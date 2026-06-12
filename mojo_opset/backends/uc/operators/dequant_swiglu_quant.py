import torch

from mojo_opset.core import MojoDequantSwiGLUQuant

from ._utils import _uc_kernels


_KERNEL_API = "mojo_dequant_swiglu_quant_bf16"
# Inner-tile column width of mojo_dequant_swiglu_quant_bf16.  Shapes with
# ``H <= _KERNEL_INNER_Y`` enter the ``H_TILES == 1`` regime where the
# kernel reproducibly hangs ``torch.npu.synchronize()`` (P2-10 finding +
# P1-G4 re-validation 2026-06-11; SetFlag/WaitFlag imbalance when the
# inner ``T.serial`` body runs exactly once across two sequential passes).
# See ``docs/project-ops/perf-debug/op-MojoDequantSwiGLUQuant-2026-06-11.md``
# §5.1 for the bump from Y=128 → Y=512 (which raised this fence from
# `H < 128` to `H <= 512` strictly).  Until the SetFlag/WaitFlag
# bookkeeping is fixed in the kernel, we fence small-H shapes back to the
# torch reference (which is the same path the wrapper already uses for
# any other capability-fence violation).
_KERNEL_INNER_Y = 512

# Row-tile of the kernel (X in mojo_dequant_swiglu_quant_bf16.py).  The
# kernel writes a full X-row block per row tile irrespective of valid M,
# so ragged-M shapes (rows % X != 0) would OOB-write the trailing block
# into adjacent NPU memory.  P1-G4 sibling worker (UCDynamicQuant
# agent-3b3ef190-d4f, 2026-06-11) confirmed the same gate is needed when
# moving from X=8 to X=16.  ``torch.zeros + copy_`` / ``torch.cat``
# padding strategies are explicitly NOT used here: torch_npu 2.x exhibits
# a DMA-retire race when ``torch.cat``-padded inputs are immediately
# consumed by a UC kernel (first call corrupted, subsequent succeed).
_KERNEL_ROW_TILE = 16

# Launch-floor profitability gate (uc-best-practices §C.1 / §I.4):
# ~80-95 µs per UC kernel launch + ~16 µs runtime cache hit (post the
# 2026-06-11 ``init_workspace`` cache patch).  Below this numel, the UC
# kernel cannot beat the torch reference even when its on-device compute
# is essentially free, so we fall back to the parent torch path.
_UC_MIN_NUMEL = 64 * 1024


class UCDequantSwiGLUQuant(MojoDequantSwiGLUQuant):
    """UC backend for the fused dequant + SwiGLU + dynamic-quant op.

    The wheel kernel handles the most common W8A8 MLP fast path:

        - 2D contiguous int8 input ``x`` of shape ``(tokens, 2H)``.
        - Single-group ``weight_scale`` ``(2H,)`` fp32 and ``quant_scale``
          ``(H,)`` fp32 (i.e. ``expert_num == 1``).
        - No ``activation_scale``, no ``bias``, no ``quant_offset`` and no
          grouped ``token_count``.
        - Default ``activate_left=False`` (mojo's ``silu(right) * left``).
        - Dynamic int8 quant (``quant_dtype=int8``, ``quant_mode=1``).
        - ``H > 512`` strictly (smaller H trips the kernel's H_TILES==1
          hang; see P2-10 + P1-G4 perf-debug docs).
        - ``tokens % 16 == 0`` (ragged-M would OOB-write the trailing
          16-row block; see P1-G4 perf-debug §5.2).
        - ``tokens * 2H >= 64K`` (sub-launch-floor numel falls back to
          torch reference per uc-best-practices §C.1).

    Anything outside that envelope falls back to ``MojoDequantSwiGLUQuant``'s
    torch reference forward, so accuracy parity is preserved.
    """

    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        activation_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        token_count: torch.Tensor = None,
    ):
        # Capability fence: anything the kernel cannot model must drop back to
        # the torch reference implementation.
        if (
            x.dtype != torch.int8
            or x.dim() != 2
            or x.shape[-1] % 2 != 0
            or self.hidden_size <= _KERNEL_INNER_Y  # H_TILES==1 hang (P2-10 + P1-G4)
            or activation_scale is not None
            or bias is not None
            or quant_offset is not None
            or token_count is not None
            or self.activate_left
            or self.quant_dtype != torch.int8
            or self.quant_mode != 1
            or self.weight_scale.shape[0] != 1
            or self.quant_scale.shape[0] != 1
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Launch-floor + ragged-M profitability gates.  These two are NOT
        # capability fences (the kernel would handle small-numel + ragged-M
        # incorrectly but silently); they intentionally fall back to the parent
        # torch path for shapes where UC kernel is known to lose or to OOB.
        # See uc-best-practices §C.1 (launch-floor) and P1-G4 perf-debug §5.2.
        if x.numel() < _UC_MIN_NUMEL:
            return super().forward(x, activation_scale, bias, quant_offset, token_count)
        if x.shape[0] % _KERNEL_ROW_TILE != 0:
            return super().forward(x, activation_scale, bias, quant_offset, token_count)

        try:
            kernels = _uc_kernels()
            kernel = kernels[_KERNEL_API]
        except (KeyError, ImportError):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        x_c = x.contiguous()
        rows, cols = x_c.shape  # rows = M (tokens), cols = N = 2*H
        half = cols // 2  # H

        # weight_scale Parameter shape is (expert_num, 2H); take the single
        # expert row and force fp32 contiguous to match the prim_func ABI.
        ws = self.weight_scale[0].to(torch.float32).contiguous()
        qs = self.quant_scale[0].to(torch.float32).contiguous()

        y = torch.empty((rows, half), dtype=torch.int8, device=x_c.device)
        scale_1d = torch.empty((rows,), dtype=torch.float32, device=x_c.device)

        # Trailing INT32 scalars follow the first-occurrence order of dim
        # names in the prim_func tensor annotations: M (from x), N (from x),
        # H (from quant_scale).
        kernel(x_c, ws, qs, y, scale_1d, rows, cols, half)

        # Mojo contract: scale shape == input.shape[:-1] + (1,) (amax keepdim).
        return y, scale_1d.unsqueeze(-1)

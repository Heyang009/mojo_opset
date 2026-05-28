import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from mojo_opset.backends.ttx.kernels import allgather_gemm_impl
from mojo_opset.backends.ttx.kernels import gemm_allreduce_impl
from mojo_opset.backends.ttx.kernels import gemm_reduce_scatter_impl
from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.core import MojoAllGatherGemm
from mojo_opset.core import MojoGemmAllReduce
from mojo_opset.core import MojoGemmReduceScatter


class TTXAllGatherGemm(MojoAllGatherGemm):
    """Triton-based fused AllGather + GEMM on Ascend NPU via aclshmem.

    Uses a hand-tuned Triton kernel that performs distributed AllGather through
    symmetric shared memory (aclshmem) and fuses it with GEMM computation.
    Currently supports fp16, gather_dim=0 only.
    """

    supported_platforms_list = ["npu"]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        gather_dim: int = 0,
    ):
        super().__init__(weight, bias, trans_weight, process_group, gather_dim)
        self._peer_mem = None
        self._rank = None
        self._world_size = None

    def _ensure_shmem(self, K: int) -> None:
        if self._peer_mem is not None:
            return

        _preload_shmem_libs()
        import shmem as ash
        from mojo_opset.backends.ttx.kernels.npu.allgather_gemm import _ensure_ash_init

        process_group = self.process_group or _get_default_group()
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)

        _ensure_ash_init(rank, world_size)

        BLOCK_SIZE_M = 128
        BLOCK_SIZE_K = 256
        pvalue = 4
        buffer_num = 2
        flat_size = BLOCK_SIZE_M * pvalue * world_size * buffer_num * max(K, BLOCK_SIZE_K)
        self._peer_mem = ash.aclshmem_create_tensor(
            [flat_size], dtype=torch.float16, device_id=rank
        )
        self._rank = rank
        self._world_size = world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return super().forward(input)

        if self.gather_dim != 0:
            input = input.movedim(self.gather_dim, 0)

        orig_shape = input.shape
        K = input.shape[-1]
        input_2d = input.reshape(-1, K).contiguous()

        self._ensure_shmem(K)

        if self.trans_weight:
            weight = self.weight
        else:
            weight = self.weight.t().contiguous()

        N = weight.shape[1]
        M = input_2d.shape[0]
        output = torch.empty(
            [M * self._world_size, N],
            dtype=input.dtype,
            device=input.device,
        )

        allgather_gemm_impl(
            input_2d, weight, output, self._peer_mem,
            self._rank, self._world_size,
        )

        if self.bias is not None:
            output = output + self.bias

        out_shape = list(orig_shape)
        out_shape[0] *= self._world_size
        out_shape[-1] = N
        output = output.reshape(out_shape)

        if self.gather_dim != 0:
            output = output.movedim(0, self.gather_dim)

        return output


def _preload_shmem_libs():
    """Preload shmem shared libraries to avoid symbol conflicts with xpu_ops."""
    import ctypes
    import importlib.util
    spec = importlib.util.find_spec("shmem")
    shmem_dir = os.path.dirname(spec.origin)
    ctypes.CDLL(os.path.join(shmem_dir, "libshmem_utils.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(shmem_dir, "libshmem.so"), mode=ctypes.RTLD_GLOBAL)


class TTXGemmAllReduce(MojoGemmAllReduce):
    """Triton-based fused GEMM + AllReduce on Ascend NPU via aclshmem."""

    supported_platforms_list = ["npu"]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(weight, bias, trans_weight, process_group)
        self._peer_mem = None
        self._rank = None
        self._world_size = None

    def _ensure_shmem(self) -> None:
        if self._peer_mem is not None:
            return

        _preload_shmem_libs()
        import shmem as ash
        from mojo_opset.backends.ttx.kernels.npu.allgather_gemm import _ensure_ash_init

        process_group = self.process_group or _get_default_group()
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)

        _ensure_ash_init(rank, world_size)

        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        ncore = get_num_cores("cube")
        pvalue = 4
        buffer_num = 2
        flat_size = BLOCK_SIZE_M * pvalue * ncore * buffer_num * BLOCK_SIZE_N
        self._peer_mem = ash.aclshmem_create_tensor(
            [flat_size], dtype=torch.float16, device_id=rank
        )
        self._rank = rank
        self._world_size = world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return super().forward(input)

        orig_shape = input.shape
        K = input.shape[-1]
        input_2d = input.reshape(-1, K).contiguous()

        self._ensure_shmem()

        if self.trans_weight:
            weight = self.weight
        else:
            weight = self.weight.t().contiguous()

        N = weight.shape[1]
        M = input_2d.shape[0]
        output = torch.zeros(
            [M, N], dtype=input.dtype, device=input.device,
        )

        gemm_allreduce_impl(
            input_2d, weight, output, self._peer_mem,
            self._rank, self._world_size,
        )

        if self.bias is not None:
            output = output + self.bias

        out_shape = list(orig_shape)
        out_shape[-1] = N
        return output.reshape(out_shape)


class TTXGemmReduceScatter(MojoGemmReduceScatter):
    """Triton-based fused GEMM + ReduceScatter on Ascend NPU via aclshmem."""

    supported_platforms_list = ["npu"]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        scatter_dim: int = 0,
    ):
        super().__init__(weight, bias, trans_weight, process_group, scatter_dim)
        self._peer_mem = None
        self._rank = None
        self._world_size = None

    def _ensure_shmem(self) -> None:
        if self._peer_mem is not None:
            return

        _preload_shmem_libs()
        import shmem as ash
        from mojo_opset.backends.ttx.kernels.npu.allgather_gemm import _ensure_ash_init

        process_group = self.process_group or _get_default_group()
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)

        _ensure_ash_init(rank, world_size)

        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        ncore = get_num_cores("cube")
        pvalue = 4
        buffer_num = 2
        flat_size = BLOCK_SIZE_M * pvalue * ncore * buffer_num * BLOCK_SIZE_N
        self._peer_mem = ash.aclshmem_create_tensor(
            [flat_size], dtype=torch.float16, device_id=rank
        )
        self._rank = rank
        self._world_size = world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return super().forward(input)

        if self.scatter_dim != 0:
            input = input.movedim(self.scatter_dim, 0)

        orig_shape = input.shape
        K = input.shape[-1]
        input_2d = input.reshape(-1, K).contiguous()

        self._ensure_shmem()

        if self.trans_weight:
            weight = self.weight
        else:
            weight = self.weight.t().contiguous()

        N = weight.shape[1]
        M = input_2d.shape[0]
        M_local = M // self._world_size
        output = torch.zeros(
            [M_local, N], dtype=input.dtype, device=input.device,
        )

        gemm_reduce_scatter_impl(
            input_2d, weight, output, self._peer_mem,
            self._rank, self._world_size,
        )

        if self.bias is not None:
            output = output + self.bias

        out_shape = list(orig_shape)
        out_shape[0] = M_local
        out_shape[-1] = N
        output = output.reshape(out_shape)

        if self.scatter_dim != 0:
            output = output.movedim(0, self.scatter_dim)

        return output

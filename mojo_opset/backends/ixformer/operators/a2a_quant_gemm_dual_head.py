from typing import Optional

import torch
import torch.distributed as dist

from ixformer import functions as ixf_f
from ixformer.distributed import symmetric_memory as symm

from mojo_opset.experimental import MojoA2AQuantGemmDualHead


class IxformerA2AQuantGemmDualHead(MojoA2AQuantGemmDualHead):
    supported_platforms_list = ["ilu"]

    @staticmethod
    def _cast_weight_scale_post_hook(module, incompatible_keys):
        module.o_proj.weight_scale = torch.nn.Parameter(
            module.o_proj.weight_scale.detach().to(torch.float32),
            requires_grad=module.o_proj.weight_scale.requires_grad,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workspace: Optional[torch.Tensor] = None

        if self.tp_size > 1 and dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(group=self.tp_group)
            self._enable_symm_mem(self.tp_group)
            workspace_bytes = ixf_f.a2a_quant_gemm_dual_head_workspace_bytes(
                5120, self.ch_local, self.world_size
            )
            device = torch.device("cuda", torch.cuda.current_device())
            self._workspace = symm.empty(workspace_bytes, dtype=torch.int8, device=device)
            self._workspace.zero_()
            symm.rendezvous(self._workspace, self.tp_group)
            dist.barrier(group=self.tp_group)
        
        self.register_load_state_dict_post_hook(self._cast_weight_scale_post_hook)



    def __del__(self):
        if getattr(self, "_workspace", None) is None:
            return
        try:
            group = getattr(self, "tp_group", None)
            group_name = getattr(group, "group_name", None)
            self._workspace = None
            symm.destroy(group_name=group_name)
        except Exception:
            pass

    def _enable_symm_mem(self, pg):
        if symm.is_nvshmem_available():
            symm.set_backend("NVSHMEM")
        symm.enable_symm_mem_for_group(pg.group_name)

    def _perm_tensors(self, device: torch.device):
        if self._full_perm is None:
            return None, None
        if self._full_perm_cache is None or self._full_perm_cache.device != device:
            self._full_perm_cache = torch.tensor(
                self._full_perm, dtype=torch.long, device=device
            )
            self._swa_perm_cache = torch.tensor(
                self._swa_perm, dtype=torch.long, device=device
            )
        return self._full_perm_cache, self._swa_perm_cache

    def forward(
        self,
        attn_int8: torch.Tensor,
        unified_scale: torch.Tensor,
    ) -> torch.Tensor:
        if attn_int8.dim() != 2:
            raise ValueError(f"attn_int8 must be 2D, got {tuple(attn_int8.shape)}")
        if attn_int8.size(1) != self.ch_local:
            raise ValueError(
                f"attn_int8 channel dim {attn_int8.size(1)} != expected ch_local {self.ch_local}"
            )
        if attn_int8.size(0) % self.tp_size != 0:
            raise ValueError(
                f"n_pad={attn_int8.size(0)} must be divisible by tp_size={self.tp_size}"
            )

        full_perm, swa_perm = self._perm_tensors(attn_int8.device)

        fmt = "TN" if self.o_proj.trans_weight else "NN"
        return ixf_f.a2a_quant_gemm_dual_head(
            attn_int8,
            self.o_proj.weight,
            unified_scale,
            self.o_proj.weight_scale,
            workspace=self._workspace,
            full_local_dim=self.full_local_dim,
            swa_local_dim=self.swa_local_dim,
            full_perm=full_perm,
            swa_perm=swa_perm,
            group=self.tp_group,
            format=fmt,
        )

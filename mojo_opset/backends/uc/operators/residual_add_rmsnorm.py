"""UC wrapper for residual-add RMSNorm.

Calls ``mojo_residual_add_rmsnorm_{bf16,fp16}`` for contiguous inputs and
weight.
"""

import torch

from mojo_opset.core import MojoResidualAddRMSNorm

from ._utils import _matrix_shape, run_kernel


class UCResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            raise NotImplementedError("UCResidualAddRMSNorm requires residual.")
        if hidden_state.shape != residual.shape or hidden_state.dtype is not residual.dtype:
            raise NotImplementedError("UCResidualAddRMSNorm requires matching input and residual.")
        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty
        if not hidden_state.is_contiguous() or not residual.is_contiguous():
            raise NotImplementedError("UCResidualAddRMSNorm requires contiguous input and residual.")
        if self.weight.dtype is not hidden_state.dtype or not self.weight.is_contiguous():
            raise NotImplementedError("UCResidualAddRMSNorm requires contiguous weight with input dtype.")

        rows, cols = _matrix_shape(hidden_state)
        output = torch.empty_like(hidden_state)
        residual_out = torch.empty_like(hidden_state)
        run_kernel(
            "mojo_residual_add_rmsnorm",
            hidden_state.dtype,
            hidden_state,
            residual,
            self.weight,
            output,
            residual_out,
            rows,
            cols,
            self.variance_epsilon,
        )
        if self.norm_pos == "pre":
            return output.reshape(hidden_state.shape), residual_out.reshape(hidden_state.shape)
        return output.reshape(hidden_state.shape), output.reshape(hidden_state.shape)

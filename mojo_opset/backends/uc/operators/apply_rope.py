"""UC wrapper for RoPE application.

The current implementation requires layout preparation outside a single
uc-kernel call, so the UC backend does not expose it.
"""

from mojo_opset.core import MojoApplyRoPE


class UCApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCApplyRoPE is not implemented as a single direct uc-kernel call.")

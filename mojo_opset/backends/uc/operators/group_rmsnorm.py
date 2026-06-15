"""UC wrapper for grouped RMSNorm."""

from mojo_opset.core import MojoGroupRMSNorm


class UCGroupRMSNorm(MojoGroupRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCGroupRMSNorm is not implemented as a single direct uc-kernel call.")

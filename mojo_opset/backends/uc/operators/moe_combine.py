"""UC wrapper for MoE combine."""

from mojo_opset.core import MojoMoECombine


class UCMoECombine(MojoMoECombine):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCMoECombine is not implemented as a single direct uc-kernel call.")

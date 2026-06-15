"""UC wrapper for MoE dispatch."""

from mojo_opset.core import MojoMoEDispatch


class UCMoEDispatch(MojoMoEDispatch):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCMoEDispatch is not implemented as a single direct uc-kernel call.")

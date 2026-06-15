"""UC wrapper for MoE dynamic quantization."""

from mojo_opset.core import MojoMoEDynamicQuant


class UCMoEDynamicQuant(MojoMoEDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCMoEDynamicQuant is not implemented as a single direct uc-kernel call.")

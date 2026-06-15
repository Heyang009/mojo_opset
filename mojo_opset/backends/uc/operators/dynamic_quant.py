"""UC wrapper for dynamic quantization."""

from mojo_opset.core import MojoDynamicQuant


class UCDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCDynamicQuant is not implemented as a single direct uc-kernel call.")

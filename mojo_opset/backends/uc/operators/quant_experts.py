"""UC wrapper for quantized experts."""

from mojo_opset.core import MojoQuantExperts


class UCQuantExperts(MojoQuantExperts):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCQuantExperts is not implemented as a single direct uc-kernel call.")

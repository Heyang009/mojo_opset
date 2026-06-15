"""UC wrapper for quantized GEMM."""

from mojo_opset.core import MojoQuantGemm


class UCQuantGemm(MojoQuantGemm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCQuantGemm is not implemented as a single direct uc-kernel call.")

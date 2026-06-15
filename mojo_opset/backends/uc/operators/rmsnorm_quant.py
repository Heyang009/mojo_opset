"""UC wrapper for RMSNorm quantization."""

from mojo_opset.core import MojoRMSNormQuant


class UCRMSNormQuant(MojoRMSNormQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCRMSNormQuant is not implemented as a single direct uc-kernel call.")

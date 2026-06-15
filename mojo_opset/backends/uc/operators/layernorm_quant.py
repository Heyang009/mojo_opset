"""UC wrapper for LayerNorm quantization."""

from mojo_opset.core import MojoLayerNormQuant


class UCLayerNormQuant(MojoLayerNormQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCLayerNormQuant is not implemented as a single direct uc-kernel call.")

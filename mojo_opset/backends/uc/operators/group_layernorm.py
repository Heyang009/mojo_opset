"""UC wrapper for grouped LayerNorm."""

from mojo_opset.experimental.operators.normalization import MojoGroupLayerNorm


class UCGroupLayerNorm(MojoGroupLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCGroupLayerNorm is not implemented as a single direct uc-kernel call.")

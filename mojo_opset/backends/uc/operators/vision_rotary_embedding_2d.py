"""UC wrapper for 2D vision rotary embedding."""

from mojo_opset.core import MojoVisionRotaryEmbedding2D


class UCVisionRotaryEmbedding2D(MojoVisionRotaryEmbedding2D):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UCVisionRotaryEmbedding2D is not implemented as a single direct uc-kernel call."
        )

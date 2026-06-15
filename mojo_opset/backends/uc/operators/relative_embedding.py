"""UC wrapper for relative embedding lookup."""

from mojo_opset.experimental.operators.position_embedding import MojoRelativeEmbedding


class UCRelativeEmbedding(MojoRelativeEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCRelativeEmbedding is not implemented as a single direct uc-kernel call.")

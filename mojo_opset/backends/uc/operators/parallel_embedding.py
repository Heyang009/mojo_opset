"""UC wrapper for vocabulary-parallel embedding."""

from mojo_opset.core import MojoParallelEmbedding


class UCParallelEmbedding(MojoParallelEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCParallelEmbedding is not implemented as a single direct uc-kernel call.")

from mojo_opset.experimental.operators.indexer import MojoIndexer


class UCIndexer(MojoIndexer):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCIndexer is not implemented by the current uc-kernel wheel."
        )

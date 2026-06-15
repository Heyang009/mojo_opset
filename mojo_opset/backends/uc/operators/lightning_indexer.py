from mojo_opset.experimental.operators.indexer import MojoLightningIndexer


class UCLightningIndexer(MojoLightningIndexer):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCLightningIndexer is not implemented by the current uc-kernel wheel."
        )

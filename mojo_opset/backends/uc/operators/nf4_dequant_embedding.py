from mojo_opset.core.operators.over_encoding import MojoNF4DequantEmbedding


class UCNF4DequantEmbedding(MojoNF4DequantEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCNF4DequantEmbedding is not implemented by the current uc-kernel wheel."
        )

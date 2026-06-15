from mojo_opset.core import MojoResidualAddLayerNormQuant


class UCResidualAddLayerNormQuant(MojoResidualAddLayerNormQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCResidualAddLayerNormQuant is not implemented by the current uc-kernel wheel."
        )

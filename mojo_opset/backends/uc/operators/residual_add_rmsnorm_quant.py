from mojo_opset.core import MojoResidualAddRMSNormQuant


class UCResidualAddRMSNormQuant(MojoResidualAddRMSNormQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCResidualAddRMSNormQuant is not implemented by the current uc-kernel wheel."
        )

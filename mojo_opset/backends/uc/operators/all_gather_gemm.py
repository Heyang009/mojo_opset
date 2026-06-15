from mojo_opset.core.operators.compute_with_comm import MojoAllGatherGemm


class UCAllGatherGemm(MojoAllGatherGemm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCAllGatherGemm is not implemented by the current uc-kernel wheel."
        )

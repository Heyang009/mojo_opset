from mojo_opset.experimental import MojoDiffusionAttentionFunction


class UCDiffusionAttentionFunction(MojoDiffusionAttentionFunction):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCDiffusionAttentionFunction is not implemented by the current uc-kernel wheel."
        )

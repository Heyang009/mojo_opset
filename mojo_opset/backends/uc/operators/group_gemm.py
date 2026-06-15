from mojo_opset.core import MojoGroupGemm


class UCGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCGroupGemm is not implemented by the current uc-kernel wheel."
        )

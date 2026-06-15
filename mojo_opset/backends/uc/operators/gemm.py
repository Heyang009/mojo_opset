from mojo_opset.core import MojoGemm


class UCGemm(MojoGemm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCGemm is not implemented by the current uc-kernel wheel."
        )

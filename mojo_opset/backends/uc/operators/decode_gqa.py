from mojo_opset.core import MojoDecodeGQA


class UCDecodeGQA(MojoDecodeGQA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCDecodeGQA is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.core import MojoOverEncodingNGram


class UCOverEncodingNGram(MojoOverEncodingNGram):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCOverEncodingNGram is not implemented by the current uc-kernel wheel."
        )

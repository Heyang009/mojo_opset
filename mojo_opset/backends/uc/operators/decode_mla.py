from mojo_opset.experimental.operators.attention import MojoDecodeMLA


class UCDecodeMLA(MojoDecodeMLA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCDecodeMLA is not implemented by the current uc-kernel wheel."
        )

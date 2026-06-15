from mojo_opset.experimental import MojoPagedDecodeMLA


class UCPagedDecodeMLA(MojoPagedDecodeMLA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedDecodeMLA is not implemented by the current uc-kernel wheel."
        )

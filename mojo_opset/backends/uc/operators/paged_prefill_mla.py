from mojo_opset.experimental import MojoPagedPrefillMLA


class UCPagedPrefillMLA(MojoPagedPrefillMLA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedPrefillMLA is not implemented by the current uc-kernel wheel."
        )

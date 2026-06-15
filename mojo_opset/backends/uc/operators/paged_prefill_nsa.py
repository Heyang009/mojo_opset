from mojo_opset.experimental.operators.attention import MojoPagedPrefillNSA


class UCPagedPrefillNSA(MojoPagedPrefillNSA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedPrefillNSA is not implemented by the current uc-kernel wheel."
        )

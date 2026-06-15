from mojo_opset.experimental.operators.attention import MojoPagedDecodeNSA


class UCPagedDecodeNSA(MojoPagedDecodeNSA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedDecodeNSA is not implemented by the current uc-kernel wheel."
        )

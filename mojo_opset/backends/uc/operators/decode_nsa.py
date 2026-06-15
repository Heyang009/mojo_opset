from mojo_opset.experimental.operators.attention import MojoDecodeNSA


class UCDecodeNSA(MojoDecodeNSA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCDecodeNSA is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.experimental.operators.attention import MojoPagedDecodeSWAWithKVDequant


class UCPagedDecodeSWAWithKVDequant(MojoPagedDecodeSWAWithKVDequant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedDecodeSWAWithKVDequant is not implemented by the current uc-kernel wheel."
        )

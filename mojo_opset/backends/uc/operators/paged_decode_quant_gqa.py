from mojo_opset.experimental.operators.attention import MojoPagedDecodeGQAWithKVDequant


class UCPagedDecodeGQAWithKVDequant(MojoPagedDecodeGQAWithKVDequant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedDecodeGQAWithKVDequant is not implemented by the current uc-kernel wheel."
        )

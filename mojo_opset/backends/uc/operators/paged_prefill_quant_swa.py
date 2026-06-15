from mojo_opset.experimental.operators.attention import MojoPagedPrefillSWAWithKVDequant


class UCPagedPrefillSWAWithKVDequant(MojoPagedPrefillSWAWithKVDequant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedPrefillSWAWithKVDequant is not implemented by the current uc-kernel wheel."
        )

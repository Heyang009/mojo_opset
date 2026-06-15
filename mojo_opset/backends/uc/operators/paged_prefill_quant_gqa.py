from mojo_opset.experimental import MojoPagedPrefillGQAWithKVDequant


class UCPagedPrefillGQAWithKVDequant(MojoPagedPrefillGQAWithKVDequant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedPrefillGQAWithKVDequant is not implemented by the current uc-kernel wheel."
        )

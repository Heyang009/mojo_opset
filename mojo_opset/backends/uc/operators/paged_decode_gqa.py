from mojo_opset.core import MojoPagedDecodeGQA


class UCPagedDecodeGQA(MojoPagedDecodeGQA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedDecodeGQA is not implemented by the current uc-kernel wheel."
        )

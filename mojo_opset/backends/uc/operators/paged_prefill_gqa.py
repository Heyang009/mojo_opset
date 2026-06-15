from mojo_opset.core import MojoPagedPrefillGQA


class UCPagedPrefillGQA(MojoPagedPrefillGQA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedPrefillGQA is not implemented by the current uc-kernel wheel."
        )

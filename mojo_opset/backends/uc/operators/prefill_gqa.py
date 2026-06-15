from mojo_opset.core import MojoPrefillGQA


class UCPrefillGQA(MojoPrefillGQA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPrefillGQA is not implemented by the current uc-kernel wheel."
        )

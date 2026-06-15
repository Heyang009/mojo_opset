from mojo_opset.experimental.operators.attention import MojoPrefillNSA


class UCPrefillNSA(MojoPrefillNSA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPrefillNSA is not implemented by the current uc-kernel wheel."
        )

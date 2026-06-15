"""UC wrapper for MLA prefill attention."""

from mojo_opset.experimental.operators.attention import MojoPrefillMLA


class UCPrefillMLA(MojoPrefillMLA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCPrefillMLA is not implemented as a single direct uc-kernel call.")

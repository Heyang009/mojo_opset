from mojo_opset.core import MojoCausalConv1dUpdateState


class UCCausalConv1dUpdateState(MojoCausalConv1dUpdateState):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCCausalConv1dUpdateState is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.core import MojoCausalConv1dFunction


class UCCausalConv1dFunction(MojoCausalConv1dFunction):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCCausalConv1dFunction is not implemented by the current uc-kernel wheel."
        )

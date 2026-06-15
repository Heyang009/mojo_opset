from mojo_opset.core import MojoFusedLinearCrossEntropyFunction


class UCFusedLinearCrossEntropyFunction(MojoFusedLinearCrossEntropyFunction):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCFusedLinearCrossEntropyFunction is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.core import MojoApplyRoPEFunction


class UCApplyRoPEFunction(MojoApplyRoPEFunction):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCApplyRoPEFunction is not implemented by the current uc-kernel wheel."
        )

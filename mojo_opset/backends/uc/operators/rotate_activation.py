from mojo_opset.experimental.operators.activation import MojoRotateActivation


class UCRotateActivation(MojoRotateActivation):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCRotateActivation is not implemented by the current uc-kernel wheel."
        )

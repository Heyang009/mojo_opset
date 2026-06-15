from mojo_opset.experimental.operators.position_embedding import MojoGridRoPE


class UCGridRoPE(MojoGridRoPE):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCGridRoPE is not implemented by the current uc-kernel wheel."
        )

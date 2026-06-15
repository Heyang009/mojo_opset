from mojo_opset.core.operators.compute_with_comm import MojoQuantGemmAll2All


class UCQuantGemmAll2All(MojoQuantGemmAll2All):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCQuantGemmAll2All is not implemented by the current uc-kernel wheel."
        )

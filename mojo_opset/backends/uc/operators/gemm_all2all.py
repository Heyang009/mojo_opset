from mojo_opset.core.operators.compute_with_comm import MojoGemmAll2All


class UCGemmAll2All(MojoGemmAll2All):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCGemmAll2All is not implemented by the current uc-kernel wheel."
        )

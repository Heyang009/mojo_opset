from mojo_opset.core.operators.compute_with_comm import MojoGemmReduceScatter


class UCGemmReduceScatter(MojoGemmReduceScatter):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCGemmReduceScatter is not implemented by the current uc-kernel wheel."
        )

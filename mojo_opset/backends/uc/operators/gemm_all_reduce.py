from mojo_opset.core.operators.compute_with_comm import MojoGemmAllReduce


class UCGemmAllReduce(MojoGemmAllReduce):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCGemmAllReduce is not implemented by the current uc-kernel wheel."
        )

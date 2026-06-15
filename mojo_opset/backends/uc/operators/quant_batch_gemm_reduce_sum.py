from mojo_opset.experimental import MojoQuantBatchGemmReduceSum


class UCQuantBatchGemmReduceSum(MojoQuantBatchGemmReduceSum):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCQuantBatchGemmReduceSum is not implemented by the current uc-kernel wheel."
        )

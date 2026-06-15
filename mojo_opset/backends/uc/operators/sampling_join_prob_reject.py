from mojo_opset.core.operators.sampling import MojoJoinProbRejectSampling


class UCJoinProbRejectSampling(MojoJoinProbRejectSampling):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCJoinProbRejectSampling is not implemented by the current uc-kernel wheel."
        )

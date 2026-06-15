from mojo_opset.core.operators.sampling import MojoTopKSampling


class UCTopKSampling(MojoTopKSampling):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCTopKSampling is not implemented by the current uc-kernel wheel."
        )

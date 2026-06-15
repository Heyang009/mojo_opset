from mojo_opset.core.operators.sampling import MojoRejectSampling


class UCRejectSampling(MojoRejectSampling):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCRejectSampling is not implemented by the current uc-kernel wheel."
        )

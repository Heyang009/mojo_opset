from mojo_opset.core.operators.sampling import MojoTopPFilter


class UCTopPFilter(MojoTopPFilter):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCTopPFilter is not implemented by the current uc-kernel wheel."
        )

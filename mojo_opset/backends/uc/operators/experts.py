from mojo_opset.core import MojoExperts


class UCExperts(MojoExperts):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCExperts is not implemented by the current uc-kernel wheel."
        )

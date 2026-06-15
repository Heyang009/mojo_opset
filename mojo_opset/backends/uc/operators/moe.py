from mojo_opset.core import MojoMoE


class UCMoE(MojoMoE):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCMoE is not implemented by the current uc-kernel wheel."
        )

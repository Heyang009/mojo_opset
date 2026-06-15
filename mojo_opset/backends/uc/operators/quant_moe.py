from mojo_opset.core import MojoQuantMoE


class UCQuantMoE(MojoQuantMoE):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCQuantMoE is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.core.operators.moe import MojoMoEGating


class UCMoEGating(MojoMoEGating):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCMoEGating is not implemented by the current uc-kernel wheel."
        )

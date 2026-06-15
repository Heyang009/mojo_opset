from mojo_opset.experimental.operators.moe import MojoMoEInitRoutingDynamicQuant


class UCMoEInitRoutingDynamicQuant(MojoMoEInitRoutingDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCMoEInitRoutingDynamicQuant is not implemented by the current uc-kernel wheel."
        )

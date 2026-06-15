from mojo_opset.experimental.operators.attention_gate import MojoFusedAttnOutputGate


class UCFusedAttnOutputGate(MojoFusedAttnOutputGate):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCFusedAttnOutputGate is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.core.operators.compute_with_comm import MojoAll2AllQuantGemm


class UCAll2AllQuantGemm(MojoAll2AllQuantGemm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCAll2AllQuantGemm is not implemented by the current uc-kernel wheel."
        )

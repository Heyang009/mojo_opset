"""UC wrapper for fused SwiGLU MoE dynamic quantization."""

from mojo_opset.experimental.operators.moe import MojoFusedSwiGLUMoEScaleDynamicQuantize


class UCFusedSwiGLUMoEScaleDynamicQuantize(MojoFusedSwiGLUMoEScaleDynamicQuantize):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UCFusedSwiGLUMoEScaleDynamicQuantize is not implemented as a single direct uc-kernel call."
        )

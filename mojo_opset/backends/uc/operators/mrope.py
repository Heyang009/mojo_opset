"""UC wrapper for multimodal RoPE."""

from mojo_opset.core import MojoMRoPE


class UCMRoPE(MojoMRoPE):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCMRoPE is not implemented as a single direct uc-kernel call.")

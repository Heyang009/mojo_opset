"""UC wrapper for channel RMSNorm."""

from mojo_opset.experimental.operators.normalization import MojoChannelRMSNorm


class UCChannelRMSNorm(MojoChannelRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCChannelRMSNorm is not implemented as a single direct uc-kernel call.")

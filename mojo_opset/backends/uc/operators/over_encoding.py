"""UC wrapper for over-encoding."""

from mojo_opset.core import MojoOverEncoding


class UCOverEncoding(MojoOverEncoding):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCOverEncoding is not implemented as a single direct uc-kernel call.")

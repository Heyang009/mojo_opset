from mojo_opset.core import MojoSWA


class UCSWA(MojoSWA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCSWA is not implemented by the current uc-kernel wheel."
        )

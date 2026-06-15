from mojo_opset.core import MojoSdpa


class UCSdpa(MojoSdpa):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCSdpa is not implemented by the current uc-kernel wheel."
        )

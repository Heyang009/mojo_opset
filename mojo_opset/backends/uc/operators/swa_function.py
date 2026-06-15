from mojo_opset.core import MojoSWAFunction


class UCSWAFunction(MojoSWAFunction):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCSWAFunction is not implemented by the current uc-kernel wheel."
        )

from mojo_opset.core import MojoSwiGLUMLP


class UCSwiGLUMLP(MojoSwiGLUMLP):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCSwiGLUMLP is not implemented by the current uc-kernel wheel."
        )

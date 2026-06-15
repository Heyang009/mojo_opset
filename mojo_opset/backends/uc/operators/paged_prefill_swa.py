from mojo_opset.core import MojoPagedPrefillSWA


class UCPagedPrefillSWA(MojoPagedPrefillSWA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedPrefillSWA is not implemented by the current uc-kernel wheel."
        )

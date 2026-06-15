from mojo_opset.core import MojoPagedDecodeSWA


class UCPagedDecodeSWA(MojoPagedDecodeSWA):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCPagedDecodeSWA is not implemented by the current uc-kernel wheel."
        )

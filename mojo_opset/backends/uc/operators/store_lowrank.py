"""UC wrapper for low-rank KV cache store."""

from mojo_opset.experimental.operators.store_lowrank import MojoStoreLowrank


class UCStoreLowrank(MojoStoreLowrank):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCStoreLowrank is not implemented as a single direct uc-kernel call.")

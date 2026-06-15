"""UC wrapper for paged KV cache store."""

from mojo_opset.core import MojoStorePagedKVCache


class UCStorePagedKVCache(MojoStorePagedKVCache):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCStorePagedKVCache is not implemented as a single direct uc-kernel call.")

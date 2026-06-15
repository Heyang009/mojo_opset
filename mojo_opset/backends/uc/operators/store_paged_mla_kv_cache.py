"""UC wrapper for paged MLA KV cache store."""

from mojo_opset.experimental.operators.kv_cache import MojoStorePagedMLAKVCache


class UCStorePagedMLAKVCache(MojoStorePagedMLAKVCache):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("UCStorePagedMLAKVCache is not implemented as a single direct uc-kernel call.")

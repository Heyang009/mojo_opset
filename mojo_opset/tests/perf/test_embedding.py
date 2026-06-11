import pytest
import torch

from mojo_opset import MojoEmbedding, MojoParallelEmbedding
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


# Representative shapes mixing latency-critical decode (N=1), small prefill
# (N=32), large prefill (N=1024), and very large prefill (N=8192) at the
# two HT-variant-aligned embedding dims the UC kernel ships.
@pytest.mark.parametrize(
    "num_embeddings, embedding_dim, num_tokens",
    [
        (1024, 128, 1024),       # H=128 path
        (1024, 128, 8192),       # H=128 path, long sequence
        (32000, 4096, 1),        # decode, smaller vocab
        (32000, 4096, 128),
        (32000, 4096, 1024),
        (128256, 4096, 1),       # llama-3 vocab, decode
        (128256, 4096, 1024),    # llama-3 vocab, prefill
        (128256, 4096, 8192),    # long prefill
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_embedding(num_embeddings, embedding_dim, num_tokens, dtype):
    device = get_torch_device()
    op = MojoEmbedding(num_embeddings, embedding_dim, device=device, dtype=dtype)
    ids = torch.randint(0, num_embeddings, (num_tokens,), dtype=torch.int64, device=device)
    perf(lambda: op(ids))  # noqa: F821


# Same shape grid for the parallel (TP-aware) variant. In single-rank /
# no-dist environments the wrapper goes through its fast-path that skips
# the parent's TP ceremony; in TP mode it does the host-side
# shift/mask/clamp/mul + all_reduce. Both paths run the lookup primitive
# (UC kernel iff ``_is_kernel_profitable`` approves, otherwise
# ``F.embedding`` → ``aclnnEmbedding``).
@pytest.mark.parametrize(
    "num_embeddings, embedding_dim, num_tokens",
    [
        (1024, 128, 1024),
        (1024, 128, 8192),
        (32000, 4096, 1),
        (32000, 4096, 128),
        (32000, 4096, 1024),
        (128256, 4096, 1),
        (128256, 4096, 1024),
        (128256, 4096, 8192),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_parallel_embedding(num_embeddings, embedding_dim, num_tokens, dtype):
    device = get_torch_device()
    op = MojoParallelEmbedding(num_embeddings, embedding_dim, device=device, dtype=dtype)
    ids = torch.randint(0, num_embeddings, (num_tokens,), dtype=torch.int64, device=device)
    perf(lambda: op(ids))  # noqa: F821

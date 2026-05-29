"""Test TTXAllGatherGemm multi-card via torchrun."""
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


def run_test():
    rank = int(os.environ["LOCAL_RANK"])
    import torch_npu  # noqa: F401
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl")
    world_size = dist.get_world_size()

    M, K, N = 32, 64, 128
    dtype = torch.float32

    torch.manual_seed(42)
    x_full = torch.randn(M, K, dtype=dtype).npu()
    w = torch.randn(N, K, dtype=dtype).npu()
    b = torch.randn(N, dtype=dtype).npu()

    # Broadcast shared tensors
    dist.broadcast(x_full, src=0)
    dist.broadcast(w, src=0)
    dist.broadcast(b, src=0)

    # Reference: full matmul
    ref = F.linear(x_full, w, b)

    # Each rank takes its shard
    m_local = M // world_size
    x_local = x_full[rank * m_local:(rank + 1) * m_local].contiguous()

    # Test torch fallback path
    from mojo_opset.core.operators.compute_with_comm import MojoAllGatherGemm
    torch_op = MojoAllGatherGemm._registry.get("torch")(
        weight=w, bias=b, trans_weight=False, gather_dim=0
    )
    torch_out = torch_op(x_local)
    torch.testing.assert_close(torch_out, ref, atol=1e-4, rtol=1e-4)
    if rank == 0:
        print(f"[PASS] torch fallback AllGatherGemm, shape={torch_out.shape}")

    dist.destroy_process_group()
    if rank == 0:
        print("[ALL PASS]")


if __name__ == "__main__":
    run_test()

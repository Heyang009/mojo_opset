"""Test TTXAllGatherGemm against torch reference (MojoAllGatherGemm fallback)."""
import os
import torch
import torch.distributed as dist


def run_test():
    rank = int(os.environ["LOCAL_RANK"])
    import torch_npu  # noqa: F401
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl")

    world_size = dist.get_world_size()
    M, K, N = 4096, 4096, 4096
    dtype = torch.float16

    # Shared weight (same on all ranks)
    torch.manual_seed(42)
    weight = torch.randn(K, N, dtype=dtype).npu()  # trans_weight=True layout: [K, N]
    bias = torch.randn(N, dtype=dtype).npu()

    # Per-rank input shard
    torch.manual_seed(42 + rank)
    input_local = torch.randn(M, K, dtype=dtype).npu()

    # --- Reference: torch AllGather + GEMM ---
    from mojo_opset.core.operators.compute_with_comm import MojoAllGatherGemm
    torch_cls = MojoAllGatherGemm._registry.get("torch")
    ref_op = torch_cls(weight=weight, bias=bias, trans_weight=True, gather_dim=0)
    ref_output = ref_op.forward(input_local)

    # --- TTX: fused AllGather + GEMM ---
    from mojo_opset.backends.ttx.operators.compute_with_comm import TTXAllGatherGemm
    ttx_op = TTXAllGatherGemm(weight=weight, bias=bias, trans_weight=True)
    ttx_output = ttx_op.forward(input_local)

    # Compare
    torch.testing.assert_close(ttx_output, ref_output, rtol=1e-3, atol=1e-3)
    if rank == 0:
        print(f"[PASS] TTXAllGatherGemm trans_weight=True, bias=True, shape={ttx_output.shape}")

    # --- Test trans_weight=False ---
    weight_nt = torch.randn(N, K, dtype=dtype).npu()  # [N, K] layout
    ref_op2 = torch_cls(weight=weight_nt, bias=None, trans_weight=False, gather_dim=0)
    ref_output2 = ref_op2.forward(input_local)

    ttx_op2 = TTXAllGatherGemm(weight=weight_nt, bias=None, trans_weight=False)
    ttx_output2 = ttx_op2.forward(input_local)

    torch.testing.assert_close(ttx_output2, ref_output2, rtol=1e-3, atol=1e-3)
    if rank == 0:
        print(f"[PASS] TTXAllGatherGemm trans_weight=False, bias=None, shape={ttx_output2.shape}")

    dist.destroy_process_group()
    if rank == 0:
        print("[ALL PASS]")


if __name__ == "__main__":
    run_test()

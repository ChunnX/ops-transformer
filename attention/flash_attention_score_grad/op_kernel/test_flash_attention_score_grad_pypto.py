"""Test script for FlashAttentionScoreGrad PyPTO Pro kernel.

Computes golden reference using PyTorch and compares with kernel output.
"""

import os
import math
import sys

import torch
import torch_npu

# Add op_kernel directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_pypto import (
    flash_attention_score_grad,
    FagTiling,
    TILE_M, TILE_N, TILE_D,
)


def golden_flash_attention_grad(q, k, v, dy, scale):
    """Compute dQ, dK, dV using PyTorch reference.

    Forward: O = softmax(Q @ K^T * scale) @ V
    Backward: given dO=dy, compute dQ, dK, dV.

    Uses FP32 for all computation.
    """
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    dy_f32 = dy.float()

    # S = Q @ K^T * scale  [B, N, S1, S2]
    s = torch.matmul(q_f32, k_f32.transpose(-1, -2)) * scale

    # P = softmax(S, dim=-1)
    s_max = s.max(dim=-1, keepdim=True).values
    s_exp = torch.exp(s - s_max)
    s_sum = s_exp.sum(dim=-1, keepdim=True)
    p = s_exp / s_sum

    # O = P @ V  (not needed for grad, but for completeness)
    # o = torch.matmul(p, v_f32)

    # Backward
    # dV = P^T @ dO  [B, N, S2, D]
    dv = torch.matmul(p.transpose(-1, -2), dy_f32)

    # dP = dO @ V^T  [B, N, S1, S2]
    dp = torch.matmul(dy_f32, v_f32.transpose(-1, -2))

    # dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    dp_sum = (dp * p).sum(dim=-1, keepdim=True)
    ds = p * (dp - dp_sum)

    # dQ = dS @ K * scale  [B, N, S1, D]
    dq = torch.matmul(ds, k_f32) * scale

    # dK = dS^T @ Q * scale  [B, N, S2, D]
    dk = torch.matmul(ds.transpose(-1, -2), q_f32) * scale

    # softmax_max and softmax_sum for kernel
    softmax_max = s_max.squeeze(-1)  # [B, N, S1]
    softmax_sum = s_sum.squeeze(-1)  # [B, N, S1]

    return dq.half(), dk.half(), dv.half(), softmax_max, softmax_sum


def run_test():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    # Test parameters — keep S1, S2, D as multiples of 128 for simplicity
    B, N, S1, S2, D = 1, 1, 256, 256, 128
    scale = 1.0 / math.sqrt(D)
    num_cores = 8

    torch.manual_seed(42)
    q = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    dy = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)

    # Compute golden reference
    dq_gold, dk_gold, dv_gold, smax_gold, ssum_gold = golden_flash_attention_grad(
        q.cpu(), k.cpu(), v.cpu(), dy.cpu(), scale)

    smax_gold = smax_gold.to(device)
    ssum_gold = ssum_gold.to(device)

    # Allocate workspaces (FP32, initialized to zero for atomic add)
    dq_ws = torch.zeros(B, S1, N, D, device=device, dtype=torch.float32)
    dk_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    dv_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)

    # dS and P workspaces (FP16)
    ds_ws = torch.empty(B, N, S1, S2, device=device, dtype=torch.float16)
    p_ws = torch.empty(B, N, S1, S2, device=device, dtype=torch.float16)

    tiling = FagTiling(
        b=B, n=N, s1=S1, s2=S2, d=D,
        scale_value=scale,
    )

    # Launch kernel
    print(f"Launching kernel: B={B}, N={N}, S1={S1}, S2={S2}, D={D}, "
          f"scale={scale:.6f}, cores={num_cores}")

    flash_attention_score_grad[None, num_cores](
        q, k, v, dy,
        smax_gold, ssum_gold,
        dq_ws, dk_ws, dv_ws,
        ds_ws, p_ws,
        tiling,
    )
    torch.npu.synchronize()

    # Cast workspace results to FP16
    dq_out = dq_ws.half()
    dk_out = dk_ws.half()
    dv_out = dv_ws.half()

    # Compare with golden
    def check(name, out, gold, rtol, atol):
        out_cpu = out.cpu()
        gold_cpu = gold.cpu()
        max_diff = (out_cpu.float() - gold_cpu.float()).abs().max().item()
        mean_diff = (out_cpu.float() - gold_cpu.float()).abs().mean().item()
        try:
            torch.testing.assert_close(out_cpu, gold_cpu, rtol=rtol, atol=atol)
            print(f"  {name}: PASS  (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
            return True
        except AssertionError as e:
            print(f"  {name}: FAIL  (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
            print(f"    {e}")
            return False

    print("\nResults:")
    # FP16 gradients typically need rtol=1e-2, atol=1e-2
    # For gradients, we may need looser tolerance
    rtol, atol = 5e-2, 5e-2
    ok_dq = check("dQ", dq_out, dq_gold, rtol, atol)
    ok_dk = check("dK", dk_out, dk_gold, rtol, atol)
    ok_dv = check("dV", dv_out, dv_gold, rtol, atol)

    if ok_dq and ok_dk and ok_dv:
        print("\n=== All tests PASSED ===")
        return 0
    else:
        print("\n=== Some tests FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(run_test())

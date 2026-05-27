import torch
import torch_npu
import npu_ops_transformer


def mhc_post_reference(x, hRes, hOut, hPost):
    """
    CPU reference implementation:
      x_{l+1} = (H_res)^T @ x + h_out ⊗ H_post

    For BSND format [B, S, N, D]:
      res_mapping = hRes.transpose(-2,-1) @ x   -> [B,S,N,D]
      post_mapping = hOut.unsqueeze(-2) * hPost.unsqueeze(-1)  -> [B,S,N,D]

    For TND format [T, N, D]:
      res_mapping = hRes.transpose(-2,-1) @ x   -> [T,N,D]
      post_mapping = hOut.unsqueeze(-2) * hPost.unsqueeze(-1)  -> [T,N,D]
    """
    x = x.float()
    hRes = hRes.float()
    hOut = hOut.float()
    hPost = hPost.float()

    res_mapping = torch.matmul(hRes.transpose(-2, -1), x)

    if x.dim() == 4:
        post_mapping = hOut.unsqueeze(-2) * hPost.unsqueeze(-1)
    elif x.dim() == 3:
        post_mapping = hOut.unsqueeze(-2) * hPost.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported x dim: {x.dim()}, expected 3 or 4")

    out = res_mapping + post_mapping
    return out


def test_mhc_post_bsnd():
    B, S, N, D = 1, 8, 4, 16

    x_cpu = torch.randn(B, S, N, D, dtype=torch.float16)
    hRes_cpu = torch.randn(B, S, N, N, dtype=torch.float32)
    hOut_cpu = torch.randn(B, S, D, dtype=torch.float16)
    hPost_cpu = torch.randn(B, S, N, dtype=torch.float32)

    ref_out = mhc_post_reference(x_cpu, hRes_cpu, hOut_cpu, hPost_cpu).half()

    x_npu = x_cpu.npu()
    hRes_npu = hRes_cpu.npu()
    hOut_npu = hOut_cpu.npu()
    hPost_npu = hPost_cpu.npu()

    npu_out = npu_ops_transformer.ops.npu_mhc_post(x_npu, hRes_npu, hOut_npu, hPost_npu)

    npu_out_cpu = npu_out.cpu()

    print(f"[BSND] x shape: {x_npu.shape}, out shape: {npu_out.shape}")
    print(f"[BSND] ref_out shape: {ref_out.shape}")
    print(f"[BSND] max abs diff: {(npu_out_cpu - ref_out).abs().max().item():.6f}")

    if torch.allclose(npu_out_cpu, ref_out, rtol=1e-2, atol=1e-3):
        print("[BSND] Verification PASSED!")
    else:
        print("[BSND] Verification FAILED!")
        print(f"  npu_out sample: {npu_out_cpu[0, 0, 0, :4]}")
        print(f"  ref_out sample: {ref_out[0, 0, 0, :4]}")


def test_mhc_post_tnd():
    T, N, D = 32, 4, 16

    x_cpu = torch.randn(T, N, D, dtype=torch.float16)
    hRes_cpu = torch.randn(T, N, N, dtype=torch.float32)
    hOut_cpu = torch.randn(T, D, dtype=torch.float16)
    hPost_cpu = torch.randn(T, N, dtype=torch.float32)

    ref_out = mhc_post_reference(x_cpu, hRes_cpu, hOut_cpu, hPost_cpu).half()

    x_npu = x_cpu.npu()
    hRes_npu = hRes_cpu.npu()
    hOut_npu = hOut_cpu.npu()
    hPost_npu = hPost_cpu.npu()

    npu_out = npu_ops_transformer.ops.npu_mhc_post(x_npu, hRes_npu, hOut_npu, hPost_npu)

    npu_out_cpu = npu_out.cpu()

    print(f"[TND] x shape: {x_npu.shape}, out shape: {npu_out.shape}")
    print(f"[TND] ref_out shape: {ref_out.shape}")
    print(f"[TND] max abs diff: {(npu_out_cpu - ref_out).abs().max().item():.6f}")

    if torch.allclose(npu_out_cpu, ref_out, rtol=1e-2, atol=1e-3):
        print("[TND] Verification PASSED!")
    else:
        print("[TND] Verification FAILED!")
        print(f"  npu_out sample: {npu_out_cpu[0, 0, :4]}")
        print(f"  ref_out sample: {ref_out[0, 0, :4]}")


def test_mhc_post_bfloat16():
    B, S, N, D = 1, 8, 4, 16

    x_cpu = torch.randn(B, S, N, D, dtype=torch.bfloat16)
    hRes_cpu = torch.randn(B, S, N, N, dtype=torch.float32)
    hOut_cpu = torch.randn(B, S, D, dtype=torch.bfloat16)
    hPost_cpu = torch.randn(B, S, N, dtype=torch.float32)

    ref_out = mhc_post_reference(x_cpu, hRes_cpu, hOut_cpu, hPost_cpu).bfloat16()

    x_npu = x_cpu.npu()
    hRes_npu = hRes_cpu.npu()
    hOut_npu = hOut_cpu.npu()
    hPost_npu = hPost_cpu.npu()

    npu_out = npu_ops_transformer.ops.npu_mhc_post(x_npu, hRes_npu, hOut_npu, hPost_npu)

    npu_out_cpu = npu_out.cpu()

    print(f"[BSND-BF16] x shape: {x_npu.shape}, out shape: {npu_out.shape}")
    print(f"[BSND-BF16] ref_out shape: {ref_out.shape}")
    print(f"[BSND-BF16] max abs diff: {(npu_out_cpu.float() - ref_out.float()).abs().max().item():.6f}")

    if torch.allclose(npu_out_cpu.float(), ref_out.float(), rtol=1e-2, atol=1e-2):
        print("[BSND-BF16] Verification PASSED!")
    else:
        print("[BSND-BF16] Verification FAILED!")
        print(f"  npu_out sample: {npu_out_cpu[0, 0, 0, :4]}")
        print(f"  ref_out sample: {ref_out[0, 0, 0, :4]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing aclnnMhcPost operator via npu_ops_transformer")
    print("=" * 60)

    test_mhc_post_bsnd()
    print()
    test_mhc_post_tnd()
    print()
    test_mhc_post_bfloat16()
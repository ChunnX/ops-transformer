# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""attn_res 绑定接口（fwd+bwd autograd）精度测试。"""

import argparse

import numpy as np
import torch
import torch_npu  # noqa: F401

import cann_ops_transformer  # noqa: F401
from cann_ops_transformer.ops import attn_res

DEFAULT_NORM_EPS = 1e-5


def attn_res_golden(prefix_sum, block_residual, proj_weight, norm_weight,
                    grad_output, norm_eps=DEFAULT_NORM_EPS):
    """Golden：与 attenres_lite.apply_attn_res 等价的 autograd backward（NPU float32 标杆）。"""
    ps = prefix_sum.detach().float().requires_grad_(True)
    br = block_residual.detach().float().requires_grad_(True)
    pw = proj_weight.detach().float().requires_grad_(True)
    nw = norm_weight.detach().float().requires_grad_(True)

    v = torch.cat((br, ps.unsqueeze(1)), dim=1)
    variance = v.pow(2).mean(-1, keepdim=True)
    k = v * torch.rsqrt(variance + norm_eps)
    score_weight = nw * pw.squeeze(0)
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden = torch.matmul(probs, v).squeeze(1)

    hidden.backward(grad_output.detach().float())
    return (ps.grad.to(prefix_sum.dtype),
            br.grad.to(block_residual.dtype),
            pw.grad.view(1, -1).to(proj_weight.dtype),
            nw.grad.to(norm_weight.dtype))


def check(name, npu, cpu, rtol=2e-2, atol=2e-2):
    a = npu.detach().float().cpu().reshape(-1)
    b = cpu.detach().float().cpu().reshape(-1)
    abs_err = (a - b).abs()
    denom = b.abs().clamp_min(1e-10)
    mere = (abs_err / denom).max().item()
    mare = abs_err.max().item() / denom.max().item()
    mae = abs_err.max().item()
    ok = torch.allclose(npu.float().cpu(), cpu.float().cpu(), rtol=rtol, atol=atol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: MERE={mere:.3e} MARE={mare:.3e} MAE={mae:.3e}")
    if not ok:
        idx = abs_err.argmax().item()
        lo = max(0, idx - 2)
        hi = min(abs_err.numel(), idx + 3)
        print(f"    {name} 最大误差点及前后2点 (元素数 {abs_err.numel()}, 原始shape {tuple(npu.shape)}):")
        for i in range(lo, hi):
            coord = np.unravel_index(i, tuple(npu.shape))
            print(f"      flat[{i}] coord={coord} npu={a[i].item():.6e} "
                  f"gold={b[i].item():.6e} abs_err={abs_err[i].item():.6e}")
    return ok


def run_case(dtype, B, N, H, seed):
    torch.manual_seed(seed)
    print(f"\n===== dtype={dtype} B={B} N={N} H={H} seed={seed} =====")

    # ---- 可微路径：attn_res 自动走 fwd+bwd
    # CPU 生成再 .npu()，生成顺序与 test_attn_res_precision.py 保持一致
    prefix_sum = torch.randn(B, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
    block_residual = torch.randn(B, N, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
    norm_weight = torch.randn(H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
    proj_weight = torch.randn(1, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
    grad_output = torch.randn(B, H, dtype=dtype, device="cpu").to("npu")

    hidden = attn_res(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=DEFAULT_NORM_EPS)
    hidden.backward(grad_output)

    g_ps, g_br, g_pw, g_nw = attn_res_golden(
        prefix_sum, block_residual, proj_weight, norm_weight, grad_output)

    results = []
    results.append(check("bwd.grad_prefix_sum", prefix_sum.grad, g_ps))
    results.append(check("bwd.grad_block_residual", block_residual.grad, g_br))
    results.append(check("bwd.grad_proj_weight", proj_weight.grad, g_pw))
    results.append(check("bwd.grad_norm_weight", norm_weight.grad, g_nw))

    # ---- 非可微路径：仅正向
    hidden_nograd = attn_res(prefix_sum.detach(), block_residual.detach(),
                             proj_weight.detach(), norm_weight.detach(),
                             norm_eps=DEFAULT_NORM_EPS)
    ok_fwd = hidden_nograd.shape == (B, H) and hidden_nograd.dtype == dtype
    results.append(ok_fwd)
    print(f"[{'PASS' if ok_fwd else 'FAIL'}] fwd.nograd shape={tuple(hidden_nograd.shape)} dtype={hidden_nograd.dtype}")

    return all(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--H", type=int, default=7168)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ok = run_case(dtype, args.B, args.N, args.H, args.seed)
    print("\n=== ALL PASS ===" if ok else "\n=== SOME CASES FAILED ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

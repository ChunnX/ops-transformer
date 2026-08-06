# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""attn_res_fwd / attention_res_grad 精度验证（torch_extension 集成）。"""

import argparse

import torch
import torch_npu  # noqa: F401

import cann_ops_transformer
from cann_ops_transformer.ops import attn_res_fwd, attention_res_grad

DEFAULT_NORM_EPS = 1e-5


# ---------------------------------------------------------------- golden reference
def attn_res_fwd_golden(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=DEFAULT_NORM_EPS):
    """Forward golden, following attention/attn_res_fwd/tests/pytest/attn_res_fwd_golden.py."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm_eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1)  # [B, N+1]
    hidden_states = (probs.unsqueeze(-1) * v_float).sum(dim=1)  # [B, H]
    inv_rms = torch.rsqrt(variance + norm_eps).squeeze(-1)  # [B, N+1]
    return hidden_states.to(v.dtype), inv_rms, probs


def attention_res_grad_golden(prefix_sum, block_residual, proj_weight, norm_weight,
                              grad_output, inv_rms, probs, norm_eps=DEFAULT_NORM_EPS):
    """Backward golden：对 attenres_lite.py 的 apply_attn_res 正向做 autograd backward。

    直接复用仓内正向标杆（attention/attn_res_fwd/tests/pytest/attenres_lite.py），
    与 kernel 的数学定义保持一致；inv_rms/probs 为正向中间量，不参与梯度计算。
    """
    ps = prefix_sum.detach().float().requires_grad_(True)
    br = block_residual.detach().float().requires_grad_(True)
    pw = proj_weight.detach().float().requires_grad_(True)
    nw = norm_weight.detach().float().requires_grad_(True)

    v = torch.cat((br, ps.unsqueeze(1)), dim=1)
    variance = v.pow(2).mean(-1, keepdim=True)
    k = v * torch.rsqrt(variance + norm_eps)
    score_weight = nw * pw.squeeze(0)
    scores = (k * score_weight).sum(-1)
    probs_fwd = scores.softmax(-1).unsqueeze(1)
    hidden = torch.matmul(probs_fwd, v).squeeze(1)

    hidden.backward(grad_output.detach().float())
    return (ps.grad.to(prefix_sum.dtype),
            br.grad.to(block_residual.dtype),
            pw.grad.view(1, -1).to(proj_weight.dtype),
            nw.grad.to(norm_weight.dtype))


# ---------------------------------------------------------------- compare helpers
def _stats(npu, cpu):
    a = npu.detach().float().cpu().reshape(-1)
    b = cpu.detach().float().cpu().reshape(-1)
    abs_err = (a - b).abs()
    denom = b.abs().clamp_min(1e-10)
    rel_err = abs_err / denom
    return float(rel_err.max()), float(abs_err.max()), float(abs_err.mean())


def check(name, npu, cpu, rtol=2e-2, atol=2e-2):
    mere, mare, mae = _stats(npu, cpu)
    ok = torch.allclose(npu.float().cpu(), cpu.float().cpu(), rtol=rtol, atol=atol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: MERE={mere:.3e} MARE={mare:.3e} MAE={mae:.3e}")
    return ok


# ---------------------------------------------------------------- main
def run_case(dtype, B, N, H, seed, with_fwd=True):
    torch.manual_seed(seed)
    print(f"\n===== dtype={dtype} B={B} N={N} H={H} seed={seed} =====")
    prefix_sum = torch.randn(B, H, dtype=dtype)
    block_residual = torch.randn(B, N, H, dtype=dtype)
    norm_weight = torch.randn(H, dtype=dtype)
    proj_weight = torch.randn(1, H, dtype=dtype)

    results = []

    if with_fwd:
        # ---- forward
        hidden, inv_rms, probs = attn_res_fwd(
            prefix_sum.npu(), block_residual.npu(), proj_weight.npu(), norm_weight.npu(),
            norm_eps=DEFAULT_NORM_EPS, need_backward=True)
        g_hidden, g_inv_rms, g_probs = attn_res_fwd_golden(prefix_sum, block_residual, proj_weight, norm_weight)

        results.append(check("fwd.hidden", hidden, g_hidden))
        results.append(check("fwd.inv_rms", inv_rms, g_inv_rms))
        results.append(check("fwd.probs", probs, g_probs))

        # ---- backward (与 fwd 输出联动)
        grad_output = torch.randn(B, H, dtype=dtype)
        grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight = attention_res_grad(
            prefix_sum.npu(), block_residual.npu(), proj_weight.npu(), norm_weight.npu(),
            grad_output.npu(), inv_rms, probs)
        g_ps, g_br, g_pw, g_nw = attention_res_grad_golden(
            prefix_sum, block_residual, proj_weight, norm_weight, grad_output, inv_rms.cpu(), probs.cpu())
    else:
        # ---- backward（独立输入，不依赖 fwd）
        inv_rms_fp32 = 0.5 + torch.rand(B, N + 1, dtype=torch.float32)
        probs_fp32 = torch.rand(B, N + 1, dtype=torch.float32)
        probs_fp32 = probs_fp32 / probs_fp32.sum(-1, keepdim=True)
        grad_output = torch.randn(B, H, dtype=dtype)
        grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight = attention_res_grad(
            prefix_sum.npu(), block_residual.npu(), proj_weight.npu(), norm_weight.npu(),
            grad_output.npu(), inv_rms_fp32.npu(), probs_fp32.npu())
        g_ps, g_br, g_pw, g_nw = attention_res_grad_golden(
            prefix_sum, block_residual, proj_weight, norm_weight, grad_output, inv_rms_fp32, probs_fp32)

    results.append(check("bwd.grad_prefix_sum", grad_prefix_sum, g_ps))
    results.append(check("bwd.grad_block_residual", grad_block_residual, g_br))
    results.append(check("bwd.grad_proj_weight", grad_proj_weight, g_pw))
    results.append(check("bwd.grad_norm_weight", grad_norm_weight, g_nw))

    return all(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    # 正向算子（910B3/arch22）仅支持 BF16；BF16 走 fwd+bwd 全链路，FP16 走 bwd 独立输入
    if args.dtype == "bf16":
        ok = run_case(dtype, args.B, args.N, args.H, args.seed, with_fwd=True)
    else:
        ok = run_case(dtype, args.B, args.N, args.H, args.seed, with_fwd=False)
    print("\n=== ALL PASS ===" if ok else "\n=== SOME CASES FAILED ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

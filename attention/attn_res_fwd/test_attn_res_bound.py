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
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch_npu  # noqa: F401

import cann_ops_transformer  # noqa: F401
from cann_ops_transformer.ops import attn_res
from cann_ops_transformer.ops import attn_res_fwd, attention_res_grad

# data_compare 与仓内 attenres_lite.py 保持一致
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR / "tests" / "pytest") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "tests" / "pytest"))
from precision_compare import data_compare  # noqa: E402

DEFAULT_NORM_EPS = 1e-5
# BF16 相对误差阈值 2^-7，与 ACLNN / pytest 一致
BF16_MERE_THRESHOLD = 1.0 / 128.0


def attn_res_golden(prefix_sum, block_residual, proj_weight, norm_weight,
                    grad_output, norm_eps=DEFAULT_NORM_EPS):
    """Golden：与 attenres_lite.apply_attn_res 等价的 autograd backward（NPU float32 标杆）。

    返回 hidden、inv_rms、probs（fwd 保存给反向的中间量）以及 4 个梯度。
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
    probs = scores.softmax(-1).unsqueeze(1)
    hidden = torch.matmul(probs, v).squeeze(1)
    inv_rms = torch.rsqrt(variance + norm_eps).squeeze(-1)

    hidden.backward(grad_output.detach().float())
    return (hidden.to(prefix_sum.dtype),
            inv_rms,
            probs.squeeze(1),
            ps.grad.to(prefix_sum.dtype),
            br.grad.to(block_residual.dtype),
            pw.grad.view(1, -1).to(proj_weight.dtype),
            nw.grad.to(norm_weight.dtype))


class _Tee:
    """同时输出到终端与日志文件的流包装。"""

    def __init__(self, stream, logf):
        self._stream = stream
        self._logf = logf

    def write(self, data):
        self._stream.write(data)
        self._logf.write(data)
        self._logf.flush()

    def flush(self):
        self._stream.flush()
        self._logf.flush()


def check(name, npu, cpu, diff_thd=BF16_MERE_THRESHOLD, pct_thd=BF16_MERE_THRESHOLD,
          max_diff_hd=BF16_MERE_THRESHOLD, show_points=False):
    """使用仓内 data_compare 进行精度比对（与 attenres_lite.py 相同阈值体系）。"""
    a = npu.detach().float().cpu().numpy()
    b = cpu.detach().float().cpu().numpy()
    result, pct, max_err = data_compare(a, b, diff_thd, pct_thd, max_diff_hd)
    ok = result in ("success", "Pass")
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: data_compare={result} pct={pct:.6f}% "
          f"max_rel_err={max_err:.6e}")
    if show_points:
        af = a.flatten()
        bf = b.flatten()
        abs_err = np.abs(af - bf)
        idx = int(abs_err.argmax())
        lo = max(0, idx - 2)
        hi = min(abs_err.size, idx + 3)
        print(f"    {name} 最大误差点及前后2点 (元素数 {abs_err.size}, 原始shape {tuple(npu.shape)}):")
        for i in range(lo, hi):
            coord = np.unravel_index(i, tuple(npu.shape))
            print(f"      flat[{i}] coord={coord} npu={af[i].item():.6e} "
                  f"gold={bf[i].item():.6e} abs_err={abs_err[i].item():.6e}")
    return ok

def _attn_res_bwd_core(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_output: torch.Tensor,
    inv_rms: torch.Tensor,
    probs: torch.Tensor,
    use_matmul_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对应 apply_attn_res / apply_attn_res_cust 的反向。

    use_matmul_out=True 时，out=matmul(probs, v) 的反向用 matmul，
    与 apply_attn_res 的 autograd 对齐；False 时用逐元素乘加，与 apply_attn_res_cust 对齐。

    使用正向保存的:
      inv_rms = rsqrt(mean(v^2) + eps)   # [num_tokens, num_blocks+1, 1]
      probs   = softmax(scores)          # [num_tokens, num_blocks+1] 或 [num_tokens, 1, num_blocks+1]
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    hidden_size = v_float.shape[-1]

    inv_rms = inv_rms.float()
    if inv_rms.dim() == 2:
        inv_rms = inv_rms.unsqueeze(-1)
    probs = probs.float()
    if probs.dim() == 3:
        probs = probs.squeeze(1)

    k = v_float * inv_rms

    norm_w = norm_weight.float()
    proj_w = proj_weight.squeeze(0).float()
    score_weight = norm_w * proj_w

    grad_out = grad_output.float()

    # out = matmul(probs.unsqueeze(1), v)  或  sum(probs * v)
    if use_matmul_out:
        # 对应 apply_attn_res: hidden = matmul(probs.unsqueeze(1), v_float).squeeze(1)
        grad_out_u = grad_out.unsqueeze(1)  # [T, 1, H]
        probs_u = probs.unsqueeze(1)  # [T, 1, B]
        grad_probs = torch.matmul(grad_out_u, v_float.transpose(-1, -2)).squeeze(1)  # [T, B]
        grad_v_from_out = torch.matmul(probs_u.transpose(-1, -2), grad_out_u)  # [T, B, H]
        # 原逐元素写法（与 matmul 数学等价，但 g_probs 归约顺序不同会有 ulp 差）：
        # grad_probs = (grad_out.unsqueeze(1) * v_float).sum(-1)
        # grad_v_from_out = grad_out.unsqueeze(1) * probs.unsqueeze(-1)
    else:
        # 对应 apply_attn_res_cust: hidden = (probs.unsqueeze(-1) * v_float).sum(dim=1)
        grad_probs = (grad_out.unsqueeze(1) * v_float).sum(-1)
        grad_v_from_out = grad_out.unsqueeze(1) * probs.unsqueeze(-1)

    # softmax backward：显式调用 aten::_softmax_backward_data，与 autograd 路径对齐
    # grad_scores = torch.ops.aten._softmax_backward_data(grad_probs, probs, -1, grad_probs.dtype)
    # 原手写 softmax backward（与 aten 数学等价，但归约顺序不同，会有 ulp 差）：
    grad_scores = probs * (grad_probs - (grad_probs * probs).sum(-1, keepdim=True))

    # scores = sum(k * score_weight)
    grad_k = grad_scores.unsqueeze(-1) * score_weight
    grad_score_weight = (grad_scores.unsqueeze(-1) * k).sum(dim=(0, 1))

    grad_norm_weight = grad_score_weight * proj_w
    grad_proj_weight = (grad_score_weight * norm_w).unsqueeze(0)

    # k = v * inv_rms
    grad_v_from_k = grad_k * inv_rms
    grad_inv_rms = (grad_k * v_float).sum(-1, keepdim=True)

    # inv_rms = rsqrt(mean(v^2) + eps)
    grad_variance = grad_inv_rms * (-0.5) * inv_rms.pow(3)
    # mean(v^2) backward：与 PyTorch 同序 (g/H)*2v，避免 ulp 差
    grad_v_from_var = (grad_variance / hidden_size) * (2.0 * v_float)
    # 原写法（数学等价，但乘除顺序不同，大 H 会有 ulp 差）：
    # grad_v_from_var = grad_variance * (2.0 * v_float / hidden_size)

    grad_v = grad_v_from_out + grad_v_from_k + grad_v_from_var

    num_blocks = block_residual.shape[1]
    grad_block_residual = grad_v[:, :num_blocks, :].to(block_residual.dtype)
    grad_prefix_sum = grad_v[:, num_blocks, :].to(prefix_sum.dtype)

    return grad_prefix_sum, grad_block_residual, grad_proj_weight.to(proj_weight.dtype), grad_norm_weight.to(
        norm_weight.dtype
    )
def _make_inputs(dtype, B, N, H, seed, grad_scale, input_mode="random"):
    """生成输入张量。input_mode='ones' 时全部为 1，用于简化基线/复现对比。"""
    torch.manual_seed(seed)
    if input_mode == "ones":
        prefix_sum = torch.ones(B, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        block_residual = torch.ones(B, N, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        norm_weight = torch.ones(H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        proj_weight = torch.ones(1, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        grad_output = torch.ones(B, H, dtype=dtype, device="cpu").to("npu") * grad_scale
    else:
        # CPU 生成再 .npu()，生成顺序与 test_attn_res_precision.py 保持一致
        prefix_sum = torch.randn(B, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        block_residual = torch.randn(B, N, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        norm_weight = torch.randn(H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        proj_weight = torch.randn(1, H, dtype=dtype, device="cpu").to("npu").requires_grad_(True)
        # 反向输入梯度放大 grad_scale 倍，验证相对误差是否随输入幅度变化
        grad_output = torch.randn(B, H, dtype=dtype, device="cpu").to("npu") * grad_scale
    return prefix_sum, block_residual, norm_weight, proj_weight, grad_output


def run_case(dtype, B, N, H, seed, mode="bound", show_points=False, grad_scale=1.0,
             input_mode="random"):
    print(f"\n===== mode={mode} dtype={dtype} B={B} N={N} H={H} seed={seed} "
          f"grad_scale={grad_scale} input_mode={input_mode} =====")

    prefix_sum, block_residual, norm_weight, proj_weight, grad_output = _make_inputs(
        dtype, B, N, H, seed, grad_scale, input_mode)

    g_hidden, g_inv_rms, g_probs, g_ps, g_br, g_pw, g_nw = attn_res_golden(
        prefix_sum, block_residual, proj_weight, norm_weight, grad_output)

    results = []
    if mode == "bound":
        # 一次调用：attn_res 自动走 fwd+bwd，同时比较正反向
        hidden = attn_res(prefix_sum, block_residual, proj_weight, norm_weight,
                          norm_eps=DEFAULT_NORM_EPS)
        hidden.backward(grad_output)
        results.append(check("fwd.hidden", hidden, g_hidden, show_points=show_points))
        results.append(check("bwd.grad_prefix_sum", prefix_sum.grad, g_ps, show_points=show_points))
        results.append(check("bwd.grad_block_residual", block_residual.grad, g_br, show_points=show_points))
        results.append(check("bwd.grad_proj_weight", proj_weight.grad, g_pw, show_points=show_points))
        results.append(check("bwd.grad_norm_weight", norm_weight.grad, g_nw, show_points=show_points))
    elif mode == "fwd":
        # 单独测正向：比较 hidden 及保存给反向的中间量 inv_rms/probs
        hidden, inv_rms, probs = attn_res_fwd(
            prefix_sum.detach(), block_residual.detach(),
            proj_weight.detach(), norm_weight.detach(),
            norm_eps=DEFAULT_NORM_EPS, need_backward=True)
        results.append(check("fwd.hidden", hidden, g_hidden, show_points=show_points))
        results.append(check("fwd.inv_rms", inv_rms, g_inv_rms, show_points=show_points))
        results.append(check("fwd.probs", probs, g_probs, show_points=show_points))
    elif mode == "bwd":
        # 单独测反向：fwd 算子产出 inv_rms/probs，手动喂给 attention_res_grad
        _, inv_rms, probs = attn_res_fwd(prefix_sum.detach(), block_residual.detach(),
                                         proj_weight.detach(), norm_weight.detach(),
                                         norm_eps=DEFAULT_NORM_EPS, need_backward=True)
        g_ps_op, g_br_op, g_pw_op, g_nw_op = attention_res_grad(
            prefix_sum.detach(), block_residual.detach(), proj_weight.detach(),
            norm_weight.detach(), grad_output.detach(), inv_rms, probs)
        results.append(check("bwd.grad_prefix_sum", g_ps_op, g_ps, show_points=show_points))
        results.append(check("bwd.grad_block_residual", g_br_op, g_br, show_points=show_points))
        results.append(check("bwd.grad_proj_weight", g_pw_op, g_pw, show_points=show_points))
        results.append(check("bwd.grad_norm_weight", g_nw_op, g_nw, show_points=show_points))
    elif mode == "mid":
        # 同输入对照：fwd 算子保存的中间量 vs 正向标杆(golden)的中间量，
        # 分别传给反向算子，比较两组反向输出 —— 区分误差来自中间量还是反向 kernel 本身
        _, inv_rms_op, probs_op = attn_res_fwd(prefix_sum.detach(), block_residual.detach(),
                                               proj_weight.detach(), norm_weight.detach(),
                                               norm_eps=DEFAULT_NORM_EPS, need_backward=True)
        g_ps_op, g_br_op, g_pw_op, g_nw_op = attention_res_grad(
            prefix_sum.detach(), block_residual.detach(), proj_weight.detach(),
            norm_weight.detach(), grad_output.detach(), inv_rms_op, probs_op)
        g_ps_gm, g_br_gm, g_pw_gm, g_nw_gm = attention_res_grad(
            prefix_sum.detach(), block_residual.detach(), proj_weight.detach(),
            norm_weight.detach(), grad_output.detach(), g_inv_rms, g_probs)

        # 1) 中间量本身：fwd 算子 vs 标杆
        results.append(check("mid.inv_rms_op_vs_golden", inv_rms_op, g_inv_rms, show_points=show_points))
        results.append(check("mid.probs_op_vs_golden", probs_op, g_probs, show_points=show_points))
        # 2) 两组反向输出互相对比
        results.append(check("mid.opmid_vs_goldenmid.grad_prefix_sum", g_ps_op, g_ps_gm, show_points=show_points))
        results.append(check("mid.opmid_vs_goldenmid.grad_block_residual", g_br_op, g_br_gm, show_points=show_points))
        results.append(check("mid.opmid_vs_goldenmid.grad_proj_weight", g_pw_op, g_pw_gm, show_points=show_points))
        results.append(check("mid.opmid_vs_goldenmid.grad_norm_weight", g_nw_op, g_nw_gm, show_points=show_points))
        # 3) 各自 vs golden 梯度
        results.append(check("mid.opmid.grad_prefix_sum", g_ps_op, g_ps, show_points=show_points))
        results.append(check("mid.opmid.grad_block_residual", g_br_op, g_br, show_points=show_points))
        results.append(check("mid.opmid.grad_proj_weight", g_pw_op, g_pw, show_points=show_points))
        results.append(check("mid.opmid.grad_norm_weight", g_nw_op, g_nw, show_points=show_points))
        results.append(check("mid.goldenmid.grad_prefix_sum", g_ps_gm, g_ps, show_points=show_points))
        results.append(check("mid.goldenmid.grad_block_residual", g_br_gm, g_br, show_points=show_points))
        results.append(check("mid.goldenmid.grad_proj_weight", g_pw_gm, g_pw, show_points=show_points))
        results.append(check("mid.goldenmid.grad_norm_weight", g_nw_gm, g_nw, show_points=show_points))
    elif mode == "vs":
        # 同输入对照：手动两段式反向 vs 绑定自动反向，两组梯度互相比对，并各自 vs golden
        _, inv_rms, probs = attn_res_fwd(prefix_sum.detach(), block_residual.detach(),
                                         proj_weight.detach(), norm_weight.detach(),
                                         norm_eps=DEFAULT_NORM_EPS, need_backward=True)
        g_ps_op, g_br_op, g_pw_op, g_nw_op = attention_res_grad(
            prefix_sum.detach(), block_residual.detach(), proj_weight.detach(),
            norm_weight.detach(), grad_output.detach(), inv_rms, probs)

        ps2 = prefix_sum.detach().requires_grad_(True)
        br2 = block_residual.detach().requires_grad_(True)
        pw2 = proj_weight.detach().requires_grad_(True)
        nw2 = norm_weight.detach().requires_grad_(True)
        hidden2 = attn_res(ps2, br2, pw2, nw2, norm_eps=DEFAULT_NORM_EPS)
        hidden2.backward(grad_output)

        results.append(check("vs.manual_vs_bound.grad_prefix_sum", g_ps_op, ps2.grad, show_points=show_points))
        results.append(check("vs.manual_vs_bound.grad_block_residual", g_br_op, br2.grad, show_points=show_points))
        results.append(check("vs.manual_vs_bound.grad_proj_weight", g_pw_op, pw2.grad, show_points=show_points))
        results.append(check("vs.manual_vs_bound.grad_norm_weight", g_nw_op, nw2.grad, show_points=show_points))
        results.append(check("vs.manual.grad_prefix_sum", g_ps_op, g_ps, show_points=show_points))
        results.append(check("vs.manual.grad_block_residual", g_br_op, g_br, show_points=show_points))
        results.append(check("vs.manual.grad_proj_weight", g_pw_op, g_pw, show_points=show_points))
        results.append(check("vs.manual.grad_norm_weight", g_nw_op, g_nw, show_points=show_points))
        results.append(check("vs.bound.grad_prefix_sum", ps2.grad, g_ps, show_points=show_points))
        results.append(check("vs.bound.grad_block_residual", br2.grad, g_br, show_points=show_points))
        results.append(check("vs.bound.grad_proj_weight", pw2.grad, g_pw, show_points=show_points))
        results.append(check("vs.bound.grad_norm_weight", nw2.grad, g_nw, show_points=show_points))
    else:
        raise ValueError(f"unknown mode: {mode}")

    return all(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--H", type=int, default=7168)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--mode", default="bound", choices=["bound", "fwd", "bwd", "vs", "mid"],
                        help="bound=正反向绑定一次调用; fwd=单独测正向; bwd=单独测反向; "
                             "vs=同输入下手动两段式 vs 绑定自动反向互相对比; "
                             "mid=算子中间量 vs 标杆中间量分别喂反向并比较")
    parser.add_argument("--show-points", action="store_true",
                        help="PASS 时也打印最大误差点及前后2点")
    parser.add_argument("--grad-scale", type=float, default=1.0,
                        help="反向输入梯度放大倍数（默认 1.0）")
    parser.add_argument("--input-mode", default="random", choices=["random", "ones"],
                        help="random=randn 输入（默认）; ones=全一输入，用于简化基线")
    parser.add_argument("--log", default="",
                        help="日志文件路径（默认 ./test_attn_res_bound.log，追加写入）")
    args = parser.parse_args()

    # 每次 compare 的结果同时打印到终端并保存到日志
    log_path = Path(args.log) if args.log else SCRIPT_DIR / "test_attn_res_bound.log"
    logf = open(log_path, "a")
    logf.write("\n===== 运行时间: %s =====\n" % __import__("datetime").datetime.now())
    sys.stdout = _Tee(sys.stdout, logf)
    sys.stderr = _Tee(sys.stderr, logf)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ok = run_case(dtype, args.B, args.N, args.H, args.seed, mode=args.mode,
                  show_points=args.show_points, grad_scale=args.grad_scale,
                  input_mode=args.input_mode)
    print("\n=== ALL PASS ===" if ok else "\n=== SOME CASES FAILED ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

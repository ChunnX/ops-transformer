#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""res 路径数值探针: torch 复刻 arch22 kernel SinkhornGrad 算法, 三方对比定位 v1/v2b 失败根因。

三方:
  gold : verify_mhc_bwd.backward_golden 的 res 路径 (_sinkhorn_grad + _exp_grad)
  sim  : torch 逐位复刻 MhcPreGradKernel::SinkhornGrad (含 kernel 特有处理:
         分母 = stored_sum + eps 再加一次; 末轮 SOFTMAX 段替代 row反传+_exp_grad)
  kernel: 真反向 kernel (torch-mids 口径, 等价 diag B 变体)

变体开关(分离根因):
  sim_full        : double_eps=True,  softmax_last=True   (完整复刻)
  sim_noeps       : double_eps=False, softmax_last=True   (去掉双 eps)
  sim_goldlast    : double_eps=True,  softmax_last=False  (末轮换 golden 式 row反传+_exp_grad)
对比层: z_res 梯度层 + 派生 ga_res / grad_bias res 段。
"""
import os
import sys

REPO = "/home/ops-transformer-ds-v4.1"
sys.path.insert(0, os.path.join(REPO, "build_out"))

import torch
import torch_npu  # noqa: F401

import verify_mhc_bwd as V
from cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_premix_backward import (  # noqa: E402
    mhc_pre_sinkhorn_premix_backward,
)


def kernel_sim(g, sum_out, norm_out, eps, double_eps=True, softmax_last=True):
    """torch 复刻 arch22 MhcPreGradKernel::SinkhornGrad (公式层, 不含布局转换)。

    kernel 语义 (mhc_pre_grad_kernel.h L397-582):
      for iter = iters-1 .. 1:
        colS' = sums[2i+1](stored, 已含 eps) + eps    (L434)
        rowS' = sums[2i](stored, 已含 eps) + eps       (L477)
        g = g/colS' - rowsum_r(g*xrn)/colS'^2          (L438-474, col 反传)
        g = g/rowS' - rowsum_j(g*xrn)/rowS'            (L483-497, row 反传)
      末轮 iter=0:
        colS0' = sums[1] + eps                          (L516)
        g = g/colS0' - rowsum_r(g*xrn0)/colS0'^2        (L519-553)
        g = xrn0 * (g - rowsum_j(g*xrn0))               (L557-568, SOFTMAX 段)
    """
    bs, seq_len, n, _ = g.shape
    iters = sum_out.shape[0] // 2
    sums = sum_out.reshape(iters, 2, bs, seq_len, n)
    norms = norm_out.reshape(iters, 2, bs, seq_len, n, n)
    g = g.clone()
    for i in range(iters - 1, -1, -1):
        xrn = norms[i][0]
        add = eps if double_eps else 0.0
        colS = (sums[i][1] + add).view(bs, seq_len, 1, n)
        g = g / colS - (g * xrn).sum(-2, keepdim=True) / (colS ** 2)
        if i == 0 and softmax_last:
            g = xrn * (g - (g * xrn).sum(-1, keepdim=True))
            return g
        rowS = (sums[i][0] + add).view(bs, seq_len, n, 1)
        g = g / rowS - (g * xrn).sum(-1, keepdim=True) / rowS
    return g  # softmax_last=False: 末轮 row 反传输出 = d comb0, 调用方接 _exp_grad


def zres_of(hcbn, inv_rms, alpha, bias, b, s, n):
    nf = hcbn * inv_rms
    return (nf[..., 2 * n:] * alpha[2] + bias[2 * n:]).reshape(b, s, n, n), nf


def rel(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


def run_scene(scene, data):
    (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
     pre_mix, grad_pre, hc_eps) = data
    h_pre, hcbn, inv_rms, sum_out, norm_out = V.forward_ref(
        x, phi, alpha, bias, V.N, V.ITERS, hc_eps)
    b, s, n = x.shape[0], x.shape[1], x.shape[2]
    zres, nf = zres_of(hcbn, inv_rms, alpha, bias, b, s, n)
    eps = float(hc_eps)

    # gold res 路径
    gold_z = V._exp_grad(zres, V._sinkhorn_grad(grad_h_res, sum_out, norm_out))
    ga_res_gold = (nf[..., 2 * n:] * gold_z.flatten(2)).sum()
    bias_res_gold = gold_z.sum((0, 1)).flatten()

    # sim 变体
    sims = {
        "sim_full    ": kernel_sim(grad_h_res, sum_out, norm_out, eps, True, True),
        "sim_noeps   ": kernel_sim(grad_h_res, sum_out, norm_out, eps, False, True),
        "sim_goldlast": V._exp_grad(
            zres, kernel_sim(grad_h_res, sum_out, norm_out, eps, True, False)),
    }

    # 真 kernel (torch-mids, 同 diag B)
    outs = mhc_pre_sinkhorn_premix_backward(
        grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
        h_pre, hcbn, inv_rms, sum_out, norm_out, pre_mix, grad_pre, hc_eps)
    torch.npu.synchronize()
    ga_res_k = outs[2][2].item()
    bias_res_k = outs[3][(V.N + V.N):].float()

    print("\n##### scene {} #####".format(scene))
    print("  gold    ga_res={:.6e}".format(ga_res_gold.item()))
    print("  kernel  ga_res={:.6e}  diff_vs_gold={:.3e} rel={:.2e}".format(
        ga_res_k, abs(ga_res_k - ga_res_gold.item()),
        abs(ga_res_k - ga_res_gold.item()) / max(abs(ga_res_gold.item()), 1e-12)))
    for name, z in sims.items():
        ga = (nf[..., 2 * n:] * z.flatten(2)).sum()
        bias_res = z.sum((0, 1)).flatten()
        print("  {} ga_res={:.6e}  diff_vs_gold={:.3e} rel={:.2e} | "
              "zgrad_maxdiff={:.3e} zgrad_rel={:.2e} | bias_res_maxdiff={:.3e} "
              "bias_vs_kernel={:.3e}".format(
                  name, ga.item(), abs(ga.item() - ga_res_gold.item()),
                  abs(ga.item() - ga_res_gold.item()) / max(abs(ga_res_gold.item()), 1e-12),
                  (z - gold_z).abs().max().item(), rel(z, gold_z),
                  (bias_res - bias_res_gold).abs().max().item(),
                  (bias_res - bias_res_k).abs().max().item()))
    # 数据特征: sums 最小值(放大因子诊断)
    sums2 = sum_out.reshape(-1, 2, V.BS, V.S, V.N)
    print("  data: min row_sum={:.3e} min col_sum={:.3e} | grad_h_res absmax={:.3e}".format(
        sums2[:, 0].min().item(), sums2[:, 1].min().item(),
        grad_h_res.abs().max().item()))


def main():
    # 按 verify 顺序重放 4 个场景数据生成(仅 RNG 消费者), 保证与失败 run 同数据
    datas = {}
    for scene, pm, gpr in (("v1", False, False), ("v1b", False, True),
                           ("v2", True, True), ("v2b", True, False)):
        datas[scene] = V.make_case(pm, gpr)
    run_scene("v1", datas["v1"])
    run_scene("v1b", datas["v1b"])
    run_scene("v2", datas["v2"])
    run_scene("v2b", datas["v2b"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

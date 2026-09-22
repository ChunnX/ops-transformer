#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""autograd 裁决实验: forward_ref 同构 sinkhorn 建图, 反向真值 vs 解析公式三方对比。

裁决 v1/v2b res 路径失败: kernel 末轮 SOFTMAX 段 vs golden (row0反传+_exp_grad) 谁等于
autograd 真值。纯 CPU, 小规模。
"""
import os
import sys

sys.path.insert(0, "/home/ops-transformer-ds-v4.1/build_out")

import torch

import verify_mhc_bwd as V  # noqa: E402


def forward_autograd(z, iters, eps):
    """forward_ref 同构 (首行归一除 raw r + eps; 其后除 S+eps), 建图。"""
    e = (z - z.max(-1, keepdim=True).values).exp()
    sums, norms = [], []
    r = e.sum(-1)
    sums.append(r + eps)
    comb = e / r[..., None] + eps
    norms.append(comb)
    colS = comb.sum(-2)
    sums.append(colS + eps)
    comb = comb / (colS[..., None, :] + eps)
    norms.append(comb)
    for _ in range(1, iters):
        rowS = comb.sum(-1)
        sums.append(rowS + eps)
        comb = comb / (rowS[..., :, None] + eps)
        norms.append(comb)
        colS = comb.sum(-2)
        sums.append(colS + eps)
        comb = comb / (colS[..., None, :] + eps)
        norms.append(comb)
    return comb, torch.stack(sums), torch.stack(norms)


def _exp_grad(x, y_grad):
    nd = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(nd, device=x.device)[None, :]
    is_max = idx == arange_idx
    y = (x - x_max).exp()
    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)
    return y * y_grad - is_max * sum_all


def sinkhorn_grad(g, sums, norms, iters, eps_add, softmax_last, fix=False):
    """kernel_sim 同构: 双 eps 开关 + 末轮 SOFTMAX/row反传 开关。

    fix=True: 修复变体 — 反传不再加 eps(存储值即除数) + SOFTMAX 权重减 eps。
    """
    bs, s, n, _ = g.shape
    sums = sums.reshape(iters, 2, bs, s, n)
    norms = norms.reshape(iters, 2, bs, s, n, n)
    g = g.clone()
    for i in range(iters - 1, -1, -1):
        xrn = norms[i][0]
        if fix:
            colS = sums[i][1].view(bs, s, 1, n)
        else:
            colS = (sums[i][1] + eps_add).view(bs, s, 1, n)
        g = g / colS - (g * xrn).sum(-2, keepdim=True) / (colS ** 2)
        if i == 0 and softmax_last:
            w = xrn - eps_add if fix else xrn
            return w * (g - (g * w).sum(-1, keepdim=True))
        if fix:
            rowS = sums[i][0].view(bs, s, n, 1)
        else:
            rowS = (sums[i][0] + eps_add).view(bs, s, n, 1)
        g = g / rowS - (g * xrn).sum(-1, keepdim=True) / rowS
    return g  # softmax_last=False: 末轮 row 反传输出 = dL/de


def run(bs, s, n, iters, eps, zscale, tag, dt=torch.float64):
    torch.manual_seed(42)
    z = torch.randn(bs, s, n, n, dtype=dt) * zscale
    z.requires_grad_(True)
    comb, sums, norms = forward_autograd(z, iters, eps)
    torch.manual_seed(7)
    gout = torch.randn(bs, s, n, n, dtype=dt)
    (comb * gout).sum().backward()
    truth = z.grad.detach()

    # gold: 无额外 add, 末轮 row 反传 + _exp_grad
    dz_gold = _exp_grad(z.detach(), sinkhorn_grad(
        gout, sums.detach(), norms.detach(), iters, 0.0, False))
    # kernel 复刻: add=eps, 末轮 SOFTMAX
    dz_kernel = sinkhorn_grad(
        gout, sums.detach(), norms.detach(), iters, eps, True)
    # kernel 变体: add=eps, 末轮 row 反传 + _exp_grad (分离 add 与 SOFTMAX)
    dz_kern_rowlast = _exp_grad(z.detach(), sinkhorn_grad(
        gout, sums.detach(), norms.detach(), iters, eps, False))

    def rpt(name, d):
        print("  {:14s} maxdiff_vs_truth={:.3e}  rel={:.2e}".format(
            name, (d - truth).abs().max().item(),
            (d - truth).abs().max().item() / truth.abs().max().item()))

    print("[{}] bs={} s={} n={} iters={} eps={:g} zscale={:g} dtype={}".format(
        tag, bs, s, n, iters, eps, zscale, str(dt).replace("torch.", "")))
    rpt("gold", dz_gold)
    rpt("kernel(SOFT)", dz_kernel)
    rpt("kernel(rowE)", dz_kern_rowlast)
    # 公式路径间互差 (不看 truth, 看 SOFTMAX vs gold 的纯公式差)
    print("  SOFTMAX_vs_gold maxdiff={:.3e}  gold_absmax={:.3e}  SOFT_absmax={:.3e}".format(
        (dz_kernel - dz_gold).abs().max().item(),
        dz_gold.abs().max().item(), dz_kernel.abs().max().item()))
    r0 = (sums.reshape(iters, 2, bs, s, n)[0][0] - eps)
    c0 = (sums.reshape(iters, 2, bs, s, n)[0][1] - eps)
    print("  data: r0 min={:.3e} max={:.3e} | c0 min={:.3e} max={:.3e}".format(
        r0.min().item(), r0.max().item(), c0.min().item(), c0.max().item()))


def run_real(scene, data, dt):
    """用缓存的真实数据(与失败 run 同 RNG)做 autograd 裁决。"""
    (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
     pre_mix, grad_pre, hc_eps) = data
    eps = float(hc_eps)
    b, s, n = x.shape[0], x.shape[1], x.shape[2]
    iters = V.ITERS

    x = x.to("cpu", dt)
    phi = phi.to("cpu", dt)
    alpha = alpha.to("cpu", dt)
    bias = bias.to("cpu", dt)
    gout = grad_h_res.to("cpu", dt)

    # zres 重算 (forward_ref 前半, 不建图)
    xf = x.reshape(b, s, n * x.shape[-1])
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    hcbn = xf @ phi.t()
    nf = hcbn * inv_rms
    zres = (nf[..., 2 * n:] * alpha[2] + bias[2 * n:]).reshape(b, s, n, n)

    z = zres.detach().clone().requires_grad_(True)
    comb, sums, norms = forward_autograd(z, iters, eps)
    (comb * gout).sum().backward()
    truth = z.grad.detach()

    dz_gold = _exp_grad(zres.detach(), sinkhorn_grad(
        gout, sums.detach(), norms.detach(), iters, 0.0, False))
    dz_kernel = sinkhorn_grad(gout, sums.detach(), norms.detach(), iters, eps, True)
    dz_kern_rowlast = _exp_grad(zres.detach(), sinkhorn_grad(
        gout, sums.detach(), norms.detach(), iters, eps, False))
    dz_kernel_fix = sinkhorn_grad(
        gout, sums.detach(), norms.detach(), iters, eps, True, fix=True)

    def rpt(name, d):
        print("  {:14s} maxdiff_vs_truth={:.3e}  rel={:.2e}".format(
            name, (d - truth).abs().max().item(),
            (d - truth).abs().max().item() / truth.abs().max().item()))

    print("[real:{}] dtype={}".format(scene, str(dt).replace("torch.", "")))
    rpt("gold", dz_gold)
    rpt("kernel(SOFT)", dz_kernel)
    rpt("kernel(rowE)", dz_kern_rowlast)
    rpt("kernel(FIX)", dz_kernel_fix)
    sums2 = sums.reshape(iters, 2, b, s, n)
    print("  data: c0 min={:.3e} | truth_absmax={:.3e}".format(
        (sums2[0][1] - eps).min().item(), truth.abs().max().item()))


def main():
    run(2, 8, 4, 3, 0.0, 1.0, "eps=0 隔离")
    run(2, 8, 4, 3, 1e-6, 1.0, "eps=1e-6 常规")
    run(2, 8, 4, 3, 1e-6, 2.0, "z 尖锐")
    run(2, 128, 4, 20, 1e-6, 1.0, "全尺寸真实")
    print("===== float32 同实验 =====")
    run(2, 8, 4, 3, 0.0, 1.0, "fp32 eps=0", torch.float32)
    run(2, 8, 4, 3, 1e-6, 1.0, "fp32 eps=1e-6", torch.float32)
    run(2, 8, 4, 3, 1e-6, 2.0, "fp32 z尖锐", torch.float32)
    run(2, 128, 4, 20, 1e-6, 1.0, "fp32 全尺寸真实", torch.float32)
    print("===== 真实失败数据 (make_case 按序重放, 与 verify 同 RNG 流) =====")
    scenes = [("v1", False, False), ("v1b", False, True),
              ("v2", True, True), ("v2b", True, False)]
    cache = {}
    for name, pm, gpr in scenes:  # 按序生成一次, 避免二次消费 RNG 漂移
        cache[name] = V.make_case(pm, gpr)
    for name, _, _ in scenes:
        run_real(name, cache[name], torch.float64)
        run_real(name, cache[name], torch.float32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

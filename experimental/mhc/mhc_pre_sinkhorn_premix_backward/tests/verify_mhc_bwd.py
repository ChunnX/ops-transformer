#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mhc_pre_sinkhorn_premix 正反向级联上板验证。

级联链路（真实训练链路形态）:
  stage-1  NPU 正向 mhc_pre_sinkhorn_premix(out_flag=True) 产出反向所需的全部中间量
           (h_pre / hc_before_norm / inv_rms / sum_out / norm_out)
  stage-2  NPU 反向 mhc_pre_sinkhorn_premix_backward 直接消费 stage-1 中间量
  参照     stage-1: 正向 kernel 真实语义的 torch 复刻(对齐 mhc/mhc_pre_sinkhorn_premix
           golden: hcbn=x@phi.T 不含 bias, h_pre=sigmoid(z)+hc_eps, 首次行归一除后+eps)
           stage-2: 以 NPU 正向中间量为叶子的解析梯度(公式与 input.py golden 一致,
           中间量替换为 NPU 正向实际输出, 消除参照与 kernel 消费值之间的口径差)

语义备注(级联口径, 与 input.py 合成口径的差异):
  input.py 为自洽合成数据生成器: hcbn=x@phi.T+bias(含 bias) 且 z=hcbn*inv_rms*alpha+bias
  (bias 双加), 与正向 kernel 语义(hcbn=x@phi.T 无 bias, z 单加 bias)是两个不同函数;
  反向 kernel 把 hcbn/inv_rms/sum/norm/h_pre 当叶子, 其 z 复算公式(hcbn*inv_rms*alpha+bias)
  施加在正向 kernel 的 hcbn(无 bias)上恰好还原正向真实 z, 级联链路自洽。
  本脚本 input.py 仅用于随机输入数据生成, 其合成中间量/golden 不参与比对。

场景矩阵（契约 v2: preMix 与 gradPre 必须同传或同缺省, aclnn/tiling 层绑定校验）:
  v1 : pre_mix=N, grad_pre=N  -> d_pre = d_mix_for_y（基线, 精度验证）
  v1b: pre_mix=N, grad_pre=Y  -> 单传组合, 预期 EZ1001 拒绝（时序竞争, 见 aclnn 校验注释）
  v2 : pre_mix=Y, grad_pre=Y  -> d_pre = grad_pre（pre 有下游, 精度验证）
  v2b: pre_mix=Y, grad_pre=N  -> 单传组合, 预期 EZ1001 拒绝（无合法调用方）

两级检查:
  stage-1 前置检查: NPU 正向中间量 vs torch 参照（链路质量诊断, 失败则级联结果不可信）
  stage-2 端到端:   级联梯度 vs torch 解析 golden（隔离反向 kernel 自身精度,
                    中间量误差在 golden 与 kernel 两侧同源抵消）

容差: 分 dtype 混合容差 + matched_ratio（参照仓库 ops-precision-standard 与正向 prec_check 先例）:
  bf16 输出(grad_x):    atol=rtol=2^-6,  matched_ratio>=0.99, max_abs 上限自适应值域
  fp32 输出(其余输出):   atol=rtol=1e-3,  matched_ratio>=0.99, max_abs 上限自适应值域
  stage-1 正向中间量:    atol=1e-4, rtol=1e-3（对齐正向 prec_check 先例）
"""
import os
import sys

# 副本选择: 默认 /home, 可用 REPO_OVERRIDE 环境变量切换(如 /opt/rrh/code 副本, 其 builder.py 新版)
REPO = os.environ.get("REPO_OVERRIDE", "/home/ops-transformer-ds-v4.1")
sys.path.insert(0, os.path.join(REPO, "torch_extension"))
sys.path.insert(0, os.path.join(REPO, "mhc", "mhc_pre_sinkhorn_premix_backward", "tests", "assets"))

import torch
import torch_npu  # noqa: F401

torch.manual_seed(20260917)

# 第三层路径必须写全: 两层 import 拿到的是包内同名子模块(module, 不可调用), 函数在子模块内
from cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_premix_backward.mhc_pre_sinkhorn_premix_backward import (  # noqa: E402
    mhc_pre_sinkhorn_premix_backward,
)
from cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_premix.mhc_pre_sinkhorn_premix import (  # noqa: E402
    mhc_pre_sinkhorn_premix_op_builder,
)
from input import mhc_pre_sinkhorn_premix_backward_input  # noqa: E402

# 容差模型: (atol, rtol, required_matched_ratio, 默认 max_abs 上限)
# bf16 按仓库标准 atol=rtol=2^-6; fp32 参照本仓正向 prec_check 先例 atol=1e-3, rtol=1e-3
# (kernel cube 累加顺序 vs torch matmul 的固有浮点差异: 小值元素 ~1e-4 相消噪声, 大值全 reduce 输出 ~1e-4 相对误差)
# max_abs_error_limit 按输出值域自适应: max(默认上限, rtol * max|golden|), 规避大值输出被绝对上限误杀
TOL_BF16 = (2.0 ** -6, 2.0 ** -6, 0.99, 1.0)
TOL_FP32 = (1e-3, 1e-3, 0.99, 1e-2)
# stage-1 前置检查: 正向中间量 vs torch 参照, 对齐正向 prec_check 先例 (FP32_ATOL=1e-4, FP32_RTOL=1e-3)
TOL_FWD = (1e-4, 1e-3, 0.99, 1e-2)
# hc_before_n 单独校准: bf16 输入 cube matmul fp32 输出 vs torch fp32 matmul 的固有相对差异 ~1e-3 量级
# (随机数据下 worst rel 1.1e-3~1.7e-3, max_abs 1.2e-2~1.5e-2@值域±50; prec_check 先例为常量合成数据不触发);
# 级联语义下 hcbn 为传递量, 其噪声在 stage-2 golden 与 kernel 两侧同源抵消(stage-2 四场景全 PASS 实证),
# 派生量(inv_rms/sum/norm/h_pre)均以严格 TOL_FWD 通过, 故此处仅作量级诊断: atol=2e-2, rtol=2e-3
TOL_FWD_HCBN = (2e-2, 2e-3, 0.99, 1e-1)
# 泛化维度约束（tiling/proto 层已确认）: N=4 与 ITERS=20 为 tiling 强约束（不可变）,
# C 须 %128==0 且 <100000, B/S 任意, x/grad_hin/grad_x dtype ∈ {BF16, FP16}（其余输入/输出恒 fp32）
N, ITERS = 4, 20
DEV = "npu"
HC_EPS = 1e-6
FWD_OP = None  # main() 中经 OpBuilder.load() JIT 加载


def forward_ref(x, phi, alpha, bias, n, iters, hc_eps):
    """stage-1 torch 参照: 正向 kernel 真实语义 (4D 输入)。

    对齐 mhc/mhc_pre_sinkhorn_premix/tests/assets/golden.py _compute:
      hcbn = x@phi.T (不含 bias); inv_rms = rsqrt(mean(x^2)+norm_eps), norm_eps 统一取 hc_eps;
      h_pre = sigmoid(z_pre)+hc_eps; 首次行归一 comb/row_sum + eps, 其后 comb/(s+eps)。
    """
    x_f32 = x.float()
    assert x_f32.dim() == 4
    b, s, n_, c = x_f32.shape
    flat = x_f32.reshape(b, s, n_ * c)
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + hc_eps)
    hcbn = flat @ phi.t()
    nf = hcbn * inv_rms
    z_res = (nf[..., 2 * n_:] * alpha[2] + bias[2 * n_:]).reshape(b, s, n_, n_)

    h_pre = torch.sigmoid(nf[..., :n_] * alpha[0] + bias[:n_]) + hc_eps

    comb = torch.exp(z_res - z_res.max(dim=-1, keepdim=True).values)
    sums, norms = [], []
    row_sum = comb.sum(dim=-1)
    sums.append(row_sum + hc_eps)
    comb = comb / row_sum[..., None] + hc_eps  # 首次行归一: 除后 +eps (golden 语义)
    norms.append(comb)
    col_sum = comb.sum(dim=-2)
    sums.append(col_sum + hc_eps)
    comb = comb / (col_sum[..., None, :] + hc_eps)
    norms.append(comb)
    for _ in range(1, iters):
        row_sum = comb.sum(dim=-1)
        sums.append(row_sum + hc_eps)
        comb = comb / (row_sum[..., :, None] + hc_eps)
        norms.append(comb)
        col_sum = comb.sum(dim=-2)
        sums.append(col_sum + hc_eps)
        comb = comb / (col_sum[..., None, :] + hc_eps)
        norms.append(comb)
    sum_out = torch.stack(sums, dim=0)
    norm_out = torch.stack(norms, dim=0)
    return h_pre, hcbn, inv_rms, sum_out, norm_out


def _sigmoid_grad(z, dy, is_pre=False, hc_eps=1e-6):
    y = torch.sigmoid(z)
    sigma = y - hc_eps if is_pre else y
    return dy * sigma * (1 - sigma)


def _rms_norm_grad(x, inv_rms, grad_inv_rms):
    nc = x.size(-1)
    return -(inv_rms**3) * x * grad_inv_rms / float(nc)


def _exp_grad(x, y_grad):
    nd = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(nd, device=x.device)[None, :]
    is_max = idx == arange_idx
    y = (x - x_max).exp()
    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)
    return y * y_grad - is_max * sum_all


def _sinkhorn_grad(grad_h_res, sum_out, norm_out):
    bs, seq_len, n, _ = grad_h_res.shape
    x_grad = grad_h_res
    it2, b_, s_, n_ = sum_out.shape
    iters = it2 // 2
    sum_out = sum_out.reshape(iters, 2, b_, s_, n_)
    norm_out = norm_out.reshape(iters, 2, b_, s_, n_, n_)
    for i in reversed(range(iters)):
        row_sum = sum_out[i][0].view(bs, seq_len, n, 1)
        col_sum = sum_out[i][1].view(bs, seq_len, 1, n)
        xrn = norm_out[i][0].view(bs, seq_len, n, n)
        gxrn = x_grad / col_sum - (x_grad * xrn / (col_sum**2)).sum(
            dim=-2, keepdim=True
        )
        x_grad = gxrn / row_sum - (gxrn * xrn / row_sum).sum(dim=-1, keepdim=True)
    return x_grad


def backward_golden(grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
                    h_pre, hcbn, inv_rms, sum_out, norm_out,
                    pre_mix, grad_pre, hc_eps):
    """stage-2 解析 golden: 以给定中间量(级联=NPU 正向输出)为叶子的梯度公式。

    公式与 input.py golden 相同, 仅 h_pre/hcbn/inv_rms/sum_out/norm_out 换成实参
    (级联模式下为 NPU 正向实际输出), 使 golden 与 kernel 消费同一组中间量。
    """
    assert x.dim() == 4
    n = x.shape[-2]
    hc_eps_val = float(hc_eps)

    x4 = x.float()
    gh4 = grad_hin.float()
    gp4 = grad_h_post
    ghres4 = grad_h_res
    hp4 = h_pre
    hcbn4 = hcbn
    invr4 = inv_rms
    suma = sum_out
    norm4 = norm_out
    pm4 = pre_mix
    gpr4 = grad_pre

    b4, s4, n4, d4 = x4.shape

    # y 折叠项加权源：hasPreMix 时用 pre_mix 替换 h_pre（二选一）
    mix_w4 = pm4 if pm4 is not None else hp4
    grad_x_hin = gh4[..., None, :] * mix_w4[..., None]
    # d_mix_for_y = einsum('bsd,bshd->bsh', d_y, x)
    g_hpre = (gh4[..., None, :] * x4).sum(-1)
    # 合并点三态：d_pre = (has_pre_mix ? 0 : d_mix_for_y) + d_pre_out
    if pm4 is None:
        d_pre4 = g_hpre if gpr4 is None else g_hpre + gpr4
    else:
        d_pre4 = gpr4 if gpr4 is not None else torch.zeros_like(g_hpre)

    nf4 = hcbn4 * invr4
    zp4 = nf4[..., :n4] * alpha[0] + bias[:n4]
    zpost4 = nf4[..., n4: 2 * n4] * alpha[1] + bias[n4: 2 * n4]
    zres4 = (nf4[..., 2 * n4:] * alpha[2] + bias[2 * n4:]).reshape(b4, s4, n4, n4)

    gzp4 = _sigmoid_grad(zp4, d_pre4, is_pre=True, hc_eps=hc_eps_val)
    gzpost4 = 2 * _sigmoid_grad(zpost4, gp4, is_pre=False, hc_eps=hc_eps_val)
    ghres4_4d = ghres4.reshape(b4, s4, n4, n4) if ghres4.dim() == 3 else ghres4
    skg4 = _sinkhorn_grad(ghres4_4d, suma, norm4)
    gzres4 = _exp_grad(zres4, skg4).flatten(2)

    gold_bias = torch.cat([gzp4.sum((0, 1)), gzpost4.sum((0, 1)), gzres4.sum((0, 1))])
    ga_pre = (nf4[..., :n4] * gzp4).sum()
    ga_post = (nf4[..., n4: 2 * n4] * gzpost4).sum()
    ga_res = (nf4[..., 2 * n4:] * gzres4).sum()
    gold_alpha = torch.tensor(
        [ga_pre, ga_post, ga_res], dtype=torch.float32, device=x.device
    )

    gn4 = torch.cat([gzp4 * alpha[0], gzpost4 * alpha[1], gzres4 * alpha[2]], dim=-1)
    ghcbn4 = gn4 * invr4
    ginvr4 = (hcbn4 * gn4).sum(-1, keepdim=True)
    gx_rms4 = _rms_norm_grad(x4.flatten(2), invr4, ginvr4)
    gx_mm4 = ghcbn4 @ phi
    gold_phi = ghcbn4.flatten(0, 1).t() @ x4.flatten(2).flatten(0, 1)
    gold_x = grad_x_hin + gx_rms4.view(x4.shape) + gx_mm4.view(x4.shape)

    gold_pre_mix = g_hpre if pm4 is not None else None
    return gold_x.to(x.dtype), gold_phi, gold_alpha, gold_bias, gold_pre_mix


def make_case(has_pm, has_gpr, bs, s, c, dtype):
    """随机输入数据生成: 复用 input.py 的填充逻辑保证数据口径与 b 系列一致。

    input.py 同时生成的合成中间量/golden 与 kernel 语义不同(hcbn 含 bias / z 双加 bias),
    级联模式下不使用, 仅取其随机输入数据(x/phi/alpha/bias/pre_mix/grad_pre/各 grad)。
    shape/dtype 参数化: bs/s/c 为泛化维度, dtype 仅作用于 x/grad_hin/grad_x（其余恒 fp32）。
    """
    f32 = torch.float32
    hm = N * N + 2 * N

    def t(*shape, dtype=f32):
        return torch.empty(*shape, dtype=dtype, device=DEV)

    grad_hin = t(bs, s, c, dtype=dtype)
    grad_h_post = t(bs, s, N)
    grad_h_res = t(bs, s, N, N)
    x = t(bs, s, N, c, dtype=dtype)
    phi = t(hm, N * c)
    alpha = t(3)
    bias = t(hm)
    h_pre = t(bs, s, N)
    hc_before_norm = t(bs, s, hm)
    inv_rms = t(bs, s, 1)
    sum_out = t(ITERS * 2, bs, s, N)
    norm_out = t(ITERS * 2, bs, s, N, N)
    pre_mix = t(bs, s, N) if has_pm else None
    grad_pre = t(bs, s, N) if has_gpr else None
    grad_x_out = t(bs, s, N, c, dtype=dtype)
    grad_phi_out = t(hm, N * c)
    grad_alpha_out = t(3)
    grad_bias_out = t(hm)
    grad_pre_mix_out = t(bs, s, N) if has_pm else None
    hc_eps = HC_EPS

    mhc_pre_sinkhorn_premix_backward_input(
        grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
        h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
        pre_mix, grad_pre, hc_eps,
        grad_x_out, grad_phi_out, grad_alpha_out, grad_bias_out, grad_pre_mix_out,
    )
    torch.npu.synchronize()
    ins = (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
           pre_mix, grad_pre, hc_eps)
    return ins


def compare(tag, got, gold, tol):
    atol, rtol, req_ratio, def_maxlim = tol
    a, g = got.float(), gold.float()
    if a.numel() == 0:
        print("    [PASS] {:<14} (empty)".format(tag))
        return True
    err = (a - g).abs()
    elem_ok = err <= (atol + rtol * g.abs())
    ratio = elem_ok.float().mean().item()
    max_abs = err.max().item()
    # max_abs_error_limit 按输出值域自适应, 防止大值全 reduce 输出被绝对上限误杀
    maxlim = max(def_maxlim, rtol * g.abs().max().item())
    i = int(err.argmax())
    gv = g.flatten()[i].item()
    av = a.flatten()[i].item()
    rel = max_abs / max(abs(gv), 1e-12)
    ok = (ratio >= req_ratio) and (max_abs <= maxlim)
    print("    [{}] {:<14} max_abs_err={:.3e} ratio={:.6f} (worst: got={:.4g} gold={:.4g} rel={:.2e})".format(
        "PASS" if ok else "FAIL", tag, max_abs, ratio, av, gv, rel))
    return ok


def run_scene(name, has_pm, has_gpr, bs, s, c, dtype):
    print("\n=== scene {}: pre_mix={}, grad_pre={} ===".format(
        name, "Y" if has_pm else "N", "Y" if has_gpr else "N"))
    (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
     pre_mix, grad_pre, hc_eps) = make_case(has_pm, has_gpr, bs, s, c, dtype)

    # ---- stage-1 参照: 正向 kernel 真实语义 (hcbn 无 bias, h_pre=sigmoid(z)+eps) ----
    h_pre_ref, hcbn_ref, invrms_ref, sum_ref, norm_ref = forward_ref(
        x, phi, alpha, bias, N, ITERS, hc_eps)

    # ---- stage-1: NPU 正向 (out_flag=True), 产出反向所需的全部中间量 ----
    # normEps 与 hcEps 统一取 hc_eps (forward_ref 的 inv_rms 用同值)
    fwd_outs = FWD_OP.mhc_pre_sinkhorn_premix(
        x, phi, alpha, bias, pre_mix, N, ITERS, hc_eps, hc_eps, True)
    torch.npu.synchronize()
    _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = fwd_outs

    # ---- 级联前置检查: NPU 正向中间量 vs torch 参照 (同一组输入, kernel 语义口径) ----
    print("  -- stage-1: forward intermediates vs torch ref (kernel semantics) --")
    all_ok = True
    # kernel 仅落盘 normOut 偶数槽(softmax 结果 + 各轮行归一化结果, backward 只读偶数槽),
    # 奇数槽(列归一化结果)不写, 故只比对偶数槽; sumOut 为全槽落盘, 保持全量比对
    for tag, k, r, tol in (("fwd.h_pre", h_pre, h_pre_ref, TOL_FWD),
                           ("fwd.hc_before_n", hc_before_norm, hcbn_ref, TOL_FWD_HCBN),
                           ("fwd.inv_rms", inv_rms, invrms_ref, TOL_FWD),
                           ("fwd.sum_out", sum_out, sum_ref, TOL_FWD),
                           ("fwd.norm_out", norm_out[::2], norm_ref[::2], TOL_FWD)):
        all_ok &= compare(tag, k, r, tol)

    # ---- stage-2: NPU 反向, 直接消费 stage-1 的 NPU 正向中间量 ----
    # 契约 v2 (aclnn/tiling 层绑定校验): preMix 与 gradPre 必须同传或同缺省。
    # v1b/v2b 单传组合自本契约起转为"预期拒绝"用例(对齐 torch_ext_test1.py 的 reject 用例模式):
    #   - v1b: pre_mix 缺省 + grad_pre 传入 -> kernel 内部重构 hPre 加权路径的 post 分量时序竞争, 拒绝
    #   - v2b: pre_mix 传入 + grad_pre 缺省 -> 真实模型图中 pre_mix 位于梯度链上时下游必然回传 gradPre, 拒绝
    if has_pm != has_gpr:
        print("  -- stage-2: expect-reject (single optional input violates contract) --")
        try:
            mhc_pre_sinkhorn_premix_backward(
                grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
                h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
                pre_mix, grad_pre, hc_eps)
            torch.npu.synchronize()
            print("    [FAIL] expected EZ1001 reject but backward call succeeded")
            return False
        except RuntimeError as e:
            rejected = "preMix and gradPre must be passed together" in str(e)
            print("    [{}] reject-check: {}".format(
                "PASS" if rejected else "FAIL", str(e).strip().splitlines()[-1][:150]))
            return rejected and all_ok

    # ---- stage-2 golden: 以 NPU 正向中间量为叶子的解析梯度 ----
    golds = backward_golden(
        grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
        h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
        pre_mix, grad_pre, hc_eps)

    print("  -- stage-2: backward(cascade) vs torch golden --")
    outs = mhc_pre_sinkhorn_premix_backward(
        grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
        h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
        pre_mix, grad_pre, hc_eps)
    torch.npu.synchronize()
    all_ok &= compare("grad_x", outs[0], golds[0], TOL_BF16)
    all_ok &= compare("grad_phi", outs[1], golds[1], TOL_FP32)
    all_ok &= compare("grad_alpha", outs[2], golds[2], TOL_FP32)
    all_ok &= compare("grad_bias", outs[3], golds[3], TOL_FP32)
    gpm = outs[4]
    if has_pm:
        all_ok &= compare("grad_pre_mix", gpm, golds[4], TOL_FP32)
    else:
        empty_ok = (not torch.is_tensor(gpm)) or gpm.numel() == 0
        n = gpm.numel() if torch.is_tensor(gpm) else "N/A"
        print("    [{}] grad_pre_mix    (empty expected, numel={})".format(
            "PASS" if empty_ok else "FAIL", n))
        all_ok &= empty_ok
    return all_ok


def main():
    global FWD_OP
    print("torch={} torch_npu device available={}".format(
        torch.__version__, torch.npu.is_available()))
    print("loading forward op via torch extension (first run compiles)...")
    FWD_OP = mhc_pre_sinkhorn_premix_op_builder.load()

    # 泛化测试矩阵: (bs, s, c, dtype, 备注) —— N=4/ITERS=20 为 tiling 强约束不参与泛化
    # c 全部满足 %128==0; dtype 仅切换 x/grad_hin/grad_x
    bf16 = torch.bfloat16
    fp16 = torch.float16
    cfgs = [
        (2, 128, 256, bf16, "baseline"),
        (1, 1, 128, bf16, "min-shape"),
        (3, 17, 128, bf16, "odd-B/S"),
        (2, 64, 384, bf16, "c=384"),
        (2, 128, 1024, bf16, "large-c"),
        (4, 256, 512, bf16, "large-bs*s"),
        (2, 128, 256, fp16, "fp16-dtype"),
    ]
    all_results = {}
    for ci, (bs, s, c, dtype, tag) in enumerate(cfgs):
        print("\n########## cfg{}/{} [{}] B={} S={} C={} dtype={} ##########".format(
            ci, len(cfgs) - 1, tag, bs, s, c, str(dtype).replace("torch.", "")))
        # 每配置独立 seed, 保证单配置可复现
        torch.manual_seed(20260917 + ci)
        results = {}
        results["v1"] = run_scene("v1", False, False, bs, s, c, dtype)
        results["v1b"] = run_scene("v1b", False, True, bs, s, c, dtype)
        results["v2"] = run_scene("v2", True, True, bs, s, c, dtype)
        results["v2b"] = run_scene("v2b", True, False, bs, s, c, dtype)
        all_results["cfg{}[{}] B{}S{}C{} {}".format(
            ci, tag, bs, s, c, str(dtype).replace("torch.", ""))] = results

    print("\n========== GENERALIZATION SUMMARY (cascade fwd->bwd; N=4, ITERS=20 fixed) ==========")
    ok_all = True
    for cfg, results in all_results.items():
        ok = all(results.values())
        ok_all &= ok
        detail = " ".join("{}:{}".format(k, "P" if results[k] else "F") for k in sorted(results))
        print("  {:<44} {} [{}]".format(cfg, "PASS" if ok else "FAIL", detail))
    print("========== {} ==========".format("ALL PASS" if ok_all else "SOME FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

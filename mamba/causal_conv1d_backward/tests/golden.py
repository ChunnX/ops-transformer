#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# ----------------------------------------------------------------------------
"""CausalConv1dBackward golden 参考实现与验证。

主体 = 标准 06 §七 的 qsl 版参考实现（CausalConv1dBackward 的正确性基准）；
验证 = 05/06 §7.1-A 方法论：被测参考实现 ↔ 框架语义参考前向（positions 版，
kda.py:414-450 同款语义）的 autograd，尺度相对误差 max|Δ|/max|golden|，
阈值 fp32 1e-5 / fp16 5e-3 / bf16 2e-2。

覆盖：W∈{2,3,4} × packed documents（等长/不规则/单 token/短于 W）× L==W × L<W ×
L=1 × B>1 × 通道非对齐 × qsl=None × activation none × fp32/fp16/bf16。

CPU 运行（无需 NPU）：
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 tests/golden.py
"""

import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 标准 06 §七 参考实现（qsl 版）—— CausalConv1dBackward 的 golden
# 口径：x/grad_y [T, C]、weight [W, C]（与 CausalConv1d 同布局）、全程 FP32、
# 最后 cast 一次、grad_x 同 x 的 dtype。
# ---------------------------------------------------------------------------


def _seg_bounds(query_start_loc, tokens):
    """query_start_loc [P+1] → [(lo, hi), …]；None 表示整轴一段。"""
    if query_start_loc is None:
        return [(0, tokens)]
    q = query_start_loc.tolist()                 # 参考实现，允许一次 D2H 同步
    return list(zip(q[:-1], q[1:]))


def _silu_grad(pre):
    """silu'(z) = σ(z)·(1 + z·(1-σ(z)))。"""
    sig = torch.sigmoid(pre)
    return sig * (1.0 + pre * (1.0 - sig))


def _seg_forward(x, weight, query_start_loc=None, *, activation="silu"):
    """段内前向：左补 W-1 个零，段边界即天然掩码。x [T,C]，weight [W,C]。"""
    tokens, channels = x.shape
    width = weight.shape[0]
    out = torch.zeros_like(x)
    for lo, hi in _seg_bounds(query_start_loc, tokens):
        seg = x[lo:hi]                               # 段内；空段 len=0 也无害
        padded = F.pad(seg, (0, 0, width - 1, 0))    # 左补零 = 文档左边界
        pre = torch.zeros_like(seg)
        for k in range(width):
            pre = pre + padded[k:k + seg.shape[0]] * weight[k]
        out[lo:hi] = pre if activation == "none" else F.silu(pre)
    return out


def causal_conv1d_backward(grad_y, x, weight, query_start_loc=None, *, activation="silu"):
    """CausalConv1dBackward 的参考实现。x/grad_y [T,C]，weight [W,C]，全程 FP32。"""
    tokens, channels = x.shape
    width = weight.shape[0]
    out_dtype = x.dtype                               # 先存下来：下面 x 会被重绑定
    compute = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    x, grad_y, weight = (t.to(compute) for t in (x, grad_y, weight))

    grad_x = torch.zeros(tokens, channels, dtype=compute, device=x.device)
    grad_weight = torch.zeros(width, channels, dtype=compute, device=x.device)
    for lo, hi in _seg_bounds(query_start_loc, tokens):
        seg, gout = x[lo:hi], grad_y[lo:hi]
        count = seg.shape[0]
        padded = F.pad(seg, (0, 0, width - 1, 0))            # [n+W-1, C]

        pre = torch.zeros_like(seg)
        for k in range(width):
            pre = pre + padded[k:k + count] * weight[k]
        dout = gout if activation == "none" else gout * _silu_grad(pre)

        # grad_x[s] = Σ_k dout[s + W-1-k] · weight[k]（右侧补零 = 段右边界）
        padded_dout = F.pad(dout, (0, 0, 0, width - 1))      # [n+W-1, C]
        for k in range(width):
            grad_x[lo:hi] += padded_dout[width - 1 - k:width - 1 - k + count] * weight[k]

        # grad_weight[k] = Σ_t dout[t] · x[t-(W-1-k)]
        for k in range(width):
            grad_weight[k] += (dout * padded[k:k + count]).sum(0)

    return grad_x.to(out_dtype), grad_weight.to(out_dtype)


# ---------------------------------------------------------------------------
# 框架语义参考前向（positions 版）—— autograd oracle
# 语义与 torchtitan-npu kda.py:414-450 causal_depthwise_conv1d 一致：
#   out[b,t] = Act( Σ_{o=0..W-1} x[b,t-o] · weight[c,0,W-1-o] · mask_o[b,t] )
#   mask_o[b,t] = (t ≥ o) ∧ (positions[b,t] == positions[b,t-o] + o)，o ≥ 1
#   fp32 计算，Act 后一次性 cast 回输入 dtype
# ---------------------------------------------------------------------------


def causal_depthwise_conv1d(x_blc, weight_c1w, positions_bl=None, *, activation="silu"):
    b, l, _ = x_blc.shape
    width = weight_c1w.shape[-1]
    x32 = x_blc.to(torch.float32)
    w_vec = weight_c1w.to(torch.float32)[:, 0, :]              # [C, W]
    output = torch.zeros(b, l, x_blc.shape[-1], dtype=torch.float32)
    for offset in range(min(width, l)):
        if offset == 0:
            shifted = x32
            valid = None
        else:
            shifted = F.pad(x32[:, :-offset], (0, 0, offset, 0))   # shifted[t] = x[t-offset]
            valid = torch.arange(l) >= offset                       # 左边界 (t>=o)
            if positions_bl is not None:
                same_document = positions_bl[:, offset:].eq(positions_bl[:, :-offset] + offset)
                valid = valid.unsqueeze(0).expand(b, l).clone()
                valid[:, offset:] &= same_document
        contribution = shifted * w_vec[:, width - 1 - offset]
        if valid is not None:
            contribution = contribution * valid.unsqueeze(-1).to(contribution.dtype)
        output = output + contribution
    out = output if activation == "none" else F.silu(output)
    return out.to(x_blc.dtype)


def _query_start_loc(positions_bl):
    """positions [B,L] → qsl [P+1] int32（框架 _query_start_loc 同款约定：
    每个文档从 0 起、逐 token +1、每行行首必为文档起点）。"""
    b, l = positions_bl.shape
    flat = positions_bl.reshape(-1).tolist()
    starts = [0]
    for t in range(1, b * l):
        if flat[t] == 0:                       # 文档起点（含每行行首）
            starts.append(t)
    starts.append(b * l)
    return torch.tensor(starts, dtype=torch.int32)


# ---------------------------------------------------------------------------
# 验证：被测 qsl 参考实现 ↔ positions 版参考前向的 autograd
# ---------------------------------------------------------------------------

THRESH = {"float32": 1e-5, "float16": 5e-3, "bfloat16": 2e-2}


def _scale_rel_err(actual, golden):
    denom = golden.abs().max().clamp_min(1e-30)
    return ((actual - golden).abs().max() / denom).item()


def run_case(name, b, l, c, width, dtype, num_docs, activation, positions_fn):
    torch.manual_seed(hash(name) & 0x7FFFFFFF)

    tokens = b * l
    positions = positions_fn(b, l, num_docs) if positions_fn is not None else None

    x32 = torch.randn(b, l, c) * 0.5
    w32 = torch.randn(c, 1, width) * 0.1
    x = x32.to(dtype)
    w = w32.to(dtype)
    go = (torch.randn(b, l, c) * 0.1).to(dtype)

    # golden：autograd 穿过参考前向（fp32 内部计算）
    x_leaf = x.detach().clone().requires_grad_(True)
    w_leaf = w.detach().clone().requires_grad_(True)
    y = causal_depthwise_conv1d(x_leaf, w_leaf, positions, activation=activation)
    loss = (y * go).sum()
    gx_auto, gw_auto = torch.autograd.grad(loss, [x_leaf, w_leaf])
    gx_auto = gx_auto.to(dtype)          # cast 后比较（半精度下与被测逐位可比）
    gw_auto = gw_auto.to(dtype)

    # 被测：qsl 版参考实现（3D 展平为 2D，weight [C,1,W] → [W,C]）
    qsl = _query_start_loc(positions) if positions is not None else None
    gx_ref, gw_ref = causal_conv1d_backward(
        go.reshape(tokens, c),
        x.reshape(tokens, c),
        w[:, 0, :].transpose(0, 1),      # [C,1,W] → [W,C]（非连续视图语义）
        qsl,
        activation=activation,
    )
    gx_ref = gx_ref.reshape(b, l, c)
    gw_ref = gw_ref.transpose(0, 1).unsqueeze(1)   # [W,C] → [C,1,W]

    err_gx = _scale_rel_err(gx_ref.float(), gx_auto.float())
    err_gw = _scale_rel_err(gw_ref.float(), gw_auto.float())
    thr = THRESH[str(dtype).split(".")[-1]]
    ok = err_gx <= thr and err_gw <= thr
    return ok, err_gx, err_gw, thr


def _positions_per_row_docs(b, l, num_docs):
    """每行切 num_docs 个文档：文档长度按不规则权重分配（含单 token、短于 W 的段）。"""
    rows = []
    for bi in range(b):
        weights = [1.0 + ((bi * 3 + d * 7) % 11) for d in range(num_docs)]
        total_w = sum(weights)
        lens = [max(1, int(round(l * w / total_w))) for w in weights]
        diff = l - sum(lens)
        while diff != 0:
            idx = max(range(num_docs), key=lambda i: lens[i]) if diff > 0 else \
                max(range(num_docs), key=lambda i: lens[i])
            step = 1 if diff > 0 else -1
            if lens[idx] + step < 1:
                break
            lens[idx] += step
            diff -= step
        pos = []
        for seg in lens:
            pos.extend(range(seg))
        rows.append(torch.tensor(pos[:l]))
    return torch.stack(rows)


def main():
    cases = []

    # ---- A 组主矩阵：W ∈ {2,3,4} × 4 组 shape × 每行文档数 {1,2,3}（06 §7.1-A） ----
    # docs=1 时也传 positions（每行一个文档）——B>1 时 qsl=None 的语义是"整轴一段"
    # （跨 batch 行），与 oracle 的逐行卷积不同；qsl=None 路径由 edge_no_qsl（B=1）覆盖。
    shapes = [(1, 8, 8), (1, 17, 16), (2, 9, 12), (2, 33, 8)]
    for width in (2, 3, 4):
        for (b, l, c) in shapes:
            for docs in (1, 2, 3):
                cases.append((f"W{width}_B{b}_L{l}_C{c}_docs{docs}_fp32",
                              b, l, c, width, torch.float32, docs, "silu",
                              _positions_per_row_docs))

    # ---- 边界与特殊场景 ----
    edge = [
        ("edge_L_eq_W", 1, 4, 16, 4, torch.float32, 1, "silu", None),            # L == W
        ("edge_L_lt_W", 1, 2, 16, 4, torch.float32, 1, "silu", None),            # L < W
        ("edge_L1", 1, 1, 16, 4, torch.float32, 1, "silu", None),                # 退化 L=1
        ("edge_single_token_docs", 1, 12, 8, 4, torch.float32, 12, "silu",
         _positions_per_row_docs),                                                # 单 token 文档
        ("edge_doc_shorter_than_W", 1, 9, 8, 4, torch.float32, 3, "silu",
         _positions_per_row_docs),                                              # 文档短于 W
        ("edge_C_unaligned", 2, 9, 7, 3, torch.float32, 2, "silu",
         _positions_per_row_docs),                                              # 通道非对齐
        ("edge_no_qsl", 1, 16, 8, 4, torch.float32, 1, "silu", None),            # qsl=None 整轴一段
        ("edge_activation_none", 1, 16, 8, 4, torch.float32, 2, "none",
         _positions_per_row_docs),                                              # activation none
        ("edge_3d_qsl_multirow", 2, 8, 8, 4, torch.float32, 2, "silu",
         _positions_per_row_docs),                                              # 3D + qsl（跨行分段）
    ]
    cases.extend(edge)

    # ---- 半精度 ----
    for width in (2, 3, 4):
        cases.append((f"bf16_W{width}_packed", 1, 33, 8, width, torch.bfloat16, 3,
                      "silu", _positions_per_row_docs))
        cases.append((f"fp16_W{width}_packed", 2, 17, 16, width, torch.float16, 2,
                      "silu", _positions_per_row_docs))
    cases.append(("bf16_anchor_activation_none", 1, 16, 8, 4, torch.bfloat16, 1,
                  "none", None))

    n_pass = 0
    rows = []
    for name, b, l, c, width, dtype, docs, act, pfn in cases:
        ok, err_gx, err_gw, thr = run_case(name, b, l, c, width, dtype, docs, act, pfn)
        rows.append((name, width, str(dtype).split(".")[-1], err_gx, err_gw, thr, ok))
        n_pass += int(ok)

    print(f"| {'用例':<34} | W | dtype | grad_x 误差 | grad_weight 误差 | 阈值 | 结果 |")
    print("|" + "-" * 36 + "|" + "-" * 4 + "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 17 + "|" + "-" * 7 + "|" + "-" * 6 + "|")
    for name, width, dts, err_gx, err_gw, thr, ok in rows:
        print(f"| {name:<34} | {width} | {dts:<8} | {err_gx:.3e} | {err_gw:.3e} | {thr:.0e} |"
              f" {'PASS' if ok else '**FAIL**'} |")

    total = len(rows)
    print(f"\n合计 {total} 例，PASS {n_pass}，FAIL {total - n_pass}（阈值按 dtype：fp32 1e-5 / fp16 5e-3 / bf16 2e-2）")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())

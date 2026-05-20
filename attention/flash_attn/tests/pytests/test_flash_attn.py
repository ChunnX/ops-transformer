#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import argparse
import importlib
import itertools

import gc
import torch
import math
import numpy as np
from cpu_impl import tforward
from test_utils import (generate_qkv, generate_pse, generate_npu_mask, trans_bnsd_to_layout,
                         data_compare_benchmark_new, Result, gen_block_table)

_DEFAULT_CASE_FILES = ["test_case", "test_case_fia_STC", "test_case_functional_single"]


def load_case_modules(module_names):
    all_cases = {}
    for mod_name in module_names:
        mod = importlib.import_module(mod_name)
        for case_name, case_dict in mod.TestCases.items():
            keys = list(case_dict.keys())
            values = [case_dict[k] for k in keys]
            combos = list(itertools.product(*values))
            if len(combos) == 1:
                all_cases[f"{mod_name}/{case_name}"] = dict(zip(keys, combos[0]))
            else:
                for i, combo in enumerate(combos):
                    all_cases[f"{mod_name}/{case_name}_{i}"] = dict(zip(keys, combo))
    return all_cases


# ------------------ 动态导入 GPU / NPU 实现 ------------------
try:
    from gpu_impl import flash_attn_gpu
    GPU_AVAILABLE = True
except ImportError:
    flash_attn_gpu = None
    GPU_AVAILABLE = False

try:
    from npu_impl import flash_attn_npu, flash_attn_metadata_only, flash_attn_npu_graph
    NPU_AVAILABLE = True
except ImportError:
    flash_attn_npu = None
    flash_attn_metadata_only = None
    flash_attn_npu_graph = None
    NPU_AVAILABLE = False

# ------------------ 辅助函数 ------------------
def save_tensor_to_txt(tensor, filepath):
    """将 tensor 展平后逐行保存为 txt，首行写入 shape 注释。"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    arr = tensor.detach().cpu().float().numpy().flatten()
    shape_str = "x".join(str(s) for s in tensor.shape)
    with open(filepath, "w") as fh:
        fh.write(f"# shape: {shape_str}  total: {arr.size}\n")
        for v in arr:
            fh.write(f"{v:.8f}\n")

def load_tensor_from_txt(filepath, target_dtype=torch.float32, target_device='cpu'):
    """从 save_tensor_to_txt 保存的文件恢复为 Tensor。"""
    with open(filepath, "r") as fh:
        header = fh.readline().strip()
        # 解析 shape 注释: # shape: 16x104x224x128  total: xxx
        if not header.startswith("# shape:"):
            raise ValueError(f"文件 {filepath} 格式不正确，缺少 shape 注释。")
        shape_str = header.split()[2]
        shape = tuple(int(s) for s in shape_str.split('x'))
        arr = np.loadtxt(fh, dtype=np.float32)
    expected = 1
    for s in shape:
        expected *= s
    if arr.size != expected:
        raise ValueError(f"形状 {shape} 与数值数 {arr.size} 不匹配。")
    tensor = torch.from_numpy(arr).reshape(shape).to(dtype=target_dtype, device=target_device)
    return tensor

# ------------------ 精度报告阈值 ------------------
RATIO_THRESHOLD = 0.005   # 单元素相对误差阈值
FAIL_RATIO_LIMIT = 0.005  # 超阈值元素占比上限


def normalize_case(raw):
    """将 test_case.py 字段转换为 call_flash_attn 所需 kwargs 格式。"""
    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    layout_kv = c.get("layout_kv", layout_q)
    c["input_layout"] = layout_q
    c.setdefault("layout_kv",   c.get("layout_kv",  layout_q))
    c.setdefault("layout_out",  c.get("layout_out", layout_q))
    if "mask_mode" in c and "sparse_mode" not in c:
        c["sparse_mode"] = c.pop("mask_mode")
    c.setdefault("N2",          c.get("N1"))
    c.setdefault("S2",          c.get("S1"))
    c.setdefault("DV",          c.get("D"))
    c.setdefault("DRope",       0)
    c.setdefault("pse_type",    0)
    c.setdefault("pse_layout",  "none")
    c.setdefault("q_start_idx", 0)
    c.setdefault("kv_start_idx",0)
    c.setdefault("keep_prob",   1.0)
    c.setdefault("seed",        0)
    c.setdefault("offset",      0)
    # win_left/win_right → pre_tokens/next_tokens（test_case_functional_single 用 win_left/win_right）
    if "win_left" in c and c["win_left"] != -1:
        c.setdefault("pre_tokens", c["win_left"])
    if "win_right" in c and c["win_right"] != -1:
        c.setdefault("next_tokens", c["win_right"])
    c.setdefault("pre_tokens",  2147483647)
    c.setdefault("next_tokens", 2147483647)
    c.setdefault("prefix",      [])
    c.setdefault("q_range",     None)
    c.setdefault("k_range",     None)
    c.setdefault("v_range",     None)
    # 清理 [None] 哨兵值 → None（test_case_functional_single.py 用 [[None]] 表示"不传"）
    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)
    if layout_q == "TND":
        c.setdefault("seqused_q", [c["cu_seqlens_q"][i+1] - c["cu_seqlens_q"][i]
                                    for i in range(len(c["cu_seqlens_q"]) - 1)])
        if layout_kv not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            # cu_seqlens_kv 未提供时默认与 cu_seqlens_q 相同（self-attention 场景，非 PA）
            c.setdefault("cu_seqlens_kv", list(c["cu_seqlens_q"]))
            c.setdefault("seqused_kv", [c["cu_seqlens_kv"][i+1] - c["cu_seqlens_kv"][i]
                                         for i in range(len(c["cu_seqlens_kv"]) - 1)])
        c["B"] = 1
    if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("cu_seqlens_kv", None)
        c.setdefault("block_size", c.get("block_size"))
        c.setdefault("block_table_shape", c.get("block_table_shape"))
        block_size = c.get("block_size", 1)
        block_table_shape = c.get("block_table_shape", [])
        # PA 需要 seqused_kv；如果未提供（被 [None] 清理掉了），默认为 [S2]*B
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)
        seqused_kv = c.get("seqused_kv")
        if ("block_table" in c.keys()) and (c.get("block_table") is not None):
            block_table_list = c.get("block_table")
            block_table = torch.tensor(block_table_list, dtype=torch.int32)
            c["block_table"] = block_table
        else:
            block_table = gen_block_table(c.get("B"), seqused_kv, block_size, block_table_shape)
            c["block_table"]        = block_table
        c["actual_b"] = block_table.shape[0]
    return c

def check_result(test_name, expect, result, except_label="CPU", comp_label="NPU", verbose_diff=False):
    """比较两个 tensor，输出精度报告。"""
    SEP = "─" * 64
    print(f"\n┌{SEP}┐")
    print(f"│  精度报告: {test_name}  ({except_label} vs {comp_label})")
    print(f"├{SEP}┤")
    if expect.shape != result.shape:
        print(f"│  [ERROR] shape不匹配: expect={tuple(expect.shape)}  {comp_label}={tuple(result.shape)}")
        print(f"└{SEP}┘")
        return False
    ef   = expect.float()
    rf   = result.float()
    diff = torch.abs(ef - rf)
    ref_abs = torch.abs(ef)
    rel_err = diff / (ref_abs + 1e-6)
    max_abs   = diff.max().item()
    mean_abs  = diff.mean().item()
    max_rel   = rel_err.max().item()
    mean_rel  = rel_err.mean().item()
    threshold = torch.max(ref_abs.mul(RATIO_THRESHOLD), torch.full_like(diff, 0.000025))
    fail_mask = diff > threshold
    fail_cnt  = int(fail_mask.sum().item())
    total     = expect.numel()
    fail_ratio = fail_cnt / total
    passed    = fail_ratio <= FAIL_RATIO_LIMIT
    print(f"│  Shape       : {tuple(expect.shape)}")
    print(f"│  MaxAbsErr   : {max_abs:.8f}")
    print(f"│  MeanAbsErr  : {mean_abs:.8f}")
    print(f"│  MaxRelErr   : {max_rel:.8f}")
    print(f"│  MeanRelErr  : {mean_rel:.8f}")
    print(f"│  FailElems   : {fail_cnt} / {total}  ({fail_ratio*100:.4f}%)")
    print(f"│  Threshold   : elemRelErr≤{RATIO_THRESHOLD*100:.2f}%  failRatio≤{FAIL_RATIO_LIMIT*100:.2f}%")
    print(f"│  结论        : {'✓ PASS' if passed else '✗ FAIL'}")
    if fail_cnt > 0:
        print(f"├{SEP}┤")
        if verbose_diff:
            all_idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).tolist()
            print(f"│  全部 {len(all_idxs)} 个超阈値元素 (relErr > {RATIO_THRESHOLD * 100:.2f}%):")
            print(f"│  {'idx':>10}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>12}")
            for i in all_idxs:
                print(f"│  {i:>10}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>12.6f}")
        else:
            idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)[:10].tolist()
            print(f"│  前{len(idxs)}个不通过元素:")
            print(f"│  {'idx':>8}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>10}")
            for i in idxs:
                print(f"│  {i:>8}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>10.6f}")
    print(f"└{SEP}┘")
    stats = {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "fail_cnt": fail_cnt,
        "total": total,
        "fail_ratio": fail_ratio,
    }
    return stats


def analyze_fail_distribution(test_name, expect, result, dump_dir="./dump_output", **kwargs):
    """分析失败元素在各 batch/head/seq 维度的分布情况，输出统计报告并保存到文件。"""
    ef = expect.float()
    rf = result.float()
    diff = torch.abs(ef - rf)
    ref_abs = torch.abs(ef)
    threshold = torch.max(ref_abs.mul(RATIO_THRESHOLD), torch.full_like(diff, 0.000025))
    fail_mask = diff > threshold

    shape = expect.shape
    ndim = len(shape)
    total_fail = int(fail_mask.sum().item())
    total_elem = expect.numel()

    SEP = "━" * 80
    lines = []
    lines.append(f"\n{SEP}")
    lines.append(f"  失败元素分布分析: {test_name}")
    lines.append(f"  Shape: {tuple(shape)}  FailElems: {total_fail}/{total_elem} ({total_fail/total_elem*100:.4f}%)")
    lines.append(SEP)

    if ndim == 4:
        B, N, S, D = shape
        # ---- 按 batch 统计 ----
        lines.append(f"\n  [按 Batch 维度统计] (共 {B} 个 batch)")
        lines.append(f"  {'Batch':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}  {'MaxAbsErr':>12}  {'MeanAbsErr':>12}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")
        batch_stats = []
        for bi in range(B):
            b_fail = fail_mask[bi]
            b_diff = diff[bi]
            cnt = int(b_fail.sum().item())
            tot = N * S * D
            ratio = cnt / tot
            max_err = b_diff.max().item() if cnt > 0 else 0.0
            mean_err = b_diff[b_fail].mean().item() if cnt > 0 else 0.0
            batch_stats.append((bi, cnt, tot, ratio, max_err, mean_err))
            lines.append(f"  {bi:>6}  {cnt:>10}  {tot:>10}  {ratio*100:>9.4f}%  {max_err:>12.8f}  {mean_err:>12.8f}")

        # ---- 按 head 统计 ----
        lines.append(f"\n  [按 Head 维度统计] (共 {N} 个 head，显示 Top-10 失败率最高)")
        lines.append(f"  {'Head':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}  {'MaxAbsErr':>12}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
        head_stats = []
        for ni in range(N):
            h_fail = fail_mask[:, ni]
            h_diff = diff[:, ni]
            cnt = int(h_fail.sum().item())
            tot = B * S * D
            ratio = cnt / tot
            max_err = h_diff.max().item() if cnt > 0 else 0.0
            head_stats.append((ni, cnt, tot, ratio, max_err))
        head_stats.sort(key=lambda x: x[3], reverse=True)
        for ni, cnt, tot, ratio, max_err in head_stats[:10]:
            lines.append(f"  {ni:>6}  {cnt:>10}  {tot:>10}  {ratio*100:>9.4f}%  {max_err:>12.8f}")

        # ---- 按 seq 位置统计（聚合） ----
        lines.append(f"\n  [按 Seq 位置统计] (共 {S} 个位置，显示 Top-10 失败率最高)")
        lines.append(f"  {'SeqPos':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}  {'MaxAbsErr':>12}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
        seq_stats = []
        for si in range(S):
            s_fail = fail_mask[:, :, si]
            s_diff = diff[:, :, si]
            cnt = int(s_fail.sum().item())
            tot = B * N * D
            ratio = cnt / tot
            max_err = s_diff.max().item() if cnt > 0 else 0.0
            seq_stats.append((si, cnt, tot, ratio, max_err))
        seq_stats.sort(key=lambda x: x[3], reverse=True)
        for si, cnt, tot, ratio, max_err in seq_stats[:10]:
            lines.append(f"  {si:>6}  {cnt:>10}  {tot:>10}  {ratio*100:>9.4f}%  {max_err:>12.8f}")

        # ---- 关联 seqused_kv 分析 ----
        seqused_kv = kwargs.get("seqused_kv", None)
        if seqused_kv is not None:
            lines.append(f"\n  [Batch vs seqused_kv 关联分析]")
            lines.append(f"  {'Batch':>6}  {'seqused_kv':>12}  {'FailRatio':>10}  {'说明':>20}")
            lines.append(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*20}")
            for bi, cnt, tot, ratio, max_err, mean_err in batch_stats:
                skv = seqused_kv[bi] if bi < len(seqused_kv) else "N/A"
                note = ""
                if ratio > 0.03:
                    note = "⚠ 超3%阈值"
                elif ratio > 0.01:
                    note = "△ 偏高"
                lines.append(f"  {bi:>6}  {skv:>12}  {ratio*100:>9.4f}%  {note:>20}")

        # ---- 错误位置连续性分析（按 D 维度展开，检测连段模式） ----
        lines.append(f"\n  [错误位置连续性分析] (选取失败率最高的 3 个 batch，分析 D 维度连段)")
        top_batches = sorted(batch_stats, key=lambda x: x[3], reverse=True)[:3]
        for bi, cnt, tot, ratio, max_err, mean_err in top_batches:
            if cnt == 0:
                continue
            lines.append(f"\n  ── Batch {bi} (FailRatio={ratio*100:.4f}%) ──")
            # 选取该 batch 中失败最多的一个 (head, seq) 位置
            b_fail = fail_mask[bi]  # (N, S, D)
            # 按 (head, seq) 聚合失败数
            hs_fail_cnt = b_fail.sum(dim=-1)  # (N, S)
            max_hs_idx = int(hs_fail_cnt.argmax().item())
            max_h = max_hs_idx // S
            max_s = max_hs_idx % S
            d_fail_vec = b_fail[max_h, max_s]  # (D,) bool
            d_fail_cnt = int(d_fail_vec.sum().item())
            lines.append(f"    最密集位置: head={max_h}, seq={max_s}, 失败D元素={d_fail_cnt}/{D}")

            # 检测连续段
            fail_positions = d_fail_vec.nonzero(as_tuple=False).squeeze(1).tolist()
            if len(fail_positions) == 0:
                lines.append(f"    无失败位置")
                continue

            # 计算连续段
            segments = []
            seg_start = fail_positions[0]
            seg_end = fail_positions[0]
            for pos in fail_positions[1:]:
                if pos == seg_end + 1:
                    seg_end = pos
                else:
                    segments.append((seg_start, seg_end))
                    seg_start = pos
                    seg_end = pos
            segments.append((seg_start, seg_end))

            num_seg = len(segments)
            max_seg_len = max(e - s + 1 for s, e in segments)
            avg_seg_len = d_fail_cnt / num_seg

            # 判断模式
            if num_seg == 1 and d_fail_cnt == D:
                pattern = "全D失败（整行错误）"
            elif avg_seg_len >= 8 and num_seg <= d_fail_cnt / 4:
                pattern = "连段集中（可能与tiling/block边界相关）"
            elif num_seg >= d_fail_cnt * 0.8:
                pattern = "随机分散（典型精度累积误差）"
            else:
                pattern = "混合模式"

            lines.append(f"    连续段数: {num_seg}  最长段: {max_seg_len}  平均段长: {avg_seg_len:.1f}")
            lines.append(f"    模式判定: {pattern}")
            # 显示前 8 个连续段
            show_segs = segments[:8]
            seg_strs = [f"[{s}:{e}]({e-s+1})" for s, e in show_segs]
            lines.append(f"    前{len(show_segs)}段: {', '.join(seg_strs)}")
            if len(segments) > 8:
                lines.append(f"    ... 共 {num_seg} 段")

            # 额外：按整个 seq 维度看该 batch+head 的错误 D 位置热图（简化）
            lines.append(f"    该 head={max_h} 在 batch={bi} 中各 seq 位置失败率:")
            seq_fail_in_bh = b_fail[max_h].sum(dim=-1)  # (S,)
            # 显示每 16 个 seq 位置聚合
            chunk_size = max(1, S // 16)
            chunks = []
            for ci in range(0, S, chunk_size):
                c_end = min(ci + chunk_size, S)
                c_cnt = int(seq_fail_in_bh[ci:c_end].sum().item())
                c_tot = (c_end - ci) * D
                chunks.append(f"{c_cnt/c_tot*100:.1f}%")
            lines.append(f"    seq分段失败率(每{chunk_size}位置): [{', '.join(chunks)}]")

    elif ndim == 3:
        T, N, D = shape
        lines.append(f"\n  [按 Head 维度统计] (TND: T={T}, N={N}, D={D})")
        lines.append(f"  {'Head':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
        for ni in range(N):
            h_fail = fail_mask[:, ni]
            cnt = int(h_fail.sum().item())
            tot = T * D
            lines.append(f"  {ni:>6}  {cnt:>10}  {tot:>10}  {cnt/tot*100:>9.4f}%")
    else:
        lines.append(f"  [维度={ndim}，跳过分维统计]")

    lines.append(f"\n{SEP}")

    report = "\n".join(lines)
    print(report)

    # 保存分析报告到文件
    dump_path = os.path.join(dump_dir, test_name)
    os.makedirs(dump_path, exist_ok=True)
    report_file = os.path.join(dump_path, "fail_analysis.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [分析报告已保存] {report_file}")

    return batch_stats if ndim == 4 else None


# ------------------ 三方精度对比 ------------------
def call_flash_attn(test_name, dump_tensors=False, dump_dir="./dump_output",
                    verbose_diff=False, visualize=False, viz_dir="./viz_output",
                    meta_only=False, compare_mode=False, graph_mode=False,
                    load_gpu_dump=None, load_npu_dump=None, fail_analysis=False,
                    **kwargs):
    b          = kwargs.get("B", 1)
    n1         = kwargs.get("N1")
    n2         = kwargs.get("N2", n1)
    sq         = kwargs.get("S1", -1)
    skv        = kwargs.get("S2", sq)
    d          = kwargs.get("D")
    d_v        = kwargs.get("DV", d)
    d_rope     = kwargs.get("DRope", 0)
    input_layout = kwargs.get("input_layout")
    layout_kv    = kwargs.get("layout_kv", input_layout)
    output_layout = kwargs.get('layout_out')
    pse_type   = int(kwargs.get("pse_type") if kwargs.get("pse_type") != '' else 0)
    pse_layout = kwargs.get("pse_layout", "none").lower()
    q_start_idx  = kwargs.get("q_start_idx", 0)
    kv_start_idx = kwargs.get("kv_start_idx", 0)
    dtype = kwargs.get("Dtype", "bf16")
    pttype = torch.float16 if dtype == "fp16" else torch.bfloat16
    input_dtype = pttype
    sparse_mode = kwargs.get("sparse_mode", None)
    pre_tokens  = kwargs.get("pre_tokens",  2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    prefix      = kwargs.get("prefix", [])
    q_range     = kwargs.get("q_range", None)
    k_range     = kwargs.get("k_range", None)
    v_range     = kwargs.get("v_range", None)
    lse_flag = kwargs.get("return_softmax_lse", 0)
    pse_b = b;  pse_s1 = sq;  pse_s2 = skv
    sq_gen = sq;  skv_gen = skv
    if input_layout == "TND":
        cu_q = kwargs["cu_seqlens_q"]
        sq = max(cu_q[i+1] - cu_q[i] for i in range(len(cu_q) - 1))
        sq_gen = cu_q[-1]
        if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            skv = max(kwargs["seqused_kv"])
            skv_gen = sum(kwargs["seqused_kv"])
        else:
            cu_kv = kwargs["cu_seqlens_kv"]
            skv = max(cu_kv[i+1] - cu_kv[i] for i in range(len(cu_kv) - 1))
            skv_gen = cu_kv[-1]
        kwargs["S1"] = sq
        kwargs["S2"] = skv
        pse_b = len(cu_q) - 1;  pse_s1 = sq;  pse_s2 = skv
        if pse_layout in ["bnhs", "1nhs"]: pse_s1 = max(1024, pse_s1)
    _, pse_npu = generate_pse(pse_b, n1, pse_s1, pse_s2, pse_type, pse_layout,
                                    pttype, q_start_idx, kv_start_idx)
    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(b, n1, n2, sq_gen, skv_gen, d, d_v, d_rope,
                                                    input_layout, input_dtype,
                                                    q_range=q_range, k_range=k_range, v_range=v_range)

    # ------------------ 1. CPU Golden 计算 (总是执行) ------------------
    print(f"[{test_name}] CPU 参考计算...")
    out_cpu, x_max, x_sum = tforward(qf, kf, v, **kwargs)
    out_cpu_out_layout = trans_bnsd_to_layout(out_cpu, output_layout, **kwargs)
    lse_out = torch.log(x_sum) + x_max
    if dump_tensors:
        dump_path = os.path.join(dump_dir, test_name)
        os.makedirs(dump_path, exist_ok=True)
        save_tensor_to_txt(q, os.path.join(dump_path, "q.txt"))
        save_tensor_to_txt(k, os.path.join(dump_path, "k.txt"))
        save_tensor_to_txt(v, os.path.join(dump_path, "v.txt"))
        save_tensor_to_txt(out_cpu_out_layout.float(), os.path.join(dump_path, "cpu_out.txt"))

    # ------------------ 2. 准备 GPU / NPU 结果张量 ------------------
    gpu_out = None
    npu_out = None

    # ---- GPU 计算或加载 dump ----
    if kwargs.get("_use_gpu", False):
        if not GPU_AVAILABLE:
            print(f"[{test_name}] 警告: GPU 不可用，但要求 GPU 模式，尝试从 dump 加载。")
        if GPU_AVAILABLE:
            print(f"[{test_name}] GPU 计算...")
            atten_mask = generate_npu_mask(b, sq, skv, sparse_mode, pre_tokens, next_tokens, prefix)
            device = torch.cuda.current_device()
            q_gpu = q.cuda(device)
            k_gpu = k.cuda(device)
            v_gpu = v.cuda(device)
            q_rope_gpu = q_rope.cuda(device) if q_rope is not None else None
            k_rope_gpu = k_rope.cuda(device) if k_rope is not None else None
            mask_gpu = atten_mask.cuda(device) if atten_mask is not None else None
            pse_gpu = pse_npu.cuda(device) if pse_npu is not None else None
            gpu_out_bnsd = flash_attn_gpu(q_gpu, k_gpu, v_gpu, q_rope_gpu, k_rope_gpu, mask_gpu, pse_gpu, **kwargs)
            gpu_out = trans_bnsd_to_layout(gpu_out_bnsd, output_layout, **kwargs).cpu()
            if dump_tensors:
                dump_path = os.path.join(dump_dir, test_name)
                os.makedirs(dump_path, exist_ok=True)
                save_tensor_to_txt(gpu_out.float(), os.path.join(dump_path, "gpu_out.txt"))
                print(f"[{test_name}] 已保存 q/k/v/gpu_out → {dump_path}/")
        else:
            print(f"[{test_name}] GPU 不可用，跳过计算。")
    if gpu_out is None or load_gpu_dump:
        gpu_dump_path = load_gpu_dump if load_gpu_dump else os.path.join(dump_dir, test_name, "gpu_out.txt")
        if os.path.exists(gpu_dump_path):
            print(f"[{test_name}] 从 dump 加载 GPU 结果: {gpu_dump_path}")
            gpu_out = load_tensor_from_txt(gpu_dump_path, target_dtype=torch.float32, target_device='cpu')
        else:
            print(f"[{test_name}] 未找到 GPU dump 文件: {gpu_dump_path}")

    # ---- NPU 计算或加载 dump ----
    if not kwargs.get("_use_gpu", False):
        if not NPU_AVAILABLE:
            print(f"[{test_name}] 警告: NPU 不可用，尝试从 dump 加载。")
        if NPU_AVAILABLE and (n1 >= n2 and n1 % n2 == 0):
            atten_mask = generate_npu_mask(b, sq, skv, sparse_mode, pre_tokens, next_tokens, prefix)
            if graph_mode:
                print(f"[{test_name}] NPU 图模式计算...")
                npu_out, lse_npu = flash_attn_npu_graph(q, k, v, q_rope, k_rope, atten_mask, pse_npu, **kwargs)
            else:
                print(f"[{test_name}] NPU 单算子模式计算...")
                npu_out, lse_npu = flash_attn_npu(q, k, v, q_rope, k_rope, atten_mask, pse_npu, **kwargs)
        elif NPU_AVAILABLE:
            print(f"[{test_name}] 跳过 NPU: N1={n1} 不满足 N1>=N2 且 N1%N2==0 (N2={n2})")
        else:
            print(f"[{test_name}] NPU 不可用，跳过计算。")
    if npu_out is None or load_npu_dump:
        npu_dump_path = load_npu_dump if load_npu_dump else os.path.join(dump_dir, test_name, "npu_out.txt")
        if os.path.exists(npu_dump_path):
            print(f"[{test_name}] 从 dump 加载 NPU 结果: {npu_dump_path}")
            npu_out = load_tensor_from_txt(npu_dump_path, target_dtype=torch.float32, target_device='cpu')
        else:
            print(f"[{test_name}] 未找到 NPU dump 文件: {npu_dump_path}")

    # ------------------ 3. 精度对比 ------------------
    passed_cpu_gpu = False
    passed_cpu_npu = False
    passed_gpu_npu = False

    if compare_mode:
        if gpu_out is not None and npu_out is not None:
            print(f"\n{'='*40} 三方对比模式（详细精度统计） {'='*40}")

            dtype_str = kwargs.get("Dtype", "fp16")
            dtype_map = {"fp16": "fp16", "bf16": "bf16", "fp32": "fp32"}
            out_dtype_key = dtype_map.get(dtype_str, dtype_str)

            params = {
                "op_name": "flash_attn",
                "case_name": test_name,
                "dtype_output": [dtype_str],
                "dtype_input": [dtype_str],
                "red_range": {
                    "fp32": "0.000001/0.00001/0.0001/0.0005",
                    "fp16": "0.001/0.002/0.005/0.01",
                    "bf16": "0.001/0.002/0.005/0.01",
                },
                "bm_cmp_std": {
                    "fp32": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 1e-06,
                        "small_value_atol": 0.0
                    },
                    "fp16": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 0.001,
                        "small_value_atol": 0.001
                    },
                    "bf16": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 1e-07,
                        "small_value_atol": 0.004
                    }
                },
            }

            npu_np = npu_out.float().numpy()
            gpu_np = gpu_out.float().numpy()
            cpu_np = out_cpu_out_layout.float().numpy()

            str1, str2, data = data_compare_benchmark_new(params, npu_np, gpu_np, cpu_np, out_dtype_key, i=0)

            print("\n====== 三方对比结果 ======")
            print(f"状态: {str1}, 原因: {str2}")
            passed_cpu_gpu = (str1 == 'Pass')
            passed_cpu_npu = (str1 == 'Pass')
            passed_gpu_npu = (str1 == 'Pass')
        else:
            print(f"[{test_name}] 缺少 GPU 或 NPU 结果，无法执行三方对比，使用标准对比模式。")
            compare_mode = False

    if not compare_mode:
        if gpu_out is not None:
            print(f"\n{'='*40} 对比 CPU vs GPU {'='*40}")
            passed_cpu_gpu = check_result(test_name + "_CPU_vs_GPU", out_cpu_out_layout.float(), gpu_out.float(), except_label="CPU", comp_label="GPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 GPU 结果，跳过 CPU vs GPU 对比。")

        if npu_out is not None:
            print(f"\n{'='*40} 对比 CPU vs NPU {'='*40}")
            passed_cpu_npu = check_result(test_name + "_CPU_vs_NPU", out_cpu_out_layout.float(), npu_out.float(), except_label="CPU", comp_label="NPU", verbose_diff=verbose_diff)
            if fail_analysis and isinstance(passed_cpu_npu, dict) and not passed_cpu_npu["passed"]:
                analyze_fail_distribution(test_name, out_cpu_out_layout, npu_out, dump_dir=dump_dir, **kwargs)
        else:
            print(f"[{test_name}] 缺少 NPU 结果，跳过 CPU vs NPU 对比。")

        if gpu_out is not None and npu_out is not None:
            print(f"\n{'='*40} 对比 GPU vs NPU {'='*40}")
            passed_gpu_npu = check_result(test_name + "_GPU_vs_NPU", gpu_out.float(), npu_out.float(), except_label="GPU", comp_label="NPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 GPU 或 NPU 结果，跳过 GPU vs NPU 对比。")

    if visualize:
        try:
            from precision_visual import visualize_from_tensors
            if npu_out is not None:
                visualize_from_tensors(out_cpu_out_layout.float(), npu_out.float(), case_name=test_name, out_dir=viz_dir)
            if gpu_out is not None:
                visualize_from_tensors(out_cpu_out_layout.float(), gpu_out.float(), case_name=test_name + "_gpu", out_dir=viz_dir)
        except ImportError:
            print("[WARN] precision_visual 导入失败，请确认 matplotlib 已安装")
        except Exception as exc:
            print(f"[WARN] 可视化异常: {exc}")

    passed_lse = 1
    if lse_flag == 1 and npu_out is not None:
        passed_lse = 0
        print(f"\n{'='*40} 对比 lse goldem vs lse npu {'='*40}")
        passed_lse = check_result("LSE CHECK", lse_out.float(), lse_npu.float(), except_label="lse_golden", comp_label="lse_npu", verbose_diff=verbose_diff)
    elif lse_flag == 1:
        print(f"[{test_name}] 缺少 NPU 结果，跳过 LSE 对比。")

    # collect structured results
    def _to_stats(val):
        """Convert check_result return (dict) or bool/int to stats dict."""
        if isinstance(val, dict):
            return val
        if val:
            return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                    "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
        return None

    attn_stats = _to_stats(passed_cpu_npu) or _to_stats(passed_cpu_gpu)
    if attn_stats is None:
        attn_stats = {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                      "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}

    if lse_flag != 1:
        lse_stats = {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                     "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    else:
        lse_stats = _to_stats(passed_lse)
        if lse_stats is None:
            lse_stats = {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                         "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}

    return {"attn": attn_stats, "lse": lse_stats}


def resolve_case_ids(case_id_arg, all_cases):
    if case_id_arg == "all":
        return sorted(all_cases.keys())
    ids = [x.strip() for x in case_id_arg.split(",")]
    result = []
    missing = []
    for cid in ids:
        full_names = [k for k in all_cases if k.endswith(f"/{cid}") or k == cid or k.rsplit("/", 1)[-1].startswith(f"{cid}_")]
        if len(full_names) >= 1:
            result.extend(full_names)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] 以下 case 不存在: {missing}")
    return result


# ------------------ main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttn 精度测试（CPU vs GPU vs NPU）")
    parser.add_argument("--case_files", type=str,
                        default=",".join(_DEFAULT_CASE_FILES),
                        help=f"逗号分隔的 test case 模块名（不含 .py） (default: {','.join(_DEFAULT_CASE_FILES)})")
    parser.add_argument("--case", type=str, default="all",
                        choices=["all", "base", "fia", "TestCases", "TestCasesFIA"],
                        help="选择测试集，覆盖 --case_files (default: all)")
    parser.add_argument("--case_id",      type=str,            default="all",
                        help="case名称，多个用逗号分隔，'all'表示全部 (default: all)")
    parser.add_argument("--device_id",    type=int,            default=0,
                        help="NPU device id (default: 0)")
    parser.add_argument("--use_gpu",      action="store_true", help="使用 GPU 进行计算，不执行 NPU")
    parser.add_argument("--gpu-device",   type=int,            default=0, help="GPU 设备 ID (默认 0)")
    parser.add_argument("--dump_tensors", action="store_true",
                        help="保存 q/k/v 及算子输出为 txt 文件")
    parser.add_argument("--dump_dir",     type=str,            default="./dump_output",
                        help="txt 文件保存根目录 (default: ./dump_output)")
    parser.add_argument("--verbose_diff", action="store_true",
                        help="逐元素输出全部超阈值精度对比表")
    parser.add_argument("--visualize",    action="store_true",
                        help="生成 CPU vs NPU 精度热力图")
    parser.add_argument("--viz_dir",      type=str,            default="./viz_output")
    parser.add_argument("--meta_only",    action="store_true",
                        help="只调用 npu_flash_attn_metadata (NPU环境)")
    parser.add_argument("--load_gpu_dump", type=str, default=None,
                        help="指定 GPU dump 文件路径（用于离线对比）")
    parser.add_argument("--load_npu_dump", type=str, default=None,
                        help="指定 NPU dump 文件路径（用于离线对比）")
    parser.add_argument("--compare_mode", action="store_true",
                        help="启用三方对比模式（CPU vs GPU vs NPU），使用详细精度统计")
    parser.add_argument("--graph_mode",  action="store_true",
                        help="使用图模式调用 NPU 算子（需要 torchair）")
    parser.add_argument("--fail_analysis", action="store_true",
                        help="分析失败元素在各 batch/head 的分布情况，配合 --dump_tensors 使用")
    args = parser.parse_args()

    # 设备初始化
    if args.use_gpu:
        if not torch.cuda.is_available():
            print("[ERROR] CUDA 不可用，无法使用 GPU 模式。")
            sys.exit(1)
        torch.cuda.set_device(args.gpu_device)
        print(f"[INFO] GPU 模式，使用设备 cuda:{args.gpu_device}")
    elif args.compare_mode:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_device)
            print(f"[INFO] Compare 模式，使用 GPU 设备 cuda:{args.gpu_device}")
        elif args.load_gpu_dump and args.load_npu_dump:
            print(f"[INFO] Compare 模式，从 dump 文件对比（无 GPU/NPU）")
        else:
            try:
                import torch_npu
                torch.npu.set_device(args.device_id)
                print(f"[INFO] Compare 模式，使用 NPU 设备 npu:{args.device_id}")
            except ImportError:
                print("[ERROR] compare_mode 需要至少一种设备（GPU/NPU）或同时指定 --load_gpu_dump 和 --load_npu_dump")
                sys.exit(1)
    else:
        try:
            import torch_npu
            torch.npu.set_device(args.device_id)
            print(f"[INFO] NPU 模式，使用设备 npu:{args.device_id}")
        except ImportError:
            print("[ERROR] torch_npu 未安装，无法使用 NPU 模式。")
            print("[提示] 使用 --use_gpu 切换到 GPU 模式，或使用 --compare_mode --use_gpu 进行三方对比")
            sys.exit(1)

    # --case 优先覆盖 --case_files（仅限 base/fia/all）
    if args.case in ("base", "TestCases"):
        case_files = "test_case"
    elif args.case in ("fia", "TestCasesFIA"):
        case_files = "test_case_fia_STC"
    else:
        case_files = args.case_files

    module_names = [x.strip() for x in case_files.split(",")]
    all_cases = load_case_modules(module_names)

    run_cases = resolve_case_ids(args.case_id, all_cases)
    if not run_cases:
        print("[ERROR] 没有可运行的 case，退出。")
        sys.exit(1)

    results = {}
    for name in run_cases:
        # 清理 NPU 缓存，防止前一个 case 的显存残留影响后续 case
        gc.collect()
        if NPU_AVAILABLE:
            torch.npu.synchronize()
            torch.npu.empty_cache()

        config = all_cases[name]
        try:
            kwargs = normalize_case(config)
        except Exception as e:
            import traceback
            print(f"[ERROR] {name} normalize_case 异常: {e}")
            traceback.print_exc()
            results[name] = {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                                       "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')},
                              "lse":  {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                                       "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}}
            continue
        if args.use_gpu:
            kwargs["_use_gpu"] = True
            auto_load_npu = args.load_npu_dump or (args.dump_tensors and not NPU_AVAILABLE)
        else:
            kwargs["_use_gpu"] = False
            auto_load_gpu = args.load_gpu_dump or (args.dump_tensors and not GPU_AVAILABLE)

        print(f"\n{'='*66}")
        print(f"  Case: {name}  "
              f"B={kwargs.get('B')} N1={kwargs.get('N1')} N2={kwargs.get('N2')} "
              f"S1={kwargs.get('S1')} S2={kwargs.get('S2')} D={kwargs.get('D')} "
              f"layout={kwargs.get('input_layout')} layout_out={kwargs.get('layout_out')} dtype={kwargs.get('Dtype')}")
        print(f"{'='*66}")

        try:
            passed = call_flash_attn(
                name,
                dump_tensors=args.dump_tensors,
                dump_dir=args.dump_dir,
                verbose_diff=args.verbose_diff,
                visualize=args.visualize,
                viz_dir=args.viz_dir,
                meta_only=args.meta_only,
                compare_mode=args.compare_mode,
                graph_mode=args.graph_mode,
                load_gpu_dump=args.load_gpu_dump,
                load_npu_dump=args.load_npu_dump,
                fail_analysis=args.fail_analysis,
                **kwargs
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] {name} 运行异常: {e}")
            traceback.print_exc()
            passed = {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                               "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')},
                      "lse":  {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                               "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}}
        results[name] = passed

    # 汇总表格
    SEP = "─" * 120
    print(f"\n┌{SEP}┐")
    print(f"│  汇总结果  ({len(run_cases)} cases)")
    print(f"├{SEP}┤")
    max_name_len = max((len(n) for n in results.keys()), default=28)
    hdr = (f"│  {'Case':<{max_name_len}}  "
           f"{'Attn':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │  "
           f"{'LSE':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │")
    print(hdr)
    print(f"├{SEP}┤")
    pass_cnt = fail_cnt = 0
    for name, res in results.items():
        a = res["attn"]
        l = res["lse"]
        a_tag = "✓ PASS" if a["passed"] else "✗ FAIL"
        l_tag = "✓ PASS" if l["passed"] else "✗ FAIL"
        a_max = f"{a['max_abs']:.6f}" if a['total'] > 0 else "N/A"
        a_fr  = f"{a['fail_ratio']*100:.4f}%" if a['total'] > 0 else "N/A"
        l_max = f"{l['max_abs']:.6f}" if l['total'] > 0 else "---"
        l_fr  = f"{l['fail_ratio']*100:.4f}%" if l['total'] > 0 else "---"
        print(f"│  {name:<{max_name_len}}  "
              f"{a_tag:>8}  {a_max:>12}  {a_fr:>10}  │  "
              f"{l_tag:>8}  {l_max:>12}  {l_fr:>10}  │")
        both_pass = a["passed"] and l["passed"]
        if both_pass: pass_cnt += 1
        else:         fail_cnt += 1
    print(f"├{SEP}┤")
    print(f"│  通过: {pass_cnt}   失败: {fail_cnt}   共: {len(run_cases)}")
    print(f"└{SEP}┘")
    sys.exit(0 if fail_cnt == 0 else 1)
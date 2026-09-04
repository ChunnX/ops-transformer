# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN " AS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FlashAttnGrad binary test（本文件是远端 test_flash_attn_grad-binary.py 的独立副本，
给 msprof 性能用；原文件可能被其他 agent 改，不要互相覆盖）。

Modes:
  fuzz / reject / compare: 与原脚本相同（compare 里的 perf 仍是墙钟，勿当纯 kernel）。
  perf:        只发 PTO / AscendC kernel（不做 golden、不用墙钟）。应被 msprof 包住。
  perf-suite:  对每条用例各自执行一次 `msprof ... python ... --mode perf`，再从
               op_summary 的 Task Duration(us) 汇总纯 kernel 时间。
               不做 host-warmup / 窗口内 warmup。iters>1 时均值丢掉第一次。

Usage:
    # 推荐入口（环境 + 逐 case msprof）：
    bash run_fag_msprof.sh
    bash run_fag_msprof.sh --filter bn2_multihead --iters 5

    # 或手动（单 case）：
    msprof --output=/home/kjc/kjc_tmp/fag_msprof_binary/case0 --ai-core=on --aic-metrics=PipeUtilization \\
           --task-time=on --ascendcl=on \\
           python -u test_flash_attn_grad-binary-msprof.py --mode perf --filter bn2_multihead
    msprof --export=on --output=/home/kjc/kjc_tmp/fag_msprof_binary/case0/<PROF_*>
"""

import ast
import csv
import glob
import math
import os
import shutil
import sys
import json
import argparse
import subprocess
import time

import torch
import torch_npu
import cann_ops_transformer

try:
    import openpyxl
except ImportError:
    openpyxl = None

DEVICE = "npu"

# compare / perf 输入：N(0,1) 截断到 [-3, 3]，避免无界 randn 的极端尾部。
INPUT_NORMAL_LO = -3.0
INPUT_NORMAL_HI = 3.0


def _trunc_normal(*shape, device=DEVICE):
    """fp32 截断正态，再搬到 device。在 CPU 上 init，避开 NPU trunc_normal_ 差异。"""
    t = torch.empty(*shape, dtype=torch.float32)
    torch.nn.init.trunc_normal_(
        t, mean=0.0, std=1.0, a=INPUT_NORMAL_LO, b=INPUT_NORMAL_HI)
    return t.to(device)


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

HEAD_FAG_START_OFFSET_INDEX = 7
FAG_SPLIT_AXIS_INDEX = 0
FAG_BLOCK_OUTER_INDEX = 1
FAG_BLOCK_STARTS_OFFSET = 8
FAG_BLOCK_ENDS_OFFSET = 44
FAG_CORE_LIST_NUM = 36
FAG_SPLIT_AXIS_BN2GS1S2 = 0
FAG_SPLIT_AXIS_BN2 = 1
# AICPU 分核轴：2=BN2S2。与 tilingkey template=3（BN2S2 预留）不是同一个数。
FAG_SPLIT_AXIS_BN2S2 = 2
FAG_PRE_AIV_NUM_INDEX = 121
FAG_DQ_ROWS_INDEX = 122
FAG_DKV_ROWS_INDEX = 123
FAG_DQ_ROW_STARTS_OFFSET = 124
FAG_DQ_ROW_ENDS_OFFSET = 196
FAG_DKV_ROW_STARTS_OFFSET = 268
FAG_DKV_ROW_ENDS_OFFSET = 340
FAG_AIV_LIST_NUM = 72


def _parse_cell(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.lower() == "null" or s == "":
            return None
        if s.startswith("[") and s.endswith("]"):
            return ast.literal_eval(s)
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    return val


def _load_xlsx_cases(xlsx_path, dtype_filter=None):
    if openpyxl is None:
        raise ImportError("openpyxl is required. Install via: pip install openpyxl")
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    cases = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        case = {}
        for i, h in enumerate(headers):
            if h is None or i >= len(row):
                continue
            case[h] = _parse_cell(row[i])
        enable_val = case.get("enable")
        if enable_val is False or (isinstance(enable_val, str) and enable_val.lower() == "false"):
            continue
        if dtype_filter is not None:
            dtype_str = case.get("dtype", "float16")
            if dtype_str not in dtype_filter:
                continue
        cases.append(case)
    return cases


def _build_case_inputs(case):
    B = case["B"]
    N1 = case["N1"]
    N2 = case["N2"]
    S1 = case["S1"]
    S2 = case["S2"]
    D = case["D"]
    Dv = case.get("Dv") or D
    dtype_str = case.get("dtype", "float16")
    torch_dtype = _DTYPE_MAP[dtype_str]
    seed = case.get("manual_seed", 0) or 0
    layout = case.get("input_layout", "BSND") or "BSND"
    mask_mode = case.get("sparse_mode", 0) or 0

    torch.manual_seed(seed)

    seqlens_q = case.get("seqlens_list_q")
    seqlens_kv = case.get("seqlens_list_kv")
    cu_seqlens_q = None
    cu_seqlens_kv = None

    if layout == "TND":
        if seqlens_q is None:
            seqlens_q = [S1] * B
        if seqlens_kv is None:
            seqlens_kv = [S2] * B
        T1 = sum(seqlens_q)
        T2 = sum(seqlens_kv)
        q = torch.randn(T1, N1, D, dtype=torch_dtype, device=DEVICE)
        k = torch.randn(T2, N2, D, dtype=torch_dtype, device=DEVICE)
        v = torch.randn(T2, N2, Dv, dtype=torch_dtype, device=DEVICE)
        dout = torch.randn(T1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        attn_out = torch.randn(T1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        softmax_lse = torch.randn(N1, T1, dtype=torch.float32, device=DEVICE)
        cu_q = [0]
        for s in seqlens_q:
            cu_q.append(cu_q[-1] + s)
        cu_kv = [0]
        for s in seqlens_kv:
            cu_kv.append(cu_kv[-1] + s)
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
        cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=DEVICE)
    elif layout == "BNSD":
        q = torch.randn(B, N1, S1, D, dtype=torch_dtype, device=DEVICE)
        k = torch.randn(B, N2, S2, D, dtype=torch_dtype, device=DEVICE)
        v = torch.randn(B, N2, S2, Dv, dtype=torch_dtype, device=DEVICE)
        dout = torch.randn(B, N1, S1, Dv, dtype=torch_dtype, device=DEVICE)
        attn_out = torch.randn(B, N1, S1, Dv, dtype=torch_dtype, device=DEVICE)
        softmax_lse = torch.randn(B, N1, S1, dtype=torch.float32, device=DEVICE)
    elif layout in ("BSND", "BSH"):
        q = torch.randn(B, S1, N1, D, dtype=torch_dtype, device=DEVICE)
        k = torch.randn(B, S2, N2, D, dtype=torch_dtype, device=DEVICE)
        v = torch.randn(B, S2, N2, Dv, dtype=torch_dtype, device=DEVICE)
        dout = torch.randn(B, S1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        attn_out = torch.randn(B, S1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        softmax_lse = torch.randn(B, N1, S1, dtype=torch.float32, device=DEVICE)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    attn_mask = None
    if mask_mode in (3, 4):
        attn_mask = torch.ones(2048, 2048, dtype=torch.int8, device=DEVICE)

    return {
        "q": q, "k": k, "v": v, "dout": dout, "attn_out": attn_out,
        "softmax_lse": softmax_lse, "attn_mask": attn_mask,
        "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_kv": cu_seqlens_kv,
        "B": B, "N1": N1, "N2": N2, "S1": S1, "S2": S2, "D": D, "Dv": Dv,
        "dtype": torch_dtype,
        "layout": layout,
        "mask_mode": mask_mode,
        "win_left": case.get("win_left", -1) if case.get("win_left") is not None else -1,
        "win_right": case.get("win_right", -1) if case.get("win_right") is not None else -1,
    }


def _extract_fag_from_metadata(metadata_tensor):
    meta = metadata_tensor.cpu().tolist()
    fag_offset = meta[HEAD_FAG_START_OFFSET_INDEX]
    if fag_offset == 0:
        return None
    fag = meta[fag_offset:]
    return {
        'block_outer': fag[FAG_BLOCK_OUTER_INDEX],
        'block_starts': fag[FAG_BLOCK_STARTS_OFFSET:FAG_BLOCK_STARTS_OFFSET + FAG_CORE_LIST_NUM],
        'block_ends': fag[FAG_BLOCK_ENDS_OFFSET:FAG_BLOCK_ENDS_OFFSET + FAG_CORE_LIST_NUM],
    }


def dump_metadata(metadata, b=None, n2=None, s1=None, s2=None, core_num=36,
                  cube_base=128, template=None, n1=None):
    """打印 AICPU 算出的 metadata，并与 kernel 自己的 BN2 分核公式对照。

    metadata 由 AICPU 在 NPU 算子之前算好，所以它是「卡在 kernel 之前还是
    之后」的第一个观察点，也是核对分核方案的唯一依据。
    """
    meta = metadata.cpu().tolist()
    print(f"  [meta] len={len(meta)} head[0:8]={meta[:8]}")
    fag_off = meta[HEAD_FAG_START_OFFSET_INDEX] if len(meta) > HEAD_FAG_START_OFFSET_INDEX else 0
    print(f"  [meta] head[{HEAD_FAG_START_OFFSET_INDEX}] fag_offset={fag_off}")
    if fag_off <= 0 or fag_off >= len(meta):
        print("  [meta] FAG region absent")
        return None
    fag = meta[fag_off:]
    bo = fag[FAG_BLOCK_OUTER_INDEX]
    starts = fag[FAG_BLOCK_STARTS_OFFSET:FAG_BLOCK_STARTS_OFFSET + FAG_CORE_LIST_NUM]
    ends = fag[FAG_BLOCK_ENDS_OFFSET:FAG_BLOCK_ENDS_OFFSET + FAG_CORE_LIST_NUM]
    used = sum(1 for i in range(FAG_CORE_LIST_NUM) if ends[i] > starts[i])
    print(f"  [meta] fag[0:8]={fag[:8]}  block_outer={bo}  used_cores={used}")
    axis = fag[FAG_SPLIT_AXIS_INDEX] if len(fag) > FAG_SPLIT_AXIS_INDEX else None
    axis_name = {0: "GS1S2", 1: "BN2", 2: "BN2S2"}.get(axis, str(axis))
    print(f"  [meta] splitAxis={axis}({axis_name}) expected_template={template}")
    mismatch = False
    if template == 1 and axis != FAG_SPLIT_AXIS_BN2:
        print("  [FAIL] AICPU splitAxis 不是 BN2，但 host tmpl=1")
        mismatch = True
    if template == 0 and axis == FAG_SPLIT_AXIS_BN2:
        print("  [FAIL] AICPU 标成 BN2，但 host tmpl=0（禁止 silently 错模板）")
        mismatch = True
    if template == 0 and axis == FAG_SPLIT_AXIS_BN2S2:
        print("  [FAIL] AICPU 标成 BN2S2（splitAxis=2），kernel 预留 tmpl=3 本轮未编")
        mismatch = True
    print(f"  [meta] blockStarts[0:12]={starts[:12]}")
    print(f"  [meta] blockEnds  [0:12]={ends[:12]}")
    n_aiv = fag[FAG_PRE_AIV_NUM_INDEX] if len(fag) > FAG_PRE_AIV_NUM_INDEX else 0
    dq_rows = fag[FAG_DQ_ROWS_INDEX] if len(fag) > FAG_DQ_ROWS_INDEX else 0
    dkv_rows = fag[FAG_DKV_ROWS_INDEX] if len(fag) > FAG_DKV_ROWS_INDEX else 0
    dq_aiv_s = fag[FAG_DQ_ROW_STARTS_OFFSET:FAG_DQ_ROW_STARTS_OFFSET + FAG_AIV_LIST_NUM]
    dq_aiv_e = fag[FAG_DQ_ROW_ENDS_OFFSET:FAG_DQ_ROW_ENDS_OFFSET + FAG_AIV_LIST_NUM]
    print(f"  [meta] nAiv={n_aiv} dqRows={dq_rows} dkvRows={dkv_rows}")
    expect_naiv = min(int(bo) * 2, FAG_AIV_LIST_NUM) if bo else 0
    if n_aiv != expect_naiv:
        print(f"  [FAIL] nAiv={n_aiv} != usedCube*2={expect_naiv}")
        mismatch = True
    if None not in (b, s1) and (n1 is not None or n2 is not None):
        n_q = n1 if n1 is not None else n2
        expect_dq = int(b) * int(s1) * int(n_q)
        if dq_rows != expect_dq:
            print(f"  [FAIL] dqRows={dq_rows} != B*S1*N1={expect_dq}")
            mismatch = True
    if None not in (b, s2, n2):
        expect_dkv = int(b) * int(s2) * int(n2)
        if dkv_rows != expect_dkv:
            print(f"  [FAIL] dkvRows={dkv_rows} != B*S2*N2={expect_dkv}")
            mismatch = True
    dq_first8 = [(dq_aiv_s[i], dq_aiv_e[i]) for i in range(min(8, len(dq_aiv_s)))]
    print(f"  [meta] dqAiv[0:8]={dq_first8}")
    live_rows = [dq_aiv_e[i] - dq_aiv_s[i] for i in range(len(dq_aiv_s))
                 if i < len(dq_aiv_e) and dq_aiv_e[i] > dq_aiv_s[i]]
    occ_span = (max(live_rows) - min(live_rows)) if live_rows else 0
    print(f"  [meta] dq live cores={len(live_rows)} max-min={occ_span}")
    if (template == 0 and n_aiv > 1 and live_rows and occ_span > 1):
        print("  [FAIL] GS1S2 pre/post 行负载 max-min>1")
        mismatch = True
    if None not in (b, n2, s1, s2):
        s1o = (s1 + cube_base - 1) // cube_base
        s2o = (s2 + cube_base - 1) // cube_base
        n_q = n1 if n1 is not None else n2
        total = b * n_q * s1o * s2o
        if template == 1:
            # BN2：BN 合轴后按核均分，每核拿整份 (s1Outer*s2Outer)
            factor = ((b * n2 + core_num - 1) // core_num) * s1o * s2o
        else:
            # BN2GS1S2：直接按总块数均分
            factor = (total + core_num - 1) // core_num
        kernel_split = []
        for c in range(core_num):
            st = c * factor
            cnt = max(min(st + factor, total) - st, 0)
            if cnt:
                kernel_split.append((c, st, st + cnt))
        print(f"  [kernel] template={template} coreNum={core_num} s1Outer={s1o} "
              f"s2Outer={s2o} totalBlocks={total} blockFactor={factor}")
        print(f"  [kernel] busy_cores={len(kernel_split)} first={kernel_split[:6]}")
        meta_split = [(i, starts[i], ends[i]) for i in range(FAG_CORE_LIST_NUM)
                      if ends[i] > starts[i]]
        print(f"  [meta ] busy_cores={len(meta_split)} first={meta_split[:6]}")
        if meta_split != kernel_split:
            print("  [note] 上面那行 [kernel] 只是旧的内置公式，用来对账；"
                  "kernel 线性分核实际读的是 metadata 的 [start, end)")
    return {"block_outer": bo, "starts": starts, "ends": ends,
            "split_axis": axis, "mismatch": mismatch, "used_cores": used,
            "n_aiv": n_aiv, "dq_rows": dq_rows, "dq_first8": dq_first8,
            "dq_span": occ_span}


def _check_fag_block_starts_ends(metadata):
    meta = metadata.cpu().tolist()
    fag_offset = meta[HEAD_FAG_START_OFFSET_INDEX] if len(meta) > HEAD_FAG_START_OFFSET_INDEX else 0
    if fag_offset == 0 or fag_offset >= len(meta):
        print(f"  [WARN] FAG region not populated (head[7]={fag_offset}), skip FAG check")
        return
    fag = meta[fag_offset:]
    bo = fag[FAG_BLOCK_OUTER_INDEX] if len(fag) > FAG_BLOCK_OUTER_INDEX else 0
    if bo <= 0 or bo > FAG_CORE_LIST_NUM:
        print(f"  [WARN] FAG block_outer={bo} (invalid), skip FAG check")
        return
    starts = fag[FAG_BLOCK_STARTS_OFFSET:FAG_BLOCK_STARTS_OFFSET + FAG_CORE_LIST_NUM]
    ends = fag[FAG_BLOCK_ENDS_OFFSET:FAG_BLOCK_ENDS_OFFSET + FAG_CORE_LIST_NUM]
    assert starts[0] == 0, f"blockStarts[0]={starts[0]}, expected 0"
    assert ends[bo - 1] > 0, f"blockEnds[{bo-1}]={ends[bo - 1]}, expected > 0"
    for i in range(bo, FAG_CORE_LIST_NUM):
        assert starts[i] == 0, f"blockStarts[{i}]={starts[i]}, expected 0 (unused core)"
        assert ends[i] == 0, f"blockEnds[{i}]={ends[i]}, expected 0 (unused core)"
    print(f"  [OK] FAG block_outer={bo}")


# ============================================================
# Fuzz mode (original)
# ============================================================

def _run_case(case):
    name = case.get("testcase_name", "unnamed")
    B = case["B"]
    N1 = case["N1"]
    N2 = case["N2"]
    S1 = case["S1"]
    S2 = case["S2"]
    D = case["D"]
    layout = case.get("input_layout", "BSND") or "BSND"
    mask_mode = case.get("sparse_mode", 0) or 0
    win_left = case.get("win_left")
    win_right = case.get("win_right")

    print(f"\n  case: {name} (B={B} N1={N1} N2={N2} S1={S1} S2={S2} D={D}"
          f" mask={mask_mode} layout={layout} dtype={case.get('dtype')})")

    inputs = _build_case_inputs(case)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    dout, attn_out = inputs["dout"], inputs["attn_out"]
    softmax_lse = inputs["softmax_lse"]
    attn_mask = inputs["attn_mask"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_kv = inputs["cu_seqlens_kv"]

    softmax_scale = 1.0 / math.sqrt(D)

    metadata = cann_ops_transformer.flash_attn_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        mask_mode=mask_mode,
        win_left=win_left if win_left is not None else -1,
        win_right=win_right if win_right is not None else -1,
        layout_q=layout,
        layout_kv=layout,
        layout_out=layout,
        is_grad_enabled=True,
    )
    assert metadata is not None, "metadata is None"
    assert metadata.dtype == torch.int32, f"metadata dtype={metadata.dtype}"
    _check_fag_block_starts_ends(metadata)
    print(f"  [OK] metadata: shape={tuple(metadata.shape)}")

    grad_kwargs = dict(
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        win_left=win_left if win_left is not None else -1,
        win_right=win_right if win_right is not None else -1,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        layout_q=layout,
        layout_kv=layout,
        layout_out=layout,
    )
    if cu_seqlens_q is not None:
        grad_kwargs["cu_seqlens_q"] = cu_seqlens_q
        grad_kwargs["cu_seqlens_kv"] = cu_seqlens_kv

    dq, dk, dv = cann_ops_transformer.flash_attn_grad(
        q, k, v, dout, attn_out, softmax_lse,
        **grad_kwargs,
    )

    assert dq.shape == q.shape, f"dq shape mismatch: {dq.shape} vs {q.shape}"
    assert dk.shape == k.shape, f"dk shape mismatch: {dk.shape} vs {k.shape}"
    assert dv.shape == v.shape, f"dv shape mismatch: {dv.shape} vs {v.shape}"
    assert dq.dtype == q.dtype, f"dq dtype mismatch: {dq.dtype} vs {q.dtype}"
    print(f"  [PASS] dq={tuple(dq.shape)} dk={tuple(dk.shape)} dv={tuple(dv.shape)}")


def _run_fuzz_mode(args):
    if args.xlsx is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.xlsx = os.path.join(script_dir, "fag_test_case.xlsx")

    print(f"[fuzz] xlsx: {args.xlsx}")
    cases = _load_xlsx_cases(args.xlsx, dtype_filter=_DTYPE_MAP.keys())
    print(f"[fuzz] Loaded {len(cases)} cases")

    if not cases:
        print("[fuzz] No valid xlsx cases found, falling back to built-in cases")
        cases = [
            {"testcase_name": "builtin_dense_small", "B": 4, "N1": 8, "N2": 4, "S1": 128,
             "S2": 128, "D": 128, "Dv": 128, "dtype": "float16", "input_layout": "BSND",
             "sparse_mode": 0, "win_left": -1, "win_right": -1, "manual_seed": 42},
            {"testcase_name": "builtin_causal", "B": 2, "N1": 4, "N2": 2, "S1": 256,
             "S2": 256, "D": 128, "Dv": 128, "dtype": "float16", "input_layout": "BSND",
             "sparse_mode": 3, "win_left": -1, "win_right": -1, "manual_seed": 42},
        ]

    pass_count = 0
    fail_count = 0
    for idx, case in enumerate(cases):
        name = case.get("testcase_name", f"case_{idx}")
        try:
            _run_case(case)
            torch.npu.synchronize()
            pass_count += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            fail_count += 1
            torch.npu.synchronize()

    return pass_count, fail_count


# ============================================================
# Reject mode (from fag_reject_case.xlsx)
# ============================================================

def _make_dummy_metadata(B, N1, N2, S1, S2, D, layout, mask_mode):
    """Create a valid metadata tensor for fag tiling rejection tests."""
    cu_seqlens_q = None
    cu_seqlens_kv = None
    if layout == "TND":
        cu_q = [0]
        cu_kv = [0]
        for _ in range(B):
            cu_q.append(cu_q[-1] + S1)
            cu_kv.append(cu_kv[-1] + S2)
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
        cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=DEVICE)

    return cann_ops_transformer.flash_attn_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        layout_q=layout,
        layout_kv=layout,
        layout_out=layout,
        is_grad_enabled=True,
    )


def _build_base_tensors(B, N1, N2, S1, S2, D, Dv, dtype_str, layout):
    """Build standard input tensors for fag tiling rejection tests."""
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float16)
    torch.manual_seed(42)
    if layout == "TND":
        T1 = S1 * B
        T2 = S2 * B
        q = torch.randn(T1, N1, D, dtype=torch_dtype, device=DEVICE)
        k = torch.randn(T2, N2, D, dtype=torch_dtype, device=DEVICE)
        v = torch.randn(T2, N2, Dv, dtype=torch_dtype, device=DEVICE)
        dout = torch.randn(T1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        attn_out = torch.randn(T1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        softmax_lse = torch.randn(N1, T1, dtype=torch.float32, device=DEVICE)
        cu_q = [0]
        cu_kv = [0]
        for _ in range(B):
            cu_q.append(cu_q[-1] + S1)
            cu_kv.append(cu_kv[-1] + S2)
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
        cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=DEVICE)
    elif layout == "BNSD":
        q = torch.randn(B, N1, S1, D, dtype=torch_dtype, device=DEVICE)
        k = torch.randn(B, N2, S2, D, dtype=torch_dtype, device=DEVICE)
        v = torch.randn(B, N2, S2, Dv, dtype=torch_dtype, device=DEVICE)
        dout = torch.randn(B, N1, S1, Dv, dtype=torch_dtype, device=DEVICE)
        attn_out = torch.randn(B, N1, S1, Dv, dtype=torch_dtype, device=DEVICE)
        softmax_lse = torch.randn(B, N1, S1, dtype=torch.float32, device=DEVICE)
        cu_seqlens_q = None
        cu_seqlens_kv = None
    else:
        q = torch.randn(B, S1, N1, D, dtype=torch_dtype, device=DEVICE)
        k = torch.randn(B, S2, N2, D, dtype=torch_dtype, device=DEVICE)
        v = torch.randn(B, S2, N2, Dv, dtype=torch_dtype, device=DEVICE)
        dout = torch.randn(B, S1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        attn_out = torch.randn(B, S1, N1, Dv, dtype=torch_dtype, device=DEVICE)
        softmax_lse = torch.randn(B, N1, S1, dtype=torch.float32, device=DEVICE)
        cu_seqlens_q = None
        cu_seqlens_kv = None
    return q, k, v, dout, attn_out, softmax_lse, cu_seqlens_q, cu_seqlens_kv


def _apply_tensor_overrides(tensors, ov, B, N1, N2, S1, S2, D, Dv, dtype_str, layout):
    """Apply override JSON to modify tensors in-place. Returns modified tensors dict."""
    q, k, v, dout, attn_out, softmax_lse, cu_q, cu_kv = tensors
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

    # dtype overrides
    if "k_dtype" in ov:
        k = k.to(_DTYPE_MAP[ov["k_dtype"]])
    if "v_dtype" in ov:
        v = v.to(_DTYPE_MAP[ov["v_dtype"]])
    if "dout_dtype" in ov:
        dout = dout.to(_DTYPE_MAP[ov["dout_dtype"]])

    # shape overrides
    if "q_ndim" in ov and ov["q_ndim"] == 5:
        q = torch.randn(B, S1, N1, D, D, dtype=torch_dtype, device=DEVICE)
    if "k_d" in ov:
        k = torch.randn(*k.shape[:-1], ov["k_d"], dtype=k.dtype, device=DEVICE)
    if "dout_d" in ov:
        dout = torch.randn(*dout.shape[:-1], ov["dout_d"], dtype=dout.dtype, device=DEVICE)
    if "dout_n" in ov:
        shape = list(dout.shape)
        n_idx = 1 if layout == "BNSD" else 2
        shape[n_idx] = ov["dout_n"]
        dout = torch.randn(*shape, dtype=dout.dtype, device=DEVICE)
    if "dout_s" in ov:
        shape = list(dout.shape)
        s_idx = 2 if layout == "BNSD" else 1
        shape[s_idx] = ov["dout_s"]
        dout = torch.randn(*shape, dtype=dout.dtype, device=DEVICE)
    if "k_b" in ov:
        shape = list(k.shape)
        shape[0] = ov["k_b"]
        k = torch.randn(*shape, dtype=k.dtype, device=DEVICE)

    # attn_mask overrides
    attn_mask = None
    mask_mode = ov.get("mask_mode", ov.get("sparse_mode", 0))
    am = ov.get("attn_mask")
    if am == "none":
        attn_mask = None
    elif am == "provided":
        attn_mask = torch.ones(2048, 2048, dtype=torch.int8, device=DEVICE)
    elif am == "3D":
        attn_mask = torch.ones(2048, 2048, 1, dtype=torch.int8, device=DEVICE)
    elif am == "wrong_shape":
        attn_mask = torch.ones(1024, 1024, dtype=torch.int8, device=DEVICE)
    elif mask_mode in (3, 4) and am != "none":
        attn_mask = torch.ones(2048, 2048, dtype=torch.int8, device=DEVICE)

    # cu_seqlens overrides
    if ov.get("cu_seqlens_q") == "missing":
        cu_q = None
    elif ov.get("cu_seqlens_q") == "force":
        cu_q = torch.tensor([0] + [S1] * B, dtype=torch.int32, device=DEVICE)
    if ov.get("cu_seqlens_kv") == "missing":
        cu_kv = None
    elif ov.get("cu_seqlens_kv") == "force":
        cu_kv = torch.tensor([0] + [S2] * B, dtype=torch.int32, device=DEVICE)

    return {
        "q": q, "k": k, "v": v, "dout": dout, "attn_out": attn_out,
        "softmax_lse": softmax_lse, "attn_mask": attn_mask,
        "cu_seqlens_q": cu_q, "cu_seqlens_kv": cu_kv,
    }


def _build_meta_params(case, ov):
    """Build metadata call params from case + override. Returns dict for flash_attn_metadata."""
    B = ov.get("batch_size", case.get("B", 2))
    N1 = ov.get("num_heads_q", case.get("N1", 4))
    N2 = ov.get("num_heads_kv", case.get("N2", 2))
    D = ov.get("head_dim", case.get("D", 128))
    S1 = ov.get("max_seqlen_q", case.get("S1", 128))
    S2 = ov.get("max_seqlen_kv", case.get("S2", 128))
    mask_mode = ov.get("mask_mode", case.get("sparse_mode", 0))
    win_left = ov.get("win_left", case.get("win_left", -1) if case.get("win_left") is not None else -1)
    win_right = ov.get("win_right", case.get("win_right", -1) if case.get("win_right") is not None else -1)
    layout_q = ov.get("layout_q", case.get("input_layout", "BSND"))
    layout_kv = ov.get("layout_kv", layout_q)
    layout_out = ov.get("layout_out", layout_q)

    cu_q = None
    cu_kv = None
    if layout_q == "TND" and ov.get("cu_seqlens_q") != "missing":
        cu_q_list = [0]
        for _ in range(B):
            cu_q_list.append(cu_q_list[-1] + S1)
        cu_q = torch.tensor(cu_q_list, dtype=torch.int32, device=DEVICE)
    if layout_kv == "TND" and ov.get("cu_seqlens_kv") != "missing":
        cu_kv_list = [0]
        for _ in range(B):
            cu_kv_list.append(cu_kv_list[-1] + S2)
        cu_kv = torch.tensor(cu_kv_list, dtype=torch.int32, device=DEVICE)

    if ov.get("cu_seqlens_q") == "force":
        cu_q = torch.tensor([0] + [S1] * B, dtype=torch.int32, device=DEVICE)
    elif ov.get("cu_seqlens_q") == "wrong_shape":
        cu_q = torch.tensor([0, S1], dtype=torch.int32, device=DEVICE)
    if ov.get("cu_seqlens_kv") == "force":
        cu_kv = torch.tensor([0] + [S2] * B, dtype=torch.int32, device=DEVICE)

    seqused_q = None
    seqused_kv = None
    if ov.get("seqused_q") == "force":
        seqused_q = torch.tensor([S1] * B, dtype=torch.int32, device=DEVICE)
    elif ov.get("seqused_q") == "wrong_shape":
        seqused_q = torch.tensor([S1], dtype=torch.int32, device=DEVICE)

    return dict(
        num_heads_q=N1, num_heads_kv=N2, head_dim=D,
        cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv,
        batch_size=B, max_seqlen_q=S1, max_seqlen_kv=S2,
        mask_mode=mask_mode, win_left=win_left, win_right=win_right,
        layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
        is_grad_enabled=True,
        seqused_q=seqused_q, seqused_kv=seqused_kv,
    )


def _run_metadata_reject_subprocess(name, meta_params):
    """Run a metadata rejection case in a subprocess to survive segfaults."""
    print(f"\n  [reject][metadata] {name}")
    script = '''
import sys, json
import torch
import torch_npu
import cann_ops_transformer

params = json.loads(sys.argv[1])
cu_q = params.pop("cu_seqlens_q", None)
cu_kv = params.pop("cu_seqlens_kv", None)
seqused_q = params.pop("seqused_q", None)
seqused_kv = params.pop("seqused_kv", None)
if cu_q is not None:
    cu_q = torch.tensor(cu_q, dtype=torch.int32, device="npu")
if cu_kv is not None:
    cu_kv = torch.tensor(cu_kv, dtype=torch.int32, device="npu")
if seqused_q is not None:
    seqused_q = torch.tensor(seqused_q, dtype=torch.int32, device="npu")
if seqused_kv is not None:
    seqused_kv = torch.tensor(seqused_kv, dtype=torch.int32, device="npu")
params["cu_seqlens_q"] = cu_q
params["cu_seqlens_kv"] = cu_kv
params["seqused_q"] = seqused_q
params["seqused_kv"] = seqused_kv

try:
    cann_ops_transformer.flash_attn_metadata(**params)
    print("RESULT:NOT_REJECTED")
except Exception as e:
    print("RESULT:REJECTED:" + str(e)[:200])
'''
    # Serialize tensors as lists
    serializable = dict(meta_params)
    for k in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        v = serializable.get(k)
        if v is not None:
            serializable[k] = v.cpu().tolist()
    params_json = json.dumps(serializable, default=str)
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-c", script, params_json],
        capture_output=True, text=True, timeout=60, env=env,
    )
    stdout = proc.stdout.strip()
    if proc.returncode != 0:
        if proc.returncode < 0:
            return True, f"crashed (signal {-proc.returncode})"
        return True, f"nonzero exit ({proc.returncode})"
    if "RESULT:NOT_REJECTED" in stdout:
        return False, "Expected rejection but call succeeded"
    for line in stdout.splitlines():
        if line.startswith("RESULT:REJECTED:"):
            return True, line[len("RESULT:REJECTED:"):]
    return True, f"unexpected output: {stdout[:200]}"


def _extract_plog_detail(before_call_line_count):
    """Search plog for the latest error detail from a FAG reject call.
    
    Searches for flash_attn_grad_tiling_check.cpp errors (tiling layer)
    and CheckSocVersionAndParam errors (framework OpDef dtype check).
    """
    pid = os.getpid()
    plog_dirs = [
        "/root/ascend/log/debug/plog",
        "/var/log/npu/log/debug/plog",
    ]
    plog_file = None
    for d in plog_dirs:
        if not os.path.isdir(d):
            continue
        candidates = [f for f in os.listdir(d) if f.startswith(f"plog-{pid}_")]
        if candidates:
            candidates.sort(reverse=True)
            plog_file = os.path.join(d, candidates[0])
            break
    if plog_file is None:
        return ""
    try:
        with open(plog_file, 'r', errors='replace') as f:
            lines = f.readlines()
        new_lines = lines[before_call_line_count:]
        skip_patterns = ["CheckParams failed", "CheckMaskAndAttnMask failed",
                         "CheckAttrs failed", "CheckLayout failed",
                         "CheckShape failed", "CheckDtypeConsistency failed",
                         "CheckInputExistence failed", "FlashAttnGradTilingFunc"]
        for line in reversed(new_lines):
            if "[ERROR]" not in line:
                continue
            if "flash_attn_grad_tiling_check.cpp" in line:
                if any(s in line for s in skip_patterns):
                    continue
                idx = line.find("OpName:[FlashAttnGrad]")
                if idx >= 0:
                    return line[idx + len("OpName:[FlashAttnGrad]"):].strip()
            if "CheckSocVersionAndParam" in line and "Reason:" in line:
                idx = line.find("Reason:")
                if idx >= 0:
                    return line[idx:].strip()
    except Exception:
        pass
    return ""


def _run_fag_reject_case(name, case, ov):
    """Run a single fag tiling rejection case. Expect it to raise an error."""
    print(f"\n  [reject][fag] {name}")
    B = ov.get("B", case.get("B", 2))
    N1 = ov.get("N1", case.get("N1", 4))
    N2 = ov.get("N2", case.get("N2", 2))
    S1 = ov.get("S1", case.get("S1", 128))
    S2 = ov.get("S2", case.get("S2", 128))
    D = case.get("D", 128)
    Dv = ov.get("Dv", case.get("Dv", D))
    dtype_str = case.get("dtype", "float16")
    layout = ov.get("layout", ov.get("input_layout", case.get("input_layout", "BSND")))

    base_tensors = _build_base_tensors(B, N1, N2, S1, S2, D, Dv, dtype_str, layout)
    tensors = _apply_tensor_overrides(base_tensors, ov, B, N1, N2, S1, S2, D, Dv, dtype_str, layout)

    meta_N1 = 4 if N1 == 5 and N2 == 3 else N1
    meta_N2 = 2 if N1 == 5 and N2 == 3 else N2
    meta_mask = ov.get("mask_mode", case.get("sparse_mode", 0))
    if meta_mask not in (0, 3, 4):
        meta_mask = 0
    metadata = _make_dummy_metadata(B, meta_N1, meta_N2, S1, S2, D, layout, meta_mask)

    mask_mode = ov.get("mask_mode", case.get("sparse_mode", 0))
    win_left = ov.get("win_left", case.get("win_left", -1) if case.get("win_left") is not None else -1)
    win_right = ov.get("win_right", case.get("win_right", -1) if case.get("win_right") is not None else -1)
    max_seqlen_q = ov.get("max_seqlen_q", S1)
    max_seqlen_kv = ov.get("max_seqlen_kv", S2)
    layout_q = ov.get("layout_q", layout)
    layout_kv = ov.get("layout_kv", layout)
    layout_out = ov.get("layout_out", layout)

    grad_kwargs = dict(
        attn_mask=tensors["attn_mask"],
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(D),
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
    )
    if tensors["cu_seqlens_q"] is not None:
        grad_kwargs["cu_seqlens_q"] = tensors["cu_seqlens_q"]
        grad_kwargs["cu_seqlens_kv"] = tensors["cu_seqlens_kv"]
    elif layout == "TND":
        grad_kwargs["cu_seqlens_q"] = None
        grad_kwargs["cu_seqlens_kv"] = None

    try:
        pid = os.getpid()
        plog_path = None
        for d in ["/root/ascend/log/debug/plog", "/var/log/npu/log/debug/plog"]:
            if os.path.isdir(d):
                cands = [f for f in os.listdir(d) if f.startswith(f"plog-{pid}_")]
                if cands:
                    cands.sort(reverse=True)
                    plog_path = os.path.join(d, cands[0])
                    break
        before_count = 0
        if plog_path and os.path.exists(plog_path):
            with open(plog_path, 'r', errors='replace') as f:
                before_count = sum(1 for _ in f)

        cann_ops_transformer.flash_attn_grad(
            tensors["q"], tensors["k"], tensors["v"],
            tensors["dout"], tensors["attn_out"], tensors["softmax_lse"],
            **grad_kwargs,
        )
        return False, "Expected rejection but call succeeded"
    except Exception as e:
        import time
        time.sleep(0.3)
        detail = _extract_plog_detail(before_count)
        if detail:
            return True, detail
        return True, str(e)[:200]


def _run_reject_mode(args):
    """Run all rejection test cases from fag_reject_case.xlsx."""
    if args.xlsx is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.xlsx = os.path.join(script_dir, "fag_reject_case.xlsx")

    print(f"[reject] xlsx: {args.xlsx}")
    cases = _load_xlsx_cases(args.xlsx)
    print(f"[reject] Loaded {len(cases)} cases")

    meta_cases = [c for c in cases if c.get("target") == "metadata"]
    fag_cases = [c for c in cases if c.get("target") == "fag"]

    if args.target in ("all", "metadata"):
        print(f"\n{'='*60}")
        print(f"  Metadata rejection tests ({len(meta_cases)} cases)")
        print(f"{'='*60}")
    pass_count = 0
    fail_count = 0

    if args.target in ("all", "metadata"):
        for case in meta_cases:
            name = case.get("testcase_name", "unnamed")
            ov_str = case.get("override", "")
            ov = json.loads(ov_str) if ov_str else {}
            try:
                meta_params = _build_meta_params(case, ov)
                rejected, msg = _run_metadata_reject_subprocess(name, meta_params)
                if rejected:
                    print(f"  [PASS] {name}: rejected ({msg})")
                    pass_count += 1
                else:
                    print(f"  [FAIL] {name}: {msg}")
                    fail_count += 1
            except Exception as e:
                print(f"  [FAIL] {name}: unexpected error: {e}")
                fail_count += 1

    if args.target in ("all", "fag"):
        print(f"\n{'='*60}")
        print(f"  FAG tiling rejection tests ({len(fag_cases)} cases)")
        print(f"{'='*60}")
        for case in fag_cases:
            name = case.get("testcase_name", "unnamed")
            ov_str = case.get("override", "")
            ov = json.loads(ov_str) if ov_str else {}
            try:
                rejected, msg = _run_fag_reject_case(name, case, ov)
                if rejected:
                    print(f"  [PASS] {name}: rejected ({msg})")
                    pass_count += 1
                else:
                    print(f"  [FAIL] {name}: {msg}")
                    fail_count += 1
            except Exception as e:
                print(f"  [FAIL] {name}: unexpected error: {e}")
                fail_count += 1
            torch.npu.synchronize()

    return pass_count, fail_count


# ============================================================
# Compare mode: xlsx 驱动，二进制 pypto vs AscendC vs fp32 golden
# ============================================================

INT32_MAX = 2147483647

# metadata / host 放行 D,Dv ∈ (0, 192]。旧白名单 {64,128,256} 已放开。
HEAD_DIM_MAX = 192
# compare 的用例表：enable=TRUE 的行就跑。
COMPARE_XLSX = "fag_test_case_kernel.xlsx"
# kernel 只实现了非 TND 的稠密路径；GQA / 不等 D 已接通。
COMPARE_LAYOUTS = ("BSND", "BNSD")
# op def 声明了 BF16，但 kernel 的 make_tensor/tile/astype 全写死 DT_FP16，
# 编出来的 bf16 二进制其实是同一份 fp16 逻辑，按 bf16 的位读进去就是垃圾值
# （实测 rel 2e36）。等 kernel 参数化 dtype 后把 bfloat16 加回来即可。
COMPARE_DTYPES = ("float16",)

CUBE_BASE = 128
L2_CACHE_BYTES = 128 * 1024 * 1024


def _norm_case(case):
    """xlsx 行 -> compare 用的字段，缺项给默认值。"""
    d = case.get("D") or case.get("Dq") or 128
    return {
        "name": case.get("testcase_name") or "unnamed",
        "b": int(case["B"]), "n1": int(case["N1"]), "n2": int(case["N2"]),
        "s1": int(case["S1"]), "s2": int(case["S2"]),
        "d": int(d), "dv": int(case.get("Dv") or d),
        "dtype": (case.get("dtype") or "float16").strip(),
        "layout": (case.get("input_layout") or "BSND").strip().upper(),
        "mask": int(case.get("sparse_mode") or 0),
        "seed": int(case.get("manual_seed") or 42),
    }


def _unsupported_reason(c):
    """挡在跑之前，让不支持的行给出原因而不是一串异常栈。"""
    if c["n2"] <= 0 or c["n1"] % c["n2"] != 0:
        return f"N1={c['n1']} 不能被 N2={c['n2']} 整除"
    if c["layout"] not in COMPARE_LAYOUTS:
        return f"layout={c['layout']} 未实现 (只有 {'/'.join(COMPARE_LAYOUTS)})"
    if c["dtype"] not in COMPARE_DTYPES:
        return (f"dtype={c['dtype']}：kernel 写死 fp16，bf16 二进制是同一份 "
                f"fp16 逻辑，跑出来是垃圾值")
    if c["mask"] != 0:
        return f"sparse_mode={c['mask']}，kernel 无 attenMask"
    if not (0 < c["d"] <= HEAD_DIM_MAX):
        return f"D={c['d']} 不在 (0, {HEAD_DIM_MAX}]"
    if not (0 < c["dv"] <= HEAD_DIM_MAX):
        return f"Dv={c['dv']} 不在 (0, {HEAD_DIM_MAX}]"
    if c["dv"] > c["d"]:
        return f"Dv={c['dv']} > D={c['d']}"
    return None


# Must match flash_attn_grad_block_outer.h / metadata InitFagBaseDims.
_FAG_S1_INNER = 64
_FAG_S2_INNER = 128
_FAG_S1CV_RATIO = 2
_FAG_S2CV_RATIO = 1


def _fag_outer(seq, inner, ratio):
    if seq == 0:
        return 0
    cv = seq if inner * ratio > seq else inner * ratio
    return (seq + cv - 1) // cv


def _fag_dense_block_outer(fused, aic_num):
    if fused <= 0 or aic_num <= 0:
        return 0
    factor = (fused + aic_num - 1) // aic_num
    return (fused + factor - 1) // factor


_BN2_MAX_S = 128
_BN2_MAX_D = 128
_BN2_MB_SEQ = 640
_BN2_MB_BN128 = 128
_BN2_MB_BN256 = 256


def _decide_bn2_multiblk(c):
    b, n1, n2 = c["b"], c["n1"], c["n2"]
    s1, s2, d = c["s1"], c["s2"], c["d"]
    dv = c.get("dv", d)
    if c.get("layout", "BSND") == "TND":
        return 0
    if (c.get("mask_mode", 0) or 0) != 0:
        return 0
    if n1 != n2 or d > _BN2_MAX_D or dv != d:
        return 0
    if not ((s1 > _BN2_MAX_S or s2 > _BN2_MAX_S) and
            s1 <= _BN2_MB_SEQ and s2 <= _BN2_MB_SEQ):
        return 0
    bn = b * n1
    if bn >= _BN2_MB_BN256:
        return 1
    if bn >= _BN2_MB_BN128 and (s1 % 128 == 0) and (s2 % 128 == 0):
        return 1
    return 0


def _decide_template(c):
    n1, n2, s1, s2, d = c["n1"], c["n2"], c["s1"], c["s2"], c["d"]
    if c.get("layout", "BSND") == "TND":
        return 0
    if n1 == n2 and d <= _BN2_MAX_D and s1 <= _BN2_MAX_S and s2 <= _BN2_MAX_S:
        return 1
    if _decide_bn2_multiblk(c):
        return 1
    return 0


def _predict_used_cube(c, aic_num=None):
    """复刻 host FagUsedCubeCores：dense 非 TND 返回实际 blockOuter，否则满核。

    返回 (usedCube, use_actual)。use_actual 为 True 时必须与 metadata fag[1] 相等。
    """
    aic_num = aic_num or int(os.environ.get("FAG_CORE_NUM", "36"))
    layout = c.get("layout", "BSND")
    mask = c.get("mask_mode", 0) or 0
    if layout == "TND" or mask != 0:
        return aic_num, False
    b, n1, n2 = c["b"], c["n1"], c["n2"]
    s1, s2, d = c["s1"], c["s2"], c["d"]
    g = n1 // n2 if n2 else 1
    tmpl = _decide_template(c)
    s1o = _fag_outer(s1, _FAG_S1_INNER, _FAG_S1CV_RATIO)
    s2o = _fag_outer(s2, _FAG_S2_INNER, _FAG_S2CV_RATIO)
    fused = b * n2 * g if tmpl == 1 else b * n2 * g * s1o * s2o
    bo = _fag_dense_block_outer(fused, aic_num)
    used = max(1, min(bo, aic_num))
    return used, True


def _predict_tiling(c, cores=None):
    """复刻 host tiling 的 template / swizzle / tilingKey，只为打进报告里对账。

    tilingKey = tmpl | layout<<2 | mask<<3 | swizzle<<4 |
                d_align_bits<<5 | dv_align_bit<<7 | is_bn2_multiblk<<8
    d_align bits: 0=64, 1=128, 2=192。GS1S2 的 D<=64 保持 d_align=64（dv_align 仍 128）。
    """
    cores = cores or int(os.environ.get("FAG_CORE_NUM", "36"))
    b, n1, n2 = c["b"], c["n1"], c["n2"]
    s1, s2, d, dv = c["s1"], c["s2"], c["d"], c["dv"]
    g = n1 // n2 if n2 else 1
    tmpl = _decide_template(c)
    is_mb = _decide_bn2_multiblk(c) if tmpl == 1 else 0
    used_cube, _ = _predict_used_cube(c, cores)
    swizzle = 0
    if tmpl == 0 and cores > 0 and used_cube == cores:
        s1_outer = -(-s1 // CUBE_BASE)
        s2_outer = -(-s2 // CUBE_BASE)
        per_head = s1_outer * s2_outer
        fused = b * n2 * g
        total = fused * per_head
        if per_head > 0:
            per_core = -(-total // cores)
            live = len({(x * per_core) // per_head for x in range(cores)})
            per_head_bytes = (3 * s1 * d * 2 + s1 * d * 4 +
                              (2 * s2 * d * 2 + 2 * s2 * d * 4) // max(g, 1))
            s1_heavy = s1_outer >= 2 * s2_outer
            threshold = (5 * L2_CACHE_BYTES // 4) if s1_heavy else (2 * L2_CACHE_BYTES)
            if (s2_outer <= s1_outer and live * per_head_bytes > threshold and
                    fused * s2_outer >= 2 * cores):
                swizzle = 1
    d_align = 64 if d <= 64 else (128 if d <= 128 else 192)
    dv_align = 128 if dv <= 128 else 192
    if d_align == 64:
        dv_align = 128
    d_bits = 0 if d_align == 64 else (1 if d_align == 128 else 2)
    dv_bit = 1 if dv_align == 192 else 0
    key = tmpl | (swizzle << 4) | (d_bits << 5) | (dv_bit << 7) | (is_mb << 8)
    return tmpl, swizzle, key


def _build_case_data(c):
    """造两侧共用的输入，并在 fp32 下逐 head 算出 golden。

    支持 GQA（N1=N2*G）和不等 D（V/dO/y/dv 用 Dv）。逐 head 是为了内存。
    """
    b, n1, n2 = c["b"], c["n1"], c["n2"]
    s1, s2, d, dv = c["s1"], c["s2"], c["d"], c["dv"]
    g = n1 // n2
    dt = _DTYPE_MAP[c["dtype"]]
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(c["seed"])

    def rnd(heads, seq, dim):
        t = _trunc_normal(b, heads, seq, dim, device=DEVICE)
        return t.to(dt).float()

    qb = rnd(n1, s1, d)
    kb = rnd(n2, s2, d)
    vb = rnd(n2, s2, dv)
    ob = rnd(n1, s1, dv)
    yb = torch.empty(b, n1, s1, dv, device=DEVICE, dtype=torch.float32)
    gdq = torch.zeros_like(qb)
    gdk = torch.zeros_like(kb)
    gdv = torch.zeros_like(vb)
    smax = torch.empty(b, n1, s1, device=DEVICE, dtype=torch.float32)
    ssum = torch.empty(b, n1, s1, device=DEVICE, dtype=torch.float32)
    lse = torch.empty(b, n1, s1, device=DEVICE, dtype=torch.float32)

    for i in range(b):
        for j in range(n1):
            kv = j // g
            q_h, k_h, v_h, o_h = qb[i, j], kb[i, kv], vb[i, kv], ob[i, j]
            sm = torch.matmul(q_h, k_h.t()) * scale
            m = sm.max(dim=-1, keepdim=True).values
            e = torch.exp(sm - m)
            l = e.sum(dim=-1, keepdim=True)
            p = e / l
            y_h = torch.matmul(p, v_h).to(dt).float()
            dp = torch.matmul(o_h, v_h.t())
            sfmg = (o_h * y_h).sum(dim=-1, keepdim=True)
            ds = p * (dp - sfmg) * scale
            gdv[i, kv] += torch.matmul(p.t(), o_h)
            gdk[i, kv] += torch.matmul(ds.t(), q_h)
            gdq[i, j] = torch.matmul(ds, k_h)
            yb[i, j] = y_h
            smax[i, j] = m.squeeze(-1)
            ssum[i, j] = l.squeeze(-1)
            lse[i, j] = (m + torch.log(l)).squeeze(-1)
            del sm, e, l, p, dp, ds

    def lay(t, cast=None):
        t = t if c["layout"] == "BNSD" else t.transpose(1, 2)
        t = t.contiguous().to(DEVICE)
        return t.to(cast) if cast is not None else t

    torch.npu.synchronize()
    return {
        "q": lay(qb, dt), "k": lay(kb, dt), "v": lay(vb, dt),
        "dout": lay(ob, dt), "y": lay(yb, dt),
        "lse": lse.contiguous().to(DEVICE),
        "smax8": smax.unsqueeze(-1).expand(-1, -1, -1, 8).contiguous().to(DEVICE),
        "ssum8": ssum.unsqueeze(-1).expand(-1, -1, -1, 8).contiguous().to(DEVICE),
        "golden": (lay(gdq), lay(gdk), lay(gdv)),
        "scale": scale,
    }


def _bench_wall(fn, warmup=3, iters=10):
    if iters <= 0:
        return float("nan")
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def _rel_err(got, ref):
    """相对误差：max|got-ref| / max|ref|。

    fp16 输出的绝对误差随 |ref| 一起长，S 越大越没法用固定 atol 判，
    所以统一折算成对参考值幅度的相对量。
    S2=1 等退化形 golden 近 0（dQ/dK 数学上全 0），相对误差会被放大到 ~1，
    判定时要同时看绝对误差（见 _err_ok）。
    """
    g = got.detach().cpu().float()
    r = ref.detach().cpu().float()
    diff = (g - r).abs()
    ref_max = r.abs().max().item()
    denom = max(ref_max, 1e-6)
    return {"max": diff.max().item(), "mean": diff.mean().item(),
            "rel_max": diff.max().item() / denom,
            "rel_mean": diff.mean().item() / denom,
            "ref_max": ref_max}


def _err_ok(e, args):
    """rel <= tol，或 abs max <= atol（近零 golden 的逃生口）。"""
    return e["rel_max"] <= args.tol or e["max"] <= args.atol


def _fmt_err(e):
    return (f"rel={e['rel_max']:.2e} max={e['max']:.2e} mean={e['mean']:.2e}"
            f" |ref|={e.get('ref_max', 0):.2e}")


def _run_one_compare(case, args):
    """一条用例：pypto 二进制、AscendC 各跑一遍，两边都跟 fp32 golden 比。"""
    c = _norm_case(case)
    tmpl, swizzle, key = _predict_tiling(c)
    used_cube, use_actual = _predict_used_cube(c)
    shape = (f"B{c['b']} N{c['n1']} S1={c['s1']} S2={c['s2']} D{c['d']} "
             f"{c['dtype']} {c['layout']}")
    print(f"\n{'='*78}\n[compare] {c['name']}\n  {shape}  "
          f"template={tmpl}({'BN2' if tmpl else 'BN2GS1S2'}) "
          f"multiblk={_decide_bn2_multiblk(c) if tmpl == 1 else 0} "
          f"swizzle={swizzle} tilingKey={key} usedCube={used_cube}"
          f"{'' if use_actual else ' (full-chip fallback)'}\n{'='*78}", flush=True)

    reason = _unsupported_reason(c)
    if reason:
        print(f"  [SKIP] {reason}")
        return {"case": c, "status": "skip", "reason": reason}

    t0 = time.perf_counter()
    data = _build_case_data(c)
    print(f"  golden (fp32, 逐 head) 用时 {time.perf_counter() - t0:.1f}s",
          flush=True)
    q, k, v = data["q"], data["k"], data["v"]
    dout, y, lse = data["dout"], data["y"], data["lse"]
    scale = data["scale"]
    gdq, gdk, gdv = data["golden"]

    metadata = cann_ops_transformer.flash_attn_metadata(
        num_heads_q=c["n1"], num_heads_kv=c["n2"], head_dim=c["d"],
        batch_size=c["b"], max_seqlen_q=c["s1"], max_seqlen_kv=c["s2"],
        mask_mode=0, win_left=-1, win_right=-1,
        layout_q=c["layout"], layout_kv=c["layout"], layout_out=c["layout"],
        is_grad_enabled=True,
    )
    _check_fag_block_starts_ends(metadata)
    meta_info = dump_metadata(
        metadata, b=c["b"], n1=c["n1"], n2=c["n2"],
        s1=c["s1"], s2=c["s2"], template=tmpl)
    if c["n1"] != c["n2"] and tmpl != 0:
        print("  [FAIL] GQA 必须 template=0")
        return {"case": c, "status": "fail", "reason": "GQA tmpl!=0",
                "tmpl": tmpl, "swizzle": swizzle, "key": key}
    if meta_info and meta_info.get("mismatch"):
        return {"case": c, "status": "fail",
                "reason": f"splitAxis={meta_info.get('split_axis')} vs tmpl={tmpl}",
                "tmpl": tmpl, "swizzle": swizzle, "key": key}
    if use_actual and meta_info:
        bo = int(meta_info.get("block_outer") or 0)
        if bo != used_cube:
            print(f"  [FAIL] tiling usedCube={used_cube} != metadata fag[1] block_outer={bo}")
            return {"case": c, "status": "fail",
                    "reason": f"usedCube {used_cube} vs fag[1] {bo}",
                    "tmpl": tmpl, "swizzle": swizzle, "key": key}
        print(f"  [OK] tiling usedCube={used_cube} == metadata block_outer")

    def pypto_once():
        return cann_ops_transformer.flash_attn_grad(
            q, k, v, dout, y, lse,
            metadata=metadata, softmax_scale=scale,
            mask_mode=0, win_left=-1, win_right=-1,
            max_seqlen_q=c["s1"], max_seqlen_kv=c["s2"],
            layout_q=c["layout"], layout_kv=c["layout"], layout_out=c["layout"],
        )

    def asc_once():
        return torch_npu.npu_fusion_attention_grad(
            q, k, v, dout, c["n1"], c["layout"],
            pse=None, padding_mask=None, atten_mask=None,
            softmax_max=data["smax8"], softmax_sum=data["ssum8"], softmax_in=None,
            attention_in=y, scale_value=scale, keep_prob=1.0,
            pre_tockens=INT32_MAX, next_tockens=INT32_MAX,
            inner_precise=0, sparse_mode=0,
        )

    res = {"case": c, "tmpl": tmpl, "swizzle": swizzle, "key": key,
           "status": "pass", "no_asc": False}
    try:
        pypto_out = pypto_once()[:3]
        torch.npu.synchronize()
    except Exception as e:
        print(f"  [FAIL] pypto 二进制: {type(e).__name__}: {e}")
        res["status"] = "fail"
        res["reason"] = f"pypto {type(e).__name__}"
        return res
    asc_out = None
    try:
        asc_out = asc_once()[:3]
        torch.npu.synchronize()
    except Exception as e:
        print(f"  [WARN] AscendC 不可用 ({type(e).__name__}: {e})，改 vs fp32 golden（无 ASC）")
        res["no_asc"] = True

    ok = True
    worst_p, worst_a, worst_pa = 0.0, 0.0, 0.0
    if asc_out is None:
        for nm, po, g in zip(("dQ", "dK", "dV"), pypto_out, (gdq, gdk, gdv)):
            ep = _rel_err(po, g)
            worst_p = max(worst_p, ep["rel_max"])
            good = _err_ok(ep, args)
            ok = ok and good
            print(f"    {nm} {'OK ' if good else 'BAD'} "
                  f"pypto/golden {_fmt_err(ep)} | ASC=n/a")
    else:
        for nm, po, ao, g in zip(("dQ", "dK", "dV"), pypto_out, asc_out,
                                 (gdq, gdk, gdv)):
            ep, ea = _rel_err(po, g), _rel_err(ao, g)
            epa = _rel_err(po, ao)
            worst_p = max(worst_p, ep["rel_max"])
            worst_a = max(worst_a, ea["rel_max"])
            worst_pa = max(worst_pa, epa["rel_max"])
            good = _err_ok(ep, args)
            ok = ok and good
            print(f"    {nm} {'OK ' if good else 'BAD'} "
                  f"pypto/golden {_fmt_err(ep)} | ASC/golden {_fmt_err(ea)} | "
                  f"pypto/ASC rel={epa['rel_max']:.2e}")
            if nm == "dQ" and not good:
                g = po.detach().cpu().float()
                r = gdq.detach().cpu().float()
                diff = (g - r).abs()
                n_big = int((diff > 0.1).sum().item())
                n_huge = int((diff > 1.0).sum().item())
                flat = int(diff.reshape(-1).argmax().item())
                loc = []
                rem = flat
                for dim in reversed(diff.shape):
                    loc.append(rem % int(dim))
                    rem //= int(dim)
                loc = tuple(reversed(loc))
                print(f"      dQ debug n>|0.1|={n_big} n>|1|={n_huge} "
                      f"argmax{loc} got={g.reshape(-1)[flat].item():.4f} "
                      f"ref={r.reshape(-1)[flat].item():.4f} "
                      f"shape={tuple(int(x) for x in diff.shape)}")
                huge = diff > 1.0
                # BSND: (B, S1, N, D)
                s1_idx = [i for i, v in enumerate(huge.any(dim=0).any(dim=1).any(dim=1).tolist()) if v]
                n_idx = [i for i, v in enumerate(huge.any(dim=0).any(dim=0).any(dim=1).tolist()) if v]
                d_idx = [i for i, v in enumerate(huge.any(dim=0).any(dim=0).any(dim=0).tolist()) if v]
                print(f"      huge s1({len(s1_idx)})={s1_idx[:32]} "
                      f"n({len(n_idx)})={n_idx[:32]} d({len(d_idx)})={d_idx[:32]}")

    # 重跑同一份输入，看结果是否逐位可重复 —— 之前 UB 竞争就是这么暴露的
    drift = 0.0
    for _ in range(max(args.repeat - 1, 0)):
        again = pypto_once()[:3]
        torch.npu.synchronize()
        for t1, t2 in zip(pypto_out, again):
            drift = max(drift, (t1.float() - t2.float()).abs().max().item())
    if args.repeat > 1:
        print(f"    重复 {args.repeat} 次的最大漂移 = {drift:.3e}"
              f"{'' if drift == 0 else '   <-- 非确定！'}")
        if drift != 0:
            ok = False
    res["drift"] = drift

    t_pypto = float("nan")
    t_asc = float("nan")
    if args.warmup > 0 or args.iters > 0:
        t_pypto = _bench_wall(pypto_once, args.warmup, args.iters)
        if asc_out is not None:
            t_asc = _bench_wall(asc_once, args.warmup, args.iters)
            print(f"    perf  pypto={t_pypto:.1f}us  ASC={t_asc:.1f}us  "
                  f"pypto/ASC={t_pypto / t_asc if t_asc else float('nan'):.2f}x")
        else:
            print(f"    perf  pypto={t_pypto:.1f}us  ASC=n/a（无 ASC）")
    else:
        print("    perf  skipped (precision-only)")

    res.update({"status": "pass" if ok else "fail", "t_pypto": t_pypto,
                "t_asc": t_asc, "err_p": worst_p, "err_a": worst_a,
                "err_pa": worst_pa})
    return res


def _select_cases(cases, flt):
    """--filter 支持逗号分隔的下标、用例名、名字前缀。"""
    if not flt:
        return cases
    sel = []
    for tok in [t.strip() for t in flt.split(",") if t.strip()]:
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(cases):
                sel.append(cases[idx])
            continue
        names = [(c.get("testcase_name") or "") for c in cases]
        exact = [c for c, n in zip(cases, names) if n == tok]
        if exact:
            sel.extend(exact)
            continue
        hit = [c for c, n in zip(cases, names) if n.startswith(tok)]
        sel.extend(hit)
    return sel


_SHORT_DT = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}


def _print_summary(rows, args):
    print(f"\n{'='*112}\n  汇总  rel = max|diff| / max|golden|（或 abs<=atol={args.atol:.0e}）；"
          f"perf = wall-clock (warmup={args.warmup} iters={args.iters})\n"
          f"  100us 以下的用例里 host 下发开销占大头，两侧比值只能当参考，"
          f"要准就 --iters 200 或用 msprof 看 device 时间\n{'='*112}")
    print(f"  {'case':<22}{'shape':<32}{'tk':>4}{'':>6}"
          f"{'pypto/gold':>12}{'ASC/gold':>11}{'pypto us':>11}{'ASC us':>10}"
          f"{'p/ASC':>8}")
    for r in rows:
        c = r["case"]
        shape = (f"B{c['b']} N{c['n1']} {c['s1']}x{c['s2']} D{c['d']} "
                 f"{_SHORT_DT.get(c['dtype'], c['dtype'])} {c['layout']}")
        head = f"  {c['name'][:21]:<22}{shape:<32}"
        if r["status"] == "skip":
            print(f"{head}{'-':>4}{'SKIP':>6}   {r.get('reason', '')}")
            continue
        if "t_pypto" not in r:
            print(f"{head}{r.get('key', 0):>4}{'FAIL':>6}   {r.get('reason', '')}")
            continue
        ratio = r["t_pypto"] / r["t_asc"] if r["t_asc"] else float("nan")
        print(f"{head}{r['key']:>4}"
              f"{('OK' if r['status'] == 'pass' else 'BAD'):>6}"
              f"{r['err_p']:>12.2e}{r['err_a']:>11.2e}"
              f"{r['t_pypto']:>11.1f}{r['t_asc']:>10.1f}{ratio:>7.2f}x")


def _run_compare_mode(args):
    # golden 得是真 fp32：NPU matmul 若走 hf32（10 位尾数），参考值的精度就跟
    # 被测对象一个量级，比出来没有意义。
    try:
        torch.npu.matmul.allow_hf32 = False
    except Exception as e:  # 老版本 torch_npu 没这个开关
        print(f"[compare] 注意：关不掉 hf32 matmul ({e})，golden 精度存疑")

    if args.xlsx is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.xlsx = os.path.join(script_dir, COMPARE_XLSX)
    cases = _select_cases(_load_xlsx_cases(args.xlsx), args.filter)

    print(f"[compare] 用例: {args.xlsx} 里 enable=TRUE 的行")
    print(f"[compare] {len(cases)} 条, tol={args.tol} warmup={args.warmup} "
          f"iters={args.iters} repeat={args.repeat}")
    if args.list:
        for i, case in enumerate(cases):
            c = _norm_case(case)
            tmpl, swz, key = _predict_tiling(c)
            why = _unsupported_reason(c)
            print(f"  [{i:2d}] {c['name']:<28} B{c['b']} N{c['n1']} "
                  f"S1={c['s1']} S2={c['s2']} D{c['d']} {c['dtype']} "
                  f"{c['layout']}  tmpl={tmpl} swizzle={swz} tk={key}"
                  f"{'  SKIP: ' + why if why else ''}")
        return 0, 0

    rows = []
    for case in cases:
        try:
            rows.append(_run_one_compare(case, args))
        except Exception as e:
            print(f"  [FAIL] {case.get('testcase_name')}: {type(e).__name__}: {e}")
            rows.append({"case": _norm_case(case), "status": "fail",
                         "reason": f"{type(e).__name__}: {e}"})
        torch.npu.synchronize()

    _print_summary(rows, args)
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    n_fail = sum(1 for r in rows if r["status"] == "fail")
    n_skip = sum(1 for r in rows if r["status"] == "skip")
    if n_skip:
        print(f"\n  跳过 {n_skip} 条（见上表 SKIP 原因）")
    return n_pass, n_fail


# ============================================================
# perf / perf-suite: msprof Task Duration 纯 kernel 口径
# ============================================================

# op_summary 里按子串匹配；二进制 PTO 是 FlashAttnGrad，ASC 融合是
# FlashAttentionScoreGrad（偶发带后缀，所以用 contains）。
PYPTO_OP_SUBSTR = os.environ.get("FAG_PYPTO_OP", "FlashAttnGrad")
ASC_OP_SUBSTR = os.environ.get("FAG_ASC_OP", "FlashAttentionScoreGrad")


def _build_perf_inputs(c):
    """只要能喂进两侧算子的输入；不做 fp32 golden（会污染 msprof）。"""
    b, n1, n2 = c["b"], c["n1"], c["n2"]
    s1, s2, d, dv = c["s1"], c["s2"], c["d"], c["dv"]
    g = n1 // n2
    dt = _DTYPE_MAP[c["dtype"]]
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(c["seed"])

    def rnd(heads, seq, dim):
        t = _trunc_normal(b, heads, seq, dim, device=DEVICE)
        return t.to(dt).float()

    qb = rnd(n1, s1, d)
    kb = rnd(n2, s2, d)
    vb = rnd(n2, s2, dv)
    ob = rnd(n1, s1, dv)
    smax = torch.empty(b, n1, s1, device=DEVICE, dtype=torch.float32)
    ssum = torch.empty(b, n1, s1, device=DEVICE, dtype=torch.float32)
    lse = torch.empty(b, n1, s1, device=DEVICE, dtype=torch.float32)
    yb = torch.empty(b, n1, s1, dv, device=DEVICE, dtype=torch.float32)
    for i in range(b):
        for j in range(n1):
            kv = j // g
            q_h, k_h, v_h = qb[i, j], kb[i, kv], vb[i, kv]
            sm = torch.matmul(q_h, k_h.t()) * scale
            m = sm.max(dim=-1, keepdim=True).values
            e = torch.exp(sm - m)
            l = e.sum(dim=-1, keepdim=True)
            yb[i, j] = torch.matmul(e / l, v_h).to(dt).float()
            smax[i, j] = m.squeeze(-1)
            ssum[i, j] = l.squeeze(-1)
            lse[i, j] = (m + torch.log(l)).squeeze(-1)
            del sm, e, l

    def lay(t, cast=None):
        t = t if c["layout"] == "BNSD" else t.transpose(1, 2)
        t = t.contiguous().to(DEVICE)
        return t.to(cast) if cast is not None else t

    torch.npu.synchronize()
    return {
        "q": lay(qb, dt), "k": lay(kb, dt), "v": lay(vb, dt),
        "dout": lay(ob, dt), "y": lay(yb, dt),
        "lse": lse.contiguous().to(DEVICE),
        "smax8": smax.unsqueeze(-1).expand(-1, -1, -1, 8).contiguous().to(DEVICE),
        "ssum8": ssum.unsqueeze(-1).expand(-1, -1, -1, 8).contiguous().to(DEVICE),
        "scale": scale,
    }


def _run_one_perf(case, args):
    """被 msprof 包住的 worker：只发 kernel，不读 Task Duration。"""
    c = _norm_case(case)
    tmpl, swizzle, key = _predict_tiling(c)
    shape = (f"B{c['b']} N{c['n1']} S1={c['s1']} S2={c['s2']} D{c['d']} "
             f"{c['dtype']} {c['layout']}")
    print(f"\n{'='*78}\n[perf] {c['name']}\n  {shape}  "
          f"template={tmpl}({'BN2' if tmpl else 'BN2GS1S2'}) "
          f"swizzle={swizzle} tilingKey={key} side={args.side}\n{'='*78}",
          flush=True)

    reason = _unsupported_reason(c)
    if reason:
        print(f"  [SKIP] {reason}")
        return {"case": c, "status": "skip", "reason": reason}

    data = _build_perf_inputs(c)
    q, k, v = data["q"], data["k"], data["v"]
    dout, y, lse = data["dout"], data["y"], data["lse"]
    scale = data["scale"]

    metadata = cann_ops_transformer.flash_attn_metadata(
        num_heads_q=c["n1"], num_heads_kv=c["n2"], head_dim=c["d"],
        batch_size=c["b"], max_seqlen_q=c["s1"], max_seqlen_kv=c["s2"],
        mask_mode=0, win_left=-1, win_right=-1,
        layout_q=c["layout"], layout_kv=c["layout"], layout_out=c["layout"],
        is_grad_enabled=True,
    )
    _check_fag_block_starts_ends(metadata)

    def pypto_once():
        return cann_ops_transformer.flash_attn_grad(
            q, k, v, dout, y, lse,
            metadata=metadata, softmax_scale=scale,
            mask_mode=0, win_left=-1, win_right=-1,
            max_seqlen_q=c["s1"], max_seqlen_kv=c["s2"],
            layout_q=c["layout"], layout_kv=c["layout"], layout_out=c["layout"],
        )

    def asc_once():
        return torch_npu.npu_fusion_attention_grad(
            q, k, v, dout, c["n1"], c["layout"],
            pse=None, padding_mask=None, atten_mask=None,
            softmax_max=data["smax8"], softmax_sum=data["ssum8"], softmax_in=None,
            attention_in=y, scale_value=scale, keep_prob=1.0,
            pre_tockens=INT32_MAX, next_tockens=INT32_MAX,
            inner_precise=0, sparse_mode=0,
        )

    side = args.side
    # perf-suite 子进程会加 --no-smoke，避免 smoke launch 混进 Task Duration。
    if not args.no_smoke:
        try:
            if side in ("both", "pypto"):
                pypto_once()
                torch.npu.synchronize()
                print("  pypto smoke OK", flush=True)
            if side in ("both", "asc"):
                asc_once()
                torch.npu.synchronize()
                print("  AscendC smoke OK", flush=True)
        except Exception as e:
            print(f"  [FAIL] smoke: {type(e).__name__}: {e}")
            return {"case": c, "status": "fail",
                    "reason": f"smoke {type(e).__name__}"}

    for _ in range(max(args.warmup, 0)):
        if side in ("both", "pypto"):
            pypto_once()
        if side in ("both", "asc"):
            asc_once()
    torch.npu.synchronize()

    print(f"  [msprof-window] iters={args.iters} side={side}", flush=True)
    for _ in range(max(args.iters, 1)):
        if side in ("both", "pypto"):
            pypto_once()
        if side in ("both", "asc"):
            asc_once()
    torch.npu.synchronize()
    print("  [msprof-window] done", flush=True)
    return {"case": c, "status": "pass", "tmpl": tmpl, "swizzle": swizzle,
            "key": key}


def _run_perf_mode(args):
    if args.xlsx is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.xlsx = os.path.join(script_dir, COMPARE_XLSX)
    cases = _select_cases(_load_xlsx_cases(args.xlsx), args.filter)
    print(f"[perf] {len(cases)} case(s) from {args.xlsx}  "
          f"warmup={args.warmup} iters={args.iters} side={args.side}")
    if args.list:
        for i, case in enumerate(cases):
            c = _norm_case(case)
            tmpl, swz, key = _predict_tiling(c)
            why = _unsupported_reason(c)
            print(f"  [{i:2d}] {c['name']:<28} B{c['b']} N{c['n1']} "
                  f"S1={c['s1']} S2={c['s2']} D{c['d']}  tk={key}"
                  f"{'  SKIP: ' + why if why else ''}")
        return 0, 0

    n_pass = n_fail = 0
    for case in cases:
        try:
            r = _run_one_perf(case, args)
        except Exception as e:
            print(f"  [FAIL] {case.get('testcase_name')}: {type(e).__name__}: {e}")
            r = {"status": "fail"}
        if r.get("status") == "pass":
            n_pass += 1
        elif r.get("status") == "skip":
            pass
        else:
            n_fail += 1
        torch.npu.synchronize()
    return n_pass, n_fail


def _ensure_msprof_sqlite():
    """msprof 解析跑在系统 python；这台机器缺 libsqlite3 时要软链一下。"""
    sqlite_lib = os.environ.get(
        "SQLITE_LIB", "/root/.conda/envs/zyj/lib/libsqlite3.so.0")
    lib_dir = os.environ.get("MSPROF_LIB_DIR", "/tmp/msprof_libs")
    if not os.path.exists(sqlite_lib):
        return
    os.makedirs(lib_dir, exist_ok=True)
    for name in ("libsqlite3.so.0", "libsqlite3.so"):
        dst = os.path.join(lib_dir, name)
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(sqlite_lib, dst)
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if lib_dir not in paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir] + [p for p in paths if p])


def _find_msprof():
    cand = os.environ.get("MSPROF")
    if cand and os.path.isfile(cand):
        return cand
    which = shutil.which("msprof")
    if which:
        return which
    fallback = "/home/kjc/kjc_cann_0814/cann-9.2.0/tools/profiler/bin/msprof"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError("msprof not found; source CANN set_env.sh first")


def _find_prof_dir(prof_root):
    """找到 case 输出目录下唯一/最新的 PROF_*。"""
    if os.path.basename(prof_root).startswith("PROF_") and os.path.isdir(prof_root):
        return prof_root
    prof_dirs = [
        d for d in glob.glob(os.path.join(prof_root, "PROF_*"))
        if os.path.isdir(d)
    ]
    if not prof_dirs:
        prof_dirs = sorted(
            glob.glob(os.path.join(prof_root, "**", "PROF_*"), recursive=True)
        )
        prof_dirs = [d for d in prof_dirs if os.path.isdir(d)]
    if not prof_dirs:
        raise FileNotFoundError(f"no PROF_* under {prof_root}")
    return max(prof_dirs, key=os.path.getmtime)


def _dedupe_profiler_outputs(prof_dir):
    """msprof 结束会自动 export；我们不再二次 export。
    若目录里已有多份同名产物（历史二次 export），只留 mtime 最早的一份。
    """
    out = os.path.join(prof_dir, "mindstudio_profiler_output")
    if not os.path.isdir(out):
        return
    groups = {}
    for path in glob.glob(os.path.join(out, "*")):
        if not os.path.isfile(path):
            continue
        base = os.path.basename(path)
        # op_summary_20260821095641.csv -> op_summary
        stem = base.split("_")[0] if "_" in base else base
        # README.txt / 无时间戳的不动
        if stem in ("README",) or "." not in base:
            continue
        # 按「去掉时间戳后的前缀」分组：op_summary / api_statistic / ...
        prefix = base.rsplit("_", 1)[0] if base.rsplit("_", 1)[-1][:8].isdigit() \
            else None
        if prefix is None:
            # msprof_20260821....json
            if base.startswith("msprof_") and base.endswith(".json"):
                prefix = "msprof"
            else:
                continue
        groups.setdefault(prefix, []).append(path)
    for prefix, paths in groups.items():
        if len(paths) <= 1:
            continue
        keep = min(paths, key=os.path.getmtime)  # 保留第一次 export
        for p in paths:
            if p != keep:
                os.remove(p)


def _ensure_one_op_summary(prof_root):
    """保证有且仅有一份 op_summary；必要时才调用 msprof --export。"""
    _ensure_msprof_sqlite()
    target = _find_prof_dir(prof_root)
    files = glob.glob(os.path.join(target, "**", "op_summary*.csv"),
                      recursive=True)
    if not files:
        msprof = _find_msprof()
        subprocess.run([msprof, "--export=on", f"--output={target}"],
                       check=False, capture_output=True, text=True)
        files = glob.glob(os.path.join(target, "**", "op_summary*.csv"),
                          recursive=True)
    if not files:
        raise FileNotFoundError(
            f"msprof export produced no op_summary under {target}")
    _dedupe_profiler_outputs(target)
    files = glob.glob(os.path.join(target, "**", "op_summary*.csv"),
                      recursive=True)
    if len(files) != 1:
        # 去重后仍多份：留最早，删其余
        keep = min(files, key=os.path.getmtime)
        for p in files:
            if p != keep:
                os.remove(p)
        files = [keep]
    return files[0], target


def _mean_drop_first(vals):
    """iters==1 保留仅有的一次；iters>1 丢掉第一次再平均。返回 (mean, n_used)。"""
    if not vals:
        return None, 0
    if len(vals) == 1:
        return float(vals[0]), 1
    used = vals[1:]
    return sum(used) / len(used), len(used)


def _extract_kernel_samples(csv_path):
    """从唯一 op_summary 抽出 PTO / ASC 每次 Task Duration(us)。"""
    py_samples, asc_samples = [], []
    py_name = asc_name = None
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("Op Name") or "").strip()
            if not name:
                continue
            try:
                dur = float(row["Task Duration(us)"])
            except (KeyError, TypeError, ValueError):
                continue
            # ASC 全名里含 FlashAttentionScoreGrad；PTO 是精确 FlashAttnGrad
            if ASC_OP_SUBSTR in name:
                asc_samples.append(dur)
                asc_name = name
            elif name == PYPTO_OP_SUBSTR or (
                    PYPTO_OP_SUBSTR in name and ASC_OP_SUBSTR not in name):
                py_samples.append(dur)
                py_name = name
    py_us, py_n = _mean_drop_first(py_samples)
    asc_us, asc_n = _mean_drop_first(asc_samples)
    return {
        "pypto_name": py_name,
        "asc_name": asc_name,
        "pypto_samples": py_samples,
        "asc_samples": asc_samples,
        "pypto_us": py_us,
        "asc_us": asc_us,
        "pypto_count": py_n,
        "asc_count": asc_n,
        "pypto_raw_count": len(py_samples),
        "asc_raw_count": len(asc_samples),
    }


def _write_case_artifacts(case_out, c, csv_path, samples, iters):
    """把唯一 op_summary 拷到 case 根目录，并写 10 次明细 CSV。"""
    stable = os.path.join(case_out, "op_summary.csv")
    shutil.copy2(csv_path, stable)

    detail = os.path.join(case_out, f"kernel_{iters}runs.csv")
    with open(detail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "shape", "side", "run", "task_duration_us", "op_name",
        ])
        shape = (f"B{c['b']}_N{c['n1']}_S{c['s1']}x{c['s2']}_D{c['d']}")
        for i, us in enumerate(samples["pypto_samples"], 1):
            w.writerow([c["name"], shape, "pypto", i, f"{us:.6f}",
                        samples["pypto_name"] or ""])
        for i, us in enumerate(samples["asc_samples"], 1):
            w.writerow([c["name"], shape, "ascendc", i, f"{us:.6f}",
                        samples["asc_name"] or ""])

    # 期望次数提示（不强制 fail，方便你对照）
    py_n = samples.get("pypto_raw_count", samples["pypto_count"])
    asc_n = samples.get("asc_raw_count", samples["asc_count"])
    if py_n != iters or asc_n != iters:
        print(f"  [warn] 期望每侧 {iters} 次，实际 "
              f"pypto={py_n} asc={asc_n}")
    return stable, detail


def _parse_prof_dir(prof_dir, iters):
    csv_path, prof_path = _ensure_one_op_summary(prof_dir)
    samples = _extract_kernel_samples(csv_path)
    # 兼容旧打印：构造轻量 agg（只含目标两行）
    agg = {}
    if samples["pypto_name"]:
        tot = sum(samples["pypto_samples"])
        agg[samples["pypto_name"]] = {
            "count": samples["pypto_count"], "total": tot,
            "mean_us": samples["pypto_us"],
        }
    if samples["asc_name"]:
        tot = sum(samples["asc_samples"])
        agg[samples["asc_name"]] = {
            "count": samples["asc_count"], "total": tot,
            "mean_us": samples["asc_us"],
        }
    return {
        "csv": csv_path,
        "prof_dir": prof_path,
        "agg": agg,
        "samples": samples,
        "pypto_name": samples["pypto_name"],
        "pypto_us": samples["pypto_us"],
        "pypto_count": samples["pypto_count"],
        "asc_name": samples["asc_name"],
        "asc_us": samples["asc_us"],
        "asc_count": samples["asc_count"],
    }


def _write_all_cases_table(out_root, rows, iters):
    """总表：每个用例 × 每侧 × 每次运行一行。"""
    path = os.path.join(out_root, f"all_cases_{iters}runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "B", "N", "S1", "S2", "D", "layout", "dtype",
            "side", "run", "task_duration_us", "op_name", "mean_us",
        ])
        for r in rows:
            if r.get("status") != "pass":
                continue
            c = r["case"]
            samples = r["samples"]
            for side, key, name_key in (
                ("pypto", "pypto_samples", "pypto_name"),
                ("ascendc", "asc_samples", "asc_name"),
            ):
                vals = samples[key]
                mean_v, _ = _mean_drop_first(vals)
                mean = "" if mean_v is None else mean_v
                for i, us in enumerate(vals, 1):
                    w.writerow([
                        c["name"], c["b"], c["n1"], c["s1"], c["s2"], c["d"],
                        c["layout"], c["dtype"],
                        side, i, f"{us:.6f}", samples[name_key] or "",
                        f"{mean:.6f}" if mean != "" else "",
                    ])
    mean_path = os.path.join(out_root, "summary_mean.csv")
    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "B", "N", "S1", "S2", "D",
            "pypto_mean_us", "asc_mean_us", "pypto_div_asc",
            "pypto_n", "asc_n", "status",
        ])
        for r in rows:
            c = r["case"]
            if r.get("status") != "pass":
                w.writerow([
                    c["name"], c["b"], c["n1"], c["s1"], c["s2"], c["d"],
                    "", "", "", "", "", r.get("status"),
                ])
                continue
            py_us, asc_us = r["pypto_us"], r["asc_us"]
            ratio = (py_us / asc_us) if (py_us and asc_us) else ""
            w.writerow([
                c["name"], c["b"], c["n1"], c["s1"], c["s2"], c["d"],
                f"{py_us:.6f}" if py_us is not None else "",
                f"{asc_us:.6f}" if asc_us is not None else "",
                f"{ratio:.6f}" if ratio != "" else "",
                r["pypto_count"], r["asc_count"], "pass",
            ])
    return path, mean_path


def _run_perf_suite(args):
    """逐 case：msprof 包一层 python --mode perf，再解析 Task Duration。"""
    if args.xlsx is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.xlsx = os.path.join(script_dir, COMPARE_XLSX)
    cases = _select_cases(_load_xlsx_cases(args.xlsx), args.filter)
    print(f"[perf-suite] {len(cases)} case(s), side={args.side} "
          f"iters={args.iters} (no warmup; mean drops first if iters>1)")
    if args.host_warmup:
        print("[perf-suite] --host-warmup 已忽略"
              "（第一次本来就会慢，均值丢掉即可）")
    if args.list:
        return _run_perf_mode(args)

    msprof = _find_msprof()
    _ensure_msprof_sqlite()
    script = os.path.abspath(__file__)
    py = sys.executable
    out_root = args.prof_dir or os.environ.get(
        "FAG_MSPROF_DIR", "/home/kjc/kjc_tmp/fag_msprof_binary")
    os.makedirs(out_root, exist_ok=True)
    print(f"[perf-suite] msprof={msprof}")
    print(f"[perf-suite] output={out_root}")
    print(f"[perf-suite] 你要看的文件：")
    print(f"  {out_root}/all_cases_{args.iters}runs.csv   # 全部用例每次运行")
    print(f"  {out_root}/summary_mean.csv                 # 均值汇总")
    print(f"  {out_root}/<case>/op_summary.csv            # 每 case 唯一一份")
    print(f"  {out_root}/<case>/kernel_{args.iters}runs.csv  # 每 case 每次明细")

    rows = []
    n_pass = n_fail = 0
    for case in cases:
        c = _norm_case(case)
        why = _unsupported_reason(c)
        if why:
            print(f"\n[perf-suite] {c['name']}: SKIP {why}")
            rows.append({"case": c, "status": "skip", "reason": why})
            continue
        tag = c["name"].replace("/", "_")
        case_out = os.path.join(out_root, tag)
        shutil.rmtree(case_out, ignore_errors=True)
        os.makedirs(case_out, exist_ok=True)

        cmd = [
            msprof,
            f"--output={case_out}",
            "--ai-core=on",
            f"--aic-metrics={args.aic_metrics}",
            "--task-time=on",
            "--ascendcl=on",
            py, "-u", script,
            "--mode", "perf",
            "--xlsx", args.xlsx,
            "--filter", c["name"],
            "--side", args.side,
            "--warmup", "0",
            "--iters", str(args.iters),
            "--no-smoke",
        ]
        print(f"\n[perf-suite] msprof {c['name']} ...", flush=True)
        log_path = os.path.join(case_out, "msprof_run.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  text=True)
        if proc.returncode != 0:
            print(f"  [FAIL] msprof exit={proc.returncode}, see {log_path}")
            rows.append({"case": c, "status": "fail",
                         "reason": f"msprof exit {proc.returncode}"})
            n_fail += 1
            continue
        try:
            parsed = _parse_prof_dir(case_out, args.iters)
            _write_case_artifacts(
                case_out, c, parsed["csv"], parsed["samples"], args.iters)
        except Exception as e:
            print(f"  [FAIL] parse: {type(e).__name__}: {e}")
            rows.append({"case": c, "status": "fail",
                         "reason": f"parse {type(e).__name__}"})
            n_fail += 1
            continue

        py_us, asc_us = parsed["pypto_us"], parsed["asc_us"]
        if args.side in ("both", "pypto") and py_us is None:
            print("  [FAIL] no FlashAttnGrad Task Duration in op_summary")
            rows.append({"case": c, "status": "fail",
                         "reason": "no pypto Task Duration"})
            n_fail += 1
            continue
        ratio = (py_us / asc_us) if (py_us and asc_us) else float("nan")
        py_s = f"{py_us:.3f}" if py_us is not None else "NA"
        asc_s = f"{asc_us:.3f}" if asc_us is not None else "NA"
        ratio_s = f"{ratio:.2f}x" if (py_us and asc_us) else "NA"
        py_raw = parsed["samples"].get("pypto_raw_count", parsed["pypto_count"])
        asc_raw = parsed["samples"].get("asc_raw_count", parsed["asc_count"])
        print(f"  Task Duration mean: pypto={py_s}us "
              f"(n={parsed['pypto_count']}/{py_raw})  "
              f"ASC={asc_s}us (n={parsed['asc_count']}/{asc_raw})  "
              f"p/ASC={ratio_s}")
        print(f"  pypto samples: "
              f"{['%.2f' % x for x in parsed['samples']['pypto_samples']]}")
        print(f"  ASC   samples: "
              f"{['%.2f' % x for x in parsed['samples']['asc_samples']]}")
        rows.append({
            "case": c, "status": "pass",
            "pypto_us": py_us, "asc_us": asc_us,
            "pypto_name": parsed["pypto_name"], "asc_name": parsed["asc_name"],
            "pypto_count": parsed["pypto_count"],
            "asc_count": parsed["asc_count"],
            "samples": parsed["samples"],
            "csv": parsed["csv"],
        })
        n_pass += 1

    all_path, mean_path = _write_all_cases_table(out_root, rows, args.iters)

    print(f"\n{'='*100}")
    print(f"  msprof Task Duration 汇总  (iters={args.iters} side={args.side}; "
          f"iters>1 丢掉第一次)")
    print(f"{'='*100}")
    print(f"  {'case':<24}{'shape':<28}{'pypto(us)':>11}{'ASC(us)':>10}{'p/ASC':>8}")
    for r in rows:
        c = r["case"]
        shape = f"B{c['b']} N{c['n1']} {c['s1']}x{c['s2']} D{c['d']}"
        head = f"  {c['name'][:23]:<24}{shape:<28}"
        if r["status"] != "pass":
            print(f"{head}{r['status'].upper():>11}  {r.get('reason', '')}")
            continue
        py_us = r.get("pypto_us")
        asc_us = r.get("asc_us")
        ratio = (py_us / asc_us) if (py_us and asc_us) else float("nan")
        py_s = f"{py_us:.1f}" if py_us is not None else "NA"
        asc_s = f"{asc_us:.1f}" if asc_us is not None else "NA"
        print(f"{head}{py_s:>11}{asc_s:>10}{ratio:>7.2f}x")
    print(f"\n已写出：\n  {all_path}\n  {mean_path}")
    print("口径：op_summary 的 Task Duration(us)；iters>1 时均值丢掉第一次；"
          "每 case 只保留一份 op_summary.csv。")
    return n_pass, n_fail


def main():
    parser = argparse.ArgumentParser(
        description="FlashAttnGrad binary test (+ msprof perf copy)")
    parser.add_argument("--mode", type=str, default="fuzz",
                        choices=["fuzz", "reject", "compare", "perf", "perf-suite"],
                        help="perf: kernel worker under msprof; "
                             "perf-suite: orchestrate msprof per case")
    parser.add_argument("--target", type=str, default="all", choices=["all", "metadata", "fag"],
                        help="reject mode: target = all/metadata/fag")
    parser.add_argument("--xlsx", type=str, default=None,
                        help=f"用例表（fuzz: fag_test_case.xlsx; reject: "
                             f"fag_reject_case.xlsx; compare/perf: {COMPARE_XLSX}）")
    # ---- compare / perf 共用 ----
    parser.add_argument("--filter", type=str, default="",
                        help="逗号分隔的下标/用例名/名字前缀，如 bn2_,3")
    parser.add_argument("--list", action="store_true",
                        help="只列用例和预测的 tilingKey，不跑")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--iters", type=int, default=10,
                        help="正式迭代次数（默认 10；写入 all_cases_Nruns.csv）")
    parser.add_argument("--repeat", type=int, default=1,
                        help="compare: 同一份输入重复跑几次查非确定性（默认 1 不查）")
    parser.add_argument("--tol", type=float, default=2e-2,
                        help="compare: 判定阈值，max|diff|/max|golden|")
    parser.add_argument("--atol", type=float, default=5e-3,
                        help="compare: 近零 golden 的绝对误差阈值（S2=1 等）")
    # ---- perf / perf-suite ----
    parser.add_argument("--side", type=str, default="both",
                        choices=["both", "pypto", "asc"],
                        help="perf: 发哪一侧 kernel")
    parser.add_argument("--prof-dir", type=str, default=None,
                        help="perf-suite: msprof 输出根目录")
    parser.add_argument("--aic-metrics", type=str, default="PipeUtilization",
                        help="perf-suite: msprof --aic-metrics")
    parser.add_argument("--host-warmup", action="store_true",
                        help="已废弃：perf-suite 不再做窗口外预热，此开关忽略")
    parser.add_argument("--no-smoke", action="store_true",
                        help="perf: 跳过首次 smoke（perf-suite 子进程默认带上）")
    args = parser.parse_args()
    if not args.filter:
        args.filter = os.environ.get("CASE_FILTER", "")

    if args.mode == "fuzz":
        pass_count, fail_count = _run_fuzz_mode(args)
    elif args.mode == "reject":
        pass_count, fail_count = _run_reject_mode(args)
    elif args.mode == "compare":
        pass_count, fail_count = _run_compare_mode(args)
    elif args.mode == "perf":
        pass_count, fail_count = _run_perf_mode(args)
    else:
        pass_count, fail_count = _run_perf_suite(args)

    total = pass_count + fail_count
    print(f"\n{'='*60}")
    print(f"  Summary: {pass_count} passed, {fail_count} failed / {total} total")
    print(f"{'='*60}")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()

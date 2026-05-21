#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Flash Attention KT Simulation Mode Runner
# ======================================================================================================================

"""
sim_runner.py
=============
Orchestrates KT (Kernel Test) simulation mode for flash_attn.
Generates metadata and tiling data on CPU, then produces binary inputs
for the CCE simulator without requiring NPU hardware.

Usage:
    python sim_runner.py --case_files test_case --case_id all --output_dir ./sim_data
"""

import os
import sys
import struct
import argparse
import importlib
import itertools
import math
import numpy as np
from typing import Dict, Any, Optional, List

from flash_attn_metadata_sim import (
    generate_metadata_cpu, adjust_sinner_and_souter,
    AIC_CORE_NUM, AIV_CORE_NUM, FA_METADATA_SIZE, FD_METADATA_SIZE,
    HEADER_SIZE, SOUTER_32, SOUTER_64,
)

# ============ Tiling Data Generation (CPU) ============

# Layout enums matching flash_attn_tiling.cpp
LAYOUT_BSND = 0
LAYOUT_BNSD = 1
LAYOUT_TND = 2
LAYOUT_BNSD_BSND = 3

# KV Layout enums
KV_LAYOUT_DEFAULT = 0
KV_LAYOUT_PA_BBH = 1
KV_LAYOUT_PA_BNBD = 2
KV_LAYOUT_PA_NZ = 3

# mask_mode → sparseMode (in tiling struct)
MASK_MODE_NO_MASK = 0
MASK_MODE_DEFAULT = 1
MASK_MODE_CAUSAL = 2
MASK_MODE_ALL_MASK = 3
MASK_MODE_BAND = 4

MASK_MODE_INT_MAX = 2147483647

CV_RATIO = 2  # AIV/AIC = 72/36 = 2


def get_layout_enum(layout_str: str) -> int:
    """Convert layout string to enum for tiling key."""
    m = {"BSND": LAYOUT_BSND, "BNSD": LAYOUT_BNSD, "TND": LAYOUT_TND, "BSH": LAYOUT_BSND}
    return m.get(layout_str, LAYOUT_BNSD)


def get_kv_layout_enum(layout_str: str) -> int:
    m = {"PA_BBND": KV_LAYOUT_PA_BBH, "PA_BNBD": KV_LAYOUT_PA_BNBD, "PA_NZ": KV_LAYOUT_PA_NZ}
    return m.get(layout_str, KV_LAYOUT_DEFAULT)


class TilingDataGenerator:
    """Generates the FlashAttnTilingData struct (binary) for KT simulation."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> bytes:
        """Generate the full FlashAttnTilingData binary blob."""
        kw = self.kwargs
        b = kw.get("B", 1)
        n1 = kw.get("N1", 1)
        n2 = kw.get("N2", n1)
        s1 = kw.get("S1", 1)
        s2 = kw.get("S2", s1)
        d = kw.get("D", 128)
        d_v = kw.get("DV", d)
        d_rope = kw.get("DRope", 0)
        input_layout = kw.get("input_layout", "BNSD")
        layout_q = kw.get("layout_q", input_layout)
        layout_kv = kw.get("layout_kv", input_layout)
        layout_out = kw.get("layout_out", input_layout)
        sparse_mode = kw.get("sparse_mode", 0)
        pre_tokens = kw.get("pre_tokens", MASK_MODE_INT_MAX)
        next_tokens = kw.get("next_tokens", MASK_MODE_INT_MAX)
        lse_flag = kw.get("return_softmax_lse", 0)

        # Determine S_outer / S_inner
        q_layout_id = 1  # BNSD
        if layout_q in ("BSH", "BSND"):
            q_layout_id = 0
        elif layout_q == "TND":
            q_layout_id = 2

        # win_left/win_right for AdjustSinnerAndSouter
        win_left = pre_tokens
        win_right = next_tokens
        if sparse_mode in (0, 2):  # NO_MASK or CAUSAL
            win_left = MASK_MODE_INT_MAX
            win_right = MASK_MODE_INT_MAX

        s_outer, s_inner = adjust_sinner_and_souter(d_v, s1, s2, sparse_mode, win_left, win_right, q_layout_id)
        m_base_size = s_outer * CV_RATIO

        # Group size
        g_size = n1 // n2 if n2 > 0 else 1

        # cuSeqLens sizes
        cu_seq_q_size = b if layout_q == "TND" else 0
        cu_seq_kv_size = b if layout_kv == "TND" else 0
        seqused_q = kw.get("seqused_q")
        seqused_kv = kw.get("seqused_kv")
        seqused_q_size = len(seqused_q) if seqused_q else 0
        seqused_kv_size = len(seqused_kv) if seqused_kv else 0

        # Attn mask params
        has_atten_mask = 1 if sparse_mode == 3 else 0
        mask_s1_size = s1 if has_atten_mask else 0
        mask_s2_size = s2 if has_atten_mask else 0
        mask_batch = b if has_atten_mask else 0

        # win_left/win_right for tiling struct
        tiling_win_left = pre_tokens if sparse_mode not in (0, 2) else MASK_MODE_INT_MAX
        tiling_win_right = next_tokens if sparse_mode not in (0, 2) else MASK_MODE_INT_MAX

        # PA params
        pa_layout_type = 0
        block_size = kw.get("block_size", 0)
        max_block_per_batch = 0
        if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            pa_layout_type = {"PA_BBND": 1, "PA_BNBD": 0, "PA_NZ": 2}.get(layout_kv, 0)
            block_table = kw.get("block_table")
            if block_table is not None and hasattr(block_table, 'shape'):
                max_block_per_batch = block_table.shape[1] if len(block_table.shape) > 1 else 0

        # Workspace sizes (compute based on typical tiling logic)
        d_v_basic_block = 64
        if d_v <= 64:
            d_v_basic_block = 64
        elif d_v <= 128:
            d_v_basic_block = 128
        elif d_v <= 256:
            d_v_basic_block = 256
        elif d_v <= 512:
            d_v_basic_block = 512

        core_num = AIC_CORE_NUM * CV_RATIO  # 72
        fa_tmp_atten_gm_size = core_num * 2 * m_base_size * d_v
        fa_tmp_lse_gm_size = core_num * 2 * m_base_size * 8

        # ---- Build baseTiling binary (FlashAttnNoQuantTilingArch35) ----
        # FlashAttnBaseParams: 21 fields
        t1_size = s1 * n1 if layout_q == "TND" else s1
        t2_size = s2 * n2 if layout_kv == "TND" else s2

        base_params = struct.pack("<IIIIIIIIIIIIIII",
            b,                  # bSize
            t1_size,            # t1Size
            t2_size,            # t2Size
            n2,                 # n2Size
            g_size,             # gSize
            s1,                 # s1Size
            s2,                 # s2Size
            d,                  # dSize
            d_v,                # dSizeV
            d_rope,             # dSizeRope
            cu_seq_q_size,      # cuSeqLensQSize
            cu_seq_kv_size,     # cuSeqLensKVSize
            seqused_q_size,     # seqUsedQSize
            seqused_kv_size,    # seqUsedKvSize
            0,                  # placeholder for scaleValue (float encoded next)
        )
        # scaleValue as float
        scale = kw.get("scale", 1.0 / math.sqrt(d))
        base_params = base_params[:-4] + struct.pack("<f", scale)
        # Remaining 6 bytes: iscuSeqLengthsNull(u8), iscuSeqLengthsKVNull(u8), isKvContinuous(u8), isSoftMaxLseEnable(u8), coreNum(u32), outputLayout(u32)
        is_cu_null = 1 if cu_seq_q_size == 0 else 0
        is_cu_kv_null = 1 if cu_seq_kv_size == 0 else 0
        base_params += struct.pack("<BBBBII",
            is_cu_null,         # iscuSeqLengthsNull
            is_cu_kv_null,      # iscuSeqLengthsKVNull
            1,                  # isKvContinuous
            lse_flag,           # isSoftMaxLseEnable
            core_num,           # coreNum
            get_layout_enum(layout_out),  # outputLayout
        )

        # FlashAttnAttenMaskParams: sparseMode(u8) + pad(3B) + winLefts(i32) + winRights(i32) + attenMaskBatch(u32) + attenMaskS1Size(u32) + attenMaskS2Size(u32) + isExistRowInvalid(u8) + pad(3B)
        mask_params = struct.pack("<BxxxiiIIIBxxx",
            sparse_mode,        # sparseMode
            tiling_win_left,    # winLefts
            tiling_win_right,   # winRights
            mask_batch,         # attenMaskBatch
            mask_s1_size,       # attenMaskS1Size
            mask_s2_size,       # attenMaskS2Size
            0,                  # isExistRowInvalid
        )

        # FlashAttnPageAttentionParams: paLayoutType(u8) + pad(3B) + blockSize(u32) + maxBlockNumPerBatch(u32)
        pa_params = struct.pack("<BxxxII",
            pa_layout_type,
            block_size,
            max_block_per_batch,
        )

        # FlashAttnWorkspaceParams: accumOutSize(u32) + logSumExpSize(u32)
        ws_params = struct.pack("<II",
            fa_tmp_atten_gm_size,
            fa_tmp_lse_gm_size,
        )

        # FlashAttnS1OuterSplitCoreParams: enableS1OutSplit(bool → u8) + pad(7B) + totalSize(u64)
        s1_split_params = struct.pack("<BxxxxxxxQ", 0, 0)

        base_tiling = base_params + mask_params + pa_params + ws_params + s1_split_params

        return base_tiling

    def generate_metadata(self) -> np.ndarray:
        """Generate AICPU metadata buffer using CPU simulator."""
        kw = self.kwargs
        input_layout = kw.get("input_layout", "BNSD")
        layout_q = kw.get("layout_q", input_layout)
        layout_kv = kw.get("layout_kv", input_layout)
        layout_out = kw.get("layout_out", input_layout)
        n1 = kw.get("N1", 1)
        n2 = kw.get("N2", n1)
        b = kw.get("B", 1)
        d = kw.get("D", 128)
        s1 = kw.get("S1", 1)
        s2 = kw.get("S2", s1)
        sparse_mode = kw.get("sparse_mode", 0)
        pre_tokens = kw.get("pre_tokens", MASK_MODE_INT_MAX)
        next_tokens = kw.get("next_tokens", MASK_MODE_INT_MAX)
        cu_q = kw.get("cu_seqlens_q")
        cu_kv = kw.get("cu_seqlens_kv")
        seqused_q = kw.get("seqused_q")
        seqused_kv = kw.get("seqused_kv")

        if sparse_mode in (0, 3):
            pre_tokens = -1
            next_tokens = -1

        # Actual batch_size
        if isinstance(cu_q, list):
            batch_size = len(cu_q) - 1
        elif isinstance(seqused_q, list):
            batch_size = len(seqused_q)
        else:
            batch_size = b

        return generate_metadata_cpu(
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            num_heads_q=n1,
            num_heads_kv=n2,
            head_dim=d,
            batch_size=batch_size,
            max_seqlen_q=s1,
            max_seqlen_kv=s2,
            mask_mode=sparse_mode,
            win_left=pre_tokens,
            win_right=next_tokens,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )


def print_sim_metadata(metadata: np.ndarray):
    """Print metadata in human-readable format (same as npu_impl.print_metadata)."""
    FA_FIELDS = ["BN2_START", "M_START", "S2_START", "BN2_END", "M_END", "S2_END", "FIRST_FD_WS_IDX"]
    FD_FIELDS = ["BN2_IDX", "M_IDX", "WS_IDX", "WS_NUM", "M_START", "M_NUM"]

    section_num = int(metadata[0])
    is_fd = int(metadata[1])
    m_base_size = int(metadata[2])
    s2_base_size = int(metadata[3])

    print(f"\n{'='*78}")
    print(f"  [Metadata Header]")
    print(f"  sectionNum  = {section_num}")
    print(f"  isFD        = {is_fd}")
    print(f"  mBaseSize   = {m_base_size}")
    print(f"  s2BaseSize  = {s2_base_size}")

    fa_base = HEADER_SIZE
    fd_base = HEADER_SIZE + AIC_CORE_NUM * FA_METADATA_SIZE

    print(f"\n{'-'*78}")
    print(f"  FA Metadata — AIC cores (active only)")
    print(f"{'-'*78}")
    cw = 12
    hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in FA_FIELDS)
    print(hdr)

    for core in range(AIC_CORE_NUM):
        off = fa_base + core * FA_METADATA_SIZE
        vals = [int(metadata[off + i]) for i in range(len(FA_FIELDS))]
        if all(v == 0 for v in vals) and core > 0:
            continue
        print(f"  AIC{core:02d}  " + "".join(f"{v:>{cw}}" for v in vals))

    print(f"\n{'-'*78}")
    print(f"  FD Metadata — active AIV cores only (M_NUM > 0)")
    print(f"{'-'*78}")
    hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in FD_FIELDS)
    print(hdr)
    active = 0
    for core in range(AIV_CORE_NUM):
        off = fd_base + core * FD_METADATA_SIZE
        vals = [int(metadata[off + i]) for i in range(len(FD_FIELDS))]
        if vals[5] == 0:  # FD_M_NUM
            continue
        print(f"  AIV{core:02d} " + "".join(f"{v:>{cw}}" for v in vals))
        active += 1
    if active == 0:
        print("  (no active FD cores)")
    print(f"{'='*78}\n")


def save_sim_data(output_dir: str, case_name: str, tiling_bin: bytes, metadata: np.ndarray, kwargs: dict):
    """Save simulation data files for KT runner."""
    case_dir = os.path.join(output_dir, case_name.replace("/", "_"))
    os.makedirs(case_dir, exist_ok=True)

    # 1. Save tiling data binary
    tiling_path = os.path.join(case_dir, "tiling_data.bin")
    with open(tiling_path, "wb") as f:
        f.write(tiling_bin)

    # 2. Save metadata binary (uint32 little-endian)
    meta_path = os.path.join(case_dir, "metadata.bin")
    metadata.tofile(meta_path)

    # 3. Save metadata as readable text
    meta_txt_path = os.path.join(case_dir, "metadata.txt")
    import io
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    print_sim_metadata(metadata)
    sys.stdout = old_stdout
    with open(meta_txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    # 4. Save case parameters as JSON
    import json
    params_path = os.path.join(case_dir, "params.json")
    # Serialize only JSON-safe values
    json_safe = {}
    for k, v in kwargs.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            json_safe[k] = v
        elif isinstance(v, list):
            json_safe[k] = v
        elif hasattr(v, 'tolist'):  # tensor/ndarray
            json_safe[k] = v.tolist() if v.numel() < 100 else f"<tensor shape={tuple(v.shape)}>"
        else:
            json_safe[k] = str(v)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)

    print(f"  [SIM] Saved: {case_dir}/")
    print(f"         tiling_data.bin ({len(tiling_bin)} bytes)")
    print(f"         metadata.bin ({metadata.nbytes} bytes)")
    print(f"         metadata.txt")
    print(f"         params.json")
    return case_dir


# ============ Main: Simulation Mode Entry ============
def run_sim_mode(case_name: str, kwargs: dict, output_dir: str = "./sim_data", verbose: bool = True):
    """Run simulation mode for a single test case.
    
    Generates metadata and tiling data on CPU without NPU hardware.
    Returns (tiling_bytes, metadata_array).
    """
    gen = TilingDataGenerator(**kwargs)
    tiling_bin = gen.generate()
    metadata = gen.generate_metadata()

    if verbose:
        print(f"\n  [SIM] Case: {case_name}")
        print(f"  [SIM] mBaseSize={int(metadata[2])}, s2BaseSize={int(metadata[3])}")
        print_sim_metadata(metadata)

    save_sim_data(output_dir, case_name, tiling_bin, metadata, kwargs)
    return tiling_bin, metadata


# ============ CLI Entry ============
_DEFAULT_CASE_FILES = ["test_case", "test_case_fia_STC", "test_case_functional_single",
                        "test_case_redline_fa_TND", "test_case_redline_fa", "test_case_redline_fia"]


def load_case_modules(module_names):
    all_cases = {}
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            print(f"[WARN] Cannot import module: {mod_name}, skipping.")
            continue
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


def normalize_case(raw):
    """Minimal normalization for sim mode (doesn't need full tensor gen)."""
    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    layout_kv = c.get("layout_kv", layout_q)
    c["input_layout"] = layout_q
    c.setdefault("layout_kv", layout_kv)
    c.setdefault("layout_out", c.get("layout_out", layout_q))
    if "mask_mode" in c and "sparse_mode" not in c:
        c["sparse_mode"] = c.pop("mask_mode")
    c.setdefault("N2", c.get("N1"))
    c.setdefault("S2", c.get("S1"))
    c.setdefault("DV", c.get("D"))
    c.setdefault("DRope", 0)
    c.setdefault("pre_tokens", 2147483647)
    c.setdefault("next_tokens", 2147483647)
    if "win_left" in c and c["win_left"] != -1:
        c.setdefault("pre_tokens", c["win_left"])
    if "win_right" in c and c["win_right"] != -1:
        c.setdefault("next_tokens", c["win_right"])
    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)
    if layout_q == "TND":
        c.setdefault("seqused_q", [c["cu_seqlens_q"][i+1] - c["cu_seqlens_q"][i]
                                    for i in range(len(c["cu_seqlens_q"]) - 1)])
        if layout_kv not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            c.setdefault("cu_seqlens_kv", list(c["cu_seqlens_q"]))
            c.setdefault("seqused_kv", [c["cu_seqlens_kv"][i+1] - c["cu_seqlens_kv"][i]
                                         for i in range(len(c["cu_seqlens_kv"]) - 1)])
        c["B"] = 1
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttn KT Simulation Mode (仿真模式)")
    parser.add_argument("--case_files", type=str,
                        default=",".join(_DEFAULT_CASE_FILES),
                        help="Comma-separated test case module names (without .py)")
    parser.add_argument("--case_id", type=str, default="all",
                        help="Case name(s), comma-separated. 'all' for all cases.")
    parser.add_argument("--output_dir", type=str, default="./sim_data",
                        help="Output directory for simulation data (default: ./sim_data)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print metadata details")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress metadata printing")
    args = parser.parse_args()

    module_names = [x.strip() for x in args.case_files.split(",")]
    all_cases = load_case_modules(module_names)

    if not all_cases:
        print("[ERROR] No test cases loaded.")
        sys.exit(1)

    # Resolve case IDs
    if args.case_id == "all":
        run_cases = list(all_cases.keys())
    else:
        requested = [x.strip() for x in args.case_id.split(",")]
        run_cases = []
        for req in requested:
            matched = [k for k in all_cases.keys() if req in k]
            if matched:
                run_cases.extend(matched)
            else:
                print(f"[WARN] No case matching '{req}'")
        run_cases = list(dict.fromkeys(run_cases))  # deduplicate preserving order

    print(f"[SIM] Running simulation mode for {len(run_cases)} cases")
    print(f"[SIM] Output directory: {os.path.abspath(args.output_dir)}")
    print(f"{'='*78}")

    success_count = 0
    fail_count = 0
    for name in run_cases:
        config = all_cases[name]
        try:
            kwargs = normalize_case(config)
            run_sim_mode(name, kwargs, output_dir=args.output_dir, verbose=not args.quiet)
            success_count += 1
        except Exception as e:
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'='*78}")
    print(f"[SIM] Complete: {success_count} succeeded, {fail_count} failed, {len(run_cases)} total")
    sys.exit(0 if fail_count == 0 else 1)

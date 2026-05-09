#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Generate the FIA performance sweep grid (cases.json).

Phase 1: BNSD/BSND no-quant paged-KV.
Phase 2: + BSH / BNSD_BSND, MLA (decode + 2 prefill modes), antiquant
         (per-channel/per-token/per-token-per-head), full_quant (per-tensor),
         contiguous KV, mask family (5 sub-tags + ALiBi pse).

tensor_list KV is intentionally out of scope (Python API key/value are Tensor,
not TensorList).

Usage:
    python3 profile_grid.py --preset <name> -o cases.json
"""

import argparse
import itertools
import json
import logging
import os
import sys

# Reuse pytest's validation helper without modifying it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTEST_DIR = os.path.normpath(os.path.join(_HERE, "..", "pytest"))
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402  (after sys.path manipulation)
import check_valid_param  # noqa: E402
import scenarios  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("profile_grid")


DTYPE_STR = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


# ============================================================================
# Phase 1 presets (existing) — preserved for backward compatibility.
# ============================================================================

PHASE1_BASE_AXES = {
    "scenario": ["no_quant"],
    "cache_layout": ["BBH"],
    "in_layout": ["BNSD", "BSND"],
    "dtype": [torch.float16, torch.bfloat16],
    "head_dim": [128],
}


PROFILE_GRID = {
    # Smoke: harness self-check
    "smoke": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [1, 128],
        "kv_head_num": [4],
        "q_head_num": [8],
        "batch_size": [2],
    },
    "decode": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128, 256],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128, 192],
        "kv_seq": [1024, 4096, 16384, 32768],
        "q_seq": [1],
        "kv_head_num": [1, 2, 4, 8],
        "q_head_num": [8, 16, 32],
        "batch_size": [1, 4, 16],
    },
    "prefill": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096, 16384],
        "q_seq": [64, 128, 1024, 4096],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 4],
    },

    # ------------------------------------------------------------------
    # Phase 2 presets
    # ------------------------------------------------------------------

    # BSH / BNSD_BSND layouts (no_quant scenario, just changing in_layout)
    "bsh_decode": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BSH"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096, 16384],
        "q_seq": [1],
        "kv_head_num": [1, 4, 8],
        "q_head_num": [8, 32],
        "batch_size": [1, 4],
    },
    "bsh_prefill": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BSH"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 4],
    },
    "bnsd_bsnd_prefill": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD_BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 4],
    },

    # MLA — three sub-modes
    "mla_decode": {
        "scenario": ["mla_decode"],
        "cache_layout": ["BNBD"],
        "block_size": [128],
        "in_layout": ["BNSD"],
        "dtype": [torch.bfloat16, torch.float16],
        "head_dim": [512],
        "kv_seq": [1024, 4096, 16384],
        "q_seq": [1],
        "kv_head_num": [1],
        "q_head_num": [8, 16, 32, 64, 128],
        "batch_size": [1, 4],
    },
    "mla_prefill_split": {
        "scenario": ["mla_prefill_split"],
        "cache_layout": ["BNBD"],
        "block_size": [128],
        "in_layout": ["BNSD"],
        "dtype": [torch.bfloat16, torch.float16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [8, 16],
        "batch_size": [1, 2],
    },
    "mla_prefill_combine": {
        "scenario": ["mla_prefill_combine"],
        "cache_layout": ["BNBD"],
        "block_size": [128],
        "in_layout": ["BNSD"],
        "dtype": [torch.bfloat16, torch.float16],
        "head_dim": [192],
        "kv_seq": [4096],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [8, 16],
        "batch_size": [1, 2],
    },

    # Antiquant
    "antiquant_perchannel": {
        "scenario": ["antiquant"],
        "antiquant_mode": [0],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096, 16384],
        "q_seq": [1, 4],
        "kv_head_num": [4, 8],
        "q_head_num": [8, 16],
        "batch_size": [1, 4],
    },
    "antiquant_pertoken": {
        "scenario": ["antiquant"],
        "antiquant_mode": [1],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.bfloat16],  # mode 1 + Q_S>=2 typically requires BF16
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [4, 16],
        "kv_head_num": [4, 8],
        "q_head_num": [8, 16],
        "batch_size": [1, 4],
    },
    "antiquant_pertoken_perhead": {
        "scenario": ["antiquant"],
        "antiquant_mode": [3],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD"],
        "dtype": [torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [4, 16],
        "kv_head_num": [4, 8],
        "q_head_num": [8, 16],
        "batch_size": [1, 2],
    },

    # Full quant (per-tensor)
    "full_quant_per_tensor": {
        "scenario": ["full_quant"],
        "quant_scale2_mode": ["per_tensor", "per_channel"],
        "out_dtype": ["float16", "bfloat16"],
        "in_layout": ["BNSD", "BSND", "BSH"],
        "dtype": ["int8"],  # input dtype is INT8 for full quant
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 4],
    },

    # Contiguous KV (no block_table)
    "contiguous_decode": {
        "scenario": ["contiguous"],
        "in_layout": ["BNSD", "BSND", "BSH"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [1024, 4096],
        "q_seq": [1],
        "kv_head_num": [4, 8],
        "q_head_num": [8, 32],
        "batch_size": [1, 4],
    },
    "contiguous_prefill": {
        "scenario": ["contiguous"],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },

    # Mask sub-tags
    "mask_no": {
        "scenario": ["mask"],
        "mask_subtype": ["mask_no"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [4096],
        "q_seq": [128],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },
    "mask_full": {
        "scenario": ["mask"],
        "mask_subtype": ["mask_full"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [128, 1024],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },
    "mask_leftup": {
        "scenario": ["mask"],
        "mask_subtype": ["mask_leftup"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [128, 1024],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },
    "mask_rightdown": {
        "scenario": ["mask"],
        "mask_subtype": ["mask_rightdown"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [128, 1024],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },
    "mask_band": {
        "scenario": ["mask"],
        "mask_subtype": ["mask_band"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [256, 1024],
        "q_seq": [256, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },
    "mask_alibi_pse": {
        "scenario": ["mask"],
        "mask_subtype": ["mask_pse_alibi"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128],
        "kv_seq": [128, 1024],
        "q_seq": [128, 1024],
        "kv_head_num": [4, 8],
        "q_head_num": [16, 32],
        "batch_size": [1, 2],
    },

    # 短 KV decode：扩展更小 kv_seq 与 batch=1，便于检验 SCALAR BOUND 触发路径。
    "small_decode": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128, 192],
        "kv_seq": [256, 512, 1024],
        "q_seq": [1],
        "kv_head_num": [1, 2, 4],
        "q_head_num": [8, 16, 32, 64],
        "batch_size": [1, 4],
    },

    # 极小尺度 + 极端 head 配置，纯压测 scalar 路径。
    "scalar_stress": {
        "scenario": ["no_quant"],
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD"],
        "dtype": [torch.float16, torch.bfloat16],
        "head_dim": [128, 192],
        "kv_seq": [128, 256],
        "q_seq": [1],
        "kv_head_num": [1, 2],
        "q_head_num": [32, 64, 128],
        "batch_size": [1],
    },
}


# Composite presets — unions of named groups.
PHASE2_GROUP_NAMES = [
    "bsh_decode", "bsh_prefill", "bnsd_bsnd_prefill",
    "mla_decode", "mla_prefill_split", "mla_prefill_combine",
    "antiquant_perchannel", "antiquant_pertoken", "antiquant_pertoken_perhead",
    "full_quant_per_tensor",
    "contiguous_decode", "contiguous_prefill",
    "mask_no", "mask_full", "mask_leftup", "mask_rightdown", "mask_band", "mask_alibi_pse",
]

ALL_GROUP_NAMES = ["smoke", "decode", "prefill"] + PHASE2_GROUP_NAMES


COMPOSITE_PRESETS = {
    "phase2_full": PHASE2_GROUP_NAMES,
    "all": ALL_GROUP_NAMES,
}


MAX_CASES_PER_PRESET = {
    "smoke": 64,
    "decode": 400,
    "prefill": 200,
    "phase2_full": 1500,
    "all": 2000,
    "small_decode": 200,
    "scalar_stress": 50,
}


def expand_preset(preset_name):
    if preset_name in COMPOSITE_PRESETS:
        merged = []
        for sub in COMPOSITE_PRESETS[preset_name]:
            merged.extend(_expand_one(sub, PROFILE_GRID[sub]))
        return merged
    if preset_name not in PROFILE_GRID:
        raise SystemExit(f"unknown preset: {preset_name}")
    return _expand_one(preset_name, PROFILE_GRID[preset_name])


def _expand_one(tag, axes):
    keys = list(axes.keys())
    out = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        params = dict(zip(keys, combo))
        params["tag"] = tag
        # Default cache_layout for scenarios that don't set it
        params.setdefault("cache_layout", "BBH")
        out.append(params)
    return out


def _normalize_dtype(v):
    if isinstance(v, str):
        return v
    return DTYPE_STR.get(v, str(v))


def serialize(case_id, params):
    out = {
        "case_id": case_id,
        "tag": params["tag"],
        "scenario": params.get("scenario", "no_quant"),
        "batch_size": params["batch_size"],
        "q_head_num": params["q_head_num"],
        "kv_head_num": params["kv_head_num"],
        "q_seq": params["q_seq"],
        "kv_seq": params["kv_seq"],
        "head_dim": params["head_dim"],
        "dtype": _normalize_dtype(params["dtype"]),
        "in_layout": params["in_layout"],
        "block_size": params.get("block_size", 128),
        "cache_layout": params.get("cache_layout", "BBH"),
    }
    # Optional sub-axis fields.
    for opt in ("antiquant_mode", "mask_subtype", "out_dtype",
                "quant_scale2_mode", "pre_tokens", "next_tokens"):
        if opt in params:
            out[opt] = params[opt]
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate FIA perf sweep cases.json.")
    all_presets = sorted(set(list(PROFILE_GRID.keys()) + list(COMPOSITE_PRESETS.keys())))
    parser.add_argument("--preset", required=True, choices=all_presets)
    parser.add_argument("-o", "--output", required=True, help="Output cases.json path ('-' for stdout)")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Override per-preset cap; useful for ad-hoc sweeps.")
    args = parser.parse_args()

    raw = expand_preset(args.preset)
    logger.info("Preset '%s': raw Cartesian = %d combinations", args.preset, len(raw))

    valid = []
    skipped_legacy = 0
    skipped_scenario = 0
    # Legacy validator only knows BSH/BSND/BNSD; skip it for newer layouts.
    LEGACY_LAYOUTS = {"BSH", "BSND", "BNSD"}
    for params in raw:
        # (i) Phase 1 legacy validator (head/seq/dtype basics) — only for legacy layouts.
        if params["in_layout"] in LEGACY_LAYOUTS:
            if isinstance(params["dtype"], str):
                dtype_for_legacy = {
                    "float16": torch.float16, "bfloat16": torch.bfloat16, "int8": torch.int8,
                }.get(params["dtype"])
            else:
                dtype_for_legacy = params["dtype"]
            cfg = (params["batch_size"], params["q_head_num"], params["kv_head_num"],
                   params["q_seq"], params["kv_seq"], params["head_dim"],
                   dtype_for_legacy, params["in_layout"])
            try:
                check_valid_param.validate_config(cfg)
            except ValueError:
                skipped_legacy += 1
                continue

        # (ii) scenario-specific validator
        # Build a serialized-style dict (with str dtype) for validate.
        serialized = serialize(0, params)
        try:
            scenarios.validate(serialized)
        except ValueError:
            skipped_scenario += 1
            continue

        valid.append(params)

    cap = args.max_cases if args.max_cases is not None else MAX_CASES_PER_PRESET.get(args.preset, len(valid))
    if len(valid) > cap:
        logger.warning("Capping %d valid cases to %d (preset cap)", len(valid), cap)
        valid = valid[:cap]

    cases = [serialize(i, p) for i, p in enumerate(valid)]
    logger.info("Valid: %d  Skipped(legacy): %d  Skipped(scenario): %d  Emitting: %d",
                len(valid), skipped_legacy, skipped_scenario, len(cases))

    payload = json.dumps(cases, indent=2, ensure_ascii=False)
    if args.output == "-":
        sys.stdout.write(payload + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        logger.info("Wrote %s (%d cases)", args.output, len(cases))


if __name__ == "__main__":
    main()

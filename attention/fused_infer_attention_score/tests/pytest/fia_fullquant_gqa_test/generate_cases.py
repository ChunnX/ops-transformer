#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate generalized test cases for FIA FullQuant GQA operator."""

import json
import os
import random

random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")

# ==============================================================================
# Normal Case Construction
# ==============================================================================
B_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
GQA_RATIOS = [1, 2, 3, 4, 5, 8, 16, 32, 64]
N_KV_VALUES = [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 32]
N_Q_MAX = 256
SPARSE_MODES = [0, 3]
ENABLE_LSE_VALUES = [True, False]

# Sequence length ranges: (label, min, max, weight)
SEQ_Q_RANGES = [
    ("TINY",   1,    128,   0.10),
    ("SHORT",  129,  512,   0.20),
    ("MEDIUM", 513,  2048,  0.30),
    ("LONG",   2049, 4096,  0.25),
    ("VLONG",  4097, 8192,  0.15),
]

SEQ_KV_RANGES = [
    ("ZERO",   0,    0,     0.03),
    ("TINY",   1,    128,   0.10),
    ("SHORT",  129,  512,   0.17),
    ("MEDIUM", 513,  2048,  0.25),
    ("LONG",   2049, 4096,  0.25),
    ("VLONG",  4097, 16384, 0.20),
]


def _make_name(b, nq, nkv, d, sq, skv, sm, lse):
    g = nq // nkv
    max_sq = max(sq)
    max_skv = max(skv)
    return (
        f"B{b}_G{g}_Nq{nq}_Nkv{nkv}_D{d}_"
        f"SM{sm}_LSE{int(lse)}_"
        f"Q{max_sq}_KV{max_skv}"
    )





def _noisy_seq_len(lo, hi):
    if lo == hi:
        return lo
    v = random.randint(lo, hi)
    if v <= 0:
        return v
    # 64-aligned → 90% prob to perturb
    if v % 64 == 0 and random.random() < 0.90:
        delta = random.choice([-3, -1, 1, 3])
        v = v + delta
    # 32-aligned (but not 64) → 70% prob to perturb
    elif v % 32 == 0 and random.random() < 0.70:
        delta = random.choice([-3, -1, 1, 3])
        v = v + delta
    # even → 50% prob to make odd
    elif v % 2 == 0 and random.random() < 0.50:
        v = v + random.choice([-1, 1])
    # clamp to range
    v = max(lo, min(hi, v))
    return v


def _make_seq_list_random(b, range_pool, weights, allow_zero=False):
    chosen = random.choices(range_pool, weights=weights, k=b)
    result = []
    for (_label, lo, hi) in chosen:
        if hi == 0:
            result.append(0)
        else:
            v = _noisy_seq_len(max(lo, 1), hi)
            if not allow_zero and v == 0:
                v = _noisy_seq_len(1, max(hi, 1))
            result.append(v)
    return result


def generate_normal_cases(count=1000):
    cases = []
    name_set = set()

    attempts = 0
    max_attempts = count * 10

    while len(cases) < count and attempts < max_attempts:
        attempts += 1

        b = random.choice(B_VALUES)
        g = random.choice(GQA_RATIOS)
        nkv = random.choice(N_KV_VALUES)

        if nkv * g > N_Q_MAX:
            nkv = random.choice([v for v in N_KV_VALUES if v * g <= N_Q_MAX] or [1])
        nq = nkv * g
        d = 128
        sm = random.choice(SPARSE_MODES)
        lse = random.choice(ENABLE_LSE_VALUES)

        q_range_pool = [(label, lo, hi) for label, lo, hi, _w in SEQ_Q_RANGES]
        q_weights = [w for _, _, _, w in SEQ_Q_RANGES]
        sq = _make_seq_list_random(b, q_range_pool, q_weights, allow_zero=False)

        kv_range_pool = [(label, lo, hi) for label, lo, hi, _w in SEQ_KV_RANGES]
        kv_weights = [w for _, _, _, w in SEQ_KV_RANGES]
        skv = _make_seq_list_random(b, kv_range_pool, kv_weights, allow_zero=True)

        name = _make_name(b, nq, nkv, d, sq, skv, sm, lse)
        if name in name_set:
            continue
        name_set.add(name)

        case = {
            "name": name,
            "B": b,
            "N_q": nq,
            "N_kv": nkv,
            "D": d,
            "actual_seq_q": sq,
            "actual_seq_kv": skv,
            "enable_pa": True,
            "enable_lse": lse,
            "block_size": 128,
            "sparse_mode": sm,
            "input_layout": "NTD_TND",
            "q_scale_layout": "NT",
            "kv_cache_layout": "BnNBsD",
            "p_scale": 1.0,
        }
        cases.append(case)

    return cases


# ==============================================================================
# Exception Case Construction
# ==============================================================================

def _valid_case_base():
    nkv = random.choice([3, 5, 7, 8])
    return {
        "name": "",
        "B": random.choice([1, 2, 3]),
        "N_q": nkv * 2,
        "N_kv": nkv,
        "D": 128,
        "actual_seq_q": [random.randint(1, 128)],
        "actual_seq_kv": [random.randint(128, 512)],
        "enable_pa": True,
        "enable_lse": True,
        "block_size": 128,
        "sparse_mode": 3,
        "input_layout": "NTD_TND",
        "q_scale_layout": "NT",
        "kv_cache_layout": "BnNBsD",
        "p_scale": 1.0,
    }


def generate_exception_cases(count=200):
    cases = []
    name_set = set()
    base = _valid_case_base()

    exception_specs = [
        # (label, proportion, mutator_fn)
        ("D_WRONG_64", 0.05, lambda c: {**c, "D": 64}),
        ("D_WRONG_256", 0.05, lambda c: {**c, "D": 256}),
        ("D_WRONG_32", 0.03, lambda c: {**c, "D": 32}),
        ("D_WRONG_512", 0.02, lambda c: {**c, "D": 512}),

        ("PA_FALSE", 0.05, lambda c: {**c, "enable_pa": False}),
        ("PA_FALSE_LSE_TRUE", 0.03, lambda c: {**c, "enable_pa": False, "enable_lse": True}),

        ("BLOCK_64", 0.05, lambda c: {**c, "block_size": 64}),
        ("BLOCK_256", 0.05, lambda c: {**c, "block_size": 256}),
        ("BLOCK_32", 0.02, lambda c: {**c, "block_size": 32}),
        ("BLOCK_512", 0.02, lambda c: {**c, "block_size": 512}),

        ("SM_1", 0.04, lambda c: {**c, "sparse_mode": 1}),
        ("SM_2", 0.04, lambda c: {**c, "sparse_mode": 2}),
        ("SM_4", 0.03, lambda c: {**c, "sparse_mode": 4}),
        ("SM_5", 0.03, lambda c: {**c, "sparse_mode": 5}),
        ("SM_NEG1", 0.02, lambda c: {**c, "sparse_mode": -1}),

        ("LAYOUT_BNSD", 0.04, lambda c: {**c, "input_layout": "BNSD"}),
        ("LAYOUT_BSND", 0.03, lambda c: {**c, "input_layout": "BSND"}),
        ("LAYOUT_TND", 0.03, lambda c: {**c, "input_layout": "TND"}),

        ("QSCALE_TN", 0.04, lambda c: {**c, "q_scale_layout": "TN"}),
        ("QSCALE_BNSD", 0.03, lambda c: {**c, "q_scale_layout": "BNSD"}),

        ("KVCACHE_DEFAULT", 0.04, lambda c: {**c, "kv_cache_layout": "default"}),
        ("KVCACHE_BSH", 0.03, lambda c: {**c, "kv_cache_layout": "BSH"}),

        ("GQA_RATIO_128", 0.03, lambda c: gen_high_gqa(128, c, SM=0)),
        ("GQA_RATIO_256", 0.02, lambda c: gen_high_gqa(256, c, SM=0)),
        ("GQA_RATIO_65", 0.03, lambda c: gen_high_gqa(65, c, SM=3)),

        ("QSEQ_OVER_64K", 0.04, lambda c: {**c, "actual_seq_q": [70000]}),
        ("QSEQ_OVER_128K", 0.02, lambda c: {**c, "actual_seq_q": [131072]}),

        ("KVSEQ_OVER_256K", 0.04, lambda c: {**c, "actual_seq_kv": [300000]}),
        ("KVSEQ_OVER_500K", 0.02, lambda c: {**c, "actual_seq_kv": [524288]}),

        ("COMBINED_PARAM_ERR", 0.05, lambda c: combined_error(c)),
    ]

    for label, proportion, mutator in exception_specs:
        n = max(1, int(count * proportion))
        for i in range(n):
            case = mutator(dict(base))
            g_ratio = case["N_q"] // case["N_kv"] if case["N_kv"] != 0 else 999
            max_q = max(case["actual_seq_q"])
            max_kv = max(case["actual_seq_kv"])
            case["name"] = f"EXCEPTION_{label}_{i:03d}_G{g_ratio}_Nq{case['N_q']}_Q{max_q}_KV{max_kv}"
            if case["name"] in name_set:
                case["name"] = f"EXCEPTION_{label}_{i:03d}_v2_G{g_ratio}"
            if case["name"] in name_set:
                continue
            name_set.add(case["name"])
            cases.append(case)

    # Fill remaining with random combinations
    while len(cases) < count:
        label = "MIXED"
        case = combined_error(dict(base))
        g_ratio = case["N_q"] // case["N_kv"] if case["N_kv"] != 0 else 999
        max_q = max(case["actual_seq_q"])
        max_kv = max(case["actual_seq_kv"])
        case["name"] = f"EXCEPTION_{label}_{len(cases):03d}_G{g_ratio}_Nq{case['N_q']}_Q{max_q}_KV{max_kv}"
        if case["name"] in name_set:
            continue
        name_set.add(case["name"])
        cases.append(case)

    return cases[:count]


def gen_high_gqa(g_ratio, base, SM=0):
    nkv = 1
    nq = min(nkv * g_ratio, 256)
    return {
        **base,
        "N_q": nq,
        "N_kv": nkv,
        "sparse_mode": SM,
    }


def combined_error(base):
    error_choices = [
        ("D", [64, 256, 32, 512]),
        ("block_size", [64, 256, 32]),
        ("sparse_mode", [1, 2, 4, 5]),
        ("input_layout", ["BNSD", "BSND", "TND"]),
        ("q_scale_layout", ["TN", "BNSD"]),
        ("kv_cache_layout", ["default", "BSH"]),
    ]
    n_errors = random.choice([1, 2])
    selected = random.sample(error_choices, n_errors)
    c = dict(base)
    for key, pool in selected:
        c[key] = random.choice(pool)
    if random.random() < 0.3:
        g_ratio = random.choice([65, 128, 256])
        c["N_kv"] = 1
        c["N_q"] = min(g_ratio, 256)
    if random.random() < 0.3:
        c["enable_pa"] = False
    return c


# ==============================================================================
# Main
# ==============================================================================
def main():
    normal_cases = generate_normal_cases(1000)
    exception_cases = generate_exception_cases(200)

    print(f"[GEN] Normal cases generated: {len(normal_cases)}")
    print(f"[GEN] Exception cases generated: {len(exception_cases)}")

    all_cases = normal_cases + exception_cases

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2)

    print(f"[GEN] Total {len(all_cases)} cases written to {OUTPUT_PATH}")

    # Print distribution summary
    normal_b_values = {}
    normal_gqa = {}
    normal_nkv = {}
    normal_sm = {}
    normal_lse = {True: 0, False: 0}
    normal_q_seq_ranges = {"tiny(1-128)": 0, "short(129-512)": 0, "medium(513-2048)": 0,
                           "long(2049-4096)": 0, "vlong(4097-8192)": 0}
    normal_kv_seq_ranges = {"zero": 0, "tiny(1-128)": 0, "short(129-512)": 0, "medium(513-2048)": 0,
                            "long(2049-4096)": 0, "vlong(4097-16384)": 0}

    for c in normal_cases:
        normal_b_values[c["B"]] = normal_b_values.get(c["B"], 0) + 1
        g = c["N_q"] // c["N_kv"]
        normal_gqa[g] = normal_gqa.get(g, 0) + 1
        normal_nkv[c["N_kv"]] = normal_nkv.get(c["N_kv"], 0) + 1
        normal_sm[c["sparse_mode"]] = normal_sm.get(c["sparse_mode"], 0) + 1
        normal_lse[c["enable_lse"]] += 1
        max_sq = max(c["actual_seq_q"])
        if max_sq <= 128:
            normal_q_seq_ranges["tiny(1-128)"] += 1
        elif max_sq <= 512:
            normal_q_seq_ranges["short(129-512)"] += 1
        elif max_sq <= 2048:
            normal_q_seq_ranges["medium(513-2048)"] += 1
        elif max_sq <= 4096:
            normal_q_seq_ranges["long(2049-4096)"] += 1
        else:
            normal_q_seq_ranges["vlong(4097-8192)"] += 1
        max_skv = max(c["actual_seq_kv"])
        if max_skv == 0:
            normal_kv_seq_ranges["zero"] += 1
        elif max_skv <= 128:
            normal_kv_seq_ranges["tiny(1-128)"] += 1
        elif max_skv <= 512:
            normal_kv_seq_ranges["short(129-512)"] += 1
        elif max_skv <= 2048:
            normal_kv_seq_ranges["medium(513-2048)"] += 1
        elif max_skv <= 4096:
            normal_kv_seq_ranges["long(2049-4096)"] += 1
        else:
            normal_kv_seq_ranges["vlong(4097-16384)"] += 1

    exception_labels = {}
    for c in exception_cases:
        label = c["name"].split("_")[1] if "EXCEPTION_" in c["name"] else "OTHER"
        exception_labels[label] = exception_labels.get(label, 0) + 1

    print("\n[Normal Case Distribution]")
    print(f"  B: {dict(sorted(normal_b_values.items()))}")
    print(f"  GQA ratios: {dict(sorted(normal_gqa.items()))}")
    print(f"  N_kv: {dict(sorted(normal_nkv.items()))}")
    print(f"  Sparse mode: {dict(sorted(normal_sm.items()))}")
    print(f"  LSE: {normal_lse}")
    print(f"  Q seq ranges: {normal_q_seq_ranges}")
    print(f"  KV seq ranges: {normal_kv_seq_ranges}")
    print(f"\n[Exception Case Distribution]")
    for k, v in sorted(exception_labels.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
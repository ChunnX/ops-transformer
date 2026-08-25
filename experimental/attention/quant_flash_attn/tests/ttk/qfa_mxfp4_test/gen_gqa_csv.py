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

"""Generate GQA test cases CSV for qfa_mxfp4.

G = N_q / N_kv, G <= 64, N_q % N_kv == 0.
Layout: BNSD only. No advanced features (no mask, no lse, no rope).

Size constraint: B * N_q * S1 <= MAX_SIZE and B * N_kv * S2 <= MAX_SIZE.
When one parameter (B/N/S) is large, the others are automatically kept small
by the size filter, so total tensor size stays manageable.

Output: qfa_mxfp4_gqa.csv (same format as qfa_mxfp4.csv)
"""

import csv
import os
import random
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))

D = 128
V_D = 128
D_HALF = D // 2          # 64, MXFP4 packed last dim
D_PER_GROUP = D // 64    # 2

PRECISION = "((0.0078125, 0.005), (0.0078125, 0.005))"
ABS_PRECISION = "(0.0001, 0.0001)"

# Size budget: B * N_q * S1 and B * N_kv * S2 must be <= this.
# B=28, N_q=40, S1=1024 -> 1,146,880 (allowed but near limit)
MAX_SIZE = 1_200_000

# GQA configs: (G, [N_kv options])
# G = N_q / N_kv, G <= 64, N_q % N_kv == 0
GQA_CONFIGS = [
    (2,  [1, 2, 4, 6, 8, 12, 16, 32]),
    (3,  [1, 2, 4, 8, 16]),
    (4,  [1, 2, 3, 4, 8, 10, 16]),
    (5,  [1, 2, 4, 8]),
    (6,  [1, 2, 4, 8]),
    (7,  [1, 2, 4, 8]),
    (8,  [1, 2, 4, 8, 16]),
    (10, [1, 2, 4]),
    (12, [1, 2, 4]),
    (16, [1, 2, 4]),
    (20, [1, 2]),
    (24, [1, 2]),
    (32, [1, 2]),
    (48, [1]),
    (64, [1]),
]

B_VALUES = [1, 2, 4, 8, 16, 20, 28]

# Diverse S pool: decode (1-2), small, medium, large, non-aligned boundaries
S_VALUES = [
    1, 2, 4, 8, 16, 32, 33, 64, 65, 96, 128, 160, 256, 420, 512,
    768, 1024, 1025, 1280, 1743, 2048, 2185, 2400, 4096, 6000, 7500, 8192,
]

DATA_RANGES = [
    "1.0", "-1,1", "0.01,1", "-100,100", "-10,10", "0,1", "-0.5,0.5",
]

# act_seq variants
ACT_VARIANTS = ["full", "varlen", "partial", "maxseq", "mixed", "maxseq_pad"]


def _v_descale_blocks(s2_effective):
    n_blocks = (s2_effective + 31) // 32
    if n_blocks % 2 != 0:
        n_blocks += 1
    return n_blocks // 2


def _gen_act(B, S1, S2, variant, rng):
    """Generate (act_q, act_kv, max_sq, max_skv) based on variant."""
    if variant == "full":
        return [S1] * B, [S2] * B, -1, -1
    elif variant == "varlen":
        aq = [max(1, S1 - rng.randint(0, max(1, S1 // 4))) for _ in range(B)]
        ak = [max(1, S2 - rng.randint(0, max(1, S2 // 4))) for _ in range(B)]
        return aq, ak, -1, -1
    elif variant == "partial":
        return [max(1, S1 * 3 // 4)] * B, [max(1, S2 * 3 // 4)] * B, -1, -1
    elif variant == "maxseq":
        return [], [], S1, S2
    elif variant == "mixed":
        aq = [S1 if i % 2 == 0 else max(1, S1 // 2) for i in range(B)]
        ak = [S2 if i % 2 == 0 else max(1, S2 // 2) for i in range(B)]
        return aq, ak, -1, -1
    elif variant == "maxseq_pad":
        return [], [], max(1, S1 // 2), max(1, S2 // 2)
    return [S1] * B, [S2] * B, -1, -1


def _make_row(name, B, N_q, N_kv, S1, S2, act_q, act_kv, max_sq, max_skv, dr):
    G = N_q // N_kv
    has_act = bool(act_q) and bool(act_kv)

    if has_act:
        s2_eff = max(act_kv)
    elif max_skv >= 0:
        s2_eff = max_skv
    else:
        s2_eff = S2

    v_descale_s = _v_descale_blocks(s2_eff)

    q_shape = (B, N_q, S1, D_HALF)
    k_shape = (B, N_kv, S2, D_HALF)
    v_shape = (B, N_kv, S2, D_HALF)
    q_descale_shape = (B, N_q, S1, D_PER_GROUP, 2)
    k_descale_shape = (B, N_kv, S2, D_PER_GROUP, 2)
    v_descale_shape = (B, N_kv, v_descale_s, V_D, 2)
    block_table_shape = (0,)

    shapes = [q_shape, k_shape, v_shape, q_descale_shape, k_descale_shape,
              v_descale_shape, block_table_shape]
    tensor_view_shapes = "(" + ",".join(f"({s})" for s in shapes) + ")"
    tensor_dtypes = "('uint8','uint8','uint8','uint8','uint8','uint8','int32')"

    if has_act:
        q_descale_dtype = "uint8"
        k_descale_dtype = "uint8"
        v_descale_dtype = "uint8"
        seqused_q_dtype = "int32"
        seqused_kv_dtype = "int32"
    else:
        q_descale_dtype = None
        k_descale_dtype = None
        v_descale_dtype = None
        seqused_q_dtype = None
        seqused_kv_dtype = None

    attrs = {
        'B': B, 'N_q': N_q, 'N_kv': N_kv, 'G': G, 'D': D, 'V_D': V_D, 'Rope_D': 0,
        'act_seq_lens_q': act_q, 'act_seq_lens_kv': act_kv,
        'cu_seqlens_q': [], 'cu_seqlens_kv': [],
        'max_seqlen_q': max_sq, 'max_seqlen_kv': max_skv,
        'input_layout': 'BNSD', 'layout_q_descale': 'BNSD',
        'layout_kv': 'BNSD', 'layout_out': 'BNSD',
        'kv_storage_mode': 'continue', 'block_size': 0,
        'q_dtype': 'fp4_e2m1', 'kv_dtype': 'fp4_e2m1', 'out_dtype': 'bfloat16',
        'q_quant_mode': 5, 'mask_mode': 0,
        'pre_tokens': 2147483647, 'next_tokens': 2147483647,
        'enable_mask': False, 'enable_lse': False, 'inner_precise': 0,
        'device_id': 0, 'graph_path': 0, 'softmax_scale': None,
        'block_table_shape': [], 'block_table_dtype': None,
        'p_scale_value': None, 'p_scale_shape': [], 'p_scale_dtype': None,
        'p_scale_datarange': None,
        'sinks_shape': [], 'sinks_dtype': None, 'sinks_datarange': None,
        'attn_mask_shape': [], 'attn_mask_dtype': None, 'attn_mask_datarange': None,
        'q_descale_dtype': q_descale_dtype,
        'k_descale_dtype': k_descale_dtype,
        'v_descale_dtype': v_descale_dtype,
        'seqused_q_dtype': seqused_q_dtype,
        'seqused_kv_dtype': seqused_kv_dtype,
        'cu_seqlens_q_dtype': None, 'cu_seqlens_kv_dtype': None,
        'softmax_lse_dtype': None,
        'data_range_q': dr, 'data_range_k': dr, 'data_range_v': dr,
    }
    attributes = repr(attrs)

    return [
        name, 'qfa_mxfp4_wrapper.npu_qfa_mxfp4',
        tensor_view_shapes, tensor_dtypes, "",
        attributes, "", "",
        PRECISION, ABS_PRECISION,
    ]


def generate_cases():
    rng = random.Random(42)
    target = 500

    # Build all (G, N_kv, N_q) pairs
    gqa_pairs = []
    for G, nkv_list in GQA_CONFIGS:
        for N_kv in nkv_list:
            N_q = G * N_kv
            if N_q <= 64 and G <= 64:
                gqa_pairs.append((G, N_kv, N_q))

    # For each pair, pre-compute valid (B, S1, S2) combos under size budget
    pair_scenarios = []
    for G, N_kv, N_q in gqa_pairs:
        valid_bss = []
        for B in B_VALUES:
            max_s1 = MAX_SIZE // (B * N_q)
            max_s2 = MAX_SIZE // (B * N_kv)
            for S1 in S_VALUES:
                if S1 > max_s1:
                    continue
                for S2 in S_VALUES:
                    if S2 > max_s2:
                        continue
                    valid_bss.append((B, S1, S2))
        rng.shuffle(valid_bss)
        pair_scenarios.append(valid_bss)

    # Round-robin across all pairs so every G value gets represented
    selected = []
    scenario_idx = [0] * len(gqa_pairs)
    pair_ptr = 0
    while len(selected) < target:
        pi = pair_ptr % len(gqa_pairs)
        si = scenario_idx[pi]
        scenarios = pair_scenarios[pi]

        if si < len(scenarios):
            B, S1, S2 = scenarios[si]
            G, N_kv, N_q = gqa_pairs[pi]
            variant = ACT_VARIANTS[si % len(ACT_VARIANTS)]
            dr = DATA_RANGES[si % len(DATA_RANGES)]
            selected.append((G, N_kv, N_q, B, S1, S2, variant, dr))
            scenario_idx[pi] += 1

        pair_ptr += 1
        if all(scenario_idx[j] >= len(pair_scenarios[j]) for j in range(len(gqa_pairs))):
            break

    rng.shuffle(selected)
    selected = selected[:target]

    # Generate rows
    cases = []
    for idx, (G, N_kv, N_q, B, S1, S2, variant, dr) in enumerate(selected, 1):
        act_q, act_kv, max_sq, max_skv = _gen_act(B, S1, S2, variant, rng)
        name = f"GQA_G{G}_B{B}_Nq{N_q}_Nkv{N_kv}_S1_{S1}_S2_{S2}_{idx:04d}"
        row = _make_row(name, B, N_q, N_kv, S1, S2,
                        act_q, act_kv, max_sq, max_skv, dr)
        cases.append(row)

    return cases


def main():
    cases = generate_cases()
    out_path = os.path.join(_HERE, "qfa_mxfp4_gqa.csv")

    header = [
        "testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
        "tensor_formats", "attributes", "output_tensor_indexes", "golden_api",
        "precision_tolerances", "absolute_precision",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in cases:
            writer.writerow(row)

    print(f"Wrote {len(cases)} GQA cases to {out_path}")

    # Stats
    import ast
    from collections import Counter
    G_counter = Counter()
    B_counter = Counter()
    S1_counter = Counter()
    S2_counter = Counter()
    variant_counter = Counter()
    dr_counter = Counter()
    max_q_size = 0
    max_kv_size = 0
    for row in cases:
        attrs = ast.literal_eval(row[5])
        B = attrs['B']
        Nq = attrs['N_q']
        Nkv = attrs['N_kv']
        G = attrs['G']
        G_counter[G] += 1
        B_counter[B] += 1

        # Extract S1/S2 from tensor_view_shapes
        import re
        shape_tuples = re.findall(r'\(([\d, ]+)\)', row[2])
        S1 = int(shape_tuples[0].split(',')[2])
        S2 = int(shape_tuples[1].split(',')[2])
        S1_counter[S1] += 1
        S2_counter[S2] += 1

        q_size = B * Nq * S1
        kv_size = B * Nkv * S2
        max_q_size = max(max_q_size, q_size)
        max_kv_size = max(max_kv_size, kv_size)

        if attrs['act_seq_lens_q']:
            variant_counter["has_act"] += 1
        else:
            variant_counter["maxseq"] += 1
        dr_counter[attrs['data_range_q']] += 1

    print(f"G distribution:      {dict(sorted(G_counter.items()))}")
    print(f"B distribution:      {dict(sorted(B_counter.items()))}")
    print(f"S1 distinct values:  {len(S1_counter)} (top: {S1_counter.most_common(5)})")
    print(f"S2 distinct values:  {len(S2_counter)} (top: {S2_counter.most_common(5)})")
    print(f"Act variant:         {dict(variant_counter)}")
    print(f"Data range dist:     {dict(dr_counter)}")
    print(f"Max B*Nq*S1:         {max_q_size}")
    print(f"Max B*Nkv*S2:        {max_kv_size}")


if __name__ == "__main__":
    main()

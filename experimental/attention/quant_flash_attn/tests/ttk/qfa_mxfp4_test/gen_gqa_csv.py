#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not be this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generate GQA test cases CSV for qfa_mxfp4.

G = N_q / N_kv, G <= 64, N_q % N_kv == 0.
Output: qfa_mxfp4_gqa.csv (same format as qfa_mxfp4.csv)
"""

import csv
import itertools
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))

D = 128
V_D = 128
D_HALF = D // 2          # 64, MXFP4 packed last dim
D_PER_GROUP = D // 64    # 2

PRECISION = "((0.0078125, 0.005), (0.0078125, 0.005))"
ABS_PRECISION = "(0.0001, 0.0001)"

DATA_RANGES = ["1.0", "-1,1", "0.01,1", "-100,100", "-10,10"]


def _v_descale_blocks(s2_effective):
    n_blocks = (s2_effective + 31) // 32
    if n_blocks % 2 != 0:
        n_blocks += 1
    return n_blocks // 2


def _make_act_seq(B, s_max, variant=0):
    """Generate act_seq_lens for B batches around s_max."""
    if variant == 0:
        return [s_max] * B
    elif variant == 1:
        return [max(1, s_max - random.randint(0, min(10, s_max - 1))) for _ in range(B)]
    elif variant == 2:
        return [max(1, s_max - i * 2) for i in range(B)]
    else:
        return [max(1, s_max // 2 + random.randint(0, s_max // 2)) for _ in range(B)]


def _make_row(name, B, N_q, N_kv, S1, S2, act_q, act_kv, max_sq, max_skv, data_range):
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
        'data_range_q': data_range, 'data_range_k': data_range,
        'data_range_v': data_range,
    }
    attributes = repr(attrs)

    return [
        name, 'qfa_mxfp4_wrapper.npu_qfa_mxfp4',
        tensor_view_shapes, tensor_dtypes, "",
        attributes, "", "",
        PRECISION, ABS_PRECISION,
    ]


# GQA configs: (G, [N_kv options])
# G = N_q / N_kv, G <= 64, N_q % N_kv == 0
GQA_CONFIGS = [
    (2,  [1, 2, 4, 6, 8, 12, 16, 32]),
    (3,  [1, 2, 4, 8, 16]),
    (4,  [1, 2, 3, 4, 8, 10, 16, 32]),
    (5,  [1, 2, 4, 8]),
    (6,  [1, 2, 4, 8, 16]),
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

# B, S1, S2 combos
B_VALUES = [1, 2, 4, 8, 16]
S1_S2_COMBOS = [
    (128, 128),
    (128, 256),
    (128, 512),
    (128, 1024),
    (256, 256),
    (256, 512),
    (256, 1024),
    (512, 512),
    (512, 1024),
    (512, 2048),
    (1024, 1024),
    (1024, 2048),
    (1024, 4096),
    (2, 512),
    (2, 1024),
    (2, 2048),
    (2, 2400),
    (2, 4096),
    (2, 8192),
    (1, 128),
    (1, 512),
    (1, 1024),
    (420, 7500),
    (1743, 2185),
    (380, 6000),
    (16, 64),
    (32, 128),
    (64, 256),
    (96, 512),
    (160, 1024),
]


def generate_cases():
    random.seed(42)
    cases = []
    idx = 0
    target = 500

    # Round-robin across G configs to ensure all G values are covered
    # Flatten all (G, N_kv) pairs
    gqa_pairs = []
    for G, nkv_list in GQA_CONFIGS:
        for N_kv in nkv_list:
            N_q = G * N_kv
            if N_q <= 128:
                gqa_pairs.append((G, N_q, N_kv))

    # Build a flat list of all (B, S1, S2, variant, dr) combos
    combos = []
    for B in B_VALUES:
        for S1, S2 in S1_S2_COMBOS:
            for variant in range(2):
                for dr_idx, dr in enumerate(DATA_RANGES):
                    combos.append((B, S1, S2, variant, dr))

    # Round-robin: iterate combos, for each combo pick next gqa_pair
    combo_idx = 0
    pair_idx = 0
    while idx < target:
        B, S1, S2, variant, dr = combos[combo_idx % len(combos)]
        G, N_q, N_kv = gqa_pairs[pair_idx % len(gqa_pairs)]

        idx += 1
        name = f"GQA_G{G}_B{B}_Nq{N_q}_Nkv{N_kv}_S1_{S1}_S2_{S2}_{idx:04d}"

        if variant == 0:
            act_q = _make_act_seq(B, S1, variant)
            act_kv = _make_act_seq(B, S2, variant)
            max_sq = -1
            max_skv = -1
        else:
            act_q = []
            act_kv = []
            max_sq = S1
            max_skv = S2

        row = _make_row(name, B, N_q, N_kv, S1, S2,
                        act_q, act_kv, max_sq, max_skv, dr)
        cases.append(row)

        pair_idx += 1
        if pair_idx % len(gqa_pairs) == 0:
            combo_idx += 1

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

    G_set = set()
    for row in cases:
        import ast
        attrs = ast.literal_eval(row[5])
        G_set.add(attrs['G'])
    print(f"G values covered: {sorted(G_set)}")


if __name__ == "__main__":
    main()

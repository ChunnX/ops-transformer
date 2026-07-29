#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Generate MLA D512 non-quant test cases that route to the restructured MLA template.

Generates N cases with randomized:
  - B, N1, S1, S2, KV layout
  - Per-batch actual_seq_kv (random, can include 0, high/low variance)
  - sparse_mode (0 only currently, field reserved for future mask support)
  - block_size (128 only, field reserved)
Ensures total input tensor size stays within memory budget to avoid OOM.

Usage:
    python gen_mla_cases.py --num 100 --max_mem_gb 25 --output mla_d512_noquant_gen.py
"""

import argparse
import math
import random


QK_D = 512
ROPE_D = 64
V_D = 512
BS = 128

S2_RANGES = [
    (128 * 1024, 128 * 1024 + 2 * 1024),
    (400 * 1024, 400 * 1024 + 2 * 1024),
    (1, 512),
    (512, 4096),
    (4096, 32768),
    (32768, 65536),
]

S1_OPTIONS = list(range(1, 17))
N1_OPTIONS = [2**i for i in range(1, 8)]
B_OPTIONS = list(range(1, 17))
PA_LAYOUTS = ["PA_BBH", "PA_BNBD", "PA_NZ", "BNSD"]

VARIANCE_PROFILES = [
    "uniform",
    "high_var",
    "with_zero",
    "small",
    "extreme",
]


def _gen_act_kv(b, s2_max, profile, rng):
    if profile == "uniform":
        return [s2_max] * b
    if profile == "high_var":
        return [rng.randint(s2_max // 4, s2_max) for _ in range(b)]
    if profile == "with_zero":
        return [rng.choice([0, s2_max]) if b > 1 else s2_max for _ in range(b)]
    if profile == "small":
        return [rng.randint(1, min(s2_max, 1024)) for _ in range(b)]
    if profile == "extreme":
        return [rng.randint(0, s2_max) for _ in range(b)]
    return [s2_max] * b


def est_mem_bytes(b, n1, n2, s1, s2_max, bs, qk_d, rope_d, v_d):
    q_bytes = b * n1 * s1 * (qk_d + rope_d) * 2
    num_blocks = math.ceil(s2_max / bs) + 1
    kv_bytes = b * num_blocks * n2 * bs * (qk_d + v_d + rope_d) * 2
    out_bytes = b * n1 * s1 * v_d * 2
    return q_bytes + kv_bytes + out_bytes


def gen_cases(num_cases, max_mem_gb, seed=2026):
    rng = random.Random(seed)
    max_mem = int(max_mem_gb * 1024**3)
    cases = []
    for i in range(num_cases):
        while True:
            layout = rng.choice(PA_LAYOUTS)
            n1 = rng.choice(N1_OPTIONS)
            n2 = 1
            b = rng.choice(B_OPTIONS)
            s1 = rng.choice(S1_OPTIONS)
            s2_lo, s2_hi = rng.choice(S2_RANGES)
            s2_max = rng.randint(s2_lo, s2_hi)
            mem = est_mem_bytes(b, n1, n2, s1, s2_max, BS, QK_D, ROPE_D, V_D)
            if mem <= max_mem:
                break

        variance = rng.choice(VARIANCE_PROFILES)
        act_kv = _gen_act_kv(b, s2_max, variance, rng)
        act_kv = [min(v, s2_max) for v in act_kv]

        cases.append({
            "id": f"case_{i+1:03d}",
            "b": b, "n1": n1, "n2": n2, "s1": s1, "s2": s2_max,
            "layout": layout,
            "act_kv": act_kv,
            "variance": variance,
        })
    return cases


def write_cases_file(cases, output_path):
    lines = [
        '#!/usr/bin/python',
        '# -*- coding: utf-8 -*-',
        '"""Auto-generated MLA D512 non-quant test cases."""',
        '',
        'BASE = {',
        '    "B": [1],',
        '    "N1": [2],',
        '    "N2": [1],',
        '    "S1": [1],',
        '    "S2": [512],',
        '    "D": [512],',
        '    "DV": [512],',
        '    "DRope": [64],',
        '    "Dtype": ["fp16"],',
        '    "layout_q": ["BNSD"],',
        '    "layout_kv": ["BNSD"],',
        '    "layout_out": ["BNSD"],',
        '    "block_size": [128],',
        '    "return_softmax_lse": [False],',
        '    "sparse_mode": [0],',
        '    "q_range": [(-5.0, 5.0)],',
        '    "k_range": [(-5.0, 5.0)],',
        '    "v_range": [(-5.0, 5.0)],',
        '    "kv_reused": [True],',
        '}',
        '',
        'TestCases = {',
    ]
    for c in cases:
        act_kv_str = "[" + ",".join(str(v) for v in c["act_kv"]) + "]"
        lines.append(
            f'    "{c["id"]}": {{**BASE,'
            f' "B": [{c["b"]}], "N1": [{c["n1"]}], "S1": [{c["s1"]}], "S2": [{c["s2"]}],'
            f' "layout_kv": ["{c["layout"]}"],'
            f' "act_kv": [{act_kv_str}],'
            f' "variance": ["{c["variance"]}"]}},'
        )
    lines.append('}')
    lines.append('')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Generated {len(cases)} cases -> {output_path}")
    max_mem = max(est_mem_bytes(c["b"], c["n1"], c["n2"], c["s1"], c["s2"], BS, QK_D, ROPE_D, V_D) for c in cases)
    print(f"Max single-case memory: {max_mem / 1024**3:.2f} GB")
    from collections import Counter
    var_counts = Counter(c["variance"] for c in cases)
    layout_counts = Counter(c["layout"] for c in cases)
    print(f"Variance profiles: {dict(var_counts)}")
    print(f"Layouts: {dict(layout_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MLA D512 non-quant test cases")
    parser.add_argument("--num", type=int, default=100, help="Number of cases to generate")
    parser.add_argument("--max_mem_gb", type=float, default=25, help="Max single-case memory in GB")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--output", type=str, default="mla_d512_noquant_gen.py",
                        help="Output test_cases file path")
    args = parser.parse_args()

    cases = gen_cases(args.num, args.max_mem_gb, args.seed)
    write_cases_file(cases, args.output)

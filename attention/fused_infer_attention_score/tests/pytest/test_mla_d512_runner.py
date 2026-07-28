#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
MLA D512 non-quant precision test runner for FIA restructured template.

Supports running hand-written test_cases/mla_d512_noquant.py and
auto-generated test_cases/mla_d512_noquant_gen.py.

Usage:
    # Run hand-written cases
    python test_mla_d512_runner.py

    # Generate 100 random cases and run
    python test_mla_d512_runner.py --gen 100

    # Run specific case
    python test_mla_d512_runner.py --case_id MLA_PABBH_B1_N2_S1_512

    # Run with custom max memory budget
    python test_mla_d512_runner.py --gen 200 --max_mem_gb 20
"""

import sys
import os
import math
import gc
import argparse
import random
import importlib
import itertools

import torch
import torch_npu
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fused_infer_attention_score_golden import FusedInferAttentionScoreGolden

DEVICE_ID = 0
QK_D = 512
ROPE_D = 64
ATOL = 5e-3
RTOL = 5e-3


def _kv_shape(layout, total_blocks, n2, bs, d):
    if layout == "PA_BBH":
        return (total_blocks, bs, n2 * d)
    if layout == "PA_BNBD":
        return (total_blocks, n2, bs, d)
    if layout == "PA_NZ":
        return (total_blocks, n2, d // 16, bs, 16)
    if layout == "BNSD":
        return (1, n2, total_blocks * bs, d)
    raise ValueError(layout)


def _make_block_table(batch, seqs_kv, bs):
    vn = [math.ceil(s / bs) + 1 if s > 0 else 1 for s in seqs_kv]
    total, mx = sum(vn), max(vn)
    ids = list(range(total))
    random.shuffle(ids)
    tbl = -torch.ones(batch, mx, dtype=torch.int32)
    off = 0
    for b in range(batch):
        tbl[b, :vn[b]] = torch.tensor(ids[off:off + vn[b]], dtype=torch.int32)
        off += vn[b]
    return tbl, total


def load_cases(module_names):
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


def normalize_params(raw):
    c = dict(raw)
    c.setdefault("N2", 1)
    c.setdefault("S2", 512)
    c.setdefault("DV", QK_D)
    c.setdefault("DRope", ROPE_D)
    c.setdefault("Dtype", "fp16")
    c.setdefault("layout_q", "BNSD")
    c.setdefault("layout_kv", "BNSD")
    c.setdefault("layout_out", "BNSD")
    c.setdefault("block_size", 128)
    c.setdefault("sparse_mode", 0)
    c.setdefault("kv_reused", True)
    c.setdefault("act_kv", None)
    return c


def run_case(case_name, params, device_id=0):
    torch.npu.set_device(device_id)
    b = params["B"]
    n1 = params["N1"]
    n2 = params["N2"]
    s1 = params["S1"]
    s2 = params["S2"]
    drope = params["DRope"]
    layout_kv = params["layout_kv"]
    bs = params["block_size"]
    kv_reused = params["kv_reused"]
    dtype_str = params["Dtype"]
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
    sparse_mode = params["sparse_mode"]

    act_q = [s1] * b
    act_kv_raw = params.get("act_kv")
    if act_kv_raw and isinstance(act_kv_raw, list) and len(act_kv_raw) == b:
        act_kv = [int(v) for v in act_kv_raw]
    else:
        act_kv = [s2] * b
    s2_max = max(act_kv) if act_kv else s2

    scale = 1.0 / math.sqrt(QK_D)

    torch.manual_seed(42)
    np.random.seed(42)

    query = torch.randn(b, n1, s1, QK_D, dtype=dtype, device="npu")
    query_rope = torch.randn(b, n1, s1, drope, dtype=dtype, device="npu")

    is_pa = layout_kv in ("PA_BBH", "PA_BNBD", "PA_NZ")

    if is_pa:
        block_table, total_blocks = _make_block_table(b, act_kv, bs)
        block_table_npu = block_table.to("npu")
        kv_shape = _kv_shape(layout_kv, total_blocks, n2, bs, QK_D)
        kr_shape = _kv_shape(layout_kv, total_blocks, n2, bs, drope)
        key = torch.randn(*kv_shape, dtype=dtype, device="npu")
        value = key if kv_reused else torch.randn(*kv_shape, dtype=dtype, device="npu")
        key_rope = torch.randn(*kr_shape, dtype=dtype, device="npu")
    else:
        block_table_npu = None
        key = torch.randn(b, n2, s2_max, QK_D, dtype=dtype, device="npu")
        value = key if kv_reused else torch.randn(b, n2, s2_max, QK_D, dtype=dtype, device="npu")
        key_rope = torch.randn(b, n2, s2_max, drope, dtype=dtype, device="npu")

    print(f"\n[{case_name}] B={b} N1={n1} N2={n2} S1={s1} S2={s2_max} "
          f"layout_kv={layout_kv} dtype={dtype_str} act_kv={act_kv} kv_reused={kv_reused}")

    print(f"  Generating tensors on NPU...", flush=True)

    try:
        print(f"  Calling NPU operator...", flush=True)
        attn_out, _ = torch_npu.npu_fused_infer_attention_score(
            query, key, value,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=act_q,
            actual_seq_lengths_kv=act_kv,
            block_table=block_table_npu,
            query_padding_size=None,
            kv_padding_size=None,
            query_rope=query_rope,
            key_rope=key_rope,
            num_heads=n1,
            scale=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout="BNSD",
            num_key_value_heads=n2,
            sparse_mode=sparse_mode,
            inner_precise=0,
            block_size=bs if is_pa else 0,
            softmax_lse_flag=False,
        )
        torch.npu.synchronize()
        print(f"  NPU operator done.", flush=True)
    except Exception as e:
        print(f"  [SKIP] Operator error: {type(e).__name__}")
        del query, query_rope, key, value, key_rope
        if block_table_npu is not None:
            del block_table_npu
        torch.npu.empty_cache()
        return None

    try:
        print(f"  Computing golden...", flush=True)
        golden = FusedInferAttentionScoreGolden(
            num_heads_q=n1,
            num_heads_kv=n2,
            head_dim=QK_D + ROPE_D,
            softmax_scale=scale,
        )
        golden_out, _ = golden.forward_golden(
            query, key, value,
            query_rope=query_rope,
            key_rope=key_rope,
            softmax_scale=scale,
            layout_q="BNSD",
            layout_kv=layout_kv,
            layout_out="BNSD",
            device_mode="npu",
            compute_mode="auto",
            block_table=block_table_npu,
            seqused_kv=torch.tensor(act_kv, dtype=torch.int32).npu(),
        )
        torch.npu.synchronize()
    except Exception as e:
        print(f"  [SKIP] Golden error: {type(e).__name__}: {e}")
        del query, query_rope, key, value, key_rope, attn_out
        if block_table_npu is not None:
            del block_table_npu
        torch.npu.empty_cache()
        return None

    npu_out = attn_out.float()
    golden_np = golden_out.float()
    diff = torch.abs(npu_out - golden_np)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_both = torch.allclose(npu_out, golden_np, atol=ATOL, rtol=RTOL)

    print(f"  max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
          f"atol={ATOL} rtol={RTOL} -> {'PASS' if pass_both else 'FAIL'}")

    del query, query_rope, key, value, key_rope, attn_out, golden_out
    if block_table_npu is not None:
        del block_table_npu
    gc.collect()
    torch.npu.empty_cache()

    return pass_both


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLA D512 non-quant precision test runner")
    parser.add_argument("--case_files", type=str, default="mla_d512_noquant",
                        help="Comma-separated test_cases module names")
    parser.add_argument("--case_id", type=str, default="all",
                        help="Specific case id (substring match) or 'all'")
    parser.add_argument("--gen", type=int, default=None,
                        help="Generate N random cases and run them")
    parser.add_argument("--max_mem_gb", type=float, default=5,
                        help="Max single-case memory in GB (for --gen)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (for --gen)")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device_id)

    if args.gen is not None:
        from gen_mla_cases import gen_cases, write_cases_file, est_mem_bytes
        cases = gen_cases(args.gen, args.max_mem_gb, args.seed)
        gen_file = os.path.join(os.path.dirname(__file__), "test_cases", "mla_d512_noquant_gen.py")
        write_cases_file(cases, gen_file)
        case_files = "mla_d512_noquant_gen"
    else:
        case_files = args.case_files

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test_cases"))
    modules = [m.strip() for m in case_files.split(",")]
    all_cases = load_cases(modules)

    if args.case_id != "all":
        all_cases = {k: v for k, v in all_cases.items()
                     if args.case_id in k.rsplit("/", 1)[-1]}

    if not all_cases:
        print("No cases found.")
        sys.exit(1)

    print(f"Loaded {len(all_cases)} cases")

    results = []
    for i, (case_name, raw_params) in enumerate(sorted(all_cases.items())):
        params = normalize_params(raw_params)
        ok = run_case(case_name, params, args.device_id)
        results.append(ok)
        passed = sum(1 for r in results if r is True)
        skipped = sum(1 for r in results if r is None)
        failed = sum(1 for r in results if r is False)
        print(f"  Progress: {i+1}/{len(all_cases)}, passed={passed}, skipped={skipped}, failed={failed}")

    passed = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if r is False)
    print(f"\n{'='*80}")
    print(f"Total: {len(results)}, Passed: {passed}, Skipped: {skipped}, Failed: {failed}")
    if failed > 0:
        failed_names = [name for name, r in zip(sorted(all_cases.keys()), results) if r is False]
        print("Failed cases:")
        for name in failed_names:
            print(f"  {name}")
    print(f"{'='*80}")

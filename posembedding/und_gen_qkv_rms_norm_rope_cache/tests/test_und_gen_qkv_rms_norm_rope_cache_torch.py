#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UndGenQkvRmsNormRopeCache torch 接口测试（NPU 上板）。

用例集与精度基准都取自同目录的 golden.py（本算子唯一 golden 实现），
覆盖 headDim=128、(Hq,Hk,Hv) ∈ {(8,1,1),(16,2,2)}、block_size=16~512、T=1~64K
（T 本身算子侧不设上限，case 集覆盖到 64K）。

前置条件：
  1. 自定义算子包已安装：bash build.sh --pkg --ops=und_gen_qkv_rms_norm_rope_cache --soc=ascend950
                        && ./build/cann-ops-transformer-custom_linux-*.run --quiet
  2. torch_extension 已安装：cd torch_extension && python3 -m build --wheel -n
                            && pip install dist/*.whl --force-reinstall --no-deps
  3. export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
  4. CANN 自带的 python/site-packages 里有一份同名旧包，会遮蔽新装的 wheel，
     跑之前要把 venv 的 site-packages 前置：
     export PYTHONPATH=<venv>/lib/python3.12/site-packages:$PYTHONPATH
  5. 若环境里的 torchair 与 torch 版本不兼容（import cann_ops_transformer.ops 直接抛
     torch_npu.dynamo._ImportError），需在 import 前把 torchair 及其子模块桩进 sys.modules；
     本算子走 aclnn 直调，不经过 torchair。

用法：
  python3 test_und_gen_qkv_rms_norm_rope_cache_torch.py           # 默认 case 集（T ≤ 10752）
  python3 test_und_gen_qkv_rms_norm_rope_cache_torch.py --full    # 追加 T=64K
  python3 test_und_gen_qkv_rms_norm_rope_cache_torch.py --case t10752_typical_h8

基线：2026-08-03 在 A5（Ascend950PR_9579）上 12/12 全 PASS（含 --full 的两个 T=64K）。
"""

import argparse
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import (  # noqa: E402
    GENERALIZED_CASES,
    HEAD_DIM,
    TYPICAL_CASES,
    build_case,
    gather_cache_rows,
    golden_dense,
)

# 直接导入本算子，避免加载其他无关算子
from cann_ops_transformer.ops.und_gen_qkv_rms_norm_rope_cache import (  # noqa: E402
    und_gen_qkv_rms_norm_rope_cache,
)

# bf16 输出的比对判据：尾数 8 bit，取 rtol=2^-8，atol 覆盖近零值
RTOL = 4e-3
ATOL = 1e-3


def _to_npu(t):
    return None if t is None else t.npu()


def compare(name, got, ref):
    """got/ref 均为 CPU 张量；ref 为 golden 的 float32 结果，按 bf16 判据比对。"""
    got_f = got.float()
    ref_f = ref.float()
    diff = (got_f - ref_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    bad = (diff > (ATOL + RTOL * ref_f.abs())).sum().item()
    ok = bad == 0
    return ok, f"{name}: max_abs={max_abs:.5f} max_rel={max_rel:.5f} 不达标元素={bad}/{ref_f.numel()}"


def run_case(spec, verbose=True):
    case = build_case(spec)
    meta = case.pop("_meta")
    total, (hq, hk, hv) = meta["total"], meta["heads"]

    # ---- golden（CPU float32）----
    ref_q, ref_k, ref_v = golden_dense(
        case["und_qkv"], case["gen_qkv"], case["und_weights_q"], case["und_weights_k"],
        case["gen_weights_q"], case["gen_weights_k"], case["cos_sin_cache"],
        case["positions"], case["cat_indices"], hq, hk, hv,
        case["norm_eps"], case["mrope_section"])

    # ---- NPU ----
    k_cache_in, v_cache_in = case["k_cache"], case["v_cache"]
    k_cache = k_cache_in.clone().npu()
    v_cache = v_cache_in.clone().npu()
    q = und_gen_qkv_rms_norm_rope_cache(
        case["und_qkv"].npu(), case["und_weights_q"].npu(), case["und_weights_k"].npu(),
        case["cos_sin_cache"].npu(), k_cache, v_cache,
        case["slot_mapping"].npu(), case["positions"].npu(),
        _to_npu(case["gen_qkv"]), _to_npu(case["gen_weights_q"]), _to_npu(case["gen_weights_k"]),
        _to_npu(case["cat_indices"]),
        num_heads_q=hq, num_heads_k=hk, num_heads_v=hv,
        norm_eps=case["norm_eps"], mrope_section=case["mrope_section"] or [])
    torch.npu.synchronize()

    # ---- 接口连通性（硬断言）----
    assert q.shape == (total, hq, HEAD_DIM), f"q shape {tuple(q.shape)} != {(total, hq, HEAD_DIM)}"
    assert q.dtype == torch.bfloat16, f"q dtype {q.dtype}"
    assert k_cache.shape == k_cache_in.shape and v_cache.shape == v_cache_in.shape
    assert k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16

    # k_cache/v_cache 未被 slot_mapping 命中的位置必须保持入参原值
    slot = case["slot_mapping"]
    untouched = torch.ones(meta["num_blocks"] * meta["block_size"], dtype=torch.bool)
    untouched[slot] = False
    k_out_rows = k_cache.cpu().reshape(-1, hk, HEAD_DIM)
    k_in_rows = k_cache_in.reshape(-1, hk, HEAD_DIM)
    untouched_ok = torch.equal(k_out_rows[untouched], k_in_rows[untouched])

    if verbose:
        # 只打印偏离默认的维度，默认集的输出保持原样不被噪声淹没
        extra = []
        if meta["mrope_section"] != [16, 16, 16]:
            extra.append(f"sec={meta['mrope_section']}")
        if meta["cat"] != "shuffled":
            extra.append(f"cat={meta['cat']}")
        if meta["slot_mode"] != "shuffled":
            extra.append(f"slot={meta['slot_mode']}")
        if meta["seed"] != 7:
            extra.append(f"seed={meta['seed']}")
        if meta["norm_eps"] != 1e-6:
            extra.append(f"eps={meta['norm_eps']:g}")
        if meta["max_pos"] != 4096:
            extra.append(f"max_pos={meta['max_pos']}")
        print(f"  [{meta['name']:<22}] 接口 OK  T={total:<6} heads=({hq},{hk},{hv}) bs={meta['block_size']:<4} "
              f"未命中槽保持原值={untouched_ok}  {' '.join(extra)}")

    # ---- 数值精度（对 golden）----
    results = [
        compare("q", q.cpu(), ref_q),
        compare("k_cache", gather_cache_rows(k_cache.cpu(), slot), ref_k),
        compare("v_cache", gather_cache_rows(v_cache.cpu(), slot), ref_v),
    ]
    numeric_ok = all(ok for ok, _ in results) and untouched_ok
    if verbose:
        for _, msg in results:
            print(f"      {msg}")
        print(f"      -> 精度 {'PASS' if numeric_ok else 'FAIL'}")
    return numeric_ok


def main():
    parser = argparse.ArgumentParser(description="UndGenQkvRmsNormRopeCache torch 接口测试")
    parser.add_argument("--full", action="store_true", help="包含 T=64K 的大 case")
    parser.add_argument("--case", default=None, help="只跑指定 case 名")
    parser.add_argument("--generalized", action="store_true",
                        help="只跑泛化集（mrope_section / cat / slot / seed / eps / max_pos 的覆盖补充）")
    parser.add_argument("--all-sets", action="store_true", help="典型集 + 泛化集一起跑")
    args = parser.parse_args()

    print(f"device: {torch.npu.get_device_name(0)}, torch {torch.__version__}, "
          f"torch_npu {torch_npu.__version__}")
    if args.generalized:
        pool = list(GENERALIZED_CASES)
    elif args.all_sets:
        pool = list(TYPICAL_CASES) + list(GENERALIZED_CASES)
    else:
        pool = list(TYPICAL_CASES)
    specs = [s for s in pool if (args.case is None or s["name"] == args.case)]
    if args.case is None and not args.full:
        skipped = [s["name"] for s in specs if s["big"]]
        specs = [s for s in specs if not s["big"]]
        if skipped:
            print(f"跳过大 case（--full 开启）: {', '.join(skipped)}")
    assert specs, "没有匹配的 case"

    results = [run_case(s) for s in specs]
    print(f"\n接口连通性: {len(results)}/{len(results)} PASS")
    print(f"数值精度:   {sum(results)}/{len(results)} PASS"
          f"{'' if all(results) else '  <-- 预期全 PASS，出现 FAIL 即为回退'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

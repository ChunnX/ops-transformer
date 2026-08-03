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
"""把同目录 golden.py 与外部参考实现的 ``_reference`` 对拍，守住"golden 语义不漂移"。

golden.py 是本算子唯一的精度基准，它自己的正确性依据是"与需求来源的参考实现
（CUDA hpc 侧的 ``test_fused_qk_rmsnorm_mrope_store_kv_fp8.py``）逐比特一致"。
参考实现属于外部代码，不入本仓；本脚本按路径加载它，找不到就 SKIP。

用法：
  export UND_GEN_REF_FILE=/path/to/test_fused_qk_rmsnorm_mrope_store_kv_fp8.py
  python3 test_golden_vs_reference.py            # 默认 case 集（T ≤ 10752）
  python3 test_golden_vs_reference.py --full     # 追加 T=64K（峰值约 4GB 内存）

对拍范围与前提：
  * 只比 ``golden_dense`` 的 Q/K/V，即 KV Cache scatter 之前的中间结果。
    scatter 部分参考实现用的是 block_ids/num_seqlen/q_index 反推，与本算子的
    slot_mapping 方案接口不同，无法直接对拍。
  * 固定 ``qk_norm_policy=2``（先 RMSNorm 再 MRoPE）、``use_hadamard=False``，
    即本算子确定的语义；参考实现的其余分支不在本算子范围内。
  * 参考实现的 ``_mrope`` 无条件索引 ``mrope_section[1]/[2]``，不接受空列表。
    "空列表退化为 [D/2, 0, 0]"是本算子自己的约定，对拍时显式展开后再喂过去，
    因此该 case 验证的是"我们对 [] 的解释 == 参考实现对 [D/2,0,0] 的行为"。
"""

import argparse
import importlib.util
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import (  # noqa: E402
    TYPICAL_CASES,
    build_case,
    golden_dense,
)

DEFAULT_REF_ENV = "UND_GEN_REF_FILE"
SKIP_EXIT_CODE = 0


def load_reference(path):
    """按文件路径加载参考实现模块。

    参考实现顶部 ``import hpc``（CUDA 侧库），本机通常没有；这里只用它的
    ``_reference``，打个空桩即可。
    """
    sys.modules.setdefault("hpc", types.ModuleType("hpc"))
    spec = importlib.util.spec_from_file_location("und_gen_external_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "_reference"):
        raise AttributeError(f"{path} 中没有 _reference，检查文件是否正确")
    return module


def compare_one(spec, ref_module):
    case = build_case(spec)
    meta = case.pop("_meta")
    num_heads_q = case["num_heads_q"]
    num_heads_k = case["num_heads_k"]
    num_heads_v = case["num_heads_v"]
    head_dim = case["und_qkv"].shape[2]
    num_heads_total = num_heads_q + num_heads_k + num_heads_v

    ours = golden_dense(
        und_qkv=case["und_qkv"],
        gen_qkv=case["gen_qkv"],
        und_weights_q=case["und_weights_q"],
        und_weights_k=case["und_weights_k"],
        gen_weights_q=case["gen_weights_q"],
        gen_weights_k=case["gen_weights_k"],
        cos_sin_cache=case["cos_sin_cache"],
        positions=case["positions"],
        cat_indices=case["cat_indices"],
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        norm_eps=case["norm_eps"],
        mrope_section=case["mrope_section"],
    )

    # 参考实现吃 2D [T, hidden]，权重顺序 [und_q, und_k, gen_q, gen_k]
    ref = ref_module._reference(
        und_qkv=case["und_qkv"].reshape(-1, num_heads_total * head_dim),
        gen_qkv=case["gen_qkv"].reshape(-1, num_heads_total * head_dim),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        weights=[
            case["und_weights_q"],
            case["und_weights_k"],
            case["gen_weights_q"],
            case["gen_weights_k"],
        ],
        norm_eps=case["norm_eps"],
        positions=case["positions"],
        cos_sin=case["cos_sin_cache"],
        cat_indices=case["cat_indices"],
        mrope_section=(case["mrope_section"] or [head_dim // 2, 0, 0]),
        qk_norm_policy=2,
        use_hadamard=False,
    )

    details = []
    for name, got, want in zip(("q", "k", "v"), ours, ref):
        if got.shape != want.shape:
            details.append((name, False, float("nan"), f"shape {tuple(got.shape)} vs {tuple(want.shape)}"))
            continue
        exact = torch.equal(got, want)
        max_diff = (got.double() - want.double()).abs().max().item()
        details.append((name, exact, max_diff, ""))
    return meta, details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=os.environ.get(DEFAULT_REF_ENV),
                        help=f"参考实现文件路径，默认取环境变量 ${DEFAULT_REF_ENV}")
    parser.add_argument("--full", action="store_true", help="追加 T=64K 的大 case（内存约 4GB）")
    parser.add_argument("--case", default=None, help="只跑指定 case 名")
    args = parser.parse_args()

    if not args.ref or not os.path.isfile(args.ref):
        print(f"SKIP: 未找到参考实现文件（--ref 或 ${DEFAULT_REF_ENV} 未设置 / 路径不存在）。")
        print("      该文件为外部代码，不随本仓分发；没有它时本对拍无法进行。")
        return SKIP_EXIT_CODE

    ref_module = load_reference(args.ref)
    print(f"参考实现: {args.ref}")
    print(f"golden:  {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden.py')}\n")

    specs = [s for s in TYPICAL_CASES if args.full or not s["big"]]
    if args.case:
        specs = [s for s in specs if s["name"] == args.case]
        if not specs:
            print(f"没有名为 {args.case} 的 case")
            return 1

    failed = []
    print(f"{'case':26} {'结论':14} {'max|diff|':>12}")
    print("-" * 56)
    for spec in specs:
        meta, details = compare_one(spec, ref_module)
        all_exact = all(exact for _, exact, _, _ in details)
        worst = max((d for _, _, d, _ in details), default=float("nan"))
        name = meta["name"] + ("" if spec["mrope_section"] else " (空 section)")
        print(f"{name:26} {'逐比特一致' if all_exact else '不一致':14} {worst:12.3e}")
        if not all_exact:
            failed.append(meta["name"])
            for tensor_name, exact, diff, note in details:
                if not exact:
                    print(f"    - {tensor_name}: max|diff|={diff:.6e} {note}")

    skipped = [s["name"] for s in TYPICAL_CASES if s["big"] and not args.full]
    if skipped:
        print(f"\n未跑（需 --full）: {', '.join(skipped)}")

    if failed:
        print(f"\nFAILED: {len(failed)} 个 case 与参考实现不一致 -> {failed}")
        return 1
    print(f"\nPASSED: {len(specs)} 个 case 与参考实现逐比特一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

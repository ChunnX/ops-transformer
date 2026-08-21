# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""用例加载 + 参数规范化（MLA 版，布局限定 TND/NZ/NTD per R1）。"""

import itertools
import importlib
import torch
from typing import Dict, List

from core.data import MLA_D_TOTAL, MLA_NZ_BLK_ELEM


def load_case_modules(module_names: List[str]) -> Dict[str, dict]:
    """从 test_cases/*.py 加载 TestCases dict，展开所有参数组合。"""
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


def normalize_params(raw: dict) -> dict:
    """将 test_cases 原始字段补全为计算所需参数。

    MLA 约定（m0070 / R1）：
      - layout_q="TND"（cu_seqlens_q 必传），layout_kv="PA_NZ"（block_table + cache_seqlens 必传），layout_out="NTD"
      - q/k_cache 最后维 576；kv head num = 1（N2 无意义，恒 1）
      - cache_seqlens = 每 batch 的 KV cache 长度（等价 flash_attn 的 seqused_kv）
    """
    c = dict(raw)
    layout_q = c.get("layout_q", "TND")
    c["input_layout"] = layout_q
    c.setdefault("layout_kv", "PA_NZ")
    c.setdefault("layout_out", "NTD")
    c["N2"] = 1
    c.setdefault("D", MLA_D_TOTAL // 2)          # qk nope 宽 / head_dim_v
    c.setdefault("DV", c["D"])
    c.setdefault("DRope", MLA_D_TOTAL - c["D"])  # 64
    c.setdefault("mask_mode", 0)
    c.setdefault("block_size", 128)

    for key in ("cu_seqlens_q", "cache_seqlens", "seqused_q"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)

    # B / cu_seqlens_q 双向推导
    if "cu_seqlens_q" not in c:
        b = c.get("B", 1)
        s1 = c.get("S1", 1)
        c["cu_seqlens_q"] = list(range(0, b * s1 + 1, s1))
    c["B"] = len(c["cu_seqlens_q"]) - 1
    c.setdefault("seqused_q", [c["cu_seqlens_q"][i + 1] - c["cu_seqlens_q"][i]
                               for i in range(len(c["cu_seqlens_q"]) - 1)])

    # KV 侧：cache_seqlens（每 batch 的 cache 长度）
    if "cache_seqlens" not in c:
        s2 = c.get("S2", c.get("S1", 1))
        c["cache_seqlens"] = [s2] * c["B"]

    # PA block_table 生成（PA_NZ 必须；kv head num = 1）
    if c.get("block_table") is not None and isinstance(c.get("block_table"), list):
        bt_raw = c["block_table"]
        c["block_table"] = torch.tensor(bt_raw, dtype=torch.int32)
        c["num_blocks"] = int(max(max(row) for row in bt_raw)) + 1
    else:
        cache_lens = c["cache_seqlens"]
        block_size = c["block_size"]
        max_blk = max((s + block_size - 1) // block_size for s in cache_lens)
        block_table = torch.full((c["B"], max_blk), -1, dtype=torch.int32)
        block_idx = 0
        for i in range(c["B"]):
            b_blk = (cache_lens[i] + block_size - 1) // block_size
            for j in range(b_blk):
                block_table[i][j] = block_idx
                block_idx += 1
        c["block_table"] = block_table
        c["num_blocks"] = block_idx
        c["max_blocks"] = max_blk

    # NZ 几何: D 总宽 576 → D_nz_sub = 36 块 × 16 元素
    c["nz_blk_elem"] = MLA_NZ_BLK_ELEM
    c["D_nz_sub"] = MLA_D_TOTAL // MLA_NZ_BLK_ELEM
    return c


def resolve_case_ids(case_id_arg: str, all_cases: Dict) -> List[str]:
    """解析 --case_id 参数（同 flash_attn harness）。"""
    if case_id_arg == "all":
        return sorted(all_cases.keys())
    ids = [x.strip() for x in case_id_arg.split(",")]
    result = []
    missing = []
    for cid in ids:
        matches = [k for k in all_cases
                   if k.endswith(f"/{cid}") or k == cid
                   or k.rsplit("/", 1)[-1].startswith(f"{cid}_")]
        if matches:
            result.extend(matches)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] 以下 case 不存在: {missing}")
    return result
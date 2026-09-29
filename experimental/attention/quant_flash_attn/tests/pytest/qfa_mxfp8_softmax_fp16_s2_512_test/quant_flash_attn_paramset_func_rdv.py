#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
功能正确性 func_rdv 用例集（S2 基本块 512 几何，C1V1-C1V1-C2V2 流水）

分块几何：Q_BLOCK=128 / BASE_BLOCK=512 / SUBLOOP=256 / MM2_KSEG=128。
维度矩阵围绕 512 几何的三层边界设计：
  1. 基础冒烟：B1/G1/S1=128/S2=512（单基本块）
  2. S2 边值：MM2_KSEG=128 段边界（±1）× SUBLOOP=256 轮边界（±1）
     × BASE_BLOCK=512 块边界（±1）+ 多块（1024/1536/2048）+ 任意尾块
  3. S1 边值：128 Q 块边界 ±1 + 小值/大值
  4. GQA/MHA：G=2/3(非2幂)/4/8 × S2 单块/多块
  5. B/bN2 行分布：B=2/4/8 + per-batch 变长（step/mix/首尾 0/交替 0）
  6. 交叉组合：S1×S2 / S2×G / B×S2

LSE 语义（每用例 _LSE 孪生）：正常行 k·ln2+ln(rowsum)；padding 行 0；
  kv=0 批有效行 -inf；q=0 批全行 0
"""


# ===== seqused 模式助手 =====
def _sq(s, n):
    return [s] * n  # 全等长


def _sq_zf(s, n):
    return [0] + [s] * (n - 1)  # 首 batch=0


def _sq_zl(s, n):
    return [s] * (n - 1) + [0]  # 末 batch=0


def _sq_alt(s, n):
    return [s if i % 2 == 0 else 0 for i in range(n)]  # 交替 0


def _sq_step(n):
    return [i * 128 + 1 for i in range(n)]  # 递增不定长（128 步进跨 Q 块）


def _sq_mix(s, n):
    return [s] * n if n <= 1 else [1] + [s] * (n - 2) + [1]  # 两头窄中间宽


_BASE = []


def _a(name, b, nq, sq_q, sq_kv, nkv=1):
    _BASE.append(
        {
            "name": name,
            "B": b,
            "N_q": nq,
            "N_kv": nkv,
            "D": 128,
            "cu_seqlens_q": None,
            "cu_seqlens_kv": None,
            "seqused_q": sq_q,
            "seqused_kv": sq_kv,
            "enable_pa": False,
            "kv_cache_layout": None,
            "block_size": None,
            "mask_mode": 0,
            "q_scale_layout": "BNSD",
            "p_scale": 1.0,
            "enable_lse": False,
        }
    )


# ===== 1. 基础冒烟（单基本块 128×512：C1V1 两轮 + C2 四次 mmad + V2 整块）=====
_a("SMOKE_B1_S2_512", 1, 1, [128], [512])

# ===== 2. S2 边值扫描（QS128 固定；三层边界 ±1 + 多块 + 任意尾块）=====
_S2_EDGES = [
    # mmad 段（128）边界
    1,
    2,
    63,
    64,
    127,
    128,
    129,
    # subloop（256）轮边界：恰一轮 / 一轮+1 / 1.5 轮
    255,
    256,
    257,
    383,
    384,
    385,
    # 基本块（512）边界：尾块 1 行 / 恰一块 / 一块+1
    511,
    512,
    513,
    # 多块与任意尾块
    640,
    767,
    768,
    769,
    1024,
    1025,
    1536,
    2048,
    2049,
]
for _s2 in _S2_EDGES:
    _a(f"S2_{_s2}", 1, 1, [128], [_s2])

# ===== 3. S1 边值扫描（KVS512 固定；Q 块 128 边界 ±1 + 分形边界）=====
_S1_EDGES = [1, 31, 32, 63, 64, 127, 128, 129, 255, 256, 383, 384, 511, 512]
for _s1 in _S1_EDGES:
    _a(f"S1_{_s1}", 1, 1, [_s1], [512])

# ===== 4. GQA/MHA =====
for _g in [2, 3, 4, 8]:
    _a(f"GQA_G{_g}_S2_512", 1, _g, [128], [512])
    _a(f"GQA_G{_g}_S2_1024", 1, _g, [128], [1024])
_a("MHA_N2_S2_512", 1, 2, [128], [512], nkv=2)

# ===== 5. B/bN2 行分布 + per-batch 变长 =====
for _b in [2, 4, 8]:
    _a(f"B{_b}_S2_512", _b, 1, _sq(128, _b), _sq(512, _b))
    _a(f"B{_b}_S2_1024", _b, 1, _sq(128, _b), _sq(1024, _b))
_a("B4_VAR_STEP", 4, 1, _sq_step(4), _sq_step(4))
_a("B4_VAR_MIX", 4, 1, _sq_mix(512, 4), _sq_mix(512, 4))
_a("B4_KV_ZF", 4, 1, _sq(128, 4), _sq_zf(512, 4))
_a("B4_KV_ZL", 4, 1, _sq(128, 4), _sq_zl(512, 4))
_a("B4_KV_ALT", 4, 1, _sq(128, 4), _sq_alt(512, 4))
_a("B4_VAR_STEP_MIX", 4, 1, _sq_step(4), _sq_mix(1024, 4))

# ===== 6. 交叉组合 =====
for _s1 in [1, 64, 128, 256, 512]:
    for _s2 in [256, 512, 1024]:
        _a(f"C_S1_{_s1}_S2_{_s2}", 1, 1, [_s1], [_s2])
for _s2 in [1, 128, 512, 2048]:
    for _g in [2, 4, 8]:
        _a(f"C_S2_{_s2}_G_{_g}", 1, _g, [128], [_s2])
for _b in [2, 4, 8]:
    for _s2 in [1, 128, 512, 1024, 2048]:
        _a(f"C_B_{_b}_S2_{_s2}", _b, 1, _sq(128, _b), _sq(_s2, _b))

# ===== LSE 展开：每用例孪生双版本（非LSE + _LSE），成对相邻 =====
CASES = []
for _c in _BASE:
    CASES.append(_c)
    _twin = dict(_c)
    _twin["name"] = _c["name"] + "_LSE"
    _twin["enable_lse"] = True
    CASES.append(_twin)

# ===== 自校验：命名唯一 + seqused 长度 == B + 每对 LSE 孪生成对 =====
_names = [c["name"] for c in CASES]
assert len(_names) == len(set(_names)), "用例重名"
for _c in CASES:
    assert len(_c["seqused_q"]) == _c["B"] == len(_c["seqused_kv"]), _c["name"]
    assert max(_c["seqused_q"]) > 0 and max(_c["seqused_kv"]) > 0, (
        _c["name"] + " 全零批不可构造"
    )
    assert _c["N_q"] % _c["N_kv"] == 0, _c["name"]

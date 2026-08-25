#!/usr/bin/python
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
import pandas as pd
import os

EXCEL_PATH = "./excel/test_lse_full_coverage.xlsx"
SHEET_NAME = "Sheet1"

COLUMNS = [
    "Testcase_Prefix",
    "layout_query",
    "layout_kv",
    "q_type",
    "kv_type",
    "B",
    "T1",
    "T2",
    "S1",
    "S2",
    "N1",
    "N2",
    "D",
    "K",
    "scale_value",
    "key_quant_mode",
    "value_quant_mode",
    "sparse_block_size",
    "tile_size",
    "rope_head_dim",
    "sparse_mode",
    "attention_mode",
    "quant_scale_repo_mode",
    "block_size",
    "block_num",
    "actual_seq_q",
    "actual_seq_kv",
    "enable_sinks",
    "return_softmax_lse",
    "range_query",
    "range_key",
    "range_query_rope",
    "range_key_rope",
    "range_dequant_scale",
    "range_sinks",
]

SCALE = 0.041666666666666664
D = 512
K = 2048


def case(
    prefix,
    lq,
    lkv,
    qt,
    kvt,
    b,
    s1,
    s2,
    n1,
    sm=3,
    sinks=False,
    lse=True,
    am=2,
    bs=128,
    asq=None,
    askv=None,
    rq="[2,10]",
    rk="[-100,100]",
    rqr="[-1,1]",
    rkr="[-10,-2]",
    rds="[0,1]",
    rsink="[-0.2,0.2]",
):
    t1 = b if lq == "TND" else ""
    t2 = s2 if lkv == "TND" else ""
    if asq is None:
        asq = str([1] * b) if s1 == 1 else str(list(range(1, b + 1)))
    if askv is None:
        askv = str([s2] * b)
    kv_type_str = kvt if kvt else ""
    return {
        "Testcase_Prefix": prefix,
        "layout_query": lq,
        "layout_kv": lkv,
        "q_type": f"torch.{'bfloat16' if qt == 'bf16' else 'float16'}",
        "kv_type": kv_type_str,
        "B": b,
        "T1": t1,
        "T2": t2,
        "S1": s1,
        "S2": s2,
        "N1": n1,
        "N2": 1,
        "D": D,
        "K": K,
        "scale_value": SCALE,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "sparse_block_size": 1,
        "tile_size": 128,
        "rope_head_dim": 64,
        "sparse_mode": sm,
        "attention_mode": am,
        "quant_scale_repo_mode": 1,
        "block_size": bs,
        "block_num": "",
        "actual_seq_q": asq,
        "actual_seq_kv": askv,
        "enable_sinks": str(sinks),
        "return_softmax_lse": str(lse),
        "range_query": rq,
        "range_key": rk,
        "range_query_rope": rqr,
        "range_key_rope": rkr,
        "range_dequant_scale": rds,
        "range_sinks": rsink,
    }


rows = []

# ========== 一、Layout 组合泛化 (LSE=True) ==========
rows.append(case("lse_bsnd_bsnd", "BSND", "BSND", "bf16", "", 2, 1, 512, 16))
rows.append(
    case(
        "lse_tnd_tnd",
        "TND",
        "TND",
        "bf16",
        "",
        3,
        1,
        128,
        16,
        asq="[1,2,3]",
        askv="[32,64,128]",
    )
)
rows.append(case("lse_bsnd_pa", "BSND", "PA_BSND", "bf16", "", 2, 1, 512, 16, bs=128))
rows.append(
    case(
        "lse_tnd_pa",
        "TND",
        "PA_BSND",
        "bf16",
        "hifloat8",
        3,
        1,
        128,
        16,
        asq="[1,2,3]",
        askv="[32,64,128]",
    )
)

# ========== 二、N1 泛化 ==========
for n1 in [1, 2, 8, 32, 48, 64]:
    rows.append(case(f"lse_n1_{n1}", "BSND", "BSND", "bf16", "", 2, 1, 256, n1))

# ========== 三、q_type 泛化 ==========
rows.append(case("lse_fp16_bsnd", "BSND", "BSND", "fp16", "", 2, 1, 256, 16))
rows.append(
    case(
        "lse_fp16_tnd_pa",
        "TND",
        "PA_BSND",
        "fp16",
        "hifloat8",
        2,
        1,
        256,
        16,
        asq="[1,2]",
        askv="[128,256]",
    )
)

# ========== 四、kv_type 泛化 ==========
rows.append(
    case("lse_kv_int8", "BSND", "BSND", "bf16", "int8", 2, 1, 256, 16, rk="[-50,50]")
)
rows.append(
    case(
        "lse_kv_hif8",
        "TND",
        "PA_BSND",
        "bf16",
        "hifloat8",
        2,
        1,
        256,
        16,
        asq="[1,2]",
        askv="[128,256]",
    )
)

# ========== 五、sparse_mode 泛化 ==========
rows.append(case("lse_mode0_bsnd", "BSND", "BSND", "bf16", "", 2, 1, 256, 16, sm=0))
rows.append(
    case(
        "lse_mode0_tnd_pa",
        "TND",
        "PA_BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        sm=0,
        asq="[1,2]",
        askv="[128,256]",
    )
)

# ========== 六、sinks + LSE 组合 ==========
rows.append(
    case("lse_sinks_bsnd", "BSND", "BSND", "bf16", "", 2, 1, 256, 16, sinks=True)
)
rows.append(
    case(
        "lse_sinks_tnd",
        "TND",
        "TND",
        "bf16",
        "",
        3,
        1,
        128,
        16,
        sinks=True,
        asq="[1,2,3]",
        askv="[32,64,128]",
    )
)
rows.append(
    case(
        "lse_sinks_pa", "BSND", "PA_BSND", "bf16", "", 2, 1, 256, 16, sinks=True, bs=128
    )
)

# ========== 七、边界场景 ==========
# B=1 最小 batch
rows.append(case("lse_b1", "BSND", "BSND", "bf16", "", 1, 1, 512, 16))
# actual_seq_kv 含 0 (空 KV 行 -> emptyFlag -> LSE=0)
rows.append(
    case("lse_empty_kv", "BSND", "BSND", "bf16", "", 3, 1, 256, 16, askv="[0,128,256]")
)
# prefill S1>1
rows.append(
    case(
        "lse_prefill",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        4,
        256,
        16,
        asq="[2,4]",
        askv="[128,256]",
    )
)
# PA block_size=16 最小
rows.append(case("lse_pa_bs16", "BSND", "PA_BSND", "bf16", "", 2, 1, 256, 16, bs=16))

# ========== 八、无 LSE 基线对照 ==========
rows.append(
    case("nolse_bsnd_pa", "BSND", "PA_BSND", "bf16", "", 2, 1, 512, 16, lse=False)
)
rows.append(
    case(
        "nolse_tnd_sinks",
        "TND",
        "TND",
        "bf16",
        "",
        2,
        1,
        128,
        16,
        sinks=True,
        lse=False,
        asq="[1,2]",
        askv="[64,128]",
    )
)

df = pd.DataFrame(rows, columns=COLUMNS)
os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)
df.to_excel(EXCEL_PATH, index=False, sheet_name=SHEET_NAME)
print(f"Excel generated: {EXCEL_PATH}")
print(f"Total cases: {len(rows)}")
print()
for i, r in enumerate(rows):
    print(
        f"  [{i + 1:02d}] {r['Testcase_Prefix']:20s} | {r['layout_query']:4s}/{r['layout_kv']:7s} | "
        f"q={r['q_type']:20s} kv={r['kv_type'] or 'default':9s} | B={r['B']} S1={r['S1']} S2={r['S2']} "
        f"N1={r['N1']:2d} | sm={r['sparse_mode']} sinks={r['enable_sinks']:5s} lse={r['return_softmax_lse']:5s} "
        f"bs={r['block_size']}"
    )

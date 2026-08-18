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

EXCEL_PATH = "./excel/test_lse_comprehensive.xlsx"
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
    "Category",
    "Test_Focus",
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
    scale=SCALE,
    rq="[2,10]",
    rk="[-100,100]",
    rqr="[-1,1]",
    rkr="[-10,-2]",
    rds="[0,1]",
    rsink="[-0.2,0.2]",
    category="",
    focus="",
):
    t1 = b if lq == "TND" else ""
    t2 = s2 if lkv == "TND" else ""
    if asq is None:
        if lq == "TND":
            asq = str(list(range(1, b + 1)))
        else:
            asq = str([s1] * b)
    if askv is None:
        if lkv == "TND":
            askv = str(list(range(s2 // b, s2 + 1, s2 // b))) if b > 0 else str([s2])
            if len(eval(askv)) != b:
                askv = str([s2] * b)
        else:
            askv = str([s2] * b)
    kv_type_str = kvt if kvt else ""
    q_type_str = f"torch.{'bfloat16' if qt == 'bf16' else 'float16'}"
    return {
        "Testcase_Prefix": prefix,
        "layout_query": lq,
        "layout_kv": lkv,
        "q_type": q_type_str,
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
        "scale_value": scale,
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
        "Category": category,
        "Test_Focus": focus,
    }


rows = []

# =====================================================================
# 一、Layout 组合 × LSE 精度验证 (return_softmax_lse=True)
#    覆盖 BSND_BSND / TND_TND / BSND_PA / TND_PA 四种合法组合
# =====================================================================
rows.append(
    case(
        "lse_bsnd_bsnd",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        512,
        16,
        category="Layout",
        focus="BSND_BSND decode LSE精度",
    )
)
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
        category="Layout",
        focus="TND_TND 变长seq LSE精度",
    )
)
rows.append(
    case(
        "lse_bsnd_pa",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        2,
        1,
        512,
        16,
        bs=128,
        category="Layout",
        focus="BSND_PA decode LSE精度",
    )
)
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
        category="Layout",
        focus="TND_PA hif8 LSE精度",
    )
)

# =====================================================================
# 二、dtype 全矩阵 (q_type × kv_type)
#    q_type: bf16 / fp16
#    kv_type: hifloat8 / float8_e4m3fn(None) / int8
# =====================================================================
for qt in ["bf16", "fp16"]:
    for kvt, label in [("", "fp8e4m3"), ("hifloat8", "hif8"), ("int8", "int8")]:
        rk_val = "[-50,50]" if kvt == "int8" else "[-100,100]"
        rows.append(
            case(
                f"lse_dtype_{qt}_{label}",
                "BSND",
                "BSND",
                qt,
                kvt,
                2,
                1,
                256,
                16,
                rk=rk_val,
                category="Dtype",
                focus=f"{qt}×{label} dtype组合 LSE精度",
            )
        )

# =====================================================================
# 三、N1 全泛化 (g = N1/N2 = N1, 影响 LSE 输出 g 维度)
#    N1: 1/2/4/8/16/32/48/64
# =====================================================================
for n1 in [1, 2, 4, 8, 16, 32, 48, 64]:
    rows.append(
        case(
            f"lse_n1_{n1}",
            "BSND",
            "BSND",
            "bf16",
            "",
            2,
            1,
            256,
            n1,
            category="N1泛化",
            focus=f"N1={n1} g维度LSE精度",
        )
    )

# =====================================================================
# 四、sparse_mode × attention_mode 全组合
# =====================================================================
for sm in [0, 3]:
    for am in [0, 2]:
        rows.append(
            case(
                f"lse_sm{sm}_am{am}",
                "BSND",
                "BSND",
                "bf16",
                "",
                2,
                1,
                256,
                16,
                sm=sm,
                am=am,
                category="sparse×attn",
                focus=f"sparse_mode={sm} attention_mode={am}",
            )
        )

# =====================================================================
# 五、sinks 开关 × layout 组合 (验证 sinks 对 LSE 的影响)
# =====================================================================
rows.append(
    case(
        "lse_sinks_bsnd",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        sinks=True,
        category="Sinks",
        focus="BSND sinks LSE精度",
    )
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
        category="Sinks",
        focus="TND sinks LSE精度",
    )
)
rows.append(
    case(
        "lse_sinks_pa",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        sinks=True,
        bs=128,
        category="Sinks",
        focus="PA sinks LSE精度",
    )
)
rows.append(
    case(
        "lse_sinks_tnd_pa",
        "TND",
        "PA_BSND",
        "bf16",
        "hifloat8",
        2,
        1,
        256,
        16,
        sinks=True,
        asq="[1,2]",
        askv="[128,256]",
        category="Sinks",
        focus="TND_PA sinks+hif8 LSE精度",
    )
)

# =====================================================================
# 六、S2 序列长度泛化 (影响 LSE 数值大小/累加项数)
# =====================================================================
for s2 in [128, 256, 512, 1024, 2048]:
    rows.append(
        case(
            f"lse_s2_{s2}",
            "BSND",
            "BSND",
            "bf16",
            "",
            2,
            1,
            s2,
            16,
            category="S2泛化",
            focus=f"S2={s2} LSE累加项数影响",
        )
    )

# =====================================================================
# 七、PA block_size 泛化 (影响 tiling/内存访问)
# =====================================================================
for bs in [16, 32, 64, 128, 256]:
    rows.append(
        case(
            f"lse_pa_bs{bs}",
            "BSND",
            "PA_BSND",
            "bf16",
            "",
            2,
            1,
            256,
            16,
            bs=bs,
            category="block_size",
            focus=f"PA block_size={bs} LSE精度",
        )
    )

# =====================================================================
# 八、B (batch) 泛化
# =====================================================================
rows.append(
    case(
        "lse_b1",
        "BSND",
        "BSND",
        "bf16",
        "",
        1,
        1,
        512,
        16,
        category="Batch",
        focus="B=1 单batch LSE精度",
    )
)
rows.append(
    case(
        "lse_b4",
        "BSND",
        "BSND",
        "bf16",
        "",
        4,
        1,
        256,
        16,
        category="Batch",
        focus="B=4 多batch LSE精度",
    )
)
rows.append(
    case(
        "lse_b4_tnd",
        "TND",
        "TND",
        "bf16",
        "",
        4,
        1,
        128,
        16,
        asq="[1,2,3,4]",
        askv="[32,64,96,128]",
        category="Batch",
        focus="B=4 TND变长 LSE精度",
    )
)

# =====================================================================
# 九、Prefill (S1>1) 场景
# =====================================================================
rows.append(
    case(
        "lse_prefill_bsnd",
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
        category="Prefill",
        focus="S1=4 prefill LSE精度",
    )
)
rows.append(
    case(
        "lse_prefill_tnd",
        "TND",
        "TND",
        "bf16",
        "",
        3,
        4,
        128,
        16,
        asq="[2,4,6]",
        askv="[32,64,128]",
        category="Prefill",
        focus="TND S1=4 prefill LSE精度",
    )
)
rows.append(
    case(
        "lse_prefill_pa",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        2,
        4,
        256,
        16,
        asq="[2,4]",
        askv="[128,256]",
        bs=128,
        category="Prefill",
        focus="PA S1=4 prefill LSE精度",
    )
)

# =====================================================================
# 十、边界场景
# =====================================================================
# 空 KV (actual_seq_kv 含 0 -> gatherKV emptyFlag -> LSE=log(1e-10))
rows.append(
    case(
        "lse_empty_kv",
        "BSND",
        "BSND",
        "bf16",
        "",
        3,
        1,
        256,
        16,
        askv="[0,128,256]",
        category="边界",
        focus="空KV行 LSE=log(1e-10)",
    )
)
rows.append(
    case(
        "lse_empty_kv_tnd",
        "TND",
        "TND",
        "bf16",
        "",
        3,
        1,
        128,
        16,
        asq="[1,2,3]",
        askv="[0,64,128]",
        category="边界",
        focus="TND 空KV LSE精度",
    )
)
# sparse_mode=3 阈值<=0 (短KV场景)
rows.append(
    case(
        "lse_short_kv_sm3",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        64,
        16,
        sm=3,
        askv="[32,64]",
        category="边界",
        focus="sm3 短KV threshold边界",
    )
)
# 全相同 seq (uniform)
rows.append(
    case(
        "lse_uniform_seq",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        3,
        1,
        256,
        16,
        askv="[256,256,256]",
        bs=128,
        category="边界",
        focus="均匀seq PA LSE精度",
    )
)
# 大 batch TND 变长
rows.append(
    case(
        "lse_tnd_varlen",
        "TND",
        "PA_BSND",
        "bf16",
        "hifloat8",
        5,
        1,
        256,
        16,
        asq="[1,2,3,4,5]",
        askv="[32,64,96,128,256]",
        category="边界",
        focus="B=5 TND_PA 变长 LSE精度",
    )
)

# =====================================================================
# 十一、scale_value 泛化 (直接影响 LSE 数值大小)
# =====================================================================
rows.append(
    case(
        "lse_scale_small",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        scale=0.01,
        category="scale",
        focus="scale=0.01 小尺度 LSE",
    )
)
rows.append(
    case(
        "lse_scale_large",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        scale=0.2,
        category="scale",
        focus="scale=0.2 大尺度 LSE",
    )
)

# =====================================================================
# 十二、range 泛化 (影响数值精度/溢出)
# =====================================================================
rows.append(
    case(
        "lse_range_extreme",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        rk="[-500,500]",
        category="range",
        focus="极端key范围 LSE稳定性",
    )
)
rows.append(
    case(
        "lse_range_narrow",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        rk="[-1,1]",
        rq="[-1,1]",
        category="range",
        focus="窄范围 LSE精度",
    )
)
rows.append(
    case(
        "lse_range_sinks_extreme",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        sinks=True,
        rsink="[-2,2]",
        category="range",
        focus="极端sinks范围 LSE稳定性",
    )
)

# =====================================================================
# 十三、LSE 不影响主输出精度 对照组 (return_softmax_lse=False)
#     与 category=Layout 中的 lse=True 用例参数完全一致，仅 lse=False
#     用于验证 LSE 特性不影响 attention_out 精度
# =====================================================================
rows.append(
    case(
        "nolse_bsnd_bsnd",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        512,
        16,
        lse=False,
        category="LSE不影响主输出",
        focus="对照 lse_bsnd_bsnd",
    )
)
rows.append(
    case(
        "nolse_tnd_tnd",
        "TND",
        "TND",
        "bf16",
        "",
        3,
        1,
        128,
        16,
        lse=False,
        asq="[1,2,3]",
        askv="[32,64,128]",
        category="LSE不影响主输出",
        focus="对照 lse_tnd_tnd",
    )
)
rows.append(
    case(
        "nolse_bsnd_pa",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        2,
        1,
        512,
        16,
        bs=128,
        lse=False,
        category="LSE不影响主输出",
        focus="对照 lse_bsnd_pa",
    )
)
rows.append(
    case(
        "nolse_tnd_pa",
        "TND",
        "PA_BSND",
        "bf16",
        "hifloat8",
        3,
        1,
        128,
        16,
        lse=False,
        asq="[1,2,3]",
        askv="[32,64,128]",
        category="LSE不影响主输出",
        focus="对照 lse_tnd_pa",
    )
)
rows.append(
    case(
        "nolse_sinks_bsnd",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        16,
        lse=False,
        sinks=True,
        category="LSE不影响主输出",
        focus="对照 sinks场景",
    )
)
rows.append(
    case(
        "nolse_prefill_bsnd",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        4,
        256,
        16,
        lse=False,
        asq="[2,4]",
        askv="[128,256]",
        category="LSE不影响主输出",
        focus="对照 prefill场景",
    )
)
rows.append(
    case(
        "nolse_n1_64",
        "BSND",
        "BSND",
        "bf16",
        "",
        2,
        1,
        256,
        64,
        lse=False,
        category="LSE不影响主输出",
        focus="对照 N1=64场景",
    )
)
rows.append(
    case(
        "nolse_fp16_hif8",
        "TND",
        "PA_BSND",
        "fp16",
        "hifloat8",
        2,
        1,
        256,
        16,
        lse=False,
        asq="[1,2]",
        askv="[128,256]",
        category="LSE不影响主输出",
        focus="对照 fp16×hif8场景",
    )
)

# =====================================================================
# 十四、交叉组合 (多因素叠加)
# =====================================================================
rows.append(
    case(
        "lse_cross_sinks_prefill",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        2,
        4,
        512,
        32,
        sinks=True,
        bs=128,
        asq="[2,4]",
        askv="[256,512]",
        category="交叉组合",
        focus="sinks+prefill+PA+N1=32 LSE",
    )
)
rows.append(
    case(
        "lse_cross_fp16_int8_sinks",
        "BSND",
        "BSND",
        "fp16",
        "int8",
        2,
        1,
        512,
        48,
        sinks=True,
        rk="[-50,50]",
        category="交叉组合",
        focus="fp16+int8+sinks+N1=48 LSE",
    )
)
rows.append(
    case(
        "lse_cross_tnd_sm0_am0",
        "TND",
        "TND",
        "bf16",
        "",
        3,
        1,
        256,
        16,
        sm=0,
        am=0,
        asq="[1,2,3]",
        askv="[64,128,256]",
        category="交叉组合",
        focus="TND+sm0+am0 LSE",
    )
)
rows.append(
    case(
        "lse_cross_large",
        "BSND",
        "PA_BSND",
        "bf16",
        "",
        4,
        1,
        2048,
        64,
        bs=256,
        askv="[512,1024,1536,2048]",
        category="交叉组合",
        focus="大S2=2048+PA+N1=64 LSE",
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
        f"  [{i + 1:02d}] {r['Testcase_Prefix']:28s} | {r['layout_query']:4s}/{r['layout_kv']:7s} | "
        f"q={r['q_type'].replace('torch.', ''):6s} kv={r['kv_type'] or 'fp8e4m3':9s} | "
        f"B={r['B']} S1={r['S1']} S2={r['S2']:4d} N1={r['N1']:2d} | "
        f"sm={r['sparse_mode']} am={r['attention_mode']} sinks={r['enable_sinks']:5s} "
        f"lse={r['return_softmax_lse']:5s} bs={r['block_size']:3d} | "
        f"[{r['Category']}] {r['Test_Focus']}"
    )

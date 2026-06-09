#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import os

def make_case(name, layout_q, layout_kv, B, S1, S2, N1, N2,
              seqused_q=None, cu_seqlens_q=None,
              seqused_ori_kv=None, cu_seqlens_ori_kv=None,
              seqused_cmp_kv=None, cu_seqlens_cmp_kv=None,
              cmp_residual_kv=None,
              K=0, cmp_ratio=4, template_run_mode="CFA",
              cmp_mask_mode=3, block_size2=128):
    return {
        "Testcase_Name": name,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "q_type": "torch.bfloat16",
        "ori_kv_type": "torch.float8_e4m3fn",
        "cmp_kv_type": "torch.float8_e4m3fn",
        "B": B,
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "D": 512,
        "K": K,
        "block_num1": None,
        "block_num2": None,
        "block_size1": 128,
        "block_size2": block_size2,
        "seqused_q": seqused_q,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_ori_kv": seqused_ori_kv,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "softmax_scale": 0.04419417,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": cmp_mask_mode,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "kv_quant_mode": 1,
        "tile_size": 64,
        "rope_head_dim": 64,
        "template_run_mode": template_run_mode,
        "actlen_mode": "full",
        "S1EQS2": "FALSE",
        "topk_value_mode": 1,
        "return_softmax_lse": False,
    }


cases = []

# =====================================================================
# Case 1: 单batch分两个核 (decode, G=128, S1=2 -> ~2 cores)
# =====================================================================
cases.append(make_case(
    "case01_single_batch_2cores",
    "BSND", "PA_BBND",
    B=1, S1=2, S2=1024, N1=128, N2=1,
))

# =====================================================================
# Case 2: 单batch分满核 (decode, G=128, S1=64 -> 64 cores)
# =====================================================================
cases.append(make_case(
    "case02_single_batch_full_cores",
    "BSND", "PA_BBND",
    B=1, S1=64, S2=1024, N1=128, N2=1,
))

# =====================================================================
# Case 3: 单batch分满核多一个基本块 (S1=129 > 2*s1BaseSize=128)
# =====================================================================
cases.append(make_case(
    "case03_single_batch_full_plus1",
    "BSND", "PA_BBND",
    B=1, S1=129, S2=1024, N1=128, N2=1,
))

# =====================================================================
# Case 4: 多batch分满核 (B=4, S1=16, total=64 tokens -> 64 cores)
# =====================================================================
cases.append(make_case(
    "case04_multi_batch_full_cores",
    "BSND", "PA_BBND",
    B=4, S1=16, S2=1024, N1=128, N2=1,
))

# =====================================================================
# Case 5: 一个核上两组规约数据，中间有不分核完整S2计算
#   B=3: batch0 S2=2048(多tile需规约), batch1 S2=128(单tile完整), batch2 S2=2048
#   TND layout, cu_seqlens_q=[0,1,2,3], seqused_q=[1,1,1]
# =====================================================================
cases.append(make_case(
    "case05_two_reductions_with_full_s2",
    "TND", "PA_BBND",
    B=3, S1=1, S2=2048, N1=128, N2=1,
    seqused_q="[1, 1, 1]",
    cu_seqlens_q="[0, 1, 2, 3]",
    seqused_ori_kv="[2048, 128, 2048]",
    cu_seqlens_ori_kv="[0, 2048, 2176, 4224]",
    seqused_cmp_kv="[512, 32, 512]",
    cu_seqlens_cmp_kv="[0, 512, 544, 1056]",
    cmp_residual_kv="[0, 0, 0]",
))

# =====================================================================
# Case 6: 一个核上两组规约数据，中间有QS=0的batch
#   B=3: batch0 S2=2048, batch1 QS=0(no work), batch2 S2=2048
# =====================================================================
cases.append(make_case(
    "case06_two_reductions_with_qs0_batch",
    "TND", "PA_BBND",
    B=3, S1=1, S2=2048, N1=128, N2=1,
    seqused_q="[1, 0, 1]",
    cu_seqlens_q="[0, 1, 1, 2]",
    seqused_ori_kv="[2048, 2048, 2048]",
    cu_seqlens_ori_kv="[0, 2048, 4096, 6144]",
    seqused_cmp_kv="[512, 512, 512]",
    cu_seqlens_cmp_kv="[0, 512, 1024, 1536]",
    cmp_residual_kv="[0, 0, 0]",
))

# =====================================================================
# Case 7: 一个核上两组规约数据，中间有cmp_KVS=0的batch
#   B=3: batch0 S2=2048, batch1 cmp_kv=0, batch2 S2=2048
# =====================================================================
cases.append(make_case(
    "case07_two_reductions_with_cmpkvs0",
    "TND", "PA_BBND",
    B=3, S1=1, S2=2048, N1=128, N2=1,
    seqused_q="[1, 1, 1]",
    cu_seqlens_q="[0, 1, 2, 3]",
    seqused_ori_kv="[2048, 2048, 2048]",
    cu_seqlens_ori_kv="[0, 2048, 4096, 6144]",
    seqused_cmp_kv="[512, 0, 512]",
    cu_seqlens_cmp_kv="[0, 512, 512, 1024]",
    cmp_residual_kv="[0, 0, 0]",
))

# =====================================================================
# Case 8: 泛化 Q-TND, K-TND (CFA)
# =====================================================================
cases.append(make_case(
    "case08_tnd_tnd_cfa",
    "TND", "TND",
    B=1, S1=1, S2=1024, N1=128, N2=1,
    seqused_q="[1]",
    cu_seqlens_q="[0, 1]",
    seqused_ori_kv="[1024]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[256]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))

# =====================================================================
# Case 9: 泛化 Q-TND, K-PA_BBND (CFA) - 默认组合，验证FD
# =====================================================================
cases.append(make_case(
    "case09_tnd_pa_cfa",
    "TND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=128, N2=1,
    seqused_q="[1]",
    cu_seqlens_q="[0, 1]",
    seqused_ori_kv="[1024]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[256]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))

# =====================================================================
# Case 10: 泛化 Q-BSND, K-PA_BBND (CFA)
# =====================================================================
cases.append(make_case(
    "case10_bsnd_pa_cfa",
    "BSND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=128, N2=1,
))

# =====================================================================
# Case 11: TND padding场景 (slot=4, actual_q=1, actual_kv=256)
# =====================================================================
cases.append(make_case(
    "case11_tnd_padding",
    "TND", "PA_BBND",
    B=1, S1=4, S2=1024, N1=128, N2=1,
    seqused_q="[1]",
    cu_seqlens_q="[0, 4]",
    seqused_ori_kv="[256]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[64]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))

# =====================================================================
# Case 12: SCFA Q-BSND K-PA_BBND
# =====================================================================
cases.append(make_case(
    "case12_bsnd_pa_scfa",
    "BSND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=128, N2=1,
    K=64, template_run_mode="SCFA",
))

# =====================================================================
# Case 13: SCFA Q-TND K-PA_BBND
# =====================================================================
cases.append(make_case(
    "case13_tnd_pa_scfa",
    "TND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=128, N2=1,
    K=64, template_run_mode="SCFA",
    seqused_q="[1]",
    cu_seqlens_q="[0, 1]",
    seqused_ori_kv="[1024]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[256]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))


# =====================================================================
# N1=64 cases (G=64, SPLIT_G=0, no FD split)
# =====================================================================

# Case 14: N1=64 单batch分两个核
cases.append(make_case(
    "case14_n64_single_batch_2cores",
    "BSND", "PA_BBND",
    B=1, S1=2, S2=1024, N1=64, N2=1,
))

# Case 15: N1=64 单batch分满核
cases.append(make_case(
    "case15_n64_single_batch_full_cores",
    "BSND", "PA_BBND",
    B=1, S1=64, S2=1024, N1=64, N2=1,
))

# Case 16: N1=64 单batch分满核多一个基本块
cases.append(make_case(
    "case16_n64_single_batch_full_plus1",
    "BSND", "PA_BBND",
    B=1, S1=129, S2=1024, N1=64, N2=1,
))

# Case 17: N1=64 多batch分满核
cases.append(make_case(
    "case17_n64_multi_batch_full_cores",
    "BSND", "PA_BBND",
    B=4, S1=16, S2=1024, N1=64, N2=1,
))

# Case 18: N1=64 两组规约+完整S2
cases.append(make_case(
    "case18_n64_two_reductions_with_full_s2",
    "TND", "PA_BBND",
    B=3, S1=1, S2=2048, N1=64, N2=1,
    seqused_q="[1, 1, 1]",
    cu_seqlens_q="[0, 1, 2, 3]",
    seqused_ori_kv="[2048, 128, 2048]",
    cu_seqlens_ori_kv="[0, 2048, 2176, 4224]",
    seqused_cmp_kv="[512, 32, 512]",
    cu_seqlens_cmp_kv="[0, 512, 544, 1056]",
    cmp_residual_kv="[0, 0, 0]",
))

# Case 19: N1=64 两组规约+QS=0 batch
cases.append(make_case(
    "case19_n64_two_reductions_with_qs0",
    "TND", "PA_BBND",
    B=3, S1=1, S2=2048, N1=64, N2=1,
    seqused_q="[1, 0, 1]",
    cu_seqlens_q="[0, 1, 1, 2]",
    seqused_ori_kv="[2048, 2048, 2048]",
    cu_seqlens_ori_kv="[0, 2048, 4096, 6144]",
    seqused_cmp_kv="[512, 512, 512]",
    cu_seqlens_cmp_kv="[0, 512, 1024, 1536]",
    cmp_residual_kv="[0, 0, 0]",
))

# Case 20: N1=64 两组规约+cmpKVS=0
cases.append(make_case(
    "case20_n64_two_reductions_with_cmpkvs0",
    "TND", "PA_BBND",
    B=3, S1=1, S2=2048, N1=64, N2=1,
    seqused_q="[1, 1, 1]",
    cu_seqlens_q="[0, 1, 2, 3]",
    seqused_ori_kv="[2048, 2048, 2048]",
    cu_seqlens_ori_kv="[0, 2048, 4096, 6144]",
    seqused_cmp_kv="[512, 0, 512]",
    cu_seqlens_cmp_kv="[0, 512, 512, 1024]",
    cmp_residual_kv="[0, 0, 0]",
))

# Case 21: N1=64 Q-TND K-TND CFA
cases.append(make_case(
    "case21_n64_tnd_tnd_cfa",
    "TND", "TND",
    B=1, S1=1, S2=1024, N1=64, N2=1,
    seqused_q="[1]",
    cu_seqlens_q="[0, 1]",
    seqused_ori_kv="[1024]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[256]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))

# Case 22: N1=64 Q-TND K-PA_BBND CFA
cases.append(make_case(
    "case22_n64_tnd_pa_cfa",
    "TND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=64, N2=1,
    seqused_q="[1]",
    cu_seqlens_q="[0, 1]",
    seqused_ori_kv="[1024]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[256]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))

# Case 23: N1=64 Q-BSND K-PA_BBND CFA
cases.append(make_case(
    "case23_n64_bsnd_pa_cfa",
    "BSND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=64, N2=1,
))

# Case 24: N1=64 TND padding
cases.append(make_case(
    "case24_n64_tnd_padding",
    "TND", "PA_BBND",
    B=1, S1=4, S2=1024, N1=64, N2=1,
    seqused_q="[1]",
    cu_seqlens_q="[0, 4]",
    seqused_ori_kv="[256]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[64]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))

# Case 25: N1=64 SCFA Q-BSND K-PA_BBND
cases.append(make_case(
    "case25_n64_bsnd_pa_scfa",
    "BSND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=64, N2=1,
    K=64, template_run_mode="SCFA",
))

# Case 26: N1=64 SCFA Q-TND K-PA_BBND
cases.append(make_case(
    "case26_n64_tnd_pa_scfa",
    "TND", "PA_BBND",
    B=1, S1=1, S2=1024, N1=64, N2=1,
    K=64, template_run_mode="SCFA",
    seqused_q="[1]",
    cu_seqlens_q="[0, 1]",
    seqused_ori_kv="[1024]",
    cu_seqlens_ori_kv="[0, 1024]",
    seqused_cmp_kv="[256]",
    cu_seqlens_cmp_kv="[0, 256]",
    cmp_residual_kv="[0]",
))


# Create DataFrame and save
columns = [
    "Testcase_Name", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type",
    "B", "S1", "S2", "N1", "N2", "D", "K", "block_num1", "block_num2",
    "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
    "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right",
    "kv_quant_mode", "tile_size", "rope_head_dim", "template_run_mode",
    "actlen_mode", "S1EQS2", "topk_value_mode", "return_softmax_lse",
    "seqused_q", "cu_seqlens_q", "seqused_ori_kv", "seqused_cmp_kv",
    "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv",
]

df = pd.DataFrame(cases, columns=columns)

os.makedirs("excel", exist_ok=True)
output_path = "excel/fd_testcases.xlsx"
df.to_excel(output_path, index=False, sheet_name="fd_cases")
print(f"Excel saved to: {output_path}")
print(f"Total cases: {len(cases)}")
for c in cases:
    print(f"  {c['Testcase_Name']}: layout_q={c['layout_q']}, layout_kv={c['layout_kv']}, "
          f"B={c['B']}, S1={c['S1']}, S2={c['S2']}, N1={c['N1']}, N2={c['N2']}, "
          f"mode={c['template_run_mode']}, K={c['K']}")

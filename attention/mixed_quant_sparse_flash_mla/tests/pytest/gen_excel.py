#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import pandas as pd
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixed_quant_sparse_flash_mla_paramset import TEST_PARAMS, ENABLED_PARAMS

def to_excel_value(v):
    if v is None:
        return None
    if isinstance(v, torch.dtype):
        if v == torch.bfloat16:
            return "torch.bfloat16"
        elif v == torch.float8_e4m3fn:
            return "torch.float8_e4m3fn"
    return v

def expand_params(param_dict):
    keys = list(param_dict.keys())
    values_lists = []
    for k in keys:
        v = param_dict[k]
        if not isinstance(v, list):
            v = [v]
        values_lists.append(v)
    combos = []
    for combo in itertools.product(*values_lists):
        combos.append(dict(zip(keys, combo)))
    return combos

all_cases = []
case_idx = 0
for params in ENABLED_PARAMS:
    for combo in expand_params(params):
        row = {}
        for k, v in combo.items():
            row[k] = to_excel_value(v)
        row["Testcase_Name"] = None
        row["topk_value_mode"] = combo.get("topk_value_mode", 1)
        row["return_softmax_lse"] = combo.get("return_softmax_lse", False)
        row["ori_kv_topk_mode"] = combo.get("ori_kv_topk_mode", "fullK")
        row["cmp_kv_topk_mode"] = combo.get("cmp_kv_topk_mode", "fullK")
        row["ori_sparse_indices_mode"] = combo.get("ori_sparse_indices_mode", "full")
        row["cmp_sparse_indices_mode"] = combo.get("cmp_sparse_indices_mode", "full")
        all_cases.append(row)
        case_idx += 1

columns = [
    "Testcase_Name", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1",
    "N2", "D", "K", "K1", "block_num1", "block_num2", "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
    "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "quant_mode", "tile_size", "rope_head_dim",
    "template_run_mode", "actlen_mode", "S1EQS2", "topk_value_mode", "return_softmax_lse",
    "ori_kv_topk_mode", "cmp_kv_topk_mode", "ori_sparse_indices_mode", "cmp_sparse_indices_mode",
    "seqused_q", "cu_seqlens_q", "seqused_ori_kv", "seqused_cmp_kv",
    "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv",
]

df = pd.DataFrame(all_cases, columns=columns)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "excel", "testcase.xlsx")
df.to_excel(output_path, index=False, sheet_name="decode")
print(f"Generated {len(df)} test cases to {output_path}")
print(df[["template_run_mode", "layout_q", "layout_kv", "quant_mode", "K1", "K"]].to_string())

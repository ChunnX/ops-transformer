#!/usr/bin/python3
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ascend_ops

from compare import compare


NPU_MASK_SIZE = 2048
GENERALIZED_MODE_ENV = "ASCEND_OPS_FA_GENERALIZED"
BASE_TEST_CASES = [
    (1, 16, 1, 2, 1024, 128),
    (1, 16, 1, 1024, 1024, 128),
]
GENERALIZED_TEST_CASES = [
    (2, 16, 1, 2, 1024, 128),       # B
    (1, 16, 1, 64, 1024, 128),      # S1 / streamK row split
    (1, 16, 1, 2, 2048, 128),       # S2
    (1, 8, 1, 2, 1024, 128),        # N1
    (1, 16, 2, 2, 1024, 128),       # N2/GQA ratio
    (2, 32, 4, 2, 1024, 128),       # combined B/N1/N2
]

def get_generalized_mode():
    return os.getenv(GENERALIZED_MODE_ENV, "0")

def build_test_cases_gene():
    B_LIST = [1, 2, 4, 5, 7, 8, 48]
    N1_LIST = [1, 2, 4, 8, 16, 32, 64]
    N2_LIST = [1, 2, 4, 8, 16, 32, 64]
    S1_LIST = [1, 2, 4, 15, 37, 111, 1024, 3077, 4096, 8101, 8192]
    S2_LIST = [1, 2, 4, 15, 37, 111, 1024, 3077, 4096, 8101, 8192, 16384]
    d = 128
    my_cases = []
    for b in B_LIST:
        for n1 in N1_LIST:
            for n2 in N2_LIST:
                if n1 % n2 != 0:
                    continue
                for s1 in S1_LIST:
                    for s2 in S2_LIST:
                        my_cases.append((b, n1, n2, s1, s2, d))
    return my_cases

def build_test_cases():
    if get_generalized_mode() == "1":
        return BASE_TEST_CASES + GENERALIZED_TEST_CASES
    elif get_generalized_mode() == "2":
        return build_test_cases_gene()
    else:
        return BASE_TEST_CASES

def case_id(case):
    b, n1, n2, sq, skv, d = case
    return f"B_{b}_S1_{sq}_S2_{skv}_N1_{n1}_N2_{n2}_D_{d}"


TEST_CASES = build_test_cases()


def make_cpu_causal_mask(sq, skv):
    q_idx = torch.arange(sq, dtype=torch.int64).unsqueeze(1)
    kv_idx = torch.arange(skv, dtype=torch.int64).unsqueeze(0)
    return kv_idx <= q_idx + (skv - sq)


def make_npu_causal_mask():
    allowed_mask = torch.tril(torch.ones(NPU_MASK_SIZE, NPU_MASK_SIZE, dtype=torch.bool))
    return ~allowed_mask


@pytest.mark.parametrize("b,n1,n2,sq,skv,d", TEST_CASES, ids=[case_id(case) for case in TEST_CASES])
def test_fa_causal(b, n1, n2, sq, skv, d):
    print(f"testcase b = {b}, n1 = {n1}, n2 = {n2}, sq = {sq}, skv = {skv}, d = {d} start")
    if torch.npu.device_count() == 0:
        pytest.skip("NPU not available")

    torch.manual_seed(0)

    q = torch.rand((b, sq, n1, d), dtype=torch.bfloat16)
    k = torch.rand((b, skv, n2, d), dtype=torch.bfloat16)
    v = torch.rand((b, skv, n2, d), dtype=torch.bfloat16)
    scale_value = 1 / math.sqrt(d)

    cpu_mask = make_cpu_causal_mask(sq, skv)
    npu_mask = make_npu_causal_mask()
    enable_gqa = n1 != n2

    q_bnsd = q.transpose(1, 2)
    k_bnsd = k.transpose(1, 2)
    v_bnsd = v.transpose(1, 2)
    cpu_out = torch.nn.functional.scaled_dot_product_attention(
        q_bnsd, k_bnsd, v_bnsd, attn_mask=cpu_mask, scale=scale_value, enable_gqa=enable_gqa)
    cpu_out = cpu_out.transpose(1, 2).contiguous()

    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    mask_npu = npu_mask.to("npu")
    npu_out = ascend_ops.ops.flash_attn(
        q_npu, k_npu, v_npu, mask_npu, scale_value, True)
    torch.npu.synchronize()
    npu_out_cpu = npu_out.to("cpu")

    assert npu_out.is_npu
    assert npu_out_cpu.shape == cpu_out.shape
    assert npu_out_cpu.dtype == torch.bfloat16

    compare_passed = compare(cpu_out.to(torch.float).numpy().flatten(),
                             npu_out_cpu.to(torch.float).numpy().flatten())
    if compare_passed == True:
        print(f"testcase b = {b}, n1 = {n1}, n2 = {n2}, sq = {sq}, skv = {skv}, d = {d} Pass")
    else:
        print(f"testcase b = {b}, n1 = {n1}, n2 = {n2}, sq = {sq}, skv = {skv}, d = {d} Failed")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
TORCH_EXTENSION_DIR = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_DIR.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

from npu_ops_transformer.ops import npu_chunk_gated_delta_rule_C128

from compare_chunk_gdr_C128_cpu_golden import make_case_args, run_cpu_golden_case

_ = npu_chunk_gated_delta_rule_C128


def run_precision_test(params):
    # 解包参数
    B, seqlen, nk, nv, dk, dv, chunk_size, data_type = params
    print(f"params = {params}")
    if data_type != torch.bfloat16:
        raise ValueError(f"chunk_gated_delta_rule_C128 only supports torch.bfloat16, got {data_type}.")
    args = make_case_args({
        "batch_size": B,
        "seqlen": seqlen,
        "nk": nk,
        "nv": nv,
        "dk": dk,
        "dv": dv,
        "chunk_size": chunk_size,
        "atol": 0.005,
        "rtol": 0.005,
    })
    assert run_cpu_golden_case(args), "CPU golden / GDN NPU / C128 NPU three-way precision failed."

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import torch
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[5]
TORCH_EXTENSION_DIR = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_DIR.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

from npu_ops_transformer.ops import npu_tri_inverse
from tri_inverse_golden import compare_tri_inverse, make_cc_input, tri_inverse_golden
from test_tri_inverse_paramset import shape_to_tile_num


def run_precision_test(shape, seed):
    torch_npu.npu.set_device(0)
    tile_num = shape_to_tile_num(shape)
    cc_in = make_cc_input(tile_num, seed=seed, device="npu")
    expected = tri_inverse_golden(cc_in.cpu()).npu()
    actual = npu_tri_inverse(cc_in)
    torch_npu.npu.synchronize()

    metrics = compare_tri_inverse(actual, expected)
    print(f"shape={shape}, tile_num={tile_num}, seed={seed}, metrics={metrics}")
    assert metrics["passed"], metrics

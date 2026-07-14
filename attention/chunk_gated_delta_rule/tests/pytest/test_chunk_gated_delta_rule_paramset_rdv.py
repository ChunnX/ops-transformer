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

DEFAULT_CHUNK_SIZE = 64


# 用例
# name, B, T, Nv, Nk, Dv, Dk, has_g, scale, data_type, state_data_type
A5_REDLINE_CASES = [
    ("ARC-001", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-002", 1, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-003", 1, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-004", 2, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-005", 4, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-006", 1, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-007", 1, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-008", 1, 131072, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-009", 1, 262144, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-010", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-011", 1, 2048, 2, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-012", 1, 2048, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-013", 1, 2048, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-014", 1, 4096, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-015", 1, 8192, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-016", 32, 256000, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-017", 1, 6000, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-018", 80, 400000, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-019", 200, 51200, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-020", 1, 10240, 8, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-021", 1, 10240, 16, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-022", 1, 10240, 32, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-023", 1, 10240, 64, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-024", 1, 32768, 8, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-025", 1, 32768, 16, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-026", 1, 32768, 32, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-027", 1, 32768, 64, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-028", 4, 40960, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-029", 4, 40960, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-030", 4, 40960, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-031", 1, 32768, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-032", 1, 32768, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-033", 1, 32768, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-034", 1, 65535, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-035", 1, 65535, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-036", 1, 65535, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-037", 1, 32768, 48, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ARC-038", 1, 32768, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
]

A5_REDLINE_CASES_FP32 = [
    ("ARC-001-FP32", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-002-FP32", 1, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-003-FP32", 1, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-004-FP32", 2, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-005-FP32", 4, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-006-FP32", 1, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-007-FP32", 1, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-008-FP32", 1, 131072, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-009-FP32", 1, 262144, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-010-FP32", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-011-FP32", 1, 2048, 2, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-012-FP32", 1, 2048, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-013-FP32", 1, 2048, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-014-FP32", 1, 4096, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-015-FP32", 1, 8192, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-016-FP32", 32, 256000, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-017-FP32", 1, 6000, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-018-FP32", 80, 400000, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-019-FP32", 200, 51200, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-020-FP32", 1, 10240, 8, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-021-FP32", 1, 10240, 16, 4, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-022-FP32", 1, 10240, 32, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-023-FP32", 1, 10240, 64, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-024-FP32", 1, 32768, 8, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-025-FP32", 1, 32768, 16, 4, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-026-FP32", 1, 32768, 32, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-027-FP32", 1, 32768, 64, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-028-FP32", 4, 40960, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-029-FP32", 4, 40960, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-030-FP32", 4, 40960, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-031-FP32", 1, 32768, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-032-FP32", 1, 32768, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-033-FP32", 1, 32768, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-034-FP32", 1, 65535, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-035-FP32", 1, 65535, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-036-FP32", 1, 65535, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-037-FP32", 1, 32768, 48, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ARC-038-FP32", 1, 32768, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
]

A5_STC_CASES = [
    ("ASC-001", 2, 2 * 4 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-002", 4, 4 * 8 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-003", 1, 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-004", 2, 2 * 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-005", 3, 3 * 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-006", 4, 4 * 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-007", 1, 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-008", 2, 2 * 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-009", 3, 3 * 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-010", 4, 4 * 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-011", 1, 128 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-012", 1, 256 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-013", 1, 2 * 1024, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-014", 1, 2 * 1024, 2, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-015", 1, 2 * 1024, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-016", 1, 2 * 1024, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-017", 1, 4 * 1024, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-018", 1, 8 * 1024, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-019", 1, 128, 32, 32, 128, 64, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-020", 4, 4 * 512, 8, 8, 64, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-021", 2, 2 * 1024, 32, 4, 64, 64, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-022", 1, 4096, 64, 1, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-023", 3, 3 * 10000, 64, 16, 50, 50, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-024", 3, 3 * 10000, 32, 16, 8, 5, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-025", 2, 2 * 300, 8, 8, 64, 64, False, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-026", 4, 4 * 512, 8, 8, 1, 1, False, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-027", 4, 4 * 1024, 8, 8, 64, 64, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-028", 4, 4 * 512, 8, 4, 64, 64, False, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-029", 4, 4 * 512, 16, 4, 64, 64, False, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-030", 8, 8 * 4096, 64, 64, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-031", 4, 4 * 128, 16, 8, 31, 31, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-032", 8, 8 * 128, 2, 2, 16, 16, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-033", 1, 128, 3, 3, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16),
    ("ASC-034", 4, 4 * 2000, 2, 2, 16, 16, True, 1.0, torch.bfloat16, torch.bfloat16),
]

A5_STC_CASES_FP32 = [
    ("ASC-001-FP32", 2, 2 * 4 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-002-FP32", 4, 4 * 8 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-003-FP32", 1, 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-004-FP32", 2, 2 * 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-005-FP32", 3, 3 * 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-006-FP32", 4, 4 * 32 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-007-FP32", 1, 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-008-FP32", 2, 2 * 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-009-FP32", 3, 3 * 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-010-FP32", 4, 4 * 64 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-011-FP32", 1, 128 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-012-FP32", 1, 256 * 1024, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-013-FP32", 1, 2 * 1024, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-014-FP32", 1, 2 * 1024, 2, 2, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-015-FP32", 1, 2 * 1024, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-016-FP32", 1, 2 * 1024, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-017-FP32", 1, 4 * 1024, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-018-FP32", 1, 8 * 1024, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-019-FP32", 1, 128, 32, 32, 128, 64, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-020-FP32", 4, 4 * 512, 8, 8, 64, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-021-FP32", 2, 2 * 1024, 32, 4, 64, 64, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-022-FP32", 1, 4096, 64, 1, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-023-FP32", 3, 3 * 10000, 64, 16, 50, 50, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-024-FP32", 3, 3 * 10000, 32, 16, 8, 5, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-025-FP32", 2, 2 * 300, 8, 8, 64, 64, False, 1.0, torch.bfloat16, torch.float32),
    ("ASC-026-FP32", 4, 4 * 512, 8, 8, 1, 1, False, 1.0, torch.bfloat16, torch.float32),
    ("ASC-027-FP32", 4, 4 * 1024, 8, 8, 64, 64, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-028-FP32", 4, 4 * 512, 8, 4, 64, 64, False, 1.0, torch.bfloat16, torch.float32),
    ("ASC-029-FP32", 4, 4 * 512, 16, 4, 64, 64, False, 1.0, torch.bfloat16, torch.float32),
    ("ASC-030-FP32", 8, 8 * 4096, 64, 64, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-031-FP32", 4, 4 * 128, 16, 8, 31, 31, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-032-FP32", 8, 8 * 128, 2, 2, 16, 16, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-033-FP32", 1, 128, 3, 3, 128, 128, True, 1.0, torch.bfloat16, torch.float32),
    ("ASC-034-FP32", 4, 4 * 2000, 2, 2, 16, 16, True, 1.0, torch.bfloat16, torch.float32),
]


def _convert_cases(cases):
    result = []
    for _name, B, T, Nv, Nk, Dv, Dk, has_g, _scale, data_type, state_dtype in cases:
        seqlen = T // B
        result.append({
            "B": [B],
            "seqlen": [seqlen],
            "nk": [Nk],
            "nv": [Nv],
            "dk": [Dk],
            "dv": [Dv],
            "chunk_size": [DEFAULT_CHUNK_SIZE],
            "data_type": [data_type],
            "state_data_type": [state_dtype],
            "has_g": [has_g],
        })
    return result


GROUP_REDLINE_BF16 = _convert_cases(A5_REDLINE_CASES)
GROUP_REDLINE_FP32 = _convert_cases(A5_REDLINE_CASES_FP32)
GROUP_STC_BF16 = _convert_cases(A5_STC_CASES)
GROUP_STC_FP32 = _convert_cases(A5_STC_CASES_FP32)


# ENABLED_PARAMS_RDV = GROUP_REDLINE_BF16 + GROUP_REDLINE_FP32 + GROUP_STC_BF16 + GROUP_STC_FP32
# fp32 待启用后开启
ENABLED_PARAMS_RDV = GROUP_REDLINE_BF16 + GROUP_STC_BF16


# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu
from cann_ops_transformer.ops import compressor


@pytest.mark.ci
@pytest.mark.parametrize("cache_mode", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [4, 8])
def test_omitted_none_and_zero_ape(cache_mode, dtype, seq_len):
    torch_npu.npu.set_device(0)
    torch.manual_seed(73)
    x = torch.randn((1, seq_len, 4096), dtype=dtype, device="npu") * 0.1
    wkv = torch.randn((1024, 4096), dtype=dtype, device="npu") * 0.1
    wgate = torch.randn_like(wkv) * 0.1
    state = torch.zeros((2, 128, 2048), dtype=torch.float32, device="npu")
    table = torch.tensor(
        [[1]] if cache_mode == 1 else [0], dtype=torch.int32, device="npu"
    )
    start = torch.zeros(1, dtype=torch.int32, device="npu")
    options = {
        "cmp_ratio": 4,
        "coff": 2,
        "cache_mode": cache_mode,
        "state_block_table": table,
        "start_pos": start,
    }
    results = []
    states = []
    for ape_options in (
        {},
        {"ape": None},
        {"ape": torch.zeros((4, 1024), dtype=torch.float32, device="npu")},
    ):
        current_state = state.clone()
        results.append(
            compressor(x, wkv, wgate, current_state, **ape_options, **options).cpu()
        )
        states.append(current_state.cpu())
    for result, updated_state in zip(results[1:], states[1:]):
        torch.testing.assert_close(result, results[0], rtol=0, atol=0)
        torch.testing.assert_close(updated_state, states[0], rtol=0, atol=0)

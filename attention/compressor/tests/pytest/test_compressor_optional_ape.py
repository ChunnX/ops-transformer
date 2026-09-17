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
from compressor_golden import run_compressor_eager


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


@pytest.mark.ci
@pytest.mark.parametrize("has_ape", [True, False])
@pytest.mark.parametrize(
    "layout_th, seq_len, capacity",
    [
        (False, 4, 11),
        (False, 8, 15),
        (False, 8, 128),
        (True, 8, 15),
    ],
)
def test_ring_mapping_golden(has_ape, layout_th, seq_len, capacity):
    start = capacity * 3 - 2
    run_compressor_eager(
        2,
        start + seq_len,
        512,
        2,
        4,
        layout_th,
        S=seq_len,
        start_pos=[start, start],
        cu_seqlens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32)
        if layout_th
        else None,
        cache_mode=2,
        ring_block_size=capacity,
        ring_block_num=8,
        ring_block_table=[3, 7],
        ape_datarange=[-1, 1] if has_ape else None,
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    "seq_len, capacity, blocks, mapping",
    [
        (8, 14, 2, [0, 1]),
        (1018, None, 2, [0, 1]),
        (8, 15, 1, [0, 1]),
        (8, 15, 2, [0, 2]),
        (8, 15, 2, [1, 1]),
    ],
)
def test_ring_harness_rejects_invalid(seq_len, capacity, blocks, mapping):
    with pytest.raises(ValueError, match="ring"):
        run_compressor_eager(
            2,
            seq_len,
            512,
            2,
            4,
            False,
            S=seq_len,
            cache_mode=2,
            ring_block_size=capacity,
            ring_block_num=blocks,
            ring_block_table=mapping,
        )


@pytest.mark.ci
@pytest.mark.parametrize("with_seqused", [False, True])
def test_bsh_ring_capacity_rejected(with_seqused):
    torch_npu.npu.set_device(0)
    x = torch.zeros((1, 8, 4096), dtype=torch.bfloat16, device="npu")
    weight = torch.zeros((1024, 4096), dtype=x.dtype, device="npu")
    state = torch.zeros((1, 14, 2048), dtype=torch.float32, device="npu")
    table = torch.zeros(1, dtype=torch.int32, device="npu")
    used = torch.ones(1, dtype=torch.int32, device="npu") if with_seqused else None
    with pytest.raises(RuntimeError):
        compressor(
            x,
            weight,
            weight,
            state,
            cmp_ratio=4,
            coff=2,
            cache_mode=2,
            state_block_table=table,
            seqused=used,
        )

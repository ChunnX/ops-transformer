# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import torch_npu
import custom_ops  # noqa: F401


DEVICE_ID = 0
DEVICE = f"npu:{DEVICE_ID}"
ROPE_HEAD_DIM = 64
BLOCK_SIZE = 128
HIDDEN_SIZE = 1024
NORM_EPS = 1e-6
ROTARY_MODE = 2
CACHE_MODE = 1
DTYPE = torch.bfloat16


def _rand(shape, dtype=DTYPE, scale=0.01):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(dtype).to(DEVICE)


def _zeros(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype, device=DEVICE)


def _make_common_tensors(batch_size, seq_len, head_dim, coff, cmp_ratio):
    out_dim = head_dim * coff * 2
    return {
        "wkv": _rand((head_dim * coff, HIDDEN_SIZE)),
        "wgate": _rand((head_dim * coff, HIDDEN_SIZE)),
        "ape": _rand((cmp_ratio, head_dim * coff), dtype=torch.float32),
        "norm_weight": torch.ones((head_dim,), dtype=torch.float32, device=DEVICE),
        "rope_sin": _rand(
            (batch_size, (seq_len + cmp_ratio - 1) // cmp_ratio, ROPE_HEAD_DIM),
            dtype=torch.float32,
        ),
        "rope_cos": _rand(
            (batch_size, (seq_len + cmp_ratio - 1) // cmp_ratio, ROPE_HEAD_DIM),
            dtype=torch.float32,
        ),
        "state_cache": _zeros((batch_size, BLOCK_SIZE, out_dim), dtype=torch.float32),
        "state_block_table": torch.arange(
            batch_size, dtype=torch.int32, device=DEVICE
        ).view(batch_size, 1),
    }


def _run_compressor(
    x,
    common,
    head_dim,
    coff,
    cmp_ratio,
    seqused,
    start_pos,
    x_cache=None,
    x_cache_slot_mapping=None,
    x_cache_update=True,
):
    return torch.ops.custom.compressor(
        x,
        common["wkv"],
        common["wgate"],
        common["state_cache"],
        common["ape"],
        common["norm_weight"],
        common["rope_sin"],
        common["rope_cos"],
        rope_head_dim=ROPE_HEAD_DIM,
        cmp_ratio=cmp_ratio,
        state_block_table=common["state_block_table"],
        seqused=seqused,
        start_pos=start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        coff=coff,
        norm_eps=NORM_EPS,
        rotary_mode=ROTARY_MODE,
        cache_mode=CACHE_MODE,
        x_cache_update=x_cache_update,
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    "head_dim,coff,cmp_ratio",
    [
        pytest.param(512, 2, 4, id="c4a"),
        pytest.param(128, 2, 4, id="c4li"),
        pytest.param(512, 1, 128, id="c128a"),
    ],
)
def test_grouped_decode_x_cache_update_only(head_dim, coff, cmp_ratio):
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 2
    x = _rand((batch_size, 1, HIDDEN_SIZE))
    common = _make_common_tensors(batch_size, 1, head_dim, coff, cmp_ratio)
    seqused = torch.ones((batch_size,), dtype=torch.int32, device=DEVICE)
    start_pos = torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE)
    x_cache = _zeros((batch_size, cmp_ratio, HIDDEN_SIZE), dtype=DTYPE)
    x_cache_slot_mapping = torch.tensor(
        [[0], [cmp_ratio]], dtype=torch.int32, device=DEVICE
    )

    _run_compressor(
        x,
        common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=True,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(x_cache[0, 0].cpu(), x[0, 0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(x_cache[1, 0].cpu(), x[1, 0].cpu(), rtol=0, atol=0)


@pytest.mark.ci
@pytest.mark.parametrize(
    "head_dim,coff,cmp_ratio",
    [
        (512, 2, 4),
        (512, 1, 128),
    ],
)
def test_grouped_decode_x_cache_updates_all_valid_mtp_tokens(head_dim, coff, cmp_ratio):
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 2
    seq_len = 4
    cache_len = cmp_ratio + seq_len - 1
    x = _rand((batch_size, seq_len, HIDDEN_SIZE))
    common = _make_common_tensors(batch_size, seq_len, head_dim, coff, cmp_ratio)
    seqused = torch.tensor([seq_len, 2], dtype=torch.int32, device=DEVICE)
    start_pos = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    positions = start_pos.view(-1, 1) + torch.arange(
        seq_len, dtype=torch.int32, device=DEVICE
    )
    x_cache_slot_mapping = (
        torch.arange(batch_size, dtype=torch.int32, device=DEVICE).view(-1, 1)
        * cache_len
        + positions % cache_len
    )
    x_cache = _zeros((batch_size, cache_len, HIDDEN_SIZE), dtype=DTYPE)

    _run_compressor(
        x,
        common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=True,
    )
    torch_npu.npu.synchronize()

    for batch_idx, valid_tokens in enumerate(seqused.cpu().tolist()):
        for seq_idx in range(valid_tokens):
            slot = int(positions[batch_idx, seq_idx].item() % cache_len)
            torch.testing.assert_close(
                x_cache[batch_idx, slot].cpu(),
                x[batch_idx, seq_idx].cpu(),
                rtol=0,
                atol=0,
            )


@pytest.mark.ci
def test_grouped_decode_li_inactive_round_is_read_only():
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 4
    head_dim = 128
    coff = 2
    cmp_ratio = 4
    x = _rand((batch_size, 1, HIDDEN_SIZE))
    common = _make_common_tensors(batch_size, 1, head_dim, coff, cmp_ratio)
    seqused = torch.ones((batch_size,), dtype=torch.int32, device=DEVICE)
    start_pos = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)
    x_cache = _rand((batch_size, cmp_ratio, HIDDEN_SIZE))
    x_cache_before = x_cache.clone()
    state_cache_before = common["state_cache"].clone()
    x_cache_slot_mapping = torch.tensor(
        [[0], [5], [8], [13]], dtype=torch.int32, device=DEVICE
    )

    _run_compressor(
        x,
        common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=False,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(x_cache.cpu(), x_cache_before.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        common["state_cache"].cpu(), state_cache_before.cpu(), rtol=0, atol=0
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    "head_dim,coff,cmp_ratio,start_pos",
    [
        pytest.param(512, 2, 4, 3, id="c4a"),
        pytest.param(128, 2, 4, 3, id="c4li"),
        pytest.param(512, 1, 128, 125, id="c128a"),
    ],
)
def test_grouped_decode_mtp_cross_boundary_matches_legacy(
    head_dim, coff, cmp_ratio, start_pos
):
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 1
    seq_len = 4
    cache_len = cmp_ratio + seq_len - 1
    prev_x = _rand((batch_size, start_pos, HIDDEN_SIZE))
    cur_x = _rand((batch_size, seq_len, HIDDEN_SIZE))
    full_group_x = torch.cat([prev_x, cur_x[:, : cmp_ratio - start_pos]], dim=1)

    grouped_common = _make_common_tensors(
        batch_size, seq_len, head_dim, coff, cmp_ratio
    )
    legacy_common = {
        name: tensor.clone() if torch.is_tensor(tensor) else tensor
        for name, tensor in grouped_common.items()
    }
    x_cache = _zeros((batch_size, cache_len, HIDDEN_SIZE), dtype=DTYPE)
    x_cache[:, :start_pos] = prev_x
    positions = start_pos + torch.arange(seq_len, dtype=torch.int32, device=DEVICE)

    grouped_out = _run_compressor(
        cur_x,
        grouped_common,
        head_dim,
        coff,
        cmp_ratio,
        torch.full((batch_size,), seq_len, dtype=torch.int32, device=DEVICE),
        torch.full((batch_size,), start_pos, dtype=torch.int32, device=DEVICE),
        x_cache=x_cache,
        x_cache_slot_mapping=positions.view(1, -1),
        x_cache_update=True,
    )
    legacy_out = _run_compressor(
        full_group_x,
        legacy_common,
        head_dim,
        coff,
        cmp_ratio,
        torch.full((batch_size,), cmp_ratio, dtype=torch.int32, device=DEVICE),
        torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE),
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(
        grouped_out.cpu(), legacy_out.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        grouped_common["state_cache"].cpu(),
        legacy_common["state_cache"].cpu(),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.ci
def test_grouped_decode_li_mixed_phase_keeps_active_batch():
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 2
    head_dim = 128
    coff = 2
    cmp_ratio = 4
    x = _rand((batch_size, 1, HIDDEN_SIZE))
    common = _make_common_tensors(batch_size, 1, head_dim, coff, cmp_ratio)
    seqused = torch.ones((batch_size,), dtype=torch.int32, device=DEVICE)
    start_pos = torch.tensor([0, cmp_ratio - 1], dtype=torch.int32, device=DEVICE)
    x_cache = _rand((batch_size, cmp_ratio, HIDDEN_SIZE))
    x_cache_slot_mapping = torch.tensor(
        [[0], [2 * cmp_ratio - 1]], dtype=torch.int32, device=DEVICE
    )

    _run_compressor(
        x,
        common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=False,
    )
    torch_npu.npu.synchronize()

    inactive_state = common["state_cache"][0].cpu()
    active_state = common["state_cache"][1].cpu()
    torch.testing.assert_close(
        inactive_state, torch.zeros_like(inactive_state), rtol=0, atol=0
    )
    assert torch.count_nonzero(active_state).item() > 0


@pytest.mark.ci
@pytest.mark.parametrize(
    "head_dim,coff,cmp_ratio",
    [
        pytest.param(512, 2, 4, id="c4a"),
        pytest.param(128, 2, 4, id="c4li"),
    ],
)
def test_grouped_decode_x_cache_read_matches_legacy(head_dim, coff, cmp_ratio):
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 1
    prev_x = _rand((batch_size, cmp_ratio - 1, HIDDEN_SIZE))
    cur_x = _rand((batch_size, 1, HIDDEN_SIZE))
    full_x = torch.cat([prev_x, cur_x], dim=1)

    grouped_common = _make_common_tensors(batch_size, 1, head_dim, coff, cmp_ratio)
    legacy_common = {
        name: tensor.clone() if torch.is_tensor(tensor) else tensor
        for name, tensor in grouped_common.items()
    }

    grouped_seqused = torch.ones((batch_size,), dtype=torch.int32, device=DEVICE)
    grouped_start_pos = torch.full(
        (batch_size,), cmp_ratio - 1, dtype=torch.int32, device=DEVICE
    )
    legacy_seqused = torch.full(
        (batch_size,), cmp_ratio, dtype=torch.int32, device=DEVICE
    )
    legacy_start_pos = torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE)
    # LI runs after KV has written the current token, so its read-only call sees
    # the complete group in x_cache.
    x_cache = full_x.clone()
    x_cache_slot_mapping = torch.full(
        (batch_size, 1), cmp_ratio - 1, dtype=torch.int32, device=DEVICE
    )

    grouped_out = _run_compressor(
        cur_x,
        grouped_common,
        head_dim,
        coff,
        cmp_ratio,
        grouped_seqused,
        grouped_start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=False,
    )
    legacy_out = _run_compressor(
        full_x,
        legacy_common,
        head_dim,
        coff,
        cmp_ratio,
        legacy_seqused,
        legacy_start_pos,
        x_cache_update=True,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(
        grouped_out.cpu(), legacy_out.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        grouped_common["state_cache"].cpu(),
        legacy_common["state_cache"].cpu(),
        rtol=5e-2,
        atol=5e-2,
    )

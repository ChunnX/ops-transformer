# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms and conditions of the CANN Open Software License Agreement Version 2.0
# (the "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root
# of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Test: compressor supports state_cache axis-0 non-contiguous
# Approach: construct non-contiguous state_cache via as_strided (padding on dim0),
#           compare NPU output and state update between contiguous and non-contiguous.
# -----------------------------------------------------------------------------------------------------------

import sys
import copy
import random
import numpy as np
import torch
import torch_npu
import pytest

sys.path.insert(0, '/home/tjj/ops-transform-IgnoreContiguous/attention/compressor/tests/pytest')
import compressor_golden
from compressor_golden import check_result, cpu_compressor
from cann_ops_transformer.ops.compressor import compressor, CacheMode

DEVICE_ID = 0
np.random.seed(42)
random.seed(42)


def generate_inputs(
    B, S_max, head_dim, coff, cmp_ratio, bs_combine_flag,
    S=0, start_pos=None, seqused=None, cu_seqlens=None,
    block_size=128, data_type=torch.bfloat16, hidden_size=4096,
    cache_mode=1,
    x_datarange=(-10, 10), wkv_datarange=(-10, 10), wgate_datarange=(-10, 10),
    ape_datarange=(-10, 10), kv_state_datarange=(-10, 10), score_state_datarange=(-10, 10),
):
    """Generate all compressor inputs and CPU golden. Returns dict of all data."""
    torch_npu.npu.set_device(DEVICE_ID)

    if bs_combine_flag:
        if cu_seqlens is None:
            T = B * S
            cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        if seqused is not None:
            S_eff = max(seqused)
        else:
            S_eff = 0
            for i in range(B):
                s = cu_seqlens[i + 1] - cu_seqlens[i]
                if s > S_eff:
                    S_eff = s
        S = S_eff
    else:
        cu_seqlens = None

    exist_start_pos = True
    if start_pos is None:
        exist_start_pos = False
        start_pos = [0] * B
    else:
        start_pos = list(start_pos)

    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)

    # ---- gen block_table & state ----
    if cache_mode == 1:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
        for i in range(B):
            cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
            cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
            if start_pos[i] % cmp_ratio == 0:
                cur_end = start_pos[i]
            cur_end = min(cur_end, start_pos[i] + S)
            cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
            cur_end_block_id = (cur_end - 1) // block_size
            for j in range(cur_start_block_id, cur_end_block_id + 1):
                if next_block_id < block_num:
                    block_table[i][j] = next_block_id
                    next_block_id += 1
            end_pos = S
            if seqused is not None:
                end_pos = int(seqused[i])
            next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
            next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
            if (start_pos[i] + end_pos) % cmp_ratio == 0:
                next_end = start_pos[i] + end_pos
            next_end = min(next_end, start_pos[i] + end_pos)
            next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
            next_end_block_id = (next_end - 1) // block_size
            for j in range(next_start_block_id, next_end_block_id + 1):
                if next_block_id < block_num and block_table[i][j] == 0:
                    block_table[i][j] = next_block_id
                    next_block_id += 1

        kv_state = torch.tensor(np.random.uniform(
            kv_state_datarange[0], kv_state_datarange[1],
            (block_num, block_size, coff * head_dim)
        )).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(
            score_state_datarange[0], score_state_datarange[1],
            (block_num, block_size, coff * head_dim)
        )).to(torch.float32)
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        block_size = min((2 * cmp_ratio + S - 1) if coff == 2 else (cmp_ratio + S - 1), 1024)
        kv_state = torch.tensor(np.random.uniform(
            kv_state_datarange[0], kv_state_datarange[1],
            (B, block_size, coff * head_dim)
        )).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(
            score_state_datarange[0], score_state_datarange[1],
            (B, block_size, coff * head_dim)
        )).to(torch.float32)

    # ---- gen other inputs ----
    if bs_combine_flag:
        x_shape = (int(cu_seqlens[-1]), hidden_size)
    else:
        x_shape = (B, S, hidden_size)

    x = torch.tensor(np.random.uniform(x_datarange[0], x_datarange[1], x_shape)).to(data_type)
    wkv = torch.tensor(np.random.uniform(
        wkv_datarange[0], wkv_datarange[1], (coff * head_dim, hidden_size)
    )).to(data_type)
    wgate = torch.tensor(np.random.uniform(
        wgate_datarange[0], wgate_datarange[1], (coff * head_dim, hidden_size)
    )).to(data_type)
    ape = torch.tensor(np.random.uniform(
        ape_datarange[0], ape_datarange[1], (cmp_ratio, coff * head_dim)
    )).to(torch.float32)

    # ---- CPU golden ----
    cpu_kv_state = kv_state.clone()
    cpu_score_state = score_state.clone()
    update_kv = torch.zeros(cpu_kv_state.shape, dtype=torch.bool)
    update_score = torch.zeros(cpu_score_state.shape, dtype=torch.bool)
    cpu_out, cmp_kv_mask = cpu_compressor(
        x, wkv, wgate, cpu_kv_state, cpu_score_state, update_kv, update_score, ape,
        block_table=block_table, cu_seqlens=cu_seqlens, seqused=seqused, start_pos=torch.tensor(start_pos),
        cmp_ratio=cmp_ratio, coff=coff, cache_mode=cache_mode,
    )

    return {
        'x': x, 'wkv': wkv, 'wgate': wgate, 'ape': ape,
        'kv_state': kv_state, 'score_state': score_state,
        'block_table': block_table, 'start_pos': start_pos,
        'cu_seqlens': cu_seqlens, 'seqused': seqused,
        'exist_start_pos': exist_start_pos,
        'cpu_out': cpu_out, 'cmp_kv_mask': cmp_kv_mask,
        'cpu_kv_state': cpu_kv_state, 'cpu_score_state': cpu_score_state,
        'update_kv': update_kv, 'update_score': update_score,
        'block_size': block_size, 'block_num': block_table.shape[0] if cache_mode == 1 else B,
        'cache_mode': cache_mode, 'coff': coff, 'cmp_ratio': cmp_ratio,
    }


def run_op_with_inputs(data, noncontiguous=False, layer_stride=3):
    """Run NPU compressor with pre-generated inputs, optionally non-contiguous state_cache."""
    DEVICE_ID = 0
    torch_npu.npu.set_device(DEVICE_ID)

    x = data['x'].clone().npu()
    wkv = data['wkv'].clone().npu()
    wgate = data['wgate'].clone().npu()
    ape = data['ape'].clone().npu()
    block_table = data['block_table'].clone().npu()
    start_pos = torch.tensor(data['start_pos']).to(torch.int32).npu()
    cu_seqlens = data['cu_seqlens'].clone().npu() if data['cu_seqlens'] is not None else None
    seqused = data['seqused'].clone().npu() if data['seqused'] is not None else None

    kv_state = data['kv_state'].clone()
    score_state = data['score_state'].clone()
    state_dim0 = kv_state.shape[0]
    state_dim1 = kv_state.shape[1]
    state_dim2 = kv_state.shape[2] * 2

    if not noncontiguous:
        state_cache = torch.zeros((state_dim0, state_dim1, state_dim2), dtype=torch.float32).npu()
        state_cache[:, :, :state_dim2 // 2] = kv_state.npu()
        state_cache[:, :, state_dim2 // 2:] = score_state.npu()
    else:
        padded_dim1 = state_dim1 + layer_stride
        state_cache_pad = torch.full(
            (state_dim0, padded_dim1 * state_dim2), float('nan'),
            dtype=torch.float32,
        ).npu()
        state_cache = torch.as_strided(
            state_cache_pad,
            size=(state_dim0, state_dim1, state_dim2),
            stride=(padded_dim1 * state_dim2, state_dim2, 1),
        )
        state_cache[:, :, :state_dim2 // 2] = kv_state.npu()
        state_cache[:, :, state_dim2 // 2:] = score_state.npu()

    npu_out = compressor(
        x, wkv, wgate, state_cache, ape,
        cmp_ratio=data['cmp_ratio'],
        state_block_table=block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos if data['exist_start_pos'] else None,
        coff=data['coff'],
        cache_mode=data['cache_mode'],
    )
    torch.npu.synchronize()

    return {
        'npu_out': npu_out,
        'state_cache': state_cache,
        'state_cache_pad': state_cache_pad if noncontiguous else None,
        'state_dim2': state_dim2,
        'noncontiguous': noncontiguous,
        'layer_stride': layer_stride if noncontiguous else 0,
    }


# ==================== Test Parameters ====================
# Requirements for each case:
#   - block_num = B * ceil(S_max / block_size) >= 4  (so >= 3 blocks are assignable)
#   - D0 = block_num >= 2  (so is_contiguous() correctly returns False for padded stride)
#   - start_pos triggers state read/write across multiple blocks
#   - coff/cmp_ratio/head_dim combinations cover diverse kernel paths
TEST_CASES = [
    {
        'name': 'prefill_d512_coff2_cmp4',
        'params': dict(B=2, S_max=512, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=8, start_pos=[0, 0],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'prefill_d128_coff2_cmp4',
        'params': dict(B=2, S_max=512, head_dim=128, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=8, start_pos=[0, 0],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'decode_d512_coff2_cmp4',
        'params': dict(B=2, S_max=1024, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=1, start_pos=[256, 512],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'decode_d128_coff1_cmp4',
        'params': dict(B=2, S_max=1024, head_dim=128, coff=1, cmp_ratio=4,
                       bs_combine_flag=True, S=1, start_pos=[256, 512],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'prefill_d512_coff1_cmp128',
        'params': dict(B=2, S_max=512, head_dim=512, coff=1, cmp_ratio=128,
                       bs_combine_flag=True, S=8, start_pos=[0, 0],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'bsh_decode_d512_coff2_cmp4',
        'params': dict(B=2, S_max=1024, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=False, S=1, start_pos=[256, 512],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    # ==================== float16 data_type ====================
    {
        'name': 'prefill_d512_coff2_cmp4_fp16',
        'params': dict(B=2, S_max=512, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=8, start_pos=[0, 0],
                       block_size=128, data_type=torch.float16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'decode_d128_coff1_cmp4_fp16',
        'params': dict(B=2, S_max=1024, head_dim=128, coff=1, cmp_ratio=4,
                       bs_combine_flag=True, S=1, start_pos=[256, 512],
                       block_size=128, data_type=torch.float16, hidden_size=4096,
                       cache_mode=1),
    },
    # NOTE: cache_mode=2 (RING_BUFFER) is not supported on arch22 (Ascend910B3),
    #       only cache_mode=1 (LINEAR_BUFFER) is supported per compressor_tiling.h
    # ==================== block_size=64 ====================
    {
        'name': 'decode_d512_coff2_cmp4_bs64',
        'params': dict(B=2, S_max=1024, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=1, start_pos=[256, 512],
                       block_size=64, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    # ==================== block_size=1024 ====================
    {
        'name': 'prefill_d512_coff2_cmp4_bs1024',
        'params': dict(B=2, S_max=2048, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=8, start_pos=[0, 0],
                       block_size=1024, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    # ==================== non-aligned start_pos ====================
    {
        'name': 'decode_d512_coff2_cmp4_unaligned',
        'params': dict(B=2, S_max=1024, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=3, start_pos=[130, 300],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    # ==================== seqused multi-batch unequal ====================
    {
        'name': 'prefill_d512_coff2_cmp4_seqused',
        'params': dict(B=2, S_max=512, head_dim=512, coff=2, cmp_ratio=4,
                       bs_combine_flag=True, S=8, start_pos=[0, 0],
                       seqused=[4, 8],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
    {
        'name': 'decode_d128_coff1_cmp4_seqused',
        'params': dict(B=3, S_max=1024, head_dim=128, coff=1, cmp_ratio=4,
                       bs_combine_flag=True, S=2, start_pos=[256, 512, 768],
                       seqused=[2, 1, 2],
                       block_size=128, data_type=torch.bfloat16, hidden_size=4096,
                       cache_mode=1),
    },
]


@pytest.mark.ci
@pytest.mark.parametrize('case', TEST_CASES, ids=[c['name'] for c in TEST_CASES])
def test_state_cache_noncontiguous_axis0(case):
    """Test compressor with non-contiguous state_cache on axis 0."""
    torch_npu.npu.set_device(DEVICE_ID)
    params = case['params']

    # --- Generate inputs ONCE (same data for both runs) ---
    np.random.seed(42)
    data = generate_inputs(**params)

    # --- 1. Contiguous baseline ---
    res_c = run_op_with_inputs(data, noncontiguous=False)

    data_type_str = str(res_c['npu_out'].dtype)
    _, result_cmp_kv_c = check_result(
        data['cpu_out'][data['cmp_kv_mask']].to(torch.float32),
        res_c['npu_out'].cpu()[data['cmp_kv_mask']].to(torch.float32),
        data_type_str,
    )
    print(f'[{case["name"]}] contiguous cmp_kv: {result_cmp_kv_c}')

    # --- 2. Non-contiguous state_cache ---
    res_nc = run_op_with_inputs(data, noncontiguous=True, layer_stride=3)

    sc = res_nc['state_cache']
    contig_stride0 = sc.shape[1] * sc.shape[2]
    block_num = data['block_num']

    # Strict assertions: verify non-contiguity is real and meaningful
    # (1) stride(0) must differ from contiguous value
    assert sc.stride(0) != contig_stride0, \
        f'state_cache stride(0)={sc.stride(0)} == contiguous {contig_stride0}: NOT non-contiguous!'
    # (2) block_num >= 2 so is_contiguous() is not fooled by D0=1
    assert block_num >= 2, \
        f'block_num={block_num} < 2: D0=1 makes is_contiguous() return True even with padding'
    # (3) at least one block must be actually assigned (next_block_id starts at 1)
    bt = data['block_table']
    assigned_blocks = (bt > 0).sum().item()
    assert assigned_blocks >= 1, \
        f'no blocks assigned in block_table: kernel never uses stateCacheStrideDim0'
    # (4) at least one state element must be updated (WriteToCacheState exercised)
    assert data['update_kv'].any(), \
        f'no kv_state updates: WriteToCacheState/ReadFromCacheState not exercised'

    print(f'  state_cache stride: {sc.stride()}, contig_stride0={contig_stride0}')
    print(f'  block_num={block_num}, assigned_blocks={assigned_blocks}, '
          f'updated_kv={data["update_kv"].sum().item()}, '
          f'updated_score={data["update_score"].sum().item()}')

    _, result_cmp_kv_nc = check_result(
        data['cpu_out'][data['cmp_kv_mask']].to(torch.float32),
        res_nc['npu_out'].cpu()[data['cmp_kv_mask']].to(torch.float32),
        data_type_str,
    )
    print(f'[{case["name"]}] non-contiguous cmp_kv: {result_cmp_kv_nc}')

    # Check state update
    sd2 = res_nc['state_dim2']
    _, result_kv_update = check_result(
        data['cpu_kv_state'][data['update_kv']].to(torch.float32),
        sc[:, :, :sd2 // 2].cpu()[data['update_kv']].to(torch.float32),
        data_type_str,
    )
    _, result_score_update = check_result(
        data['cpu_score_state'][data['update_score']].to(torch.float32),
        sc[:, :, sd2 // 2:].cpu()[data['update_score']].to(torch.float32),
        data_type_str,
    )
    print(f'  kv_state_update: {result_kv_update}')
    print(f'  score_state_update: {result_score_update}')

    # Trap data check: padding region must remain NaN (kernel must not read/write it)
    pad = res_nc['state_cache_pad'].cpu()
    # Reshape pad to (D0, padded_dim1, state_dim2) to isolate padding rows
    padded_dim1 = sc.shape[1] + res_nc['layer_stride']
    pad_view = pad[:, :padded_dim1 * sd2].reshape(sc.shape[0], padded_dim1, sd2)
    padding_region = pad_view[:, sc.shape[1]:, :]  # the extra layer_stride rows
    trap_count = torch.isnan(padding_region).sum().item()
    trap_total = padding_region.numel()
    print(f'  trap data: {trap_count}/{trap_total} NaN in padding region '
          f'(should be all NaN if kernel respects stride)')
    assert trap_count == trap_total, \
        f'{case["name"]}: padding region corrupted ({trap_total - trap_count} non-NaN): ' \
        f'kernel may be using wrong stride!'

    # --- 3. Contiguous vs Non-contiguous: compare only valid (masked) output ---
    mask = data['cmp_kv_mask']
    out_c_valid = res_c['npu_out'].cpu()[mask].to(torch.float32)
    out_nc_valid = res_nc['npu_out'].cpu()[mask].to(torch.float32)
    if out_c_valid.numel() > 0:
        max_abs_diff = (out_c_valid - out_nc_valid).abs().max().item()
        allclose = torch.allclose(out_c_valid, out_nc_valid, rtol=0.005, atol=0.01)
        print(f'  contig vs noncontig (valid only): max_abs_diff={max_abs_diff:.6f}, allclose={allclose}')
        assert allclose, f'{case["name"]}: contig vs noncontig output not close (max_abs_diff={max_abs_diff})'
    else:
        print(f'  contig vs noncontig: no valid output elements (empty mask), skipping output comparison')

    # Assertions
    assert result_cmp_kv_c == 'Pass', f'{case["name"]}: contiguous baseline FAILED'
    assert result_cmp_kv_nc == 'Pass', f'{case["name"]}: non-contiguous test FAILED'
    assert result_kv_update == 'Pass', f'{case["name"]}: kv_state update FAILED'
    assert result_score_update == 'Pass', f'{case["name"]}: score_state update FAILED'
    print(f'[{case["name"]}] PASS - state_cache axis-0 non-contiguous supported')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'ci'])

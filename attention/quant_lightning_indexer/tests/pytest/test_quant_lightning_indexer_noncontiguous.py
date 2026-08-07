#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Test: quant_lightning_indexer supports key / key_dequant_scale non-contiguous on axis 0
# Approach: reference v2 blockFusion method to construct non-contiguous inputs
# -----------------------------------------------------------------------------------------------------------

import sys
import math
import numpy as np
import torch
import torch_npu
import pytest

sys.path.insert(0, '/home/tjj/ops-transform-IgnoreContiguous/attention/quant_lightning_indexer/tests/pytest')
import quant_lightning_indexer_golden as golden
import result_compare_method


def make_block_fusion_noncontiguous(key, key_dequant_scale, qk_dtype, dequant_dtype):
    """
    Reference v2 golden DISCONTINUOUS_KEYS approach:
    Fuse key and key_dequant_scale into a single blockFusion tensor,
    then slice them out so both are non-contiguous on axis 0.

    key:               (block_num, block_size, k_head_num, head_dim)  dtype=qk_dtype
    key_dequant_scale: (block_num, block_size, k_head_num)            dtype=dequant_dtype
    """
    block_num, block_size, k_head_num, head_dim = key.shape
    key_elem_size = key.element_size()
    scale_elem_size = key_dequant_scale.element_size()
    bytes_per_token = head_dim + scale_elem_size // key_elem_size

    blockFusion = torch.zeros(
        (block_num, block_size * k_head_num * bytes_per_token),
        dtype=qk_dtype,
        device=key.device,
    )
    key_flat = key.view(block_num, block_size * k_head_num * head_dim)
    scale_flat = (
        key_dequant_scale
        .view(block_num, block_size * k_head_num)
        .view(qk_dtype)
    )
    key_cols = block_size * k_head_num * head_dim
    blockFusion[:, :key_cols] = key_flat
    blockFusion[:, key_cols:] = scale_flat

    nc_key = blockFusion[:, :key_cols].view(block_num, block_size, k_head_num, head_dim)
    nc_scale = (
        blockFusion[:, key_cols:]
        .view(dequant_dtype)
        .view(block_num, block_size, k_head_num)
    )
    return nc_key, nc_scale, blockFusion


def run_op(query, key, weights, q_scale, k_scale,
           act_seq_q, act_seq_k, block_table,
           query_quant_mode, key_quant_mode,
           layout_query, layout_key,
           sparse_count, sparse_mode):
    npu_result = torch_npu.npu_quant_lightning_indexer(
        query, key, weights, q_scale, k_scale,
        actual_seq_lengths_query=act_seq_q,
        actual_seq_lengths_key=act_seq_k,
        block_table=block_table,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
    )
    torch.npu.synchronize()
    return npu_result


def generate_inputs_and_golden(params):
    """Reuse v1 golden input generation and CPU golden computation."""
    (batch_size, q_seq, k_seq, q_t_size, k_t_size,
     q_head_num, k_head_num, head_dim, block_size, block_num,
     qk_dtype, weight_dtype, dequant_dtype, actual_seq_dtype,
     act_seq_q, act_seq_k, query_quant_mode, key_quant_mode,
     layout_query, layout_key, sparse_count, sparse_mode,
     query_datarange, key_datarange, weights_datarange,
     q_scale_datarange, k_scale_datarange) = params

    test_qli = golden.GeneralizedQLI(
        batch_size, q_seq, k_seq, q_t_size, k_t_size,
        q_head_num, k_head_num, head_dim, block_size, block_num,
        qk_dtype, weight_dtype, dequant_dtype, actual_seq_dtype,
        act_seq_q, act_seq_k, query_quant_mode, key_quant_mode,
        layout_query, layout_key, sparse_count, sparse_mode,
    )

    actual_seq_lengths_query = (
        torch.tensor(act_seq_q).to(actual_seq_dtype).npu()
        if act_seq_q is not None
        else torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(actual_seq_dtype).npu()
    )
    actual_seq_lengths_key = (
        torch.tensor(act_seq_k).to(actual_seq_dtype).npu()
        if act_seq_k is not None
        else torch.tensor(np.random.uniform(k_seq, k_seq, batch_size)).to(actual_seq_dtype).npu()
    )

    if layout_query == 'BSND':
        query = torch.tensor(np.random.uniform(
            query_datarange[0], query_datarange[1],
            (batch_size, q_seq, q_head_num, head_dim)
        )).to(qk_dtype).npu()
        query_dequant_scale = torch.tensor(np.random.uniform(
            q_scale_datarange[0], q_scale_datarange[1],
            (batch_size, q_seq, q_head_num)
        )).to(dequant_dtype).npu()
        weights = torch.tensor(np.random.uniform(
            weights_datarange[0], weights_datarange[1],
            (batch_size, q_seq, q_head_num)
        )).to(weight_dtype).npu()

    if layout_key == 'BSND':
        key = torch.tensor(np.random.uniform(
            key_datarange[0], key_datarange[1],
            (batch_size, k_seq, k_head_num, head_dim)
        )).to(qk_dtype).npu()
        key_dequant_scale = torch.tensor(np.random.uniform(
            k_scale_datarange[0], k_scale_datarange[1],
            (batch_size, k_seq, k_head_num)
        )).to(dequant_dtype).npu()
        block_table = None
        cpu_result, topk_value = test_qli.forward(
            query, key, weights, query_dequant_scale, key_dequant_scale,
            actual_seq_lengths_query, actual_seq_lengths_key, block_table,
        )

    elif layout_key == 'PA_BSND':
        k_max_s2 = math.floor(max(act_seq_k))
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size)
        key_bnsd = torch.tensor(np.random.uniform(
            key_datarange[0], key_datarange[1],
            (batch_size, k_head_num, k_max_s2, head_dim)
        )).to(qk_dtype)
        key_dequant_scale_bns = torch.tensor(np.random.uniform(
            k_scale_datarange[0], k_scale_datarange[1],
            (batch_size, k_head_num, k_max_s2)
        )).to(dequant_dtype)

        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        assert block_num >= key_block_num_sum, f'block_num={block_num} < needed={key_block_num_sum}'

        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table_np = np.full(
            (batch_size, k_max_block_num_per_batch), fill_value=-1, dtype=np.int32
        )
        for batch_idx in range(batch_size):
            for i_block_id in range(key_block_num_per_batch[batch_idx]):
                block_table_np[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1

        key_expand = torch.zeros(
            (batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim),
            dtype=qk_dtype,
        )
        key_expand[:, :, :k_max_s2, :] = key_bnsd
        key = torch.zeros(
            (block_num, block_size, k_head_num, head_dim), dtype=qk_dtype
        )
        for i_batch in range(batch_size):
            for i_block, cur_block_id_bt in enumerate(block_table_np[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id_bt == -1:
                    continue
                for i_n in range(k_head_num):
                    key[cur_block_id_bt, :, i_n, :] = key_expand[
                        i_batch, i_n,
                        block_start_pos:block_start_pos + block_size, :
                    ]
        key = key.npu()

        key_dequant_scale_expand = torch.zeros(
            (batch_size, k_head_num, k_max_block_num_per_batch * block_size),
            dtype=dequant_dtype,
        )
        key_dequant_scale_expand[:, :, :k_max_s2] = key_dequant_scale_bns
        key_dequant_scale = torch.zeros(
            (block_num, block_size, k_head_num), dtype=dequant_dtype
        )
        for i_batch in range(batch_size):
            for i_block, cur_block_id_bt in enumerate(block_table_np[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id_bt == -1:
                    continue
                for i_n in range(k_head_num):
                    key_dequant_scale[cur_block_id_bt, :, i_n] = key_dequant_scale_expand[
                        i_batch, i_n,
                        block_start_pos:block_start_pos + block_size
                    ]
        key_dequant_scale = key_dequant_scale.npu()

        cpu_result, topk_value = test_qli.forward(
            query, key_bnsd, weights, query_dequant_scale,
            key_dequant_scale_bns,
            actual_seq_lengths_query, actual_seq_lengths_key, block_table_np,
        )
        block_table = torch.from_numpy(block_table_np).to(dtype=torch.int32).npu()

    return {
        'cpu_result': cpu_result,
        'topk_value': topk_value,
        'query': query,
        'key': key,
        'weights': weights,
        'query_dequant_scale': query_dequant_scale,
        'key_dequant_scale': key_dequant_scale,
        'actual_seq_lengths_query': actual_seq_lengths_query,
        'actual_seq_lengths_key': actual_seq_lengths_key,
        'block_table': block_table,
        'qk_dtype': qk_dtype,
        'dequant_dtype': dequant_dtype,
        'params': params,
    }


# ==================== Test Parameters ====================
# 910B3: qk_dtype=int8, dequant_dtype=float16, weight_dtype=float16
# group_size (q_head_num / k_head_num) must be 64
#
# NOTE: Only PA_BSND layout truly supports axis-0 non-contiguous key.
#   - In kernel, keyStride0 is used ONLY in KeyNd2NzForPA() (PA_BSND path).
#   - BSND's KeyNd2Nz() computes offset with contiguous stride
#     (bIdx * kSeqSize * kHeadNum * headDim), ignoring keyStride0.
#   - BSND non-contiguous "works" only because ACL framework does an implicit
#     contiguous copy, which adds overhead and is NOT true non-contiguous support.
NONCONTIGUOUS_TEST_PARAMS = [
    {
        'name': 'bsnd_pa_bsnd_basic',
        'params': (
            2, 4, 64, 1, 1,
            64, 1, 128,
            128, 4,
            torch.int8, torch.float16, torch.float16, torch.int32,
            [4, 4],
            [64, 64],
            0, 0,
            'BSND', 'PA_BSND',
            32, 0,
            [-1, 1], [-1, 1], [-25, 25], [0, 255], [0, 65504],
        ),
    },
    {
        'name': 'bsnd_pa_bsnd_sparse3',
        'params': (
            2, 4, 64, 1, 1,
            64, 1, 128,
            128, 4,
            torch.int8, torch.float16, torch.float16, torch.int32,
            [4, 4],
            [64, 64],
            0, 0,
            'BSND', 'PA_BSND',
            32, 3,
            [-1, 1], [-1, 1], [-25, 25], [0, 255], [0, 65504],
        ),
    },
    {
        'name': 'bsnd_pa_bsnd_multi_block',
        'params': (
            4, 4, 256, 1, 1,
            64, 1, 128,
            64, 20,
            torch.int8, torch.float16, torch.float16, torch.int32,
            [4, 4, 4, 4],
            [128, 256, 64, 200],
            0, 0,
            'BSND', 'PA_BSND',
            64, 3,
            [-1, 1], [-1, 1], [-25, 25], [0, 255], [0, 65504],
        ),
    },
    {
        'name': 'bsnd_pa_bsnd_large',
        'params': (
            3, 4, 1024, 1, 1,
            64, 1, 128,
            128, 30,
            torch.int8, torch.float16, torch.float16, torch.int32,
            [4, 4, 4],
            [512, 1024, 256],
            0, 0,
            'BSND', 'PA_BSND',
            128, 3,
            [-1, 1], [-1, 1], [-25, 25], [0, 255], [0, 65504],
        ),
    },
]


@pytest.mark.ci
@pytest.mark.parametrize('case', NONCONTIGUOUS_TEST_PARAMS, ids=[c['name'] for c in NONCONTIGUOUS_TEST_PARAMS])
def test_key_noncontiguous_axis0(case):
    """Test key / key_dequant_scale non-contiguous on axis 0."""
    torch_npu.npu.set_device(0)
    params = case['params']
    layout_key = params[19]
    p = params

    data = generate_inputs_and_golden(params)
    cpu_result = data['cpu_result']
    topk_value = data['topk_value']

    # --- 1. Contiguous baseline ---
    npu_contiguous = run_op(
        data['query'], data['key'], data['weights'],
        data['query_dequant_scale'], data['key_dequant_scale'],
        data['actual_seq_lengths_query'], data['actual_seq_lengths_key'],
        data['block_table'],
        p[16], p[17], p[18], p[19], p[20], p[21],
    )
    result_contig, pct_contig = result_compare_method.check_result(
        cpu_result, npu_contiguous, topk_value, params,
    )
    print(f'[{case["name"]}] contiguous: result={result_contig}, pct={pct_contig:.2f}%')

    # --- 2. Non-contiguous key / key_dequant_scale (PA_BSND only) ---
    if layout_key != 'PA_BSND':
        pytest.skip(f'layout_key={layout_key}: only PA_BSND truly supports axis-0 non-contiguous')
    nc_key, nc_scale, blockFusion = make_block_fusion_noncontiguous(
        data['key'], data['key_dequant_scale'],
        data['qk_dtype'], data['dequant_dtype'],
    )

    # Verify non-contiguity
    assert not nc_key.is_contiguous(), f'nc_key should be non-contiguous, stride={nc_key.stride()}'
    assert not nc_scale.is_contiguous(), f'nc_scale should be non-contiguous, stride={nc_scale.stride()}'
    print(f'  nc_key stride: {nc_key.stride()}, contiguous={nc_key.is_contiguous()}')
    print(f'  nc_scale stride: {nc_scale.stride()}, contiguous={nc_scale.is_contiguous()}')

    # Verify data consistency
    assert torch.equal(nc_key.contiguous(), data['key']), 'nc_key data mismatch with original key'
    assert torch.equal(nc_scale.contiguous(), data['key_dequant_scale']), 'nc_scale data mismatch'

    npu_noncontig = run_op(
        data['query'], nc_key, data['weights'],
        data['query_dequant_scale'], nc_scale,
        data['actual_seq_lengths_query'], data['actual_seq_lengths_key'],
        data['block_table'],
        p[16], p[17], p[18], p[19], p[20], p[21],
    )
    result_nc, pct_nc = result_compare_method.check_result(
        cpu_result, npu_noncontig, topk_value, params,
    )
    print(f'[{case["name"]}] non-contiguous: result={result_nc}, pct={pct_nc:.2f}%')

    # --- 3. Contiguous vs Non-contiguous should be identical ---
    diff = (npu_contiguous.cpu() != npu_noncontig.cpu()).sum().item()
    total = npu_contiguous.numel()
    print(f'  contig vs noncontig: diff={diff}/{total}')

    # Assertions
    assert result_contig == 'Pass', f'{case["name"]}: contiguous baseline FAILED (result={result_contig})'
    assert result_nc == 'Pass', f'{case["name"]}: non-contiguous test FAILED (result={result_nc})'
    assert diff == 0, f'{case["name"]}: contig vs noncontig differ ({diff}/{total})'
    print(f'[{case["name"]}] PASS - PA_BSND axis-0 non-contiguous supported (keyStride0 used in kernel)')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'ci'])

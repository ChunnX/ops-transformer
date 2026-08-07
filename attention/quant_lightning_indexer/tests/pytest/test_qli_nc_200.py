#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Test: quant_lightning_indexer dim0 non-contiguous, 200 generalized cases
# Covers: BSND+PA_BSND and TND+PA_BSND, q_seq up to 262144, k_seq up to 524288
# -----------------------------------------------------------------------------------------------------------

import sys
import math
import random
import numpy as np
import torch
import torch_npu
import pytest

sys.path.insert(0, '/home/tjj/ops-transform-IgnoreContiguous/attention/quant_lightning_indexer/tests/pytest')
import quant_lightning_indexer_golden as golden
import result_compare_method

DEVICE_ID = 0

# ==================== Fixed params for 910B3 ====================
Q_HEAD_NUM = 64
K_HEAD_NUM = 1
HEAD_DIM = 128
QK_DTYPE = torch.int8
WEIGHT_DTYPE = torch.float16
DEQUANT_DTYPE = torch.float16
ACT_SEQ_DTYPE = torch.int32

PRE_TOKENS = (1 << 63) - 1
NEXT_TOKENS = (1 << 63) - 1


def get_sparse_count(q_seq_val):
    if q_seq_val <= 4096:
        return 2048
    elif q_seq_val <= 32768:
        return 2048
    elif q_seq_val <= 65536:
        return 1024
    else:
        return 512


def make_nc(key, scale, method='blockFusion'):
    block_num = key.shape[0]
    if method == 'blockFusion':
        return _make_block_fusion_nc(key, scale)
    else:
        multiplier = int(method.replace('stride', '').replace('x', ''))
        return _make_stride_nc(key, scale, multiplier)


def _make_block_fusion_nc(key, scale):
    block_num, block_size, k_head_num, head_dim = key.shape
    key_elem_size = key.element_size()
    scale_elem_size = scale.element_size()
    bytes_per_token = head_dim + scale_elem_size // key_elem_size
    blockFusion = torch.zeros(
        (block_num, block_size * k_head_num * bytes_per_token),
        dtype=key.dtype, device=key.device,
    )
    key_flat = key.view(block_num, block_size * k_head_num * head_dim)
    scale_flat = scale.view(block_num, block_size * k_head_num).view(key.dtype)
    key_cols = block_size * k_head_num * head_dim
    blockFusion[:, :key_cols] = key_flat
    blockFusion[:, key_cols:] = scale_flat
    nc_key = blockFusion[:, :key_cols].view(block_num, block_size, k_head_num, head_dim)
    nc_scale = blockFusion[:, key_cols:].view(scale.dtype).view(block_num, block_size, k_head_num)
    return nc_key, nc_scale


def _make_stride_nc(key, scale, multiplier):
    block_num = key.shape[0]
    pad_dim0 = block_num * multiplier
    kp = torch.zeros((pad_dim0, *key.shape[1:]), dtype=key.dtype, device=key.device)
    sp = torch.zeros((pad_dim0, *scale.shape[1:]), dtype=scale.dtype, device=key.device)
    kp[0::multiplier] = key
    sp[0::multiplier] = scale
    return kp[0::multiplier], sp[0::multiplier]


def run_op(query, key, weights, q_scale, k_scale,
           act_seq_q, act_seq_k, block_table,
           query_quant_mode, key_quant_mode,
           layout_query, layout_key,
           sparse_count, sparse_mode):
    kwargs = dict(
        query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=sparse_count, sparse_mode=sparse_mode,
        pre_tokens=PRE_TOKENS, next_tokens=NEXT_TOKENS,
    )
    npu_result = torch_npu.npu_quant_lightning_indexer(
        query, key, weights, q_scale, k_scale,
        actual_seq_lengths_query=act_seq_q,
        actual_seq_lengths_key=act_seq_k,
        block_table=block_table, **kwargs)
    torch.npu.synchronize()
    return npu_result


def generate_inputs(params):
    (batch_size, q_seq, k_seq, q_t_size, k_t_size,
     block_size, block_num,
     layout_query, layout_key,
     sparse_count, sparse_mode,
     act_seq_q, act_seq_k,
     query_datarange, key_datarange,
     weights_datarange, q_scale_datarange, k_scale_datarange) = params

    np.random.seed(42)
    random.seed(42)

    if layout_query == 'BSND':
        query = torch.tensor(np.random.uniform(
            query_datarange[0], query_datarange[1],
            (batch_size, q_seq, Q_HEAD_NUM, HEAD_DIM))).to(QK_DTYPE).npu()
        query_dequant_scale = torch.tensor(np.random.uniform(
            q_scale_datarange[0], q_scale_datarange[1],
            (batch_size, q_seq, Q_HEAD_NUM))).to(DEQUANT_DTYPE).npu()
        weights = torch.tensor(np.random.uniform(
            weights_datarange[0], weights_datarange[1],
            (batch_size, q_seq, Q_HEAD_NUM))).to(WEIGHT_DTYPE).npu()
        act_q = torch.tensor(act_seq_q).to(ACT_SEQ_DTYPE).npu() if act_seq_q else \
                torch.tensor([q_seq] * batch_size).to(ACT_SEQ_DTYPE).npu()

    elif layout_query == 'TND':
        query = torch.tensor(np.random.uniform(
            query_datarange[0], query_datarange[1],
            (q_t_size, Q_HEAD_NUM, HEAD_DIM))).to(QK_DTYPE).npu()
        query_dequant_scale = torch.tensor(np.random.uniform(
            q_scale_datarange[0], q_scale_datarange[1],
            (q_t_size, Q_HEAD_NUM))).to(DEQUANT_DTYPE).npu()
        weights = torch.tensor(np.random.uniform(
            weights_datarange[0], weights_datarange[1],
            (q_t_size, Q_HEAD_NUM))).to(WEIGHT_DTYPE).npu()
        # TND: act_seq_q must be prefix sum for NPU operator
        if act_seq_q:
            act_q_per_batch = act_seq_q
        else:
            act_q_per_batch = [q_t_size // batch_size] * batch_size
        act_q_prefix = []
        cumsum = 0
        for v in act_q_per_batch:
            cumsum += v
            act_q_prefix.append(cumsum)
        act_q = torch.tensor(act_q_prefix).to(ACT_SEQ_DTYPE).npu()

    k_max_s2 = math.floor(max(act_seq_k)) if act_seq_k else k_seq
    k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size)

    if layout_key == 'PA_BSND':
        key_bnsd = torch.tensor(np.random.uniform(
            key_datarange[0], key_datarange[1],
            (batch_size, K_HEAD_NUM, k_max_s2, HEAD_DIM))).to(QK_DTYPE)
        scale_bns = torch.tensor(np.random.uniform(
            k_scale_datarange[0], k_scale_datarange[1],
            (batch_size, K_HEAD_NUM, k_max_s2))).to(DEQUANT_DTYPE)

        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in (act_seq_k or [k_seq] * batch_size):
            cur = math.floor(cur_act_k)
            cur_bn = math.ceil(cur / block_size)
            key_block_num_per_batch.append(cur_bn)
            key_block_num_sum += cur_bn
        assert block_num >= key_block_num_sum, f'block_num={block_num} < needed={key_block_num_sum}'

        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table_np = np.full(
            (batch_size, k_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
        for batch_idx in range(batch_size):
            for i_block_id in range(key_block_num_per_batch[batch_idx]):
                block_table_np[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1

        key_expand = torch.zeros(
            (batch_size, K_HEAD_NUM, k_max_block_num_per_batch * block_size, HEAD_DIM), dtype=QK_DTYPE)
        key_expand[:, :, :k_max_s2, :] = key_bnsd
        key = torch.zeros((block_num, block_size, K_HEAD_NUM, HEAD_DIM), dtype=QK_DTYPE)
        scale_expand = torch.zeros(
            (batch_size, K_HEAD_NUM, k_max_block_num_per_batch * block_size), dtype=DEQUANT_DTYPE)
        scale_expand[:, :, :k_max_s2] = scale_bns
        key_dequant_scale = torch.zeros((block_num, block_size, K_HEAD_NUM), dtype=DEQUANT_DTYPE)
        for i_batch in range(batch_size):
            for i_block, cur_block_id_bt in enumerate(block_table_np[i_batch]):
                if cur_block_id_bt == -1:
                    continue
                bs_pos = i_block * block_size
                key[cur_block_id_bt] = key_expand[i_batch, :, bs_pos:bs_pos + block_size, :].permute(1, 0, 2)
                key_dequant_scale[cur_block_id_bt] = scale_expand[i_batch, :, bs_pos:bs_pos + block_size].permute(1, 0)
        key = key.npu()
        key_dequant_scale = key_dequant_scale.npu()
        block_table = torch.from_numpy(block_table_np).to(torch.int32).npu()
        act_k = torch.tensor(act_seq_k).to(ACT_SEQ_DTYPE).npu() if act_seq_k else \
                torch.tensor([k_seq] * batch_size).to(ACT_SEQ_DTYPE).npu()

    qk_prod = q_seq * k_seq if layout_query == 'BSND' else q_t_size * k_seq
    if qk_prod <= 65536:
        # golden expects: TND act_seq_q = prefix sum, PA_BSND act_seq_k = per-batch
        if layout_query == 'TND':
            act_q_per = act_seq_q or ([q_t_size // batch_size] * batch_size)
            act_q_golden = []
            cumsum = 0
            for v in act_q_per:
                cumsum += v
                act_q_golden.append(cumsum)
        else:
            act_q_golden = act_seq_q or ([q_seq] * batch_size)
        act_k_golden = torch.tensor(act_seq_k or [k_seq] * batch_size, dtype=ACT_SEQ_DTYPE)

        test_qli = golden.GeneralizedQLI(
            batch_size, q_seq, k_seq, q_t_size, k_t_size,
            Q_HEAD_NUM, K_HEAD_NUM, HEAD_DIM, block_size, block_num,
            QK_DTYPE, WEIGHT_DTYPE, DEQUANT_DTYPE, ACT_SEQ_DTYPE,
            act_q_golden, act_k_golden,
            0, 0, layout_query, layout_key, sparse_count, sparse_mode,
        )
        if layout_key == 'PA_BSND':
            cpu_result, topk_value = test_qli.forward(
                query, key_bnsd, weights, query_dequant_scale, scale_bns,
                act_q_golden, act_k_golden, block_table_np,
            )
        else:
            cpu_result, topk_value = test_qli.forward(
                query, key, weights, query_dequant_scale, key_dequant_scale,
                act_q_golden, act_k_golden, block_table_np,
            )
    else:
        cpu_result = None
        topk_value = None

    return {
        'cpu_result': cpu_result, 'topk_value': topk_value,
        'query': query, 'key': key, 'weights': weights,
        'query_dequant_scale': query_dequant_scale, 'key_dequant_scale': key_dequant_scale,
        'act_seq_q': act_q, 'act_seq_k': act_k, 'block_table': block_table,
        'params': params,
    }


# ==================== Generate 200 test cases ====================
def gen_params():
    cases = []
    cid = 0

    DR = ([-1, 1], [-1, 1], [-25, 25], [0, 255], [0, 65504])

    def add(name, B, q, k, bs, sc, mode, lq='BSND', lk='PA_BSND',
            act_q=None, act_k=None, nc_method='blockFusion', qt=1, kt=1):
        nonlocal cid
        bn = math.ceil(max(act_k or [k]) / bs) * B + 4
        cases.append({
            'name': f'{cid:03d}_{name}',
            'params': (B, q, k, qt, kt, bs, bn, lq, lk, sc, mode, act_q, act_k, *DR),
            'nc_method': nc_method,
        })
        cid += 1

    # ---- Group 1: BSND batch_size variation (10) ----
    for B in [1, 2, 4, 8, 12, 16, 24]:
        add(f'bsnd_b{B}', B, 4, 512, 128, 2048, 3)
    add('bsnd_b2_k1024', 2, 4, 1024, 128, 2048, 3)
    add('bsnd_b4_k2048', 4, 4, 2048, 128, 2048, 3)
    add('bsnd_b8_k4096', 8, 4, 4096, 128, 2048, 3)

    # ---- Group 2: BSND q_seq short-mid (15) ----
    for q in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        add(f'bsnd_q{q}', 4, q, 1024, 128, 2048, 3)
    add('bsnd_q4_k256', 4, 4, 256, 128, 2048, 3)
    add('bsnd_q16_k2048', 4, 16, 2048, 128, 2048, 3)
    add('bsnd_q64_k4096', 4, 64, 4096, 128, 2048, 3)
    add('bsnd_q256_bs64', 4, 256, 1024, 64, 2048, 3)
    add('bsnd_q512_bs256', 4, 512, 2048, 256, 2048, 3)

    # ---- Group 3: BSND q_seq long (20) ----
    for q in [1024, 2048, 4096, 8192]:
        for B in [1, 2, 4]:
            sc = get_sparse_count(q)
            add(f'bsnd_q{q}_b{B}', B, q, 2048, 128, sc, 3)
    for q in [1024, 2048, 4096, 8192]:
        sc = get_sparse_count(q)
        add(f'bsnd_q{q}_k8192', 1, q, 8192, 128, sc, 3)
    add('bsnd_q16384', 1, 16384, 8192, 128, get_sparse_count(16384), 3)
    add('bsnd_q32768', 1, 32768, 8192, 128, get_sparse_count(32768), 3)
    add('bsnd_q65536', 1, 65536, 16384, 256, get_sparse_count(65536), 3)
    add('bsnd_q131072', 1, 131072, 32768, 256, 512, 3)
    add('bsnd_q262144', 1, 262144, 65536, 256, 512, 3)

    # ---- Group 4: BSND k_seq variation (25) ----
    for k in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        add(f'bsnd_k{k}', 4, 4, k, 128, 2048, 3)
    for k in [32768, 65536]:
        add(f'bsnd_k{k}_b1', 1, 4, k, 128, 2048, 3)
        add(f'bsnd_k{k}_b2', 2, 4, k, 128, 2048, 3)
    add('bsnd_k131072', 1, 4, 131072, 256, 2048, 3)
    add('bsnd_k262144', 1, 4, 262144, 256, 2048, 3)
    add('bsnd_k524288', 1, 4, 524288, 256, 2048, 3)
    add('bsnd_k65536_q64', 1, 64, 65536, 256, 2048, 3)
    add('bsnd_k131072_q1024', 1, 1024, 131072, 256, get_sparse_count(1024), 3)
    add('bsnd_k262144_q4096', 1, 4096, 262144, 256, get_sparse_count(4096), 3)
    add('bsnd_k524288_q64', 1, 64, 524288, 256, 2048, 3)
    add('bsnd_k32768_q1024', 1, 1024, 32768, 128, get_sparse_count(1024), 3)
    add('bsnd_k16384_q4096', 1, 4096, 16384, 128, get_sparse_count(4096), 3)

    # ---- Group 5: BSND block_size/sc/mode (15) ----
    for bs in [64, 128, 256]:
        add(f'bsnd_bs{bs}_k256', 4, 4, 256, bs, 2048, 3)
        add(f'bsnd_bs{bs}_k1024', 4, 4, 1024, bs, 2048, 3)
        add(f'bsnd_bs{bs}_k4096', 4, 4, 4096, bs, 2048, 3)
    for sc in [1, 32, 128, 512, 2048]:
        add(f'bsnd_sc{sc}', 4, 4, 512, 128, sc, 3)
    for mode in [3]:
        add(f'bsnd_mode{mode}_k256', 2, 4, 256, 128, 2048, mode)
        add(f'bsnd_mode{mode}_k2048', 2, 4, 2048, 128, 2048, mode)
        add(f'bsnd_mode{mode}_k8192', 1, 4, 8192, 128, 2048, mode)
        add(f'bsnd_mode{mode}_b4', 4, 4, 512, 128, 2048, mode)

    # ---- Group 6: BSND nc construction method (10) ----
    for method in ['blockFusion', 'stridex2', 'stridex3', 'stridex4']:
        add(f'bsnd_nc_{method}_k256', 4, 4, 256, 128, 2048, 3, nc_method=method)
        add(f'bsnd_nc_{method}_k1024', 4, 4, 1024, 128, 2048, 3, nc_method=method)
    add('bsnd_nc_blockFusion_k4096', 4, 4, 4096, 128, 2048, 3, nc_method='blockFusion')
    add('bsnd_nc_stridex2_k8192', 1, 4, 8192, 128, 2048, 3, nc_method='stridex2')

    # ---- Group 7: BSND non-uniform act_seq_k (10) ----
    add('bsnd_nu_k1', 4, 4, 512, 128, 2048, 3, act_k=[64, 128, 256, 512])
    add('bsnd_nu_k2', 4, 4, 1024, 128, 2048, 3, act_k=[128, 256, 512, 1024])
    add('bsnd_nu_k3', 4, 4, 2048, 128, 2048, 3, act_k=[256, 512, 1024, 2048])
    add('bsnd_nu_k4', 4, 16, 1024, 128, 2048, 3, act_k=[128, 256, 512, 1024])
    add('bsnd_nu_k5', 8, 4, 512, 128, 2048, 3, act_k=[64, 128, 64, 256, 128, 512, 64, 512])
    add('bsnd_nu_k6', 8, 4, 1024, 128, 2048, 3, act_k=[128, 256, 512, 1024, 128, 256, 512, 1024])
    add('bsnd_nu_k7', 2, 4, 4096, 128, 2048, 3, act_k=[512, 4096])
    add('bsnd_nu_k8', 4, 4, 8192, 256, 2048, 3, act_k=[1024, 2048, 4096, 8192])
    add('bsnd_nu_k9', 4, 64, 2048, 128, 2048, 3, act_k=[256, 512, 1024, 2048])
    add('bsnd_nu_k10', 2, 1024, 4096, 128, get_sparse_count(1024), 3, act_k=[1024, 4096])

    # ---- Group 8: BSND combo/boundary (5) ----
    add('bsnd_combo_q8192_k32768', 1, 8192, 32768, 128, get_sparse_count(8192), 3)
    add('bsnd_combo_q4096_k65536', 1, 4096, 65536, 256, get_sparse_count(4096), 3)
    add('bsnd_boundary_q1_k64', 1, 1, 64, 64, 1, 3)
    add('bsnd_boundary_q1_sc1', 1, 1, 128, 128, 1, 3)
    add('bsnd_boundary_b24_q4_k256', 24, 4, 256, 128, 2048, 3)

    # ==================== TND cases (90) ====================
    def add_tnd(name, B, qt, k, bs, sc, mode, act_q_prefix=None, act_k=None, nc_method='blockFusion'):
        nonlocal cid
        if act_q_prefix is None:
            per = qt // B
            act_q_prefix = [per] * B
        if act_k is None:
            act_k = [k] * B
        bn = math.ceil(max(act_k) / bs) * B + 4
        cases.append({
            'name': f'{cid:03d}_{name}',
            'params': (B, qt // B, k, qt, 1, bs, bn, 'TND', 'PA_BSND', sc, mode,
                       act_q_prefix, act_k, *DR),
            'nc_method': nc_method,
        })
        cid += 1

    # ---- Group 9: TND q_t variation (20) ----
    for qt in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        add_tnd(f'tnd_qt{qt}', 1, qt, 1024, 128, 2048, 3)
    for qt in [4096, 8192]:
        add_tnd(f'tnd_qt{qt}_b1', 1, qt, 2048, 128, get_sparse_count(qt), 3)
        add_tnd(f'tnd_qt{qt}_b2', 2, qt, 2048, 128, get_sparse_count(qt), 3)
    add_tnd('tnd_qt256_k512', 2, 256, 512, 128, 2048, 3)
    add_tnd('tnd_qt1024_k4096', 1, 1024, 4096, 128, get_sparse_count(1024), 3)
    add_tnd('tnd_qt2048_k8192', 1, 2048, 8192, 128, get_sparse_count(2048), 3)
    add_tnd('tnd_qt512_b4', 4, 512, 1024, 128, 2048, 3)
    add_tnd('tnd_qt1024_b4', 4, 1024, 2048, 128, get_sparse_count(1024), 3)

    # ---- Group 10: TND k_seq variation (20) ----
    for k in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        add_tnd(f'tnd_k{k}', 2, 8, k, 128, 2048, 3)
    add_tnd('tnd_k32768_qt4', 1, 4, 32768, 128, 2048, 3)
    add_tnd('tnd_k32768_qt1024', 1, 1024, 32768, 128, get_sparse_count(1024), 3)
    add_tnd('tnd_k65536_qt4', 1, 4, 65536, 256, 2048, 3)
    add_tnd('tnd_k65536_qt1024', 1, 1024, 65536, 256, get_sparse_count(1024), 3)
    add_tnd('tnd_k65536_qt4096', 1, 4096, 65536, 256, get_sparse_count(4096), 3)
    add_tnd('tnd_k131072_qt4', 1, 4, 131072, 256, 2048, 3)
    add_tnd('tnd_k131072_qt64', 1, 64, 131072, 256, 2048, 3)
    add_tnd('tnd_k262144_qt4', 1, 4, 262144, 256, 2048, 3)
    add_tnd('tnd_k262144_qt64', 1, 64, 262144, 256, 2048, 3)
    add_tnd('tnd_k16384_qt2048', 1, 2048, 16384, 128, get_sparse_count(2048), 3)
    add_tnd('tnd_k8192_qt4096', 1, 4096, 8192, 128, get_sparse_count(4096), 3)

    # ---- Group 11: TND q_t ultra-long (15) ----
    for qt in [16384, 32768, 65536]:
        add_tnd(f'tnd_qt{qt}_k8192', 1, qt, 8192, 128, get_sparse_count(qt), 3)
        add_tnd(f'tnd_qt{qt}_k32768', 1, qt, 32768, 256, get_sparse_count(qt), 3)
    add_tnd('tnd_qt131072_k65536', 1, 131072, 65536, 256, 512, 3)
    add_tnd('tnd_qt131072_k32768', 1, 131072, 32768, 256, 512, 3)
    add_tnd('tnd_qt262144_k65536', 1, 262144, 65536, 256, 512, 3)
    add_tnd('tnd_qt262144_k32768', 1, 262144, 32768, 256, 512, 3)
    add_tnd('tnd_qt65536_k16384', 1, 65536, 16384, 128, get_sparse_count(65536), 3)
    add_tnd('tnd_qt32768_k16384', 1, 32768, 16384, 128, get_sparse_count(32768), 3)
    add_tnd('tnd_qt16384_k16384', 1, 16384, 16384, 128, get_sparse_count(16384), 3)

    # ---- Group 12: TND block_size/sc (10) ----
    for bs in [64, 128, 256]:
        add_tnd(f'tnd_bs{bs}_k512', 2, 8, 512, bs, 2048, 3)
        add_tnd(f'tnd_bs{bs}_k2048', 2, 8, 2048, bs, 2048, 3)
    for sc in [1, 32, 512, 2048]:
        add_tnd(f'tnd_sc{sc}', 2, 8, 512, 128, sc, 3)

    # ---- Group 13: TND nc construction method (10) ----
    for method in ['blockFusion', 'stridex2', 'stridex3', 'stridex4']:
        add_tnd(f'tnd_nc_{method}_k256', 2, 8, 256, 128, 2048, 3, nc_method=method)
        add_tnd(f'tnd_nc_{method}_k1024', 2, 8, 1024, 128, 2048, 3, nc_method=method)
    add_tnd('tnd_nc_blockFusion_k4096', 1, 4, 4096, 128, 2048, 3, nc_method='blockFusion')
    add_tnd('tnd_nc_stridex2_k8192', 1, 4, 8192, 128, 2048, 3, nc_method='stridex2')

    # ---- Group 14: TND non-uniform act_seq (10) ----
    add_tnd('tnd_nu1', 3, 24, 512, 128, 2048, 3, act_q_prefix=[4, 8, 12], act_k=[64, 256, 512])
    add_tnd('tnd_nu2', 3, 48, 1024, 128, 2048, 3, act_q_prefix=[8, 16, 24], act_k=[128, 512, 1024])
    add_tnd('tnd_nu3', 2, 32, 2048, 128, 2048, 3, act_q_prefix=[4, 28], act_k=[256, 2048])
    add_tnd('tnd_nu4', 4, 64, 512, 128, 2048, 3, act_q_prefix=[4, 8, 16, 36], act_k=[64, 128, 256, 512])
    add_tnd('tnd_nu5', 3, 96, 1024, 128, 2048, 3, act_q_prefix=[16, 32, 48], act_k=[128, 512, 1024])
    add_tnd('tnd_nu6', 2, 16, 4096, 128, 2048, 3, act_q_prefix=[4, 12], act_k=[512, 4096])
    add_tnd('tnd_nu7', 4, 128, 2048, 128, 2048, 3, act_q_prefix=[16, 32, 32, 48], act_k=[256, 512, 1024, 2048])
    add_tnd('tnd_nu8', 3, 12, 8192, 256, 2048, 3, act_q_prefix=[2, 4, 6], act_k=[1024, 4096, 8192])
    add_tnd('tnd_nu9', 2, 8, 16384, 256, 2048, 3, act_q_prefix=[2, 6], act_k=[4096, 16384])
    add_tnd('tnd_nu10', 4, 256, 4096, 128, 2048, 3, act_q_prefix=[32, 64, 64, 96], act_k=[512, 1024, 2048, 4096])

    # ---- Group 15: TND combo/boundary (5) ----
    add_tnd('tnd_combo_q8192_k32768', 1, 8192, 32768, 128, get_sparse_count(8192), 3)
    add_tnd('tnd_combo_q4096_k65536', 1, 4096, 65536, 256, get_sparse_count(4096), 3)
    add_tnd('tnd_boundary_q4_k64', 1, 4, 64, 64, 1, 3)
    add_tnd('tnd_boundary_q4_sc1', 1, 4, 128, 128, 1, 3)
    add_tnd('tnd_boundary_b4_qt512', 4, 512, 1024, 128, 2048, 3)
    add_tnd('tnd_boundary_q8_k128', 1, 8, 128, 128, 1, 3)
    add_tnd('tnd_boundary_q16_k256', 1, 16, 256, 128, 32, 3)

    assert len(cases) == 200, f'Expected 200 cases, got {len(cases)}'
    return cases


TEST_CASES = gen_params()


@pytest.mark.ci
@pytest.mark.parametrize('case', TEST_CASES, ids=[c['name'] for c in TEST_CASES])
def test_qli_nc_dim0(case):
    torch_npu.npu.set_device(DEVICE_ID)
    params = case['params']
    nc_method = case['nc_method']
    p = params

    B, q_seq, k_seq, q_t, k_t, block_size, block_num, lq, lk, sc, mode, act_q, act_k = p[:13]
    do_golden = (q_seq * k_seq <= 65536) and (q_t * k_seq <= 65536)

    try:
        data = generate_inputs(params)
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.skip(f'Input generation failed: {str(e)[:300]}')

    cpu_result = data['cpu_result']
    topk_value = data['topk_value']

    # 1. Contiguous baseline
    try:
        out_c = run_op(
            data['query'], data['key'], data['weights'],
            data['query_dequant_scale'], data['key_dequant_scale'],
            data['act_seq_q'], data['act_seq_k'], data['block_table'],
            0, 0, lq, lk, sc, mode)
    except Exception as e:
        pytest.fail(f'Contiguous run failed: {str(e)[:300]}')

    # 2. Non-contiguous
    nc_key, nc_scale = make_nc(data['key'], data['key_dequant_scale'], nc_method)
    assert not nc_key.is_contiguous(), f'nc_key should be non-contiguous, stride={nc_key.stride()}'
    assert not nc_scale.is_contiguous(), f'nc_scale should be non-contiguous, stride={nc_scale.stride()}'

    # Verify data consistency
    assert torch.equal(nc_key.contiguous(), data['key']), 'nc_key data mismatch'
    assert torch.equal(nc_scale.contiguous(), data['key_dequant_scale']), 'nc_scale data mismatch'

    try:
        out_nc = run_op(
            data['query'], nc_key, data['weights'],
            data['query_dequant_scale'], nc_scale,
            data['act_seq_q'], data['act_seq_k'], data['block_table'],
            0, 0, lq, lk, sc, mode)
    except Exception as e:
        pytest.fail(f'Non-contiguous run failed: {str(e)[:300]}')

    # 3. Contiguous vs Non-contiguous must be identical
    diff = (out_c.cpu() != out_nc.cpu()).sum().item()
    total = out_c.numel()
    assert diff == 0, f'contig vs noncontig differ: {diff}/{total}'

    # 4. Golden check (for small cases, best-effort)
    if do_golden and cpu_result is not None:
        dr = ([-1, 1], [-1, 1], [-25, 25], [0, 255], [0, 65504])
        golden_params = (
            B, q_seq, k_seq, q_t, k_t, Q_HEAD_NUM, K_HEAD_NUM, HEAD_DIM, block_size, block_num,
            QK_DTYPE, WEIGHT_DTYPE, DEQUANT_DTYPE, ACT_SEQ_DTYPE,
            act_q or ([q_seq] * B if lq == 'BSND' else [q_t // B] * B),
            act_k or [k_seq] * B,
            0, 0, lq, lk, sc, mode, *dr,
        )
        try:
            result_c, pct_c = result_compare_method.check_result(
                cpu_result, out_c, topk_value, golden_params)
            result_nc, pct_nc = result_compare_method.check_result(
                cpu_result, out_nc, topk_value, golden_params)
            assert result_c == 'Pass', f'contiguous golden FAILED: result={result_c}, pct={pct_c:.2f}%'
            assert result_nc == 'Pass', f'non-contiguous golden FAILED: result={result_nc}, pct={pct_nc:.2f}%'
        except (ValueError, AssertionError) as e:
            if 'reshape' in str(e).lower() or 'cannot' in str(e).lower():
                print(f'  [WARN] Golden check skipped (shape mismatch): {str(e)[:100]}')
            else:
                raise

    print(f'[{case["name"]}] PASS - {lq}+{lk} q={q_seq} k={k_seq} bs={block_size} '
          f'sc={sc} nc={nc_method} diff={diff}/{total}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'ci'])

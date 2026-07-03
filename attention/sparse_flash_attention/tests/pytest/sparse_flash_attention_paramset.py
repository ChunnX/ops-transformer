#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import math
import torch


TEST_PARAMS = {
    "bsnd_basic": {
        "Testcase_Prefix": ["sfa_bsnd_basic"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "S1": [5],
        "S2": [262144],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [16],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [False],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[4]],
        "range_query": [[-10.0, 100.0]],
        "range_key": [[5.0, 100.0]],
    },
    "bsnd_multi_batch": {
        "Testcase_Prefix": ["sfa_bsnd_multi_batch"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.bfloat16],
        "kv_type": [torch.bfloat16],
        "B": [32],
        "S1": [6],
        "S2": [512],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [32],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [False],
        "actual_seq_q": [[2,5,4,2,6,6,0,6,6,2,2,6,3,4,6,4,5,5,6,4,0,4,3,2,3,0,0,3,3,6,2,3]],
        "actual_seq_kv": [[2,5,4,2,6,6,0,6,6,2,2,6,3,4,6,4,5,5,6,4,0,4,3,2,3,0,0,3,3,6,2,3]],
    },
    "pa_bsnd": {
        "Testcase_Prefix": ["sfa_pa_bsnd"],
        "layout_query": ["BSND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [2],
        "S1": [8192],
        "S2": [1024],
        "N1": [16],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [False],
        "use_sinks": [False],
        "block_size": [256],
        "block_num": [1051],
        "actual_seq_q": [[18,93]],
        "actual_seq_kv": [[1024,1024]],
    },
    "tnd_basic": {
        "Testcase_Prefix": ["sfa_tnd_basic"],
        "layout_query": ["TND"],
        "layout_kv": ["TND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "T1": [12],
        "T2": [902],
        "S1": [16],
        "S2": [1024],
        "N1": [12],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [False],
        "actual_seq_q": [[12]],
        "actual_seq_kv": [[902]],
    },
    "tnd_pa_multi_batch": {
        "Testcase_Prefix": ["sfa_tnd_pa_multi_batch"],
        "layout_query": ["TND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.bfloat16],
        "kv_type": [torch.bfloat16],
        "B": [4],
        "T1": [15],
        "S1": [5],
        "S2": [3072],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [64],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [False],
        "use_sinks": [False],
        "block_size": [512],
        "block_num": [74],
        "actual_seq_q": [[2, 6, 11, 15]],
        "actual_seq_kv": [[1024, 1024, 1024, 1024]],
    },
    "bsnd_sinks": {
        "Testcase_Prefix": ["sfa_bsnd_sinks"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "S1": [5],
        "S2": [128],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [16],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [True],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[4]],
        "range_sinks": [[-0.2, 0.2]],
    },
    "tnd_sinks": {
        "Testcase_Prefix": ["sfa_tnd_sinks"],
        "layout_query": ["TND"],
        "layout_kv": ["TND"],
        "q_type": [torch.bfloat16],
        "kv_type": [torch.bfloat16],
        "B": [2],
        "T1": [8],
        "T2": [256],
        "S1": [4],
        "S2": [256],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [32],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [True],
        "actual_seq_q": [[4, 8]],
        "actual_seq_kv": [[128, 256]],
        "range_sinks": [[-0.2, 0.2]],
    },
    "bsnd_sinks_fail1": {
        "Testcase_Prefix": ["sfa_bsnd_sinks_fail1"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [8],
        "S1": [1],
        "S2": [256],
        "N1": [24],
        "N2": [1],
        "D": [512],
        "K": [256],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [True],
        "actual_seq_q": [[1, 1, 1, 1, 1, 1, 1, 1]],
        "actual_seq_kv": [[170, 181, 192, 203, 214, 225, 236, 247]],
        "range_sinks": [[-0.2, 0.2]],
    },
    "bsnd_sinks_fail2": {
        "Testcase_Prefix": ["sfa_bsnd_sinks_fail2"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "S1": [8],
        "S2": [32],
        "N1": [12],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [False],
        "use_sinks": [True],
        "actual_seq_q": [[3]],
        "actual_seq_kv": [[32]],
        "range_sinks": [[0.0, 0.0]],
    },
    "tnd_sinks_fail1": {
        "Testcase_Prefix": ["sfa_tnd_sinks_fail1"],
        "layout_query": ["TND"],
        "layout_kv": ["TND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [4],
        "T1": [18],
        "T2": [5714],
        "S1": [8],
        "S2": [2048],
        "N1": [24],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [False],
        "use_sinks": [True],
        "actual_seq_q": [[4, 11, 13, 18]],
        "actual_seq_kv": [[1412, 2835, 4269, 5714]],
        "range_sinks": [[-0.2, 0.2]],
    },
    "tnd_sinks_fail2": {
        "Testcase_Prefix": ["sfa_tnd_sinks_fail2"],
        "layout_query": ["TND"],
        "layout_kv": ["TND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "T1": [14],
        "T2": [1446],
        "S1": [16],
        "S2": [2048],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [2048],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [False],
        "use_sinks": [True],
        "actual_seq_q": [[14]],
        "actual_seq_kv": [[1446]],
        "range_sinks": [[0.0, 0.0]],
    },
}


def _as_list_param(value):
    return [value]


def _make_param(
        prefix, layout, q_type, B, S1, S2, N1, K,
        sparse_mode, return_softmax_lse, use_sinks, actual_seq_q, actual_seq_kv,
        T1=None, T2=None, range_sinks=None):
    param = {
        "Testcase_Prefix": _as_list_param(prefix),
        "layout_query": _as_list_param(layout),
        "layout_kv": _as_list_param(layout),
        "q_type": _as_list_param(q_type),
        "kv_type": _as_list_param(q_type),
        "B": _as_list_param(B),
        "S1": _as_list_param(S1),
        "S2": _as_list_param(S2),
        "N1": _as_list_param(N1),
        "N2": _as_list_param(1),
        "D": _as_list_param(512),
        "K": _as_list_param(K),
        "scale_value": _as_list_param(1.0 / math.sqrt(576)),
        "sparse_block_size": _as_list_param(1),
        "rope_head_dim": _as_list_param(64),
        "sparse_mode": _as_list_param(sparse_mode),
        "attention_mode": _as_list_param(2),
        "return_softmax_lse": _as_list_param(return_softmax_lse),
        "use_sinks": _as_list_param(use_sinks),
        "actual_seq_q": _as_list_param(actual_seq_q),
        "actual_seq_kv": _as_list_param(actual_seq_kv),
    }
    if layout == "TND":
        param["T1"] = _as_list_param(T1 if T1 is not None else actual_seq_q[-1])
        param["T2"] = _as_list_param(T2 if T2 is not None else actual_seq_kv[-1])
    if use_sinks:
        param["range_sinks"] = _as_list_param(range_sinks or [-0.2, 0.2])
    return param


def _make_pa_param(
        prefix, layout_query, q_type, B, S1, S2, N1, K, block_size,
        actual_seq_q, actual_seq_kv, T1=None):
    block_num = sum(math.ceil(seq / block_size) for seq in actual_seq_kv)
    param = {
        "Testcase_Prefix": _as_list_param(prefix),
        "layout_query": _as_list_param(layout_query),
        "layout_kv": _as_list_param("PA_BSND"),
        "q_type": _as_list_param(q_type),
        "kv_type": _as_list_param(q_type),
        "B": _as_list_param(B),
        "S1": _as_list_param(S1),
        "S2": _as_list_param(S2),
        "N1": _as_list_param(N1),
        "N2": _as_list_param(1),
        "D": _as_list_param(512),
        "K": _as_list_param(K),
        "scale_value": _as_list_param(1.0 / math.sqrt(576)),
        "sparse_block_size": _as_list_param(1),
        "rope_head_dim": _as_list_param(64),
        "sparse_mode": _as_list_param(3),
        "attention_mode": _as_list_param(2),
        "return_softmax_lse": _as_list_param(False),
        "use_sinks": _as_list_param(False),
        "block_size": _as_list_param(block_size),
        "block_num": _as_list_param(block_num),
        "actual_seq_q": _as_list_param(actual_seq_q),
        "actual_seq_kv": _as_list_param(actual_seq_kv),
    }
    if layout_query == "TND":
        param["T1"] = _as_list_param(T1 if T1 is not None else actual_seq_q[-1])
    return param


def _spread_lengths(batch, upper_bound, seed, minimum=1):
    return [
        minimum + ((seed * 17 + idx * 11) % (upper_bound - minimum + 1))
        for idx in range(batch)
    ]


def _cumulative(lengths):
    total = 0
    actual_seq = []
    for length in lengths:
        total += length
        actual_seq.append(total)
    return actual_seq


def _select_k(seq_len, sparse_block_size, seed):
    # K is top-k count. Keep 2048 as the dominant case, with a few
    # smaller/larger points for coverage. sparse_block_size is fixed to 1.
    del seq_len, sparse_block_size
    if seed % 5 != 0:
        return 2048
    return [16, 32, 64, 128, 256, 512, 1024, 4096][(seed // 5) % 8]


def _select_n1(seed):
    main_head_nums = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
    rare_head_nums = [5, 7, 9, 10, 11, 14, 15, 17, 31, 63, 96, 127]
    if seed % 19 == 0:
        return rare_head_nums[(seed // 19) % len(rare_head_nums)]
    return main_head_nums[seed % len(main_head_nums)]


def _generate_bsnd_sinks_params():
    params = []
    batches = [1, 1, 2, 2, 4, 4, 8]
    s1_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    s2_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    sparse_block_size = 1
    sink_ranges = [[-0.2, 0.2], [0.0, 0.0], [-1.0, 1.0], [-6.0, 0.0], [0.0, 6.0]]
    large_configs = [
        (1, 1024, 16384, 8),
        (1, 2048, 32768, 4),
        (1, 4096, 8192, 2),
        (1, 8192, 4096, 1),
        (1, 64, 65536, 1),
        (2, 512, 16384, 2),
        (1, 128, 32768, 14),
        (1, 8192, 1024, 17),
        (1, 256, 65536, 3),
    ]

    for idx in range(145):
        if idx % 17 == 0:
            large_idx = (idx // 17) % len(large_configs)
            B, S1, S2, N1 = large_configs[large_idx]
        else:
            B = batches[idx % len(batches)]
            S1 = s1_sizes[(idx // 2) % len(s1_sizes)]
            S2 = s2_sizes[(idx // 5) % len(s2_sizes)]
            N1 = _select_n1(idx * 3)
        K = _select_k(S2, sparse_block_size, idx)
        actual_seq_q = [S1] * B if idx % 23 == 0 else _spread_lengths(B, S1, idx)
        actual_seq_kv = [S2] * B if idx % 17 == 0 else _spread_lengths(B, S2, idx + 5)
        params.append(_make_param(
            f"sfa_bsnd_sinks_gen_{idx:03d}",
            "BSND",
            torch.float16 if idx % 2 == 0 else torch.bfloat16,
            B,
            S1,
            S2,
            N1,
            K,
            0 if idx % 3 == 0 else 3,
            idx % 4 == 0,
            True,
            actual_seq_q,
            actual_seq_kv,
            range_sinks=sink_ranges[idx % len(sink_ranges)],
        ))
    return params


def _generate_tnd_sinks_params():
    params = []
    batches = [1, 1, 2, 2, 4, 8]
    per_batch_q_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    per_batch_kv_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    sparse_block_size = 1
    sink_ranges = [[-0.2, 0.2], [-0.5, 0.5], [0.0, 0.0], [-4.0, 2.0]]
    large_configs = [
        (1, 512, 8192, 8),
        (1, 1024, 16384, 4),
        (1, 2048, 32768, 2),
        (1, 4096, 4096, 1),
        (1, 128, 65536, 14),
        (2, 512, 8192, 3),
        (1, 8192, 2048, 17),
    ]

    for idx in range(80):
        if idx % 13 == 0:
            large_idx = (idx // 13) % len(large_configs)
            B, S1, S2, N1 = large_configs[large_idx]
        else:
            B = batches[idx % len(batches)]
            S1 = per_batch_q_sizes[(idx // 3) % len(per_batch_q_sizes)]
            S2 = per_batch_kv_sizes[(idx // 4) % len(per_batch_kv_sizes)]
            N1 = _select_n1(idx * 5)
        q_lengths = [S1] * B if idx % 29 == 0 else _spread_lengths(B, S1, idx + 11)
        kv_lengths = [S2] * B if idx % 13 == 0 else _spread_lengths(B, S2, idx + 19)
        actual_seq_q = _cumulative(q_lengths)
        actual_seq_kv = _cumulative(kv_lengths)
        K = _select_k(actual_seq_kv[-1], sparse_block_size, idx)
        params.append(_make_param(
            f"sfa_tnd_sinks_gen_{idx:03d}",
            "TND",
            torch.float16 if idx % 2 == 0 else torch.bfloat16,
            B,
            S1,
            S2,
            N1,
            K,
            0 if idx % 4 == 0 else 3,
            idx % 5 == 0,
            True,
            actual_seq_q,
            actual_seq_kv,
            T1=actual_seq_q[-1],
            T2=actual_seq_kv[-1],
            range_sinks=sink_ranges[idx % len(sink_ranges)],
        ))
    return params


def _generate_no_sinks_params():
    params = []
    sparse_block_size = 1
    for idx in range(15):
        layout = "BSND" if idx < 12 else "TND"
        if idx in (5, 11, 15, 19):
            B = 1
            S1 = [1024, 4096, 2048, 8192][(idx // 5) - 1]
            S2 = [8192, 32768, 16384, 65536][(idx // 5) - 1]
            N1 = [8, 14, 2, 17][(idx // 5) - 1]
        else:
            B = [1, 2, 4, 8][idx % 4]
            S1 = [1, 4, 8, 16, 32, 64, 128, 256, 512][idx % 9]
            S2 = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096][(idx // 2) % 9]
            N1 = _select_n1(idx * 7)
        if layout == "TND":
            actual_seq_q = _cumulative(_spread_lengths(B, S1, idx + 31))
            actual_seq_kv = _cumulative(_spread_lengths(B, S2, idx + 41))
            K = _select_k(actual_seq_kv[-1], sparse_block_size, idx)
            params.append(_make_param(
                f"sfa_tnd_no_sinks_gen_{idx:03d}",
                layout,
                torch.bfloat16 if idx % 2 else torch.float16,
                B,
                S1,
                S2,
                N1,
                K,
                0 if idx % 3 == 0 else 3,
                idx % 4 == 0,
                False,
                actual_seq_q,
                actual_seq_kv,
                T1=actual_seq_q[-1],
                T2=actual_seq_kv[-1],
            ))
        else:
            K = _select_k(S2, sparse_block_size, idx)
            params.append(_make_param(
                f"sfa_bsnd_no_sinks_gen_{idx:03d}",
                layout,
                torch.bfloat16 if idx % 2 else torch.float16,
                B,
                S1,
                S2,
                N1,
                K,
                0 if idx % 3 == 0 else 3,
                idx % 4 == 0,
                False,
                _spread_lengths(B, S1, idx + 31),
                _spread_lengths(B, S2, idx + 41),
            ))
    return params


def _generate_pa_params():
    params = []
    configs = [
        ("BSND", 1, 4, 128, 8, 16),
        ("BSND", 2, 8, 256, 16, 32),
        ("BSND", 4, 16, 512, 32, 64),
        ("BSND", 8, 8, 1024, 64, 128),
        ("BSND", 1, 512, 8192, 14, 256),
        ("TND", 2, 4, 128, 8, 16),
        ("TND", 4, 8, 256, 16, 32),
        ("TND", 8, 16, 512, 32, 64),
        ("TND", 1, 512, 8192, 17, 128),
        ("TND", 1, 1024, 16384, 4, 256),
    ]
    for idx, (layout_query, B, S1, S2, N1, block_size) in enumerate(configs):
        K = _select_k(S2, 1, idx)
        q_lengths = _spread_lengths(B, S1, idx + 53)
        kv_lengths = _spread_lengths(B, S2, idx + 61)
        actual_seq_q = _cumulative(q_lengths) if layout_query == "TND" else q_lengths
        params.append(_make_pa_param(
            f"sfa_pa_no_sinks_gen_{idx:03d}",
            layout_query,
            torch.float16 if idx % 2 == 0 else torch.bfloat16,
            B,
            S1,
            S2,
            N1,
            K,
            block_size,
            actual_seq_q,
            kv_lengths,
            T1=actual_seq_q[-1] if layout_query == "TND" else None,
        ))
    return params


def _generate_training_longseq_params():
    params = []
    s1_sizes = [2048, 4096, 8192]
    s2_sizes = [8192, 16384, 32768, 65536]
    for idx, (S1, S2) in enumerate((s1, s2) for s1 in s1_sizes for s2 in s2_sizes):
        N1 = [1, 2, 4, 8][idx % 4]
        params.append(_make_param(
            f"sfa_bsnd_sinks_train_longseq_{idx:03d}",
            "BSND",
            torch.float16 if idx % 2 == 0 else torch.bfloat16,
            1,
            S1,
            S2,
            N1,
            2048,
            3,
            False,
            True,
            [S1],
            [S2],
            range_sinks=[-0.2, 0.2],
        ))
    return params


def _generate_generalized_params():
    # Most generated tensors are mid-sized. A few point cases cover the upper
    # S1/S2 range with small B/N1 to avoid excessive device pressure.
    # sparse_block_size is fixed to 1 for all enabled generated cases.
    # The generated entries are single-point params, so combin_params does not create a cartesian explosion.
    return (
        _generate_bsnd_sinks_params() +
        _generate_tnd_sinks_params() +
        _generate_training_longseq_params() +
        _generate_no_sinks_params() +
        _generate_pa_params()
    )


# ENABLED_PARAMS = _generate_generalized_params()
ENABLED_PARAMS = [
    TEST_PARAMS["bsnd_basic"],
    # TEST_PARAMS["tnd_basic"],
    # TEST_PARAMS["bsnd_sinks_fail1"],
    # TEST_PARAMS["bsnd_sinks_fail2"],
    # TEST_PARAMS["tnd_sinks_fail1"],
    # TEST_PARAMS["tnd_sinks_fail2"],
]

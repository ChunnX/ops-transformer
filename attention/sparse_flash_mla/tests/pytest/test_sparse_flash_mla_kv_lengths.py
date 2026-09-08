#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU regressions for independently specified original and compressed KV lengths."""

import pytest
import torch

from sparse_flash_mla_golden import gen_data, generate_cpu_golden


def _params(**overrides):
    params = {
        "testcase_name": "independent_kv_lengths_cpu",
        "layout_q": "BSND",
        "layout_kv": "PA_BBND",
        "q_type": torch.float32,
        "ori_kv_type": torch.float32,
        "cmp_kv_type": torch.float32,
        "B": 1,
        "S1": 4,
        "S2": 3,
        "T1": None,
        "T2": None,
        "T3": None,
        "N1": 1,
        "N2": 1,
        "D": 4,
        "K1": 8,
        "K": 8,
        "block_num1": 2,
        "block_num2": 2,
        "block_size1": 2,
        "block_size2": 2,
        "cu_seqlens_q": None,
        "cu_seqlens_ori_kv": None,
        "cu_seqlens_cmp_kv": None,
        "seqused_q": [4],
        "seqused_ori_kv": [3],
        "seqused_cmp_kv": [3],
        "cmp_residual_kv": [1],
        "softmax_scale": 1.0,
        "cmp_ratio": 2,
        "ori_mask_mode": 0,
        "cmp_mask_mode": 3,
        "ori_win_left": -1,
        "ori_win_right": -1,
        "q_datarange": [0.0, 0.0],
        "ori_kv_datarange": [0.0, 0.0],
        "cmp_kv_datarange": [1.0, 1.0],
        "ori_kv_topk_mode": "fullK",
        "cmp_kv_topk_mode": "fullK",
        "ori_sparse_indices_mode": "full",
        "cmp_sparse_indices_mode": "full",
        "return_softmax_lse": True,
        "template_mode": "CSA",
    }
    params.update(overrides)
    return params


def _assert_full_indices(indices, counts):
    for row, count in zip(indices, counts):
        assert sorted(row[row >= 0].tolist()) == list(range(count))
        assert torch.all(row[count:] == -1)


@pytest.mark.parametrize("template_mode", ["HCA", "CSA", "ORI_CMP_SPARSE"])
@pytest.mark.parametrize(
    "mask_mode,residual,counts",
    [
        (3, [1], [2, 2, 3, 3]),
        (3, None, [1, 2, 2, 3]),
        (0, [1], [3, 3, 3, 3]),
    ],
    ids=["causal-residual", "causal-no-residual", "unmasked"],
)
def test_pa_compressed_golden_uses_its_own_lengths(
    template_mode, mask_mode, residual, counts
):
    # Original length is 3; restored compressed length is 7 (or 6 without
    # residual). The expected causal counts therefore cannot come from ori KV.
    data = gen_data(
        _params(
            template_mode=template_mode,
            cmp_mask_mode=mask_mode,
            cmp_residual_kv=residual,
        ),
        prepare_device_storage=False,
    )
    cmp_counts = torch.tensor(counts, dtype=torch.float32)
    # All scores and the sink are zero; original values are zero and compressed
    # values are one. Each output is a simple average including the sink.
    expected = (cmp_counts / (3 + cmp_counts + 1)).view(1, 4, 1, 1)
    torch.testing.assert_close(data["cpu_output"], expected.expand(1, 4, 1, 4))
    expected_lse = torch.log(3 + cmp_counts + 1).view(1, 1, 4, 1)
    torch.testing.assert_close(data["softmax_lse"], expected_lse)
    if template_mode != "HCA":
        _assert_full_indices(data["input"]["cmp_sparse_indices"][0, :, 0], counts)
    if template_mode == "ORI_CMP_SPARSE":
        _assert_full_indices(data["input"]["ori_sparse_indices"][0, :, 0], [3] * 4)


@pytest.mark.parametrize(
    "cmp_used,batch_counts",
    [
        ([3, 2], [[2, 2, 3, 3], [1, 1, 2]]),
        (None, [[4, 4, 5, 5], [3, 3, 4]]),
    ],
    ids=["used-before-cumulative", "cumulative-without-used"],
)
def test_tnd_compressed_lengths_residuals_and_padding(cmp_used, batch_counts):
    data = gen_data(
        _params(
            layout_q="TND",
            layout_kv="TND",
            B=2,
            S1=5,
            S2=9,
            T1=10,
            T2=16,
            T3=9,
            cu_seqlens_q=[0, 5, 10],
            cu_seqlens_ori_kv=[0, 7, 16],
            cu_seqlens_cmp_kv=[0, 5, 9],
            seqused_q=[4, 3],
            seqused_ori_kv=[3, 4],
            seqused_cmp_kv=cmp_used,
            cmp_residual_kv=[1, 0],
        ),
        prepare_device_storage=False,
    )
    expected = torch.zeros(10, 1, 4)
    expected_lse = torch.zeros(1, 10, 1)
    indices = data["input"]["cmp_sparse_indices"][:, 0]
    for start, ori_count, counts in zip([0, 5], [3, 4], batch_counts):
        cmp_counts = torch.tensor(counts, dtype=torch.float32)
        total = ori_count + cmp_counts + 1
        end = start + len(counts)
        expected[start:end] = (cmp_counts / total).view(-1, 1, 1)
        expected_lse[0, start:end, 0] = torch.log(total)
        _assert_full_indices(indices[start:end], counts)
    torch.testing.assert_close(data["cpu_output"], expected)
    torch.testing.assert_close(data["softmax_lse"], expected_lse)
    assert torch.all(indices[[4, 8, 9]] == -1)
    assert data["input"]["cmp_kv"].shape == (9, 1, 4)


def test_bsnd_allocates_compressed_storage_from_compressed_used_lengths():
    data = gen_data(
        _params(
            layout_kv="BSND",
            template_mode="HCA",
            B=2,
            S1=1,
            seqused_q=[1, 1],
            seqused_ori_kv=[3, 1],
            seqused_cmp_kv=[3, 5],
            cmp_residual_kv=[0, 1],
            cmp_mask_mode=0,
        ),
        prepare_device_storage=False,
    )
    assert data["input"]["cmp_kv"].shape == (2, 5, 1, 4)
    assert data["metadata_input"]["max_seqlen_cmp_kv"] == 5
    expected = torch.tensor([3 / 7, 5 / 7]).view(2, 1, 1, 1)
    torch.testing.assert_close(data["cpu_output"], expected.expand(2, 1, 1, 4))


def _paged_storage_params(**overrides):
    params = _params(
        B=2,
        S1=1,
        seqused_q=[1, 1],
        seqused_ori_kv=[3, 1],
        seqused_cmp_kv=[5, 3],
        cmp_residual_kv=[0, 1],
        cmp_mask_mode=0,
        block_num1=3,
        block_num2=5,
        cmp_kv_datarange=[-1.0, 1.0],
    )
    params.update(overrides)
    return params


def test_pa_compressed_blocks_preserve_all_compressed_tokens():
    data = gen_data(
        _paged_storage_params(), prepare_device_storage=False, generate_golden=False
    )
    table = data["input"]["cmp_block_table"]
    paged = data["input"]["cmp_kv"]
    logical = data["golden_state"]["cmp_k"]
    assert logical.shape == (2, 1, 5, 4)
    assert table.shape == (2, 3)
    assert (table >= 0).sum(dim=1).tolist() == [3, 2]
    assert len(set(table[table >= 0].tolist())) == 5
    for batch, length in enumerate([5, 3]):
        block_ids = table[batch][table[batch] >= 0].long()
        restored = paged[block_ids].reshape(-1, 1, 4)
        torch.testing.assert_close(restored[:length, 0], logical[batch, 0, :length])
        _assert_full_indices(data["input"]["cmp_sparse_indices"][batch, :, 0], [length])


def test_pa_rejects_capacity_below_independent_compressed_block_count():
    with pytest.raises(ValueError, match=r"cmp_kv.*4 < 5"):
        gen_data(
            _paged_storage_params(block_num2=4),
            prepare_device_storage=False,
            generate_golden=False,
        )


def test_pa_materializes_compressed_used_lengths_from_compressed_cumulative_lengths():
    data = gen_data(
        _paged_storage_params(
            seqused_cmp_kv=None,
            cu_seqlens_cmp_kv=[0, 5, 8],
        ),
        prepare_device_storage=False,
        generate_golden=False,
    )
    assert data["input"]["cu_seqlens_cmp_kv"] is None
    assert data["input"]["seqused_cmp_kv"].tolist() == [5, 3]
    assert (data["input"]["cmp_block_table"] >= 0).sum(dim=1).tolist() == [3, 2]


@pytest.mark.parametrize("layout_q", ["BSND", "TND"])
def test_random_compressed_indices_remain_empty_before_causal_boundary(layout_q):
    params = _params(
        layout_q=layout_q,
        seqused_cmp_kv=[1],
        cmp_residual_kv=[0],
        cmp_sparse_indices_mode="random",
    )
    if layout_q == "TND":
        params.update(
            layout_kv="TND",
            T1=4,
            T2=3,
            T3=1,
            cu_seqlens_q=[0, 4],
            cu_seqlens_ori_kv=[0, 3],
            cu_seqlens_cmp_kv=[0, 1],
        )
    data = gen_data(params, prepare_device_storage=False)
    indices = data["input"]["cmp_sparse_indices"].reshape(4, 8)
    _assert_full_indices(indices, [0, 0, 0, 1])
    expected = torch.tensor([0.0, 0.0, 0.0, 0.2]).view(4, 1)
    torch.testing.assert_close(data["cpu_output"].reshape(4, 4), expected.expand(4, 4))


def test_hca_golden_falls_back_to_compressed_tensor_length():
    data = gen_data(
        _params(layout_kv="BSND", template_mode="HCA"),
        prepare_device_storage=False,
        generate_golden=False,
    )
    state = data["golden_state"]
    state["seqused_cmp_kv"] = None
    state["cu_seqlens_cmp_kv"] = None
    state["cmp_residual_kv"] = None
    output, softmax_lse = generate_cpu_golden(data)
    # Three stored compressed tokens restore length six, while ori length is
    # only three. Missing residual metadata contributes zero to restoration.
    counts = torch.tensor([1, 2, 2, 3], dtype=torch.float32)
    expected = (counts / (3 + counts + 1)).view(1, 4, 1, 1)
    torch.testing.assert_close(output, expected.expand(1, 4, 1, 4))
    torch.testing.assert_close(softmax_lse, torch.log(3 + counts + 1).view(1, 1, 4, 1))

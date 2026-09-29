# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch

from attn_prologue_golden import apply_tail_rope, attn_prologue_golden, make_golden_inputs
from fp8_criterion import adjudicate
from generalization_inputs import all_cases


def test_zero_token_and_paged_cache():
    inputs = make_golden_inputs(3, force_zero_qr_group=True, block_size=2,
                                cache_index=torch.tensor([5, 0, 3]))
    before = inputs.kv_cache.clone()
    output = attn_prologue_golden(inputs, norm_eps=1e-6)
    assert output.q.shape == (3, 2, 64)
    assert torch.count_nonzero(output.q[0]) == 0
    assert torch.count_nonzero(output.qr[0].float()) == 0
    assert torch.count_nonzero(output.descale_qr[0]) == 0
    assert torch.count_nonzero(output.kv_cache.view(-1, 64)[5].view(torch.float8_e4m3fn).float()) == 0
    untouched = torch.ones(before.numel() // 64, dtype=torch.bool)
    untouched[inputs.cache_index] = False
    assert torch.equal(output.kv_cache.view(-1, 64)[untouched], before.view(-1, 64)[untouched])


def test_rope_interleaved_pairs_with_signed_sin():
    data = torch.tensor([[99., 98., 1., 2., 3., 4.]])
    sin = torch.tensor([[-1., -1., 1., 1.]])
    cos = torch.zeros_like(sin)
    result = apply_tail_rope(data, sin, cos)
    torch.testing.assert_close(result, torch.tensor([[99., 98., -2., -4., 1., 3.]]))


def test_fp64_rule_rejects_a_wrong_payload():
    got = torch.tensor([1.0, 2.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    ref = torch.tensor([1.125, 1.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    result = adjudicate(got, ref, torch.tensor([1.0625, 1.0], dtype=torch.float64))
    assert result.legitimate == 1
    assert result.hard == 1


def test_full_case_matrix():
    cases = all_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 32
    assert sum(case.phase == "decode" for case in cases) == 12
    assert sum(case.phase == "prefill" for case in cases) == 20
    assert {1, 72, 262144} <= {case.T for case in cases}

from __future__ import annotations

import pytest
import torch

from key_pool_golden import run_key_pool_golden
from key_pool_test_utils import make_inputs, run_pair


def test_bsh_base_and_incomplete_tail():
    inputs = make_inputs("BSH", [5], 4, 4, [0])
    output = run_pair(*inputs, cmp_ratio=4)
    assert output.shape == (1, 2, 8)
    assert torch.count_nonzero(output[:, 1:]) == 0


def test_bsh_norm_and_historical_cache():
    inputs = make_inputs("BSH", [5], 4, 4, [4], norm=True)
    run_pair(*inputs, cmp_ratio=4)


def test_cache_update_region_and_untouched_region():
    inputs = make_inputs("BSH", [3], 4, 4, [2])
    before = inputs[4].clone()
    golden_args = [item.clone() if isinstance(item, torch.Tensor) else item for item in inputs]
    _, intermediates = run_key_pool_golden(
        *golden_args, cmp_ratio=4, enforce_model_dims=False
    )
    after = intermediates["state_cache"]
    table = inputs[5]
    touched = torch.zeros_like(before, dtype=torch.bool)
    # end_pos=5 and R=4 leave only logical position 4 as the persistent tail.
    for logical_pos in range(4, 5):
        logical_block, offset = divmod(logical_pos, before.size(1))
        physical_block = int(table[0, logical_block])
        touched[physical_block, offset] = True
    assert torch.count_nonzero(after[touched] != before[touched]) > 0
    torch.testing.assert_close(after[~touched], before[~touched], rtol=0, atol=0)


def test_th_unequal_and_zero_length_batch():
    inputs = make_inputs("TH", [0, 4, 5], 4, 4, [0, 0, 0])
    output = run_pair(*inputs, cmp_ratio=4)
    assert output.shape[0] == 3


@pytest.mark.parametrize(
    "cu_seqlens,error",
    [
        (torch.tensor([1, 4, 8], dtype=torch.int32), "cu_seqlens\\[0\\]"),
        (torch.tensor([0, 5, 4], dtype=torch.int32), "nondecreasing"),
        (torch.tensor([0, 4, 7], dtype=torch.int32), "shape\\[0\\]"),
    ],
)
def test_th_cu_seqlens_contract(cu_seqlens, error):
    inputs = list(make_inputs("TH", [4, 4], 4, 4, [0, 0]))
    inputs[11] = cu_seqlens
    with pytest.raises(ValueError, match=error):
        run_pair(*inputs, cmp_ratio=4)


def test_th_requires_cu_seqlens_and_bsh_rejects_it():
    th_inputs = list(make_inputs("TH", [4, 4], 4, 4, [0, 0]))
    th_inputs[11] = None
    with pytest.raises(ValueError, match="required"):
        run_pair(*th_inputs, cmp_ratio=4)

    bsh_inputs = list(make_inputs("BSH", [4, 4], 4, 4, [0, 0]))
    bsh_inputs[11] = torch.tensor([0, 4, 8], dtype=torch.int32)
    with pytest.raises(ValueError, match="absent"):
        run_pair(*bsh_inputs, cmp_ratio=4)


def test_seqused_is_reserved():
    inputs = list(make_inputs("TH", [4, 4], 4, 4, [0, 0]))
    inputs[12] = torch.tensor([4, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="seqused"):
        run_pair(*inputs, cmp_ratio=4)


def test_all_supported_ratios():
    for ratio in (2, 4, 8, 16, 32, 64, 128):
        length = ratio
        inputs = make_inputs("BSH", [length], ratio, ratio, [0])
        output = run_pair(*inputs, cmp_ratio=ratio)
        assert output.shape[1] == 1


def test_rope_is_explicitly_rejected():
    inputs = make_inputs("BSH", [4], 4, 4, [0])
    hidden, wk, gate, ape, cache, table, start, norm_w, norm_b, _, _, cu_seqlens, seqused = inputs
    cos = torch.ones((1, 4, 4), dtype=torch.bfloat16)
    sin = torch.ones_like(cos)
    with pytest.raises(NotImplementedError, match="RoPE"):
        run_key_pool_golden(
            hidden, wk, gate, ape, cache, table, start,
            norm_w, norm_b, cos, sin, cu_seqlens, seqused, 4, enforce_model_dims=False,
        )

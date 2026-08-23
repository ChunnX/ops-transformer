from __future__ import annotations

import pytest
import torch

from key_pool_golden import run_key_pool_golden
from key_pool_test_utils import make_inputs, run_pair


BASELINE_CASES = [
    ("bsh_r2_smoke", "BSH", [2], 2, 1, [0], False, False),
    ("bsh_r4_incomplete", "BSH", [3], 4, 2, [0], False, False),
    ("bsh_r4_history_norm", "BSH", [4], 4, 4, [4], True, False),
    ("bsh_r4_cross_block", "BSH", [6], 4, 4, [2], False, False),
    ("bsh_r8_multi_batch", "BSH", [8, 8], 8, 4, [0, 4], True, False),
    ("bsh_r16_history", "BSH", [16], 16, 8, [8], False, False),
    ("bsh_r32_block_shuffle", "BSH", [32], 32, 16, [0], True, True),
    ("bsh_r64_history", "BSH", [64], 64, 32, [64], False, True),
    ("th_r2_single_batch", "TH", [4], 2, 2, [0], False, False),
    ("th_r4_zero_batch", "TH", [0, 4, 5], 4, 4, [0, 0, 0], False, False),
    ("th_r4_uneven_history", "TH", [2, 2], 4, 4, [2, 0], True, False),
    ("th_r4_cross_block", "TH", [1, 7, 3], 4, 2, [3, 1, 5], False, True),
    ("th_r8_multi_batch", "TH", [8, 8], 8, 4, [0, 4], True, False),
    ("th_r16_zero_batch", "TH", [16, 0], 16, 8, [0, 16], False, False),
    ("th_r32_history", "TH", [32], 32, 16, [16], True, True),
    ("th_r128_full", "TH", [128], 128, 32, [0], False, False),
]


@pytest.mark.parametrize(
    "case_id,layout,lengths,ratio,block_size,start_pos,norm,shuffle_blocks",
    BASELINE_CASES,
    ids=[case[0] for case in BASELINE_CASES],
)
def test_step01_baseline_golden_cases(
    case_id,
    layout,
    lengths,
    ratio,
    block_size,
    start_pos,
    norm,
    shuffle_blocks,
):
    inputs = list(make_inputs(layout, lengths, ratio, block_size, start_pos, norm=norm))
    if shuffle_blocks:
        inputs[5] = torch.flip(inputs[5], dims=[1]).contiguous()

    before_cache = inputs[4].clone()
    output = run_pair(*inputs, cmp_ratio=ratio)
    expected_capacity = (inputs[5].size(1) * block_size + ratio - 1) // ratio
    assert output.shape == (len(lengths), expected_capacity, inputs[1].size(0)), case_id

    golden_inputs = [
        tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in inputs
    ]
    golden_output, intermediates = run_key_pool_golden(
        *golden_inputs, cmp_ratio=ratio, enforce_model_dims=False
    )
    assert torch.equal(golden_output, output), case_id

    touched = torch.zeros_like(before_cache, dtype=torch.bool)
    table = inputs[5]
    for batch, (start, length) in enumerate(zip(start_pos, lengths)):
        for logical_pos in range(start, start + length):
            logical_block, offset = divmod(logical_pos, block_size)
            physical_block = int(table[batch, logical_block])
            touched[physical_block, offset] = True
        valid_pool_count = (start + length) // ratio - start // ratio
        assert torch.count_nonzero(output[batch, valid_pool_count:]) == 0, case_id

    after_cache = intermediates["state_cache"]
    torch.testing.assert_close(
        after_cache[~touched],
        before_cache[~touched],
        rtol=0,
        atol=0,
    )

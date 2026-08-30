# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Focused CPU/TestSpec contract tests for MoeTutelDispatch golden."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


_GOLDEN_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location(
    "moe_tutel_dispatch_golden", _GOLDEN_PATH
)
golden = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(golden)


@pytest.mark.parametrize("route_k, expert_e", [(5, 2), (2, 5)])
def test_golden_preserves_caller_expert_dimension(route_k, expert_e):
    samples, hidden, capacity = 3, 4, 2
    x = np.arange(samples * hidden, dtype=np.float32).reshape(samples, hidden)
    gates = np.zeros((route_k, samples), dtype=np.float32)
    indices = np.zeros((route_k, samples), dtype=np.int32)
    locations = np.zeros((route_k, samples), dtype=np.int32)
    gates[0, 0] = 1.0
    indices[0, 0] = expert_e - 1
    locations[0, 0] = 1

    out = golden.MoeTutelDispatchKernelSpec.golden(
        x,
        gates,
        indices,
        locations,
        capacity=capacity,
        output_ori_shapes=((expert_e, capacity, hidden),),
    )[0]

    assert out.shape == (expert_e, capacity, hidden)
    np.testing.assert_array_equal(
        out[expert_e - 1, 1].view(np.uint32), x[0].view(np.uint32)
    )
    assert np.all(out.view(np.uint32)[out == 0] == 0)


def test_golden_preserves_fixed_fp32_nan_payload_bits():
    raw = np.array([0x7FC12345, 0xFFC54321, 0x3F800000], dtype=np.uint32)
    x = raw.view(np.float32).reshape(1, 3)
    gates = np.ones((1, 1), dtype=np.float32)
    indices = np.zeros((1, 1), dtype=np.int32)
    locations = np.zeros((1, 1), dtype=np.int32)

    out = golden.MoeTutelDispatchKernelSpec.golden(
        x,
        gates,
        indices,
        locations,
        capacity=2,
        output_ori_shapes=((1, 2, 3),),
    )[0]

    np.testing.assert_array_equal(out[0, 0].view(np.uint32), raw)
    np.testing.assert_array_equal(
        out[0, 1].view(np.uint32), np.zeros(3, dtype=np.uint32)
    )


def test_third_party_compose_uses_output_shapes_not_gate_k():
    compose = golden._Compose(capacity=2, output_shapes=((5, 2, 3),))
    out = compose(
        torch.ones((1, 3), dtype=torch.float32),
        torch.ones((2, 1), dtype=torch.float32),
        torch.tensor([[4], [-1]], dtype=torch.int32),
        torch.tensor([[1], [0]], dtype=torch.int32),
    )[0]
    assert tuple(out.shape) == (5, 2, 3)


def test_missing_caller_output_shape_is_rejected():
    with pytest.raises(ValueError, match="caller-provided output shape"):
        golden.MoeTutelDispatchKernelSpec.golden(
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int32),
            capacity=1,
        )

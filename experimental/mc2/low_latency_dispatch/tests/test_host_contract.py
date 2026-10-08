# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host rejection tests run without Torch, CANN, SHMEM, or an NPU."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ops_transformer_mc2 import _runtime
from ops_transformer_mc2.buffers import elastic


class Tensor:
    def __init__(self, shape, dtype, device=0, contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device = SimpleNamespace(
            type="cpu" if device is None else "npu", index=device
        )
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=Tensor,
            bfloat16="bf16",
            float16="fp16",
            float8_e4m3fn="fp8",
            int64="i64",
            float32="f32",
            int32="i32",
        ),
    )
    calls = []

    def layout(*args):
        calls.append(args)
        return {"capacity_rows": 1024}

    monkeypatch.setattr(
        elastic, "require_native", lambda _: SimpleNamespace(low_latency_layout=layout)
    )
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer._runtime = SimpleNamespace(
        group=None, rank=0, world_size=16, device=0, closed=True
    )
    buffer.hidden = buffer.num_topk = 0
    return buffer, calls


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
def test_supported_inputs_reach_native_layout(plan, dtype):
    buffer, calls = plan
    x = Tensor((8, 7168), dtype)
    if dtype == "fp8":
        x = x, Tensor((8, 56), "i32")
    assert buffer._low_latency_plan(
        x, Tensor((8, 8), "i64"), Tensor((8, 8), "f32"), 128, 8
    )
    assert len(calls) == 1


@pytest.mark.parametrize("device", [None, 1])
@pytest.mark.parametrize("argument", ["x", "indices", "weights", "scales"])
def test_wrong_device_is_rejected_before_native_launch(plan, device, argument):
    buffer, calls = plan
    values = {
        "x": Tensor((8, 7168), "fp8"),
        "indices": Tensor((8, 8), "i64"),
        "weights": Tensor((8, 8), "f32"),
        "scales": Tensor((8, 56), "i32"),
    }
    values[argument].device = SimpleNamespace(
        type="cpu" if device is None else "npu", index=device
    )
    assert (
        buffer._low_latency_plan(
            (values["x"], values["scales"]),
            values["indices"],
            values["weights"],
            128,
            8,
        )
        is None
    )
    assert not calls


@pytest.mark.parametrize("capacity", [0, 7, 256, True])
def test_bad_capacity_does_not_reach_native_layout(plan, capacity):
    buffer, calls = plan
    assert (
        buffer._low_latency_plan(
            Tensor((8, 7168), "bf16"),
            Tensor((8, 8), "i64"),
            Tensor((8, 8), "f32"),
            128,
            capacity,
        )
        is None
    )
    assert not calls


@pytest.mark.parametrize("world", [2, 8, 256])
def test_wrong_topology_is_rejected(plan, world):
    buffer, calls = plan
    buffer._runtime.world_size = world
    assert (
        buffer._low_latency_plan(
            Tensor((8, 7168), "bf16"),
            Tensor((8, 8), "i64"),
            Tensor((8, 8), "f32"),
            128,
            8,
        )
        is None
    )
    assert not calls


def test_explicit_rank_identity():
    assert _runtime.rank_and_world(None, 15, 16) == (15, 16)
    with pytest.raises(ValueError):
        _runtime.rank_and_world(None, 16, 16)
    with pytest.raises(ValueError):
        _runtime.rank_and_world(None, 0, 8)


def test_no_group_does_not_read_inputs(plan):
    assert plan[0]._notify_low_latency(object(), 128) is None


def test_package_import_does_not_import_frameworks():
    mc2 = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(mc2 / "common/deepep/python"),
                str(mc2 / "low_latency_dispatch/python"),
            ]
        ),
    )
    paths = [
        str(mc2 / "common/deepep/python"),
        str(mc2 / "low_latency_dispatch/python"),
    ]
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path[:0] = {paths!r}; import ops_transformer_mc2; "
            "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules; "
            "assert 'deep_ep' not in sys.modules",
        ],
        env=env,
        check=True,
    )

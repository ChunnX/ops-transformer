# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""依据 aclnnChunkGatedDeltaRule.md 的合法输入泛化、精度与语义回归。"""

import json
import os
import random

import pytest
import torch

from cgdr_generalization_utils import aclnn, assert_precision, make_inputs, recurrent_reference


def precision_cases():
    cases = []

    def add(name, **kwargs):
        for dtype in ("bfloat16", "float32"):
            case = dict(name=name + "_" + dtype, state_dtype=dtype, **kwargs)
            cases.append(pytest.param(case, id=case["name"]))

    # 维度在lane、block及两AIV切片边界两侧取值；包括所有合法维度的端点。
    dims = (1, 2, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128)
    for dim in dims:
        add(f"diagonal_{dim}", dk=dim, dv=dim)
        if dim != 128:
            add(f"dk_{dim}_dv128", dk=dim, dv=128)
            add(f"dk128_dv_{dim}", dk=128, dv=dim)
    for length in (
        1,
        2,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        191,
        192,
        193,
        255,
        256,
        257,
    ):
        add(f"length_{length}", lengths=(length,))
    for length in (4095, 4096, 4097):
        add(f"group_boundary_{length}", lengths=(length,), nk=1, nv=2, dk=16, dv=16)
    for nk, nv in (
        (1, 1),
        (1, 64),
        (64, 64),
        (16, 32),
        (14, 28),
        (31, 31),
        (32, 32),
        (33, 33),
        (63, 63),
        (3, 6),
        (5, 15),
    ):
        add(f"heads_{nk}_{nv}", lengths=(65,), nk=nk, nv=nv, dk=17, dv=65)
    for lengths in (
        (1, 63, 64, 65, 129),
        (0, 1, 64, 0, 65, 0),
        (1, 1, 1),
        (65, 1, 129),
        (0, 0, 65),
        (65, 0, 0),
    ):
        add("batch_" + "_".join(map(str, lengths)), lengths=lengths, nk=16, nv=32, dk=16, dv=17)
    for dk, dv in ((5, 8), (128, 128)):
        for gate in ("none", "zero", "minus_one", "near_zero", "alternating"):
            add(f"gate_{gate}_{dk}_{dv}", dk=dk, dv=dv, gate=gate)
        for scale in (0.0, 1.0, -0.5, 0.001, 2.0):
            add(f"scale_{scale}_{dk}_{dv}", dk=dk, dv=dv, scale=scale)
        for beta in ("near_zero", "near_one"):
            for qk in ("one_hot", "alternating"):
                add(f"beta_{beta}_{qk}_{dk}_{dv}", dk=dk, dv=dv, beta=beta, qk=qk, gate="zero")
        add(f"zero_state_{dk}_{dv}", dk=dk, dv=dv, zero_state=True)
        add(f"zero_value_{dk}_{dv}", dk=dk, dv=dv, zero_value=True)
        add(f"amplitude_{dk}_{dv}", dk=dk, dv=dv, value_scale=64, state_scale=16)
    for name, axes in [
        ("q", range(3)),
        ("k", range(3)),
        ("v", range(3)),
        ("beta", range(2)),
        ("g", range(2)),
        ("lengths", range(1)),
        ("state", range(2)),
    ]:
        for axis in axes:
            for dk, dv in ((5, 8), (65, 127)):
                add(
                    f"layout_{name}_{axis}_{dk}_{dv}",
                    lengths=(1, 63, 65),
                    dk=dk,
                    dv=dv,
                    layouts={name: axis},
                )
    add(
        "all_noncontiguous",
        lengths=(1, 63, 65),
        dk=128,
        dv=128,
        layouts={"q": 2, "k": 0, "v": 1, "beta": 1, "g": 0, "lengths": 0, "state": 1},
    )
    # 多seed固定随机组合，重现性与单一fixture随机seed解耦。
    rng = random.Random(20260908)
    for i in range(16):
        nk = rng.choice((1, 2, 3, 8, 16))
        nv = nk * rng.choice((1, 2, 4))
        add(
            f"random_{i}",
            seed=1000 + i,
            nk=nk,
            nv=nv,
            dk=rng.choice(dims),
            dv=rng.choice(dims),
            lengths=tuple(rng.choice((0, 1, 17, 63, 64, 65, 129)) for _ in range(3)) + (1,),
            gate=rng.choice(("none", "random", "zero", "minus_one")),
            scale=rng.choice((0.125, 1.0, -1.0)),
        )
    return cases


def check_case(case, aclnn, record_property):
    inputs = make_inputs(case, device=f"npu:{torch.npu.current_device()}")
    tensors = inputs["tensors"]
    # 检查输出shape/dtype/有限性，完整输入及有洞storage不可被算子覆盖。
    before = [None if x is None else x.cpu().clone() for x in tensors]
    padding_before = [x.cpu().clone() for x in inputs["backings"]]
    expected = recurrent_reference(inputs)
    actual = aclnn.call(inputs)
    assert actual[0].shape == tensors[2].shape and actual[0].dtype == torch.bfloat16
    assert actual[1].shape == tensors[4].shape and actual[1].dtype == tensors[4].dtype
    assert all(x.is_contiguous() for x in actual)
    metrics = {
        name: assert_precision(out, ref, name)
        for name, out, ref in zip(("attention", "final_state"), actual, expected)
    }
    for old, new in zip(before, tensors):
        if old is not None:
            assert torch.equal(old, new.cpu()), "算子修改了输入"
    for old, new in zip(padding_before, inputs["backings"]):
        assert torch.equal(old, new.cpu()), "算子修改了非连续输入的padding"
    for name, value in [("case", case), ("precision", metrics), ("opapi", aclnn.path)]:
        record_property(name, json.dumps(value, ensure_ascii=False))


@pytest.mark.ci
@pytest.mark.parametrize("case", precision_cases())
def test_generalization_precision(case, aclnn, record_property):
    check_case(case, aclnn, record_property)


@pytest.mark.ci
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(
                name=f"network_{length}_{nk}_{dtype}",
                lengths=(length,),
                nk=nk,
                nv=2 * nk,
                state_dtype=dtype,
            ),
            id=f"T{length}_Nk{nk}_{dtype}",
        )
        for length in (3500, 7000, 8000, 11264)
        for nk in (16, 14)
        for dtype in ("bfloat16", "float32")
    ],
)
def test_network_precision(case, aclnn, record_property):
    check_case(case, aclnn, record_property)


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dk,dv", [(5, 8), (128, 128)])
@pytest.mark.parametrize(
    "relation",
    [
        "determinism",
        "gate_none_zero",
        "scale_zero",
        "scale_sign",
        "batch_independence",
        "head_permutation",
        "layout_equivalence",
    ],
)
def test_semantic_relations(dtype, dk, dv, relation, aclnn):
    case = dict(lengths=(63, 65), nk=2, nv=4, dk=dk, dv=dv, state_dtype=dtype)
    inputs = make_inputs(case, device=f"npu:{torch.npu.current_device()}")
    if relation == "gate_none_zero":
        inputs["tensors"][-1] = None
    if relation == "head_permutation":
        # Nk=1时所有value head共享同一Q/K，value head置换不改变组间Q/K映射。
        case.update(nk=1)
        inputs = make_inputs(case, device=f"npu:{torch.npu.current_device()}")
    expected = aclnn.call(inputs)
    changed = dict(tensors=list(inputs["tensors"]), scale=inputs["scale"])
    if relation == "determinism":
        actual = aclnn.call(inputs)
    elif relation == "gate_none_zero":
        changed["tensors"][-1] = torch.zeros_like(inputs["tensors"][3], dtype=torch.float32)
        actual = aclnn.call(changed)
    elif relation == "scale_zero":
        changed["scale"] = 0.0
        actual = aclnn.call(changed)
        assert torch.count_nonzero(actual[0]).item() == 0
        assert torch.equal(actual[1], expected[1])
        return
    elif relation == "scale_sign":
        changed["scale"] = -inputs["scale"]
        actual = aclnn.call(changed)
        expected = [-expected[0], expected[1]]
    elif relation == "head_permutation":
        permutation = torch.tensor([3, 1, 0, 2], device=inputs["tensors"][0].device)
        for index in (2, 3, 4, 6):
            changed["tensors"][index] = inputs["tensors"][index].index_select(1, permutation)
        actual = aclnn.call(changed)
        expected = [x.index_select(1, permutation) for x in expected]
    elif relation == "layout_equivalence":
        changed = make_inputs(
            dict(
                case, layouts={"q": 2, "k": 0, "v": 1, "beta": 1, "g": 0, "lengths": 0, "state": 1}
            ),
            device=inputs["tensors"][0].device,
        )
        actual = aclnn.call(changed)
    else:
        outputs = []
        states = []
        offset = 0
        for batch, length in enumerate(case["lengths"]):
            parts = [x[offset : offset + length] for x in inputs["tensors"][:4]]
            parts += [
                inputs["tensors"][4][batch : batch + 1],
                inputs["tensors"][5][batch : batch + 1],
                inputs["tensors"][6][offset : offset + length],
            ]
            out, state = aclnn.call(dict(tensors=parts, scale=inputs["scale"]))
            outputs.append(out)
            states.append(state)
            offset += length
        actual = [torch.cat(outputs), torch.cat(states)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected)), f"{relation} 不满足逐元素相等"

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""不依赖同一个golden的闭式解、因果性及输出边界检查。"""

import math

import pytest
import torch

from cgdr_generalization_utils import aclnn, assert_precision, make_inputs, recurrent_reference


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dim", [1, 128])
@pytest.mark.parametrize("gate", [0.0, -math.log(2)])
@pytest.mark.parametrize("scale", [0.0, 0.25, -2.0])
def test_one_token_closed_form(dtype, dim, gate, scale, aclnn):
    inputs = make_inputs(
        dict(lengths=(1,), nk=1, nv=2, dk=dim, dv=dim, state_dtype=dtype, scale=scale),
        device=f"npu:{torch.npu.current_device()}",
    )
    q, k, v, beta, state, _, g = inputs["tensors"]
    q.zero_()
    q[..., 0] = -1
    k.zero_()
    k[..., 0] = 1
    v.fill_(4)
    beta.fill_(0.5)
    state.fill_(2)
    g.fill_(gate)
    alpha = float(g[0, 0].cpu().double().exp())
    expected_state = torch.full(state.shape, 2 * alpha, dtype=torch.float64)
    expected_state[..., 0] = 2 + alpha
    expected_out = torch.full(v.shape, -(2 + alpha) * scale, dtype=torch.float64)
    # 同时验证独立参考本身，防止参考函数悄悄丢失scale、gate或head映射。
    reference = recurrent_reference(inputs)
    torch.testing.assert_close(reference[0], expected_out, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reference[1], expected_state, rtol=1e-12, atol=1e-12)
    actual = aclnn.call(inputs)
    for label, out, ref in zip(
        ("attention", "final_state"), actual, (expected_out, expected_state)
    ):
        assert_precision(out, ref, label)


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dk,dv", [(5, 8), (128, 128)])
def test_future_values_do_not_change_prefix(dtype, dk, dv, aclnn):
    inputs = make_inputs(
        dict(lengths=(129,), dk=dk, dv=dv, state_dtype=dtype),
        device=f"npu:{torch.npu.current_device()}",
    )
    first = aclnn.call(inputs)
    inputs["tensors"][2][32:].neg_()  # 在同一个chunk内部开始修改，检查上三角屏蔽。
    second = aclnn.call(inputs)
    assert torch.equal(first[0][:32], second[0][:32]), "未来value影响了过去attention"


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dk,dv", [(5, 8), (128, 128)])
def test_zero_state_and_value_stay_zero(dtype, dk, dv, aclnn):
    inputs = make_inputs(
        dict(
            lengths=(0, 63, 65),
            dk=dk,
            dv=dv,
            state_dtype=dtype,
            zero_state=True,
            zero_value=True,
            gate="minus_one",
        ),
        device=f"npu:{torch.npu.current_device()}",
    )
    for output in aclnn.call(inputs):
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output).item() == 0


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dk,dv", [(5, 8), (65, 127), (128, 128)])
def test_contiguous_outputs_respect_storage_bounds(dtype, dk, dv, aclnn):
    inputs = make_inputs(
        dict(lengths=(1, 63, 65), dk=dk, dv=dv, state_dtype=dtype),
        device=f"npu:{torch.npu.current_device()}",
    )
    outputs = []
    backings = []
    for tensor in (inputs["tensors"][2], inputs["tensors"][4]):
        shape = list(tensor.shape)
        shape[0] += 2
        backing = torch.full(shape, 17, dtype=tensor.dtype, device=tensor.device)
        output = backing[1:-1]
        assert output.is_contiguous() and output.storage_offset() > 0
        output.fill_(float("nan"))  # 输出的每个有效元素都必须被写入。
        outputs.append(output)
        backings.append(backing)
    expected = recurrent_reference(inputs)
    actual = aclnn.call(inputs, outputs=outputs)
    for name, out, ref, backing in zip(("attention", "final_state"), actual, expected, backings):
        assert_precision(out, ref, name)
        assert torch.all(backing[0] == 17) and torch.all(
            backing[-1] == 17
        ), "输出越过storage view边界"


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dk,dv", [(5, 8), (128, 128)])
@pytest.mark.parametrize(
    "layout",
    [
        ("q", 0),
        ("k", 1),
        ("v", 0),
        ("beta", 1),
        ("g", 0),
        ("state", 0),
        ("state", 1),
        ("lengths", 0),
        ("state_both", None),
    ],
)
def test_broadcast_and_state_strides(dtype, dk, dv, layout, aclnn):
    inputs = make_inputs(
        dict(lengths=(65, 65, 65), dk=dk, dv=dv, state_dtype=dtype),
        device=f"npu:{torch.npu.current_device()}",
    )
    name, axis = layout
    if name == "state_both":
        state = inputs["tensors"][4]
        b, nv, *tail = state.shape
        backing = torch.full(
            (2 * b + 1, 2 * nv + 1, *tail), 17, dtype=state.dtype, device=state.device
        )
        view = backing[1 : 2 * b + 1 : 2, 1 : 2 * nv + 1 : 2]
        view.copy_(state)
        inputs["tensors"][4] = view
        padding_before = backing.cpu().clone()
    else:
        index = ("q", "k", "v", "beta", "state", "lengths", "g").index(name)
        tensor = inputs["tensors"][index]
        inputs["tensors"][index] = tensor.narrow(axis, 0, 1).expand(tensor.shape)
        assert inputs["tensors"][index].stride(axis) == 0
    expected = recurrent_reference(inputs)
    actual = aclnn.call(inputs)
    for label, out, ref in zip(("attention", "final_state"), actual, expected):
        assert_precision(out, ref, label)
    if name == "state_both":
        assert torch.equal(backing.cpu(), padding_before)


@pytest.mark.ci
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("dk,dv", [(9, 9), (9, 11), (9, 12), (128, 1), (1, 128)])
@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 129])
def test_streaming_cube_transition(dtype, dk, dv, length, aclnn, record_property):
    # 同一小维度形状跨chunk边界，能覆盖静态分流改变及不同VF路径；
    # 正确性仍只与接口递推公式比较，不把实现中的分流公式作为golden。
    inputs = make_inputs(
        dict(lengths=(length,), dk=dk, dv=dv, state_dtype=dtype),
        device=f"npu:{torch.npu.current_device()}",
    )
    expected = recurrent_reference(inputs)
    actual = aclnn.call(inputs)
    for name, out, ref in zip(("attention", "final_state"), actual, expected):
        record_property(name, assert_precision(out, ref, name))

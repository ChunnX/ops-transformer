# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""非法输入仅做ACLNN第一段校验，不执行可能越界的kernel。"""

import json

import pytest
import torch

from cgdr_generalization_utils import aclnn, make_inputs

NAMES = ("q", "k", "v", "beta", "state", "lengths", "g", "out", "final_state")


def invalid_cases():
    result = []

    def add(kind, name, detail=None):
        label = f"{kind}_{name}_{detail}" if detail is not None else f"{kind}_{name}"
        result.append(pytest.param((kind, name, detail), id=label))

    for name in NAMES:
        if name != "g":
            add("null", name)
        add("rank", name)
        add("empty", name)
    for name in ("q", "k", "v", "beta", "state", "out", "final_state"):
        add("dtype", name, "float16")
    for name in ("q", "k", "v", "beta", "out"):
        add("dtype", name, "float32")
    for dtype in ("float16", "bfloat16", "int32"):
        add("dtype", "g", dtype)
    for dtype in ("float32", "int64"):
        add("dtype", "lengths", dtype)
    # 每个shape关联轴分别破坏，输出也通过ACLNN描述符直接测试。
    for name, axes in [
        ("k", (0, 1, 2)),
        ("v", (0,)),
        ("beta", (0, 1)),
        ("state", (1, 3)),
        ("lengths", (0,)),
        ("g", (0, 1)),
        ("out", (0, 1, 2)),
        ("final_state", (0, 1, 2, 3)),
    ]:
        for axis in axes:
            add("shape", name, axis)
    for field, value in [
        ("nk", 65),
        ("nv", 65),
        ("dk", 129),
        ("dv", 129),
        ("nk", 0),
        ("nv", 0),
        ("dk", 0),
        ("dv", 0),
        ("B", 0),
        ("T", 0),
    ]:
        add("dimension", field, value)
    add("head_ratio", "nk", 3)
    return result


@pytest.mark.ci
@pytest.mark.parametrize("invalid", invalid_cases())
def test_invalid_input_is_rejected(invalid, aclnn, record_property):
    inputs = make_inputs(
        dict(lengths=(2, 3), nk=2, nv=4, dk=8, dv=8), device=f"npu:{torch.npu.current_device()}"
    )
    tensors = list(inputs["tensors"])
    tensors += [torch.empty_like(tensors[2]), torch.empty_like(tensors[4])]
    kind, name, detail = invalid
    if kind in ("dimension", "head_ratio"):
        dims = dict(T=5, B=2, nk=2, nv=4, dk=8, dv=8)
        dims[name] = detail
        t, b, nk, nv, dk, dv = [dims[n] for n in ("T", "B", "nk", "nv", "dk", "dv")]
        shapes = [
            (t, nk, dk),
            (t, nk, dk),
            (t, nv, dv),
            (t, nv),
            (b, nv, dv, dk),
            (b,),
            (t, nv),
            (t, nv, dv),
            (b, nv, dv, dk),
        ]
        tensors = [
            torch.empty(shape, dtype=old.dtype, device=old.device)
            for old, shape in zip(tensors, shapes)
        ]
    else:
        index = NAMES.index(name)
        old = tensors[index]
        if kind == "null":
            tensors[index] = None
        elif kind == "dtype":
            tensors[index] = old.to(getattr(torch, detail))
        elif kind == "rank":
            tensors[index] = old.unsqueeze(0)
        else:
            shape = list(old.shape)
            shape[0 if kind == "empty" else detail] = 0 if kind == "empty" else shape[detail] + 1
            tensors[index] = torch.empty(shape, dtype=old.dtype, device=old.device)
    inputs["tensors"] = tensors[:7]
    code = aclnn.call(inputs, outputs=tensors[7:], preflight=True)
    record_property("invalid_case", json.dumps(invalid))
    record_property("aclnn_status", str(code))
    # 先覆盖“第一段必须拒绝”的功能约束；实际错误码入报告供对照文档，
    # 不用pytest.raises(Exception)把测试代码错误或设备初始化错误误算成拒绝成功。
    assert code != 0, f"非法输入被GetWorkspaceSize接受: {invalid}"

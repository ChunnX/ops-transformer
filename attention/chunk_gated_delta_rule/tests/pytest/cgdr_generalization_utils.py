# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CGDR 泛化测试工具：独立 FP64 递推参考、受约束输入及公开 ACLNN 调用。"""

from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

# BF16 的单位舍入误差约为 2^-8。固定阈值同时限制整体 RMS 和局部离群误差，
# 不用候选算子的结果调阈值，也不把所有接近零的元素误差除以接近零的参考值。
RMS_LIMIT = 0.02
MAX_LIMIT = 0.10
ABS_FLOOR = 2e-5


def recurrent_reference(inputs):
    """严格按接口公式逐 token 更新，使用真实输入值；不重新归一化、不重设 scale。"""
    q, k, v, beta, state, lengths, g = inputs["tensors"]
    q, k, v, beta, state = [x.detach().cpu().double() for x in (q, k, v, beta, state)]
    g = None if g is None else g.detach().cpu().double()
    lengths = lengths.cpu().tolist()
    nv, nk = v.shape[1], q.shape[1]
    q = q.repeat_interleave(nv // nk, dim=1)
    k = k.repeat_interleave(nv // nk, dim=1)
    output = torch.empty(v.shape, dtype=torch.float64)
    result = state.clone()
    offset = 0
    for batch, length in enumerate(lengths):
        current = state[batch].clone()
        for token in range(offset, offset + length):
            # state: [Nv,Dv,Dk]，每个 value head 使用 floor(head/(Nv/Nk)) 对应的 Q/K。
            alpha = 1.0 if g is None else g[token].exp()[:, None, None]
            decayed = current * alpha
            residual = v[token] - (decayed * k[token, :, None, :]).sum(-1)
            current = decayed + (beta[token, :, None] * residual)[:, :, None] * k[token, :, None, :]
            output[token] = (current * q[token, :, None, :]).sum(-1) * inputs["scale"]
        result[batch] = current
        offset += length
    assert offset == q.shape[0]
    return output, result


def precision_metrics(actual, reference):
    actual = actual.detach().cpu().double()
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all(), "算子输出存在 NaN/Inf"
    assert torch.isfinite(reference).all(), "参考结果存在 NaN/Inf"
    if actual.numel() == 0:
        return dict(relative_rms=0.0, normalized_max=0.0, max_abs=0.0)
    diff = actual - reference
    rms = reference.square().mean().sqrt().item()
    error_rms = diff.square().mean().sqrt().item()
    # 同时看全局误差及单元素误差；最大误差使用 max(|ref_i|, RMS(ref)) 归一化。
    normalized = diff.abs() / torch.maximum(
        reference.abs(), torch.full_like(reference, rms)
    ).clamp_min(ABS_FLOOR)
    return dict(
        relative_rms=error_rms / max(rms, ABS_FLOOR),
        normalized_max=normalized.max().item(),
        max_abs=diff.abs().max().item(),
    )


def assert_precision(actual, reference, label):
    metrics = precision_metrics(actual, reference)
    assert (
        metrics["relative_rms"] <= RMS_LIMIT and metrics["normalized_max"] <= MAX_LIMIT
    ), f"{label}: {metrics}; limits RMS={RMS_LIMIT}, normalized_max={MAX_LIMIT}"
    return metrics


def strided_view(value, axis):
    """非零 storage_offset + 指定轴间隔切片；保留 backing 用哨兵检查只读输入未被破坏。"""
    shape = list(value.shape)
    shape[axis] = shape[axis] * 2 + 1
    backing = torch.full(shape, 17, dtype=value.dtype, device=value.device)
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(1, 2 * value.shape[axis] + 1, 2)
    view = backing[tuple(slices)]
    view.copy_(value)
    assert view.storage_offset() > 0
    if value.shape[axis] > 1:
        assert not view.is_contiguous()
    return view, backing


def make_inputs(case, device="cpu"):
    """所有合法用例都在量化后核对约束；beta 避免 BF16 舍入到 0 或 1。"""
    gen = torch.Generator().manual_seed(case.get("seed", 20260908))
    lengths = tuple(case.get("lengths", (65,)))
    nk, nv, dk, dv = [
        case.get(key, default) for key, default in [("nk", 2), ("nv", 4), ("dk", 128), ("dv", 128)]
    ]
    total = sum(lengths)
    assert total > 0 and all(x >= 0 for x in lengths)
    assert 0 < nk <= 64 and 0 < nv <= 64 and nv % nk == 0
    assert 0 < dk <= 128 and 0 < dv <= 128

    def signed(shape):
        return torch.rand(shape, generator=gen) * 2 - 1

    def normalized(shape):
        value = signed(shape)
        if case.get("qk") == "one_hot":
            value.zero_()
            value[..., 0] = -1
        elif case.get("qk") == "alternating":
            value = torch.arange(shape[-1]).remainder(2).mul(2).sub(1).float().expand(shape).clone()
        value = torch.nn.functional.normalize(value, dim=-1).bfloat16()
        assert value.abs().max() <= 1
        assert torch.allclose(
            value.float().norm(dim=-1), torch.ones(shape[:-1]), atol=0.004, rtol=0
        )
        return value

    q = normalized((total, nk, dk))
    k = normalized((total, nk, dk))
    v = (signed((total, nv, dv)) * case.get("value_scale", 1)).bfloat16()
    beta = (0.01 + 0.98 * torch.rand((total, nv), generator=gen)).bfloat16()
    if case.get("beta") == "near_zero":
        beta.fill_(2**-12)
    if case.get("beta") == "near_one":
        beta.fill_(1 - 2**-8)
    assert ((beta > 0) & (beta < 1)).all()
    mode = case.get("gate", "random")
    g = -torch.rand((total, nv), generator=gen)
    if mode == "none":
        g = None
    elif mode == "zero":
        g.zero_()
    elif mode == "minus_one":
        g.fill_(-1)
    elif mode == "near_zero":
        g.fill_(-(2**-16))
    elif mode == "alternating":
        g = -torch.arange(total).remainder(2).float()[:, None].expand(total, nv).clone()
    assert g is None or ((g >= -1) & (g <= 0)).all()
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[
        case.get("state_dtype", "bfloat16")
    ]
    state = (signed((len(lengths), nv, dv, dk)) * case.get("state_scale", 0.25)).to(dtype)
    if case.get("zero_state"):
        state.zero_()
    if case.get("zero_value"):
        v.zero_()
    tensors = [q, k, v, beta, state, torch.tensor(lengths, dtype=torch.int32), g]
    tensors = [None if t is None else t.to(device) for t in tensors]
    # 支持输入所有轴的非连续；state仅对接口承诺的0/1轴构造有洞stride。
    backings = []
    names = ["q", "k", "v", "beta", "state", "lengths", "g"]
    for name, axis in case.get("layouts", {}).items():
        index = names.index(name)
        tensors[index], backing = strided_view(tensors[index], axis)
        backings.append(backing)
    return dict(
        tensors=tensors,
        scale=float(C.c_float(case.get("scale", dk**-0.5)).value),
        backings=backings,
    )


class AclnnCgdr:
    """直接调用公开两段式接口，非法输入仅调用第一段，绝不发射非法kernel。"""

    def __init__(self):
        vendor = os.environ.get("CGDR_TEST_VENDOR")
        self.path = str(Path(vendor) / "op_api/lib/libcust_opapi.so") if vendor else "libopapi.so"
        self.lib = C.CDLL(self.path)
        ptr, i64, u64 = C.c_void_p, C.c_int64, C.c_uint64
        self.lib.aclCreateTensor.argtypes = [
            C.POINTER(i64),
            u64,
            C.c_int,
            C.POINTER(i64),
            i64,
            C.c_int,
            C.POINTER(i64),
            u64,
            ptr,
        ]
        self.lib.aclCreateTensor.restype = ptr
        self.lib.aclDestroyTensor.argtypes = [ptr]
        self.lib.aclDestroyTensor.restype = C.c_int
        self.lib.aclDestroyAclOpExecutor.argtypes = [ptr]
        self.lib.aclDestroyAclOpExecutor.restype = C.c_int
        self.lib.aclnnChunkGatedDeltaRuleGetWorkspaceSize.argtypes = (
            [ptr] * 7 + [C.c_float] + [ptr] * 2 + [C.POINTER(u64), C.POINTER(ptr)]
        )
        self.lib.aclnnChunkGatedDeltaRuleGetWorkspaceSize.restype = C.c_int
        self.lib.aclnnChunkGatedDeltaRule.argtypes = [ptr, u64, ptr, ptr]
        self.lib.aclnnChunkGatedDeltaRule.restype = C.c_int

    def descriptor(self, value):
        if value is None:
            return None
        dims = (C.c_int64 * value.ndim)(*value.shape)
        strides = (C.c_int64 * value.ndim)(*value.stride())
        storage = (C.c_int64 * 1)(value.untyped_storage().nbytes() // value.element_size())
        dtype = {
            torch.bfloat16: 27,
            torch.float32: 0,
            torch.float16: 1,
            torch.int32: 3,
            torch.int64: 9,
        }[value.dtype]
        result = self.lib.aclCreateTensor(
            dims,
            value.ndim,
            dtype,
            strides,
            value.storage_offset(),
            2,
            storage,
            1,
            value.untyped_storage().data_ptr(),
        )
        assert result, "aclCreateTensor returned nullptr"
        return result

    def call(self, inputs, outputs=None, preflight=False):
        tensors = inputs["tensors"]
        if outputs is None:
            outputs = [
                torch.empty(tensors[2].shape, dtype=torch.bfloat16, device=tensors[2].device),
                torch.empty(tensors[4].shape, dtype=tensors[4].dtype, device=tensors[4].device),
            ]
        torch.npu.synchronize()  # 刷出 torch_npu 的异步提交队列后，直接在同一stream上调用ACL。
        descriptors = []
        executor, size = C.c_void_p(), C.c_uint64()
        executed = False
        try:
            for value in list(tensors) + list(outputs):
                descriptors.append(self.descriptor(value))
            status = self.lib.aclnnChunkGatedDeltaRuleGetWorkspaceSize(
                *descriptors[:7],
                inputs["scale"],
                *descriptors[7:],
                C.byref(size),
                C.byref(executor),
            )
            if preflight:
                return status
            assert status == 0, f"GetWorkspaceSize returned {status}"
            workspace = torch.empty(size.value, dtype=torch.uint8, device=tensors[0].device)
            status = self.lib.aclnnChunkGatedDeltaRule(
                workspace.data_ptr(), size.value, executor, torch.npu.current_stream().npu_stream
            )
            executed = True  # 执行器由第二段接口管理；仅第一段成功但未执行时需手工销毁。
            assert status == 0, f"aclnnChunkGatedDeltaRule returned {status}"
            torch.npu.synchronize()
            return outputs
        finally:
            if executor.value and not executed:
                self.lib.aclDestroyAclOpExecutor(executor)
            for desc in descriptors:
                if desc:
                    self.lib.aclDestroyTensor(desc)


@pytest.fixture(scope="module")
def aclnn():
    if not torch.npu.is_available():
        pytest.skip("需要Ascend NPU和torch_npu")
    torch.npu.set_device(int(os.environ.get("CGDR_TEST_DEVICE", "0")))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vendor = os.environ.get("CGDR_TEST_VENDOR")
    if vendor:
        assert str(Path(vendor).resolve()) in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(
            ":"
        ), "需在进程启动前设置ASCEND_CUSTOM_OPP_PATH"
    api = AclnnCgdr()
    try:
        yield api
    finally:
        audit = os.environ.get("CGDR_TEST_AUDIT")
        if audit:
            maps = Path("/proc/self/maps").read_text()
            libraries = sorted(
                set(
                    # maps 路径后可能带“(deleted)”，保留完整路径用于版本审计。
                    line.split(maxsplit=5)[-1]
                    for line in maps.splitlines()
                    if "/vendors/" in line or "libopapi" in line
                )
            )
            Path(audit).write_text(
                json.dumps(
                    dict(
                        opapi=api.path,
                        vendor=vendor,
                        libraries=libraries,
                        torch=torch.__version__,
                        torch_npu=torch_npu.__version__,
                    ),
                    indent=2,
                )
                + "\n"
            )
        torch.set_num_threads(threads)

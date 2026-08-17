#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MoeTutelCombineX 多通路 golden（TestSpec 范式）。

通路支持面（照抄 01_requirement.md §3.3 / §5，与下方 __spec__ 注册项逐条相等）：

| 通路   | 支持 | 依据                                                          |
|--------|------|---------------------------------------------------------------|
| kernel | 是   | op_kernel/arch35/ 有 Ascend950 regbase 实现                   |
| geir   | 是   | op_graph/moe_tutel_combine_x_proto.h 有 REG_OP(MoeTutelCombineX) |
| aclnn  | 否   | 无 op_api/、无 docs/aclnnMoeTutelCombineX.md，canndev 侧亦无   |
| e2e    | 否   | torch_npu 二进制无 aclnnMoeTutelCombineX 符号                 |

语义：
    valid(k, s) = indices[k, s] >= 0 and locations[k, s] < C
    x_grad[s, :] = sum_k valid(k, s) ? gates[k, s] * y_grad[indices[k, s], locations[k, s], :] : 0
累加从 0 起，K = 0 或全部路由无效时输出为 0。
"""

import zlib

import numpy as np
import torch

__spec__ = {
    # kernel + geir 共用同一个注册键（算子蛇形名），geir 不另写。
    "moe_tutel_combine_x": "MoeTutelCombineXKernelSpec",
}

# ── 判据 tolerance ───────────────────────────────────────────────────────────
# 浮点输出全部配 cross_check：它是唯一会去拉三方数据判精度的标准，不配则
# third_party 写了也不会被取用，三方精度那一栏永远是空的。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


# ── 路由输入定制 ─────────────────────────────────────────────────────────────
# indices / locations 不能用框架的通用随机数：
#   · indices 的合法域是 [-1, E)，随机 int32 会给出 >= E 的专家号，而判据
#     `expertId >= 0 && location < C` 照抄 Tutel/canndev，**不判 expertId < E**，
#     越界值会直接被当成合法下标去读 y_grad（越界读，golden 侧同样是未定义）；
#   · locations 的合法域是 [0, +inf)，>= C 表示该 token 被丢弃（Tutel 的 drop 语义），
#     但 **负数不合法**——判据同样不判 location >= 0，负 location 会读到 y_grad 之前。
# 因此路由表由本函数按用例名里的场景后缀生成，其余两个浮点输入沿用 CSV 的随机范围。
_ROUTE_SCENARIOS = ("all", "mix", "none", "neg", "ooc", "edge", "dup", "sparse")
_ROUTE_SUFFIX = "_r"


def _scenario_of(testcase_name):
    """用例名以 `_r<scenario>` 结尾时取该场景，否则默认 mix。"""
    tail = str(testcase_name or "").rsplit(_ROUTE_SUFFIX, 1)
    if len(tail) == 2 and tail[1] in _ROUTE_SCENARIOS:
        return tail[1]
    return "mix"


def _make_route(
    scenario, expert_num, capacity, top_k, samples, idx_dtype, loc_dtype, seed_key
):
    """按场景生成 [K, S] 的 indices / locations。种子取用例名 CRC32，跨轮可复现。"""
    rng = np.random.default_rng(zlib.crc32(str(seed_key).encode("utf-8")))
    shape = (top_k, samples)
    if top_k == 0 or samples == 0:
        return (np.zeros(shape, dtype=idx_dtype), np.zeros(shape, dtype=loc_dtype))

    hi_e = max(expert_num, 1)
    hi_c = max(capacity, 1)
    # 基准：全合法。
    idx = rng.integers(0, hi_e, size=shape)
    loc = rng.integers(0, hi_c, size=shape)
    if expert_num == 0:
        # E == 0 时任何非负专家号都越界，只能整表判无效。
        idx = np.full(shape, -1)

    if scenario == "all":
        pass
    elif scenario == "neg":
        idx = np.full(shape, -1)
    elif scenario == "ooc":
        loc = rng.integers(capacity, capacity + 8, size=shape)
    elif scenario == "none":
        half = rng.random(shape) < 0.5
        idx = np.where(half, -1, idx)
        loc = np.where(half, loc, rng.integers(capacity, capacity + 8, size=shape))
    elif scenario == "edge":
        # 只取判据的边界值：expert 取 {-1, 0, E-1}，location 取 {0, C-1, C}。
        idx = rng.choice(np.array([-1, 0, hi_e - 1]), size=shape)
        loc = rng.choice(np.array([0, hi_c - 1, capacity]), size=shape)
    elif scenario == "dup":
        # 一个 token 的 K 项全部指向同一个 (expert, location)：重复 gather + 纯累加。
        idx = np.repeat(rng.integers(0, hi_e, size=(1, samples)), top_k, axis=0)
        loc = np.repeat(rng.integers(0, hi_c, size=(1, samples)), top_k, axis=0)
    elif scenario == "sparse":
        keep = rng.random(shape) < 0.1
        idx = np.where(keep, idx, -1)
    else:  # mix
        drop_i = rng.random(shape) < 0.25
        drop_l = rng.random(shape) < 0.25
        idx = np.where(drop_i, -1, idx)
        loc = np.where(drop_l, rng.integers(capacity, capacity + 8, size=shape), loc)

    if expert_num == 0:
        idx = np.full(shape, -1)
    return (idx.astype(idx_dtype, copy=False), loc.astype(loc_dtype, copy=False))


def _to_tensor(value):
    """numpy/torch 统一成 torch 张量。

    非 Promote 通路（如 --compare stat_rel_err）里 bf16 输入是 ml_dtypes.bfloat16 的
    numpy 数组，torch.from_numpy 不认；经 fp32 中转再落回 torch.bfloat16 是无损的
    （bf16 -> fp32 -> bf16 逐位可逆）。
    """
    if isinstance(value, torch.Tensor):
        return value
    arr = np.ascontiguousarray(value)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


def _to_numpy(tensor, dtype_name=None):
    """torch 张量转 numpy；torch.bfloat16 没有 numpy 对应体，按位搬到 ml_dtypes.bfloat16。"""
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            arr = tensor.view(torch.int16).numpy().view(np.dtype("bfloat16"))
        else:
            arr = tensor.numpy()
    else:
        arr = np.asarray(tensor)
    if dtype_name is not None and arr.dtype.name != dtype_name:
        arr = arr.astype(dtype_name)
    return arr


def _float_workspace(tensor):
    """浮点输入的工作精度：只向上兜底，绝不向下砍。

    tolerance 配 cross_check 时框架按 golden_mode=Promote 把输入抬一档喂进来
    （fp16/bf16 -> fp32、fp32 -> fp64），此处保持不动；只有 CPU 上残缺的
    half/bfloat16 才抬到 fp32。
    """
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


def _compute(y_grad, gates, indices, locations, **kwargs):
    """torch 竞品接口实现（R3：禁 numpy 纯公式）。

    y_grad   : [E, C, H]
    gates    : [K, S]
    indices  : [K, S] int32
    locations: [K, S] int32
    返回     : [list([S, H])]，顺序照 def.cpp 的输出序。
    """
    del kwargs
    y_grad = _float_workspace(_to_tensor(y_grad))
    gates = _float_workspace(_to_tensor(gates))
    indices = _to_tensor(indices).to(torch.int64)
    locations = _to_tensor(locations).to(torch.int64)

    capacity = y_grad.shape[1]
    hidden = y_grad.shape[2]
    samples = gates.shape[1]
    work_dtype = y_grad.dtype

    out = torch.zeros((samples, hidden), dtype=work_dtype)
    # E*C == 0 时 y_grad 一行都没有：E==0 无合法专家、C==0 使 location < C 恒假，
    # 两种情形下全部路由必然无效，输出恒为 0（此时 index_select 也不能调用）。
    if (
        gates.shape[0] == 0
        or samples == 0
        or hidden == 0
        or y_grad.shape[0] * capacity == 0
    ):
        return [out]

    flat = y_grad.reshape(-1, hidden)
    valid = torch.logical_and(indices >= 0, locations < capacity)
    # 无效项的行号先夹到 0，避免 index_select 报越界；其贡献随后被 where 抹成 0。
    row = torch.where(valid, indices * capacity + locations, torch.zeros_like(indices))
    zero_row = torch.zeros((samples, hidden), dtype=work_dtype)

    for k in range(gates.shape[0]):
        gathered = torch.index_select(flat, 0, row[k])
        weighted = torch.mul(gathered, gates[k].reshape(-1, 1))
        # 乘法与加法分两步写，不用 addcmul 之类的融合算子（FMA 会改运算序列）。
        out = torch.add(out, torch.where(valid[k].reshape(-1, 1), weighted, zero_row))
    return [out]


class _Compose:
    """竞品标杆：用与 _compute 不同的 torch 路径拼出同一语义。

    _compute 走 index_select + 逐 k 循环累加；这里走高级索引一次取出 [K, S, H]
    再用 masked_fill + sum 一次归约，两条路径相互独立，避免"同错一起过"。
    输出必须 cast 回 NPU 的输出 dtype，否则竞品对 golden 的误差远小于 NPU，
    cross_check 的 ratio 会凭空爆表。
    """

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, y_grad, gates, indices, locations, **kwargs):
        del kwargs
        y_grad_t = _to_tensor(y_grad)
        out_dtype = y_grad_t.dtype
        work = _float_workspace(y_grad_t)
        gates_t = _float_workspace(_to_tensor(gates))
        indices_t = _to_tensor(indices).to(torch.int64)
        locations_t = _to_tensor(locations).to(torch.int64)

        capacity = work.shape[1]
        hidden = work.shape[2]
        samples = gates_t.shape[1]
        # E*C == 0：无合法专家 / location < C 恒假，全部路由无效，输出恒为 0。
        if (
            gates_t.shape[0] == 0
            or samples == 0
            or hidden == 0
            or work.shape[0] * capacity == 0
        ):
            return [torch.zeros((samples, hidden), dtype=out_dtype)]

        valid = torch.logical_and(indices_t >= 0, locations_t < capacity)
        safe_expert = torch.where(valid, indices_t, torch.zeros_like(indices_t))
        safe_loc = torch.where(valid, locations_t, torch.zeros_like(locations_t))
        gathered = work[safe_expert, safe_loc]  # [K, S, H]
        scaled = gathered * gates_t.unsqueeze(-1)  # [K, S, H]
        masked = scaled.masked_fill(~valid.unsqueeze(-1), 0)
        return [masked.sum(dim=0).to(out_dtype)]


class MoeTutelCombineXKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。

    入参名取自 op_host/moe_tutel_combine_x_def.cpp（无属性，全部按位置传）。
    """

    def customize_inputs(y_grad, gates, indices, locations, **kwargs):
        """按用例名的 `_r<scenario>` 后缀重写路由表，浮点输入保持框架随机值。"""
        scenario = _scenario_of(kwargs.get("testcase_name"))
        new_idx, new_loc = _make_route(
            scenario,
            int(y_grad.shape[0]),
            int(y_grad.shape[1]),
            int(gates.shape[0]),
            int(gates.shape[1]),
            indices.dtype,
            locations.dtype,
            kwargs.get("testcase_name"),
        )
        return (y_grad, gates, new_idx, new_loc)

    def golden(*inputs, **kwargs):
        outs = _compute(*inputs, **kwargs)
        out_dtypes = kwargs.get("output_dtypes") or []
        out_dtypes = [
            d[0] if isinstance(d, (list, tuple)) else str(d) for d in out_dtypes
        ]
        result = []
        for i, o in enumerate(outs):
            result.append(_to_numpy(o, out_dtypes[i] if i < len(out_dtypes) else None))
        return result

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路：本算子无 op_host/op_api/、无 docs/aclnnMoeTutelCombineX.md，
#   canndev 侧同样没有同名 aclnn，README「调用说明」只列图模式调用。
# 【不存在】e2e 通路：strings libtorch_npu.so | grep -c aclnnMoeTutelCombineX 结果为 0
#   （对照组 aclnnMoeFinalizeRouting/V2/V4 有命中，证明查法有效）。

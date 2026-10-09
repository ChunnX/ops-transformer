#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoeTokenPermuteWithEp golden（精度标杆）。

纯搬运 + 稳定排序实现（paddedMode=false 唯一支持模式，无浮点算术）：
  - topK：indices 1 维视作 topK=1，2 维为 dim(1)；totalLength = N * topK
  - 两次稳定 argsort：order = argsort_stable(flatten(indices))；
    sorted_indices = argsort_stable(order)（= rank，固定 INT32 输出）
  - 窗口：range 提供时 [start,end)=(range[0],range[1])，为空时 end=num_out_tokens；
    numOutTokens = end-start，<=0 时环绕 +totalLength，clamp [0,totalLength]；
    numOutTokens == totalLength 时窗口为全量 [0,totalLength)
  - 搬运：permute_tokens[r] = tokens[order[start+r] // topK]；
    permute_probs[r] = probs.flatten()[order[start+r]]
  - probs 缺省时 permute_probs 返回 None（比对按 SUPPRESSED 跳过）

精度标准：纯搬运无浮点算术，全输出 binary 精确比对（rtol=0/atol=0）；
bfloat16 经 float32 无损桥接。
"""

import tensorflow as tf

__spec__ = {
    "moe_token_permute_with_ep": "MoeTokenPermuteWithEpTfTestSpec",
    "aclnnMoeTokenPermuteWithEp": "AclnnMoeTokenPermuteWithEpTfTestSpec",
}

_TOLERANCE = {
    "float32": {"standard": "binary_equal", "rtol": 0, "atol": 0},
    "float16": {"standard": "binary_equal", "rtol": 0, "atol": 0},
    "bfloat16": {"standard": "binary_equal", "rtol": 0, "atol": 0},
    "int32": {"standard": "binary_equal", "rtol": 0, "atol": 0},
}

# tf.bfloat16 哨兵（用于识别 numpy ml_dtypes.bfloat16 输入并桥接）
try:
    import ml_dtypes

    _BF16_NUMPY = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover - TTK 环境必然提供
    _BF16_NUMPY = None


def _tf_to_numpy(tensor):
    """tf.Tensor -> numpy；bfloat16 经 fp32 无损桥接（纯搬运无舍入，往返 bit-exact）。"""
    if tensor is None:
        return None
    arr = tensor.numpy()
    if tensor.dtype == tf.bfloat16:
        arr = arr.astype("float32")
        if _BF16_NUMPY is not None:
            return arr.astype(_BF16_NUMPY)
    return arr


def _to_tf(t):
    """框架 Tensor / numpy -> tf.Tensor（torch.Tensor / tf.Tensor / numpy 统一入口）。

    torch/numpy 的 dtype 与 tf dtype 同名对齐，tf.constant 自动推断；
    numpy 无原生 bf16，torch.bfloat16 / ml_dtypes.bfloat16 经 fp32 无损桥接。"""
    if t is None:
        return None
    if hasattr(t, "detach"):  # torch.Tensor
        arr = t.detach().to("cpu")
        if str(arr.dtype) == "torch.bfloat16":
            # numpy 无原生 bf16，经 fp32 无损桥接（fp16 numpy 原生支持，无需中转）
            return tf.constant(arr.float().numpy(), dtype=tf.bfloat16)
        return tf.constant(arr.numpy())  # dtype 由 numpy 自动推断，无需映射表
    if hasattr(
        t, "numpy"
    ):  # tf.Tensor：已是目标形态，直接返回（避免 tf→numpy→tf 往返拷贝）
        return t
    if _BF16_NUMPY is not None and hasattr(t, "dtype") and t.dtype == _BF16_NUMPY:
        return tf.constant(t.astype("float32"), dtype=tf.bfloat16)
    return tf.constant(t)


def _normalize_range(range_value):
    """把 CSV/ACLNN 传入的 range 属性规整为 (start, end) 或 None。

    支持 list/tuple；空列表/None 均视为回退语义（返回 None，num_out_tokens 生效）。
    """
    if range_value is None:
        return None
    if hasattr(range_value, "numpy"):  # tf.Tensor
        range_value = range_value.numpy().flatten().tolist()
    if isinstance(range_value, (int, float)):
        return None
    range_list = [int(v) for v in range_value]
    if len(range_list) == 0:
        return None
    if len(range_list) != 2:
        raise ValueError("range must be a ListInt of size 2")
    return range_list


def _optional_probs(probs):
    """可选 probs 归一化：None / 空 tensor 统一为 None（缺席语义）。

    调用点均已前置转换为 tf.Tensor（_numpy_to_tf / _to_tf），无需 numpy 分支。
    """
    if probs is None:
        return None
    if probs.shape.num_elements() == 0:
        return None
    return probs


class _Prepared:
    """_prepare_inputs 的产物：dtype 已就位、窗口端点已预计算的计算输入。"""

    __slots__ = ("tokens", "indices", "probs", "start", "end")

    def __init__(self, tokens, indices, probs, start, end):
        self.tokens = tokens
        self.indices = indices
        self.probs = probs
        self.start = start
        self.end = end


def _prepare_inputs(
    tokens,
    indices,
    probs,
    num_out_tokens,
    range_value,
    padded_mode=False,
):
    """输入预处理：窗口端点预计算（dtype 转换已由调用方前置完成）。

    返回 _Prepared；probs 输入为 None 时 permute_probs 亦为 None。
    """
    if padded_mode:
        raise ValueError(
            "padded_mode=true is not supported (SE 5.4: tiling 报错 561002)"
        )

    # range 语义：range 提供时 [start, end)；为空时 start=0、end=0（[0,0] 环绕公式
    # 得全量输出，与 tiling/910 实际行为一致——num_out_tokens 在真实链路中不生效。
    # 保持和 A2 兼容）
    if range_value is not None:
        start, end = int(range_value[0]), int(range_value[1])
    else:
        start, end = 0, 0

    return _Prepared(tokens, indices, probs, start, end)


def _compute_core(prepared):
    """纯计算骨架：两次稳定 argsort + 窗口选择 + gather 搬运（零 dtype 载体转换）。

    稳定排序顺序只取决于元素比较结果（int32→int64 单调映射，order/rank 逐位一致）；
    tf.argsort 输出恒为 int32，满足 sorted_indices 固定 INT32 规格（SE 6.5）。
    返回 (permute_tokens, rank, permute_probs)；probs 缺省时第三项为 None。
    """
    indices_shape = prepared.indices.shape
    if len(indices_shape) == 1:
        total_length = int(indices_shape[0])
        top_k = 1
    else:
        total_length = int(indices_shape[0]) * int(indices_shape[1])
        top_k = int(indices_shape[1])

    flat_idx = tf.reshape(prepared.indices, [-1])
    order = tf.argsort(flat_idx, axis=-1, direction="ASCENDING", stable=True)

    rank = tf.argsort(order, axis=-1, direction="ASCENDING", stable=True)

    # numOutTokens：end-start，<=0 时环绕 +totalLength，clamp 到 [0, totalLength]
    num_out = prepared.end - prepared.start
    if num_out <= 0:
        num_out += total_length
    num_out = max(0, min(num_out, total_length))

    # 窗口：全量分支（num_out == total_length）取 [0, totalLength)；切片分支取
    # [start, min(start + num_out, totalLength))
    if num_out == total_length:
        lo, hi = 0, total_length
    else:
        lo, hi = prepared.start, min(prepared.start + num_out, total_length)
    sel = order[lo:hi]

    permute_tokens = tf.gather(prepared.tokens, sel // top_k, axis=0)
    permute_probs = None
    if prepared.probs is not None:
        permute_probs = tf.gather(tf.reshape(prepared.probs, [-1]), sel, axis=0)
    # sorted_indices 输出规格固定 INT32（SE 6.5）；tf.argsort 输出恒为 int32，cast 为规格兜底
    return permute_tokens, tf.cast(rank, tf.int32), permute_probs


def _prepare_typed(tokens, indices, probs, num_out_tokens, range_, padded_mode):
    """统一输入管线：duck-typed 转换（_to_tf）-> _prepare_inputs。"""
    return _prepare_inputs(
        _to_tf(tokens),
        _to_tf(indices),
        _optional_probs(_to_tf(probs)),
        num_out_tokens,
        _normalize_range(range_),
        padded_mode,
    )


def _run_golden_flow(tokens, indices, probs, num_out_tokens, range_, padded_mode):
    """golden 流：numpy/framework Tensor 入 -> [numpy, numpy, numpy|None] 出。"""
    pt, si, pp = _compute_core(
        _prepare_typed(tokens, indices, probs, num_out_tokens, range_, padded_mode)
    )
    return [_tf_to_numpy(pt), _tf_to_numpy(si), _tf_to_numpy(pp)]


class MoeTokenPermuteWithEpTfTestSpec:
    """Kernel/GEIR 共用 TestSpec（CSV op_name = moe_token_permute_with_ep）。

    输入按 def.cpp 顺序位置传入，属性按名传入（range / num_out_tokens / padded_mode）。
    golden 接收 numpy，返回 numpy list [permute_tokens, sorted_indices, permute_probs]
    （与 def.cpp Output 顺序一致）；probs 缺省时第三个元素为 None（比对跳过）。
    """

    @staticmethod
    def golden(
        tokens,
        indices,
        probs=None,
        *,
        range=None,
        num_out_tokens=0,
        padded_mode=False,
        **kwargs,
    ):
        return _run_golden_flow(
            tokens, indices, probs, num_out_tokens, range, padded_mode
        )

    class ThirdPartyImpl:
        """tf provider：输入转换前置 __init__，__call__ 链内零转换。"""

        def __init__(
            self,
            tokens=None,
            indices=None,
            probs=None,
            num_out_tokens=0,
            range=None,
            padded_mode=False,
            **kwargs,
        ):
            self._prepared = _prepare_typed(
                tokens, indices, probs, num_out_tokens, range, padded_mode
            )

        def __call__(self, **kwargs):
            return list(_compute_core(self._prepared))

    third_party = {"tf": ThirdPartyImpl}
    tolerance = _TOLERANCE


class AclnnMoeTokenPermuteWithEpTfTestSpec:
    """ACLNN TestSpec（api_name = aclnnMoeTokenPermuteWithEp）。

    参数顺序与 aclnnMoeTokenPermuteWithEpGetWorkspaceSize 一致
    （去掉 workspaceSize/executor）：3 输入 + 3 属性 + 3 输出。
    golden 收框架 Tensor（torch/tf 均可，经 numpy 桥接），返回 numpy list；
    probsOptional 缺省时 permute_probs 为 None（比对跳过）。
    """

    @staticmethod
    def golden(
        tokens,
        indices,
        probsOptional=None,
        rangeOptional=None,
        numOutTokens=0,
        paddedMode=False,
        permuteTokensOut=None,
        sortedIndicesOut=None,
        permuteProbsOut=None,
        **kwargs,
    ):
        return _run_golden_flow(
            tokens, indices, probsOptional, numOutTokens, rangeOptional, paddedMode
        )

    class ThirdPartyImpl:
        """tf provider：输入转换前置 __init__，__call__ 链内零转换。"""

        def __init__(
            self,
            tokens=None,
            indices=None,
            probsOptional=None,
            rangeOptional=None,
            numOutTokens=0,
            paddedMode=False,
            **kwargs,
        ):
            self._prepared = _prepare_typed(
                tokens, indices, probsOptional, numOutTokens, rangeOptional, paddedMode
            )

        def __call__(self, **kwargs):
            return list(_compute_core(self._prepared))

    third_party = {"tf": ThirdPartyImpl}
    tolerance = _TOLERANCE

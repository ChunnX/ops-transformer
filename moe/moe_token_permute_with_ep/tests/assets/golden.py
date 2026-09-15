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

计算语义（paddedMode=false 唯一支持模式，与 SE 第 7 章 golden 及 kernel CopyOut
分支逐行一致，纯搬运 + 稳定排序，无任何浮点算术）：
  1) topK 与 totalLength：indices 1 维视作 topK=1，2 维 topK=dim(1)；
     totalLength = N * topK
  2) 两次稳定 argsort：
     - order          = argsort_stable(flatten(indices))  按专家索引升序，
       相同值保持原相对顺序（与 kernel 归并排序稳定语义对齐）
     - sorted_indices = argsort_stable(order)             = 各原位置的秩 rank，
       固定 INT32 输出（order 为无重复排列，结果稳定唯一）
  3) 窗口语义（与 kernel CopyOut 分支一致）：
     - range 提供时 [start, end) = (range[0], range[1])；为空时 start=0、
       end=num_out_tokens（语义回退 aclnnMoeTokenPermute）
     - numOutTokens = end - start；<=0 时环绕 +totalLength；clamp 到
       [0, totalLength]
     - 全量分支（numOutTokens == totalLength，含 num_out_tokens=0 与 range 窗口
       恰为全长如 range=[2, total+2]）：窗口 [0, totalLength)，输出无偏移全量
       permute（kernel ifNumOutTokens=false 路径，注意与"窗口切片"直觉不同）
     - 切片分支（numOutTokens != totalLength，含 num_out_tokens 负切片）：
       窗口 [start, min(start + numOutTokens, totalLength))，输出行 r 对应排序后
       第 start + r 位
  4) 纯搬运：permute_tokens[r] = tokens[order[start + r] // topK]，
     permute_probs[r] = probs.flatten()[order[start + r]]
  5) probs 缺省（None / 空 tensor）：permute_probs 返回 None（比对引擎按
     SUPPRESSED 跳过该输出）
  6) 环绕场景（range[0] >= range[1]，如 range=[4,1]，SE 5.4 口径）：kernel 实际
     输出 shape 的 dim(0) 为 numOutTokens（含环绕补足部分），但仅有效窗口
     [start, min(start + numOutTokens, totalLength)) 内的位置被写入，超出部分为
     未初始化内存；本 golden 返回有效窗口内的填充结果（元素个数 = 有效窗口长度）。
     精度验收用例应避开环绕场景（A2 ST 用例全部为 start < end），环绕场景因含
     未定义内容不作为 binary 比对对象

精度标准：纯搬运 + 稳定排序类算子，无浮点计算误差，全输出 binary 精确比对
（rtol=0 / atol=0，bit-exact）；sorted_indices 为整数秩精确比对；bfloat16 经
float32 无损桥接（bf16 是 fp32 的截断，往返 bit-exact）。

sorted_indices 数学定义自检：rank[order[j]] = j 对任意 j 成立。
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

_TORCH_DT_MAP = {
    "torch.bfloat16": tf.bfloat16,
    "torch.float16": tf.float16,
    "torch.float32": tf.float32,
    "torch.float64": tf.float64,
    "torch.int32": tf.int32,
    "torch.int64": tf.int64,
}


def _numpy_to_tf(arr):
    """numpy -> tf.Tensor；ml_dtypes.bfloat16 经 fp32 无损桥接。"""
    if arr is None:
        return None
    if _BF16_NUMPY is not None and arr.dtype == _BF16_NUMPY:
        return tf.constant(arr.astype("float32"), dtype=tf.bfloat16)
    return tf.constant(arr)


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
    """框架 Tensor / numpy -> tf.Tensor（duck typing，保留原始 dtype）。

    兼容 torch.Tensor / tf.Tensor / numpy（含 ml_dtypes.bfloat16）。
    """
    if t is None:
        return None
    if hasattr(t, "detach"):  # torch.Tensor：保留原始 dtype 到 tf
        arr = t.detach().to("cpu")
        np_arr = (
            arr.float().numpy()
            if str(arr.dtype) in ("torch.bfloat16", "torch.float16")
            else arr.numpy()
        )
        return tf.constant(np_arr, dtype=_TORCH_DT_MAP.get(str(arr.dtype), tf.float32))
    if hasattr(t, "numpy"):  # tf.Tensor
        return tf.constant(t.numpy())
    # numpy（ml_dtypes.bfloat16 桥接回 fp32 再标 bf16 dtype）
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
    """可选 probs 输入归一化：空 tensor / 0 元素 / None 统一为 None（缺席语义）。"""
    if probs is None:
        return None
    if (
        hasattr(probs, "shape")
        and hasattr(probs.shape, "num_elements")
        and probs.shape.num_elements() == 0
    ):
        return None
    if hasattr(probs, "numpy") and probs.numpy().size == 0:
        return None
    if hasattr(probs, "size") and probs.size == 0:
        return None
    return probs


class _Prepared:
    """_prepare_inputs 的产物：可直接参与计算的预备数据（dtype 已就位、窗口端点已预计算）。"""

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
    """输入预处理：入参校验 + 窗口端点预计算（dtype 转换已由调用方前置完成）。

    所有 dtype 转换集中于此之前（golden 流在 golden 入口调用、third_party 流在
    __init__ 调用），保证 _compute_core 调用链内零转换。
    返回 _Prepared；probs 输入为 None 时 permute_probs 亦为 None。
    """
    if padded_mode:
        raise ValueError("padded_mode=true is not supported (SE 5.4: tiling 报错 561002)")

    # range 语义（SE 5.2）：range 提供时 [start, end)；为空时 start=0、end=num_out_tokens
    if range_value is not None:
        start, end = int(range_value[0]), int(range_value[1])
    else:
        start, end = 0, int(num_out_tokens)

    return _Prepared(tokens, indices, probs, start, end)


def _compute_core(prepared):
    """纯计算核心（零 dtype 转换）：两次稳定 argsort + kernel CopyOut 窗口搬运。

    与 SE 第 7 章 golden_moe_token_permute_with_ep 及 TTK golden 计算核心逐行一致：
      - 稳定排序：tf.argsort(..., stable=True) 对齐 kernel 归并排序稳定语义，
        相同专家索引保持原相对顺序
      - 第二次 argsort 输入为无重复排列，结果稳定唯一（= rank，固定 INT32 输出，
        与输入 dtype 无关）
      - 搬运窗口为 host 标量分支 + gather（纯搬运，无浮点算术）
    返回 (permute_tokens, sorted_indices, permute_probs)；probs 缺省时第三项为 None。
    """
    # topK 与 totalLength（indices 1 维时 topK=1，2 维时 topK=dim(1)，SE 1.2）
    indices_shape = prepared.indices.shape
    if len(indices_shape) == 1:
        total_length = int(indices_shape[0])
        top_k = 1
    else:
        total_length = int(indices_shape[0]) * int(indices_shape[1])
        top_k = int(indices_shape[1])

    # 第一次稳定 argsort：按专家索引值升序，相同值保持原相对顺序（sortedIndicesFirst）
    flat_idx = tf.cast(tf.reshape(prepared.indices, [-1]), tf.int64)
    order = tf.argsort(flat_idx, axis=-1, direction="ASCENDING", stable=True)

    # 第二次 argsort：得到每个原位置的秩 rank（order 为无重复排列，结果稳定唯一）
    order64 = tf.cast(order, tf.int64)
    sorted_indices = tf.cast(
        tf.argsort(order64, axis=-1, direction="ASCENDING", stable=True), tf.int32
    )

    # numOutTokens（与 tiling/infershape 语义一致）：end-start，<=0 时环绕
    # +totalLength，clamp 到 [0, totalLength]
    num_out = prepared.end - prepared.start
    if num_out <= 0:
        num_out += total_length
    num_out = max(0, min(num_out, total_length))

    # 搬运窗口（与 kernel CopyOut 分支语义一致，SE 5.2/第 7 章）：
    #   全量分支 ifNumOutTokens=false（num_out == total_length）：dst[rank[i]] 无偏移
    #   切片分支 ifNumOutTokens=true（num_out != total_length）：
    #       rank ∈ [start, min(start + num_out, total_length))
    if num_out == total_length:
        lo, hi = 0, total_length
    else:
        lo, hi = prepared.start, min(prepared.start + num_out, total_length)
    sel = tf.cast(order[lo:hi], tf.int64)  # 排序后第 j 位（j ∈ [lo, hi)）对应的原索引

    # 纯搬运：permute_tokens[r] = tokens[order[j] // topK]，permute_probs[r] = probs[order[j]]
    permute_tokens = tf.gather(prepared.tokens, sel // top_k, axis=0)
    permute_probs = None
    if prepared.probs is not None:
        permute_probs = tf.gather(tf.reshape(prepared.probs, [-1]), sel, axis=0)
    return permute_tokens, sorted_indices, permute_probs


def _run_numpy_flow(tokens, indices, probs, num_out_tokens, range_, padded_mode):
    """golden 流完整管线：numpy 入 -> [numpy, numpy, numpy|None] 出。

    `_tf_to_numpy` 对 None 直接透传（probs 缺省时第三个元素保持 None）。
    """
    prepared = _prepare_inputs(
        _numpy_to_tf(tokens), _numpy_to_tf(indices), _optional_probs(_numpy_to_tf(probs)),
        num_out_tokens, _normalize_range(range_), padded_mode,
    )
    pt, si, pp = _compute_core(prepared)
    return [_tf_to_numpy(pt), _tf_to_numpy(si), _tf_to_numpy(pp)]


def _prepare_framework_inputs(tokens, indices, probs, num_out_tokens, range_, padded_mode):
    """third_party 流前置转换（TTK 绑定约定：转换在 __init__ 完成）：框架 Tensor/numpy 入 -> _Prepared。"""
    return _prepare_inputs(
        _to_tf(tokens), _to_tf(indices), _optional_probs(_to_tf(probs)),
        num_out_tokens, _normalize_range(range_), padded_mode,
    )


def _run_framework_flow(tokens, indices, probs, num_out_tokens, range_, padded_mode):
    """third_party 流完整管线：框架 Tensor/numpy 入 -> [tf, tf, tf|None] 出。"""
    prepared = _prepare_framework_inputs(tokens, indices, probs, num_out_tokens, range_, padded_mode)
    return list(_compute_core(prepared))


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
        return _run_numpy_flow(tokens, indices, probs, num_out_tokens, range, padded_mode)

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
            self._prepared = _prepare_framework_inputs(
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
        return [_tf_to_numpy(r) for r in _run_framework_flow(
            tokens, indices, probsOptional, numOutTokens, rangeOptional, paddedMode)]

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
            self._prepared = _prepare_framework_inputs(
                tokens, indices, probsOptional, numOutTokens, rangeOptional, paddedMode
            )

        def __call__(self, **kwargs):
            return list(_compute_core(self._prepared))

    third_party = {"tf": ThirdPartyImpl}
    tolerance = _TOLERANCE

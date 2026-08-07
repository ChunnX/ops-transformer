#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UndGenQkvRmsNormRopeCache 的 TTK 输入/golden 插件。

用法：
    python3 -m ttk kernel -i <csv> -b=release --plugin <此文件>

两个插件都注册在 kernel 级：

- ``__input__``：算子有三个索引类输入带取值约束，随机 range 满足不了，必须改写。
  最关键的是 ``slot_mapping`` 必须互不重复——重复槽位会让多个核写同一 cache 行，
  写入顺序与最终结果都不确定，随机整数几乎必然撞号，不改写就会得到一个
  「每次跑都不一样」的 golden 比对。
- ``__golden__``：直接复用同目录上一层的 ``tests/golden.py``。该实现与需求来源的
  外部参考实现逐比特一致（见 ``tests/test_golden_vs_reference.py``），不在这里
  重写一份，避免两份 golden 各自演化后对不上。

``k_cache`` / ``v_cache`` 是原地更新的输入输出：IR 里输出名与输入名相同，
TTK 会据此自动填 ``output_inplace_indexes``，CSV 不必显式给。
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)

# IR 输入序（与 op_host/und_gen_qkv_rms_norm_rope_cache_def.cpp 一致）
IDX_UND_QKV = 0
IDX_UND_WEIGHTS_Q = 1
IDX_UND_WEIGHTS_K = 2
IDX_COS_SIN_CACHE = 3
IDX_K_CACHE = 4
IDX_V_CACHE = 5
IDX_SLOT_MAPPING = 6
IDX_POSITIONS = 7
IDX_GEN_QKV = 8
IDX_GEN_WEIGHTS_Q = 9
IDX_GEN_WEIGHTS_K = 10
IDX_CAT_INDICES = 11

INPUT_NUM = 12


def _load_repo_golden():
    """按需加载 tests/golden.py（torch 实现），失败时给出可操作的报错。"""
    if _TESTS_DIR not in sys.path:
        sys.path.insert(0, _TESTS_DIR)
    try:
        from golden import golden_und_gen_qkv_rms_norm_rope_cache
    except ImportError as exc:  # torch 缺失是最可能的原因
        raise RuntimeError(
            "加载 %s/golden.py 失败：%s。该 golden 依赖 torch（CPU 即可），"
            "请在跑 TTK 的环境里安装 torch 后重试。" % (_TESTS_DIR, exc)
        ) from exc
    return golden_und_gen_qkv_rms_norm_rope_cache


def _np_to_torch(array):
    """numpy -> torch，bf16 经 float32 中转（numpy 侧是 ml_dtypes.bfloat16）。"""
    import torch

    if array is None:
        return None
    if str(array.dtype) == "bfloat16":
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(array))


def _torch_to_np(tensor, like):
    """torch -> numpy，dtype 对齐到 TTK 侧对应输出的 numpy dtype。"""
    import torch

    if tensor.dtype == torch.bfloat16:
        out = tensor.float().numpy()
        return out.astype(like) if like is not None else out
    out = tensor.numpy()
    return out.astype(like) if like is not None else out


def und_gen_qkv_rms_norm_rope_cache_input(*inputs, **kwargs):
    """把三个索引类输入改写成满足算子约束的取值。

    随机整数不满足的约束：
      - slot_mapping：必须落在 [0, Bn*Bs-1] 且**互不重复**
      - cat_indices ：必须落在 [0, T-1]
      - positions   ：必须落在 [0, max_pos-1]，否则越界读 cos_sin_cache
    """
    arrays = list(inputs)
    if len(arrays) != INPUT_NUM:
        raise RuntimeError(
            "und_gen_qkv_rms_norm_rope_cache 期望 %d 个输入，实际收到 %d 个"
            % (INPUT_NUM, len(arrays))
        )

    slot_mapping = arrays[IDX_SLOT_MAPPING]
    k_cache = arrays[IDX_K_CACHE]
    total = int(slot_mapping.shape[0])
    block_num, block_size = int(k_cache.shape[0]), int(k_cache.shape[1])
    max_slot = block_num * block_size
    if max_slot < total:
        raise RuntimeError(
            "KV Cache 容量不足：Bn*Bs=%d < T=%d，请调整用例的 k_cache/v_cache shape"
            % (max_slot, total)
        )

    name = str(kwargs.get("testcase_name", ""))
    rng = np.random.default_rng(_seed_of(kwargs))

    # slot_mapping 形态由用例名后缀决定，CSV 里的 slot_form 列与此一一对应
    if "_slotseq" in name:
        slots = np.arange(total)  # 连续槽位：写入地址完全顺序
    else:
        slots = rng.choice(max_slot, size=total, replace=False)  # 散布在整块 cache 上
    arrays[IDX_SLOT_MAPPING] = slots.astype(slot_mapping.dtype)

    # cat_indices 形态同理，决定 undMask 与 gamma 的选择路径
    cat_indices = arrays[IDX_CAT_INDICES]
    if cat_indices is not None:
        und_len = int(arrays[IDX_UND_QKV].shape[0])
        if "_catid" in name:
            cat = np.arange(total)                       # 恒等映射
        elif "_catrev" in name:
            cat = np.arange(total)[::-1].copy()          # 逆序
        elif "_catund" in name:
            cat = rng.integers(0, und_len, size=total)   # 全部落在 und 段
        elif "_catgen" in name:
            cat = rng.integers(und_len, total, size=total)  # 全部落在 gen 段
        else:
            cat = rng.permutation(total)                 # 乱序
        arrays[IDX_CAT_INDICES] = cat.astype(cat_indices.dtype)

    positions = arrays[IDX_POSITIONS]
    if positions is not None:
        max_pos = int(arrays[IDX_COS_SIN_CACHE].shape[0])
        arrays[IDX_POSITIONS] = rng.integers(0, max_pos, size=positions.shape).astype(positions.dtype)

    return arrays


def _seed_of(kwargs):
    """按用例名派生种子：同一条用例每次跑拿到同一组索引，便于复现。"""
    name = str(kwargs.get("testcase_name", "und_gen_qkv_rms_norm_rope_cache"))
    return abs(hash(name)) % (2 ** 31)


def und_gen_qkv_rms_norm_rope_cache_golden(und_qkv,
                                           und_weights_q,
                                           und_weights_k,
                                           cos_sin_cache,
                                           k_cache,
                                           v_cache,
                                           slot_mapping,
                                           positions,
                                           gen_qkv=None,
                                           gen_weights_q=None,
                                           gen_weights_k=None,
                                           cat_indices=None,
                                           **kwargs):
    '''Kernel golden for und_gen_qkv_rms_norm_rope_cache.

    参数序与 @und_gen_qkv_rms_norm_rope_cache_def.cpp 的输入一致（不含输出）。
    所有输入张量是 numpy.ndarray；bf16 由 ml_dtypes.bfloat16 承载。
    kwargs 含算子属性（num_heads_q/num_heads_k/num_heads_v/norm_eps/mrope_section）
    以及 short_soc_version、input_dtypes、output_dtypes 等元信息。

    返回 (q, k_cache, v_cache)：k_cache / v_cache 是写入 slot 后的整块 cache，
    未被 slot_mapping 命中的位置保持入参原值。
    '''
    golden_fn = _load_repo_golden()

    num_heads_q = int(kwargs.get("num_heads_q", 8))
    num_heads_k = int(kwargs.get("num_heads_k", 1))
    num_heads_v = int(kwargs.get("num_heads_v", 1))
    norm_eps = float(kwargs.get("norm_eps", 1e-6))
    mrope_section = kwargs.get("mrope_section", None)
    if mrope_section is not None:
        mrope_section = list(mrope_section)

    out_dtypes = kwargs.get("output_dtypes", None)
    q_dtype = k_cache.dtype
    kv_dtype = k_cache.dtype
    if out_dtypes:
        q_dtype = np.dtype(out_dtypes[0]) if not str(out_dtypes[0]).startswith("bfloat") else k_cache.dtype

    q_t, k_t, v_t = golden_fn(
        _np_to_torch(und_qkv),
        _np_to_torch(und_weights_q),
        _np_to_torch(und_weights_k),
        _np_to_torch(cos_sin_cache),
        _np_to_torch(k_cache),
        _np_to_torch(v_cache),
        _np_to_torch(slot_mapping),
        _np_to_torch(positions),
        gen_qkv=_np_to_torch(gen_qkv),
        gen_weights_q=_np_to_torch(gen_weights_q),
        gen_weights_k=_np_to_torch(gen_weights_k),
        cat_indices=_np_to_torch(cat_indices),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        norm_eps=norm_eps,
        mrope_section=mrope_section,
        inplace=False,
    )

    return (
        _torch_to_np(q_t, q_dtype),
        _torch_to_np(k_t, kv_dtype),
        _torch_to_np(v_t, kv_dtype),
    )


__input__ = {
    "kernel": {
        "und_gen_qkv_rms_norm_rope_cache": "und_gen_qkv_rms_norm_rope_cache_input"
    }
}

__golden__ = {
    "kernel": {
        "und_gen_qkv_rms_norm_rope_cache": "und_gen_qkv_rms_norm_rope_cache_golden"
    }
}

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mla_prolog_v5 Python 接口（pybind 薄封装）。

使用前：
1. 编译并安装算子包：bash build.sh --soc=ascend910b --install
2. source ~/Ascend/opp/vendors/mla_prolog_v5_custom/bin/set_env.bash
3. 编译扩展：bash torch_extension/build.sh

示例：
    import torch, torch_npu
    from mla_prolog_v5 import mla_prolog
    query, query_rope, _, query_norm, _ = mla_prolog(...)
"""

import importlib.machinery
import importlib.util
import os
from typing import Optional, Tuple

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

_EXT_NAME = "mla_prolog_v5_ext"
_DEFAULT_SO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "build", f"lib{_EXT_NAME}.so"
)

_ext = None


def _load_ext(so_path: Optional[str] = None):
    global _ext
    if _ext is not None:
        return _ext
    path = so_path or os.environ.get("MLA_PROLOG_V5_EXT") or _DEFAULT_SO
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pybind extension not found: {path}\n"
            "run: bash torch_extension/build.sh  (under operators/mla_prolog_v5)"
        )
    loader = importlib.machinery.ExtensionFileLoader(_EXT_NAME, path)
    spec = importlib.util.spec_from_loader(_EXT_NAME, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    _ext = mod
    return _ext


def mla_prolog(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    *,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-05,
    rmsnorm_epsilon_ckv: float = 1e-05,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
    ext_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MlaPrologV5 前向（原地更新 kv_cache/kr_cache），封装 aclnnMlaPrologV5WeightNz 两段式接口。

    RoPE 开关由 rope_sin/rope_cos 是否成对传入决定（V4/V5 语义）。
    返回: (query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm)
    """
    ext = _load_ext(ext_path)
    return ext.mla_prolog(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        kv_cache,
        kr_cache,
        rope_sin,
        rope_cos,
        cache_index,
        dequant_scale_x,
        dequant_scale_w_dq,
        dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr,
        quant_scale_ckv,
        quant_scale_ckr,
        smooth_scales_cq,
        actual_seq_len,
        k_nope_clip_alpha,
        rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv,
        cache_mode,
        query_norm_flag,
        weight_quant_mode,
        kv_cache_quant_mode,
        query_quant_mode,
        ckvkr_repo_mode,
        quant_scale_repo_mode,
        tile_size,
        qc_qr_scale,
        kc_scale,
        token_x_dtype,
        weight_dq_dtype,
        weight_uq_qr_dtype,
        weight_dkv_kr_dtype,
        kv_cache_dtype,
    )


__all__ = ["mla_prolog"]

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""实验性质算子 fused_moe_small_bs_w4a16 的自包含 torch 绑定加载器。

不依赖 cann_ops_transformer 共享扩展包：
- 通过 torch.utils.cpp_extension.load JIT 编译同目录的 binding cpp，
  直接调用 custom_transformer 算子包（libcust_opapi.so）中的 aclnn 两段式接口；
- 同时注册 torch.library（schema + Meta + PrivateUse1），使算子可被
  torch.npu.NPUGraph（aclgraph）捕获。

环境要求：
- 已安装 custom_transformer 算子包（build_out/cann-ops-transformer-custom_linux-*.run）
- 设置 ASCEND_HOME_PATH（默认 /usr/local/Ascend/cann-9.0.0）
"""

import os

import torch
import torch_npu
from torch.library import Library, impl
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
_OPP_API_DIR = os.path.join(_ASCEND_HOME_PATH, "opp", "vendors", "custom_transformer", "op_api")
_TORCH_NPU_DIR = os.path.dirname(os.path.abspath(torch_npu.__file__))

if not os.path.isdir(_OPP_API_DIR):
    raise RuntimeError(
        f"custom_transformer op package not found at {_OPP_API_DIR}, "
        "please install build_out/cann-ops-transformer-custom_linux-*.run first."
    )

_op_module = load(
    name="fused_moe_small_bs_w4a16_experimental",
    sources=[os.path.join(_HERE, "fused_moe_small_bs_w4a16_binding.cpp")],
    extra_include_paths=[
        os.path.join(_TORCH_NPU_DIR, "include"),
        os.path.join(_ASCEND_HOME_PATH, "include"),
        os.path.join(_OPP_API_DIR, "include"),
    ],
    extra_cflags=["-O2", "-w"],
    extra_ldflags=[
        "-L" + os.path.join(_OPP_API_DIR, "lib"), "-lcust_opapi",
        "-Wl,-rpath," + os.path.join(_OPP_API_DIR, "lib"),
        "-L" + os.path.join(_ASCEND_HOME_PATH, "lib64"), "-lascendcl",
        "-L" + os.path.join(_TORCH_NPU_DIR, "lib"), "-ltorch_npu",
    ],
    verbose=False,
)

# 注册 torch.library，使算子可被 torch.npu.NPUGraph 捕获、并支持 FakeTensor
_LIBRARY = Library("fused_moe_experimental", "DEF")
_LIBRARY.define(
    "fused_moe_small_bs_w4a16(Tensor x, Tensor w13_weight, Tensor w13_scales, "
    "Tensor w2_weight, Tensor w2_scales, Tensor expert_ids, Tensor topk_weights) -> Tensor"
)


@impl(_LIBRARY, "fused_moe_small_bs_w4a16", "Meta")
def _fused_moe_small_bs_w4a16_meta(x, w13_weight, w13_scales, w2_weight, w2_scales,
                                   expert_ids, topk_weights):
    torch._check(x.dim() == 2, lambda: f"x must be 2D [BatchSize, HiddenSize], got {x.dim()}D")
    return x.new_empty((x.size(0), x.size(1)))


@impl(_LIBRARY, "fused_moe_small_bs_w4a16", "PrivateUse1")
def _fused_moe_small_bs_w4a16(x, w13_weight, w13_scales, w2_weight, w2_scales,
                              expert_ids, topk_weights):
    return _op_module.fused_moe_small_bs_w4a16(x, w13_weight, w13_scales, w2_weight, w2_scales,
                                               expert_ids, topk_weights)


def fused_moe_small_bs_w4a16(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """小 batch 场景的 W4A16 融合 MoE 前向计算（实验性质，自包含绑定）。

    单 kernel 完成 W13(broadcast x) -> SwiGLU -> W2(topk 加权累加) 全流程。

    Args:
        x (Tensor): 输入隐状态，shape [BatchSize, HiddenSize]，dtype float16/bfloat16。
        w13_weight (Tensor): W13 打包权重（8 个 int4 打包为一个 int32，低位在前），
            shape [NumExperts, HiddenSize, 2 * InterSize / 8]，dtype int32。
        w13_scales (Tensor): W13 分组量化 scale（group_size=32，对称量化），
            shape [NumExperts, HiddenSize / 32, 2 * InterSize]，dtype 与 x 一致。
        w2_weight (Tensor): W2 打包权重，shape [NumExperts, InterSize, HiddenSize / 8]，dtype int32。
        w2_scales (Tensor): W2 分组量化 scale，shape [NumExperts, InterSize / 32, HiddenSize]，
            dtype 与 x 一致。
        expert_ids (Tensor): 每个 token 路由到的专家 id，shape [BatchSize, TopK]，dtype int32。
        topk_weights (Tensor): 每个 token 路由权重，shape [BatchSize, TopK]，dtype float32。

    Returns:
        Tensor: 输出隐状态，shape [BatchSize, HiddenSize]，dtype 与 x 一致。

    约束：HiddenSize 与 InterSize 必须为 128 的倍数。
    """
    return torch.ops.fused_moe_experimental.fused_moe_small_bs_w4a16(
        x.contiguous(),
        w13_weight.contiguous(),
        w13_scales.contiguous(),
        w2_weight.contiguous(),
        w2_scales.contiguous(),
        expert_ids.contiguous(),
        topk_weights.contiguous(),
    )

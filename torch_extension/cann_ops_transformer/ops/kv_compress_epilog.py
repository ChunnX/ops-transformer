# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class KvCompressEpilogOpBuilder(OpBuilder):
    def __init__(self):
        super(KvCompressEpilogOpBuilder, self).__init__("npu_kv_compress_epilog")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/kv_compress_epilog.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_kv_compress_epilog(Tensor cache, Tensor x, Tensor slot_mapping, *, " \
               "int quant_group_size=64, int quant_mode=1, bool round_scale=True, " \
               "float x_scale=1.0) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for FakeTensor / graph mode support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_kv_compress_epilog_meta(cache, x, slot_mapping, *, quant_group_size=64,
                                        quant_mode=1, round_scale=True, x_scale=1.0):
            # 输出为更新后的 cache，shape/dtype 与输入 cache 相同
            return torch.empty_like(cache)


# Instantiate the builder
npu_kv_compress_epilog_op_builder = KvCompressEpilogOpBuilder()
op_module = npu_kv_compress_epilog_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, npu_kv_compress_epilog_op_builder.name, "PrivateUse1")
def npu_kv_compress_epilog(cache, x, slot_mapping, *, quant_group_size=64,
                           quant_mode=1, round_scale=True, x_scale=1.0):
    """在 KV Cache 的 Epilog 阶段对 cache 进行逐组动态量化压缩更新，封装 aclnnKvCompressEpilog。

    将 bfloat16 激活值 x 按尾轴每 64 个元素一组动态量化为 FP8（以 uint8 打包），并按 slot_mapping
    将量化结果散写到 cache，slot_mapping 中值为 -1 的 token 跳过不处理。返回更新后的 cache（函数式
    语义，底层 aclnn 接口为原地实现）。

    Args:
        cache (Tensor): 当前层的 KV Cache 向量缓存，dtype 为 uint8，shape 为 [num_slots, kvCacheCol]，
            每行包含 rope/fp8/scale/pad 各区段。
        x (Tensor): 待量化的激活输入，dtype 为 bfloat16，shape 为 [bs, d]，d 需满足 d % 64 == 0 且 d > 64。
        slot_mapping (Tensor): token 到 cache slot 的索引映射，dtype 为 int32 或 int64，shape 为 [bs]，
            元素取值范围 [-1, num_slots - 1]，-1 表示跳过该 token。维度需等于 x 的维度减 1。
        quant_group_size (int): 量化分组大小，与内核实现一致固定为 64，默认 64。
        quant_mode (int): 量化模式，0=group(bf16 scale)、1=group(e8m0 scale)，默认 1。
        round_scale (bool): group 模式下是否对每组 scale 向上取到 2 的幂，默认 True。
        x_scale (float): 预留参数，当前未使用，默认 1.0。

    Returns:
        Tensor: 更新后的 KV Cache，shape/dtype 与输入 cache 相同。

    仅支持 Ascend 950PR/Ascend 950DT。
    """
    return op_module.npu_kv_compress_epilog(cache, x, slot_mapping, quant_group_size,
                                            quant_mode, round_scale, x_scale)
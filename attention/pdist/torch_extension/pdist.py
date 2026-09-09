# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


PDIST_OP_NAME = "pdist"


class PdistOpBuilder(OpBuilder):
    def __init__(self):
        super(PdistOpBuilder, self).__init__("pdist", category="attention")

    def sources(self):
        return ["csrc/attention/pdist.cpp"]

    def schema(self) -> str:
        return "pdist(Tensor x, float p=2.0) -> Tensor"

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + PDIST_OP_NAME)
        def pdist_meta(x, p=2.0):
            n = x.shape[0]
            out_len = n * (n - 1) // 2
            return torch.empty((out_len,), dtype=x.dtype, device="meta")


pdist_op_builder = PdistOpBuilder()


@impl(get_as_library(), PDIST_OP_NAME, "PrivateUse1")
def pdist(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """pdist 前向算子，封装 aclnnPdist。

    计算行向量两两之间的 p-范数距离，输出为上三角行优先向量：
    y[idx] = ||x_i - x_j||_p，idx = i*(2*N-i-1)/2 + (j-i-1)，0 <= i < j <= N-1。

    Args:
        x (Tensor): 输入，shape (N, M)，N >= 2，dtype 支持 float16/float32。
        p (float): 范数阶，p >= 0 或 +inf，默认 2.0（欧氏距离）。
            p=0 统计不相等维度数，p=+inf 取最大绝对差，0<p<inf 为幂和的 1/p 次方。

    Returns:
        Tensor: shape (N*(N-1)/2,) 的距离向量，dtype 与 x 一致。
    """
    op_module = pdist_op_builder.load()
    return op_module.pdist(x, p)


@torch.library.register_kernel("cann_ops_transformer::" + PDIST_OP_NAME, None)
def pdist_fallback(x, p=2.0):
    return pdist(x, p)


torch.compiler.allow_in_graph(pdist)

pdist_op_builder._ensure_initialized()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MhcPreSplitPostResOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSplitPostResOpBuilder, self).__init__(
            "mhc_pre_split_post_res", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_split_post_res.cpp"]

    def schema(self) -> str:
        return (
            "mhc_pre_split_post_res(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            "float norm_eps=1e-6) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_split_post_res_meta(x, phi, alpha, bias, norm_eps=1e-6):
            # norm_eps 默认值与 schema 对齐：torch.compile(fake tensor) 不会替 Meta 实现
            # 填充 schema 默认值，签名缺默认值会报 missing positional argument 并打断 fullgraph 捕获
            # h_res 为可选输出：phi 行数为 N（仅 post 形态）时第二个返回为 0 元素 tensor
            n = x.size(-2)
            has_res = phi.size(0) == n * n + n
            if x.dim() == 3:
                t = x.size(0)
                h_post = torch.empty(t, n, dtype=phi.dtype, device="meta")
                h_res = torch.empty(t, n, n, dtype=phi.dtype, device="meta") if has_res \
                    else torch.empty(0, dtype=phi.dtype, device="meta")
            else:
                b = x.size(0)
                s = x.size(1)
                h_post = torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                h_res = torch.empty(b, s, n, n, dtype=phi.dtype, device="meta") if has_res \
                    else torch.empty(0, dtype=phi.dtype, device="meta")
            return (h_post, h_res)


mhc_pre_split_post_res_op_builder = MhcPreSplitPostResOpBuilder()
mhc_pre_split_post_res_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_split_post_res_op_builder.name, "PrivateUse1")
def mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps=1e-6):
    op_module = mhc_pre_split_post_res_op_builder.load()
    return op_module.mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps)

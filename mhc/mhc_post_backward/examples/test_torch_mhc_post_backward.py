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

from cann_ops_transformer.ops import mhc_post_backward

# 参数设置
B = 2
S = 8
n = 4
D = 128

# 创建输入 tensor
grad_output = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

# 调用算子
grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
    grad_output, x, h_res, h_out, h_post
)
torch.npu.synchronize()

assert grad_x.shape == x.shape, grad_x.shape
assert grad_x.dtype == torch.bfloat16, grad_x.dtype
assert grad_h_res.shape == h_res.shape, grad_h_res.shape
assert grad_h_res.dtype == torch.float32, grad_h_res.dtype
assert grad_h_out.shape == h_out.shape, grad_h_out.shape
assert grad_h_out.dtype == torch.bfloat16, grad_h_out.dtype
assert grad_h_post.shape == h_post.shape, grad_h_post.shape
assert grad_h_post.dtype == torch.float32, grad_h_post.dtype

print(f"grad_x shape: {grad_x.shape}, dtype: {grad_x.dtype}")
print(f"grad_h_res shape: {grad_h_res.shape}, dtype: {grad_h_res.dtype}")
print(f"grad_h_out shape: {grad_h_out.shape}, dtype: {grad_h_out.dtype}")
print(f"grad_h_post shape: {grad_h_post.shape}, dtype: {grad_h_post.dtype}")

# h_res 传入 None，跳过 Res Mapping
# 仅 Ascend 950PR&950DT系列产品支持，其他产品形态会返回 ACLNN_ERR_PARAM_NULLPTR
grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
    grad_output, x, None, h_out, h_post
)
torch.npu.synchronize()

assert grad_h_res.shape == (B, S, n, n), grad_h_res.shape
assert grad_x.shape == x.shape, grad_x.shape

print(f"h_res=None, grad_x shape: {grad_x.shape}")
print(f"h_res=None, grad_h_res shape: {grad_h_res.shape}")

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


# 1. Sigmoid反向：已修正h_pre的hc_eps处理

def SigmoidGrad(z: torch.Tensor, dy: torch.Tensor, is_pre: bool = False, hc_eps: float = 1e-6):

    """

    Sigmoid激活函数的梯度计算

    参数：

        z: 激活前的输入值

        dy: 损失对激活输出的梯度

        is_pre: 是否是h_pre支路的梯度（h_pre正向加了hc_eps）

        hc_eps: 正向中h_pre添加的数值稳定项

    """

    y = torch.sigmoid(z)          # sigmoid(z) = 1/(1 + exp(-z))

    if is_pre:

        # 正向：h_pre = sigmoid(z) + hc_eps → 导数中的sigmoid(z) = y - hc_eps

        sigma = y - hc_eps

    else:

        # 正向：h_post = 2 * sigmoid(z) → 没有加eps

        sigma = y

    return dy * sigma * (1 - sigma)


# 2. RMSNorm倒数梯度：标准命名版

def RMSNormGrad(x: torch.Tensor, inv_rms: torch.Tensor, grad_inv_rms: torch.Tensor):

    """

    RMSNorm倒数的梯度计算

    参数：

        x: 原始输入特征，形状[B*S, n*C]

        inv_rms: 正向计算的RMS倒数，形状[B*S, 1]

        grad_inv_rms: 损失对inv_rms的梯度，形状[B*S, 1]

    """

    NC = x.size(-1)

    grad_x = - (inv_rms ** 3) * x * grad_inv_rms / float(NC)

    return grad_x


# 3. exp激活梯度：标准命名版

def ExpGrad(x: torch.Tensor, y_grad: torch.Tensor):

    """

    数值稳定的exp激活梯度计算（对应正向的softmax第一步）

    """

    n = x.size(-1)

    x_max, idx = x.max(dim=-1, keepdim=True)

    arange_idx = torch.arange(n, device=x.device)[None, :]

    is_max = (idx == arange_idx)

    y = (x - x_max).exp()

    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)

    return y * y_grad - is_max * sum_all


# 4. Sinkhorn-Knopp反向：标准命名版

def SinkhornGrad(

    grad_h_res: torch.Tensor,

    sum_out: torch.Tensor,

    norm_out: torch.Tensor

):

    """

    Sinkhorn-Knopp算法的梯度计算

    参数：

        grad_h_res: 损失对h_res的梯度，形状[B, S, n, n]

        sum_out: 正向保存的每一步行和与列和，形状[iters, 2, B, S, n]

        norm_out: 正向保存的每一步行归一化与列归一化结果，形状[iters, 2, B, S, n, n]

    """

    bs, seq_len, n, _ = grad_h_res.shape

    x_grad = grad_h_res

    iters = sum_out.shape[0]



    for i in reversed(range(iters)):

        row_sum = sum_out[i][0].view(bs, seq_len, n, 1)

        col_sum = sum_out[i][1].view(bs, seq_len, 1, n)

        x_row_normed = norm_out[i][0].view(bs, seq_len, n, n)


        # 列归一化的梯度

        grad_x_row_normed = x_grad / col_sum - (x_grad * x_row_normed / (col_sum ** 2)).sum(dim=-2, keepdim=True)

        # 行归一化的梯度

        x_grad = grad_x_row_normed / row_sum - (grad_x_row_normed * x_row_normed / row_sum).sum(dim=-1, keepdim=True)

    

    return x_grad


# 主梯度计算函数：完全对齐算子原型命名

def mhc_pre_sinkhorn_backward_golden(

    grad_hin: torch.Tensor,

    grad_h_post: torch.Tensor,

    grad_h_res: torch.Tensor,

    x: torch.Tensor,

    phi: torch.Tensor,

    alpha: torch.Tensor,

    bias: torch.Tensor,

    h_pre: torch.Tensor,

    hc_before_norm: torch.Tensor,

    inv_rms: torch.Tensor,

    sum_out: torch.Tensor,

    norm_out: torch.Tensor,

    hc_eps: float = 1e-6

) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    # 获取形状信息

    B, S, n, d = x.shape


    # 转换为float32计算，保证精度

    grad_hin_fp32 = grad_hin.float()

    x_fp32 = x.float()


    # -------------------------- 步骤1：输入聚合的梯度 --------------------------

    # grad_x_from_hin: 损失对x的梯度（来自h_in支路）

    grad_x_from_hin = grad_hin_fp32[..., None, :] * h_pre[..., None]

    # grad_h_pre: 损失对h_pre的梯度

    grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)


    # -------------------------- 步骤2：计算正向的中间变量 --------------------------

    # norm_out = hc_before_norm * inv_rms（正向步骤4）

    norm_out_forward = hc_before_norm * inv_rms

    # 拆分出三个支路的线性输出

    z_pre = norm_out_forward[..., :n] * alpha[0] + bias[:n]

    z_post = norm_out_forward[..., n:2*n] * alpha[1] + bias[n:2*n]

    z_res = (norm_out_forward[..., 2*n:] * alpha[2] + bias[2*n:]).reshape(B, S, n, n)


    # -------------------------- 步骤3：三路激活的梯度 --------------------------

    # h_pre支路：需要传入is_pre=True，修正hc_eps

    grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)

    # h_post支路：不需要修正

    grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False, hc_eps=hc_eps)

    # h_res支路：Sinkhorn梯度 + exp梯度

    sk_grad = SinkhornGrad(grad_h_res, sum_out, norm_out)

    grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)


    # -------------------------- 步骤4：偏置bias的梯度 --------------------------

    grad_bias = torch.cat([

        grad_z_pre.sum((0, 1)),

        grad_z_post.sum((0, 1)),

        grad_z_res.sum((0, 1))

    ])


    # -------------------------- 步骤5：缩放因子alpha的梯度 --------------------------

    grad_alpha_pre = (norm_out_forward[..., :n] * grad_z_pre).sum()

    grad_alpha_post = (norm_out_forward[..., n:2*n] * grad_z_post).sum()

    grad_alpha_res = (norm_out_forward[..., 2*n:] * grad_z_res).sum()

    grad_alpha = torch.tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res], dtype=torch.float32)


    # -------------------------- 步骤6：线性变换的梯度 --------------------------

    # 对norm_out的梯度

    grad_norm_out_pre = grad_z_pre * alpha[0]

    grad_norm_out_post = grad_z_post * alpha[1]

    grad_norm_out_res = grad_z_res * alpha[2]

    grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)


    # -------------------------- 步骤7：乘法的梯度 --------------------------

    # 对hc_before_norm的梯度

    grad_hc_before_norm = grad_norm_out * inv_rms

    # 对inv_rms的梯度

    grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)


    # -------------------------- 步骤8：RMSNorm倒数的梯度 --------------------------

    x_flat = x_fp32.flatten(2)

    grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)


    # -------------------------- 步骤9：矩阵乘法的梯度 --------------------------

    # 对x的梯度（来自矩阵乘法支路）

    grad_x_from_matmul = grad_hc_before_norm @ phi

    # 对phi的梯度

    grad_hc_before_norm_flat = grad_hc_before_norm.flatten(0, 1)

    x_flat_flat = x_flat.flatten(0, 1)

    grad_phi = grad_hc_before_norm_flat.T @ x_flat_flat


    # -------------------------- 步骤10：汇总所有梯度 --------------------------

    grad_x = (

        grad_x_from_hin.view(x.shape)

        + grad_x_from_rms.view(x.shape)

        + grad_x_from_matmul.view(x.shape)

    ).to(x.dtype)



    return grad_x, grad_phi, grad_alpha, grad_bias

    
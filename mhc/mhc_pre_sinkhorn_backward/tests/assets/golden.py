# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MhcPreSinkhornBackward 算子 Golden实现

算子原型：MhcPreSinkhornBackward
功能：MhcPreSinkhorn反向传播梯度计算
平台：Ascend NPU (arch35)

================================================================================
算子维度约束（官方定义）
================================================================================

【输入维度约束】
    grad_hin:       [B, S, D] 或 [T, D]       (T = B*S, D = feature_dim)
    grad_h_post:    [B, S, N] 或 [T, N]       (N = hidden_dim)
    grad_h_res:     [B, S, N, N] 或 [T, N, N] (Sinkhorn矩阵梯度)
    x:              [B, S, N, D] 或 [T, N, D] (输入特征矩阵)
    phi:            [N²+2N, N*D]              (全局投影矩阵) ★关键
    alpha:          [3]                        (三路缩放系数)
    bias:           [N²+2N]                    (全局偏置)
    h_pre:          [B, S, N] 或 [T, N]       (sigmoid门控输出)
    hc_before_norm: [B, S, N²+2N] 或 [T, N²+2N] (投影后输出)
    inv_rms:        [B, S, 1] 或 [T, 1]       (RMS倒数)
    sum_out:        [2*iter, B, S, N]         (Sinkhorn迭代sum记录)
    norm_out:       [2*iter, B, S, N, N]      (Sinkhorn迭代norm记录)

【输出维度约束】
    grad_x:         [B, S, N, D]              (对x的梯度)
    grad_phi:       [N²+2N, N*D]              (对phi的梯度) ★关键
    grad_alpha:     [3]                        (对alpha的梯度)
    grad_bias:      [N²+2N]                    (对bias的梯度)

【关键理解】
    1. phi形状：[N²+2N, N*D] = [24, 256]
       - N²+2N = 16+8 = 24 (三路输出维度)
       - N*D = 4*64 = 256 (输入特征维度)
       
    2. phi作用：全局投影矩阵
       - 正向：hc_before_norm = x_flat @ phi.T + bias
       - x_flat: [T, N*D] = [16, 256]
       - phi.T: [N*D, N²+2N] = [256, 24]
       - 输出：[T, N²+2N] = [16, 24]
       
    3. grad_phi计算公式（矩阵乘法反向）：
       - grad_phi = grad_hc_before_norm.T @ x_flat
       - grad_hc.T: [N²+2N, T] = [24, 16]
       - x_flat: [T, N*D] = [16, 256]
       - 结果：[N²+2N, N*D] = [24, 256] ✓

================================================================================
"""

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

        x: 原始输入特征，形状任意维度（最后一维是n*C）

        inv_rms: 正向计算的RMS倒数，形状与x相同（最后一维是1）

        grad_inv_rms: 损失对inv_rms的梯度，形状与inv_rms相同

    """

    NC = x.size(-1)  # 兼容任意维度tensor，获取最后一个维度大小

    grad_x = - inv_rms.pow(3) * x * grad_inv_rms / NC

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

    iters_tims_2, B, S, n = sum_out.shape
    iters = iters_tims_2 // 2

    sum_out = sum_out.reshape(iters, 2, B, S, n)
    norm_out = norm_out.reshape(iters, 2, B, S, n, n)

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
    # ============================================================================
    # 【步骤1详细说明】输入聚合梯度计算
    # ============================================================================
    # 【功能】计算正向传播 h_in = x * h_pre 的反向梯度
    # 
    # 【正向传播公式】
    #   h_in = sum_{i=1}^{n} (x[..., i, :] * h_pre[..., i]) 
    #   输入: x[B,S,n,C], h_pre[B,S,n]
    #   输出: h_in[B,S,C]  (聚合n个hidden_dim的特征)
    # 
    # 【反向传播推导】
    #   已知: grad_h_in[B,S,C] - 损失函数对h_in的梯度
    #   目标: 求 grad_x[B,S,n,C] 和 grad_h_pre[B,S,n]
    #   
    #   grad_x[..., i, :] = grad_h_in * h_pre[..., i]  (broadcast)
    #   grad_h_pre[..., i] = sum_{c=1}^{C} (grad_h_in[..., c] * x[..., i, c])
    # 
    # 【输入约束】
    #   - grad_hin: torch.Tensor, shape=[B,S,C], dtype=float32/16
    #     含义: 损失函数对聚合输出h_in的梯度
    #   - x: torch.Tensor, shape=[B,S,n,C], dtype=float16
    #     含义: 输入特征矩阵，n个hidden_dim，每个dim有C个特征
    #   - h_pre: torch.Tensor, shape=[B,S,n], dtype=float16  
    #     含义: sigmoid激活输出，作为聚合权重
    # 
    # 【输出约束】
    #   - grad_x_from_hin: torch.Tensor, shape=[B,S,n,C], dtype=float32
    #     含义: 对x的梯度（来自h_in支路）
    #   - grad_h_pre: torch.Tensor, shape=[B,S,n], dtype=float32
    #     含义: 对h_pre的梯度，后续传入sigmoid反向
    # 
    # 【关键理解】
    #   1. h_pre是sigmoid输出，范围(0,1)，作为聚合权重
    #   2. x[..., i, :]表示第i个hidden_dim的所有C个特征
    #   3. broadcast机制: grad_h_in[B,S,C] * h_pre[..., i][B,S,1] → [B,S,C]
    #   4. sum归约: 对C维求和得到grad_h_pre[B,S,n]
    # ============================================================================

    # grad_x_from_hin: 损失对x的梯度（来自h_in支路）

    grad_x_from_hin = grad_hin_fp32[..., None, :] * h_pre[..., None]

    # grad_h_pre: 损失对h_pre的梯度

    grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)


    # -------------------------- 步骤2：计算正向的中间变量 --------------------------
    # ============================================================================
    # 【步骤2详细说明】正向中间变量重计算
    # ============================================================================
    # 【功能】重新计算正向传播的中间变量，用于反向传播推导
    # 
    # 【正向传播链】
    #   hc_before_norm[B,S,n*n+2*n] = x_flat[B*S,n*C] @ phi[n*C,n*n+2*n] + bias[n*n+2*n]
    #   inv_rms[B,S,1] = 1 / sqrt(mean(hc^2) + eps)
    #   norm_out[B,S,n*n+2*n] = hc_before_norm * inv_rms
    #   z_pre = norm_out[..., :n] * alpha[0] + bias[:n]
    #   z_post = norm_out[..., n:2*n] * alpha[1] + bias[n:2*n]
    #   z_res = (norm_out[..., 2*n:] * alpha[2] + bias[2*n:]).reshape(B,S,n,n)
    # 
    # 【输入约束】
    #   - hc_before_norm: [B,S,n*n+2*n] - 矩阵乘法输出（未归一化）
    #   - inv_rms: [B,S,1] - RMSNorm倒数（平方均值倒数的平方根）
    #   - alpha: [3] - 三路缩放因子[alpha_pre, alpha_post, alpha_res]
    #   - bias: [n*n+2*n] - 三路偏置[bias_pre(n), bias_post(n), bias_res(n*n)]
    # 
    # 【输出约束】
    #   - norm_out_forward: [B,S,n*n+2*n] - RMSNorm输出
    #   - z_pre: [B,S,n] - pre支路的线性输出（sigmoid输入）
    #   - z_post: [B,S,n] - post支路的线性输出（sigmoid输入）
    #   - z_res: [B,S,n,n] - res支路的线性输出（exp+Sinkhorn输入）
    # 
    # 【关键理解】
    #   1. norm_out分成三段: [:n]为pre支路, [n:2*n]为post支路, [2*n:]为res支路
    #   2. z是激活前的线性输出: z = norm_out * alpha + bias
    #   3. res支路特殊: [n*n] reshape为[B,S,n,n]矩阵形式
    #   4. alpha/bias分段长度: pre(n), post(n), res(n*n) → 总长度n*n+2*n
    # ============================================================================

    # norm_out = hc_before_norm * inv_rms（正向步骤4）

    norm_out_forward = hc_before_norm * inv_rms

    # 拆分出三个支路的线性输出

    z_pre = norm_out_forward[..., :n] * alpha[0] + bias[:n]

    z_post = norm_out_forward[..., n:2*n] * alpha[1] + bias[n:2*n]

    z_res = (norm_out_forward[..., 2*n:] * alpha[2] + bias[2*n:]).reshape(B, S, n, n)


    # -------------------------- 步骤3：三路激活的梯度 --------------------------
    # ============================================================================
    # 【步骤3详细说明】三路激活函数反向梯度
    # ============================================================================
    # 【功能】计算sigmoid和exp激活函数的反向梯度
    # 
    # 【三路激活类型】
    #   1. pre支路: sigmoid(z_pre) → h_pre
    #      反向: SigmoidGrad(z_pre, grad_h_pre) with hc_eps修正
    #   2. post支路: sigmoid(z_post) → h_post
    #      反向: 2 * SigmoidGrad(z_post, grad_h_post) (无修正)
    #   3. res支路: exp(z_res) → Sinkhorn → h_res
    #      反向: ExpGrad(z_res, SinkhornGrad(...))
    # 
    # 【输入约束】
    #   - z_pre: [B,S,n] - pre支路sigmoid输入
    #   - grad_h_pre: [B,S,n] - 对sigmoid输出的梯度（来自步骤1）
    #   - z_post: [B,S,n] - post支路sigmoid输入
    #   - grad_h_post: [B,S,n] - 对sigmoid输出的梯度（外部输入）
    #   - z_res: [B,S,n,n] - res支路exp输入
    #   - grad_h_res: [B,S,n,n] - 对Sinkhorn输出的梯度（外部输入）
    #   - sum_out: [iters,2,B,S,n] - Sinkhorn正向保存的行/列和
    #   - norm_out: [iters,2,B,S,n,n] - Sinkhorn正向保存的归一化矩阵
    # 
    # 【输出约束】
    #   - grad_z_pre: [B,S,n] - 对z_pre的梯度（传给步骤4/5/6）
    #   - grad_z_post: [B,S,n] - 对z_post的梯度
    #   - grad_z_res: [B,S,n*n] - 对z_res的梯度（flatten后）
    # 
    # 【关键理解】
    #   1. pre支路hc_eps修正: sigmoid(h_pre*(1+eps))的特殊处理
    #   2. post支路系数2: sigmoid的梯度公式中因子为2
    #   3. res支路复杂: exp(z_res) → Sinkhorn迭代 → h_res
    #      SinkhornGrad需要正向保存的sum_out和norm_out
    #   4. grad_z_res flatten: [B,S,n,n] → [B,S,n*n]以便后续concat
    # ============================================================================

    # h_pre支路：需要传入is_pre=True，修正hc_eps

    grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)

    # h_post支路：不需要修正

    grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False, hc_eps=hc_eps)

    # h_res支路：Sinkhorn梯度 + exp梯度

    sk_grad = SinkhornGrad(grad_h_res, sum_out, norm_out)

    grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)


    # -------------------------- 步骤4：偏置bias的梯度 --------------------------
    # ============================================================================
    # 【步骤4详细说明】偏置bias梯度计算（全局归约）
    # ============================================================================
    # 【功能】计算三路偏置的梯度，通过全局归约累加
    # 
    # 【正向传播】
    #   z_pre = norm_out[..., :n] * alpha[0] + bias[:n]
    #   z_post = norm_out[..., n:2*n] * alpha[1] + bias[n:2*n]
    #   z_res = norm_out[..., 2*n:] * alpha[2] + bias[2*n:]
    # 
    # 【反向传播】
    #   grad_bias_pre[c] = sum_{b,s} grad_z_pre[b,s,c]  (对所有batch和sequence累加)
    #   grad_bias_post[c] = sum_{b,s} grad_z_post[b,s,c]
    #   grad_bias_res[i] = sum_{b,s} grad_z_res[b,s,i]
    # 
    # 【输入约束】
    #   - grad_z_pre: [B,S,n] - pre支路对z的梯度（来自步骤3）
    #   - grad_z_post: [B,S,n] - post支路对z的梯度
    #   - grad_z_res: [B,S,n*n] - res支路对z的梯度
    # 
    # 【输出约束】
    #   - grad_bias: [n*n+2*n] - 对bias的梯度
    #     组成: [grad_bias_pre(n), grad_bias_post(n), grad_bias_res(n*n)]
    # 
    # 【关键理解】
    #   1. 全局归约: sum over (B,S) → 所有batch和sequence位置累加
    #   2. 偏置是全局参数: 每个位置的bias相同，梯度需要累加
    #   3. 三段concat: pre支路[n], post支路[n], res支路[n*n]
    #   4. 总长度: n + n + n*n = n*n + 2*n (例如n=4时为24)
    # ============================================================================

    grad_bias = torch.cat([

        grad_z_pre.sum((0, 1)),

        grad_z_post.sum((0, 1)),

        grad_z_res.sum((0, 1))

    ])


    # -------------------------- 步骤5：缩放因子alpha的梯度 --------------------------
    # ============================================================================
    # 【步骤5详细说明】缩放因子alpha梯度计算（全局归约）
    # ============================================================================
    # 【功能】计算三路缩放因子的梯度，通过全局归约累加
    # 
    # 【正向传播】
    #   z_pre = norm_out[..., :n] * alpha[0] + bias[:n]
    #   z_post = norm_out[..., n:2*n] * alpha[1] + bias[n:2*n]
    #   z_res = norm_out[..., 2*n:] * alpha[2] + bias[2*n:]
    # 
    # 【反向传播】
    #   grad_alpha[0] = sum_{b,s,c} (norm_out[b,s,c] * grad_z_pre[b,s,c])  (对所有元素累加)
    #   grad_alpha[1] = sum_{b,s,c} (norm_out[b,s,n+c] * grad_z_post[b,s,c])
    #   grad_alpha[2] = sum_{b,s,i} (norm_out[b,s,2*n+i] * grad_z_res[b,s,i])
    # 
    # 【输入约束】
    #   - norm_out_forward: [B,S,n*n+2*n] - RMSNorm输出
    #   - grad_z_pre: [B,S,n] - pre支路对z的梯度
    #   - grad_z_post: [B,S,n] - post支路对z的梯度
    #   - grad_z_res: [B,S,n*n] - res支路对z的梯度
    # 
    # 【输出约束】
    #   - grad_alpha: [3] - 对alpha的梯度
    #     组成: [grad_alpha_pre, grad_alpha_post, grad_alpha_res]
    # 
    # 【关键理解】
    #   1. 全局归约: sum over 所有元素(B,S,dim)
    #   2. alpha是全局标量: 每个支路一个alpha值，梯度需要累加
    #   3. 逐元素相乘: norm_out * grad_z，然后sum
    #   4. 三段对应: [:n]对应pre, [n:2*n]对应post, [2*n:]对应res
    # ============================================================================

    grad_alpha_pre = (norm_out_forward[..., :n] * grad_z_pre).sum()

    grad_alpha_post = (norm_out_forward[..., n:2*n] * grad_z_post).sum()

    grad_alpha_res = (norm_out_forward[..., 2*n:] * grad_z_res).sum()

    grad_alpha = torch.tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res], dtype=torch.float32)


    # -------------------------- 步骤6：线性变换的梯度 --------------------------
    # ============================================================================
    # 【步骤6详细说明】线性变换梯度计算
    # ============================================================================
    # 【功能】计算线性变换 z = norm_out * alpha 的反向梯度
    # 
    # 【正向传播】
    #   z_pre = norm_out[..., :n] * alpha[0]
    #   z_post = norm_out[..., n:2*n] * alpha[1]
    #   z_res = norm_out[..., 2*n:] * alpha[2]
    #   (bias在步骤4单独处理)
    # 
    # 【反向传播】
    #   grad_norm_out[..., :n] = grad_z_pre * alpha[0]
    #   grad_norm_out[..., n:2*n] = grad_z_post * alpha[1]
    #   grad_norm_out[..., 2*n:] = grad_z_res * alpha[2]
    # 
    # 【输入约束】
    #   - grad_z_pre: [B,S,n] - pre支路对z的梯度
    #   - grad_z_post: [B,S,n] - post支路对z的梯度
    #   - grad_z_res: [B,S,n*n] - res支路对z的梯度
    #   - alpha: [3] - 缩放因子
    # 
    # 【输出约束】
    #   - grad_norm_out: [B,S,n*n+2*n] - 对norm_out的梯度
    #     组成: concat([grad_norm_out_pre(n), grad_norm_out_post(n), grad_norm_out_res(n*n)])
    # 
    # 【关键理解】
    #   1. 线性变换反向: grad_input = grad_output * weight
    #   2. alpha是标量: broadcast到整个支路
    #   3. 三段concat: 保持与norm_out相同的分段结构
    #   4. 这是norm_out梯度的第一部分，后续还有inv_rms梯度（步骤7）
    # ============================================================================

    # 对norm_out的梯度

    grad_norm_out_pre = grad_z_pre * alpha[0]

    grad_norm_out_post = grad_z_post * alpha[1]

    grad_norm_out_res = grad_z_res * alpha[2]

    grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)


    # -------------------------- 步骤7：乘法的梯度 --------------------------
    # ============================================================================
    # 【步骤7详细说明】乘法梯度计算
    # ============================================================================
    # 【功能】计算 norm_out = hc_before_norm * inv_rms 的反向梯度
    # 
    # 【正向传播】
    #   norm_out[b,s,i] = hc_before_norm[b,s,i] * inv_rms[b,s,0]
    #   (inv_rms broadcast到最后一维)
    # 
    # 【反向传播】
    #   grad_hc_before_norm = grad_norm_out * inv_rms  (broadcast)
    #   grad_inv_rms[b,s,0] = sum_{i} (hc_before_norm[b,s,i] * grad_norm_out[b,s,i])
    # 
    # 【输入约束】
    #   - grad_norm_out: [B,S,n*n+2*n] - 对norm_out的梯度（来自步骤6）
    #   - inv_rms: [B,S,1] - RMSNorm倒数（平方均值倒数的平方根）
    #   - hc_before_norm: [B,S,n*n+2*n] - RMSNorm输入（未归一化）
    # 
    # 【输出约束】
    #   - grad_hc_before_norm: [B,S,n*n+2*n] - 对hc_before_norm的梯度
    #     含义: 矩阵乘法输出的梯度，传给步骤9
    #   - grad_inv_rms: [B,S,1] - 对inv_rms的梯度
    #     含义: RMSNorm倒数的梯度，传给步骤8
    # 
    # 【关键理解】
    #   1. 乘法反向有两支: grad_hc和grad_inv_rms
    #   2. grad_hc: 直接相乘（broadcast inv_rms）
    #   3. grad_inv_rms: 归约sum（最后一维累加）
    #   4. 这是梯度分叉点: 一支去步骤9(matmul)，一支去步骤8(RMSNorm倒数)
    # ============================================================================

    # 对hc_before_norm的梯度

    grad_hc_before_norm = grad_norm_out * inv_rms

    # 对inv_rms的梯度

    grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)


    # -------------------------- 步骤8：RMSNorm倒数的梯度 --------------------------
    # ============================================================================
    # 【步骤8详细说明】RMSNorm倒数梯度计算（倒数倒数）
    # ============================================================================
    # 【功能】计算RMSNorm倒数 inv_rms 的反向梯度（推导倒数倒数）
    # 
    # 【正向传播RMSNorm】
    #   rms = sqrt(mean(hc_before_norm^2) + eps)
    #   inv_rms = 1 / rms
    #   norm_out = hc_before_norm * inv_rms
    # 
    # 【inv_rms的梯度推导】
    #   inv_rms = 1 / sqrt(mean(x^2) + eps)  (x = hc_before_norm的一部分)
    #   设: rms^2 = mean(x^2) + eps
    #   则: inv_rms = rms^{-1}
    #   
    #   grad_x = d(loss)/d(inv_rms) * d(inv_rms)/d(x)
    #         = grad_inv_rms * (-1) * rms^{-2} * (1/2) * rms^{-1} * (2*x/NC)
    #         = - inv_rms^3 * x * grad_inv_rms / NC
    # 
    # 【输入约束】
    #   - x_fp32: [B,S,n,C] - 输入特征（hc_before_norm的来源）
    #   - inv_rms: [B,S,1] - RMSNorm倒数
    #   - grad_inv_rms: [B,S,1] - 对inv_rms的梯度（来自步骤7）
    # 
    # 【输出约束】
    #   - grad_x_from_rms: [B*S, n*C] - 对x的梯度（flatten后）
    #     含义: RMSNorm倒数倒数梯度，传给步骤10汇总
    # 
    # 【关键理解】
    #   1. 倒数倒数: inv_rms的倒数导致三次方inv_rms^3
    #   2. flatten维度: [B,S,n,C] → [B*S, n*C]
    #      正确方式: flatten(0,1).flatten(1) (合并前2维，再合并剩余)
    #      错误方式: flatten(2) (只flatten后2维，产生[B,S,n*C] 3D错误)
    #   3. NC = n*C: 特征总数（例如n=4,C=64 → NC=256）
    #   4. RMSNormGrad函数: 专门处理inv_rms反向，避免数值不稳定
    # ============================================================================
    x_flat = x_fp32.flatten(2)

    grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)

# -------------------------- 步骤9：矩阵乘法的梯度 --------------------------
    # ============================================================================
    # 【步骤9详细说明】矩阵乘法梯度计算
    # ============================================================================
    # 【功能】计算 hc_before_norm = x @ phi.T + bias 的反向梯度
    # 
    # 【phi形状】基于官方约束：[N²+2N, N*D] = [24, 256]
    #   - N²+2N = n² + 2n = 16 + 8 = 24（三路输出维度：pre/post/res）
    #   - N*D = n * C = 4 * 64 = 256（输入特征维度）
    #   - phi是全局投影矩阵，对所有(B,S)位置相同
    # 
    # 【正向公式】
    #   hc_before_norm[B,S,n*n+2*n] = x_flat[B*S,n*C] @ phi.T[n*C,n*n+2*n] + bias[n*n+2*n]
    #   
    #   关键理解:
    #   - x: [B,S,n,C] → flatten为 [B*S,n*C] (例如[16,256])
    #   - phi: [N²+2N, N*D] = [24,256] (全局投影矩阵)
    #   - phi.T: [N*D, N²+2N] = [256,24]
    #   - hc: [B*S,n*n+2*n] = [16,24] (矩阵乘法结果)
    #   - bias: [n*n+2*n] = [24] (全局偏置)
    # 
    # 【反向传播】基于矩阵乘法链式法则
    #   grad_x = grad_hc @ phi
    #         grad_hc_flat[B*S,n*n+2*n] @ phi[n*n+2*n,n*C] → [B*S,n*C]
    #         reshape为 [B,S,n,C]
    #   
    #   grad_phi = grad_hc.T @ x_flat
    #         grad_hc_flat.T[n*n+2*n,B*S] @ x_flat[B*S,n*C] → [n*n+2*n,n*C]
    #         全局归约: 所有batch和sequence位置累加
    # 
    # 【输入约束】
    #   - grad_hc_before_norm: [B,S,n*n+2*n] - 对hc的梯度（来自步骤7）
    #   - phi: [N²+2N, N*D] = [24,256] - 全局投影矩阵
    #   - x_fp32: [B,S,n,C] - 输入特征矩阵
    # 
    # 【输出约束】
    #   - grad_x_from_matmul: [B,S,n,C] - 对x的梯度
    #   - grad_phi: [N²+2N, N*D] = [24,256] - 对phi的梯度（全局归约）
    # 
    # 【关键理解】
    #   1. phi的正确形状: [N²+2N, N*D] = [24,256] (官方约束)
    #   2. 正向需要transpose: hc = x_flat @ phi.T
    #   3. 反向不需要transpose: grad_x = grad_hc @ phi
    #   4. 全局归约: grad_phi累加所有(B,S)位置
    # ============================================================================
    
    # 对x的梯度
    grad_x_from_matmul = grad_hc_before_norm @ phi
    
    # 对phi的梯度（全局归约：所有batch和sequence位置累加）
    grad_hc_before_norm_flat = grad_hc_before_norm.flatten(0, 1)
    x_flat_flat = x_flat.flatten(0, 1)
    grad_phi = grad_hc_before_norm_flat.T @ x_flat_flat


    # -------------------------- 步骤10：汇总所有梯度 --------------------------
    # ============================================================================
    # 【步骤10详细说明】梯度汇总计算
    # ============================================================================
    # 【功能】汇总所有对x的梯度来源，得到最终梯度
    # 
    # 【x梯度的三个来源】
    #   1. grad_x_from_hin[B,S,n,C]: 来自h_in支路（步骤1）
    #      h_in = sum(x * h_pre)
    #   
    #   2. grad_x_from_rms[B*S,n*C]: 来自RMSNorm倒数倒数（步骤8）
    #      inv_rms = 1/sqrt(mean(x^2)+eps)
    #   
    #   3. grad_x_from_matmul[B,S,n,C]: 来自矩阵乘法反向（步骤9）
    #      hc_before_norm = x @ phi + bias
    # 
    # 【输入约束】
    #   - grad_x_from_hin: [B,S,n,C] - 来自步骤1
    #   - grad_x_from_rms: [B*S,n*C] - 来自步骤8（flatten）
    #   - grad_x_from_matmul: [B,S,n,C] - 来自步骤9
    #   - x: [B,S,n,C] - 原始输入（用于恢复dtype）
    # 
    # 【输出约束】
    #   - grad_x: [B,S,n,C] - 最终对x的梯度
    #     dtype: 与原始x相同（float16）
    #     组成: grad_x_from_hin + grad_x_from_rms.view + grad_x_from_matmul
    # 
    # 【关键理解】
    #   1. 三项累加: 三个梯度来源相加
    #   2. view恢复: grad_x_from_rms需要view回[B,S,n,C]
    #   3. dtype恢复: 转回float16（计算过程用float32保证精度）
    #   4. 不包括grad_phi/alpha/bias: 这些是全局参数，x是输入特征
    # ============================================================================

    grad_x = (

        grad_x_from_hin.view(x.shape)

        + grad_x_from_rms.view(x.shape)

        + grad_x_from_matmul.view(x.shape)

    ).to(x.dtype)



    return grad_x, grad_phi, grad_alpha, grad_bias

    
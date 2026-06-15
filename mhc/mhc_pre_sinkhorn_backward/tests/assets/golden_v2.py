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
       - N²+2N = 16+8 = 24 (三路输出维度：pre/post/res)
       - N*D = 4*64 = 256 (输入特征维度)
       
    2. phi作用：全局投影矩阵
       - 正向：hc_before_norm = x_flat @ phi.T + bias
       - x_flat: [T, N*D] = [16, 256]
       - phi.T: [N*D, N²+2N] = [256, 24]
       - 输出：[T, N²+2N] = [16, 24]

================================================================================
"""

import torch


def SigmoidGrad(z: torch.Tensor, dy: torch.Tensor, is_pre: bool = False, hc_eps: float = 1e-6):
    """Sigmoid激活函数的梯度计算"""
    y = torch.sigmoid(z)
    if is_pre:
        sigma = y - hc_eps
    else:
        sigma = y
    return dy * sigma * (1 - sigma)


def RMSNormGrad(x: torch.Tensor, inv_rms: torch.Tensor, grad_inv_rms: torch.Tensor):
    """RMSNorm倒数的梯度计算"""
    NC = x.size(-1)
    grad_x = - (inv_rms ** 3) * x * grad_inv_rms / float(NC)
    return grad_x


def ExpGrad(x: torch.Tensor, y_grad: torch.Tensor):
    """数值稳定的exp激活梯度计算"""
    n = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(n, device=x.device)[None, :]
    is_max = (idx == arange_idx)
    y = (x - x_max).exp()
    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)
    return y * y_grad - is_max * sum_all


def SinkhornGrad(grad_h_res: torch.Tensor, sum_out: torch.Tensor, norm_out: torch.Tensor):
    """Sinkhorn-Knopp算法的梯度计算"""
    bs, seq_len, n, _ = grad_h_res.shape
    x_grad = grad_h_res
    iters = sum_out.shape[0]
    
    for i in reversed(range(iters)):
        row_sum = sum_out[i][0].view(bs, seq_len, n, 1)
        col_sum = sum_out[i][1].view(bs, seq_len, 1, n)
        x_row_normed = norm_out[i][0].view(bs, seq_len, n, n)
        
        grad_x_row_normed = x_grad / col_sum - (x_grad * x_row_normed / (col_sum ** 2)).sum(dim=-2, keepdim=True)
        x_grad = grad_x_row_normed / row_sum - (grad_x_row_normed * x_row_normed / row_sum).sum(dim=-1, keepdim=True)
    
    return x_grad


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
    
    B, S, n, d = x.shape
    grad_hin_fp32 = grad_hin.float()
    x_fp32 = x.float()
    
    # ================================================================================
    # 步骤1：输入聚合的梯度（h_in门控）
    # ================================================================================
    grad_x_from_hin = grad_hin_fp32[..., None, :] * h_pre[..., None]
    grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
    
    # ================================================================================
    # 步骤2：计算正向的中间变量（先完整计算，再拆分）
    # ================================================================================
    norm_out_forward = hc_before_norm * inv_rms
    norm_out_forward_part1 = norm_out_forward[..., :n]
    norm_out_forward_part2 = norm_out_forward[..., n:2*n]
    norm_out_forward_part3 = norm_out_forward[..., 2*n:]
    
    z_pre = norm_out_forward_part1 * alpha[0] + bias[:n]
    z_post = norm_out_forward_part2 * alpha[1] + bias[n:2*n]
    z_res = (norm_out_forward_part3 * alpha[2] + bias[2*n:]).reshape(B, S, n, n)
    
    # ================================================================================
    # 步骤3：pre/post两路激活函数的梯度
    # ================================================================================
    grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)
    grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False, hc_eps=hc_eps)
    
    # ================================================================================
    # 步骤4：pre/post两路缩放因子alpha的梯度
    # ================================================================================
    grad_alpha_pre = (norm_out_forward_part1 * grad_z_pre).sum()
    grad_alpha_post = (norm_out_forward_part2 * grad_z_post).sum()
    
    # ================================================================================
    # 步骤5：pre/post两路偏置bias的梯度（先分计算）
    # ================================================================================
    grad_bias_part1 = grad_z_pre.sum((0, 1))
    grad_bias_part2 = grad_z_post.sum((0, 1))
    
    # ========== 【llh负责：步骤1-5】 ==========
    
    # ================================================================================
    # 步骤6：res支路激活函数的梯度计算
    # ================================================================================
    sk_grad = SinkhornGrad(grad_h_res, sum_out, norm_out)
    grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
    
    # ================================================================================
    # 步骤7：res支路缩放因子alpha的梯度
    # ================================================================================
    grad_alpha_res = (norm_out_forward_part3 * grad_z_res).sum()
    
    # ================================================================================
    # 步骤8：res支路偏置bias的梯度
    # ================================================================================
    grad_bias_part3 = grad_z_res.sum((0, 1))
    
    # ========== 【hbj负责：步骤6-8】 ==========
    
    # ========== 【lhp负责：步骤9-15】 ==========
    
    # ================================================================================
    # 步骤9：pre/post两路线性变换的梯度
    # ================================================================================
    grad_norm_out_pre = grad_z_pre * alpha[0]
    grad_norm_out_post = grad_z_post * alpha[1]

    # ================================================================================
    # 步骤10：res支路线性变换的梯度
    # ================================================================================
    grad_norm_out_res = grad_z_res * alpha[2]
    
    # ================================================================================
    # 步骤11：组装完整的三路梯度
    # ================================================================================
    grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
    
    # ================================================================================
    # 步骤12：完整的三路乘法梯度
    # ================================================================================
    grad_hc_before_norm = grad_norm_out * inv_rms
    grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)
    

    
    # ================================================================================
    # 步骤13：偏置bias的梯度组装
    # ================================================================================
    
    # ================================================================================
    # 步骤14：RMSNorm倒数的梯度
    # ================================================================================
    # 【功能】计算RMSNorm倒数对x的梯度
    # 【反向】grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)
    # 【维度】x[B,S,N,D] → x_flat[B,S,N*D]=[B,S,256]
    x_flat = x_fp32.flatten(2)
    grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)
    
    # ================================================================================
    # 步骤15：矩阵乘法的梯度 + grad_x汇总
    # ================================================================================
    # 【功能】计算矩阵乘法梯度 + 汇总所有对x的梯度
    # 【反向】
    #   - grad_x_from_matmul = grad_hc @ phi
    #   - grad_phi = grad_hc.T @ x_flat
    #   - grad_x = grad_x_from_hin + grad_x_from_rms + grad_x_from_matmul
    # 【维度】
    #   - grad_hc[B,S,24] @ phi[24,256] → grad_x_from_matmul[B,S,256]
    #   - grad_hc.T[24,B*S] @ x_flat[B*S,256] → grad_phi[24,256]
    #   - grad_x[B,S,N,D] + [B,S,N*D] + [B,S,N*D] → grad_x[B,S,N,D]
    grad_x_from_matmul = grad_hc_before_norm @ phi
    grad_hc_before_norm_flat = grad_hc_before_norm.flatten(0, 1)
    x_flat_flat = x_flat.flatten(0, 1)
    grad_phi = grad_hc_before_norm_flat.T @ x_flat_flat
    
    grad_x = (
        grad_x_from_hin.view(x.shape)
        + grad_x_from_rms.view(x.shape)
        + grad_x_from_matmul.view(x.shape)
    ).to(x.dtype)

    # ================================================================================
    # 步骤16：组装grad_alpha和grad_bias【zyf负责】
    # ================================================================================
    # 【功能】组装三路的alpha梯度 + 三路的bias梯度
    # 【组装】
    #   - grad_alpha = tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res])
    #   - grad_bias = cat([grad_bias_part1, grad_bias_part2, grad_bias_part3])
    # 【维度】
    #   - grad_alpha[3]
    #   - grad_bias[N²+2N] = [24]
    grad_alpha = torch.tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res], dtype=torch.float32)
    grad_bias = torch.cat([grad_bias_part1, grad_bias_part2, grad_bias_part3])

    # ========== 【zyf负责最后 ：grad_alpha grad_bias 组装】==========

    
    return grad_x, grad_phi, grad_alpha, grad_bias
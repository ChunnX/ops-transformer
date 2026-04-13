from typing import List

import torch
import torch_npu
import torch.nn as nn
from torch.optim.adamw import _single_tensor_adamw


class Model(nn.Module):
    """
    实现adamW优化器功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor, grad: torch.Tensor,
                step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float, amsgrad: bool,
                maximize: bool) -> List[torch.Tensor]:
        """
        实现adamW优化器功能。

        Args:
            var_ref: 待计算的权重输入同时也是输出，公式中的theta
            m_ref: adamw优化器中m参数，公式中的m
            v_ref: adamw优化器中v参数，公式中的v
            max_grad_norm_ref: 保存v参数的最大值，公式中的v_max
            grad: 梯度数据，公式中的gt
            step: 迭代次数，公式中的t
            lr: 学习率，公式中的gamma
            beta1: beta1参数
            beta2: beta2参数
            weight_decay: 权重衰减系数, 公式中的λ
            eps: 防止除数为0, 公式中的ε
            amsgrad: 是否使用算法的AMSGrad变体
            maximize: 是否最大化参数

        Returns:
            更新后的var_ref, m_ref, v_ref和max_grad_norm_ref列表
        """
        dtype1 = var_ref.dtype
        if grad.dtype != dtype1:
            grad = grad.to(dtype1)
        if max_grad_norm_ref.dtype != dtype1:
            max_grad_norm_ref = max_grad_norm_ref.to(dtype1)

        # 与 kernel 内 float 累加一致：参考在 float32 上算，再写回原 dtype
        compute_dtype = torch.float32
        p = var_ref.to(compute_dtype).clone()
        exp_avg = m_ref.to(compute_dtype).clone()
        exp_avg_sq = v_ref.to(compute_dtype).clone()
        g = grad.to(compute_dtype).clone()
        if amsgrad:
            max_exp = max_grad_norm_ref.to(compute_dtype).clone()
            max_list: List[torch.Tensor] = [max_exp]
        else:
            max_list = [torch.zeros(1, dtype=compute_dtype, device=p.device)]
        st = step.clone()

        _single_tensor_adamw(
            [p],
            [g],
            [exp_avg],
            [exp_avg_sq],
            max_list,
            [st],
            grad_scale=None,
            found_inf=None,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            has_complex=False,
        )

        var_ref.copy_(p.to(dtype1))
        m_ref.copy_(exp_avg.to(dtype1))
        v_ref.copy_(exp_avg_sq.to(dtype1))
        if amsgrad:
            max_grad_norm_ref.copy_(max_list[0].to(dtype1))

        if amsgrad:
            return [var_ref, m_ref, v_ref, max_grad_norm_ref]
        return [var_ref, m_ref, v_ref]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor, grad: torch.Tensor,
                step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float, amsgrad: bool,
                maximize: bool) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.apply_adam_wv2(var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref,
                                             lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
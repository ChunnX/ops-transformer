# -*- coding: utf-8 -*-
# gdd_executor_premix.py - mhc_pre_sinkhorn_premix 正反向 ATK 执行器插件
# 基于 41 环境 mhcPreSinkhorn / mhcPreSinkhornBackward 模板移植:
#   - CPU golden 公式与正式版可选输入能力版完全一致 (e9-4 修复口径: sinkhorn 加法 eps)
#   - 注册名改为 experimental premix 算子: aclnn_mhc_pre_sinkhorn_premix(_backward)
#   - premix C API: fwd 10输入+8输出 (premix=idx4 可选); bwd 14张量+hcEps+5输出
#     (preMix=idx12/gradPre=idx13 可选, gradPreMix 第5输出仅 hasPreMix 时写)
#   - needHPre = needBackward || hasPreMix (kernel m_split_core.h L268 实证)
# 4 注册类:
#   function_aclnn_mhc_pre_sinkhorn_premix           - fwd CPU golden
#   AclnnBaseApi_aclnn_mhc_pre_sinkhorn_premix       - fwd NPU pyaclnn
#   function_aclnn_mhc_pre_sinkhorn_premix_backward  - bwd CPU golden
#   AclnnBaseApi_aclnn_mhc_pre_sinkhorn_premix_backward - bwd NPU pyaclnn
import os
import warnings
from typing import Any

import torch
import torch_npu
import torch.nn.functional as F

import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase, AclDataType, AclFormat, TensorPtr

torch.use_deterministic_algorithms(True)


def pm_save_tensors(tag, kwargs, outputs=None):
    """调试用: 保存执行器输入/输出, 路径 = $PM_SAVE_INPUT.{tag}.pt (环境变量控制)"""
    path = os.environ.get("PM_SAVE_INPUT")
    if not path:
        return
    try:
        data = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in (kwargs or {}).items()}
        if outputs is not None:
            data["__outputs__"] = [o.detach().cpu().clone() if isinstance(o, torch.Tensor) else o
                                   for o in outputs]
        out_path = f"{path}.{tag}.pt"
        torch.save(data, out_path)
        print(f"[PM_SAVE] saved {tag} -> {out_path}")
    except Exception as e:
        print(f"[PM_SAVE] save {tag} failed: {e}")


def custreducesum(x, dim):
    cs = x.shape
    csL = list(cs)
    csL[len(cs) + dim] = 1
    out = torch.zeros(csL, dtype=x.dtype, device=x.device)
    for i in range(x.shape[dim]):
        ss = i
        out += x.select(dim, ss).unsqueeze(dim)
    return out


def sinkhorn_backward(grad_output, sum_out, norm_out, num_iters=20, eps=1e-6):
    """
    Sinkhorn变换反向传播 (e9-4 修复口径: 前向为加法 eps)
    """
    grad_curr = grad_output

    for i in range(num_iters - 1, 0, -1):
        idx = i * 2 + 1
        # D2: iter>=1 的 sum_out 存储值已含 +eps（前向 Adds 后 CopyOut），直接使用
        grad_curr = grad_curr / sum_out[idx].unsqueeze(-2) - custreducesum(grad_curr * norm_out[2 * i], -2) / sum_out[idx].unsqueeze(-2) ** 2
        idx = i * 2

        grad_curr = grad_curr / sum_out[idx].unsqueeze(-1) - (grad_curr * (norm_out[idx]) / sum_out[idx].unsqueeze(-1)).sum(dim=-1, keepdim=True)

    # step1（i=0）保持 +eps：sum_out[1] 为 raw 落盘，+eps 还原前向真实除数
    grad_curr = grad_curr / (sum_out[1] + eps).unsqueeze(-2) - custreducesum(grad_curr * norm_out[0], -2) / (sum_out[1] + eps).unsqueeze(-2) ** 2
    # D3: softmax 乘数还原 p = norm_out[0] − eps（step1 的 Σ(g·norm) 项仍用 p+eps 本值）
    p = norm_out[0] - eps
    dot_prod = (grad_curr * p).sum(dim=-1, keepdim=True)

    grad_input = (grad_curr - dot_prod) * p
    return grad_input


def RMSNormGrad(x, rsqrt, grad_rsqrt):
    _, _, NC = x.shape
    grad_x = (- (rsqrt ** 3)) * grad_rsqrt / float(NC) * x.flatten(2)
    return grad_x.view(x.shape)


def SigmoidGrad(x, grad_output):
    x_sigmoid = 1 / (1 + (-x).exp())
    return x_sigmoid * (1 - x_sigmoid) * grad_output


# =====================================================================
# Forward: CPU golden (双标杆 cpu benchmark 节点)
# =====================================================================
@register("function_aclnn_mhc_pre_sinkhorn_premix")
class FunctionMhcPreSinkhornPremixApi(BaseApi):
    def _fp32_to_tf32(self, data: torch.Tensor) -> torch.Tensor:
        """将 FP32 张量转换为模拟 NPU/TensorCore TF32 格式的张量。"""
        data = data.to(torch.float32).contiguous()
        u = data.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        u = ((u + 0x1000) >> 13) << 13
        u = u & 0xFFFFFFFF
        return u.to(torch.int32).view(torch.float32)

    def softmax_func(self, x, axis=None):
        x_max, _ = x.max(axis=axis, keepdims=True)
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(axis=axis, keepdims=True)
        ans = y / x_sum
        return ans, x_max, x_sum

    def _hc_split_sinkhorn_torch_full(
            self,
            mixes: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            hc_mult: int,
            sinkhorn_iters: int,
            eps: float,
            need_backward: bool):
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
        post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

        comb, _, row_sum = self.softmax_func(comb, -1)
        comb = comb + eps
        sum_out_list = []
        norm_out_list = []
        sum_out_list.append(row_sum.squeeze(-1))
        norm_out_list.append(comb)
        col_sum = comb.sum(-2, keepdim=True)
        sum_out_list.append(col_sum.squeeze(-2))
        comb = comb / (col_sum + eps)
        # 性能优化不再需要填充数据 (奇数槽零占位, 与 kernel 落盘口径对齐)
        zero_sum = torch.zeros_like(comb)
        norm_out_list.append(zero_sum)
        for x in range(sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            row_sum = row_sum + eps
            sum_out_list.append(row_sum.squeeze(-1))
            comb = comb / row_sum
            norm_out_list.append(comb)
            col_sum = comb.sum(-2, keepdim=True)
            col_sum = col_sum + eps
            sum_out_list.append(col_sum.squeeze(-2))
            comb = comb / col_sum
            zero_sum = torch.zeros_like(comb)
            norm_out_list.append(zero_sum)
        sum_out = torch.stack(sum_out_list, dim=0)
        norm_out = torch.stack(norm_out_list, dim=0)
        return pre, post, comb, sum_out, norm_out

    def _hc_pre_full(
            self,
            x: torch.Tensor,
            hc_fn: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            hc_mult: int,
            hc_sinkhorn_iters: int,
            norm_eps: float,
            hc_eps: float,
            need_backward: bool,
            pre_mix: torch.Tensor = None):
        shape, dtype = x.size(), x.dtype
        if x.dim() == 4:
            x_flat = x.flatten(2)
        elif x.dim() == 3:
            x_flat = x.flatten(1)
        else:
            raise ValueError(f"x must be 3D or 4D, got {x.dim()}D")

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        inv_rms = rsqrt.unsqueeze(-1)

        if (x.dtype != torch.float64):
            x_flat = self._fp32_to_tf32(x_flat)
            hc_fn = self._fp32_to_tf32(hc_fn)

        mixes = F.linear(x_flat, hc_fn)
        hc_before_norm = mixes
        mixes = mixes * rsqrt

        pre, post, comb, sum_out, norm_out = self._hc_split_sinkhorn_torch_full(
            mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, need_backward)

        # kernel 语义 (m_split_core.h): hasPreMix 时 y 的 mix 权重用 pre_mix 替换 pre (二选一)
        if pre_mix is not None:
            mix_weight = pre_mix
        else:
            mix_weight = pre
        y = torch.sum(mix_weight.unsqueeze(-1) * x.view(shape), dim=2)
        inv_rms = inv_rms.flatten(start_dim=2)
        comb = comb.flatten(start_dim=2)

        return y, post, comb, pre, hc_before_norm, inv_rms, sum_out, norm_out

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        phi = input_data.kwargs["phi"]
        alpha = input_data.kwargs["alpha"]
        bias = input_data.kwargs["bias"]
        hc_mult = input_data.kwargs["hcMult"]
        num_iters = input_data.kwargs["numIters"]
        hc_eps = input_data.kwargs["hcEps"]
        norm_eps = input_data.kwargs["normEps"]
        need_backward = input_data.kwargs["needGrad"]
        # 可选输入 preMix: 存在场景为 [bs, seq, n] fp32 tensor, 不存在场景为 None
        pre_mix = input_data.kwargs.get("preMix")

        if self.device == 'cpu':
            try:
                ##----双标杆使用
                x_torch = x.to(phi.dtype)
                phi_torch = phi
                alpha_torch = alpha
                bias_torch = bias
                pre_mix_torch = pre_mix.to(phi.dtype) if pre_mix is not None else None

                if torch.any(torch.isinf(x_torch)) or torch.any(torch.isnan(x_torch)):
                    warnings.warn("Input x contains inf or nan values.", RuntimeWarning)

                (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut) = self._hc_pre_full(
                    x_torch, phi_torch, alpha_torch, bias_torch,
                    hc_mult, num_iters, norm_eps, hc_eps, need_backward, pre_mix_torch
                )

                hin = hin.to(x.dtype)
                hPost = hPost
                hRes = hRes
                if need_backward:
                    hPre = hPre
                    hcBeforeNorm = hcBeforeNorm
                    invRms = invRms
                    sumOut = sumOut
                    normOut = normOut
                    return hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut
                else:
                    # needGrad=false: kernel 仅输出 hin/hPost/hRes, hPre 仅在 hasPreMix 时输出,
                    # hcBeforeNorm/invRms/sumOut/normOut 为空, 用零占位与 NPU 侧对齐
                    zero = torch.tensor([0]).contiguous()
                    hPre = hPre if pre_mix is not None else zero
                    return hin, hPost, hRes, hPre, zero, zero, zero, zero
            except Exception:
                (bs, seq_len, n, c) = x.shape
                hin = torch.zeros(bs, seq_len, c).to(x.dtype)
                hPost = torch.zeros(bs, seq_len, n)
                hRes = torch.zeros(bs, seq_len, n * n)
                hPre = torch.zeros(bs, seq_len, n)
                hcBeforeNorm = torch.zeros(bs, seq_len, n * n + 2 * n)
                invRms = torch.zeros(bs, seq_len, 1)
                sumOut = torch.zeros(2 * num_iters, bs, seq_len, n)
                normOut = torch.zeros(2 * num_iters, bs, seq_len, n, n)
                return hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut


# =====================================================================
# Forward: NPU pyaclnn 执行器
# =====================================================================
@register("AclnnBaseApi_aclnn_mhc_pre_sinkhorn_premix")
class AclnnMhcPreSinkhornPremixApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        self.Backward_flag = input_data.kwargs["needGrad"]
        # 可选输入 preMix: 用例 JSON 中缺失时 kwargs 里无该键 -> hasPreMix=False
        self.PreMix_flag = input_data.kwargs.get("preMix") is not None
        if not self.PreMix_flag:
            # YAML 未声明 preMix 时, input_args 只有 9 输入 + 8 输出 (与 C API 18 参错位)。
            # 在第 5 位 (bias 之后) 插入 NULL 的 aclTensor 指针:
            # aclnn 实现侧 c10::optional<aclTensor*> 收到 nullptr -> nullopt,
            # kernel tiling 侧 GetInputShape(4) == nullptr -> hasPreMix=0。
            input_args.insert(4, TensorPtr())
        return input_args, output_packages

    # needBackward=true 返回 8 输出 (normOut 偶数槽掩码, kernel 奇数槽未清零);
    # needBackward=false 返回前 3 项 + hPre(hasPreMix 时为真实值, 否则零占位) + 4 个零占位
    def after_call(self, output_packages):
        if self.Backward_flag:
            norm_out_tensor = self.acl_tensor_to_torch(output_packages[7])
            zero_norm = torch.zeros_like(norm_out_tensor)
            zero_norm[::2, :, :, :] = norm_out_tensor[::2, :, :, :]
            zero_norm = zero_norm.contiguous()
            selected = output_packages[:7]
            return [self.acl_tensor_to_torch(el).cpu() for el in selected] + [zero_norm]
        else:
            zero = torch.tensor([0]).contiguous()
            result = [self.acl_tensor_to_torch(el).cpu() for el in output_packages[:3]]
            # hPre: needGrad=false 但 preMix 存在时仍有真实输出 (needHPre = needBackward || hasPreMix)
            result.append(self.acl_tensor_to_torch(output_packages[3]).cpu() if self.PreMix_flag else zero)
            result += [zero] * 4
            return result


# =====================================================================
# Backward: CPU golden (双标杆 cpu benchmark 节点)
# =====================================================================
@register("function_aclnn_mhc_pre_sinkhorn_premix_backward")
class FunctionMhcPreSinkhornPremixBackwardApi(BaseApi):
    def softmax_func(self, x, axis=None):
        x_max, _ = x.max(axis=axis, keepdims=True)
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(axis=axis, keepdims=True)
        ans = y / x_sum
        return ans, x_max, x_sum

    def _hc_split_sinkhorn_torch_full(
            self,
            mixes: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            hc_mult: int,
            sinkhorn_iters: int,
            eps: float,
            need_backward: bool):
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
        post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

        comb1, _, row_sum = self.softmax_func(comb, -1)

        comb = comb1 + eps

        sum_out_list = []
        norm_out_list = []
        sum_out_list.append(row_sum.squeeze(-1))
        norm_out_list.append(comb)
        col_sum = comb.sum(-2, keepdim=True)
        sum_out_list.append(col_sum.squeeze(-2))
        comb = comb / (col_sum + eps)
        norm_out_list.append(comb)

        for x in range(sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            sum_out_list.append(row_sum.squeeze(-1) + eps)
            comb = comb / (row_sum + eps)
            norm_out_list.append(comb)
            col_sum = comb.sum(-2, keepdim=True)
            sum_out_list.append(col_sum.squeeze(-2) + eps)
            comb = comb / (col_sum + eps)
            norm_out_list.append(comb)
        sum_out = torch.stack(sum_out_list, dim=0)
        norm_out = torch.stack(norm_out_list, dim=0)
        return pre, post, comb, sum_out, norm_out

    def _hc_pre_full(
            self,
            x: torch.Tensor,
            hc_fn: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            hc_mult: int,
            hc_sinkhorn_iters: int,
            norm_eps: float,
            hc_eps: float,
            need_backward: bool):
        shape, dtype = x.size(), x.dtype
        if x.dim() == 4:
            x_flat = x.flatten(2)
        elif x.dim() == 3:
            x_flat = x.flatten(1)
        else:
            raise ValueError(f"x must be 3D or 4D, got {x.dim()}D")

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        inv_rms = rsqrt.unsqueeze(-1)

        x_np = x_flat.to(dtype=hc_fn.dtype).detach().cpu().numpy()
        hc_np = hc_fn.detach().cpu().numpy()

        mixes_np = x_np @ hc_np.T  # hc_np 形状 (out_features, in_features)

        # 转回 PyTorch 张量，放回原设备
        device = x_flat.device
        mixes = torch.from_numpy(mixes_np).to(device=device, dtype=hc_fn.dtype)

        hc_before_norm = mixes
        mixes = mixes * rsqrt

        pre, post, comb, sum_out, norm_out = self._hc_split_sinkhorn_torch_full(
            mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, need_backward)

        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        inv_rms = inv_rms.flatten(start_dim=2)
        comb = comb.flatten(start_dim=2)

        return y, post, comb, pre, hc_before_norm, inv_rms, sum_out, norm_out

    def _golden(self, grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias, h_pre,
                hc_before_norm, inv_rms, sum_out, norm_out, hc_eps, pre_mix, grad_pre_in):
        """CPU golden实现（公式依据 docs/pre_mix_backward_design.md, e9-4 修复口径）。
        pre_mix/grad_pre_in均为None: v1场景(不传可选输入), 返回4元组;
        均传入: v2场景(传入可选输入), 返回5元组(末位为grad_pre_mix=d_mix_for_y)。
          分支A: grad_x_3加权源 = pre_mix(v2) / pre即h_pre(v1)
          分支B合并点: d_pre = (v2: 0 + grad_pre_in) / (v1: d_mix_for_y)
          新可选输出: grad_pre_mix = d_mix_for_y = Σ_c(d_y·x), 仅v2
        """
        try:
            orig_dtype = x.dtype

            B, S, N, D = x.shape
            hc_mult = N

            sk_iter_count = sum_out.shape[0] // 2
            compute_dtype = torch.float64 if x.dtype == torch.float32 else torch.float32

            x_fp32 = x.to(compute_dtype)

            hc_fn = phi
            pre = h_pre
            grad_y = grad_hin.to(compute_dtype)
            grad_post = grad_h_post
            hc_scale = alpha
            hc_base = bias
            h_hat2 = hc_before_norm
            rsqrt = inv_rms
            x_shape = x.shape
            # 分支A: y折叠反传, grad_x_3加权源: v2用pre_mix, v1用pre(h_pre)
            if pre_mix is not None:
                grad_x_3 = grad_y[..., None, :] * pre_mix[..., None].to(compute_dtype)
            else:
                grad_x_3 = grad_y[..., None, :] * pre[..., None]    # [bs, seq_len, n, c]  # 修正D10: 不做中间降精度(对齐kernel fp32行为)
            # y→pre边梯度: d_mix_for_y = Σ_c(d_y·x)  # 修正D6: 先升精度fp32再乘
            d_mix_for_y = custreducesum((grad_y[..., None, :] * x), -1).squeeze(-1)
            # 分支B合并点: d_pre = (has_pre_mix ? 0 : d_mix_for_y) + d_pre_in
            if pre_mix is not None:
                grad_pre_mix = d_mix_for_y  # 新可选输出: d_mix_for_y落盘(fp32)
                d_pre = grad_pre_in
            else:
                d_pre = d_mix_for_y

            # h_hat2 -> h_hat
            h_hat2_rsqrt = h_hat2 * rsqrt

            h_hat_pre = h_hat2[..., :hc_mult] * hc_scale[0] * rsqrt + hc_base[..., :hc_mult]
            h_hat_post = h_hat2[..., hc_mult:(hc_mult * 2)] * hc_scale[1] * rsqrt + hc_base[..., hc_mult:(hc_mult * 2)]
            h_hat_res = (h_hat2[..., (hc_mult * 2):] * hc_scale[2] * rsqrt + hc_base[..., (hc_mult * 2):]).reshape(x_shape[0], x_shape[1], hc_mult, hc_mult)

            grad_h_hat_pre = SigmoidGrad(h_hat_pre, d_pre)

            grad_h_hat_post = 2 * SigmoidGrad(h_hat_post, grad_post)

            grad_h_hat_res = sinkhorn_backward(grad_h_res, sum_out, norm_out, sk_iter_count, hc_eps)

            grad_h_hat_res = grad_h_hat_res.flatten(2)

            grad_hc_base = torch.cat(
                [custreducesum(grad_h_hat_pre.reshape(-1, grad_h_hat_pre.shape[-1]), -2).flatten(),
                 custreducesum(grad_h_hat_post.reshape(-1, grad_h_hat_post.shape[-1]),
                               -2).flatten()
                    , custreducesum(grad_h_hat_res.reshape(-1, N * N), -2).flatten()])
            grad_hc_base1 = torch.cat(
                [grad_h_hat_pre.sum((0, 1)), grad_h_hat_post.sum((0, 1)), grad_h_hat_res.sum((0, 1)).flatten()])

            grad_hc_scale_pre = custreducesum((h_hat2[..., :hc_mult] * (grad_h_hat_pre * rsqrt)).reshape(-1, 1), -2)
            grad_hc_scale_post = custreducesum(
                (h_hat2[..., hc_mult:(hc_mult * 2)] * (grad_h_hat_post * rsqrt)).reshape(-1, 1), -2)
            grad_hc_scale_res = custreducesum(
                (h_hat2[..., (hc_mult * 2):] * (grad_h_hat_res * rsqrt)).reshape(-1, 1),
                -2)
            grad_hc_scale = torch.stack([grad_hc_scale_pre, grad_hc_scale_post, grad_hc_scale_res]).reshape(-1)

            grad_h_hat2_pre = grad_h_hat_pre * hc_scale[0]
            grad_h_hat2_post = grad_h_hat_post * hc_scale[1]
            grad_h_hat2_res = grad_h_hat_res * hc_scale[2]

            grad_h_hat2 = torch.cat([grad_h_hat2_pre, grad_h_hat2_post, grad_h_hat2_res], dim=-1)

            grad_rsqrt = (h_hat2 * grad_h_hat2).sum(-1, keepdim=True)

            grad_x_1 = RMSNormGrad(x_fp32.flatten(2), rsqrt, grad_rsqrt)
            grad_h_hat2 = grad_h_hat2 * rsqrt

            grad_h_hat2cast = grad_h_hat2[..., None, :]
            if (grad_h_hat2.dtype == torch.float32):
                grad_x_2 = grad_h_hat2cast.to(compute_dtype) @ hc_fn.to(compute_dtype)
                grad_hc_fn = grad_h_hat2cast.flatten(0, 1).reshape(-1, N * N + 2 * N).T.to(compute_dtype) @ x_fp32.flatten(2).flatten(0, 1).to(compute_dtype)
            else:
                grad_x_2 = (grad_h_hat2cast @ hc_fn)
                grad_hc_fn = (grad_h_hat2cast.flatten(0, 1).reshape(-1, N * N + 2 * N).T @ x_fp32.flatten(2).flatten(0, 1))

            grad_x = grad_x_3.view(x_shape) + grad_x_1.view(x_shape) + grad_x_2.view(x_shape)

            if pre_mix is not None:
                return (grad_x.to(orig_dtype), grad_hc_fn.to(phi.dtype), grad_hc_scale, grad_hc_base,
                        grad_pre_mix)
            return grad_x.to(orig_dtype), grad_hc_fn.to(phi.dtype), grad_hc_scale, grad_hc_base
        except Exception:
            (B, S, C) = grad_hin.shape
            N = grad_h_post.shape[-1]
            grad_x = torch.zeros(B, S, N, C).to(grad_hin.dtype)
            grad_hc_fn = torch.zeros(2 * N + N * N, N * C)
            grad_hc_scale = torch.zeros(3)
            grad_hc_base = torch.zeros(2 * N + N * N)
            if pre_mix is not None:
                grad_pre_mix = torch.zeros(B, S, N)
                return grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base, grad_pre_mix
            return grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base

    def init_by_input_data(self, input_data: InputDataset):
        try:
            torch.npu.synchronize()

            x = input_data.kwargs.get("x")
            phi = input_data.kwargs.get("phi")
            alpha = input_data.kwargs.get("alpha")
            bias = input_data.kwargs.get("bias")
            sum_out = input_data.kwargs.get("sum_out")
            iter = sum_out.shape[0] // 2
            hc_eps = input_data.kwargs.get("hc_eps", 1e-6)

            input_data.kwargs["hc_eps"] = 1e-6
            torch.use_deterministic_algorithms(True)

            (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut) = self._hc_pre_full(
                x, phi, alpha, bias,
                4, iter, hc_eps, hc_eps, True
            )

            input_data.kwargs["h_pre"].copy_(hPre)
            input_data.kwargs["hc_before_norm"].copy_(hcBeforeNorm)
            input_data.kwargs["inv_rms"].copy_(invRms)
            input_data.kwargs["sum_out"].copy_(sumOut)
            input_data.kwargs["norm_out"].copy_(normOut)
            torch.npu.synchronize()
        except Exception:
            print("input is invalid, for check aclnn error code ")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        torch.use_deterministic_algorithms(True)

        grad_hin = input_data.kwargs.get("grad_hin")
        grad_h_post = input_data.kwargs.get("grad_h_post")
        grad_h_res = input_data.kwargs.get("grad_h_res")
        x = input_data.kwargs.get("x")
        phi = input_data.kwargs.get("phi")
        alpha = input_data.kwargs.get("alpha")
        bias = input_data.kwargs.get("bias")
        h_pre = input_data.kwargs.get("h_pre")
        hc_before_norm = input_data.kwargs.get("hc_before_norm")
        inv_rms = input_data.kwargs.get("inv_rms")
        sum_out = input_data.kwargs.get("sum_out")
        norm_out = input_data.kwargs.get("norm_out")
        hc_eps = input_data.kwargs.get("hc_eps", 1e-6)
        pre_mix = input_data.kwargs.get("pre_mix")
        grad_pre = input_data.kwargs.get("grad_pre")

        if self.device == "cpu":
            # 单场景模式: 场景由 case JSON 是否含 pre_mix 决定,
            # CPU golden 与 NPU pyaclnn 两侧读同一 case, 自然对齐
            common_args = (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
                           h_pre, hc_before_norm, inv_rms, sum_out, norm_out, hc_eps)
            if pre_mix is None:
                # v1 场景: 不传可选输入 pre_mix/grad_pre。
                # 仍必须返回 5 元组(第 5 个 grad_pre_mix 为零占位): 基类按 CPU golden
                # 返回个数分配 NPU 侧 outTensor, 若只返 4 元组则 GetWorkspaceSize 仅
                # 19 参, 与 C API 20 参错位, 报 EZ1001(executor/workspaceSize nullptr)
                outs = self._golden(*common_args, None, None)
                B, S, N = grad_h_post.shape
                return outs + (torch.zeros(B, S, N, dtype=torch.float32),)
            # v2 场景: 传入可选输入, 返回5输出(末位 grad_pre_mix)
            outs = self._golden(*common_args, pre_mix, grad_pre)
            pm_save_tensors("golden_v2", input_data.kwargs, outs)
            return outs


# =====================================================================
# Backward: NPU pyaclnn 执行器
# =====================================================================
@register("AclnnBaseApi_aclnn_mhc_pre_sinkhorn_premix_backward")
class AclnnMhcPreSinkhornPremixBackwardApi(AclnnBaseApi):
    """NPU pyaclnn 执行器: 场景由 case JSON 是否声明 pre_mix/grad_pre 决定"""

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # 可选输入 pre_mix: case JSON 缺失时 kwargs 无该键 -> v1 场景
        self.Optional_flag = input_data.kwargs.get("pre_mix") is not None
        if not self.Optional_flag:
            # v1 场景: YAML 未声明 pre_mix/grad_pre, input_args 为 12 张量 + 1 attr + 5 输出。
            # 在 attr(hc_eps) 前插入两个 NULL 的 aclTensor 指针, 对齐 C API 20 参:
            # preMix=idx12, gradPre=idx13; aclnn 实现侧收到 nullptr -> hasPreMix=0
            input_args.insert(12, TensorPtr())
            input_args.insert(13, TensorPtr())
        pm_save_tensors("npu_input", input_data.kwargs)
        return input_args, output_packages

    def after_call(self, output_packages):
        if self.Optional_flag:
            # v2: gradX/gradPhi/gradAlpha/gradBias/gradPreMix 全为真实输出
            return [self.acl_tensor_to_torch(el).cpu() for el in output_packages[:5]]
        # v1: 与 CPU golden 5 元组对齐必须返回 5 项。前 4 项为真实输出;
        # 第 5 项(gradPreMix) kernel 不写, 值未初始化, 清零与 CPU golden 零占位对齐
        result = [self.acl_tensor_to_torch(el).cpu() for el in output_packages[:5]]
        result[4] = torch.zeros_like(result[4])
        return result

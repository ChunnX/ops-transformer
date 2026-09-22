from typing import Any

from typing import Any

import torch
import torch_npu
import torch.nn.functional as F

import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase, AclDataType, AclFormat

# torch.set_##printoptions(threshold=10)
torch.use_deterministic_algorithms(True)


def custreducesum(x, dim):
    cs = x.shape
    ifhighper = x.dtype == torch.float64

    csL = list(cs)
    csL[len(cs) + dim] = 1

    out = torch.zeros(csL, dtype=x.dtype, device=x.device)
    for i in range(x.shape[dim]):
        ss = i
        # if ifhighper:
        #     ss = x.shape[dim] - i - 1

        out += x.select(dim, ss).unsqueeze(dim)

    return out


def _fp32_to_hf32(data: torch.Tensor) -> torch.Tensor:
    """
    将 FP32 张量转换为模拟 NPU HF32 格式的张量。
    参数：
        data: 任意形状的 torch.float32 张量
    返回：
        转换后的 torch.float32 张量，形状与输入相同
    """
    # 确保输入为 float32 类型，并转为连续内存（便于 view 操作）
    data = data.to(torch.float32).contiguous()

    # 将浮点位模式重新解释为有符号 32 位整数
    data_int = data.view(torch.int32)

    # 模拟 HF32 转换：右移 11 位 -> 加 1 -> 右移 1 位 -> 左移 12 位
    data_int = torch.bitwise_right_shift(data_int, 11)  # >> 11
    data_int = data_int + 1
    data_int = torch.bitwise_right_shift(data_int, 1)  # >> 1
    data_int = torch.bitwise_left_shift(data_int, 12)  # << 12

    # 重新解释为 float32 并返回
    return data_int.view(torch.float32)


def sinkhorn_backward(grad_output, sum_out, norm_out, num_iters=20, eps=1e-6):
    """
    Sinkhorn变换反向传播
    """
    grad_curr = grad_output

    for i in range(num_iters - 1, 0, -1):
        idx = i * 2 + 1
        # D2: iter>=1 的 sum_out 存储值已含 +eps（前向 Adds 后 CopyOut），直接使用
        grad_curr = grad_curr / sum_out[idx].unsqueeze(-2) - custreducesum(grad_curr * norm_out[2 * i],-2 ) / sum_out[idx].unsqueeze(-2) ** 2
        idx = i * 2

        grad_curr = grad_curr / sum_out[idx].unsqueeze(-1) - (grad_curr * (norm_out[idx] ) / sum_out[idx].unsqueeze(-1) ).sum(dim=-1, keepdim=True)

    # step1（i=0）保持 +eps：sum_out[1] 为 raw 落盘，+eps 还原前向真实除数
    grad_curr = grad_curr / (sum_out[1] + eps).unsqueeze(-2) - custreducesum(grad_curr * norm_out[0],-2) / (sum_out[1] + eps).unsqueeze(-2) ** 2
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


@register("function_aclnn_mhc_pre_sinkhorn_backward")
class FunctionApi(BaseApi):
    def softmax_func(self, x, axis=None):
        # x = x.astype(np.float32)
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
        if need_backward:
            sum_out = torch.stack(sum_out_list, dim=0)
            norm_out = torch.stack(norm_out_list, dim=0)
            return pre, post, comb, sum_out, norm_out
        else:
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
            raise ValueError(f"x must be 3D or 4D, got {x.dim()}D")

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        inv_rms = rsqrt.unsqueeze(-1)

        x_np = x_flat.to(dtype=hc_fn.dtype).detach().cpu().numpy()
        hc_np = hc_fn.detach().cpu().numpy()

        mixes_np = x_np @ hc_np.T  # hc_np 形状 (out_features, in_features)

        # 3. 转回 PyTorch 张量，放回原设备
        device = x_flat.device
        mixes = torch.from_numpy(mixes_np).to(device=device, dtype=hc_fn.dtype)

        hc_before_norm = mixes
        mixes = mixes * rsqrt

        pre, post, comb, sum_out, norm_out = self._hc_split_sinkhorn_torch_full(
            mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, need_backward)

        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        inv_rms = inv_rms.flatten(start_dim=2)
        comb = comb.flatten(start_dim=2)

        if need_backward:
            return y, post, comb, pre, hc_before_norm, inv_rms, sum_out, norm_out
        else:
            return y, post, comb, pre, hc_before_norm, inv_rms, sum_out, norm_out

    def init_by_input_data(self, input_data: InputDataset):
        try:
            torch.npu.synchronize()

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
        except:
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

        if self.device == "cpu":
            try:
                orig_dtype = x.dtype
                device = x.device

                B, S, N, D = x.shape
                ND = N * D
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
                b, s, n, c = x.shape
                x_shape = x.shape
                # compute grad_x_3
                grad_x_3 = grad_y[..., None, :] * pre[..., None]    # [bs, seq_len, n, c]  # 修正D10: 不做中间降精度(对齐kernel fp32行为)
                # compute pre_grad
                grad_pre = custreducesum((grad_y[..., None, :].float() * x.float()),-1).squeeze(-1)  # 修正D6: 先升精度fp32再乘

                # h_hat2 -> h_hat
                h_hat2_rsqrt = h_hat2 * rsqrt

                h_hat_pre = h_hat2[..., :hc_mult] * hc_scale[0] * rsqrt + hc_base[..., :hc_mult]
                h_hat_post = h_hat2[..., hc_mult:(hc_mult *2)] * hc_scale[1] * rsqrt + hc_base[..., hc_mult:(hc_mult *2)]
                h_hat_res = (h_hat2[..., (hc_mult *2):] * hc_scale[2] * rsqrt + hc_base[..., (hc_mult *2):]).reshape(x_shape[0], x_shape[1], hc_mult, hc_mult)

                grad_h_hat_pre = SigmoidGrad(h_hat_pre.float(), grad_pre.float())

                grad_h_hat_post = 2 * SigmoidGrad(h_hat_post.float(), grad_post.float())

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
                    grad_hc_fn = grad_h_hat2cast.flatten(0, 1).reshape(-1,N*N+2*N).T.to(compute_dtype) @ x_fp32.flatten(2).flatten(0, 1).to(compute_dtype)
                else:
                    grad_x_2 = (grad_h_hat2cast @ hc_fn)
                    grad_hc_fn = (grad_h_hat2cast.flatten(0, 1).reshape(-1,N*N+2*N).T @ xhf32.flatten(2).flatten(0, 1))


                grad_x = grad_x_3.view(x_shape) + grad_x_1.view(x_shape) + grad_x_2.view(x_shape)

                return grad_x.to(orig_dtype), grad_hc_fn.to(phi.dtype), grad_hc_scale, grad_hc_base
            except:
                (B, S, C) = grad_hin.shape
                N = grad_h_post.shape[-1]
                grad_x = torch.zeros(B, S, N, C).to(grad_hin.dtype)
                grad_hc_fn = torch.zeros(2 * N + N * N, N * C).to(torch.float32)
                grad_hc_scale = torch.zeros(3).to(torch.float32)
                grad_hc_base = torch.zeros(2 * N + N * N).to(torch.float32)
                return grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base

        elif self.device == 'npu':
            x.requires_grad_(True)
            phi.requires_grad_(True)
            alpha.requires_grad_(True)
            bias.requires_grad_(True)
            import cann_ops_transformer.ops.mhc_pre_sinkhorn_backward as _backward_mod
            (gradX, grad_hc_fn, grad_hc_scale, grad_hc_base) = _backward_mod(grad_hin, grad_h_post, grad_h_res,
                                    x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, hc_eps)

            return (gradX, grad_hc_fn, grad_hc_scale, grad_hc_base)

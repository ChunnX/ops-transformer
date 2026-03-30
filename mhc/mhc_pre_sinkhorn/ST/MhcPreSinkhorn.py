# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
import torch.nn.functional as F
import warnings
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import OutputData
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


@register("ascend_function_mhc_pre_sinkhorn")
class MethodTorchMhcPreSinkhornApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchMhcPreSinkhornApi, self).__init__(task_result)

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

        comb = comb.softmax(-1) + eps

        sum_out_list = []
        norm_out_list = []

        row_sum = comb.sum(-1, keepdim=True)

        sum_out_list.append(row_sum.squeeze(-1))
        comb = comb / (row_sum + eps)
        norm_out_list.append(comb)
        col_sum = comb.sum(-2, keepdim=True)
        sum_out_list.append(col_sum.squeeze(-2))
        comb = comb / (col_sum + eps)
        norm_out_list.append(comb)

        for x in range(sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            sum_out_list.append(row_sum.squeeze(-1))
            comb = comb / (row_sum + eps)
            norm_out_list.append(comb)
            col_sum = comb.sum(-2, keepdim=True)
            sum_out_list.append(col_sum.squeeze(-2))
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
            x_flat = x.flatten(2).float()
        elif x.dim() == 3:
            x_flat = x.flatten(1).float()
        else:
            raise ValueError(f"x must be 3D or 4D, got {x.dim()}D")

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        inv_rms = rsqrt.unsqueeze(-1)

        mixes = F.linear(x_flat, hc_fn)
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

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        phi = input_data.kwargs["phi"]
        alpha = input_data.kwargs["alpha"]
        bias = input_data.kwargs["bias"]
        hc_mult = input_data.kwargs["hcMult"]
        num_iters = input_data.kwargs["numIters"]
        hc_eps = input_data.kwargs["hcEps"]
        norm_eps = input_data.kwargs["normEps"]
        need_backward = input_data.kwargs["needBackward"]

        x_torch = x.to(torch.float32)
        phi_torch = phi.to(torch.float32)
        alpha_torch = alpha.to(torch.float32)
        bias_torch = bias.to(torch.float32)

        if torch.any(torch.isinf(x_torch)) or torch.any(torch.isnan(x_torch)):
            warnings.warn("Input x contains inf or nan values.", RuntimeWarning)

        (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut) = self._hc_pre_full(
            x_torch, phi_torch, alpha_torch, bias_torch,
            hc_mult, num_iters, norm_eps, hc_eps, need_backward
        )

        hin = hin.to(torch.bfloat16)
        hPost = hPost.to(torch.float32)
        hRes = hRes.to(torch.float32)
        if need_backward:
            hPre = hPre.to(torch.float32)
            hcBeforeNorm = hcBeforeNorm.to(torch.float32)
            invRms = invRms.to(torch.float32)
            sumOut = sumOut.to(torch.float32)
            normOut = normOut.to(torch.float32)
            return hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut
        else:
            no_output = torch.tensor([0]).to(torch.float32).contiguous()
            return hin, hPost, hRes, no_output, no_output, no_output, no_output, no_output

@register("AclnnBaseApi_aclnn_mhc_pre_sinkhorn")
class AclnnMhcPreSinkHornApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        self.Backward_flag = input_data.kwargs["needBackward"]
        return input_args, output_packages
    
    #如果是true就拼3个，否则为8个
    def after_call(self, output_packages):
        print("---hyh--self.Backward_flag-\n",self.Backward_flag)
        if self.Backward_flag:
            selected = output_packages[:9]
            return [self.acl_tensor_to_torch(el) for el in selected]
        else:
            zero = torch.tensor([0]).to(torch.float32).contiguous()
            selected = output_packages[:3] 
        return [self.acl_tensor_to_torch(el) for el in selected] + [zero] * 5

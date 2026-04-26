# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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


def sinkhorn_backward(grad_output, sum_out, norm_out, num_iters=20, eps=1e-6):
    """
    Sinkhorn变换反向传播
    """
    grad_curr = grad_output
    print("norm_out.shape")
    for i in range(num_iters - 1, 0, -1):
        idx = i * 2 + 1
        # print("col",i,norm_out[idx])

        dot_prod = (grad_curr * norm_out[idx]).sum(dim=-2, keepdim=True)
        # print("col",i,sum_out[idx] + eps)
        # print("col",i,sum_out[idx])
        # print("t01",i,grad_curr)
        # print("t01",i,sum_out[idx].shape)
        # print("t01",i,norm_out[idx].shape)

        t0 = (sum_out[idx] + eps).unsqueeze(-2)
        t1 = grad_curr / ((sum_out[idx] + eps).unsqueeze(-2))
        t2 = (grad_curr * norm_out[2 *i]).sum(dim=-2, keepdim=True)
        t3 = t2 / ((sum_out[idx] + eps)*(sum_out[idx] + eps)).unsqueeze(-2)
        # print("t0",i,t0[:,:,:].shape)
        # print("t1",i,t1,t1.shape)
        # print("t2",i,t2)
        # print("t3",i,t3)
        # print("t4",i,t1-t3)
        # print("norm_out[idx]",i,norm_out[idx])
        # print("norm_out[idx]",i,norm_out[2 *i]/(sum_out[idx] + eps))

        # print("tt",i, dot_prod / (sum_out[idx] + eps).unsqueeze(-2))
        # print("tt1",i, (grad_curr * norm_out[2 *i]).sum(dim=-2, keepdim=True) / ((sum_out[idx] + eps)*(sum_out[idx] + eps)).unsqueeze(-2))
        # print("tt1",i, (grad_curr * norm_out[2 *i]/ (sum_out[idx] + eps)).sum(dim=-2, keepdim=True) / ((sum_out[idx] + eps)).unsqueeze(-2))

        grad_curr = (grad_curr - dot_prod) / (sum_out[idx] + eps).unsqueeze(-2)


        # print("coldot_prod",i,dot_prod)
        print("col",i,grad_curr)

        idx = i * 2

        dot_prod = (grad_curr * norm_out[idx]).sum(dim=-1, keepdim=True)

        # print("grad_curr / ((sum_out[idx] + eps).unsqueeze(-1))",i, grad_curr / ((sum_out[idx] + eps).unsqueeze(-1)))

        # print("dot_prod / ((sum_out[idx] + eps).unsqueeze(-1))",i, dot_prod / ((sum_out[idx] + eps).unsqueeze(-1)))
        # print("(grad_curr * norm_out[idx]) / ((sum_out[idx] + eps).unsqueeze(-1))",i, (grad_curr * norm_out[idx]) / ((sum_out[idx] + eps).unsqueeze(-1)))

        # print("row(sum_out[idx] + eps).unsqueeze(-1)ow",i,(sum_out[idx] + eps).unsqueeze(-1))
        dot_prod = (grad_curr * norm_out[idx]).sum(dim=-1, keepdim=True)
        grad_curr = (grad_curr - dot_prod) / ((sum_out[idx] + eps).unsqueeze(-1))
        print("row",i,grad_curr)


    dot_prod = (grad_curr * norm_out[1]).sum(dim=-2, keepdim=True)
    grad_curr = (grad_curr - dot_prod) / (sum_out[1] + eps).unsqueeze(-2)


    dot_prod = (grad_curr * norm_out[0]).sum(dim=-1, keepdim=True)

    grad_input = (grad_curr - dot_prod) * norm_out[0]
    print("ngrad_input",grad_input)

    return grad_input

def RMSNormGrad(x, rsqrt, grad_rsqrt):
    _, _, NC = x.shape
    print("rsqrtffff1",grad_rsqrt)

    print("rsqrtffff",(- (rsqrt ** 3) ) * grad_rsqrt / float(NC))
    grad_x = (- (rsqrt ** 3) * x.flatten(2)) * grad_rsqrt / float(NC)
    return grad_x.view(x.shape)

def SigmoidGrad(x, grad_output):
    # x_sigmoid = x.sigmoid()
    x_sigmoid = 1 / (1 + (-x).exp())
    print("x_sigmoid",x_sigmoid)
    print("11111",x_sigmoid* grad_output)
    print("222222",x_sigmoid*x_sigmoid* grad_output)
    print("33333",x_sigmoid * (1 - x_sigmoid) )

    return x_sigmoid * (1 - x_sigmoid) * grad_output
@register("function_aclnn_mhc_pre_sinkhorn_backward")
class FunctionApi(BaseApi):
    def softmax_func(self, x, axis=None):
        # x = x.astype(np.float32)
        x_max, _ = x.max(axis=axis, keepdims = True)
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(axis=axis, keepdims = True)
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

        # softmax之后的这个结果不要加了
        # 从中间把0结果取了
	    # comb = comb.softmax(-1) + eps
        comb, _, row_sum = self.softmax_func(comb, -1)
        comb = comb + eps

        sum_out_list = []
        norm_out_list = []
        # row_sum = comb.sum(-1, keepdim=True)
        sum_out_list.append(row_sum.squeeze(-1))
        # comb = comb / (row_sum + eps)
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
            x_flat = x.flatten(2)
        elif x.dim() == 3:
            x_flat = x.flatten(1)
            raise ValueError(f"x must be 3D or 4D, got {x.dim()}D")
        print("norm_eps",norm_eps)
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        inv_rms = rsqrt.unsqueeze(-1)



        # mixes的计算精度达不到，考虑使用float64或双标杆
        mixes = F.linear(x_flat.to(dtype=hc_fn.dtype), hc_fn) #todo 换成torch.matmul
        # mixes = torch.matmul(x_flat, hc_fn)
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

        # torch.npu.synchronize()
        print(f"init_by_input_data")

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

        (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut) = self._hc_pre_full(
            x, phi, alpha, bias,
            4, 20, hc_eps, hc_eps, True
        )
        input_data.kwargs["h_pre"] = hPre
        input_data.kwargs["hc_before_norm"] = hcBeforeNorm
        input_data.kwargs["inv_rms"] = inv_rms

        input_data.kwargs["sum_out"] = sumOut 
        # sumOut[:,:,:,0]=0
        # sumOut[:,:,:,1]=1
        # sumOut[:,:,:,2]=2
        # sumOut[:,:,:,3]=3
        # for i in range (x.shape[0]):
        #     x[i]=1
        # for i in range (grad_hin.shape[0]):
        #     grad_hin[i]=i
        input_data.kwargs["x"] = x 
        input_data.kwargs["grad_hin"] = grad_hin 

        # for i in range (grad_h_res.shape[0]):
        #     grad_h_res[i]=i
        # input_data.kwargs["grad_h_res"] = grad_h_res

        #     for j in range (4
        #         normOut[:,:,:,i,j]=1 * 4 + j
        input_data.kwargs["norm_out"] = normOut
        # for i in range (4):
        #     for j in range (4):
        #         normOut[:,:,:,i,j]=1 * 4 + j
        print(f"sum_out",sumOut.shape)
        print(f"norm_out",normOut.shape)

        print(f"init_by_input_data")





    def __call__(self, input_data: InputDataset, with_output: bool = False):
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
        
        print(f"---------------MhcPreSinkhornBackward---------------------")
        print(f"grad_hin shape: {grad_hin.shape}")
        print(f"grad_h_post shape: {grad_h_post.shape}")
        print(f"grad_h_res shape: {grad_h_res.shape}")
        print(f"x shape: {x.shape}")
        print(f"phi shape: {phi.shape}")
        print(f"alpha shape: {alpha.shape}")
        print(f"bias shape: {bias.shape}")
        print(f"h_pre shape: {h_pre.shape}")
        print(f"hc_before_norm shape: {hc_before_norm.shape}")
        print(f"inv_rms shape: {inv_rms.shape}")
        print(f"sum_out shape: {sum_out.shape}")
        print(f"norm_out shape: {norm_out.shape}")

        orig_dtype = x.dtype
        device = x.device
        
        B, S, N, D = x.shape
        ND = N * D
        hc_mult = N

        sk_iter_count = sum_out.shape[0] // 2
        
        grad_hin_fp32 = grad_hin.cpu().float()
        grad_h_post_fp32 = grad_h_post.cpu().float()
        grad_h_res_fp32 = grad_h_res.cpu().float()
        x_fp32 = x.cpu().float()
        phi_fp32 = phi.cpu().float()
        alpha_fp32 = alpha.cpu().float()
        bias_fp32 = bias.cpu().float()
        h_pre_fp32 = h_pre.cpu().float()
        hc_before_norm_fp32 = hc_before_norm.cpu().float()
        inv_rms_fp32 = inv_rms.cpu().float()
        sum_out_fp32 = sum_out.cpu().float()
        norm_out_fp32 = norm_out.cpu().float()
        x_fp32 = x.to(dtype=phi.dtype).clone()
        hc_fn = phi
        pre = h_pre
        grad_y = grad_hin
        grad_post = grad_h_post
        hc_scale = alpha
        hc_base = bias
        h_hat2 = hc_before_norm
        rsqrt = inv_rms
        b, s, n, c = x.shape
        x_shape = x.shape
        grad_y_fp32 = grad_y.to(dtype=pre.dtype)
        # compute grad_x_3
        grad_x_3 = grad_y_fp32[..., None, :] * pre[..., None]    # [bs, seq_len, n, c]
        # compute pre_grad
        grad_pre = (grad_y_fp32[..., None, :] * x_fp32).sum(-1)

        # h_hat2 -> h_hat
        h_hat2_rsqrt = h_hat2 * rsqrt
        h_hat_pre = h_hat2_rsqrt[..., :hc_mult] * hc_scale[0] + hc_base[..., :hc_mult]
        h_hat_post = h_hat2_rsqrt[..., hc_mult:(hc_mult *2)] * hc_scale[1] + hc_base[..., hc_mult:(hc_mult *2)]
        h_hat_res = (h_hat2_rsqrt[..., (hc_mult *2):] * hc_scale[2] + hc_base[..., (hc_mult *2):]).reshape(x_shape[0], x_shape[1], hc_mult, hc_mult)
        print("h_hat_pre",h_hat_pre)
        print("h_hat_post",h_hat_post) 
        print("h_hat_res",h_hat_res) 
        print("grad_pre",grad_pre) 
        print("grad_pre[:256,0,0]",grad_pre[:256,0,0])
        print("x_fp32",x_fp32) 

        print("grad_post",grad_post) 

        grad_h_hat_pre = SigmoidGrad(h_hat_pre, grad_pre)
        grad_h_hat_post = 2 * SigmoidGrad(h_hat_post, grad_post)
        print("grad_h_hat_pre",grad_h_hat_pre)
        print("grad_h_hat_post",grad_h_hat_post)

        grad_h_hat_res = sinkhorn_backward(grad_h_res, sum_out, norm_out, sk_iter_count, hc_eps)
        print("grad_h_hat_res",grad_h_hat_res)
        print("grad_h_hat_res",grad_h_hat_res)

        grad_h_hat_res = grad_h_hat_res.flatten(2)
        print("grad_h_hat_res[:256,0,0]",grad_h_hat_res[:256,0,0])

        grad_hc_base1 = torch.cat([grad_h_hat_pre, grad_h_hat_post, grad_h_hat_res.reshape(grad_h_hat_res.shape[0],grad_h_hat_res.shape[1],-1)],dim=-1)
        print(grad_hc_base1)

        grad_hc_base = torch.cat([grad_h_hat_pre.sum((0, 1)), grad_h_hat_post.sum((0, 1)), grad_h_hat_res.sum((0, 1)).flatten()])
        print("grad_hc_base",grad_hc_base)

        # print("hcbase")
        # print(grad_h_hat_pre)
        # print(grad_h_hat_post)
        print(grad_hc_base1 * h_hat2_rsqrt,"grad_hc_scale_11") 
        grad_hc_scale_pre = (h_hat2_rsqrt[..., :hc_mult] * grad_h_hat_pre).sum()
        grad_hc_scale_post = (h_hat2_rsqrt[..., hc_mult:(hc_mult *2)] * grad_h_hat_post).sum()
        grad_hc_scale_res = (h_hat2_rsqrt[..., (hc_mult *2):] * grad_h_hat_res).sum()
        grad_hc_scale = torch.stack([grad_hc_scale_pre, grad_hc_scale_post, grad_hc_scale_res])
        print("grad_hc_scale",grad_hc_scale)
        # print("hcscale")
        # print(grad_hc_scale_pre)
        # print(grad_hc_scale_post)
        # print(h_hat2_rsqrt[..., :hc_mult] * grad_h_hat_pre)
        # print(h_hat2_rsqrt[..., hc_mult:(hc_mult *2)] * grad_h_hat_post)

        grad_h_hat2_pre = grad_h_hat_pre * hc_scale[0]
        grad_h_hat2_post = grad_h_hat_post * hc_scale[1]
        grad_h_hat2_res = grad_h_hat_res * hc_scale[2]

        grad_h_hat2 = torch.cat([grad_h_hat2_pre, grad_h_hat2_post, grad_h_hat2_res], dim=-1)
        print("grad_h_hat2",grad_h_hat2)

        grad_rsqrt = (h_hat2 * grad_h_hat2).sum(-1, keepdim=True)

        # print(grad_rsqrt)
        # print(h_hat2 * grad_h_hat2)
        # print(h_hat2)
        # print((h_hat2 * grad_h_hat2)[..., :8].sum(-1))
        # print((h_hat2 * grad_h_hat2)[..., 8:].sum(-1))

        grad_x_1 = RMSNormGrad(x_fp32.flatten(2), rsqrt, grad_rsqrt)
        print("grad_h_hat2",grad_h_hat2)

        grad_h_hat2 = grad_h_hat2 * rsqrt
        print("grad_h_hat2",grad_h_hat2)
        grad_x_2 = grad_h_hat2[..., None, :] @ hc_fn
        print("grad_x1",grad_x_3.view(x_shape) )
        print("grad_x2", grad_x_1.view(x_shape) )
        print("grad_x3",grad_x_2.view(x_shape))


        print("grad_x11",grad_x_3.view(x_shape) + grad_x_1.view(x_shape) + grad_x_2.view(x_shape))

        grad_x = grad_x_3.view(x_shape) + grad_x_1.view(x_shape) + grad_x_2.view(x_shape)
        # grad_x = grad_x_1.view(x_shape) + grad_x_3.view(x_shape)
        # grad_x = grad_x_3.view(x_shape)
        print("grad_x",grad_x)
        print("grad_x11",grad_x_3.view(x_shape) + grad_x_1.view(x_shape) + grad_x_2.view(x_shape))
        print("grad_x",grad_x)

        grad_hc_fn = grad_h_hat2.flatten(0, 1).T @ x_fp32.flatten(2).flatten(0, 1)
        
        debug_info = {
            "grad_x_3": grad_x_3.to(torch.bfloat16),
            "grad_x_1": grad_x_1.to(torch.bfloat16), 
            "grad_x_1 + grad_x_3": (grad_x_3.flatten() + grad_x_1.flatten()).to(torch.bfloat16),
            "grad_pre": grad_pre,
            "grad_h_hat2_pre": grad_h_hat2_pre,
            "grad_h_hat2_post": grad_h_hat2_post
        }
        print("grad_x",grad_x)
        print("grad_hc_fn",grad_hc_fn)

        print("grad_hc_scale",grad_hc_scale)
        print("grad_hc_base",grad_hc_base)
        return grad_x.to(orig_dtype), grad_hc_fn, grad_hc_scale, grad_hc_base

@register("function_pyaclnn_mhc_pre_sinkhorn_backward")
class PyAclnnMhcPreSinkhornBackward(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        torch.npu.synchronize()
        
        input_args, output_packages = super().init_by_input_data(input_data)
        print("input_args")
        print("output_packages")

        # print(output_packages)
        # self.input_args, tmp = super().init_by_input_data(input_data)
        # self_tensor = self.acl_tensor_to_torch(self.input_args[0])
        # self.self_shape = self_tensor.size()
        
        # for i, output_data in enumerate(self.task_result.output_info_list):
        #     dtype = output_data.dtype
        #     torch_dtype = getattr(torch, dtype.replace("torch.", ""))
            
        #     if output_data.stride:
        #         empty_tensor = torch.empty_strided(output_data.shape, output_data.stride,
        #                                             dtype=torch_dtype, device='npu')
        #     else:
        #         empty_tensor = torch.empty(output_data.shape, dtype=torch_dtype, device='npu')
            
        #     output_packages[i] = self.torch_tensor_to_acl_tensor(empty_tensor)
        
        # input_args[-len(output_packages):] = output_packages
        torch.npu.synchronize()
        
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for i, output_pack in enumerate(output_packages):
            output.append(self.acl_tensor_to_torch(output_pack))
        # print(output[2][:256,0,0],"outputssssss")
        print(output,"outputssssss")

        return output

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize("
            "const aclTensor *gradHin, "
            "const aclTensor *gradHPost, "
            "const aclTensor *gradHRes, "
            "const aclTensor *x, "
            "const aclTensor *phi, "
            "const aclTensor *alpha, "
            "const aclTensor *bias, "
            "const aclTensor *hPre, "
            "const aclTensor *hcBeforeNorm, "
            "const aclTensor *invRms, "
            "const aclTensor *sumOut, "
            "const aclTensor *normOut, "
            "double hcEps, "
            "aclTensor *gradX, "
            "aclTensor *gradPhi, "
            "aclTensor *gradAlpha, "
            "aclTensor *gradBias, "
            "uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )

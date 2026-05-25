import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import random
import ctypes
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from typing import Optional
import numpy as np

@register("ascend_aclnn_mhc_post_backward")
class FunctionMhcPostBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionMhcPostBackwardApi, self).__init__(task_result)

    def vec_matmul_mknk(self, A, B_T):
        # print("A.shape : ",A.shape)
        bs, m, k = A.shape
        bs, n, k_b = B_T.shape
        
        assert k == k_b, f"A 的列数 {k} 必须等于 B 的行数 {k_b}"
        
        alignN = n
        step=1024
        C = torch.zeros((bs, m, alignN), dtype = A.dtype, device = A.device)
        broadcastBuffer = torch.zeros((bs, n, k), dtype = A.dtype, device = A.device)
        tmp = torch.zeros((bs, n), dtype = A.dtype, device = A.device)
        for i in range(0,m):
            broadcastBuffer[:,0] = A[:, i] * B_T[:,0]
            broadcastBuffer[:,1] = A[:, i] * B_T[:,1]
            broadcastBuffer[:,2] = A[:, i] * B_T[:,2]
            broadcastBuffer[:,3] = A[:, i] * B_T[:,3]
            tmp.zero_()
            for j in range(0,k):
                tmp += broadcastBuffer[:,:,j]
            C[:,i] = tmp
        return C
    def vec_matmul_mkkn(self, A, B):
        bs, m, k = A.shape
        bs, k_b, n = B.shape
        
        assert k == k_b, f"A 的列数 {k} 必须等于 B 的行数 {k_b}"
        
        C = torch.zeros((bs,m, n), dtype = A.dtype, device = A.device)
        A_flat = A.flatten(start_dim=1)  # [m*k]
        broadcastBuffer = A_flat.unsqueeze(2).expand(bs, m * k, n)
        
        # 对应 AscendC 的双重 for 循环
        for i in range(m):
            for j in range(k):
                broadcastBuffer=broadcastBuffer.clone()
                broadcastBuffer[:, i * k + j] = broadcastBuffer[:,i * k + j] * B[:,j]
                C[:,i] = C[:,i] + broadcastBuffer[:,i * k + j]
        return C

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        grad_y_dtype=input_data.kwargs['grad_y'].dtype
        # print(grad_y_dtype)
        if grad_y_dtype==torch.float32:
            dtype=torch.float64
            cast_dtype=grad_y_dtype
        else:
            dtype=torch.float32
            cast_dtype=grad_y_dtype
        grad_y=input_data.kwargs['grad_y'].to(dtype)
        grad_y_trans=grad_y.transpose(-2,-1).to(dtype)
        x=input_data.kwargs['x'].to(dtype)
        h_res=input_data.kwargs['h_res'].to(dtype)
        h_out=input_data.kwargs['h_out'].to(dtype)
        h_post=input_data.kwargs['h_post'].to(dtype)
        b_s = grad_y.shape[0]

        dimN = len(grad_y.shape)
        if dimN == 4:
            B,S,n,C =grad_y.shape
            grad_y=grad_y.reshape(grad_y.shape[0] * grad_y.shape[1], *grad_y.shape[2:]).to(dtype)
            h_out=h_out.reshape(h_out.shape[0] * h_out.shape[1], *h_out.shape[2:]).to(dtype)
            h_post=h_post.reshape(h_post.shape[0] * h_post.shape[1], *h_post.shape[2:]).to(dtype)
            x=x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]).to(dtype)
            h_res=h_res.reshape(h_res.shape[0] * h_res.shape[1], *h_res.shape[2:]).to(dtype)

        T, D = h_out.shape
        h_out = h_out.reshape(T, 1, D).to(dtype)
        T, n = h_post.shape
        h_post = h_post.reshape(T, 1, n).to(dtype)

        grad_y = grad_y.to(device)
        h_out = h_out.to(device)
        x = x.to(device)
        h_post = h_post.to(device)
        h_res = h_res.to(device)

        grad_h_post=self.vec_matmul_mknk(h_out, grad_y)
        grad_h_res=self.vec_matmul_mknk(x, grad_y)

        grad_h_out=self.vec_matmul_mkkn(h_post, grad_y)
        grad_x=self.vec_matmul_mkkn(h_res, grad_y)
        if dimN == 4:
            return grad_x.to(grad_y_dtype).reshape(B,S,n,C), grad_h_res.reshape(B,S,n,n), grad_h_out.to(grad_y_dtype).reshape(B,S,D), grad_h_post.reshape(B,S,n)
        else:
            return grad_x.to(grad_y_dtype), grad_h_res, grad_h_out.to(grad_y_dtype).reshape(T, D), grad_h_post.reshape(T, n)

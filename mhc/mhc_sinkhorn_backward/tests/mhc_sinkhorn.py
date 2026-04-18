#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# import tbetoolkits
# from .registry import register_golden
# from tbetoolkits.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor
import torch
import torch_npu

import os
import sys
import logging
import numpy as np
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass
import math
import shutil
from typing import Optional
import time

np.random.seed(1)
torch.manual_seed(1)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))




def do_op_tiling(T, n, n_align):
    DATA_BLOCK_NUM = 8
    SIZEOF_FLOAT = 4
    VF_T_FACTOR = 8

    def ceil_align(val):
        return (val + DATA_BLOCK_NUM - 1) // DATA_BLOCK_NUM * DATA_BLOCK_NUM
    def cal_occupy_size(ubfactor):
        inque_size = ceil_align(ubfactor * n * n)
        outque_size = ceil_align(ubfactor * n * n)
        norm_que_size = ceil_align(2 * ubfactor * n * n_align)
        sum_que_size = ceil_align(2 * ubfactor * n_align)
        return inque_size + outque_size + norm_que_size + sum_que_size
    
    double_buffer = 2
    total_core = 64
    aviable_ub_size = (248 * 1024 - 64 - 512) // double_buffer
    # total_core = 2 # debug用
    # aviable_ub_size = 2 * 1024 // double_buffer # debug用: 2K可以装2个
    
    t_main = (T + total_core - 1) // total_core
    used_core_num = (T + t_main - 1) // t_main
    t_tail = T - t_main * (used_core_num - 1)

    ubfactor = t_main
    occupy_size = cal_occupy_size(ubfactor) * SIZEOF_FLOAT

    print(f"occupy_size={occupy_size}")
    print(f"aviable_ub_size={aviable_ub_size}")

    if occupy_size > aviable_ub_size:
        one_pice_size = cal_occupy_size(1) * SIZEOF_FLOAT
        ubfactor = aviable_ub_size // one_pice_size
        print(f"ubfactor first={ubfactor}")
        print(f"one_pice_size={one_pice_size}")
        ubfactor = ubfactor if ubfactor // VF_T_FACTOR == 0 else ubfactor // VF_T_FACTOR * VF_T_FACTOR

    print("=======================PRINT TILING INFO==========================")
    print("t_main=tNormCore=", t_main)
    print("t_tail=tTailCore=", t_tail)
    print("used_core_num=", used_core_num)
    print("ubfactor=ubfactor=", ubfactor)
    return t_main, t_tail, used_core_num, ubfactor, n, n_align

def proc_norm_single_core(norm_cache, blockIdx, tiling_info):
    import torch as torch

    t_main, t_tail, used_core_num, ubfactor, *rest = tiling_info
    t_num = t_tail if blockIdx == (used_core_num - 1) else t_main

    # 取出每个核要处理的内存
    norm_cache_single_core = []
    for iter_norm in norm_cache:
        start = blockIdx * t_main
        norm_cache_single_core.append(iter_norm[start:start + t_num])

    # print("norm_cache_single_core=", norm_cache_single_core)
    loop = (t_num + ubfactor - 1) // ubfactor
    main_num = ubfactor * (loop - 1)
    tail_num = t_num - main_num
    
    iter_main_cache = []
    iter_tail_cache = []
    for norm in norm_cache_single_core:
        if loop > 1:
            norm_main = norm[:main_num]
            main_reshape = norm_main.view(loop - 1, ubfactor, *list(norm.shape[-2:]))
            main_loop = main_reshape.permute(0,2,1,3) # [loop - 1, 4, ubfactor, 8]
            print("=======main_loop.shape", main_loop.shape)
            iter_main_cache.append(main_loop)

        norm_tail = norm[main_num:]
        tail_reshape = norm_tail.view(1, tail_num, *list(norm.shape[-2:]))
        tail_loop = tail_reshape.permute(0,2,1,3) # [1, 4, ubfactor, 8]
        print("=======tail_loop.shape", tail_loop.shape)
        # print("tail_loop.size=", tail_loop.size())
        iter_tail_cache.append(tail_loop)

    if loop > 1:
        main_layout = torch.cat(iter_main_cache, dim=1)
        tail_layout = torch.cat(iter_tail_cache, dim=1)
        print("main_layout.shape", main_layout.shape)
        print("tail_layout.shape", tail_layout.shape)
        main_layout = main_layout.reshape(-1, main_layout.shape[-1])
        tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
        norm_out_single_core = torch.cat((main_layout, tail_layout), dim=0)
    else:
        tail_layout = torch.cat(iter_tail_cache, dim=1)
        print("tail_layout.shape", tail_layout.shape)
        tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
        norm_out_single_core = tail_layout

    return norm_out_single_core



def proc_sum_single_core(sum_col_cache, sum_row_cache, blockIdx, tiling_info):
    import torch as torch
    import torch.nn.functional as F

    t_main, t_tail, used_core_num, ubfactor, n_, n_align = tiling_info
    t_num = t_tail if blockIdx == (used_core_num - 1) else t_main
    # n_ = 4
    # n_align = 8

    # 取出每个核要处理的内存
    sum_col_cache_single_core = []
    sum_row_cache_single_core = []
    for iter_sum in sum_col_cache:
        start = blockIdx * t_main
        sum_col_cache_single_core.append(iter_sum[start:start + t_num]) # col尾轴已经按8对齐pad
    for iter_sum in sum_row_cache:
        start = blockIdx * t_main
        sum_row_cache_single_core.append(iter_sum[start:start + t_num])

    loop = (t_num + ubfactor - 1) // ubfactor
    main_num = ubfactor * (loop - 1)
    tail_num = t_num - main_num
    
    print("t_num=", t_num)
    print(f"blockIdx={blockIdx}, loop={loop}, main_num={main_num}, tail_num={tail_num}")

    iter_main_cache = []
    iter_tail_cache = []
    for i in range(len(sum_col_cache_single_core)):
        sum_col = sum_col_cache_single_core[i]
        sum_row = sum_row_cache_single_core[i]

        if loop > 1:
            # mainloop处理
            sum_col_main = sum_col[:main_num]
            sum_row_main = sum_row[:main_num]
            col_main_reshape = sum_col_main.view(loop - 1, ubfactor, n_align) # [1, 2, 8]
            row_main_reshape = sum_row_main.view(loop - 1, ubfactor, n_).permute(0,2,1) # [1, 4, 2]
            # 在1轴上n0*t_对齐为n0_align*t_
            row_main_reshape = F.pad(row_main_reshape, (0, 0, 0, n_align - n_), mode='constant', value=1) # [1, 8, 2]
            # reshape之后cat
            col_ = col_main_reshape.reshape(loop - 1, -1) # [2, 2 * 8]
            row_ = row_main_reshape.reshape(loop - 1, -1) # [2, 8 * 2]
            main_loop = torch.cat((row_, col_), dim=1) # [2, 32]
            print(f"blockIdx={blockIdx}, main_loop.size()={main_loop.size()}")
            iter_main_cache.append(main_loop)
            
        # tailloop处理
        sum_col_tail = sum_col[main_num:]
        sum_row_tail = sum_row[main_num:]
        # print(f"blockIdx={blockIdx}, sum_col_tail.size()={sum_col_tail.size()}")
        # print(f"blockIdx={blockIdx}, sum_row_tail.size()={sum_row_tail.size()}")
        col_tail_reshape = sum_col_tail.view(1, tail_num, n_align) # [1, 1, 8]
        row_tail_reshape = sum_row_tail.view(1, tail_num, n_).permute(0,2,1) # [1, 4, 1]
        row_tail_reshape = F.pad(row_tail_reshape, (0, 0, 0, n_align - n_), mode='constant', value=1) # [1, 8, 1]
        # reshape之后cat
        col_ = col_tail_reshape.reshape(1, -1) # [1, 1 * 8]
        row_ = row_tail_reshape.reshape(1, -1) # [1, 8 * 1]
        tail_loop = torch.cat((row_, col_), dim=1) # [2, 8]
        print(f"blockIdx={blockIdx}, tail_loop.size()={tail_loop.size()}")
        iter_tail_cache.append(tail_loop)
    
    if loop > 1:
        main_layout = torch.cat(iter_main_cache, dim=1) # [40, 16]
        # print(f"blockIdx={blockIdx},  main_layout.size()={ main_layout.size()}")
        tail_layout = torch.cat(iter_tail_cache, dim=1) # [40, 8]
        # print(f"blockIdx={blockIdx}, tail_layout.size()={tail_layout.size()}")
        # main_layout = main_layout.reshape(-1, main_layout.shape[-1]) # [40, 16]
        main_layout = main_layout.reshape(-1)
        print(f"blockIdx={blockIdx}, main_layou.size()={main_layout.size()}")
        # tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1]) # [40, 8]
        tail_layout = tail_layout.reshape(-1)
        print(f"blockIdx={blockIdx}, tail_layout.size()={tail_layout.size()}")
        sum_out = torch.cat((main_layout, tail_layout), dim=0)
        print(f"blockIdx={blockIdx}, sum_out.size()={sum_out.size()}")
    else:
        tail_layout = torch.cat(iter_tail_cache, dim=1)
        tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
        sum_out = tail_layout

    return sum_out.reshape(-1)



def proc_norm(norm_cache, sum_col_cache, sum_row_cache, tiling_info):
    import torch as torch

    t_main, t_tail, used_core_num, ubfactor, *rest = tiling_info

    norm_out_cache = []
    sum_out_cache = []
    # print(f"used_core_num={used_core_num}, t_tail={t_tail}, t_main={t_main}, ubfactor={ubfactor}")

    for blockIdx in range(used_core_num):
        norm_on_core = proc_norm_single_core(norm_cache, blockIdx, tiling_info)
        sum_on_core = proc_sum_single_core(sum_col_cache, sum_row_cache, blockIdx, tiling_info)
        # norm_on_core = proc_norm_single_core_custom(norm_cache, blockIdx, tiling_info)
        # sum_on_core = proc_sum_single_core_custom(sum_col_cache, sum_row_cache, blockIdx, tiling_info)
        print("blockIdx====", blockIdx)
        print("norm_on_core====", norm_on_core.size())
        print("sum_on_core====", sum_on_core.size())

        norm_out_cache.append(norm_on_core)
        sum_out_cache.append(sum_on_core)

    norm_out = torch.cat(norm_out_cache, dim=0)
    sum_out = torch.cat(sum_out_cache, dim=0)

    return norm_out, sum_out

def softmax(x, dim=-1):
    """计算softmax"""
    return torch.softmax(x, dim=dim)


def sinkhorn_forward_simple(x, num_iters, eps):
    """
    Sinkhorn前向传播（简化版）

    参数:
        x: 输入张量 [T, n, n]
        num_iters: 迭代次数
        eps: 小常数防止除零

    返回:
        output: 输出 [T, n, n]
        norm_list: 归一化输出列表
        sum_list: 求和结果列表
    """
    norm_list = []
    sum_list = []

    # Step 0: Softmax
    prob = softmax(x, dim=-1)
    curr = prob + eps
    norm_list.append(prob.clone())
    sum_list.append(None)

    # Step 1: Initial Col Norm
    col_sum = curr.sum(dim=-2, keepdim=True) + eps
    curr = curr / col_sum
    norm_list.append(curr.clone())
    sum_list.append(col_sum.clone())

    # Step 2: Loop
    for i in range(num_iters - 1):
        # Row Norm
        row_sum = curr.sum(dim=-1, keepdim=True) + eps
        curr = curr / row_sum
        norm_list.append(curr.clone())
        sum_list.append(row_sum.clone())

        # Col Norm
        col_sum = curr.sum(dim=-2, keepdim=True) + eps
        curr = curr / col_sum
        norm_list.append(curr.clone())
        sum_list.append(col_sum.clone())

    return curr, norm_list, sum_list

def sinkhorn_backward_golden(grad_output, norm_out, sum_out, num_iters):
    # 将grad_output从[T, n, n]转换为[n, n, T]以匹配norm_out的格式
    grad_curr = grad_output.permute(1, 2, 0)  # [T, n, n] -> [n, n, T]
    # 共进行num_iters次迭代
    for i in range(num_iters - 1, 0, -1):
        # 列归一化的反向传播
        col_idx = 2 * i + 1
        # grad_curr: [n, n, T], norm_out[col_idx]: [n, n, T]
        # 列归一化是在axis=0方向求和（第一个n维度）
        dot_prod = torch.sum(grad_curr * norm_out[col_idx], dim=0, keepdim=True)  # [1, n, T]
        # sum_out[col_idx]: [n, T], 需要扩展为 [1, n, T]
        grad_curr = (grad_curr - dot_prod) / sum_out[col_idx].unsqueeze(0)

        # 行归一化的反向传播
        row_idx = 2 * i
        # grad_curr: [n, n, T], norm_out[row_idx]: [n, n, T]
        # 行归一化是在axis=1方向求和（第二个n维度）
        dot_prod = torch.sum(grad_curr * norm_out[row_idx], dim=1, keepdim=True)  # [n, 1, T]
        grad_curr = (grad_curr - dot_prod) / sum_out[row_idx].unsqueeze(1)


    # 最后一次迭代：列归一化
    print("最后一次迭代：列归一化")
    dot_prod = torch.sum(grad_curr * norm_out[1], dim=0, keepdim=True)  # [1, n, T]
    print("norm_out[1]: ", norm_out[1])
    print("dot_prod: ", dot_prod)
    grad_curr = (grad_curr - dot_prod) / sum_out[1].unsqueeze(0)
    print("sum_out[1]: ", sum_out[1])
    print("grad_curr: ", grad_curr)
    
    print("最后一次迭代: 行softmax")
    dot_prod = torch.sum(grad_curr * norm_out[0], dim=1, keepdim=True)  # [n, 1, T]
    print("norm_out[0]: ", norm_out[0])
    print("dot_prod: ", dot_prod)
    grad_input = norm_out[0] * (grad_curr - dot_prod)  # [n, n, T]
    print("grad_input: ", grad_input)
    # print(f"soft max")
    # print(grad_curr)
    # 将结果从[n, n, T]转回[T, n, n]
    grad_input = grad_input.permute(2, 0, 1).contiguous()  # [n, n, T] -> [T, n, n]
    print("grad_input: ", grad_input)

    return grad_input


def convert_to_npu_format(norm_list, sum_list, num_iters):
    """
    将列表格式的中间结果转换为NPU格式

    参数:
        norm_list: 归一化输出列表，每个元素shape为[T, n, n]
        sum_list: 求和结果列表，每个元素shape为[T, 1, n]或[T, n, 1]
        num_iters: 迭代次数

    返回:
        norm_out: [2*num_iters, n, n, T]
        sum_out: [2*num_iters, n, T]
    """
    T, n, _ = norm_list[0].shape

    # 构建norm_out: [2*num_iters, n, n, T]
    norm_out = torch.zeros(2 * num_iters, n, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, norm_tensor in enumerate(norm_list):
        # [T, n, n] -> [n, n, T]
        norm_out[idx] = norm_tensor.permute(1, 2, 0)

    # 构建sum_out: [2*num_iters, n, T]
    sum_out = torch.zeros(2 * num_iters, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, sum_tensor in enumerate(sum_list):
        if sum_tensor is None:
            continue
        # [T, 1, n] or [T, n, 1] -> [n, T]
        if sum_tensor.shape[-2] == 1:  # col_sum: [T, 1, n]
            sum_out[idx] = sum_tensor.squeeze(-2).T
        else:  # row_sum: [T, n, 1]
            sum_out[idx] = sum_tensor.squeeze(-1).T

    return norm_out, sum_out

# @register_golden(["mhc_sinkhorn"])

def mhc_sinkhorn(debug=False):
    import torch as torch
    import torch.nn.functional as F
    import numpy as np
    print("hhhh")
    # x = context.input_arrays[0]
    # # tensor_x = numpy_to_torch_tensor(x).clone()
    # tensor_x = x.clone().form_numpy()
    T = 20000
    n0, n1 = 4, 4
    tensor_x = torch.randn(T, n0, n1, dtype=torch.float32)
    x = tensor_x.numpy()
    # n0, n1 = x.shape[-2:]
    # T = torch.numel(tensor_x) // (n0 * n1)

    ## 获取属性
    # eps = context.other_compilation_params['eps']
    eps = 1e-6
    eps = float(eps)
    # num_iters = context.other_compilation_params['num_iters']
    num_iters = 20
    # out_flag = context.other_compilation_params['out_flag']
    out_flag = 1

    if n0 != n1:
        raise RuntimeError(f"The last two dimensions of `x` must be the same. Got: {n0} x {n1}")

    if n0 not in [4, 6, 8]:
        raise RuntimeError(f"n0 must be 4, 6, or 8, but got {n0}")

    if "float32" not in str(x.dtype):
        raise RuntimeError(f"xDtype must be float32, got {x.dtype}")

    if num_iters <= 0 or num_iters > 100:
        raise RuntimeError(f"num_iters must be greater than 0 and less than or equal to 100, got {num_iters}")

    n_align = 8
    def pad_norm(norm):
        norm_pad = F.pad(norm, (0, n_align - n1), mode='constant', value=0)
        norm_reshape = norm_pad.reshape(-1, *norm_pad.shape[-2:])
        return norm_reshape

    def pad_sum(sum_col):
        sum_pad = F.pad(sum_col, (0, n_align - n1), mode='constant', value=0)
        sum_reshape = sum_pad.reshape(-1, *sum_pad.shape[-1:])
        return sum_reshape
    
    tiling_info = []
    if out_flag != 0:
        tiling_info = do_op_tiling(T, n0, n_align)

    norm_cache = []
    sum_col_cache = []
    sum_row_cache = []
    # Sinkhorn-Knopp算法功能
    prob = torch.softmax(tensor_x, dim=-1)
    print("prob: ", prob.shape)
    h_comb_softmax = prob + eps
    print("h_comb_softmax: ", prob.shape)
    sum_col = h_comb_softmax.sum(dim=-2, keepdim=True) + eps
    print("sum_col: ", sum_col.shape)
    h_comb = h_comb_softmax / sum_col
    print("h_comb: ", prob.shape)
    
    norm_cache.append(pad_norm(prob))
    print("pad_norm(prob): ", pad_norm(prob).shape)
    norm_cache.append(pad_norm(h_comb))
    print("pad_norm(h_comb): ", pad_norm(h_comb).shape)
    sum_row_cache.append(torch.ones(T, n0))
    sum_col_cache.append(pad_sum(sum_col))
    print("pad_sum(sum_col): ", pad_sum(sum_col).shape)
    
    for i in range(max(num_iters - 1, 0)):
        # Row Norm
        sum_row = (h_comb.sum(dim=-1, keepdim=True) + eps)
        h_comb = h_comb / sum_row
        norm_cache.append(pad_norm(h_comb))
        sum_row_cache.append(sum_row.reshape(-1, *sum_row.shape[-2:]))
        print("sum_row.reshape(-1, *sum_row.shape[-2:]): ", sum_row.reshape(-1, *sum_row.shape[-2:]).shape)

        # Col Norm
        sum_col = (h_comb.sum(dim=-2, keepdim=True) + eps)
        h_comb = h_comb / sum_col
        norm_cache.append(pad_norm(h_comb))
        sum_col_cache.append(pad_sum(sum_col))
        print("pad_sum(sum_col): ", pad_sum(sum_col).shape)

    if out_flag != 0:
        if debug:
            norm_cache = []
            sum_col_cache = []
            sum_row_cache = []
            for i in range(max(num_iters, 0)):
                norm_cache.append(pad_norm(torch.arange(2*i * T * n0 * n0, (2*i+1) * T * n0 * n0, 1, dtype=torch.int64).reshape(T, n0, n0)))
                sum_col_cache.append(pad_sum(torch.arange(i * T * n0, (i+1) * T * n0, 1, dtype=torch.int64).reshape(T, n0)))
                norm_cache.append(pad_norm(torch.arange((2*i+1) * T * n0 * n0, (2*i+2) * T * n0 * n0, 1, dtype=torch.int64).reshape(T, n0, n0)))
                sum_row_cache.append(torch.arange(10000 + i * T * n0, 10000 + (i+1) * T * n0, 1, dtype=torch.int64).reshape(T, n0))
                
        norm_out, sum_out = proc_norm(norm_cache, sum_col_cache, sum_row_cache, tiling_info)
        norm_out = norm_out.reshape(-1)
        sum_out = sum_out.reshape(-1)

        if debug:
            torch.set_printoptions(threshold=float('inf'))
            torch.set_printoptions(linewidth=200)
            print("norm_out.shape ==", norm_out.shape)
            print("norm_out ==", norm_out)
            torch.set_printoptions(linewidth=235)
            print("sum_out.shape ==", sum_out.shape)
            print("sum_out ==", sum_out)


        print("norm_out====", norm_out.size())
        print("sum_out====", sum_out.size())

        # return torch_to_numpy_tensor(h_comb), torch_to_numpy_tensor(norm_out), torch_to_numpy_tensor(sum_out)
    # grad_input = torch.ops.custom.npu_sinkhorn_grad(h_comb.npu(), norm_out.npu(), sum_out.npu())
    
    if not debug:
        y, norm_list_golden, sum_list_golden = sinkhorn_forward_simple(tensor_x, num_iters, eps)
        # 构造输出梯度（假设loss = sum(y^2)）
        grad_output = 2 * y
        norm_out_golden, sum_out_golden = convert_to_npu_format(norm_list_golden, sum_list_golden, num_iters)
        grad_input_golden = sinkhorn_backward_golden(grad_output, norm_out_golden, sum_out_golden, num_iters)

        print("===================PRINT GOLDEN INPUT=============================")
        print("norm_out_golden.shape: ", norm_out_golden.shape) # 16, 1, 40, 4, 128, 8
        print("norm_out_golden: ", norm_out_golden) 
        print("=======================================================")
        print("sum_out_golden.shape: ", sum_out_golden.shape) # 16, 1, 40, 8*128
        print("sum_out_golden: ", sum_out_golden) 
        print("=======================================================")
        print("grad_output.shape: ", grad_output.shape)
        print("grad_output.shape: ", grad_output)
        print("===================PRINT GOLDEN OUTPUT==============================")
        print("grad_input_golden.shape: ", grad_input_golden.shape)
        print("grad_input_golden: ", grad_input_golden)


        print("===================PRINT NPU INPUT=============================")
        print("norm_out.shape: ", norm_out.shape)
        print("snorm_out: ", norm_out) 
        print("=======================================================")
        print("sum_out.shape: ", sum_out.shape)
        print("sum_out: ", sum_out) 
        torch.npu.synchronize()
        grad_input = torch_npu.npu_mhc_sinkhorn_backward(grad_output.npu(), norm_out.npu(), sum_out.npu())
        torch.npu.synchronize()
        # time.sleep(15)
        print("===================PRINT NPU OUTPUT==============================")
        print("grad_input.shape: ", grad_input.shape)
        print("grad_input: ", grad_input)
        grad_input = grad_input.cpu()
        
        # save tensor
        data_dir = os.path.join(WORKSPACE, "data")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        norm_out_golden.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "norm_out_cpu.bin"))
        sum_out_golden.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "sum_out_cpu.bin"))
        grad_output.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "grad_output.bin"))
        grad_input_golden.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "grad_input_cpu.bin"))

        norm_out.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "norm_out_npu.bin"))
        sum_out.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "sum_out_npu.bin"))
        grad_input.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "grad_input_npu.bin"))



if __name__ == "__main__":
    debug = int(sys.argv[1])
    if debug == 1:
        debug_flag = True
    else:
        debug_flag = False
    mhc_sinkhorn(debug_flag)

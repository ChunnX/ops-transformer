

def do_op_tiling(T, n, n_align):
    def set_ubblock_align(quesize):
        return (quesize + 31) // 32
    def cal_occupy_size(ubfactor):
        inque_size = ubfactor * n * n 
        outque_size= ubfactor * n * n 
        norm_que_size = ubfactor * n * n_align
        sum_que_size = ubfactor * n_align
        return inque_size + outque_size + norm_que_size + sum_que_size

    total_core = 64
    double_buffer = 2
    aviable_ub_size = (256 * 1024 - 8 * 1024 - 64 - 512) // double_buffer

    t_main = (T + total_core - 1) // total_core
    used_core_num = (T + t_main - 1) // t_main
    t_tail = T - t_main * (used_core_num - 1)

    ubfactor = t_main
    occupy_size = cal_occupy_size(t_main) * 4
    if occupy_size > aviable_ub_size:
        one_pice_size = cal_occupy_size(1) * 4
        ubfactor = aviable_ub_size // one_pice_size
    else:
        ubfactor = t_main

    return t_main, t_tail, used_core_num, ubfactor, n, n_align

def proc_norm_single_core(norm_cache, blockIdx, tiling_info):
    import torch as torch

    t_main, t_tail, used_core_num, ubfactor, *rest = tiling_info
    t_num = t_tail if blockIdx == (used_core_num - 1) else t_main

    norm_cache_single_core = []
    for iter_norm in norm_cache:
        start = blockIdx * t_main
        norm_cache_single_core.append(iter_norm[start:start + t_num])

    loop = (t_num + ubfactor - 1) // ubfactor
    main_num = ubfactor * (loop - 1)
    tail_num = t_num - main_num

    iter_main_cache = []
    iter_tail_cache = []
    for norm in norm_cache_single_core:
        if loop > 1:
            norm_main = norm[:main_num]
            main_reshape = norm_main.view(loop - 1, ubfactor, *list(norm.shape(-2:)))
            main_loop = main_reshape.permute(0, 2, 1, 3)
            iter_main_cache.append(main_loop)

        norm_tail = norm[main_num:]

        tail_reshape = norm_tail.view(1, tail_num, *list(norm.shape[-2:]))
        tail_loop = tail_reshape.permute(0, 2, 1, 3)

        iter_tail_cache.append(tail_loop)

    if loop > 1:
        main_layout = torch.cat(iter_main_cache, dim=1)
        tail_layout = torch.cat(iter_tail_cache, dim=1)
        main_layout = main_layout.reshape(-1, main_layout.shape[-1])
        tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
        norm_out_single_core = torch.cat((main_layout, tail_layout), dim=0)
    else:
        tail_layout = torch.cat(iter_tail_cache, dim=1)
        tail_layout = tail_layout.rehsape(-1, tail_layout.shape[-1])
        norm_out_single_core = tail_layout

    return norm_out_single_core

def proc_sum_single_core(sum_col_cache, sum_row_cache, blockIdx, tiling_info):
    import torch as torch
    import torch.nn.functional as F 

    t_main, t_tail, used_core_num, ubfactor, n_, n_align = tiling_info
    t_num = t_tail if blockIdx == (used_core_num - 1) else t_main

    sum_col_cache_single_core = []
    sum_row_cache_single_core = []
    for iter_sum in sum_col_cache:
        start = blockIdx * t_main
        sum_col_cache_single_core.append(iter_sum[start:start + t_num])
    for iter_sum in sum_row_cache:
        start = blockIdx * t_main
        sum_row_cache_single_core.append(iter_sum[start:start + t_num])

    loop = (t_num + ubfactor - 1) // ubfactor
    main_num = ubfactor * (loop - 1)
    tail_num = t_num - main_num

    iter_main_cache = []
    iter_tail_cache = []
    for i in range(len(sum_col_cache_single_core)):
        sum_col = sum_col_cache_single_core[i]
        sum_row = sum_row_cache_single_core[i]

        if loop > 1:
            sum_col_main = sum_col[:main_num]
            sum_row_main = sum_row[:main_num]
            col_main_reshape = sum_col_main.view(loop - 1, ubfactor, n_align)
            row_main_reshape = sum_row_main.view(loop - 1, ubfactor, n_).permute(0, 2, 1)

            row_main_reshape = F.pad(row_main_reshape, (0, 0, 0, n_align - n), mode='constant', value = 1)
            col_ = col_main_reshape.reshape(loop - 1, -1)
            row_ = row_main_reshape.reshape(loop - 1, -1)
            main_loop = torch.cat((row_, col_), dim=0)
            iter_main_cache.append(main_loop)

        sum_col_tail = sum_col[main_num:]
        sum_row_tail = sum_row[main_num:]

        col_tail_reshape = sum_col_tail.view(1, tail_num, n_align)
        row_tail_reshape = sum_row_tail.view(1, tail_num, n_).permute(0, 2, 1)
        row_tail_reshape = F.pad(row_tail_reshape, (0, 0, 0, n_align - n_), mode='constant', value=1)

        col_ = col_tail_reshape.reshape(1, -1)
        row_ = row_tail_reshape.reshape(1, -1)
        tail_loop = torch.cat((row_, col_), dim=0)
        iter_tail_cache.append(tail_loop)

    if loop > 1:
        main_layout = torch.cat(iter_main_cache, dim=0)
        tail_layout = torch.cat(iter_tail_cache, dim=0)
        main_layout = main_layout.reshape(-1, main_layout.shape[-1])
        tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
        sum_out = torch.cat((main_layout, tail_layout), dim=0)
    else:
        tail_layout = torch.cat(iter_tail_cache, dim=0)
        tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
        sum_out = tail_layout

    return sum_out.reshape(-1)

def proc_norm(norm_cache, sum_col_cache, sum_row_cache, tiling_info):
    import torch as torch
    t_main, t_tail, used_core_num, ubfactor, *rest = tiling_info

    norm_out_cache = []
    sum_out_cache = []

    for blockIdx in range(used_core_num):
        norm_on_core = proc_norm_single_core(norm_cache, blockIdx, tiling_info)
        sum_on_core = proc_sum_single_core(sum_col_cache, sum_row_cache, blockIdx, tiling_info)

        norm_out_cache.append(norm_on_core)
        sum_out_cache.append(sum_on_core)
    
    norm_out = torch.cat(norm_out_cache, dim=0)
    sum_out = torch.cat(sum_out_cache, dim=0)

    return norm_out, sum_out 

def mhc_sinkhorn():
    import torch 
    import torch.nn.functional as F 
    import numpy as np 
    T, n0, n1=128, 4, 4
    tensor_x = torch.randn(T, n0, n1, dtype=torch.float32)
    x = tensor_x.numpy()

    eps = 1e-6
    num_iters = 1
    out_flag = 1
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
    prob = torch.softmax(tensor_x, dim=-1)
    h_comb_softmax = prob + eps
    sum_col = h_comb_softmax.sum(dim=-2, keepdim=True) + eps
    h_comb = h_comb_softmax / sum_col

    norm_cache.append(pad_norm(prob))
    norm_cache.append(pad_norm(h_comb))
    sum_row_cache.append(torch.ones(T, n0))
    sum_col_cache.append(pad_sum(sum_col))

    for i in range(max(num_iters - 1, 0)):
        sum_row = (h_comb.sum(dim=-1, keepdim=True) + eps)
        h_comb = h_comb / sum_row
        norm_cache.append(pad_norm(h_comb))
        sum_row_cache.append(sum_row.reshape(-1, *sum_row.shape[-2:]))

        sum_col = (h_comb.sum(dim=-2, keepdim=True) + eps)
        h_comb = h_comb / sum_col
        norm_cache.append(pad_norm(h_comb))
        sum_col_cache.append(pad_sum(sum_col))

    if out_flag != 0:
        norm_out, sum_out = proc_norm(norm_cache, sum_col_cache, sum_row_cache, tiling_info)
        norm_out = norm_out.reshape(2*num_iters*T, n0, n_align)
        sum_out = sum_out.reshape(2 * num_iters * T, n_align)

if __name__ == "__main__":
    mhc_sinkhorn()
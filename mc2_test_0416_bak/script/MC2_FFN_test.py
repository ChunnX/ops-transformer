import datetime
import os
import random
import argparse
import time
import os
from multiprocessing import Process, Manager
import socket
import hashlib
from functools import reduce
import torch_npu
import numpy as np
import torch
import torch.distributed as dist
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
import torchair._contrib.custom_torch_ops


def gen_unique_topk_array(low, high, bs, k):
    array = []
    for i in range(bs):
        top_idx = list(np.arange(low, high, dtype=np.int32))
        random.shuffle(top_idx)
        array.append(top_idx[0:k])
    return np.array(array)


def gen_x(shape):
    array = []
    for i in range(shape[0]):
        array.append([i] * shape[1])
    return np.array(array)


def get_new_group_npu(rank):
    # tp_world_size = 2
    # ep_world_size = 8
    for i in range(tp_world_size):
        # [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
        ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_t = ep_group
            # print(f"rank:{rank} ep_ranks:{ep_ranks}")
    for i in range(ep_world_size):
        # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
        tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_t = tp_group
            # print(f"rank:{rank} tp_ranks:{tp_ranks}")
    return ep_group_t, tp_group_t

def get_new_group_cpu(rank):
    # tp_world_size = 2
    # ep_world_size = 8
    for i in range(tp_world_size):
        # [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
        ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
        ep_group = dist.new_group(backend="gloo", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_t = ep_group
            # print(f"rank:{rank} ep_ranks:{ep_ranks}")
    for i in range(ep_world_size):
        # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
        tp_group = dist.new_group(backend="gloo", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_t = tp_group
            # print(f"rank:{rank} tp_ranks:{tp_ranks}")
    return ep_group_t, tp_group_t

def cal_quant(y, scales):
    if scales:
        y_tmp = y.to(torch.float32) * scales
    else:
        y_tmp = y.to(torch.float32)
    # print(y_tmp.shape)
    y_max, _ = torch.max(torch.abs(y_tmp), dim=0)
    dynamic_scales = 127.0 / y_max
    if quant_mode == 2:
        y_tmp = y_tmp * dynamic_scales
    return torch.round(y_tmp).to(torch.int8), 1/dynamic_scales

def get_hcomm_info(rank, comm_group):
    if torch.__version__ > '2.0.1':
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info

def compare_data(x, y, err_thd, err_pct):
    if len(x) != len(y):
        print("ERROR: golden and npu output num not equal, golden output num %s, npu output num %s" %(len(x), len(y)))
        return False
    precision_status = "PASS"
    for i in range(len(x)):
        if x[i].shape != y[i].shape:
            print("ERROR: golden and npu output shape not equal, output index %s, golden output shape %s, npu output shape %s" %(i, x[i].shape, y[i].shape))
            return False
        size = reduce(lambda x, y: x * y, x[i].shape)
        diff_results = torch.isclose(x[i], y[i], rtol=err_thd, atol=1e-8, equal_nan=True)
        count = torch.sum(diff_results)
        se = (size - count) / size
        if se > err_pct:
            print("ERROR: npu output is compared failed with golden, output index %s" % i)
            x[i] = torch.reshape(x[i], (-1,))
            y[i] = torch.reshape(y[i], (-1,))
            print("element_index-------------golden-------------npu_output")
            for j in range(min(len(x[i]), 10)):
                print("%s-------------%s-------------%s" %(j, x[i][j], y[i][j]))
            precision_status = "Failed"
    return precision_status

class MC2_FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, rank, x, expert_ids, ep_group_hcomm_info, tp_group_hcomm_info, scale_dispatch, expert_scale, 
        weight_gmm1, scale_swiglu, group_tensor, output_dtype, bias, quant_scale, quant_offset, group_index, weight_gmm2, scale_gmm2):
        output = torch_npu.npu_moe_distribute_dispatch(x=x, expert_ids=expert_ids, group_ep=ep_group_hcomm_info,
                                                    group_tp=tp_group_hcomm_info,
                                                    ep_world_size=ep_world_size, tp_world_size=tp_world_size,
                                                    ep_rank_id=rank // tp_world_size, tp_rank_id=rank % tp_world_size,
                                                    expert_shard_type=0, shared_expert_rank_num=shared_expert_rank_num,
                                                    moe_expert_num=moe_expert_num, scales=scale_dispatch, quant_mode=quant_mode, global_bs=0)
        expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = output
        dynamic_scales = torch.reciprocal(dynamic_scales)
        pertoken_scale = torch.mul(dynamic_scales, 0.0000005)
        #GMM1_OUT = [M,N] INT32,输出的是Tensor_List
        GMM1_OUT = torch_npu.npu_grouped_matmul([expand_x], weight_gmm1, scale=None, bias=bias,
                                                group_list=group_tensor,
                                                per_token_scale=None,
                                                antiquant_scale=None, antiquant_offset=None,
                                                act_type=0, split_item=2, group_type=0,
                                                output_dtype=torch.int32)

        # y = [M, N//2]int8, gmm2_pertoken_scale = [M]
        y, gmm2_pertoken_scale = torch.ops.npu_inference.npu_dequant_swiglu_quant(GMM1_OUT[0], scale_swiglu[0], pertoken_scale, bias,
                                                                                    quant_scale, quant_offset, group_index,
                                                                                    activate_left=False, quant_mode=1)

        #OUT = [M, K]
        GMM2_OUT = torch_npu.npu_grouped_matmul([y], weight_gmm2, scale=scale_gmm2, bias=None,
                                                group_list=group_tensor,
                                                per_token_scale=[gmm2_pertoken_scale],
                                                antiquant_scale=None, antiquant_offset=None,
                                                act_type=0, split_item=2, group_type=0,
                                                output_dtype=output_dtype)
        output_x = torch_npu.npu_moe_distribute_combine(expand_x=GMM2_OUT[0], expert_ids=expert_ids, expand_idx=expand_idx,
                                                        ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts,
                                                        expert_scales=expert_scale,
                                                        group_ep=ep_group_hcomm_info, group_tp=tp_group_hcomm_info,
                                                        ep_world_size=ep_world_size, tp_world_size=tp_world_size,
                                                        ep_rank_id=rank // tp_world_size, tp_rank_id=rank % tp_world_size,
                                                        expert_shard_type=0,
                                                        shared_expert_rank_num=shared_expert_rank_num, moe_expert_num=moe_expert_num,
                                                        global_bs=0)
        return output_x


def run_npu(rank, p_rank, x, expert_ids, scale_dispatch, expert_scale, weight_gmm1, scale_swiglu, group_tensor, output_dtype, bias, 
        quant_scale, quant_offset, group_index, weight_gmm2, scale_gmm2):
    md5_list = []
    torch_npu.npu.set_device(p_rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
    ep_group, tp_group = get_new_group_npu(rank)
    ep_group_hcomm_info = get_hcomm_info(rank, ep_group)
    tp_group_hcomm_info = get_hcomm_info(rank, tp_group)

    x = x.npu()
    expert_ids = expert_ids.npu()
    scale_dispatch = scale_dispatch.npu() if scale_dispatch else scale_dispatch
    expert_scale = expert_scale.npu()
    weight_gmm1 = weight_gmm1.npu()
    scale_swiglu = [scale_swiglu[0].npu()] if scale_swiglu else scale_swiglu
    group_tensor = group_tensor.npu()
    bias = bias.npu() if bias else bias
    quant_scale = quant_scale.npu()
    quant_offset = quant_offset.npu() if quant_offset else quant_offset
    group_index = group_index.npu()
    weight_gmm2 = weight_gmm2.npu()
    scale_gmm2 = [scale_gmm2[0].npu()] if scale_gmm2 else scale_gmm2

    model = MC2_FFN()
    model = model.npu()
    model.eval()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    if False:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.MemoryL0,
            l2_cache=False
        )
        profile_path = "./output"
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=True,  # 采集torch op的函数调用栈的开关
                record_shapes=True,  # 采集torch op的input shape和input type的开关
                profile_memory=True,  # 采集memory相关数据的开关
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),  # 导出tensorboard可呈现的数据形式
                experimental_config=experimental_config
        ) as prof:
            for _ in range(5):
                output_x = model(rank, x, expert_ids, ep_group_hcomm_info, tp_group_hcomm_info, scale_dispatch, expert_scale, 
                    [weight_gmm1], scale_swiglu, group_tensor, output_dtype, bias, quant_scale, quant_offset, group_index, [weight_gmm2], scale_gmm2)
                prof.step()
    else:
        output_x = model(rank, x, expert_ids, ep_group_hcomm_info, tp_group_hcomm_info, scale_dispatch, expert_scale, 
            [weight_gmm1], scale_swiglu, group_tensor, output_dtype, bias, quant_scale, quant_offset, group_index, [weight_gmm2], scale_gmm2)
    torch.npu.synchronize()
    return output_x.cpu()

def run_cpu(rank, p_rank, recv_rank_idx_2d_array, global_recv_rank_idx_2d_array, x, expert_ids, scale_dispatch, expert_scale, 
        weight_gmm1, scale_swiglu, group_tensor, output_dtype, bias, quant_scale, quant_offset, group_index, weight_gmm2, scale_gmm2):
    #-----------------------------------------------dispatch golden------------------------------------------------------------------------
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
    ep_group, tp_group = get_new_group_cpu(rank)

    global_expert_scale_list = []
    for _ in range(ep_world_size):
        gather_output_i = torch.zeros(size=[bs, k], dtype=torch.float32)
        global_expert_scale_list.append(gather_output_i)
    dist.all_gather(global_expert_scale_list, expert_scale, group=ep_group)

    if dtype == "bf16":
        ep_tensor = torch.tensor([]).to(torch.bfloat16)
    else:
        ep_tensor = torch.tensor([]).to(torch.float16)

    if rank // 2 < shared_expert_rank_num:
        a = global_bs // shared_expert_rank_num
    else:
        a = global_bs * local_moe_expert_num

    p2p_op_list = []
    recv_tensor_list = []

    # moe专家数据发送
    for index, recv_rank_idx_1d_array in enumerate(recv_rank_idx_2d_array):
        recv_rank_idx_1d_list = list(recv_rank_idx_1d_array)
        for recv_rank_idx in recv_rank_idx_1d_list:
            if rank != recv_rank_idx:
                send_op = dist.P2POp(dist.isend, x[index], recv_rank_idx, group=ep_group)
                p2p_op_list.append(send_op)
    # 共享专家数据发送
    if rank // 2 < shared_expert_rank_num:
        # 本卡数据copy
        pass
        # ep_tensor = torch.cat((ep_tensor, x), dim=0)
    else:
        recv_shared_expert_idx = rank // 2 % shared_expert_rank_num
        if rank % 2 == 0:
            recv_shared_expert_rank_idx = recv_shared_expert_idx * 2
        else:
            recv_shared_expert_rank_idx = recv_shared_expert_idx * 2 + 1
        # print(f"send x from {rank} to {recv_shared_expert_rank_idx}")
        send_op = dist.P2POp(dist.isend, x, recv_shared_expert_rank_idx, group=ep_group)
        p2p_op_list.append(send_op)

    ep_recv_counts = []
    ep_count = 0
    # moe专家数据接收 + 计算ep_recv_counts
    for index, global_recv_rank_idx_1d_array in enumerate(global_recv_rank_idx_2d_array):
        global_recv_rank_idx_1d_list = list(global_recv_rank_idx_1d_array)
        global_recv_rank_idx_1d_list.sort()
        for recv_rank_idx in global_recv_rank_idx_1d_list:
            if recv_rank_idx == rank:
                src_rank_idx = index // bs * 2 if rank % 2 == 0 else index // bs * 2 + 1
                if dtype == "bf16":
                    recv_tensor = torch.empty((1, h), dtype=torch.bfloat16)
                else:
                    recv_tensor = torch.empty((1, h), dtype=torch.float16)
                if rank != src_rank_idx:
                    recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank_idx, group=ep_group)
                    p2p_op_list.append(recv_op)
                else:
                    x_index = index % bs
                    # 接收本卡的数据发送
                    recv_tensor = x[x_index].reshape([1, h])
                recv_tensor_list.append(recv_tensor)
                ep_count += 1
        if index % bs == (bs - 1):
            ep_recv_counts.append(ep_count)
    # 共享专家数据接收
    if rank // 2 < shared_expert_rank_num:
        ep_tensor = torch.cat((ep_tensor, x), dim=0)
        for src_rank_idx in range(world_size):
            recv_tensor = torch.empty((bs, h), dtype=torch_dtype_map[dtype])
            if src_rank_idx // 2 >= shared_expert_rank_num:
                if rank == src_rank_idx % (shared_expert_rank_num * 2):
                    # print(f"recv x from {src_rank_idx} to {rank}")
                    recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank_idx, group=ep_group)
                    p2p_op_list.append(recv_op)
                    recv_tensor_list.append(recv_tensor)

    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()
    for recv_tensor in recv_tensor_list:
        ep_tensor = torch.cat((ep_tensor, recv_tensor), dim=0)
    ep_tensor_dim0 = ep_tensor.shape[0]
    fill_size = a - ep_tensor_dim0
    size_list[p_rank] = fill_size
    if dtype == "bf16":
        fill_tensor = torch.zeros((fill_size, h), dtype=torch.bfloat16)
    else:
        fill_tensor = torch.zeros((fill_size, h), dtype=torch.float16)
    ep_tensor = torch.cat((ep_tensor, fill_tensor), dim=0)
    # tp域内做allGather
    gather_output_tensor_list = []
    for _ in range(tp_world_size):
        if dtype == "bf16":
            gather_output_i = torch.zeros(size=[a, h], dtype=torch.bfloat16)
        else:
            gather_output_i = torch.zeros(size=[a, h], dtype=torch.float16)
        gather_output_tensor_list.append(gather_output_i)
    dist.all_gather(gather_output_tensor_list, ep_tensor, group=tp_group)
    # 集群同步
    dist.barrier()
    # 切块后重新拼接
    if rank % 2 == 0:
        ep_tensor0, fill_tensor0 = torch.split(gather_output_tensor_list[0], [a - size_list[p_rank], size_list[p_rank]], dim=0)
        ep_tensor1, fill_tensor1 = torch.split(gather_output_tensor_list[1], [a - size_list[p_rank + 1], size_list[p_rank + 1]], dim=0)
        expand_x = torch.cat((ep_tensor0, ep_tensor1, fill_tensor0, fill_tensor1), dim=0)
    else:
        ep_tensor1, fill_tensor1 = torch.split(gather_output_tensor_list[1], [a - size_list[p_rank], size_list[p_rank]], dim=0)
        ep_tensor0, fill_tensor0 = torch.split(gather_output_tensor_list[0], [a - size_list[p_rank - 1], size_list[p_rank - 1]], dim=0)
        expand_x = torch.cat((ep_tensor1, ep_tensor0, fill_tensor1, fill_tensor0), dim=0)
    dynamic_scales_list = []
    for i in range(expand_x.shape[0]):
        expand_x[i,:], dynamic_scales_tmp = cal_quant(expand_x[i,:], None)
        dynamic_scales_list.append(dynamic_scales_tmp)
    dynamic_scales = torch.tensor(dynamic_scales_list).to(torch.float32)
    expand_x = expand_x.to(torch.int8)

    # 计算expand_idx
    count_dict = {}
    expand_idx = []
    for recv_rank_idx_1d_array in recv_rank_idx_2d_array:
        for recv_rank_idx in recv_rank_idx_1d_array:
            if recv_rank_idx not in count_dict:
                count_dict[recv_rank_idx] = 0
                expand_idx.append(0)
            else:
                count_dict[recv_rank_idx] += 1
                expand_idx.append(count_dict[recv_rank_idx])
    expand_idx = torch.from_numpy(np.array(expand_idx).astype(np.int32))
    if rank % 2 == 0:
        tp_recv_counts = [a - size_list[p_rank], a - size_list[p_rank + 1]]
    else:
        tp_recv_counts = [a - size_list[p_rank - 1], a - size_list[p_rank]]
    ep_recv_counts = torch.from_numpy(np.array(ep_recv_counts).astype(np.int32))
    tp_recv_counts = torch.from_numpy(np.array(tp_recv_counts).astype(np.int32))

    #-----------------------------------------------FFN golden------------------------------------------------------------------------
    # reciprocal + mul
    dynamic_scales = torch.reciprocal(dynamic_scales)
    pertoken_scale = torch.mul(dynamic_scales, 0.0000005)
    pertoken_scale = pertoken_scale.reshape(-1, 1)

    # GMM1
    expand_x = expand_x.to(torch.int32)
    weight_gmm1 = weight_gmm1.to(torch.int32)
    GMM1_OUT = torch.matmul(expand_x, weight_gmm1)
    if bias:
        GMM1_OUT = GMM1_OUT + bias.to(torch.int32)
    GMM1_OUT = GMM1_OUT.squeeze(0)

    # dequnt_swiglu_quant
    x_shape = GMM1_OUT.shape
    if group_index is not None:
        GMM1_OUT = GMM1_OUT[0:group_index.numpy(),:]
        pertoken_scale = pertoken_scale[0:group_index.numpy(),:]
    if bias is not None:
        GMM1_OUT = torch.add(GMM1_OUT, bias)
    res = torch.mul(GMM1_OUT, scale_swiglu)
    res = torch.mul(res, pertoken_scale)
    out = torch.chunk(res, 2, dim=-1)
    self_tensor = out[1]
    other = out[0]
    output = torch.nn.functional.silu(self_tensor) * other  # 100

    output = torch.mul(output, quant_scale)  # 100
    scale_out = torch.zeros([GMM1_OUT.shape[0]], dtype=torch.float64)
    output_abs = torch.abs(output)
    max_values = torch.amax(output_abs, dim=-1)
    scale_out = max_values / 127
    factors = 127 / max_values
    output = output * factors.unsqueeze(1)
    output = torch.clamp(output, -128, 127)
    output = torch.round(output)
    if group_index is not None:
        output_swiglu = torch.zeros([x_shape[0],x_shape[1]//2], dtype=torch.int8)
        output_swiglu[0:group_index.numpy(),:] = output
        gmm2_pertoken_scale = torch.zeros([x_shape[0]], dtype=torch.float64)
        gmm2_pertoken_scale[0:group_index.numpy()] = scale_out
    else:
        output_swiglu, gmm2_pertoken_scale = output.to(torch.int8), scale_out

    # GMM2
    output_swiglu = output_swiglu.to(torch.int32)
    weight_gmm2 = weight_gmm2.to(torch.int32)
    GMM2_OUT = torch.matmul(output_swiglu, weight_gmm2)
    if bias:
        GMM2_OUT = GMM2_OUT + bias.to(torch.int32)
    GMM2_OUT = GMM2_OUT * scale_gmm2
    GMM2_OUT = GMM2_OUT.to(torch.float32)
    gmm2_pertoken_scale = gmm2_pertoken_scale.reshape(-1, 1)
    GMM2_OUT = GMM2_OUT * gmm2_pertoken_scale
    GMM2_OUT = GMM2_OUT.to(torch.float16)
    GMM2_OUT = GMM2_OUT.squeeze(0)

    #-----------------------------------------------combine golden------------------------------------------------------------------------
    # combine逻辑
    p2p_op_list = []
    recv_tensor_list = []
    local_recv_tensor_list = []
    if rank % 2 == 0:
        split_list = [a - size_list[p_rank], a - size_list[p_rank + 1], size_list[p_rank], size_list[p_rank + 1]]
        ep_tensor0, ep_tensor1, fill_tensor0, fill_tensor1 = torch.split(GMM2_OUT, split_list, dim=0)
    else:
        split_list = [a - size_list[p_rank], a - size_list[p_rank - 1], size_list[p_rank], size_list[p_rank - 1]]
        ep_tensor1, ep_tensor0, fill_tensor1, fill_tensor0 = torch.split(GMM2_OUT, split_list, dim=0)
    ep_tensor0 = torch.cat([ep_tensor0, fill_tensor0], dim=0)
    ep_tensor1 = torch.cat([ep_tensor1, fill_tensor1], dim=0)

    dist.all_reduce(ep_tensor0, op=dist.ReduceOp.SUM, group=tp_group)
    dist.all_reduce(ep_tensor1, op=dist.ReduceOp.SUM, group=tp_group)
    ep_tensor = ep_tensor0 if rank % 2 == 0 else ep_tensor1
    ep_tensor, _ = torch.split(ep_tensor, [a - size_list[p_rank], size_list[p_rank]], dim=0)
    ep_send_counts = list(ep_recv_counts.numpy())
    # moe专家卡向所有卡发送数据
    if rank // 2 >= shared_expert_rank_num:
        for index in range(ep_world_size):
            send_rank_idx_2d_array = global_recv_rank_idx_2d_array[bs * index:bs * (index + 1)]
            recv_rank_idx = index * 2 if rank % 2 == 0 else index * 2 + 1
            send_tensor_list = []
            count = 0
            for idx, send_rank_idx_1d_array in enumerate(send_rank_idx_2d_array):
                if rank not in send_rank_idx_1d_array:
                    send_tensor = torch.zeros(size=[1, h], dtype=torch_dtype_map[dtype])
                else:
                    if index == 0:
                        send_tensor = ep_tensor[count].reshape([1, h])
                    else:
                        send_tensor = ep_tensor[ep_send_counts[index - 1] + count].reshape([1, h])
                    # 发送前先乘以对应的expert_scales
                    rank_index = list(send_rank_idx_1d_array).index(rank)
                    scale = global_expert_scale_list[index][count][rank_index]
                    send_tensor = (send_tensor.to(torch.float32) * scale).to(torch_dtype_map[dtype])
                    count += 1
                send_tensor_list.append(send_tensor)
            send_tensor = torch.cat(send_tensor_list, dim=0)
            if rank != recv_rank_idx:
                send_op = dist.P2POp(dist.isend, send_tensor, recv_rank_idx, group=ep_group)
                p2p_op_list.append(send_op)
            else:
                local_recv_tensor_list.append(send_tensor)
    # 共享专家卡向所有卡发送数据
    if rank // 2 < shared_expert_rank_num:
        for recv_rank_idx in range(world_size):
            if rank == recv_rank_idx % (shared_expert_rank_num * 2):
                offset = recv_rank_idx // (shared_expert_rank_num * 2)
                if offset > 0:
                    send_tensor = ep_tensor[bs * offset:bs * (offset + 1)]
                    send_op = dist.P2POp(dist.isend, send_tensor, recv_rank_idx, group=ep_group)
                    p2p_op_list.append(send_op)
                else:
                    send_tensor = ep_tensor[bs * offset:bs * (offset + 1)]
                    local_recv_tensor_list.append(send_tensor)
    # moe专家卡接收数据
    for index in range(ep_world_size):
        src_rank_idx = index * 2 if rank % 2 == 0 else index * 2 + 1
        if src_rank_idx // 2 >= shared_expert_rank_num:
            if rank != src_rank_idx:
                recv_tensor = torch.empty((bs, h), dtype=torch_dtype_map[dtype])
                recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank_idx, group=ep_group)
                p2p_op_list.append(recv_op)
            else:
                recv_tensor = local_recv_tensor_list[0]
            recv_tensor_list.append(recv_tensor)
    # 共享专家卡接收数据
    expert_index = rank // 2 % shared_expert_rank_num
    src_rank_idx = expert_index * 2 if rank % 2 == 0 else expert_index * 2 + 1
    if rank != src_rank_idx:
        recv_tensor = torch.empty((bs, h), dtype=torch_dtype_map[dtype])
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank_idx, group=ep_group)
        p2p_op_list.append(recv_op)
    else:
        recv_tensor = local_recv_tensor_list[0]
    recv_tensor_list.append(recv_tensor)
    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()
    output_x = torch.zeros(size=[bs, h], dtype=torch.float32)
    for tensor in recv_tensor_list:
        tensor = tensor.to(torch.float32)
        output_x = torch.add(output_x, tensor)
    output_x = output_x.to(torch_dtype_map[dtype])
    torch.cpu.synchronize()
    dist.destroy_process_group()
    return output_x

def exec_op(p_rank):
    rank = p_rank + 16 * args.server_index
    np.random.seed(rank)
    #-----------------------------------------------MC2 input------------------------------------------------------------------------
    # token_num = random.randint(4, 8)
    # token = np.random.uniform(-1, 1, [1, h]).astype(np.float32)
    # repeated_token = token.repeat(token_num, axis=0)
    # token_nan = np.full([8 - token_num, h], np.nan).astype(np.float32)
    # x = np.concatenate((repeated_token, token_nan), axis=0)
    if dtype == "bf16":
        x = np.random.uniform(-1, 1, [bs, h]).astype(bfloat16.as_numpy_dtype)
    else:
        x = np.random.uniform(-1, 1, [bs, h]).astype(np.float16)
        # x = np.ones([bs, h]).astype(np.float16)
    # 所有token的topk个专家组成的二维tensor,shape为 [bs, k]
    if rank % 2 == 0:
        expert_ids = golbal_expert_idx_ep0[bs * rank // 2:bs * (rank // 2 + 1)].astype(np.int32)
        recv_rank_idx_2d_array = golbal_recv_rank_id_ep0[bs * rank // 2:bs * (rank // 2 + 1)]
        global_recv_rank_idx_2d_array = golbal_recv_rank_id_ep0
    else:
        expert_ids = golbal_expert_idx_ep1[bs * (rank - 1) // 2:bs * ((rank - 1) // 2 + 1)].astype(np.int32)
        recv_rank_idx_2d_array = golbal_recv_rank_id_ep1[bs * (rank - 1) // 2:bs * ((rank - 1) // 2 + 1)]
        global_recv_rank_idx_2d_array = golbal_recv_rank_id_ep1

    if dtype == "bf16":
        x = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16)
    else:
        x = torch.from_numpy(x.astype(np.float32)).to(torch.float16)
    expert_ids = torch.from_numpy(expert_ids)
    # expert_scale = np.random.uniform(-1, 1, [bs, k]).astype(np.float32)
    expert_scale = np.ones([bs, k]).astype(np.float32)
    expert_scale = torch.from_numpy(expert_scale)
    scale_dispatch = None

    #-----------------------------------------------FFN input------------------------------------------------------------------------
    out_dtype = "fp16"
    #GMM1
    weight_gmm1 = torch.randn(E, K, N).to(torch.int8)

    group_list = [M]
    group_tensor = torch.tensor(group_list).to(torch.int64)

    # print(f"------------------gmm_flag:{gmm_flag} model_flag:{model_flag}")
    # print(f"------------------out_dtype:{out_dtype}")
    if gmm_flag == 2 or model_flag == 1:
        if out_dtype == "fp16":
            scale_swiglu = [torch.randn(E, N).to(torch.float32)]
            scale_gmm2 = [torch.randn(E, K).to(torch.float32)]
        else:
            scale_swiglu = [torch.randn(E, N).to(torch.bfloat16)]
            scale_gmm2 = [torch.randn(E, K).to(torch.bfloat16)]
    else:
        scale_gmm2 = None
        scale_swiglu = None
        out_dtype = 'int32'

    bias = None

    outdtype_dict = {'int8': torch.int8, 'bf16': torch.bfloat16, 'fp16': torch.float16,
                    'int32': torch.int32, 'fp32': torch.float32}

    output_dtype = outdtype_dict[out_dtype]

    #------npu_dequant_swiglu_quant----
    quant_scale = torch.randn(N//2,).to(torch.float32)
    quant_offset = None
    group_index = torch.tensor(M).to(torch.int64)

    #-------GMM2--------X = [M, N//2]
    weight_gmm2 = torch.randn(E, N//2, K).to(torch.int8)

    output_cpu = run_cpu(rank, p_rank, recv_rank_idx_2d_array, global_recv_rank_idx_2d_array, x, expert_ids, scale_dispatch, expert_scale, 
        weight_gmm1, scale_swiglu[0], group_tensor, output_dtype, bias, quant_scale, quant_offset, group_index, weight_gmm2, scale_gmm2[0])

    output_npu = run_npu(rank, p_rank, x, expert_ids, scale_dispatch, expert_scale, weight_gmm1, scale_swiglu, group_tensor, output_dtype, bias, quant_scale, 
        quant_offset, group_index, weight_gmm2, scale_gmm2)

    # compare result
    print(f"rank:{rank}, compare result, output_cpu.shape:{output_cpu.shape} output_cpu.dtype:{output_cpu.dtype} output_npu.shape:{output_npu.shape} output_npu.dtype:{output_npu.dtype}")
    err_thd = 0.005
    err_pct = 0.005
    result = compare_data([output_cpu,], [output_npu,], err_thd, err_pct)
    print(f"rank:{rank}, compare_data result is {result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8, help="token number")
    parser.add_argument("--h", type=int, default=7168, help="token length")
    parser.add_argument("--k", type=int, default=6, help="topk expert number")
    parser.add_argument("--N", type=int, default=2048, help="GMM N dim")
    parser.add_argument("--server_num", type=int, default=1, help="server_num")
    parser.add_argument("--server_index", type=int, default=0, help="server_index")
    args = parser.parse_args()

    os.system("rm -rf /root/ascend/log")
    torch_dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    server_index = args.server_index
    server_num = args.server_num
    port = 50501
    # 多机泛化
    master_ip = "127.0.0.1" if args.server_num == 1 else "71.20.45.121"
    h = args.h
    tp_world_size = 2
    ep_world_size = 8 * args.server_num
    world_size = tp_world_size * ep_world_size
    local_moe_expert_num = 1
    bs = args.bs
    random.seed(0)
    dtype = random.choice(["fp16"])
    global_bs = bs * ep_world_size
    shared_expert_rank_num = 1
    moe_expert_num = ep_world_size - shared_expert_rank_num
    k = args.k
    # Note: k should not bigger than moe_expert_num
    quant_mode = random.choice([2])
    # 主进程生成所有rank的expert_idx, shape为[global_bs, k]
    golbal_expert_idx_ep0 = gen_unique_topk_array(0, moe_expert_num, global_bs, k)
    golbal_expert_idx_ep1 = gen_unique_topk_array(0, moe_expert_num, global_bs, k)

    # 对应tp域内所有token接收端对应的rank
    golbal_recv_rank_id_ep0 = (golbal_expert_idx_ep0 + shared_expert_rank_num) * 2
    golbal_recv_rank_id_ep1 = (golbal_expert_idx_ep1 + shared_expert_rank_num) * 2 + 1

    #-----------------------------------------------FFN config------------------------------------------------------------------------
    M = global_bs * tp_world_size
    K = h
    N = args.N
    E = 1
    gmm_flag = 1
    # 1 是GMM1-INT32外抛， 2是GMM2-Perotoken
    model_flag = 1
    # 0 是GMM， 1是FFN-三个算子组合
    print(f"bs:{bs}, h:{h}, k:{k}, N:{N}, server_index:{server_index}, server_num:{server_num}")
    size_list = Manager().list([0] * world_size)
    p_list = []
    loop_world_size = world_size // args.server_num
    for _rank in range(loop_world_size):
        p = Process(target=exec_op, args=(_rank,))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

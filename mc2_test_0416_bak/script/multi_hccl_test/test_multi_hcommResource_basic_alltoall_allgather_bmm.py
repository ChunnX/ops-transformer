import random
import time
import os


import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm


input_dtype = torch.bfloat16         # 输出dtype

server_index = 0
port = 50001
master_ip = '127.0.0.1'
world_size = 8 # server_num * dev_num
ep_world_size = 4
tp_world_size = 2


def setup_ep_tp(rank, tp_size, ep_size, backend_type):


def get_ep_tp_hcomm_info(rank, ep_size, tp_size):
    ep_group, tp_group = setup_ep_tp(rank, tp_size, ep_size, "hccl")

    return ep_hcomm_info, tp_hcomm_info


def run_npu_process(rank):
    torch_npu.npu.set_device(rank)
    rank = rank + 16 * server_index
    dist.init_process_group(backend='hccl', rank=rank, world_size=16, init_method=f'tcp://{master_ip}:{port}')
    gp_moe = [0, 1, 2, 3, 4, 5, 6, 7] ## alltoall_allgather_bmm通信域
    gp_hccl = [8, 9, 10, 11, 12, 13, 14, 15] ## reduce_scatter通信域

    group_hccl = dist.new_group(gp_hccl)
    print(f"hccl group created success!")

    if rank in gp_moe:
        ep_group, tp_group = get_ep_tp_hcomm_info(rank, ep_world_size, tp_world_size)
        print(f"rank{rank}, ep_hcomm_info is {ep_group}, tp_hcomm_info is {tp_group}")
        # 创建moe输入
        # 创建输入tensor
        dtype = torch.float16
        x_shard_type = 1
        out_y2_flag = True
        out_y3_flag = False
        act_type = "None"
        transpose_weight = False
        # dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {ep_group}, {tp_group}')
        E, C, H, M = 4, 1024, 1024, 8192
        if x_shard_type == 0:
            x_shape = (E, C, H / tp_world_size)
        elif x_shard_type == 1:
            x_shape = (E, C / tp_world_size, H)
        else:
            x_shape = (E / ep_world_size, tp_world_size * ep_world_size * C, M / tp_world_size)
        weight_shape = (E / ep_world_size, H, M / tp_world_size)
        if transpose_weight == True:
            weight_shape = (E / ep_world_size, M / tp_world_size, H)
        bias_shape = (E / ep_world_size, 1, M / tp_world_size)
        
        x_shape = tuple(int(item) for item in x_shape)
        weight_shape = tuple(int(item) for item in weight_shape)
        bias_shape = tuple(int(item) for item in bias_shape)
        x = torch.rand(x_shape)
        weight = torch.rand(weight_shape)
        bias = torch.rand(bias_shape)
        x_npu = x.npu().to(dtype)
        weight_npu = weight.npu().to(dtype)
        if transpose_weight == True:
            print(f'!!!!before transpose, weight_npu.size()={weight_npu.size()}')
            weight_npu = weight_npu.transpose(1, 2)
            print(f'!!!!after transpose, weight_npu.size()={weight_npu.size()}')
            print(f'!!!!after transpose, weight_npu.is_contiguous()={weight_npu.is_contiguous()}')
        bias_npu = bias.npu().to(dtype)
        # 赋值None可以验证bias为空的场景
        bias_npu = None
        y_npu = npu_alltoall_allgather_bmm(x_npu,
                                        weight_npu,
                                        ep_group,
                                        ep_world_size,
                                        tp_group,
                                        tp_world_size,
                                        bias=bias_npu,
                                        shard_type=x_shard_type,
                                        act_type=act_type,
                                        need_allgather_out=out_y2_flag,
                                        need_activation_feature=out_y3_flag)
        if rank == 0:
            for i, y in enumerate(y_npu[0]):
                y.cpu().numpy().tofile(f"./y_{i}.bin")

        print(f"rank{rank} {y_npu}:alltoall_allgather_bmm finished!!!!")
        
    elif rank in gp_hccl:
        # 创建hccl 输入
        input_shape = [16, 512]
        input_dtype_hccl = torch.float16
        output_shape = [2, 512]
        np_input = np.random.uniform(1, 1, size=input_shape).astype(np.float32)
        input_hccl = torch.tensor(np_input).to(input_dtype_hccl)
        output_hccl = torch.zeros(output_shape, dtype=input_dtype_hccl).npu()

        # hcomm_info = group_hccl._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        dist._reduce_scatter_base(output_hccl, input_hccl.npu(), op=ReduceOp.SUM, group=group_hccl)
        output_hccl_cpu = output_hccl.cpu()
        print(f"hccl通信域，执行算子成功，输出为：{output_hccl_cpu}")

if __name__ == "__main__":
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序开始')
    print(f"tp_world_size={tp_world_size}", flush=True)
    print(f"ep_world_size={ep_world_size}", flush=True)

    p_list = []
    for rank in range(16):
        p = Process(target=run_npu_process, args=(rank,))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    print("run npu success.")
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序结束')




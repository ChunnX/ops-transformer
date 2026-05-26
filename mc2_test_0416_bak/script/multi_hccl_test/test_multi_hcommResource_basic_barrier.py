import random
import time
import os


import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.distributed.distributed_c10d import ReduceOp


# for test
def run_mc2_npu(rank, port, world_size):

    print("---rank:", rank, "port:", port, "world_size:", world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
    print(f"rank{rank}:init success")
    gp1 = [0, 1, 2, 3]
    gp2 = [4, 5, 6, 7]
    # gp2 = [4, 5, 6, 7]
    tp_group1 = dist.new_group(gp1)
    tp_group2 = dist.new_group(gp2)
    print(f"rank{rank}:new_group success")


    input_shape = [16, 512]
    input_dtype = 'fp16'
    np_input = np.random.uniform(1, 1, size=input_shape).astype(np.float32)
    input1 = torch.tensor(np_input).npu()
    input1 = input1.to(torch.bfloat16)
    input2 = torch.tensor(1).npu()

    weight_shape = [512, 2]
    weight_dtype = 'fp16'
    np_weight = np.random.uniform(1, 1, size=weight_shape).astype(np.float32)
    weight = torch.tensor(np_weight).npu()
    weight = weight.to(torch.bfloat16)

    if rank in gp1:
        hcomm_info = tp_group1._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output_npu = torch_npu._npu_distribute_barrier(
            x_ref=input2,
            group=hcomm_info,
            world_size=len(gp1))
        print("output_npu, rank", rank, output_npu)


    elif rank in gp2:
        all_reduce_output = torch.zeros(input1.size(), dtype=input1.dtype).npu()
        dist.all_reduce(all_reduce_output, op=dist.ReduceOp.SUM, group=tp_group2)
        print('all_reduce2, rank', rank, all_reduce_output)

    else:
        print(f"rank{rank} not in gp1/gp2")
    torch.npu.synchronize()


def aaa_npu():
    world_size = 8
    comm_size = 1
    p_list = []
    port = 36666 + random.randint(0, 10000)
    mp.set_start_method("forkserver", force=True)
    for rank in range(world_size):
        p = Process(target = run_mc2_npu, args=(rank % (world_size // comm_size), port + rank // (world_size // comm_size), world_size // comm_size))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


if __name__ == "__main__":
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序开始')
    # npu_v2()
    # npu_v3()
    aaa_npu()
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序结束')



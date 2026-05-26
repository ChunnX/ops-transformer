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
    gp1 = [0, 1, 2, 3, 4, 5, 6, 7]
    gp2 = [8, 9, 10, 11 ,12, 13, 14, 15]
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

    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.bfloat16
    # dtype = torch.float16

    # all_gather
    tensor_allgather_shape = x1_shape
    single_shape = [x1_shape[0] // world_size, x1_shape[1]]

    input_ = torch.randn(single_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()

    # alltoallv_gmm   
    e = 4
    BS = 512
    K = 8
    gmm_x_shape = [BS*K, 2048]
    gmm_weight_shape = [e, 2048, 2048]
    send_counts = [128] * (e * len(gp1))
    recv_counts = [128] * (e * len(gp2))
    input_alltoallv = torch.randn(gmm_x_shape, dtype=dtype).npu()
    weight_alltoallv = torch.randn(gmm_weight_shape, dtype=dtype).npu()
    if rank in gp1:
        hcomm_info = tp_group1._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output_npu = torch_npu.npu_alltoallv_gmm(gmm_x =input_alltoallv,
                                                gmm_weight = weight_alltoallv,
                                                hcom= hcomm_info,
                                                ep_world_size = len(gp1),
                                                send_counts = list(send_counts),
                                                recv_counts = list(recv_counts),
                                                send_counts_tensor = None, #send_counts_tensor,
                                                recv_counts_tensor = None, #recv_counts_tensor,
                                                mm_x = None,
                                                mm_weight = None,
                                                trans_gmm_weight = False,
                                                trans_mm_weight  = False,
                                                permute_out_flag  = False)

        print("output_npu, rank", rank, '\n', output_npu)


    elif rank in gp2:
        scatter_shape = [input1.size()[0] // len(gp2), input1.size()[1]]
        scatter_output = torch.zeros(scatter_shape, dtype=input1.dtype).npu()
        dist._reduce_scatter_base(scatter_output, input1, op=ReduceOp.SUM, group=tp_group2)
        print('scatter_output2, rank', rank, '\n', scatter_output)
    else:
        print(f"rank{rank} not in gp1/gp2")
    torch.npu.synchronize()


def aaa_npu():
    world_size = 16
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



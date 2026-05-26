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

    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16
    tensor_allgather_shape = x1_shape
    single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    input_allgather = torch.randn(single_shape, dtype=dtype).npu()
    weight_allgather = torch.randn(x2_shape, dtype=dtype).npu()
    if rank in gp1:
        hcomm_info = tp_group1._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        # output_npu = torch_npu.npu_mm_reduce_scatter_base(input1, weight, hcomm_info, len(gp1), reduce_op="sum")
        output_npu = torch_npu.npu_all_gather_base_mm(input_allgather, weight_allgather, hcomm_info, world_size)
        print("output_npu, rank", rank, '\n', output_npu)


    elif rank in gp2:
        sub_rank = gp2.index(rank)  # 子组内相对rank（4-7）
        sub_world_size = len(gp2)
        
        # 构造bfloat16输入数据
        input_rows = 8
        input_tensor = torch.randn(input_rows, 8, dtype=torch.bfloat16,  device='npu:{}'.format(rank))
        
        # 定义split大小（每个子rank发送/接收2行）
        input_split_sizes = [2] * sub_world_size
        output_split_sizes = [2] * sub_world_size
        output_total = sum(output_split_sizes)
        
        # 预分配输出tensor
        output_tensor = torch.empty(output_total, 8, dtype=torch.bfloat16,  device='npu:{}'.format(rank))
        
        # 调用all_to_all_single
        dist.all_to_all_single(
            output=output_tensor,
            input=input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=tp_group2,
            async_op=False
        )
        
        print(f"Rank {rank} (gp2, sub_rank {sub_rank}) all_to_all output:\n"
              f"Shape: {output_tensor.shape}, Dtype: {output_tensor.dtype}\n"
              f"Sample:\n{output_tensor[:2, :2]}\n")



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



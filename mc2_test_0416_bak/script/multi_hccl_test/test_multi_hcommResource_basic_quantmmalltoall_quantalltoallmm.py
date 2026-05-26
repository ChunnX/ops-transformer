import random
import time
import os
import argparse

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from ml_dtypes import float8_e4m3fn, float8_e5m2, bfloat16, int4
from en_dtypes import float8_e8m0, float4_e2m1, float4_e1m2, hifloat8
from torch.multiprocessing import Process
from torch.distributed.distributed_c10d import ReduceOp
from generate_mx_data import mx_quantize, to_tensor, pack_int4, unpack_group_size


# for test
def run_mc2_npu(rank, port, world_size, version="v1"):

    print("---rank:", rank, "port:", port, "world_size:", world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
    print(f"rank{rank}:init success")
    gp1 = [0, 1]
    gp2 = [4, 5]
    # gp2 = [4, 5, 6, 7]
    tp_group1 = dist.new_group(gp1)
    # tp_group2 = dist.new_group(gp2)
    print(f"rank{rank}:new_group success")

    if rank in gp1:
        x1_shape = [1024, 256]
        x2_shape = [256, 3072]
        bias_shape = [3072]
        x1_scale_shape = [1024]
        x2_scale_shape = [3072]
        x1_np = np.random.uniform(-1, 1, x1_shape).astype(float8_e4m3fn)
        x1_tensor = torch.from_numpy(x1_np.astype(np.float32)).to(torch.float8_e4m3fn).npu()
        x2_np = np.random.uniform(-1, 1, x2_shape).astype(float8_e4m3fn)
        x2_tensor = torch.from_numpy(x2_np.astype(np.float32)).to(torch.float8_e4m3fn).npu()
        bias_tensor = torch.randn(bias_shape, dtype=torch.float32).npu()
        x1_scale_tensor = torch.randn(x1_scale_shape, dtype=torch.float32).npu()
        x2_scale_tensor = torch.randn(x2_scale_shape, dtype=torch.float32).npu()
        hcomm_info = tp_group1._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output = torch_npu.npu_quant_matmul_all_to_all(
            x1_tensor,
            x2_tensor,
            hcomm_info,
            len(gp1),
            bias_tensor,
            x1_scale_tensor,
            x2_scale_tensor
        )
        print("output_npu, rank", rank, '\n', output)
    elif rank in gp2:
        x1_shape = [1024, 256]
        x2_shape = [512, 3072]
        bias_shape = [3072]
        x2_scale_shape = [3072]
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_np = np.random.uniform(-1, 1, x2_shape).astype(float8_e4m3fn)
        x2_tensor = torch.from_numpy(x2_np.astype(np.float32)).to(torch.float8_e4m3fn).npu()
        bias_tensor = torch.randn(bias_shape, dtype=torch.float32).npu()
        x2_scale_tensor = torch.randn(x2_scale_shape, dtype=torch.float32).npu()
        hcomm_info2 = tp_group2._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output2, alltoallout = torch_npu.npu_all_to_all_quant_matmul(
            x1_tensor,
            x2_tensor,
            hcomm_info2,
            len(gp2),
            bias=bias_tensor,
            x2_scale=x2_scale_tensor)
        print("output_npu2, rank", rank, '\n', output2)
    else:
        print(f"rank{rank} not in gp1/gp2")
    torch.npu.synchronize()


def aaa_npu(version):
    world_size = 2
    comm_size = 1
    p_list = []
    port = 36666 + random.randint(0, 10000)
    mp.set_start_method("forkserver", force=True)
    for rank in range(world_size):
        p = Process(target = run_mc2_npu, args=(rank % (world_size // comm_size), port + rank // (world_size // comm_size), world_size // comm_size, version))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, default="v1", help="MatMulReduceScatter or MatMulReduceScatterV2")

    args = parser.parse_args()
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序开始')
    aaa_npu(args.version)
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序结束')



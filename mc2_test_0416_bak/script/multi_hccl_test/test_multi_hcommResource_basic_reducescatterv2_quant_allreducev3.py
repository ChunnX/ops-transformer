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
    gp1 = [0, 1, 2, 3]
    gp2 = [4, 5, 6, 7]
    tp_group1 = dist.new_group(gp1)
    tp_group2 = dist.new_group(gp2)
    print(f"rank{rank}:new_group success")

    if rank in gp1:
        x1_shape = [8192, 1024]
        x2_shape = [8192, 1024]
        bias_shape = [8192]
        x1_scale_shape = [8192, 16, 2]
        x2_scale_shape = [8192, 16, 2]
        x1 = torch.tensor(np.random.uniform(-1, 1, x1_shape).astype("float32")).to(torch.float8_e5m2).npu()
        x2 = torch.tensor(np.random.uniform(-1, 1, x2_shape).astype("float32")).to(torch.float8_e4m3fn).npu()
        x2 = x2.t()
        bias = torch.randn(bias_shape, dtype=torch.float32).npu()
        x1_scale = torch.randint(0, 128, x1_scale_shape, dtype=torch.uint8).npu()
        x2_scale = torch.randint(0, 128, x2_scale_shape, dtype=torch.uint8).npu()
        group_sizes = unpack_group_size(4295032864)
        hcomm_info = tp_group1._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output, amax_out = torch_npu.npu_quant_mm_reduce_scatter(x1, x2, hcomm_info, len(gp1),
                                                                 bias=bias,
                                                                 x1_scale=x1_scale,
                                                                 x2_scale=x2_scale,
                                                                 group_sizes=group_sizes,
                                                                 y_dtype=torch.bfloat16,
                                                                 x1_scale_dtype=293,
                                                                 x2_scale_dtype=293)

        print("output_npu, rank", rank, '\n', output)
    elif rank in gp2:
        x1_shape = [1, 51, 1408]
        x2_shape = [11264, 1408]
        x3_shape = [1, 51, 11264]
        bias_shape = [11264]
        dequant_scale_shape = [1, 11264]
        pertoken_scale_shape = [51]
        commquant_scale1_shape = [1, 11264]
        commquant_scale2_shape = [1, 11264]
        x3 = torch.randn(x3_shape, dtype=torch.float16).npu()
        x1 = torch.randint(-128, 127, x1_shape, dtype=torch.int8).npu()
        x2 = torch.randint(-128, 127, x2_shape, dtype=torch.int8).npu()
        dequant_scale = torch.randn(dequant_scale_shape, dtype=torch.float32).npu()
        pertoken_scale = torch.randn(pertoken_scale_shape, dtype=torch.float32).npu()
        commquant_scale1 = torch.randn(commquant_scale1_shape, dtype=torch.float16).npu()
        commquant_scale2 = torch.randn(commquant_scale2_shape, dtype=torch.float16).npu()
        bias = torch.randint(-128, 127, bias_shape, dtype=torch.int32).npu()
        x2 = x2.t()
        hcomm_info2 = tp_group2._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output2 = torch_npu.npu_mm_all_reduce_base(
            x1=x1,
            x2=x2,
            x3=x3,
            bias=bias,
            hcom=hcomm_info2,
            dequant_scale=dequant_scale,
            pertoken_scale=pertoken_scale,
            comm_quant_scale_1=commquant_scale1,
            comm_quant_scale_2=commquant_scale2
        )
        print("output_npu2, rank", rank, '\n', output2)
    else:
        print(f"rank{rank} not in gp1/gp2")
    torch.npu.synchronize()


def aaa_npu(version):
    world_size = 8
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
    # npu_v2()
    # npu_v3()
    aaa_npu(args.version)
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序结束')



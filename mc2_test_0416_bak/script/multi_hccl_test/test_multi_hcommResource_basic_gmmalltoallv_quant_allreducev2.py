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

    x1_shape = [1, 51, 1408]
    x2_shape = [11264, 1408]
    x3_shape = [1, 51, 11264]
    bias_shape = [11264]
    dequant_scale_shape = [1, 11264]
    pertoken_scale_shape = [51]
    x3 = torch.randn(x3_shape, dtype=torch.float16).npu()

    if rank in gp1:
        dtype = torch.bfloat16
        e = 4
        BS = 512
        K = 8
        gmm_x_shape = [BS * K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        # Fix: Use correct world size for send/recv counts
        send_counts = [BS * K // (e * len(gp1))] * (e * len(gp1))
        recv_counts = [BS * K // (e * len(gp1))] * (e * len(gp1))
        input_gmm_alltoallv = torch.randn(gmm_x_shape, dtype=dtype).npu()
        weight_gmm_alltoallv = torch.randn(gmm_weight_shape, dtype=dtype).npu()
        hcomm_info = tp_group1._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output = torch_npu.npu_gmm_alltoallv(gmm_x=input_gmm_alltoallv,
                                             gmm_weight=weight_gmm_alltoallv,
                                             hcom=hcomm_info,
                                             ep_world_size=len(gp1),
                                             send_counts=list(send_counts),
                                             recv_counts=list(recv_counts),
                                             send_counts_tensor=None,  # send_counts_tensor,
                                             recv_counts_tensor=None,  # recv_counts_tensor,
                                             mm_x=None,
                                             mm_weight=None,
                                             trans_gmm_weight=False,
                                             trans_mm_weight=False)

        print("output_npu, rank", rank, '\n', output)
    elif rank in gp2:
        dtype = torch.float32
        x1 = torch.randint(-128, 127, x1_shape, dtype=torch.int8).npu()
        x2 = torch.randint(-128, 127, x2_shape, dtype=torch.int8).npu()
        bias = torch.randint(-128, 127, bias_shape, dtype=torch.int32).npu()
        x2 = x2.t()
        pertoken_scale = torch.randn(pertoken_scale_shape, dtype=dtype).npu()
        dequant_scale = torch.randn(dequant_scale_shape, dtype=dtype).npu()
        hcomm_info2 = tp_group2._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        output2 = torch_npu.npu_mm_all_reduce_base(
            x1=x1,
            x2=x2,
            x3=x3,
            bias=bias,
            hcom=hcomm_info2,
            dequant_scale=dequant_scale,
            pertoken_scale=pertoken_scale,
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



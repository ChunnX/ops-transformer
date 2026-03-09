#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import socket
import torch
import csv
import time
import torch_npu
import ascend_ops
from typing import List
import numpy as np
from torch.multiprocessing import Process, Manager, Pool, Barrier
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import pandas as pd
from math import ceil
from enum import Enum
server_num = 1
server_index = 0 if server_num > 1 else 0
master_ip = '127.0.0.1' if server_num > 1 else '127.0.0.1'
rank_per_dev = 16
world_size = server_num * rank_per_dev
de_world_size = world_size
ep_world_size = world_size

ep_ranks_list = [list(range(0, ep_world_size))]
de_ranks_list = [list(range(0, de_world_size))]
ba_ranks_list = [list(range(0, de_world_size))]


def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank):
    print(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")

    print(f"device {rank} 初始化EP域")
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group = tmp_group
    
    print(f"device {rank} 初始化DETECTION域")
    for de_ranks in de_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=de_ranks)
        if rank in de_ranks:
            de_group = tmp_group

    print(f"device {rank} 初始化BARRIER域")
    for ba_ranks in ba_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ba_ranks)
        if rank in ba_ranks:
            ba_group = tmp_group
    
    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    de_hcomm_info = de_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    ba_hcomm_info = ba_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    return ep_hcomm_info, de_hcomm_info, ba_hcomm_info, ep_group, de_group, ba_group


def warmup_barrier(ep_hcomm_info):
    x_ref = torch.ones(1, dtype=torch.int32)
    warmup_barriers = {
        'x_ref': x_ref.npu(),
        'group': ep_hcomm_info,
        'world_size': world_size
    }

    out_res = torch_npu._npu_distribute_barrier(**warmup_barriers)

    return out_res


def run_detection_npu(barrier, rank):
    set_device(rank)
    ep_hcomm_info, de_hcomm_info, ba_hcomm_info, ep_group, de_group, ba_group = init_hccl_comm(rank)
    print(f'[INFO] device_{rank} 构造detection算子的输入数据')
    tensor_x = torch.zeros(2, 2)
    tensor_x = tensor_x.to(torch.int32).npu()

    barrier.wait()

    warmup_barrier(ba_hcomm_info)
    print(f'rank {rank} epid {rank} npu barrier warm_up!')

    print(f'rank {rank} epid {rank} npu start \n')

    elastic_info = torch.ops.ascend_ops.detection(tensor_x, ep_hcomm_info, ep_world_size, rank, de_hcomm_info,
                                                  de_world_size, rank, ba_hcomm_info, de_world_size, rank, 1000)
    time.sleep(5)
    barrier.wait()
    print(f'rank {rank} epid {rank} npu finished! Prepare for secend time \n')

    elastic_info2 = torch.ops.ascend_ops.detection(tensor_x, ep_hcomm_info, ep_world_size, rank, de_hcomm_info,
                                                   de_world_size, rank, ba_hcomm_info, de_world_size, rank, 150)
    torch.npu.synchronize()
    print(f'rank {rank} epid {rank} npu finished ! \n')
    print(elastic_info2)

if __name__ == "__main__":
    p_list = []
    barrier = Barrier(rank_per_dev)
    tasks = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    
    def warpper(task):
        return run_detection_npu(barrier, task)
    
    with Pool(processes=rank_per_dev) as pool:
        pool.map(warpper, tasks)
        pool.close()
        pool.join()

    print("detection run npu success")
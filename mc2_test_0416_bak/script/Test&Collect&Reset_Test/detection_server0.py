import os,socket,torch,csv,torch,time,torch_npu
from typing import List
import numpy as np
from torch.multiprocessing import Process, Manager, Pool, Barrier
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import pandas as pd
from math import ceil
from enum import Enum
import argparse

server_num = 1
server_index = 0 if server_num > 1 else 0                        # 多机每个脚本保持一致并递增index值
master_ip = '100.102.180.141' if server_num > 1 else '127.0.0.1' 
is_dispatch = 1
is_combine = 0
is_cascade =0
is_elastic=1
quant_mode = 2
is_mask = 0 
rank_per_dev = 16 
world_size=server_num * rank_per_dev
de_world_size= world_size
ep_world_size=world_size
bs = 16
h =7168
k = 2
#通行域
ep_ranks_list = [list(range(0, ep_world_size))]
de_ranks_list = [list(range(0, de_world_size))]
def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")

def init_hccl_comm(rank):
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")
    
    print(f"device {rank} 初始化EP域")
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group = tmp_group

    print(f"device {rank} 初始化DECTECTION域")
    for de_ranks in de_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=de_ranks)
        if rank in de_ranks:
            de_group = tmp_group

    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    de_hcomm_info = de_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    
    return ep_hcomm_info, de_hcomm_info, ep_group, de_group

def get_detection_kwargs(
    expand_x,group_ep,ep_world_size,ep_rank_id,group_detection,detection_world_size,detection_rank_id
    ):
    return{
       "expand_x" :expand_x, 
       "group_ep" :group_ep,
       "ep_world_size":ep_world_size,
       "ep_rank_id":ep_rank_id,
       "group_detection":group_detection,
       "detection_world_size":detection_world_size,
       "detection_rank_id":detection_rank_id
    }

def warmup_barrier(ep_hcomm_info):
    x_ref = torch.ones(1, dtype=torch.int32)
    timeout = torch.tensor([1000000], dtype=torch.int32)
    warmup_barriers={
        'x_ref':x_ref.npu(),
        #'time_out':timeout.npu(),
        'group':ep_hcomm_info,
        'world_size': world_size,
    }

    out_res=torch_npu._npu_distribute_barrier(**warmup_barriers)

    return out_res

def run_dectection_npu(barrier, rank, data):
    out = open(f"stdout_{rank}.txt", "w")
    err = open(f"stderr_{rank}.txt", "w")

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(out.fileno(), 1)
    os.dup2(err.fileno(), 2)
    set_device(rank)
    ep_hcomm_info, de_hcomm_info, ep_group, de_group = init_hccl_comm(rank)
    print(f'[INFO] device_{rank} 构造dispatch算子输入数据')
    tensor_x = torch.zeros(2,2)
    tensor_x = tensor_x.to(torch.int32).npu()
    detection_kwargs=get_detection_kwargs(
        expand_x=tensor_x,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        group_detection=de_hcomm_info,
        detection_world_size=de_world_size,
        detection_rank_id=rank,
    )
    barrier.wait()
    
    warmup_barrier(ep_hcomm_info)
    print(f'rank {rank} epid {rank}  npu barrier warm_up! \n')
   
    print(f'rank {rank} epid {rank}  npu start! \n')

    elestic_info=torch_npu.npu_fault_detection(**detection_kwargs)
    #print(elestic_info)

    time.sleep(5)
    barrier.wait()
    print(f'rank {rank} epid {rank}  npu finished! Prepare for second time\n')

    print(f"******************************rank data:{rank}*****************************")
    print(data)
    
    if data is not None:
        if rank in data:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            return

    elestic_info2=torch_npu.npu_fault_detection(**detection_kwargs)

    torch.npu.synchronize()
    print(f'rank {rank} epid {rank}  npu finished! \n')
    print(elestic_info2.cpu())

    os.dup2(old_stdout, 1)
    os.dup2(old_stderr, 2)
    os.close(old_stdout)
    os.close(old_stderr)

if __name__=="__main__":
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--nargs-int-type', type=int, nargs='+', help='an integer array')


    for _,data in cmdline.parse_args()._get_kwargs():
        if data is not None:
            print(data)







    p_list = []
    # for rank in range(rank_per_dev):
    #     p = Process(target=run_dectection_npu, args=(rank,))
    #     p_list.append(p)
    # for p in p_list:
    #     p.start()
    # for p in p_list:
    #     p.join()
    barrier = Barrier(rank_per_dev)
    tasks = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))

    def warpper(task):
        return run_dectection_npu(barrier, task, data)

    with Pool(processes=rank_per_dev) as pool:
        pool.map(warpper, tasks)
        pool.close()
        pool.join()
        
    print("dectection run npu success.")

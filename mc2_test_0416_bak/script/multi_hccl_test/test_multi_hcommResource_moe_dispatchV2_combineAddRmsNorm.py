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

# 控制模式
quant_mode = 2                       # 2为动态量化
is_dispatch_scales = True            # 动态量化可选择是否传scales
input_dtype = torch.bfloat16         # 输出dtype
server_num = 1
server_index = 0
port = 50001
master_ip = '127.0.0.1'
dev_num = 8
world_size = server_num * dev_num
rank_per_dev = int(world_size / server_num)  # 每个host有几个die
shared_expert_rank_num = 0                      # 共享专家数
moe_expert_num = 8                            # moe专家数
bs = 8                                       # token数量
h = 7168                                     # 每个token的长度
k = 8
random_seed = 0
tp_world_size = 1
ep_world_size = int(world_size / tp_world_size)
moe_rank_num = ep_world_size - shared_expert_rank_num
local_moe_expert_num = moe_expert_num // moe_rank_num
globalBS = bs * ep_world_size
is_shared = (shared_expert_rank_num > 0)
is_quant = (quant_mode > 0)
zero_expert_num = 0
copy_expert_num = 0
const_expert_num = 0


elastic_ranks = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def gen_elastic_info(elastic_ranks, is_elastic=1, actual_rank_num=14, elastic_shared_rank=4, elastic_moe_num=10, ep_world_size=16):
    if not is_elastic: return None
    elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype=torch.int32)
    elastic_info[0] = is_elastic
    elastic_info[1] = actual_rank_num
    elastic_info[2] = shared_expert_rank_num
    elastic_info[3] = moe_expert_num
    table1 = [-1] * ep_world_size
    table2 = [-1] * ep_world_size

    for local_rank_id, ep_rank_id in enumerate(elastic_ranks):
        if ep_rank_id < ep_world_size:
            table1[ep_rank_id] = local_rank_id
            table2[local_rank_id] = ep_rank_id
    for i in range(ep_world_size):
        elastic_info[4 + i] = table1[i]
    for i in range(ep_world_size):
        elastic_info[4 + ep_world_size + i] = table2[i]
    assert elastic_info.shape[0] == 4 + 2 * ep_world_size
    if is_elastic:
        table1 = elastic_info[4:4+ep_world_size]
        table2 = elastic_info[4+ep_world_size:4+2*ep_world_size]
    return elastic_info


def gen_const_expert_alpha_1():
    const_expert_alpha_1 = torch.empty(size=[const_expert_num ,h], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_alpha_1

def gen_const_expert_alpha_2():
    const_expert_alpha_2 = torch.empty(size=[const_expert_num ,h], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_alpha_2

def gen_const_expert_v():
    const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_v

def gen_unique_topk_array(low, high, bs, k):
    array = []
    for i in range(bs):
        top_idx = list(np.arange(low, high, dtype=np.int32))
        random.shuffle(top_idx)
        array.append(top_idx[0:k])
    return np.array(array)

def get_new_group(rank):
    for i in range(tp_world_size):
        # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
        ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_t = ep_group
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
    for i in range(ep_world_size):
        # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
        tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_t = tp_group
            print(f"rank:{rank} tp_ranks:{tp_ranks}")
    return ep_group_t, tp_group_t

def get_hcomm_info(rank, comm_group):
    if torch.__version__ > '2.0.1':
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info

def warm_up_dispatch(rank, group_ep, group_tp):
    x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
    expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()

    dispatch_kwargs_before = get_dispatch_kwargs_warmup(
        x_warm_up=x_warm_up,
        expert_ids_warm_up=expert_ids_warm_up,
        group_ep=group_ep,
        group_tp=group_tp,
        ep_rank_id=rank//tp_world_size,
        tp_rank_id=rank%tp_world_size,
    )

    (
        expand_x, dynamic_scales, expand_idx,
        expert_token_nums, ep_recv_counts, tp_recv_counts, _
    ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs_before)


    return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

def get_dispatch_kwargs_warmup(
    x_warm_up, expert_ids_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id, 
):
    x_warm_up = x_warm_up.to(input_dtype).npu()
    expert_ids_warm_up = expert_ids_warm_up.to(torch.int32).npu()

    return {
        'x': x_warm_up,
        'expert_ids': expert_ids_warm_up,
        'x_active_mask': None,
        'group_ep': group_ep,
        'group_tp': group_tp,
        'ep_rank_id': ep_rank_id,
        'tp_rank_id': tp_rank_id,
        'ep_world_size': ep_world_size,
        'tp_world_size': tp_world_size,
        'expert_shard_type': 0,
        'shared_expert_num': 0,
        'shared_expert_rank_num': shared_expert_rank_num,
        'moe_expert_num': moe_expert_num,
        'scales': None,
        'quant_mode': 2,
        'global_bs': 16,
    }


def run_npu_process(rank):
    torch_npu.npu.set_device(rank)
    rank = rank + 16 * server_index
    dist.init_process_group(backend='hccl', rank=rank, world_size=16, init_method=f'tcp://{master_ip}:{port}')
    gp_moe = [0, 1, 2, 3, 4, 5, 6, 7] ## moe通信域
    gp_hccl = [8, 9, 10, 11, 12, 13, 14, 15] ## allgather通信域

    group_hccl = dist.new_group(gp_hccl)
    print(f"hccl group created success!")

    if rank in gp_moe:
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)
        print(f"rank{rank}, ep_hcomm_info is {ep_hcomm_info}, tp_hcomm_info is {tp_hcomm_info}")
        # 创建moe输入
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        # expert_ids（Tensor）：必选参数，每个token的topK个专家索引，要求为2D的Tensor，shape为(BS, K)。数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应torch_npu.npu_moe_distribute_dispatch的expert_ids输入，张量里value取值范围为[0, moe_expert_num)，且同一行中的K个value不能重复。
        expert_ids = torch.tensor([[ 5,  7, 1,  4,  2,  6, 0, 3],
            [5,  2, 1,  4,  7,  6, 0, 3],
            [1,  7, 5,  4,  2,  3, 0, 6],
            [7,  5, 1,  2,  4,  0, 6, 3],
            [2,  7, 1,  4,  2,  6, 0, 3],
            [3,  7, 1,  4,  2,  6, 0, 5],
            [4,  7, 1,  5,  2,  6, 0, 3],
            [6,  7, 1,  4,  2,  5, 0, 3]], dtype=torch.int32).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        # elastic_info = torch.tensor([ 1, 10, 0, 20,
        #                             -1,  0, 1,  2, -1, 3, -1,  4, -1,  5,  6,  7, -1,  8,  9, -1,
        #                             1,  2, 3,  5,  7, 9, 10, 11, 13, 14, -1, -1, -1, -1, -1,  -1], dtype=torch.int32).npu()
        # available_ranks = [1, 2, 3, 5, 7, 9, 10, 11, 13, 14]
        elastic_info = None
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()

        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)

        if rank in gp_moe:
            expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                group_ep=ep_hcomm_info,
                group_tp=tp_hcomm_info,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                ep_rank_id=rank // tp_world_size,
                tp_rank_id=rank % tp_world_size,
                expert_shard_type=0,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                scales=scales,
                quant_mode=quant_mode,
                global_bs=globalBS,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num)

            if is_quant:
                expand_x = expand_x.to(input_dtype)
        
            bs_local = expert_ids.shape[0]
            torch.manual_seed(42)
            residual_x = torch.rand((bs_local, 1, h), dtype=torch.bfloat16).npu()
            torch.manual_seed(random_seed)
            gamma = torch.ones(h).to(input_dtype).npu()
            norm_eps = 1e-6
            y, rstd_out, x = torch_npu.npu_moe_distribute_combine_add_rms_norm(
                expand_x=expand_x,
                residual_x=residual_x,
                gamma=gamma,
                norm_eps=norm_eps,
                expert_ids=expert_ids,
                expand_idx=expand_idx,
                ep_send_counts=ep_recv_counts,
                tp_send_counts=tp_recv_counts,
                expert_scales=expert_scales,
                group_ep=ep_hcomm_info,
                group_tp=tp_hcomm_info,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                ep_rank_id=rank // tp_world_size,
                tp_rank_id=rank % tp_world_size,
                expert_shard_type=0,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                global_bs=globalBS,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
        time.sleep(10)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')
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
    print(f"bs={bs}")
    print(f"global_bs={globalBS}")
    print(f"shared_expert_rank_num={shared_expert_rank_num}")
    print(f"moe_expert_num={moe_expert_num}")
    print(f"k={k}")
    print(f"quant_mode={quant_mode}", flush=True)
    print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
    print(f"tp_world_size={tp_world_size}", flush=True)
    print(f"ep_world_size={ep_world_size}", flush=True)

    if tp_world_size != 1 and local_moe_expert_num > 1:
        print("unSupported tp = 2 and local moe > 1")
        exit(0)

    if shared_expert_rank_num > ep_world_size:
        print("shared_expert_rank_num 不能大于 ep_world_size")
        exit(0)

    if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
        print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
        exit(0)

    if moe_expert_num % moe_rank_num != 0:
        print("moe_expert_num 必须是 moe_rank_num 的整数倍")
        exit(0)


    p_list = []
    for rank in range(16):
        p = Process(target=run_npu_process, args=(rank,))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    print("run npu success.")


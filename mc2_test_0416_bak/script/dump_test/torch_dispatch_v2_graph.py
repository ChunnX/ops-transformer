# 仅支持静态图
import os
import torch
import random
import torch_npu
import torchair
import numpy as np
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.distributed import ReduceOp
import time


# 控制模式
quant_mode = 2  # 2为动态量化
is_dispatch_scales = True  # 动态量化可选择是否传scales
input_dtype = torch.bfloat16  # 输出dtype
server_num = 1
server_index = 0
port = 50001
master_ip = '127.0.0.1'
dev_num = 16
world_size = server_num * dev_num
rank_per_dev = int(world_size / server_num)  # 每个host有几个die
shared_expert_rank_num = 0  # 共享专家数
moe_expert_num = 32  # moe专家数
bs = 8  # token数量
h = 7168  # 每个token的长度
k = 8
random_seed = 0
tp_world_size = 1
ep_world_size = int(world_size / tp_world_size)
moe_rank_num = ep_world_size - shared_expert_rank_num
local_moe_expert_num = moe_expert_num // moe_rank_num
globalBS = bs * ep_world_size
is_shared = (shared_expert_rank_num > 0)
is_quant = (quant_mode > 0)

zero_expert_num = 1
copy_expert_num = 1
const_expert_num = 1


class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                scales, quant_mode, global_bs, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
        output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            group_ep=group_ep,
            group_tp=group_tp,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=ep_rank_id,
            tp_rank_id=tp_rank_id,
            expert_shard_type=expert_shard_type,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=moe_expert_num,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=global_bs,
            elastic_info=elastic_info,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num
        )
        expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
        if expand_x_npu.dtype == torch.int8:
            expand_x_npu = expand_x_npu.to(input_dtype)

        output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expand_x_npu,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine_npu,
            ep_send_counts=ep_recv_counts_npu,
            tp_send_counts=tp_recv_counts_npu,
            expert_scales=expert_scales,
            group_ep=group_ep,
            group_tp=group_tp,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=ep_rank_id,
            tp_rank_id=tp_rank_id,
            expert_shard_type=expert_shard_type,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=moe_expert_num,
            global_bs=global_bs,
            elastic_info=elastic_info,
            ori_x=x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num
        )
        x = output_combine_npu
        x_combine_res = output_combine_npu
        return [x_combine_res, output_combine_npu]


def gen_const_expert_alpha_1():
    const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_alpha_1


def gen_const_expert_alpha_2():
    const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_alpha_2


def gen_const_expert_v():
    const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_v


def get_new_group(rank):
    for i in range(tp_world_size):
        ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_t = ep_group
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
    for i in range(ep_world_size):
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
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{master_ip}:{port}'
    )
    ep_group, tp_group = get_new_group(rank)
    ep_hcomm_info = get_hcomm_info(rank, ep_group)
    tp_hcomm_info = get_hcomm_info(rank, tp_group)

    # 创建输入tensor
    x = torch.randn(bs, h, dtype=input_dtype).npu()
    expert_ids = torch.tensor([
        [0, 8, 4, 1, 6, 12, 14, 17],
        [14, 10, 7, 3, 0, 12, 11, 17],
        [12, 0, 5, 11, 19, 4, 6, 18],
        [17, 3, 4, 10, 18, 0, 1, 2],
        [13, 16, 9, 10, 15, 6, 7, 14],
        [17, 15, 14, 8, 16, 18, 3, 12],
        [4, 12, 2, 17, 15, 3, 9, 10],
        [16, 7, 12, 9, 18, 3, 19, 17]
    ], dtype=torch.int32).npu()

    expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
    scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
    if is_dispatch_scales:
        scales = torch.randn(scales_shape, dtype=torch.float32).npu()
    else:
        scales = None

    elastic_info = torch.tensor([
        1, 10, 0, 20,
        -1, 0, 1, 2, -1, 3, -1, 4, -1, 5, 6, 7, -1, 8, 9, -1,
        1, 2, 3, 5, 7, 9, 10, 11, 13, 14, -1, -1, -1, -1, -1, -1
    ], dtype=torch.int32).npu()
    available_ranks = [1, 2, 3, 5, 7, 9, 10, 11, 13, 14]
    const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
    const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
    const_expert_v = gen_const_expert_v().npu()
    out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)

    model = MOE_DISTRIBUTE_GRAPH_Model()
    model = model.npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    #if rank in available_ranks:
    output = model.forward(
        x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
        rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, moe_expert_num, scales,
        quant_mode, globalBS, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
        zero_expert_num, copy_expert_num, const_expert_num
    )
    torch.npu.synchronize()
    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    time.sleep(10)


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
    for rank in range(rank_per_dev):
        p = Process(target=run_npu_process, args=(rank,))
        p_list.append(p)

    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

    print("run npu success.")
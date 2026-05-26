import os
import torch
import random
import torch_npu
import numpy as np
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.distributed import ReduceOp

# 控制模式
quant_mode = 0  # dispatch量化模式
is_dispatch_scales = False  # 动态量化可选择是否传scales
input_dtype = torch.bfloat16  # 输出dtype
y_dtype = "bfloat16"
server_num = 1
master_ip = "127.0.0.1"
dev_num = 2
world_size = server_num * dev_num
rank_per_dev = int(world_size / server_num)  # 每个host有几个die
shared_expert_rank_num = 0  # 共享专家数
moe_expert_num = 32  # moe专家数
shared_expert_num = 0
shared_expert_rank_num = 0
bs = 8  # token数量
h = 7168  # 每个token的长度
k = 8
tp_world_size = 1
ep_world_size = int(world_size / tp_world_size)
moe_rank_num = ep_world_size - shared_expert_rank_num
local_moe_expert_num = moe_expert_num // moe_rank_num
global_bs = bs * ep_world_size
is_shared = shared_expert_rank_num > 0
is_quant = quant_mode > 0
is_quant_scales = False
comm_quant_mode = 0
is_topk_balance = True  # 是否均衡topK
is_mask = False
ep_ranks_list = [list(range(0, world_size))]

os.environ["HCCL_BUFFSIZE"] = "5000"

DTYPE_TO_INT_VALUE_MAP = {
    "float16": 5,
    "bfloat16": 15,
    "float8_e5m2": 23,
    "float8_e4m3fn": 24,
}


def gen_expert_ids(rank):
    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]

    def gen_balanced_expert_ids_element(round_index):
        return (
            torch.arange(round_index, round_index + moe_expert_num, dtype=torch.int32)
            % moe_expert_num
        )

    ep_expert_ids_list = []
    if is_topk_balance:
        ep_expert_ids = [
            gen_balanced_expert_ids_element(round_index)
            for round_index in range((bs * ep_world_size * k) // moe_expert_num + 1)
        ]
    else:
        ep_expert_ids = [
            gen_random_token_expert_ids(k, 0, moe_expert_num)
            for _ in range(bs * ep_world_size)
        ]
    # print(ep_expert_ids)
    ep_expert_ids_list.append(
        torch.cat(ep_expert_ids)[: bs * ep_world_size * k].view([bs * ep_world_size, k])
    )

    return ep_expert_ids_list[0][rank * bs : (rank + 1) * bs, :]


def init_hccl_comm(rank):
    print(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50001",
    )
    print(f"device_{rank} init_process_group success")

    print(f"device {rank} 初始化EP域")
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks_list[0])
        if rank in ep_ranks:
            ep_group = tmp_group

    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    return ep_hcomm_info, ep_group


def get_dispatch_setup_kwargs(
    x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_shard_type,
    shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, comm_type, comm_alg
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask else None
    if is_quant and is_quant_scales:
        scales = scales.to(torch.float32).npu()
    return {
        "x": x,
        "expert_ids": expert_ids,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "scales": scales,
        "x_active_mask": x_active_mask,
        "expert_shard_type": expert_shard_type,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "quant_mode": quant_mode,
        "global_bs": global_bs,
        "comm_type": comm_type,
        "comm_alg": comm_alg,
        "y_dtype": DTYPE_TO_INT_VALUE_MAP[y_dtype],
    }


def get_dispatch_teardown_kwargs(
    x, y, expert_ids, comm_cmd_info, group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type,
    shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_type, comm_alg
):
    x = x.to(input_dtype).npu()
    y = y.npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    comm_cmd_info = comm_cmd_info.to(torch.int32).npu()
    return {
        "x": x,
        "y": y,
        "expert_ids": expert_ids,
        "comm_cmd_info": comm_cmd_info,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "expert_shard_type": expert_shard_type,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "quant_mode": quant_mode,
        "global_bs": global_bs,
        "expert_token_nums_type": expert_token_nums_type,
        "comm_type": comm_type,
        "comm_alg": comm_alg,
    }


def get_combine_setup_kwargs(
    expand_x, expert_ids, assist_info_for_combine, shared_expert_rank_num, global_bs, comm_quant_mode, group_ep,
    ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, comm_alg, comm_type
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    assist_info_for_combine = assist_info_for_combine.to(torch.int32).npu()

    return {
        "expand_x": expand_x,
        "expert_ids": expert_ids,
        "assist_info_for_combine": assist_info_for_combine,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "expert_shard_type": expert_shard_type,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "global_bs": global_bs,
        "comm_quant_mode": comm_quant_mode,
        "comm_type": comm_type,
        "comm_alg": comm_alg,
    }


def get_combine_teardown_kwargs(
    expand_x, quant_expand_x, expert_ids, expand_idx, expert_scales, ep_world_size, shared_expert_x, expert_shard_type,
    x_active_mask, comm_cmd_info, group_ep, ep_rank_id, moe_expert_num, shared_expert_num, shared_expert_rank_num,
    global_bs, comm_quant_mode, comm_type, comm_alg
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu()
    expert_scales = expert_scales.to(torch.float32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask else None
    quant_expand_x = quant_expand_x.to(torch.int8).npu()
    comm_cmd_info = comm_cmd_info.to(torch.int32).npu()

    return {
        "expand_x": expand_x,
        "quant_expand_x": quant_expand_x,
        "expert_ids": expert_ids,
        "expand_idx": expand_idx,
        "expert_scales": expert_scales,
        "comm_cmd_info": comm_cmd_info,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "x_active_mask": x_active_mask,
        "shared_expert_x": None,
        "expert_shard_type": 0,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "global_bs": global_bs,
        "comm_quant_mode": comm_quant_mode,
        "comm_type": comm_type,
        "comm_alg": comm_alg,
    }


def run_npu_process(rank):
    torch_npu.npu.set_device(rank)
    ep_hcomm_info, _ = init_hccl_comm(rank)

    # 创建输入tensor
    x = torch.randn(bs, h, dtype=input_dtype).npu()
    expert_ids = gen_expert_ids(rank).npu()
    expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
    x_active_mask = None

    scales_shape = (
        (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
    )
    if is_dispatch_scales:
        scales = torch.randn(scales_shape, dtype=torch.float32).npu()
    else:
        scales = None

    dispatch_setup_kwargs = get_dispatch_setup_kwargs(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=x_active_mask,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        scales=scales,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=2,
        comm_alg="",
    )
    dispatch_teardown_kwargs = get_dispatch_teardown_kwargs(
        x=x,
        y=x,
        comm_cmd_info=x,
        expert_ids=expert_ids,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=2,
        expert_token_nums_type=1,
        comm_alg="",
    )

    combine_setup_kwargs = get_combine_setup_kwargs(
        expand_x=x,
        expert_ids=expert_ids,
        assist_info_for_combine=x,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        comm_quant_mode=comm_quant_mode,
        comm_type=2,
        comm_alg="",
    )
    combine_teardown_kwargs = get_combine_teardown_kwargs(
        expand_x=x,
        quant_expand_x=x,
        expert_ids=expert_ids,
        expand_idx=x,
        expert_scales=expert_scales,
        comm_cmd_info=x,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        x_active_mask=x_active_mask,
        shared_expert_x=None,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        comm_quant_mode=comm_quant_mode,
        comm_type=2,
        comm_alg="",
    )

    y, expand_idx, comm_cmd_info = torch_npu.npu_moe_distribute_dispatch_setup(
        **dispatch_setup_kwargs
    )
    dispatch_teardown_kwargs["y"] = y
    dispatch_teardown_kwargs["comm_cmd_info"] = comm_cmd_info
    expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums = (
        torch_npu.npu_moe_distribute_dispatch_teardown(**dispatch_teardown_kwargs)
    )

    if is_quant:
        expand_x = expand_x.to(input_dtype)

    combine_setup_kwargs["expand_x"] = expand_x
    combine_setup_kwargs["assist_info_for_combine"] = assist_info_for_combine
    quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(
        **combine_setup_kwargs
    )
    combine_teardown_kwargs["expand_x"] = expand_x
    combine_teardown_kwargs["quant_expand_x"] = quant_expand_x
    combine_teardown_kwargs["expand_idx"] = expand_idx
    combine_teardown_kwargs["comm_cmd_info"] = comm_cmd_info
    x = torch_npu.npu_moe_distribute_combine_teardown(**combine_teardown_kwargs)

    print(
        f"rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n{x}"
    )


if __name__ == "__main__":
    print(f"bs={bs}")
    print(f"global_bs={global_bs}")
    print(f"shared_expert_rank_num={shared_expert_rank_num}")
    print(f"moe_expert_num={moe_expert_num}")
    print(f"k={k}")
    print(f"quant_mode={quant_mode}")
    print(f"local_moe_expert_num={local_moe_expert_num}")
    print(f"ep_world_size={ep_world_size}")

    p_list = []
    for rank in range(rank_per_dev):
        p = Process(target=run_npu_process, args=(rank,))
        p_list.append(p)

    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

    print("run finish.")

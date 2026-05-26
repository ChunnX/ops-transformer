# 修改graph_type支持静态图、动态图
import os
import torch
import random
import torch_npu
import torchair
import numpy as np
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch._dynamo

import torch, os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging
# logger.setLevel(logging.DEBUG)

import numpy as np

torch._dynamo.config.suppress_errors = True

# 控制模式
quant_mode = 2
is_dispatch_scales = True
input_dtype = torch.bfloat16
server_num = 1
server_index = 0
port = 50001
master_ip = '127.0.0.1'
dev_num = 16
world_size = server_num * dev_num
rank_per_dev = int(world_size / server_num)
shared_expert_rank_num = 0
moe_expert_num = 8
bs = 8
h = 7168
k = 8
random_seed = 0
tp_world_size = 2
ep_world_size = int(world_size / tp_world_size)
moe_rank_num = ep_world_size - shared_expert_rank_num
local_moe_expert_num = moe_expert_num // moe_rank_num
globalBS = bs * ep_world_size
is_shared = (shared_expert_rank_num > 0)
is_quant = (quant_mode > 0)
is_pruning = 1
balance_mode = 1


class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size, ep_rank_id, tp_rank_id,
                expert_shard_type, shared_expert_rank_num, moe_expert_num, scales, quant_mode, global_bs, expert_scales,
                eplb_table, pruning_threshold, active_mask, balance_mode, rank=0):
        # if rank == 0:
        #     print(x.shape, expert_ids.shape, ep_world_size, tp_world_size, ep_rank_id, tp_rank_id,
        #         expert_shard_type, shared_expert_rank_num, moe_expert_num, scales.shape, quant_mode, global_bs, expert_scales.shape,
        #         eplb_table.shape, pruning_threshold.shape, active_mask.shape, balance_mode)
        e, c, h, m, k, n = [32, 1024, 1024, 1024, 16384, 16384]
        device = torch.device("npu")
        np_input = np.random.uniform(-1, 1, [m, k]).astype(np.float16)
        np_weight = np.random.uniform(-1, 1, [k, n]).astype(np.float16)
        input_ = torch.from_numpy(np_input).to(device)
        weight_ = torch.from_numpy(np_weight).to(device)

        with tng.scope.npu_stream_switch('1'):
            torch.matmul(input_, weight_)

        with tng.scope.npu_stream_switch('2'):
            balanced_expert_ids, balanced_active_mask = torch_npu.npu_moe_update_expert(expert_ids=expert_ids,
                                                                                        eplb_table=eplb_table,
                                                                                        expert_scales=expert_scales if is_pruning else None,
                                                                                        pruning_threshold=pruning_threshold,
                                                                                        active_mask=active_mask,
                                                                                        local_rank_id=ep_rank_id,
                                                                                        world_size=ep_world_size,
                                                                                        balance_mode=balance_mode)
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=x,
                                                                           expert_ids=balanced_expert_ids,
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
                                                                           global_bs=global_bs)
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x_npu,
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
                                                                         global_bs=global_bs)
            x = output_combine_npu
            x_combine_res = output_combine_npu
        return [x_combine_res, output_combine_npu]


def gen_unique_topk_array(low, high, bs, k):
    array = []
    for i in range(bs):
        top_idx = list(np.arange(low, high, dtype=np.int32))
        random.shuffle(top_idx)
        array.append(top_idx[0:k])
    return np.array(array)


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


def run_npu_process(rank):
    torch_npu.npu.set_device(rank)
    rank = rank + 16 * server_index
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
    ep_group, tp_group = get_new_group(rank)
    ep_hcomm_info = get_hcomm_info(rank, ep_group)
    tp_hcomm_info = get_hcomm_info(rank, tp_group)

    # 创建输入tensor
    x = torch.randn(bs, h, dtype=input_dtype).npu()
    expert_ids = gen_unique_topk_array(0, moe_expert_num, bs, k).astype(np.int32)
    expert_ids = torch.from_numpy(expert_ids).npu()
    eplb_table = np.array(
        [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12],
         [1, 13]]).astype(np.int32)
    eplb_table = torch.from_numpy(eplb_table).npu()
    expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()

    pruning_threshold = None
    active_mask = None
    if is_pruning:
        expert_scales_temp = -np.sort(-np.random.uniform(low=0, high=0.25, size=(bs, k)), axis=1)
        pruning_threshold_temp = np.random.uniform(low=0, high=0.15, size=(1, k))
        num_true = np.random.randint(0, bs + 1)
        active_mask_temp = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(bs - num_true, dtype=bool)])
        expert_scales_tensor = torch.from_numpy(expert_scales_temp).to(torch.float32)
        pruning_threshold_tensor = torch.from_numpy(pruning_threshold_temp).to(torch.float32)
        active_mask_tensor = torch.from_numpy(active_mask_temp).to(torch.bool)
        expert_scales = expert_scales_tensor.npu()
        pruning_threshold = pruning_threshold_tensor.npu()
        active_mask = active_mask_tensor.npu()

    scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
    if is_dispatch_scales:
        scales = torch.randn(scales_shape, dtype=torch.float32).npu()
    else:
        scales = None

    model = MOE_DISTRIBUTE_GRAPH_Model()
    model = model.npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    # 打印参数（PyTorch专属，参数名+值/张量shape）
    if rank == 0:
        print(f"x: {x.shape if isinstance(x, torch.Tensor) else x}")
        print(f"expert_ids: {expert_ids.shape if isinstance(expert_ids, torch.Tensor) else expert_ids}")
        print(f"group_ep: {ep_hcomm_info.shape if isinstance(ep_hcomm_info, torch.Tensor) else ep_hcomm_info}")
        print(f"group_tp: {tp_hcomm_info.shape if isinstance(tp_hcomm_info, torch.Tensor) else tp_hcomm_info}")
        print(f"ep_world_size: {ep_world_size.shape if isinstance(ep_world_size, torch.Tensor) else ep_world_size}")
        print(f"tp_world_size: {tp_world_size.shape if isinstance(tp_world_size, torch.Tensor) else tp_world_size}")
        print(
            f"ep_rank_id: {rank // tp_world_size.shape if isinstance(rank // tp_world_size, torch.Tensor) else rank // tp_world_size}")
        print(
            f"tp_rank_id: {rank % tp_world_size.shape if isinstance(rank % tp_world_size, torch.Tensor) else rank % tp_world_size}")
        print(f"expert_shard_type: 0")
        print(
            f"shared_expert_rank_num: {shared_expert_rank_num.shape if isinstance(shared_expert_rank_num, torch.Tensor) else shared_expert_rank_num}")
        print(f"moe_expert_num: {moe_expert_num.shape if isinstance(moe_expert_num, torch.Tensor) else moe_expert_num}")
        print(f"scales: {scales.shape if isinstance(scales, torch.Tensor) else scales}")
        print(f"quant_mode: {quant_mode.shape if isinstance(quant_mode, torch.Tensor) else quant_mode}")
        print(f"global_bs: {globalBS.shape if isinstance(globalBS, torch.Tensor) else globalBS}")
        print(f"expert_scales: {expert_scales.shape if isinstance(expert_scales, torch.Tensor) else expert_scales}")
        print(f"eplb_table: {eplb_table.shape if isinstance(eplb_table, torch.Tensor) else eplb_table}")
        print(
            f"pruning_threshold: {pruning_threshold.shape if isinstance(pruning_threshold, torch.Tensor) else pruning_threshold}")
        print(f"active_mask: {active_mask.shape if isinstance(active_mask, torch.Tensor) else active_mask}")
        print(f"balance_mode: {balance_mode.shape if isinstance(balance_mode, torch.Tensor) else balance_mode}")
    output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                           rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, moe_expert_num,
                           scales,
                           quant_mode, globalBS, expert_scales, eplb_table, pruning_threshold, active_mask,
                           balance_mode)
    torch.npu.synchronize()
    if rank == 0:
        print(output)
    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')


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
    print("run npu success")
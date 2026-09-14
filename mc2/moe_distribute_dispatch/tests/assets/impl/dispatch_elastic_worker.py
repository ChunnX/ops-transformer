#!/usr/bin/env python3

import json
import base64
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import torchair
from cann_ops_transformer_mc2moe.ops import MoeDistributeBuffer



def _to_torch(value, dtype):
    if value.size == 0:
        return None
    tensor = torch.from_numpy(value.copy())
    if dtype == "bfloat16":
        return tensor.view(torch.bfloat16).npu().contiguous()
    if dtype is not None:
        tensor = tensor.to(getattr(torch, dtype))
    return tensor.npu().contiguous()


def _serialize(tensor):
    if tensor is None:
        return np.zeros(0), (0,)
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy(), tuple(value.shape)


def main():
    rank = int(os.environ["MOE_ELASTIC_DISPATCH_RANK"])
    device_id = int(os.environ["MOE_ELASTIC_DISPATCH_DEVICE"])
    print(f"Moe elastic Dispatch worker stage=enter rank={rank}", flush=True)
    with open(os.environ["MOE_ELASTIC_DISPATCH_PLAN"], encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    print(f"Moe elastic Dispatch worker stage=plan-loaded rank={rank}", flush=True)

    attributes = plan["attributes"]
    world_size = int(attributes["epWorldSize"]) * int(attributes["tpWorldSize"])
    ep_size = int(attributes["epWorldSize"])
    tp_size = int(attributes["tpWorldSize"])
    torch_npu.npu.set_device(device_id)
    print(f"Moe elastic Dispatch worker stage=device-set rank={rank}", flush=True)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=os.environ["MOE_ELASTIC_DISPATCH_ENDPOINT"],
    )
    print(f"Moe elastic Dispatch worker stage=process-group rank={rank}", flush=True)
    ep_ranks_list = [
        list(range(tp_rank, world_size, tp_size)) for tp_rank in range(tp_size)
    ]
    tp_ranks_list = [
        list(range(ep_rank * tp_size, (ep_rank + 1) * tp_size))
        for ep_rank in range(ep_size)
    ]
    ep_group = None
    ep2_group = None
    tp_group = None
    for ranks in ep_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            ep_group = group
    ep2_group = None
    for ranks in ep_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            ep2_group = group
    dist.barrier()
    for ranks in tp_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            tp_group = group
    print(
        f"Moe elastic Dispatch worker stage=groups rank={rank} "
        f"ep={ep_group is not None} ep2={ep2_group is not None} "
        f"tp={tp_group is not None}",
        flush=True,
    )

    x = torch.frombuffer(
        base64.b64decode(plan["x"]), dtype=getattr(torch, plan["inputDtype"])
    ).clone()
    x = x.view(plan["xShape"])
    expert_ids = torch.frombuffer(
        base64.b64decode(plan["expertIds"]), dtype=torch.int32
    ).clone()
    expert_ids = expert_ids.view(plan["expertIdsShape"])
    x_active_mask = torch.frombuffer(
        base64.b64decode(plan["xActiveMask"]), dtype=torch.bool
    ).clone()
    x_active_mask = x_active_mask.view(plan["xActiveMaskShape"])
    elastic_info = torch.from_numpy(
        np.array(plan.get("elasticInfo", []), copy=True, dtype=np.int32)
    )
    scales = (
        torch.frombuffer(base64.b64decode(plan["scales"]), dtype=torch.float32).clone()
        if plan.get("scales")
        else None
    )
    if scales is not None:
        scales = scales.view(plan["scalesShape"])
    topk = int(expert_ids.shape[-1])
    hidden_size = int(x.shape[-1])
    input_dtype = x.dtype
    print(
        f"Moe elastic Dispatch worker inputs rank={rank} "
        f"x={x[:, 0].int().tolist()} topk_idx={expert_ids[:, :1].int().flatten().tolist()}",
        flush=True,
    )
    summary_dir = Path(os.environ.get("TTK_MOE_CACHE_DIR", ".")).expanduser()
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "dispatch_worker_inputs.log").open("a", encoding="utf-8") as f:
        f.write(
            f"rank={rank} x={x[:, 0].int().tolist()} "
            f"topk_idx={expert_ids[:, :1].int().flatten().tolist()}\n"
        )
    distribute_buffer = MoeDistributeBuffer(ep_group)
    print(f"Moe elastic Dispatch worker stage=buffer rank={rank}", flush=True)
    output = None
    with torchair.scope.limit_core_num(
        int(attributes.get("aicCoreNum", 24)),
        int(attributes.get("aivCoreNum", 48)),
    ):
        def run_dispatch(expert_ids_input, effective_elastic_info):
            return distribute_buffer.low_latency_dispatch(
                x=x.to(input_dtype).npu(),
                topk_idx=expert_ids_input.to(torch.int32).npu(),
                num_experts=int(attributes["moeExpertNum"]),
                quant_mode=int(attributes.get("quantMode", 0)),
                comm_alg=str(attributes.get("commAlg", "")),
                x_smooth_scale=(
                    scales.to(torch.float32).npu() if scales is not None else None
                ),
                x_active_mask=x_active_mask.to(torch.bool).npu(),
                elastic_info=(
                    effective_elastic_info.to(torch.int32).npu()
                    if effective_elastic_info is not None
                    and effective_elastic_info.numel()
                    else None
                ),
                expert_shard_type=int(attributes.get("expertShardType", 0)),
                shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
                shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
                expert_token_nums_type=int(attributes.get("expertTokenNumsType", 1)),
                zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
                copy_expert_num=int(attributes.get("copyExpertNum", 0)),
                const_expert_num=int(attributes.get("constExpertNum", 0)),
                num_max_dispatch_tokens_per_rank=(
                    int(attributes["globalBs"]) // ep_size
                ),
            )

        output = None
        available_ranks = set(plan["availableRanks"])
        has_elastic_info = elastic_info.numel() > 0
        if int(attributes.get("isElastic", 0)) == 1 or has_elastic_info:
            warmup_x = torch.empty(
                (1, hidden_size), dtype=input_dtype
            ).uniform_(-1024, 1024).npu()
            warmup_expert_ids = torch.arange(
                topk, dtype=torch.int32
            ).unsqueeze(0).npu()
            distribute_buffer.low_latency_dispatch(
                x=warmup_x,
                topk_idx=warmup_expert_ids,
                num_experts=int(attributes["moeExpertNum"]),
                quant_mode=2,
                comm_alg=str(attributes.get("commAlg", "")),
                num_max_dispatch_tokens_per_rank=(
                    int(attributes["globalBs"]) // ep_size
                ),
                expert_shard_type=int(attributes.get("expertShardType", 0)),
                shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
                shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
            )
            if rank in available_ranks:
                output = run_dispatch(expert_ids, elastic_info)

        if int(attributes.get("isExpand", 0)) == 1:
            distribute_buffer.update_ctx(ep2_group)
            expand_expert_ids = torch.frombuffer(
                base64.b64decode(plan["expertIdsExpand"]), dtype=torch.int32
            ).clone()
            expand_expert_ids = expand_expert_ids.view(plan["expertIdsExpandShape"])
            output = run_dispatch(expand_expert_ids, None)
        elif output is None:
            with open(plan["outputPath"], "wb"):
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        print(
            f"Moe elastic Dispatch worker stage=dispatch rank={rank} expand={int(attributes.get('isExpand', 0))}",
            flush=True,
        )
    torch.npu.synchronize()
    dist.destroy_process_group()
    print(
        "Moe elastic Dispatch worker output "
        f"rank={rank} expand_x={tuple(output[0].shape)} "
        f"first={output[0].flatten()[:4].tolist()} "
        f"abs_sum={float(output[0].float().abs().sum().item())}",
        flush=True,
    )

    tp_recv_counts = x.new_zeros((tp_size,), dtype=torch.int32)
    output = tuple(output[:5]) + (tp_recv_counts, None)
    serialized = [_serialize(item) for item in output]
    np.savez(
        plan["outputPath"],
        rank=np.array(rank, dtype=np.int32),
        output_0=serialized[0][0],
        output_1=serialized[1][0],
        output_2=serialized[2][0],
        output_3=serialized[3][0],
        output_4=serialized[4][0],
        output_5=serialized[5][0],
        output_6=serialized[6][0],
        output_shapes=np.array(
            [value for item in serialized for value in item[1]], dtype=np.int64
        ),
        output_dims=np.array([len(item[1]) for item in serialized], dtype=np.int64),
    )


if __name__ == "__main__":
    main()

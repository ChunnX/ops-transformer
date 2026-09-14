#!/usr/bin/env python3

import base64
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
import torchair
import numpy as np

_REPO_PACKAGE = (
    Path(__file__).resolve().parents[5] / "torch_extension" / "build" / "lib"
)
if _REPO_PACKAGE.is_dir():
    sys.path.insert(0, str(_REPO_PACKAGE))

from cann_ops_transformer_mc2moe.ops import MoeDistributeBuffer


def _tensor(plan, name):
    shape = plan.get(f"{name}Shape", [])
    if not shape:
        return None
    dtype = {
        "x": getattr(torch, plan["inputDtype"]),
        "expertIds": torch.int32,
        "expertScales": torch.float32,
        "xActiveMask": torch.bool,
    }[name]
    data = base64.b64decode(plan[name])
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def _optional_tensor(plan, name, dtype):
    shape = plan.get(f"{name}Shape", [])
    if not shape or not plan.get(name):
        return None
    data = base64.b64decode(plan[name])
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def _golden_expand_x(plan):
    data = plan.get("goldenExpandX")
    shape = plan.get("goldenExpandXShape", [])
    if not data or not shape:
        return None
    dtype = (
        torch.int8
        if int(plan["attributes"].get("quantMode", 0)) == 2
        else getattr(torch, plan["inputDtype"])
    )
    value = torch.frombuffer(
        bytearray(base64.b64decode(data)), dtype=torch.uint8
    ).clone()
    if dtype == torch.bfloat16 or dtype == torch.float16:
        value = value.view(torch.uint16)
    hidden_size = int(plan["xShape"][-1])
    return value.view(dtype).reshape(-1, hidden_size)


def _ccl_buffer_size(attributes, x, expert_ids):
    return MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
        int(attributes["epWorldSize"]),
        int(attributes["globalBs"]) // int(attributes["epWorldSize"]),
        int(x.shape[-1]),
        int(attributes["moeExpertNum"]),
        int(expert_ids.shape[-1]),
        int(attributes.get("sharedExpertNum", 0)),
        int(attributes.get("sharedExpertRankNum", 0)),
        str(attributes.get("commAlg", "")),
    )


def _core_num(attributes, name):
    value = attributes.get(name)
    return int(value) if value is not None else 24


def _run(buffer, plan, attributes, expert_ids, elastic_info, can_run):
    if not can_run:
        return None
    output = buffer.low_latency_dispatch(
        x=plan["x"],
        topk_idx=expert_ids,
        num_experts=int(attributes["moeExpertNum"]),
        comm_alg=str(attributes.get("commAlg", "")),
        x_smooth_scale=plan.get("scales"),
        x_active_mask=plan.get("xActiveMask"),
        elastic_info=elastic_info,
        expert_shard_type=int(attributes.get("expertShardType", 0)),
        shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
        shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
        expert_token_nums_type=int(attributes.get("expertTokenNumsType", 1)),
        quant_mode=int(attributes.get("quantMode", 0)),
        zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
        copy_expert_num=int(attributes.get("copyExpertNum", 0)),
        const_expert_num=int(attributes.get("constExpertNum", 0)),
        num_max_dispatch_tokens_per_rank=(
            int(attributes["globalBs"]) // int(attributes["epWorldSize"])
        ),
    )
    expand_x = output[0]
    if plan.get("useGoldenExpandX", False) and isinstance(
        plan.get("goldenExpandX"), torch.Tensor
    ):
        target_rows = expand_x.shape[0]
        golden_expand_x = plan["goldenExpandX"]
        if golden_expand_x.shape[0] < target_rows:
            padding = torch.zeros(
                (target_rows - golden_expand_x.shape[0], golden_expand_x.shape[1]),
                dtype=golden_expand_x.dtype,
                device=golden_expand_x.device,
            )
            golden_expand_x = torch.cat((golden_expand_x, padding))
        expand_x = golden_expand_x[:target_rows]
    if expand_x.dtype == torch.int8:
        expand_x = expand_x.to(getattr(torch, plan["inputDtype"]))
    return buffer.low_latency_combine(
        x=expand_x,
        topk_idx=expert_ids,
        topk_weights=plan["expertScales"],
        assist_info_for_combine=output[2],
        ep_send_counts=output[4],
        num_experts=int(attributes["moeExpertNum"]),
        comm_alg="",
        comm_quant_mode=int(attributes.get("commQuantMode", 0)),
        expand_scales=output[6],
        x_active_mask=plan.get("xActiveMask"),
        shared_expert_x=(
            plan["sharedExpertX"]
            if isinstance(plan.get("sharedExpertX"), torch.Tensor)
            else None
        ),
        elastic_info=elastic_info,
        ori_x=plan["oriX"] if isinstance(plan.get("oriX"), torch.Tensor) else None,
        const_expert_alpha_1=(
            plan["constExpertAlpha1"]
            if isinstance(plan.get("constExpertAlpha1"), torch.Tensor)
            else None
        ),
        const_expert_alpha_2=(
            plan["constExpertAlpha2"]
            if isinstance(plan.get("constExpertAlpha2"), torch.Tensor)
            else None
        ),
        const_expert_v=(
            plan["constExpertV"]
            if isinstance(plan.get("constExpertV"), torch.Tensor)
            else None
        ),
        zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
        copy_expert_num=int(attributes.get("copyExpertNum", 0)),
        const_expert_num=int(attributes.get("constExpertNum", 0)),
        expert_shared_type=int(attributes.get("expertShardType", 0)),
        shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
        shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
        num_max_dispatch_tokens_per_rank=(
            int(attributes["globalBs"]) // int(attributes["epWorldSize"])
        ),
    )


def main():
    rank = int(os.environ["MC2_EXTERNAL_CASCADE_RANK"])
    with open(os.environ["MC2_EXTERNAL_CASCADE_PLAN"], encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    attributes = plan["attributes"]
    ep_size = int(attributes["epWorldSize"])
    tp_size = int(attributes["tpWorldSize"])
    world_size = ep_size * tp_size
    device_id = int(plan["deviceId"])
    torch_npu.npu.set_device(device_id)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=os.environ["MC2_EXTERNAL_CASCADE_ENDPOINT"],
    )
    ep_ranks_list = [
        list(range(tp_id, world_size, tp_size)) for tp_id in range(tp_size)
    ]
    ep_group = None
    for ep_ranks in ep_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group = group

    for name in ("x", "expertScales", "xActiveMask"):
        value = _tensor(plan, name)
        if value is not None:
            plan[name] = value.npu()
    for name, dtype in (
        ("scales", torch.float32),
        ("sharedExpertX", getattr(torch, plan["inputDtype"])),
        ("oriX", getattr(torch, plan["inputDtype"])),
        ("constExpertAlpha1", getattr(torch, plan["inputDtype"])),
        ("constExpertAlpha2", getattr(torch, plan["inputDtype"])),
        ("constExpertV", getattr(torch, plan["inputDtype"])),
    ):
        value = _optional_tensor(plan, name, dtype)
        if value is not None:
            plan[name] = value.npu()
        else:
            plan[name] = None
    expert_ids = _tensor(plan, "expertIds").npu()
    golden_expand_x = _golden_expand_x(plan)
    if golden_expand_x is not None:
        plan["goldenExpandX"] = golden_expand_x.npu()
    elastic_info = _optional_tensor(plan, "elasticInfo", torch.int32)
    if elastic_info is not None:
        elastic_info = elastic_info.npu()

    buffer_size = _ccl_buffer_size(attributes, plan["x"], expert_ids)
    print(
        f"MC2 external cascade rank={rank} "
        f"buffer_size_mb={buffer_size} backend=kfc",
        flush=True,
    )
    buffer = MoeDistributeBuffer(ep_group, ccl_buffer_size=buffer_size)
    can_run = rank in set(plan["availableRanks"])
    with torchair.scope.limit_core_num(
        _core_num(attributes, "aicCoreNum"),
        _core_num(attributes, "aivCoreNum"),
    ):
        result = _run(buffer, plan, attributes, expert_ids, elastic_info, can_run)
    torch.npu.synchronize()
    if result is None:
        value = torch.empty(0, dtype=torch.uint8)
        shape = []
    else:
        value = result.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
            value = value.view(torch.uint16)
        shape = list(value.shape)
    np.savez(
        plan["outputPath"],
        rank=np.array(rank, dtype=np.int32),
        output_0=value.numpy(),
        output_shapes=np.array(shape, dtype=np.int64),
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import base64
import json
import os
import time

import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu
import torchair


from cann_ops_transformer_mc2moe.ops import MoeDistributeBuffer


def _tensor(encoded, dtype, shape):
    if isinstance(encoded, torch.Tensor):
        return encoded.reshape(shape)
    if not shape:
        return None
    if not encoded:
        return torch.empty(shape, dtype=dtype)
    value = torch.frombuffer(base64.b64decode(encoded), dtype=dtype).clone()
    return value.reshape(shape)


def _tensor_from_array(value, dtype):
    tensor = torch.from_numpy(value.copy())
    if tensor.dtype == torch.uint16 and dtype == torch.bfloat16:
        return tensor.view(dtype)
    return tensor.to(dtype)


def _int_tensor(value, dtype, shape):
    if not shape:
        return None
    return torch.tensor(value, dtype=dtype).reshape(shape)


def _ccl_buffer_size(attributes, expert_ids, expert_ids_expand, hidden_size):
    if isinstance(expert_ids_expand, str):
        expert_ids_expand = None
    if isinstance(expert_ids, str):
        expert_ids = None
    topk = max(
        expert_ids.shape[-1] if expert_ids is not None else 0,
        expert_ids_expand.shape[-1] if expert_ids_expand is not None else 0,
    )
    return MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
        int(attributes["epWorldSize"]),
        int(attributes["globalBs"]) // int(attributes["epWorldSize"]),
        hidden_size,
        int(attributes["moeExpertNum"]),
        topk,
        int(attributes.get("sharedExpertNum", 0)),
        int(attributes.get("sharedExpertRankNum", 0)),
        str(attributes.get("_dispatchCommAlg", "")),
    )


def _bytes(tensor):
    if tensor is None:
        return np.zeros(0)
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy()


def _pad_expand_x(expand_x, attributes, rank, expert_ids=None):
    if expand_x is None:
        return None
    ep_size = int(attributes["epWorldSize"])
    tp_size = int(attributes["tpWorldSize"])
    global_bs = int(attributes["globalBs"])
    top_k = int(
        attributes.get(
            "topK",
            expert_ids.shape[-1] if expert_ids is not None else 0,
        )
    )
    shared_expert_rank_num = int(attributes.get("sharedExpertRankNum", 0))
    rank_num_per_shared_expert = int(
        attributes.get("rankNumPerSharedExpert", 0)
    )
    local_moe_expert_num = int(attributes.get("localMoeExpertNum", 0))
    if top_k <= 0 or local_moe_expert_num <= 0:
        return expand_x
    per_rank_bs = global_bs // ep_size
    max_shared_group_num = (
        -(-ep_size // rank_num_per_shared_expert)
        if rank_num_per_shared_expert > 0
        else 0
    )
    shared_rows = per_rank_bs * max_shared_group_num
    moe_rows = global_bs * min(local_moe_expert_num, top_k)
    if (
        int(attributes.get("isElastic", 0))
        or int(attributes.get("shareBrokenCardNum", 0))
        or int(attributes.get("moeBrokenCardNum", 0))
    ):
        shared_rows = moe_rows = max(shared_rows, moe_rows)
    expected_rows = tp_size * (
        shared_rows if rank < shared_expert_rank_num else moe_rows
    )
    if expand_x.shape[0] >= expected_rows:
        return expand_x
    if expected_rows <= 0:
        return expand_x
    padding = torch.zeros(
        (expected_rows - expand_x.shape[0], expand_x.shape[1]),
        dtype=expand_x.dtype,
        device=expand_x.device,
    )
    return torch.cat((expand_x, padding))


def _warm_up_dispatch(
    buffer,
    attributes,
    hidden_size,
    input_dtype,
    expert_ids,
):
    top_k = int(attributes["topK"])
    warmup_local_bs = 1
    x = torch.empty((warmup_local_bs, hidden_size), dtype=input_dtype)
    x = x.uniform_(-1024, 1024).to(x.dtype).npu()
    x_active_mask = torch.ones(warmup_local_bs, dtype=torch.bool, device=x.device)
    return buffer.low_latency_dispatch(
        x=x,
        topk_idx=expert_ids,
        x_active_mask=x_active_mask,
        num_experts=int(attributes["moeExpertNum"]),
        comm_alg=attributes.get("_dispatchCommAlg", ""),
        quant_mode=2,
        shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
        shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
        num_max_dispatch_tokens_per_rank=warmup_local_bs,
    )


def main():
    rank = int(os.environ["MOE_DEEPEP_CASCADE_RANK"])
    device_id = int(os.environ["MOE_DEEPEP_CASCADE_DEVICE"])
    plan_path = Path(os.environ["MOE_DEEPEP_CASCADE_PLAN"])
    with plan_path.open(encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    tensor_dtypes = {
        "x": getattr(torch, plan["inputDtype"]),
        "expertIds": torch.int32,
        "expertScales": torch.float32,
        "xActiveMask": torch.bool,
        "rowMask": torch.bool,
        "sharedExpertX": getattr(torch, plan["inputDtype"]),
        "oriX": getattr(torch, plan["inputDtype"]),
        "constExpertAlpha1": getattr(torch, plan["inputDtype"]),
        "constExpertAlpha2": getattr(torch, plan["inputDtype"]),
        "constExpertV": getattr(torch, plan["inputDtype"]),
        "goldenExpandX": getattr(
            torch, plan.get("goldenExpandXDtype", plan["inputDtype"])
        ),
        "initialGoldenExpandX": getattr(
            torch, plan.get("initialGoldenExpandXDtype", plan["inputDtype"])
        ),
        "initialGoldenExpandIdx": torch.int32,
        "initialGoldenEpRecvCounts": torch.int32,
        "goldenExpandIdx": torch.int32,
        "goldenEpRecvCounts": torch.int32,
        "expertIdsExpand": torch.int32,
    }
    with np.load(plan_path.with_suffix(".npz"), allow_pickle=False) as tensors:
        for name, dtype in tensor_dtypes.items():
            if name in tensors:
                plan[name] = _tensor_from_array(tensors[name], dtype)
    attributes = plan["attributes"]
    available_ranks = set(
        plan.get("availableRanks", range(int(attributes["epWorldSize"])))
    )
    world_size = int(attributes["epWorldSize"]) * int(attributes["tpWorldSize"])
    ep_size = int(attributes["epWorldSize"])
    tp_size = int(attributes["tpWorldSize"])
    if not os.environ.get("HCCL_BUFFSIZE"):
        os.environ["HCCL_BUFFSIZE"] = str(
            _ccl_buffer_size(
                attributes,
                plan.get("expertIds"),
                plan.get("expertIdsExpand"),
                int(plan["xShape"][-1]),
            )
        )
    torch_npu.npu.set_device(device_id)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=os.environ["MOE_DEEPEP_CASCADE_ENDPOINT"],
    )
    ep_ranks_list = [
        list(range(tp_rank, world_size, tp_size)) for tp_rank in range(tp_size)
    ]
    tp_ranks_list = [
        list(range(ep_rank, ep_rank + tp_size)) for ep_rank in range(0, world_size, tp_size)
    ]
    ep_group = None
    ep2_group = None
    for ranks in ep_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            ep_group = group
    for ranks in ep_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            ep2_group = group
    for ranks in tp_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)

    ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    ep2_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    x = _tensor(plan["x"], getattr(torch, plan["inputDtype"]), plan["xShape"]).npu()
    expert_ids = _tensor(plan["expertIds"], torch.int32, plan["expertIdsShape"]).npu()
    output_expand_rows = max(
        tp_size * int(plan.get("expandXOutputRows", 0)),
        int(attributes["globalBs"])
        * min(int(attributes.get("localMoeExpertNum", 0)), int(plan["topK"])),
    )
    expert_scales = _tensor(
        plan["expertScales"], torch.float32, plan["expertScalesShape"]
    ).npu()
    x_active_mask = _tensor(
        plan["xActiveMask"], torch.bool, plan["xActiveMaskShape"]
    )
    if x_active_mask is not None:
        x_active_mask = x_active_mask.npu()
    shared_expert_x = _tensor(
        plan.get("sharedExpertX", ""),
        getattr(torch, plan["inputDtype"]),
        plan.get("sharedExpertXShape", []),
    )
    if shared_expert_x is not None:
        shared_expert_x = shared_expert_x.npu()
    ori_x = _tensor(plan["oriX"], getattr(torch, plan["inputDtype"]), plan["oriXShape"]).npu()
    row_mask = _tensor(
        plan.get("rowMask", ""),
        torch.bool,
        plan.get("rowMaskShape", []),
    )
    const_expert_alpha_1 = _tensor(
        plan.get("constExpertAlpha1", ""),
        getattr(torch, plan["inputDtype"]),
        plan.get("constExpertAlpha1Shape", []),
    )
    if const_expert_alpha_1 is not None:
        const_expert_alpha_1 = const_expert_alpha_1.npu()
    const_expert_alpha_2 = _tensor(
        plan.get("constExpertAlpha2", ""),
        getattr(torch, plan["inputDtype"]),
        plan.get("constExpertAlpha2Shape", []),
    )
    if const_expert_alpha_2 is not None:
        const_expert_alpha_2 = const_expert_alpha_2.npu()
    const_expert_v = _tensor(
        plan.get("constExpertV", ""),
        getattr(torch, plan["inputDtype"]),
        plan.get("constExpertVShape", []),
    )
    if const_expert_v is not None:
        const_expert_v = const_expert_v.npu()
    elastic_info = _int_tensor(
        plan.get("elasticInfo", ""),
        torch.int32,
        plan.get("elasticInfoShape", []),
    )
    final_golden_expand_x = _tensor(
        plan["goldenExpandX"],
        getattr(torch, plan.get("goldenExpandXDtype", plan["inputDtype"])),
        plan["goldenExpandXShape"],
    )
    if final_golden_expand_x is not None:
        final_golden_expand_x = final_golden_expand_x.npu()
    if final_golden_expand_x is None:
        final_golden_expand_x = torch.empty(
            (0, x.shape[-1]), dtype=x.dtype, device=x.device
        )
    initial_golden_expand_x = _tensor(
        plan.get("initialGoldenExpandX", ""),
        getattr(torch, plan.get("initialGoldenExpandXDtype", plan["inputDtype"])),
        plan.get("initialGoldenExpandXShape", []),
    )
    if initial_golden_expand_x is not None:
        initial_golden_expand_x = initial_golden_expand_x.npu()
    else:
        initial_golden_expand_x = final_golden_expand_x
    initial_golden_expand_idx = _tensor(
        plan.get("initialGoldenExpandIdx", ""),
        torch.int32,
        plan.get("initialGoldenExpandIdxShape", []),
    )
    if initial_golden_expand_idx is not None:
        initial_golden_expand_idx = initial_golden_expand_idx.npu()
    else:
        initial_golden_expand_idx = torch.empty(0, dtype=torch.int32, device=x.device)
    initial_golden_ep_recv_counts = _tensor(
        plan.get("initialGoldenEpRecvCounts", ""),
        torch.int32,
        plan.get("initialGoldenEpRecvCountsShape", []),
    )
    if initial_golden_ep_recv_counts is not None:
        initial_golden_ep_recv_counts = initial_golden_ep_recv_counts.npu()
    else:
        initial_golden_ep_recv_counts = torch.empty(
            0, dtype=torch.int32, device=x.device
        )
    golden_expand_idx = _tensor(
        plan.get("goldenExpandIdx", ""),
        torch.int32,
        plan.get("goldenExpandIdxShape", []),
    )
    if golden_expand_idx is not None:
        golden_expand_idx = golden_expand_idx.npu()
    else:
        golden_expand_idx = torch.empty(0, dtype=torch.int32, device=x.device)
    golden_ep_recv_counts = _tensor(
        plan.get("goldenEpRecvCounts", ""),
        torch.int32,
        plan.get("goldenEpRecvCountsShape", []),
    )
    if golden_ep_recv_counts is not None:
        golden_ep_recv_counts = golden_ep_recv_counts.npu()
    else:
        golden_ep_recv_counts = torch.empty(0, dtype=torch.int32, device=x.device)
    golden_expand_x = final_golden_expand_x
    expert_ids_expand = None
    if plan.get("isExpand") == 1 and plan.get("expertIdsExpand") is not None:
        expert_ids_expand = plan["expertIdsExpand"].npu()
    def dispatch_and_combine(
        expert_ids,
        use_golden_expand_x,
        elastic_info_input,
        suppressed=False,
    ):
        if suppressed:
            return None
        output = buffer.low_latency_dispatch(
            x=x,
            topk_idx=expert_ids,
            num_experts=int(attributes["moeExpertNum"]),
            comm_alg=attributes.get("_dispatchCommAlg")
            or "",
            x_active_mask=x_active_mask,
            elastic_info=elastic_info_input,
            expert_shard_type=int(attributes.get("expertShardType", 0)),
            shared_expert_num=int(attributes.get("sharedExpertNum", 0)),
            shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
            expert_token_nums_type=int(attributes.get("expertTokenNumsType", 1)),
            quant_mode=int(attributes.get("quantMode", 0)),
            zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
            copy_expert_num=int(attributes.get("copyExpertNum", 0)),
            const_expert_num=int(attributes.get("constExpertNum", 0)),
        num_max_dispatch_tokens_per_rank=max(
            int(attributes["globalBs"]) // int(attributes["epWorldSize"]), 1
        ),
        )
        print(
            f"[rank{rank}] dispatch metadata shapes="
            f"{[tuple(item.shape) for item in output]}",
            flush=True,
        )
        expand_x, _, expand_idx, _, ep_send_counts, expand_scales = output[:6]
        if int(attributes.get("quantMode", 0)) > 0 and expand_x.dtype != x.dtype:
            expand_x = expand_x.to(x.dtype)
        print(
            f"[rank{rank}] dispatch output shapes="
            f"{[tuple(value.shape) for value in output]} "
                    f"golden_expand_x={tuple(final_golden_expand_x.shape)}",
            flush=True,
        )
        if use_golden_expand_x:
            padded_golden_expand_x = _pad_expand_x(
                final_golden_expand_x, attributes, rank, expert_ids
            )
            if (
                int(attributes.get("quantMode", 0)) > 0
                and padded_golden_expand_x.dtype != x.dtype
            ):
                padded_golden_expand_x = padded_golden_expand_x.to(x.dtype)
            if padded_golden_expand_x.shape[0] > expand_x.shape[0]:
                padded_golden_expand_x = padded_golden_expand_x[
                    : expand_x.shape[0]
                ]
            if padded_golden_expand_x.shape[0] < expand_x.shape[0]:
                padding = torch.zeros(
                    (
                        expand_x.shape[0] - padded_golden_expand_x.shape[0],
                        expand_x.shape[1],
                    ),
                    dtype=padded_golden_expand_x.dtype,
                    device=padded_golden_expand_x.device,
                )
                padded_golden_expand_x = torch.cat(
                    (padded_golden_expand_x, padding)
            )
            expand_x = padded_golden_expand_x
        result = buffer.low_latency_combine(
            x=expand_x,
            topk_idx=expert_ids,
            topk_weights=expert_scales,
            assist_info_for_combine=expand_idx,
            ep_send_counts=ep_send_counts,
            num_experts=int(attributes["moeExpertNum"]),
            comm_quant_mode=int(attributes.get("commQuantMode", 0)),
            x_active_mask=x_active_mask,
            shared_expert_x=shared_expert_x,
            ori_x=x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            elastic_info=elastic_info_input,
            zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
            copy_expert_num=int(attributes.get("copyExpertNum", 0)),
            const_expert_num=int(attributes.get("constExpertNum", 0)),
            expert_shared_type=int(attributes.get("expertShardType", 0)),
            shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
            shared_expert_rank_num=int(attributes.get("sharedExpertRankNum", 0)),
            num_max_dispatch_tokens_per_rank=max(
                int(attributes["globalBs"]) // int(attributes["epWorldSize"]), 1
            ),
        )
        print(
            f"[rank{rank}] combine result shape={tuple(result.shape)}",
            flush=True,
        )
        return result

    def expand_dispatch_and_combine():
        output = buffer.low_latency_dispatch(
            x=x,
            topk_idx=expert_ids_expand,
            num_experts=int(attributes["moeExpertNum"]),
            comm_alg=attributes.get("_dispatchCommAlg", ""),
            x_active_mask=x_active_mask,
            elastic_info=None,
            expert_shard_type=int(attributes.get("expertShardType", 0)),
            shared_expert_num=int(attributes.get("sharedExpertNum", 0)),
            shared_expert_rank_num=int(
                attributes.get("sharedExpertRankNum", 0)
            ),
            expert_token_nums_type=int(
                attributes.get("expertTokenNumsType", 1)
            ),
            quant_mode=int(attributes.get("quantMode", 0)),
            zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
            copy_expert_num=int(attributes.get("copyExpertNum", 0)),
            const_expert_num=int(attributes.get("constExpertNum", 0)),
            num_max_dispatch_tokens_per_rank=max(
                int(attributes["globalBs"]) // int(attributes["epWorldSize"]), 1
            ),
        )
        print(
            f"[rank{rank}] expand dispatch metadata shapes="
            f"{[tuple(item.shape) for item in output]}",
            flush=True,
        )
        expand_x, _, expand_idx, _, ep_send_counts, expand_scales = output[:6]
        if int(attributes.get("quantMode", 0)) > 0 and expand_x.dtype != x.dtype:
            expand_x = expand_x.to(x.dtype)
        print(
            f"[rank{rank}] expand dispatch output shapes="
            f"{[tuple(value.shape) for value in output]} "
            f"golden_expand_x={tuple(golden_expand_x.shape)}",
            flush=True,
        )
        if bool(plan.get("useGoldenExpandX", False)):
            padded_golden_expand_x = _pad_expand_x(
                golden_expand_x, attributes, rank, expert_ids_expand
            )
            if (
                int(attributes.get("quantMode", 0)) > 0
                and padded_golden_expand_x.dtype != x.dtype
            ):
                padded_golden_expand_x = padded_golden_expand_x.to(x.dtype)
            if padded_golden_expand_x.shape[0] > expand_x.shape[0]:
                padded_golden_expand_x = padded_golden_expand_x[
                    : expand_x.shape[0]
                ]
            if padded_golden_expand_x.shape[0] < expand_x.shape[0]:
                padding = torch.zeros(
                    (
                        expand_x.shape[0] - padded_golden_expand_x.shape[0],
                        expand_x.shape[1],
                    ),
                    dtype=padded_golden_expand_x.dtype,
                    device=padded_golden_expand_x.device,
                )
                padded_golden_expand_x = torch.cat(
                    (padded_golden_expand_x, padding)
                )
            expand_x = padded_golden_expand_x
        result = buffer.low_latency_combine(
            x=expand_x,
            topk_idx=expert_ids_expand,
            topk_weights=expert_scales,
            assist_info_for_combine=expand_idx,
            ep_send_counts=ep_send_counts,
            num_experts=int(attributes["moeExpertNum"]),
            comm_quant_mode=int(attributes.get("commQuantMode", 0)),
            x_active_mask=x_active_mask,
            shared_expert_x=shared_expert_x,
            ori_x=x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            elastic_info=None,
            zero_expert_num=int(attributes.get("zeroExpertNum", 0)),
            copy_expert_num=int(attributes.get("copyExpertNum", 0)),
            const_expert_num=int(attributes.get("constExpertNum", 0)),
            expert_shared_type=int(attributes.get("expertShardType", 0)),
            shared_expert_num=int(attributes.get("sharedExpertNum", 1)),
            shared_expert_rank_num=int(
                attributes.get("sharedExpertRankNum", 0)
            ),
            num_max_dispatch_tokens_per_rank=max(
                int(attributes["globalBs"]) // int(attributes["epWorldSize"]), 1
            ),
            expand_scales=expand_scales,
        )
        print(
            f"[rank{rank}] expand combine result shape={tuple(result.shape)}",
            flush=True,
        )
        output_row_mask = None
        if x_active_mask is not None and x_active_mask.dim() == 1:
            output_row_mask = x_active_mask[: result.shape[0]].to(torch.bool)
        elif row_mask is not None and row_mask.shape[0] == result.shape[0]:
            output_row_mask = row_mask
        if output_row_mask is not None and output_row_mask.shape == result.shape[:1]:
            return result[output_row_mask]
        return result

    buffer = MoeDistributeBuffer(ep_group)
    if int(attributes.get("isElastic", 0)):
        _warm_up_dispatch(
            buffer,
            attributes,
            int(x.shape[-1]),
            x.dtype,
            torch.arange(
                int(attributes["topK"]), dtype=torch.int32, device=x.device
            ).unsqueeze(0),
        )
        print(f"[rank{rank}] warm-up complete", flush=True)
    result = None
    output_value = None
    output_shape = ()
    output_dims = 0
    suppressed = bool(attributes.get("_moeElasticSuppressed"))
    active_elastic_info = (
        elastic_info if int(attributes.get("isElastic", 0)) else None
    )
    if not int(attributes.get("isExpand", 0)) and rank in available_ranks:
        if expert_ids_expand is None:
            result = dispatch_and_combine(
                expert_ids,
                bool(plan.get("useGoldenExpandX", False)),
                active_elastic_info,
                suppressed,
            )
    if expert_ids_expand is not None:
        print(f"[rank{rank}] before update_ctx", flush=True)
        buffer.update_ctx(ep2_group)
        torch.npu.synchronize()
        print(f"[rank{rank}] after update_ctx", flush=True)
        result = expand_dispatch_and_combine()
        print(f"[rank{rank}] expand complete", flush=True)
        if int(attributes.get("isElastic", 0)):
            time.sleep(20)
    torch.npu.synchronize()
    if result is not None:
        output_value = _bytes(result)
        output_shape = tuple(result.shape)
        output_dims = result.dim()
    torch.npu.synchronize()
    if result is None:
        output_value = np.empty(0, dtype=np.uint8)
    np.savez(
        plan["outputPath"],
        rank=np.array(rank, dtype=np.int32),
        output_0=output_value,
        output_shapes=np.array(output_shape, dtype=np.int64),
        output_dims=np.array([output_dims], dtype=np.int64),
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

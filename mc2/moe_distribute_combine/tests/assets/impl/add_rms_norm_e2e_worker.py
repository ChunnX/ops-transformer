#!/usr/bin/env python3

import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import torchair


def load_input(data, name, dtype=None):
    value = data[name]
    if value.size == 0:
        return None
    tensor = torch.from_numpy(value.copy())
    if dtype == "bfloat16" or tensor.dtype == torch.uint16:
        return tensor.view(torch.bfloat16).npu().contiguous()
    if dtype is not None:
        tensor = tensor.to(getattr(torch, dtype))
    return tensor.npu().contiguous()


def _pad_first_dim(tensor, target_rows, hidden_size):
    if tensor.shape[0] >= target_rows:
        return tensor
    padding = torch.zeros(
        (target_rows - tensor.shape[0], hidden_size), dtype=tensor.dtype
    )
    return torch.cat([tensor, padding], dim=0)


def _pad_output_rows(tensor, target_rows):
    if tensor.ndim != 2 or tensor.shape[0] >= target_rows:
        return tensor
    padding = torch.zeros(
        (target_rows - tensor.shape[0], tensor.shape[1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=0)


def main():
    rank = int(os.environ["MOE_ADD_RMS_NORM_RANK"])
    device_id = int(os.environ["MOE_ADD_RMS_NORM_DEVICE"])
    with open(os.environ["MOE_ADD_RMS_NORM_PLAN"], encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    attributes = plan["attributes"]
    output_path = plan["output_path"]

    torch_npu.npu.set_device(device_id)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=int(attributes["epWorldSize"]) * int(attributes["tpWorldSize"]),
        init_method=os.environ["MOE_ADD_RMS_NORM_ENDPOINT"],
    )
    world_size = int(attributes["epWorldSize"]) * int(attributes["tpWorldSize"])
    ep_size = int(attributes["epWorldSize"])
    tp_size = int(attributes["tpWorldSize"])
    ep_group = None
    tp_group = None
    ep_ranks_list = [
        list(range(tp_id, world_size, tp_size)) for tp_id in range(tp_size)
    ]
    tp_ranks_list = [
        list(range(ep_id * tp_size, (ep_id + 1) * tp_size))
        for ep_id in range(ep_size)
    ]
    for ranks in ep_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            ep_group = group
    for ranks in tp_ranks_list:
        group = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            tp_group = group
    group_ep = str(ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(device_id))
    group_tp = str(tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(device_id))

    input_path = plan["input_path"]
    with np.load(input_path, allow_pickle=False) as input_data:
        def load_tensor(name, dtype, raw=False):
            value = np.array(input_data[name], copy=True)
            if value.size == 0:
                return None
            if raw:
                return torch.from_numpy(value)
            if dtype == torch.bfloat16:
                return torch.from_numpy(value).view(torch.bfloat16)
            return torch.from_numpy(value).to(dtype)

        expand_x = load_tensor("expand_x", torch.bfloat16)
        expert_ids = load_tensor("expert_ids", torch.int32)
        expand_idx = load_tensor("expand_idx", torch.int32)
        ep_send_counts = load_tensor("ep_send_counts", torch.int32)
        tp_send_counts = load_tensor("tp_send_counts", torch.int32)
        expert_scales = load_tensor("expert_scales", torch.float32)
        x_active_mask = load_tensor("x_active_mask", torch.bool)
        elastic_info = load_tensor("elastic_info", torch.int32)
        residual_x = load_tensor("residual_x", torch.bfloat16)
        gamma = load_tensor("gamma", None if False else torch.float32)
        shared_expert_x = load_tensor("shared_expert_x", torch.bfloat16)
        ori_x = load_tensor("ori_x", torch.bfloat16)
        const_expert_alpha_1 = load_tensor("const_expert_alpha_1", torch.bfloat16)
        const_expert_alpha_2 = load_tensor("const_expert_alpha_2", torch.bfloat16)
        const_expert_v = load_tensor("const_expert_v", torch.bfloat16)

    hidden_size = int(expand_x.shape[-1])
    is_elastic = int(attributes.get("isElastic", 0)) == 1
    x_active_mask = (
        x_active_mask.bool().npu()
        if int(attributes.get("isActivateMask", 0)) and x_active_mask is not None
        else None
    )
    target_rows = (
        int(attributes["globalBs"])
        * min(
            int(
                attributes.get(
                    "localMoeExpertNum",
                    int(attributes["moeExpertNum"])
                    // max(int(attributes["epWorldSize"]), 1),
                )
            ),
            int(expert_ids.shape[-1]),
        )
    )
    expand_x = _pad_first_dim(expand_x, target_rows, hidden_size)
    expand_idx = torch.cat(
        [expand_idx, torch.zeros(target_rows * 128 - expand_idx.numel(), dtype=expand_idx.dtype)]
    )
    combine_kwargs = {
        "expand_x": expand_x.npu(),
        "expert_ids": expert_ids.npu(),
        "assist_info_for_combine": expand_idx.npu(),
        "ep_send_counts": ep_send_counts.npu(),
        "tp_send_counts": tp_send_counts.npu() if tp_send_counts is not None else None,
        "expert_scales": expert_scales.float().npu(),
        "x_active_mask": x_active_mask,
        "shared_expert_x": shared_expert_x.npu() if shared_expert_x is not None else None,
        "elastic_info": elastic_info.reshape(-1).npu() if is_elastic and elastic_info is not None else None,
        "ori_x": ori_x.npu() if ori_x is not None else None,
        "const_expert_alpha_1": (
            const_expert_alpha_1.npu() if const_expert_alpha_1 is not None else None
        ),
        "const_expert_alpha_2": (
            const_expert_alpha_2.npu() if const_expert_alpha_2 is not None else None
        ),
        "const_expert_v": const_expert_v.npu() if const_expert_v is not None else None,
        "group_ep": group_ep,
        "group_tp": group_tp,
        "ep_rank_id": rank // tp_size,
        "tp_rank_id": rank % tp_size,
        "ep_world_size": int(attributes["epWorldSize"]),
        "tp_world_size": int(attributes["tpWorldSize"]),
        "expert_shard_type": int(attributes.get("expertShardType", 0)),
        "shared_expert_num": int(attributes.get("sharedExpertNum", 0)),
        "shared_expert_rank_num": int(attributes.get("sharedExpertRankNum", 0)),
        "moe_expert_num": int(attributes["moeExpertNum"]),
        "comm_quant_mode": int(attributes.get("commQuantMode", 0)),
        "global_bs": int(attributes["globalBs"]),
        "zero_expert_num": int(attributes.get("zeroExpertNum", 0)),
        "copy_expert_num": int(attributes.get("copyExpertNum", 0)),
        "const_expert_num": int(attributes.get("constExpertNum", 0)),
    }
    hidden_size = int(combine_kwargs["expand_x"].shape[-1])
    expert_num_per_token = int(combine_kwargs["expert_ids"].shape[-1])
    bs_local = int(combine_kwargs["expert_ids"].shape[0])
    if residual_x is None:
        residual_x = torch.zeros((bs_local, hidden_size), dtype=torch.bfloat16)
    elif residual_x.numel() == bs_local * hidden_size:
        residual_x = residual_x.reshape(bs_local, hidden_size)
        if residual_x.dtype != torch.bfloat16:
            residual_x = residual_x.to(torch.bfloat16)
    else:
        raise ValueError(
            f"AddRmsNorm residual_x size {residual_x.numel()} does not match "
            f"{bs_local} x {hidden_size}"
        )
    residual_x = residual_x.npu()
    gamma = (
        gamma.reshape(hidden_size).to(combine_kwargs["expand_x"].dtype).npu()
        if gamma is not None
        else torch.ones((hidden_size,), dtype=combine_kwargs["expand_x"].dtype).npu()
    )
    norm_eps = float(attributes.get("normEps", 1e-6))
    actual_kwargs = dict(combine_kwargs)
    actual_kwargs.update(
        {
            "expand_idx": actual_kwargs.pop("assist_info_for_combine"),
            "residual_x": residual_x.view(-1, 1, hidden_size),
            "gamma": gamma,
            "norm_eps": norm_eps,
        }
    )
    if is_elastic:
        warmup_x = (
        torch.empty((1, hidden_size), dtype=combine_kwargs["expand_x"].dtype)
        .uniform_(-1024, 1024)
        .npu()
    )
        warmup_expert_ids = (
        torch.arange(expert_num_per_token, dtype=torch.int32).unsqueeze(0).npu()
    )
        torch_npu.npu_moe_distribute_dispatch_v2(
        x=warmup_x,
        expert_ids=warmup_expert_ids,
        x_active_mask=None,
        group_ep=combine_kwargs["group_ep"],
        group_tp=combine_kwargs["group_tp"],
        ep_rank_id=combine_kwargs["ep_rank_id"],
        tp_rank_id=combine_kwargs["tp_rank_id"],
        ep_world_size=combine_kwargs["ep_world_size"],
        tp_world_size=combine_kwargs["tp_world_size"],
        expert_shard_type=combine_kwargs["expert_shard_type"],
        shared_expert_num=combine_kwargs["shared_expert_num"],
        shared_expert_rank_num=combine_kwargs["shared_expert_rank_num"],
        moe_expert_num=combine_kwargs["moe_expert_num"],
        scales=None,
        quant_mode=2,
        global_bs=combine_kwargs["ep_world_size"],
    )
        torch.npu.synchronize()

    if rank not in set(plan.get("availableRanks", range(world_size))):
        dist.destroy_process_group()
        return

    summary = {
        name: (tuple(value.shape), str(value.dtype), float(value.float().sum()))
        for name, value in actual_kwargs.items()
        if torch.is_tensor(value)
    }
    print(
        f"[AddRmsNormV2 rank={rank} input-summary] "
        f"{json.dumps(summary, sort_keys=True)}",
        flush=True,
    )
    actual_output = torch_npu.npu_moe_distribute_combine_add_rms_norm(
        **actual_kwargs
    )
    actual_output = (
        actual_output[0],
        torch.where(
            torch.all((actual_output[2] >= 0) & (actual_output[2] <= 1), dim=-1, keepdim=True),
            actual_output[1].new_full(actual_output[1].shape, 0.0),
            actual_output[1],
        ),
        actual_output[2],
    )
    torch.npu.synchronize()

    is_masked = (
        int(attributes.get("isActivateMask", 0)) == 1
        and int(attributes.get("xActiveMaskDim", 0)) == 1
    )
    is_zero_compute_expert = any(
        int(attributes.get(name, 0)) != 0
        for name in ("zeroExpertNum", "copyExpertNum", "constExpertNum")
    )
    if is_masked and not is_zero_compute_expert:
        row_mask = combine_kwargs["x_active_mask"].reshape(-1).to(torch.bool)
        combine_kwargs["expert_ids"] = combine_kwargs["expert_ids"][row_mask]
        combine_kwargs["expert_scales"] = combine_kwargs["expert_scales"][row_mask]
        combine_kwargs["shared_expert_x"] = (
            _pad_output_rows(combine_kwargs["shared_expert_x"][row_mask], bs_local)
            if combine_kwargs["shared_expert_x"] is not None
            else None
        )
        combine_kwargs["ori_x"] = (
            _pad_output_rows(combine_kwargs["ori_x"][row_mask], bs_local)
            if combine_kwargs["ori_x"] is not None
            else None
        )
        combine_kwargs["x_active_mask"] = None
        if combine_kwargs["shared_expert_x"] is not None:
            combine_kwargs["shared_expert_x"] = (
                combine_kwargs["shared_expert_x"].view(-1, 1, hidden_size)
            )
        if combine_kwargs["ori_x"] is not None:
            combine_kwargs["ori_x"] = combine_kwargs["ori_x"].view(
                -1, 1, hidden_size
            )
        residual_x = residual_x[row_mask]

    x1 = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
    x1 = x1.view(-1, 1, hidden_size)
    residual_x = residual_x.view(-1, 1, hidden_size)
    reference_output = torch_npu.npu_add_rms_norm(
        x1, residual_x, gamma, norm_eps
    )
    if is_masked and not is_zero_compute_expert:
        actual_output = tuple(
            value.reshape(-1, *value.shape[2:]) for value in actual_output
        )
        reference_output = tuple(
            _pad_output_rows(
                value.reshape(-1, *value.shape[2:]),
                bs_local,
            )
            for value in reference_output
        )
    if bool(int(attributes.get("_moeVariableBs", 0))):
        actual_output = tuple(
            _pad_output_rows(value.view(-1, value.shape[-1]), target_rows)
            if value.ndim >= 2
            else _pad_output_rows(value.view(-1, 1), target_rows)
            for value in actual_output
        )
        reference_output = tuple(
            _pad_output_rows(value.view(-1, value.shape[-1]), target_rows)
            if value.ndim >= 2
            else _pad_output_rows(value.view(-1, 1), target_rows)
            for value in reference_output
        )
    torch.npu.synchronize()

    def serialize_outputs(items):
        outputs = []
        for item in items:
            value = item.detach().cpu().contiguous()
            if value.dtype == torch.bfloat16:
                value = value.view(torch.uint16)
            outputs.append(value.numpy())
        return outputs

    actual_outputs = serialize_outputs(actual_output)
    reference_outputs = serialize_outputs(reference_output)
    if [item.shape for item in actual_outputs] != [
        item.shape for item in reference_outputs
    ]:
        raise RuntimeError(
            "Moe AddRmsNorm V2 actual/reference output shapes mismatch"
        )

    np.savez(
        output_path,
        rank=np.array(rank, dtype=np.int32),
        actual_0=actual_outputs[0],
        actual_1=actual_outputs[1],
        actual_2=actual_outputs[2],
        reference_0=reference_outputs[0],
        reference_1=reference_outputs[1],
        reference_2=reference_outputs[2],
        output_shapes=np.array(
            [item.shape for item in actual_outputs], dtype=np.int64
        ),
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import hashlib
import json
import os
import random
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

_moe_case_cache_lock = threading.Lock()


@dataclass(frozen=True)
class MoeCaseConfig:
    global_bs: int
    hidden_size: int
    top_k: int
    seed: int
    ep_world_size: int
    tp_world_size: int
    moe_expert_num: int
    local_moe_expert_num_override: int = 0
    shared_expert_num: int = 0
    shared_expert_rank_num: int = 0
    const_expert_num: int = 0
    zero_expert_num: int = 0
    copy_expert_num: int = 0
    rank_num_per_shared_expert: int = 0
    rank_expert_nums_override: tuple = ()
    input_dtype: str = "bfloat16"
    is_mask: bool = False
    is_variable_bs: bool = False
    is_quant: bool = False
    is_dispatch_scales: bool = False
    is_topk_balance: bool = False
    is_variable_local_moe_expert_num: bool = False
    local_moe_expert_nums_override: tuple = ()
    is_deepep: bool = False
    is_x_random: bool = False
    is_elastic: bool = False
    elastic_shared_rank_num: int = 0
    elastic_moe_rank_num: int = 0
    comm_quant_mode: int = 0
    is_combine_x: bool = False
    is_shared_expert_x: bool = False
    is_add_rms_norm: bool = False
    is_expand: bool = False
    active_mask_dim: int = 0

    @property
    def local_moe_expert_num(self):
        if self.local_moe_expert_num_override:
            return self.local_moe_expert_num_override
        if self.ep_world_size <= self.shared_expert_rank_num:
            return 0
        return self.moe_expert_num // max(
            self.ep_world_size - self.shared_expert_rank_num, 1
        )

    @property
    def local_moe_expert_nums(self):
        if self.local_moe_expert_nums_override:
            return self.local_moe_expert_nums_override
        moe_rank_num = (
            self.ep_world_size - self.shared_expert_rank_num
        )
        min_local_moe_expert_num = (
            1 if self.is_variable_local_moe_expert_num else self.local_moe_expert_num
        )
        # The reference generator always consumes RNG here, even when the fixed
        # local-expert range produces one possible value.
        if self.tp_world_size == 1:
            return tuple(
            torch.randint(
                min_local_moe_expert_num,
                self.local_moe_expert_num + 1,
                (moe_rank_num,),
                ).tolist()
            )
        return tuple(
            tuple(
                torch.randint(
                    min_local_moe_expert_num,
                    self.local_moe_expert_num + 1,
                    (moe_rank_num,),
                ).tolist()
            )
            for _ in range(self.tp_world_size)
        )

    @property
    def rank_expert_nums(self):
        if self.rank_expert_nums_override:
            return self.rank_expert_nums_override
        local_expert_nums = (
            self.local_moe_expert_num_list[0]
            if self.tp_world_size == 1
            else self.local_moe_expert_num_list
        )
        return (1,) * self.shared_expert_rank_num + tuple(local_expert_nums)

    @property
    def cached_rank_expert_nums(self):
        return self.rank_expert_nums

    @property
    def local_moe_expert_num_list(self):
        if self.tp_world_size == 1:
            return [self.local_moe_expert_nums]
        return list(self.local_moe_expert_nums)

    @property
    def is_shared(self):
        return self.shared_expert_rank_num > 0

    @property
    def special_expert_num(self):
        if self.is_expand:
            return self.moe_expert_num + (
                self.zero_expert_num + self.copy_expert_num + self.const_expert_num
            )
        return (
            self.elastic_moe_expert_num
            + self.zero_expert_num
            + self.copy_expert_num
            + self.const_expert_num
        )

    @property
    def elastic_moe_expert_num(self):
        return (
            len(self.available_ranks) - self.elastic_shared_expert_rank_num
        ) * self.local_moe_expert_num

    @property
    def expand_moe_expert_num(self):
        if not self.is_expand:
            return self.moe_expert_num
        return self.moe_expert_num + (
            self.zero_expert_num + self.copy_expert_num + self.const_expert_num
        )

    @property
    def elastic_shared_expert_rank_num(self):
        if not self.is_elastic:
            return self.shared_expert_rank_num
        if (
            self.shared_expert_num == 0
            or not self.elastic_shared_rank_num
            or self.elastic_shared_rank_num % self.shared_expert_num
            or self.elastic_shared_rank_num == self.shared_expert_rank_num
        ):
            return self.shared_expert_rank_num
        return self.shared_expert_rank_num - self.elastic_shared_rank_num

    @property
    def elastic_rank_num_per_shared_expert(self):
        if not self.is_elastic:
            return self.rank_num_per_shared_expert
        return self.elastic_shared_expert_rank_num // max(self.shared_expert_num, 1)

    @property
    def available_ranks(self):
        if not self.is_elastic:
            return list(range(self.ep_world_size))
        shared_ranks = list(range(self.shared_expert_rank_num))
        moe_ranks = list(range(self.shared_expert_rank_num, self.ep_world_size))
        if self.elastic_shared_rank_num:
            random.seed(self.seed)
            for rank in random.sample(shared_ranks, self.elastic_shared_rank_num):
                shared_ranks.remove(rank)
            random.seed(self.seed)
        if self.elastic_moe_rank_num:
            random.seed(self.seed)
            for rank in random.sample(moe_ranks, self.elastic_moe_rank_num):
                moe_ranks.remove(rank)
        return sorted(shared_ranks + moe_ranks)

    @property
    def elastic_info(self):
        info = torch.zeros(4 + 2 * self.ep_world_size, dtype=torch.int32)
        available_ranks = self.available_ranks
        info[0] = int(self.is_elastic)
        info[1] = len(available_ranks) // self.tp_world_size
        info[2] = self.elastic_shared_expert_rank_num
        info[3] = self.elastic_moe_expert_num
        table1 = [-1] * self.ep_world_size
        table2 = [-1] * self.ep_world_size
        for local_rank_id, ep_rank_id in enumerate(available_ranks):
            table1[ep_rank_id] = local_rank_id
            table2[local_rank_id] = ep_rank_id
        info[4 : 4 + self.ep_world_size] = torch.tensor(table1, dtype=torch.int32)
        info[4 + self.ep_world_size :] = torch.tensor(table2, dtype=torch.int32)
        return info

    def __post_init__(self):
        if self.shared_expert_num > 0 and self.rank_num_per_shared_expert <= 0:
            raise ValueError(
                "sharedExpertNum > 0 requires sharedExpertRankNum and rankNumPerSharedExpert > 0"
            )


def _dtype(config):
    return torch.bfloat16 if config.input_dtype == "bfloat16" else torch.float16


def _shared_expert_send_card_ids(config, shared_rank_id):
    card_id_in_shared_group = shared_rank_id % config.rank_num_per_shared_expert
    return range(
        card_id_in_shared_group,
        config.ep_world_size,
        config.rank_num_per_shared_expert,
    )


def _binary_tree_sum(tensor):
    if tensor.numel() == 1:
        return tensor
    if tensor.numel() % 2:
        tensor = torch.cat((tensor, torch.zeros(1, dtype=tensor.dtype)))
    return _binary_tree_sum(tensor.view(-1, 2).sum(dim=1))


def _comm_quant_round_trip(tensor, config):
    if config.comm_quant_mode == 0:
        return tensor
    if tensor.numel() == 0:
        return tensor

    quant_dtype = torch.int8 if config.comm_quant_mode == 2 else torch.int16
    quant_dtype_max = 127.0 if config.comm_quant_mode == 2 else 2047.0
    element_per_block = 128 if config.comm_quant_mode == 2 else 8
    token_pad_length = (
        element_per_block - (tensor.size(1) % element_per_block)
    ) % element_per_block
    padded_tensor = torch.cat(
        (
            tensor,
            tensor.new_zeros(tensor.size(0), token_pad_length),
        ),
        dim=1,
    )
    blocks = padded_tensor.reshape(-1, element_per_block).float()
    blocks_max, _ = blocks.abs().max(dim=-1, keepdim=True)
    blocks_scales = blocks_max / quant_dtype_max
    blocks_scales[blocks_scales == 0.0] = 1.0
    blocks_quant = (blocks / blocks_scales).to(torch.float16).round().to(quant_dtype)
    blocks_dequant = (
        blocks_quant.to(torch.float16).to(torch.float32)
        * blocks_scales.to(_dtype(config)).to(torch.float32)
    ).to(_dtype(config))
    return (
        blocks_dequant.reshape(tensor.size(0), -1)[
            :, : tensor.size(1)
        ].reshape(tensor.shape)
    )


def _quantize_dispatch_values(values, config):
    values = values.float()
    output_scale = values.abs().max(dim=-1, keepdim=True).values / 127.0
    output_scale[output_scale == 0.0] = 1.0
    quant_values = (values / output_scale).round().to(torch.int8)
    return quant_values, output_scale


def _dequantize_dispatch_values(tensor, config):
    if not config.is_quant:
        return tensor
    return tensor.to(_dtype(config))


def _split_rank(tensor, config, ep_rank_id, tp_rank_id=0):
    if config.tp_world_size == 1:
        return tensor[ep_rank_id]
    return tensor[tp_rank_id][ep_rank_id]


def _apply_rank_mapping(tensor, config):
    if not config.is_elastic:
        return tensor
    if tensor is None:
        return None
    tensor = tensor.reshape(-1)
    expected_size = config.ep_world_size * max(config.local_moe_expert_num, 1)
    if tensor.numel() < expected_size:
        tensor = torch.cat(
            (
                tensor,
                torch.full(
                    (expected_size - tensor.numel(),),
                    int(tensor[-1]) if tensor.numel() else 0,
                    dtype=tensor.dtype,
                ),
            )
        )
    mapped = tensor.clone()
    rank_stride = max(config.rank_num_per_shared_expert, 1)
    rank_positions = torch.arange(mapped.numel(), dtype=torch.int64) % rank_stride == 0
    local_rank_ids = (mapped[rank_positions] / rank_stride).to(torch.int64)
    available_ranks = torch.tensor(config.available_ranks, dtype=mapped.dtype)
    local_rank_ids = local_rank_ids.clamp(max=available_ranks.numel() - 1)
    mapped[rank_positions] = available_ranks[local_rank_ids]
    return mapped


def _generate_case(config):
    has_special_expert = any(
        count > 0
        for count in (
            config.zero_expert_num,
            config.copy_expert_num,
            config.const_expert_num,
        )
    )
    active_mask_dim = config.active_mask_dim
    shared_ranks = list(range(config.shared_expert_rank_num))
    moe_ranks = list(range(config.shared_expert_rank_num, config.ep_world_size))
    if config.is_elastic and config.elastic_shared_rank_num:
        random.seed(config.seed)
        for rank in random.sample(shared_ranks, config.elastic_shared_rank_num):
            shared_ranks.remove(rank)
    if config.is_elastic and config.elastic_moe_rank_num:
        random.seed(config.seed)
        for rank in random.sample(moe_ranks, config.elastic_moe_rank_num):
            moe_ranks.remove(rank)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    local_moe_expert_num_list = config.local_moe_expert_num_list
    x_list = []
    expert_ids_list = []
    x_active_mask_list = []
    x_active_mask_elastic_list = []
    x_active_mask_input_list = []
    x_active_compute_mask_list = []
    x_active_compute_expand_mask_list = []
    x_active_compute_initial_mask_list = []
    scales_list = []
    padded_scales_list = []
    shared_expert_x_list = []
    expert_ids_expand_list = []

    def generate_expert_ids(expert_num):
        if config.is_topk_balance:
            blocks = []
            block_count = config.global_bs * config.top_k // expert_num + 1
            for round_index in range(block_count):
                blocks.append(
                    torch.arange(
                        round_index,
                        round_index + expert_num,
                        dtype=torch.int32,
                    )
                    % expert_num
                )
            return torch.cat(blocks)[: config.global_bs * config.top_k].view(
                config.global_bs, config.top_k
            )

        blocks = []
        for _ in range(config.global_bs):
            blocks.append(torch.randperm(expert_num, dtype=torch.int32)[: config.top_k])
        return torch.stack(blocks)

    per_rank_bs = max(config.global_bs // config.ep_world_size, 1)
    for tp_rank_id in range(config.tp_world_size):
        has_special_expert = any(
            count > 0
            for count in (
                config.zero_expert_num,
                config.copy_expert_num,
                config.const_expert_num,
            )
        )
        local_moe_expert_nums = local_moe_expert_num_list[tp_rank_id]
        expert_ids = generate_expert_ids(
            config.elastic_moe_expert_num
            + config.zero_expert_num
            + config.copy_expert_num
            + config.const_expert_num
        )
        if config.is_elastic:
            expert_ids[expert_ids >= config.elastic_moe_expert_num] += (
                config.moe_expert_num
                - config.elastic_moe_expert_num
            )
        expert_ids_expand = generate_expert_ids(
            config.moe_expert_num
            + config.zero_expert_num
            + config.copy_expert_num
            + config.const_expert_num
        )
        expert_ids_expand_list.append(expert_ids_expand)

        masks = []
        for _ in range(config.ep_world_size):
            has_token_count = config.is_mask or config.is_variable_bs
            if has_token_count:
                min_real_token_num = 0 if config.is_mask and active_mask_dim == 1 else 1
                real_token_num = torch.randint(
                    min_real_token_num, per_rank_bs + 1, (1,)
                ).item()
                masks.append(
                    torch.cat(
                        (
                            torch.ones(real_token_num),
                            torch.zeros(per_rank_bs - real_token_num),
                        )
                    ).bool()
                )
            else:
                masks.append(torch.ones(per_rank_bs, dtype=torch.bool))

        if config.is_variable_bs:
            upper_bs_card_ep_id = torch.randint(0, config.ep_world_size, (1,)).item()
            masks[upper_bs_card_ep_id] = torch.ones(per_rank_bs, dtype=torch.bool)
        shared_expert_x = []
        if config.is_shared_expert_x:
            for rank_id in range(config.ep_world_size):
                rows = (
                    per_rank_bs
                    if active_mask_dim == 2
                    else int(masks[rank_id].sum())
                )
                shared_expert_x.append(
                    torch.empty(
                        rows, config.hidden_size, dtype=_dtype(config)
                    ).uniform_(0, 1)
                )
        shared_expert_x_list.append(shared_expert_x)
        if active_mask_dim == 2:
            for mask_id, mask in enumerate(masks):
                true_indices = torch.nonzero(mask, as_tuple=False).flatten()
                expanded_mask = torch.zeros(
                    per_rank_bs, max(config.top_k, 1), dtype=torch.bool
                )
                if true_indices.numel() > 0:
                    expanded_mask[true_indices] = (
                        torch.rand(true_indices.numel(), config.top_k) > 0.5
                    )
                masks[mask_id] = expanded_mask
        x_active_mask_input = torch.cat(masks)
        x_active_mask = x_active_mask_input.clone()
        if config.is_elastic:
            for rank_id in range(config.ep_world_size):
                if rank_id not in config.available_ranks:
                    x_active_mask[
                        rank_id * per_rank_bs : (rank_id + 1) * per_rank_bs
                    ] = False
        x_active_mask_expand = (
            x_active_mask_input.clone()
            if not config.is_expand
            else x_active_mask_input.clone()
        )
        x_active_mask_ela_ori = x_active_mask_input.clone()

        has_special_expert = any(
            count > 0
            for count in (
                config.zero_expert_num,
                config.copy_expert_num,
                config.const_expert_num,
            )
        )
        if has_special_expert:
            compute_mask_base = (
                x_active_mask_input if config.is_expand else x_active_mask
            )
            x_active_compute_mask = (
                compute_mask_base.reshape(-1, 1).repeat(1, config.top_k)
                if compute_mask_base.dim() == 1
                else compute_mask_base.clone()
            )
            source_expert_ids = (
                expert_ids_expand if config.is_expand else expert_ids
            )
            x_active_compute_mask[source_expert_ids >= config.moe_expert_num] = False
            x_active_compute_expand_mask = (
                x_active_mask_expand.reshape(-1, 1).repeat(1, config.top_k)
                if x_active_mask_expand.dim() == 1
                else x_active_mask_expand.clone()
            )
            x_active_compute_expand_mask[
                expert_ids_expand >= config.moe_expert_num
            ] = False
        else:
            x_active_compute_mask = (
                x_active_mask_expand if config.is_expand else x_active_mask
            )
            x_active_compute_expand_mask = x_active_mask_expand
        if has_special_expert:
            x_active_compute_initial_mask = (
                x_active_mask.reshape(-1, 1).repeat(1, config.top_k)
                if x_active_mask.dim() == 1
                else x_active_mask.clone()
            )
            x_active_compute_initial_mask[expert_ids >= config.moe_expert_num] = False
        else:
            x_active_compute_initial_mask = x_active_mask

        if config.is_x_random:
            x = torch.empty(
                config.global_bs, config.hidden_size, dtype=_dtype(config)
            ).uniform_(-1024, 1024)
        else:
            x = (
                torch.arange(1, config.global_bs + 1, dtype=_dtype(config))
                .view(-1, 1)
                .repeat(1, config.hidden_size)
            )
        if config.is_quant and config.is_dispatch_scales:
            scales = torch.empty(
                config.shared_expert_num + config.moe_expert_num, config.hidden_size
            ).uniform_(-1, 1)
            shared_scales = scales[: config.shared_expert_num]
            if config.shared_expert_rank_num:
                shared_scales = shared_scales.repeat_interleave(
                    config.rank_num_per_shared_expert, dim=0
                )
            padded_scales = torch.cat(
                (shared_scales, scales[config.shared_expert_num :])
            )
        else:
            scales = None
            padded_scales = None

        x_list.append(x)
        expert_ids_list.append(expert_ids)
        x_active_mask_elastic_list.append(x_active_mask)
        x_active_mask_list.append(x_active_mask_input)
        x_active_mask_input_list.append(x_active_mask_input.clone())
        x_active_compute_mask_list.append(x_active_compute_mask)
        x_active_compute_expand_mask_list.append(x_active_compute_expand_mask)
        x_active_compute_initial_mask_list.append(x_active_compute_initial_mask)
        scales_list.append(scales)
        padded_scales_list.append(padded_scales)

    return {
        "x_list": x_list,
        "expert_ids_list": expert_ids_list,
        "expert_ids_expand_list": [
            [chunk.contiguous() for chunk in expert_ids.chunk(config.ep_world_size)]
            for expert_ids in expert_ids_expand_list
        ],
        "x_rank_list": [chunk.contiguous() for x in x_list for chunk in x.chunk(config.ep_world_size)],
        "expert_ids_rank_list": [
            chunk.contiguous()
            for expert_ids in expert_ids_list
            for chunk in expert_ids.chunk(config.ep_world_size)
        ],
        "expert_ids_rank_expand_list": [
            chunk.contiguous()
            for expert_ids in expert_ids_expand_list
            for chunk in expert_ids.chunk(config.ep_world_size)
        ],
        "x_active_mask_list": x_active_mask_list,
        "x_active_mask_elastic_list": x_active_mask_elastic_list,
        "x_active_mask_input_list": x_active_mask_input_list,
        "x_active_compute_mask_list": x_active_compute_mask_list,
        "x_active_compute_expand_mask_list": x_active_compute_expand_mask_list,
        "x_active_compute_initial_mask_list": x_active_compute_initial_mask_list,
        "scales_list": scales_list,
        "padded_scales_list": padded_scales_list,
        "local_moe_expert_nums": local_moe_expert_num_list,
        "rank_expert_nums": ((1,) * config.shared_expert_rank_num + tuple(local_moe_expert_num_list[0] if config.tp_world_size == 1 else local_moe_expert_num_list)),
        "shared_expert_x_list": (
            shared_expert_x_list
            if shared_expert_x_list
            else [
                torch.empty(0, dtype=_dtype(config))
                for _ in range(config.ep_world_size)
            ]
        ),
        "per_rank_bs": per_rank_bs,
    }


def _generate_dispatch_golden(data, config, initial_elastic=False):
    local_moe_expert_num_list = data["local_moe_expert_nums"]
    is_initial_elastic = config.is_elastic and initial_elastic

    def shared_source_cards(shared_rank_id):
        use_elastic = is_initial_elastic
        if not use_elastic:
            return _shared_expert_send_card_ids(config, shared_rank_id)
        rank_num_per_shared_expert = config.elastic_rank_num_per_shared_expert
        card_id_in_shared_group = shared_rank_id % max(rank_num_per_shared_expert, 1)
        local_card_ids = range(
            card_id_in_shared_group,
            len(config.available_ranks),
            rank_num_per_shared_expert,
        )
        return [config.available_ranks[local_card_id] for local_card_id in local_card_ids]

    def dispatch_shared_source_cards(shared_rank_id):
        return _shared_expert_send_card_ids(config, shared_rank_id)

    expand_x_list = []
    expand_idx_list = []
    tokens_list = []
    ep_recv_count_list = []
    dynamic_scales_list = []
    middle_idx_tp_list = []
    special_expert_count = (
        config.elastic_moe_expert_num
        + config.zero_expert_num
        + config.copy_expert_num
        + config.const_expert_num
        if is_initial_elastic
        else (
            config.moe_expert_num
            + config.zero_expert_num
            + config.copy_expert_num
            + config.const_expert_num
        )
    )
    shared_rank_num = config.shared_expert_rank_num
    rank_slot_ids_length = (
        config.elastic_shared_expert_rank_num
        if is_initial_elastic
        else config.shared_expert_rank_num
    )
    output_slot_num = shared_rank_num
    moe_rank_num = config.ep_world_size - config.shared_expert_rank_num
    middle_idx_tp_list = []
    per_rank_bs = max(config.global_bs // config.ep_world_size, 1)
    for tp_rank_id in range(len(data["x_list"])):
        local_moe_expert_nums = local_moe_expert_num_list[tp_rank_id]
        expected_moe_expert_count = sum(local_moe_expert_nums)
        if not is_initial_elastic and config.is_expand:
            expected_moe_expert_count = config.moe_expert_num
        if is_initial_elastic:
            expected_moe_expert_count = config.elastic_moe_expert_num
        quant_output_scales = []
        x = data["x_list"][tp_rank_id]
        expert_ids = (
            data["expert_ids_rank_list"]
            if is_initial_elastic
            else data["expert_ids_expand_list"][tp_rank_id]
        )
        if not is_initial_elastic and not config.is_expand:
            expert_ids = data["expert_ids_list"][tp_rank_id]
        x_active_mask = data["x_active_mask_elastic_list"][tp_rank_id]
        x_active_mask_ela_ori = data["x_active_mask_list"][tp_rank_id]
        if config.is_expand and not is_initial_elastic:
            x_active_mask = data["x_active_mask_list"][tp_rank_id]
        has_special_expert = any(
            count > 0
            for count in (
                config.zero_expert_num,
                config.copy_expert_num,
                config.const_expert_num,
            )
        )
        if not has_special_expert:
            x_active_compute_mask = x_active_mask
        elif config.is_expand:
            x_active_compute_mask = data[
                "x_active_compute_initial_mask_list"
                if is_initial_elastic
                else "x_active_compute_expand_mask_list"
            ][tp_rank_id]
        else:
            x_active_compute_mask = data["x_active_compute_initial_mask_list"][tp_rank_id]
        x_chunks = [tensor.contiguous() for tensor in x.chunk(config.ep_world_size)]
        expert_ids_chunks = (
            list(torch.cat(expert_ids).chunk(config.ep_world_size))
            if isinstance(expert_ids, list)
            else list(expert_ids.chunk(config.ep_world_size))
        )
        expert_ids = torch.cat(expert_ids) if isinstance(expert_ids, list) else expert_ids

        if x_active_compute_mask.dim() == 1:
            selected_rows = torch.nonzero(
                x_active_compute_mask.reshape(-1), as_tuple=False
            ).flatten()
            valid_flat_index = (
                selected_rows[:, None] * config.top_k
                + torch.arange(config.top_k, device=selected_rows.device)
            ).reshape(-1)
            order = None
        else:
            valid_flat_index = torch.nonzero(
                x_active_compute_mask.reshape(-1), as_tuple=False
            ).flatten()
        valid_flat_index = valid_flat_index[valid_flat_index < expert_ids.numel()]
        valid_expert_ids = (
            torch.cat(expert_ids_chunks).reshape(-1)[valid_flat_index]
        )
        valid_moe_mask = valid_expert_ids < expected_moe_expert_count
        order = torch.argsort(valid_expert_ids, stable=True)
        sorted_flat_index = valid_flat_index[order]
        sorted_valid_moe_mask = valid_moe_mask[order]
        moe_middle_idx = sorted_flat_index[sorted_valid_moe_mask]
        expanded_x = torch.cat(x_chunks).to(_dtype(config))
        expanded_x = expanded_x[sorted_flat_index // config.top_k]
        expanded_x = expanded_x[valid_moe_mask]
        moe_tokens = torch.bincount(
            valid_expert_ids[valid_moe_mask].flatten(),
            minlength=expected_moe_expert_count,
        )
        if int(moe_tokens.sum()) != expanded_x.shape[0]:
            raise ValueError(
                "MOE dispatch token mismatch: "
                f"tokens={int(moe_tokens.sum())}, expand_x={expanded_x.shape[0]}"
            )

        shared_expand_x = []
        shared_tokens = []
        if config.is_shared:
            real_token_nums = (
                x_active_mask_ela_ori.view(
                    config.ep_world_size, per_rank_bs
                ).sum(dim=1)
                if x_active_mask_ela_ori.dim() == 1
                else x_active_mask_ela_ori.view(
                    config.ep_world_size, per_rank_bs, config.top_k
                )
                .any(dim=-1)
                .sum(dim=1)
            ).tolist()
            for shared_rank_id in range(rank_slot_ids_length):
                send_card_ids = dispatch_shared_source_cards(shared_rank_id)
                if not send_card_ids:
                    continue

                def card_rows(card_id):
                    card_mask = x_active_mask_ela_ori[
                        card_id * per_rank_bs : (card_id + 1) * per_rank_bs
                    ]
                    return card_mask.any(dim=-1) if card_mask.dim() == 2 else card_mask

                chunks = [
                    x_chunks[card_id][card_rows(card_id)].to(_dtype(config))
                    for card_id in send_card_ids
                ]
                tokens = sum(int(real_token_nums[card_id]) for card_id in send_card_ids)
                chunk = torch.cat(chunks)
                if config.is_quant:
                    scale_values = (
                        data["padded_scales_list"][tp_rank_id][shared_rank_id].float()
                        if config.is_dispatch_scales
                        else None
                    )
                    chunk_values = (
                        chunk.float() * scale_values
                        if scale_values is not None
                        else chunk.float()
                    )
                    chunk, chunk_scale = _quantize_dispatch_values(chunk_values, config)
                    quant_output_scales.append(chunk_scale.flatten())
                shared_expand_x.append(chunk)
                shared_tokens.append(tokens)
        else:
            shared_expand_x = [
                torch.empty(0, config.hidden_size, dtype=_dtype(config))
                for _ in range(shared_rank_num)
            ]

        per_expert_x = list(expanded_x.split(moe_tokens.tolist()))
        if len(per_expert_x) < expected_moe_expert_count:
            per_expert_x.extend(
                torch.empty(0, config.hidden_size, dtype=_dtype(config))
                for _ in range(expected_moe_expert_count - len(per_expert_x))
            )
        per_card_x = []
        moe_rank_num = config.ep_world_size - config.shared_expert_rank_num
        full_per_card_expert_nums = local_moe_expert_num_list[tp_rank_id]
        if is_initial_elastic:
            per_card_expert_nums = full_per_card_expert_nums
        elif config.is_expand:
            per_card_expert_num = -(-config.moe_expert_num // max(moe_rank_num, 1))
            per_card_expert_nums = []
            remaining_expert_num = config.moe_expert_num
            for _ in range(moe_rank_num):
                expert_count = min(per_card_expert_num, remaining_expert_num)
                per_card_expert_nums.append(expert_count)
                remaining_expert_num -= expert_count
        else:
            per_card_expert_nums = full_per_card_expert_nums
        per_card_scale_nums = []
        if not config.is_variable_local_moe_expert_num:
            expected_count = sum(per_card_expert_nums)
            per_card_count = (
                len(per_card_expert_nums)
                if per_card_expert_nums
                else moe_rank_num
            )
            if config.is_expand and expected_count % max(per_card_count, 1):
                per_card_expert_num = -(-expected_count // max(per_card_count, 1))
                per_card_expert_nums = []
                remaining_expert_num = expected_count
                for _ in range(per_card_count):
                    expert_count = min(per_card_expert_num, remaining_expert_num)
                    per_card_expert_nums.append(expert_count)
                    remaining_expert_num -= expert_count
            else:
                per_card_expert_nums = (expected_count // max(per_card_count, 1),) * (
                    max(per_card_count, 1)
                )
        local_moe_expert_bounds = []
        expert_start = 0
        for per_card_expert_num in per_card_expert_nums:
            local_moe_expert_bounds.append(
                (expert_start, expert_start + per_card_expert_num)
            )
            expert_start += per_card_expert_num
        expected_moe_expert_num = sum(per_card_expert_nums)
        expert_start = 0
        for moe_rank_id in range(moe_rank_num):
            start = expert_start
            end = start + per_card_expert_nums[moe_rank_id]
            expert_segments = []
            scale_segment_count = 0
            for expert_offset, expert_tensor in enumerate(per_expert_x[start:end]):
                expert_tensor = (
                    expert_tensor
                    if expert_tensor.numel()
                    else torch.empty(0, config.hidden_size, dtype=_dtype(config))
                )
                expert_global_id = config.shared_expert_rank_num + start + expert_offset
                if config.is_quant:
                    scale_values = (
                        data["padded_scales_list"][tp_rank_id][expert_global_id].float()
                        if config.is_dispatch_scales
                        else None
                    )
                    quant_values = expert_tensor.float()
                    if scale_values is not None:
                        quant_values = quant_values * scale_values
                    expert_tensor, row_scale = _quantize_dispatch_values(
                        quant_values, config
                    )
                    quant_output_scales.append(row_scale.flatten())
                    scale_segment_count += 1
                expert_segments.append(expert_tensor)
            per_card_x.append(
                torch.cat(expert_segments)
                if expert_segments
                else torch.empty(0, config.hidden_size, dtype=_dtype(config))
            )
            expert_start = end
            per_card_scale_nums.append(scale_segment_count)
        if is_initial_elastic and config.is_shared:
            shared_expand_x.extend(
                torch.empty(0, config.hidden_size, dtype=_dtype(config))
                for _ in range(shared_rank_num - rank_slot_ids_length)
            )
            shared_tokens.extend(0 for _ in range(shared_rank_num - rank_slot_ids_length))
        expand_x = shared_expand_x + per_card_x
        golden_moe_tokens = torch.tensor(
            [
                int(moe_tokens[start:end].sum())
                for start, end in local_moe_expert_bounds
            ],
            dtype=torch.int64,
        )
        middle_idx_tp_list.append(
            list(moe_middle_idx.split(golden_moe_tokens.tolist()))
        )
        tokens = torch.tensor(
            shared_tokens + [int(value) for value in golden_moe_tokens],
            dtype=torch.int64,
        )
        tokens = list(tokens.split([1] * len(tokens)))

        shared_recv_counts = []
        if config.is_shared:
            counts = torch.zeros(
                shared_rank_num,
                config.ep_world_size,
                dtype=torch.int32,
            )
            real_token_nums = (
                x_active_mask_ela_ori.view(
                    config.ep_world_size, per_rank_bs
                ).sum(dim=1)
                if x_active_mask_ela_ori.dim() == 1
                else x_active_mask_ela_ori.view(
                    config.ep_world_size, per_rank_bs, config.top_k
                )
                .any(dim=-1)
                .sum(dim=1)
            ).tolist()
            for card_id in range(config.ep_world_size):
                for shared_rank_id in range(counts.shape[0]):
                    send_card_ids = shared_source_cards(shared_rank_id)
                    if not send_card_ids:
                        continue
                    if card_id in send_card_ids:
                        counts[shared_rank_id, card_id] = real_token_nums[card_id]
            shared_recv_counts = list(counts.cumsum(1))
        else:
            shared_recv_counts = [
                torch.zeros(config.ep_world_size, dtype=torch.int32)
                for _ in range(shared_rank_num)
            ]

        moe_count_columns = (
            config.available_ranks if config.is_elastic else range(len(expert_ids_chunks))
        )
        moe_counts = torch.zeros(
            special_expert_count,
            len(moe_count_columns),
            dtype=torch.int32,
        )
        count_columns = moe_count_columns
        for column_id, rank_id in enumerate(count_columns):
            segment = expert_ids_chunks[rank_id].reshape(-1)
            rank_segment_rows = min(
                per_rank_bs,
                max(config.global_bs - rank_id * per_rank_bs, 0),
            )
            rank_mask = x_active_compute_mask[
                rank_id * per_rank_bs : rank_id * per_rank_bs + rank_segment_rows
            ]
            if rank_mask.dim() == 1:
                rank_mask = (
                    rank_mask.reshape(-1, 1).repeat_interleave(config.top_k).reshape(-1)
                )
            else:
                rank_mask = rank_mask[:, : config.top_k].reshape(-1)
            segment = segment[rank_mask]
            if segment.numel():
                segment = segment[segment < special_expert_count]
            moe_counts[:, column_id] = torch.bincount(
                segment, minlength=special_expert_count
            )
        rank_expert_bounds = [0]
        rank_expert_bounds.extend(
            rank_expert_bounds[-1] + count for count in per_card_expert_nums
        )
        rank_expert_bounds = [
            (start, end)
            for start, end in zip(rank_expert_bounds, rank_expert_bounds[1:])
        ]
        moe_counts = [
            moe_counts[start:end]
            .flatten()
            .cumsum(0)
            .clone()
            .to(torch.int32)
            for start, end in rank_expert_bounds
        ]
        ep_recv_count = shared_recv_counts + list(moe_counts)
        if config.is_quant:
            if quant_output_scales:
                dynamic_scales = quant_output_scales[
                    : config.shared_expert_rank_num
                ] + [
                    (
                        torch.cat(
                            quant_output_scales[
                                config.shared_expert_rank_num
                                + sum(per_card_scale_nums[:moe_rank_id]) : (
                                    config.shared_expert_rank_num
                                    + sum(per_card_scale_nums[: moe_rank_id + 1])
                                )
                            ]
                        )
                        if (
                            per_card_scale_nums[moe_rank_id]
                            and config.shared_expert_rank_num
                            + sum(per_card_scale_nums[: moe_rank_id + 1])
                            <= len(quant_output_scales)
                        )
                        else torch.empty(0, dtype=torch.float32)
                    )
                    for moe_rank_id in range(moe_rank_num)
                ]
            else:
                dynamic_scales = torch.empty(0, dtype=torch.float32)
        else:
            dynamic_scales = torch.empty(0, dtype=torch.float32)

        def shared_expand_idx(shared_rank_id):
            if not config.is_shared:
                return []
            values = []
            if not is_initial_elastic:
                send_card_ids = _shared_expert_send_card_ids(config, shared_rank_id)
                shared_expert_index = (
                    shared_rank_id // config.rank_num_per_shared_expert
                )
                for card_id in send_card_ids:
                    count = int(real_token_nums[card_id])
                    for token_id in range(count):
                        values.extend(
                            (card_id, token_id, config.top_k + shared_expert_index)
                        )
            else:
                available_ranks = config.available_ranks
                card_id_in_shared_group = (
                    shared_rank_id
                    % max(config.elastic_rank_num_per_shared_expert, 1)
                )
                shared_expert_index = (
                    shared_rank_id
                    // max(config.elastic_rank_num_per_shared_expert, 1)
                )
                local_card_ids = range(
                    card_id_in_shared_group,
                    len(available_ranks),
                    max(config.elastic_rank_num_per_shared_expert, 1),
                )
                for local_card_id in local_card_ids:
                    card_id = available_ranks[local_card_id]
                    count = int(real_token_nums[card_id])
                    for token_id in range(count):
                        values.extend(
                            (card_id, token_id, config.top_k + shared_expert_index)
                        )
            return values

        def moe_expand_idx(moe_rank_id):
            values = []
            start, end = local_moe_expert_bounds[moe_rank_id]
            for rank_id in range(config.ep_world_size):
                segment = expert_ids_chunks[rank_id].reshape(-1)
                if x_active_compute_mask.dim() == 1:
                    segment_mask = (
                        x_active_compute_mask[
                            rank_id * per_rank_bs : (rank_id + 1) * per_rank_bs
                        ]
                        .reshape(-1, 1)
                        .repeat(1, config.top_k)
                        .reshape(-1)
                    )
                else:
                    segment_mask = x_active_compute_mask[
                        rank_id * per_rank_bs : (rank_id + 1) * per_rank_bs
                    ].reshape(-1)
                for flat_id in (
                    torch.nonzero(segment_mask, as_tuple=False).flatten().tolist()
                ):
                    expert_id = int(segment[flat_id])
                    if start <= expert_id < end:
                        values.append((rank_id, expert_id, flat_id))
            values.sort(key=lambda item: item[1])
            triplets = [
                (rank_id, flat_id // config.top_k, flat_id % config.top_k)
                for rank_id, _, flat_id in values
            ]
            return [element for triplet in triplets for element in triplet]

        expand_idx = []
        for shared_rank_id in range(
            shared_rank_num
        ):
            expand_idx.append(
                torch.tensor(shared_expand_idx(shared_rank_id), dtype=torch.int32)
            )
        for moe_rank_id in range(moe_rank_num):
            expand_idx.append(
                torch.tensor(moe_expand_idx(moe_rank_id), dtype=torch.int32)
            )

        if is_initial_elastic and len(expand_x) < config.ep_world_size:
            expand_x.extend(
                torch.empty(0, config.hidden_size, dtype=_dtype(config))
                for _ in range(config.ep_world_size - len(expand_x))
            )
            expand_idx.extend(
                torch.empty(0, dtype=torch.int32)
                for _ in range(config.ep_world_size - len(expand_idx))
            )
            tokens.extend(
                torch.zeros(1, dtype=torch.int64)
                for _ in range(config.ep_world_size - len(tokens))
            )
            ep_recv_count.extend(
                torch.zeros(
                    config.ep_world_size * max(config.local_moe_expert_num, 1),
                    dtype=torch.int32,
                )
                for _ in range(config.ep_world_size - len(ep_recv_count))
            )

        expand_x_list.append(expand_x)
        expand_idx_list.append(expand_idx)
        tokens_list.append(tokens)
        ep_recv_count_list.append(ep_recv_count)
        dynamic_scales_list.append(dynamic_scales)

    return {
        "expand_x_list": expand_x_list,
        "expand_idx_list": expand_idx_list,
        "tokens_list": tokens_list,
        "ep_recv_count_list": ep_recv_count_list,
        "middle_idx_list": middle_idx_tp_list,
        "dynamic_scales_list": dynamic_scales_list,
    }


def _generate_combine_golden(data, dispatch_golden, config):
    golden_x = []
    row_mask_list = []
    per_rank_bs = max(config.global_bs // config.ep_world_size, 1)
    rank_num_per_shared_expert = config.rank_num_per_shared_expert
    shared_rank_num = config.shared_expert_rank_num
    for tp_rank_id in range(config.tp_world_size):
        has_special_expert = any(
            count > 0
            for count in (
                config.zero_expert_num,
                config.copy_expert_num,
                config.const_expert_num,
            )
        )
        expert_ids = (
            data["expert_ids_expand_list"][tp_rank_id]
            if config.is_expand
            else data["expert_ids_list"][tp_rank_id]
        )
        expert_ids = (
            torch.cat(expert_ids) if isinstance(expert_ids, list) else expert_ids
        )
        expert_scales = (
            data["expert_scales_list"][tp_rank_id].reshape(
                config.global_bs, config.top_k
            )
        )
        x_active_mask = data["x_active_mask_input_list"][tp_rank_id]
        x_active_mask_ela_ori = data["x_active_mask_input_list"][tp_rank_id]
        if has_special_expert and x_active_mask_ela_ori.dim() == 1:
            x_active_mask_ela_ori = x_active_mask_ela_ori.reshape(-1, 1).repeat(
                1, config.top_k
            )
        x_active_mask_output = x_active_mask_ela_ori
        x = data["x_list"][tp_rank_id]
        ori_x = torch.cat(list(x.chunk(config.ep_world_size))).float()
        result = torch.zeros(config.global_bs, config.hidden_size, dtype=torch.float32)
        expand_x = dispatch_golden["expand_x_list"][tp_rank_id]
        if config.is_quant:
            expand_x = [
                _dequantize_dispatch_values(tensor, config) for tensor in expand_x
                ]
        expand_x = [tensor.to(_dtype(config)) for tensor in expand_x]
        available_moe_slot_ids = [
            rank
            for rank in range(config.shared_expert_rank_num)
            if rank in config.available_ranks
        ] + [
            rank
            for rank in config.available_ranks
            if rank >= config.shared_expert_rank_num
        ]
        available_moe_count = (
            config.elastic_shared_expert_rank_num
            if config.is_elastic and not config.is_expand
            else config.shared_expert_rank_num
        )
        moe_expand_x = [
            tensor.view(-1, config.hidden_size)
            for tensor in (
                expand_x[config.shared_expert_rank_num:]
                if config.is_expand
                else expand_x[available_moe_count:]
            )
        ]
        middle_idx = torch.cat(dispatch_golden["middle_idx_list"][tp_rank_id])
        if middle_idx.numel() != sum(
            tensor.shape[0] for tensor in moe_expand_x
        ):
            raise ValueError(
                "MOE dispatch shape mismatch: "
                f"middle_idx={middle_idx.numel()}, "
                f"expand_x={sum(tensor.shape[0] for tensor in moe_expand_x)}"
            )
        moe_expand_x = torch.cat(moe_expand_x)
        expandx_row_to_x_row = torch.argsort(middle_idx)
        moe_expand_x = moe_expand_x[expandx_row_to_x_row].float()
        moe_expand_x = moe_expand_x[torch.argsort(expandx_row_to_x_row)]
        moe_expand_x = _comm_quant_round_trip(moe_expand_x, config)
        moe_expand_x = moe_expand_x[expandx_row_to_x_row].float()
        expand_index = 0
        for token_id in range(config.global_bs):
            for top_id in range(config.top_k):
                expert_id = int(expert_ids[token_id, top_id])
                token_mask = (
                    x_active_mask[token_id].any()
                    if x_active_mask.dim() == 2
                    else x_active_mask[token_id]
                )
                if x_active_mask.dim() == 2:
                    token_mask = bool(x_active_mask[token_id, top_id])
                if not bool(token_mask):
                    continue
                if expert_id < config.moe_expert_num:
                    result[token_id] += moe_expand_x[expand_index] * float(
                        expert_scales[token_id, top_id]
                    )
                    expand_index += 1
                    continue
                if expert_id < config.moe_expert_num + config.zero_expert_num:
                    continue
                elif expert_id < (
                    config.moe_expert_num
                    + config.zero_expert_num
                    + config.copy_expert_num
                ):
                    result[token_id] += ori_x[token_id] * float(
                        expert_scales[token_id, top_id]
                    )
                else:
                    const_id = (
                        expert_id
                        - config.moe_expert_num
                        - config.zero_expert_num
                        - config.copy_expert_num
                    )
                    alpha_1_values = (
                        data["const_expert_alpha_1_list"][tp_rank_id][const_id].float()
                        * ori_x[token_id]
                    )
                    alpha_2_values = (
                        data["const_expert_alpha_2_list"][tp_rank_id][const_id].float()
                        * ori_x[token_id]
                    )
                    alpha_1_sum = _binary_tree_sum(alpha_1_values)
                    alpha_2_sum = _binary_tree_sum(alpha_2_values)
                    alpha_max = max(float(alpha_1_sum), float(alpha_2_sum))
                    denom = (
                        torch.exp(alpha_1_sum - alpha_max).double()
                        + torch.exp(alpha_2_sum - alpha_max).double()
                    ).float()
                    alpha_1 = torch.exp(alpha_1_sum - alpha_max).float() / denom
                    alpha_2 = torch.exp(alpha_2_sum - alpha_max).float() / denom
                    const_v = data["const_expert_v_list"][tp_rank_id][const_id].float()
                    result[token_id] += (
                        alpha_1 * ori_x[token_id].float() + alpha_2 * const_v
                    ) * float(expert_scales[token_id, top_id])
        if config.is_shared:
            shared_token_mask = x_active_mask_output
            if shared_token_mask.dim() == 1:
                real_token_mask = shared_token_mask
            else:
                real_token_mask = shared_token_mask.any(dim=-1)
            real_token_nums = (
                real_token_mask.view(config.ep_world_size, per_rank_bs)
                .sum(dim=1)
                .tolist()
            )
            shared_results = [
                [torch.empty(0, config.hidden_size) for _ in range(config.ep_world_size)]
                for _ in range(config.shared_expert_num)
            ]
            shared_rank_stride = (
                config.elastic_rank_num_per_shared_expert
                if config.is_elastic and not config.is_expand
                else rank_num_per_shared_expert
            )
            for shared_rank_id in range(
                config.elastic_shared_expert_rank_num
                if config.is_elastic and not config.is_expand
                else shared_rank_num
            ):
                shared_rank_num_per_expert = max(shared_rank_stride, 1)
                shared_expert_index = shared_rank_id // shared_rank_num_per_expert
                card_id_in_shared_group = shared_rank_id % shared_rank_num_per_expert
                if not config.is_elastic or config.is_expand:
                    send_card_ids = range(
                        card_id_in_shared_group,
                        config.ep_world_size,
                        shared_rank_num_per_expert,
                    )
                else:
                    send_card_ids = [
                        config.available_ranks[index]
                        for index in range(
                            card_id_in_shared_group,
                            len(config.available_ranks),
                            shared_rank_num_per_expert,
                        )
                    ]
                chunks = dispatch_golden["expand_x_list"][tp_rank_id][shared_rank_id]
                offset = 0
                for card_id in send_card_ids:
                    count = int(real_token_nums[card_id])
                    chunk = chunks[offset : offset + count]
                    offset += count
                    shared_results[shared_expert_index][card_id] = chunk.float()
            for tensors in shared_results:
                round_trip = _comm_quant_round_trip(torch.cat(tensors), config)
                scaled = torch.zeros_like(result)
                rank_masks = real_token_mask.view(
                    config.ep_world_size, per_rank_bs
                )
                if round_trip.shape[0] != int(real_token_mask.sum()):
                    raise ValueError(
                        "shared expert row count does not match active mask"
                    )
                valid_rows = torch.nonzero(
                    real_token_mask, as_tuple=False
                ).flatten()
                scaled[valid_rows] = round_trip.float()
                result = result + scaled
        row_mask = (
            x_active_mask_output.any(dim=-1)
            if x_active_mask_output.dim() == 2
            else x_active_mask_output
        )
        if config.is_shared_expert_x:
            rank_masks = row_mask.view(config.ep_world_size, per_rank_bs)
            shared_tensors = data["shared_expert_x_list"][tp_rank_id]
            for rank_id, shared_tensor in enumerate(shared_tensors):
                shared_values = shared_tensor.float()
                if shared_values.numel() == 0:
                    continue
                if shared_values.shape[0] == per_rank_bs:
                    result[
                        rank_id * per_rank_bs : (rank_id + 1) * per_rank_bs
                    ] += shared_values
                    continue
                valid_rows = torch.nonzero(rank_masks[rank_id], as_tuple=False).flatten()
                if valid_rows.numel() != shared_values.shape[0]:
                    raise ValueError(
                        "shared expert row count does not match active mask"
                    )
                result[rank_id * per_rank_bs + valid_rows] += shared_values
        per_rank_real_token_nums = (
            row_mask.view(config.ep_world_size, per_rank_bs).sum(dim=1).tolist()
        )
        row_mask_list.extend(row_mask.view(config.ep_world_size, per_rank_bs))
        if not config.is_add_rms_norm:
            result = result.to(_dtype(config))
        for rank_id, rank_result in enumerate(result.split(per_rank_bs)):
            if row_mask.dim() == 1:
                rank_mask = row_mask_list[rank_id]
                rank_result = rank_result * rank_mask.unsqueeze(1).to(
                    rank_result.dtype
                )
            golden_x.append(rank_result)
    golden_x = golden_x[: config.ep_world_size * config.tp_world_size]
    if config.tp_world_size > 1:
        reordered = []
        for ep_rank_id in range(config.ep_world_size):
            for tp_rank_id in range(config.tp_world_size):
                reordered.append(
                    golden_x[tp_rank_id * config.ep_world_size + ep_rank_id]
                )
        golden_x = reordered
        row_mask_list = [
            row_mask_list[tp_rank_id * config.ep_world_size + ep_rank_id]
            for ep_rank_id in range(config.ep_world_size)
            for tp_rank_id in range(config.tp_world_size)
        ]
    return golden_x, row_mask_list


def _generate_add_rms_norm_golden(golden_x, config):
    if not golden_x:
        raise ValueError("AddRmsNorm golden requires non-empty combine golden")
    if config.is_elastic:
        raise NotImplementedError("Elastic AddRmsNorm golden is not supported yet")
    hidden_size = config.hidden_size
    generator = torch.Generator().manual_seed(42)
    residual_x = torch.rand(
        (config.global_bs, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    gamma = torch.ones(hidden_size, dtype=torch.float32)
    per_rank_bs = config.global_bs // max(config.ep_world_size, 1)
    residual_x = residual_x[:per_rank_bs]
    normalized = [
        tensor.float() + residual_x.float() for tensor in golden_x
    ]
    rstd = [
        torch.rsqrt(tensor.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        for tensor in normalized
    ]
    golden_y = [
        tensor * rstd_value * gamma for tensor, rstd_value in zip(normalized, rstd)
    ]
    golden_rstd = [value.reshape(-1) for value in rstd]
    return (
        golden_y,
        golden_rstd,
        normalized,
        [
            residual_x.clone()
            for rank in range(config.ep_world_size * config.tp_world_size)
        ],
    )


def generate_case(config):
    data = _generate_case(config)
    initial_dispatch_golden = None
    if config.is_expand:
        initial_dispatch_golden = _generate_dispatch_golden(
            data, config, initial_elastic=True
        )
    dispatch_golden = _generate_dispatch_golden(data, config, initial_elastic=False)
    const_expert_alpha_1 = torch.empty(
        config.const_expert_num, config.hidden_size, dtype=_dtype(config)
    ).uniform_(-1, 1)
    const_expert_alpha_2 = torch.empty(
        config.const_expert_num, config.hidden_size, dtype=_dtype(config)
    ).uniform_(-1, 1)
    const_expert_v = torch.empty(
        config.const_expert_num, config.hidden_size, dtype=_dtype(config)
    ).uniform_(-1, 1)
    expert_scales_list = [
        torch.empty(config.global_bs, config.top_k, dtype=torch.float32).uniform_(-1, 1)
        for _ in range(config.tp_world_size)
    ]
    per_rank_expert_scales_list = []
    per_rank_bs = max(config.global_bs // config.ep_world_size, 1)
    for expert_scales in expert_scales_list:
        flat_scales = expert_scales.reshape(-1)
        per_rank_expert_scales_list.append(
            torch.stack(
                [
                    flat_scales[
                        rank_id
                        * per_rank_bs
                        * config.top_k : (rank_id + 1)
                        * per_rank_bs
                        * config.top_k
                    ].reshape(per_rank_bs, config.top_k)
                    for rank_id in range(config.ep_world_size)
                ]
            )
        )
    elastic_info = (
        config.elastic_info
        if config.ep_world_size == 16
        else torch.zeros(36, dtype=torch.int32)
    )
    data.update(
        {
            "ori_x_list": data["x_list"],
            "elastic_info_list": [elastic_info for _ in range(config.tp_world_size)],
            "const_expert_alpha_1_list": [
                const_expert_alpha_1 for _ in range(config.tp_world_size)
            ],
            "const_expert_alpha_2_list": [
                const_expert_alpha_2 for _ in range(config.tp_world_size)
            ],
            "const_expert_v_list": [
                const_expert_v for _ in range(config.tp_world_size)
            ],
            "expert_scales_list": per_rank_expert_scales_list,
        }
    )
    combine_dispatch_golden = dispatch_golden
    if config.is_quant:
        combine_dispatch_golden = {
            **combine_dispatch_golden,
            "expand_x_list": [
                [tensor.to(_dtype(config)) for tensor in tensors]
                for tensors in combine_dispatch_golden["expand_x_list"]
            ],
        }
    combine_result = _generate_combine_golden(data, combine_dispatch_golden, config)
    result = {
        **data,
        **dispatch_golden,
        "golden_x": combine_result[0],
        "row_mask_list": combine_result[1],
    }
    if config.is_add_rms_norm:
        add_rms_result = _generate_add_rms_norm_golden(result["golden_x"], config)
        result.update(
            {
                "golden_y": add_rms_result[0],
                "golden_rstd_out": add_rms_result[1],
                "add_rms_norm_x": add_rms_result[2],
                "residual_x": add_rms_result[3],
            }
        )
    if initial_dispatch_golden is not None:
        result.update(
            {
                "initial_expand_x_list": initial_dispatch_golden["expand_x_list"],
                "initial_expand_idx_list": initial_dispatch_golden["expand_idx_list"],
                "initial_tokens_list": initial_dispatch_golden["tokens_list"],
                "initial_ep_recv_count_list": initial_dispatch_golden[
                    "ep_recv_count_list"
                ],
                "initial_middle_idx_list": initial_dispatch_golden["middle_idx_list"],
            }
        )
    return result


def _cache_identity(config):
    generator_source = Path(__file__).read_bytes()
    payload = {
        "generator_source": hashlib.sha256(generator_source).hexdigest(),
        "version": int(_moe_config().generator_version),
        **asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _moe_config():
    from ttk.config.loader import get_moe_config, load_config

    try:
        get_moe_config()
    except RuntimeError:
        load_config()

    return get_moe_config()


def _disable_rank_case(config):
    return bool(getattr(_moe_config(), "disable_rank_case", False)) and (
        config.is_elastic or config.is_expand
    )


def _cache_root():
    root = Path(_moe_config().cache_dir)
    root = root.expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def _load_cache(path, expected_manifest):
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized_manifest = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in expected_manifest.items()
    }
    if any(manifest.get(key) != value for key, value in normalized_manifest.items()):
        raise ValueError(f"Moe cache manifest mismatch: {path}")
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "case.pt"
        with open(archive_path, "wb") as output:
            for part_name in manifest["parts"]:
                part_path = path / part_name
                digest = hashlib.sha256()
                with open(part_path, "rb") as part:
                    while chunk := part.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != manifest["checksums"][part_name]:
                    raise ValueError(f"Moe cache checksum mismatch: {part_path}")
        return torch.load(archive_path, map_location="cpu", weights_only=True)


def _save_cache(path, case, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    temp_path.mkdir()
    try:
        archive_path = temp_path / "case.pt"
        torch.save(case, archive_path)
        part_size = 512 * 1024 * 1024
        checksums = {}
        with open(archive_path, "rb") as archive:
            part_index = 0
            while chunk := archive.read(part_size):
                part_name = f"case-{part_index:04d}.part"
                with open(temp_path / part_name, "wb") as part:
                    part.write(chunk)
                checksums[part_name] = hashlib.sha256(chunk).hexdigest()
                part_index += 1
        archive_path.unlink()
        (temp_path / "manifest.json").write_text(
            json.dumps(
                {**manifest, "parts": sorted(checksums), "checksums": checksums},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if path.exists():
            backup = path.with_name(f".{path.name}.old-{os.getpid()}")
            path.rename(backup)
            temp_path.rename(path)
            shutil.rmtree(backup)
        else:
            temp_path.rename(path)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


def load_or_generate_case(config):
    with _moe_case_cache_lock:
        mode = _moe_config().cache_mode
        identity, manifest = _cache_identity(config)
        path = _cache_root() / identity[:2] / identity
        if mode in ("read-write", "read-only"):
            try:
                return {**_load_cache(path, manifest), "available_ranks": config.available_ranks}
            except (FileNotFoundError, ValueError, RuntimeError):
                if mode == "read-only":
                    raise
        if mode not in ("read-write",):
            raise ValueError(f"Unsupported Moe cache mode: {mode}")
        case = generate_case(config)
        try:
            _save_cache(path, case, manifest)
        except (FileExistsError, OSError):
            try:
                return {**_load_cache(path, manifest), "available_ranks": config.available_ranks}
            except (FileNotFoundError, ValueError, RuntimeError):
                pass
        return {**case, "available_ranks": config.available_ranks}


def _select_rank_tensor(tensors, config, ep_rank_id):
    if not config.is_elastic:
        return tensors[ep_rank_id]
    if ep_rank_id not in config.available_ranks:
        return torch.empty(0, config.hidden_size, dtype=_dtype(config))
    return tensors[ep_rank_id]


def _select_rank_1d_tensor(tensors, config, ep_rank_id):
    if not config.is_elastic:
        return tensors[ep_rank_id]
    if ep_rank_id not in config.available_ranks:
        return torch.empty(0, dtype=torch.int32)
    return tensors[ep_rank_id]


def rank_case(case, config, ep_rank_id, tp_rank_id=0):
    dispatch_index = tp_rank_id
    rank = tp_rank_id * config.ep_world_size + ep_rank_id
    per_rank_bs = case.get("per_rank_bs", config.global_bs)
    x = case["x_list"][tp_rank_id]
    expert_ids = case["expert_ids_list"][tp_rank_id]
    initial_expand_x = (
        _select_rank_tensor(
            case.get("initial_expand_x_list", case["expand_x_list"])[dispatch_index],
            config,
            ep_rank_id,
        )
        if config.is_expand
        else case["expand_x_list"][dispatch_index][ep_rank_id]
    )
    if initial_expand_x is None:
        initial_expand_x = torch.empty(0, config.hidden_size, dtype=_dtype(config))
    initial_expand_idx = (
        _select_rank_1d_tensor(
            case.get("initial_expand_idx_list", case["expand_idx_list"])[dispatch_index],
            config,
            ep_rank_id,
        )
        if config.is_expand
        else case["expand_idx_list"][dispatch_index][ep_rank_id]
    )
    if initial_expand_idx is None:
        initial_expand_idx = torch.empty(0, dtype=torch.int32)
    initial_ep_recv_counts = (
        _apply_rank_mapping(
            _select_rank_1d_tensor(
                case.get(
                    "initial_ep_recv_count_list", case["ep_recv_count_list"]
                )[dispatch_index],
                config,
                ep_rank_id,
            ),
            config,
        )
        if config.is_expand
        else case["ep_recv_count_list"][dispatch_index][ep_rank_id]
    )
    expert_ids_expand = (
        case["expert_ids_expand_list"][tp_rank_id] if config.is_expand else None
    )
    rank_index = tp_rank_id * config.ep_world_size + ep_rank_id
    expert_scales = case["expert_scales_list"][tp_rank_id]
    x_active_mask = case["x_active_mask_list"][tp_rank_id][
        ep_rank_id * per_rank_bs : (ep_rank_id + 1) * per_rank_bs
    ]
    ori_x = case["ori_x_list"][tp_rank_id][
        ep_rank_id * per_rank_bs : (ep_rank_id + 1) * per_rank_bs
    ]
    return {
        "available_ranks": config.available_ranks,
        "x": case["x_rank_list"][rank_index],
        "expert_ids": case["expert_ids_rank_list"][rank_index],
        "expert_ids_expand": (
            case["expert_ids_rank_expand_list"][rank_index]
            if config.is_expand
            else None
        ),
        "x_active_mask": x_active_mask,
        "x_active_compute_mask": (
            case["x_active_compute_mask_list"][tp_rank_id][
                ep_rank_id * per_rank_bs : (ep_rank_id + 1) * per_rank_bs
            ]
        ),
        "x_active_mask_input": (
            case.get("x_active_mask_input_list", [case["x_active_mask_list"]])[
                tp_rank_id
            ][ep_rank_id * per_rank_bs : (ep_rank_id + 1) * per_rank_bs]
        ),
        "x_active_compute_expand_mask": (
            case["x_active_compute_expand_mask_list"][tp_rank_id][
                ep_rank_id * per_rank_bs : (ep_rank_id + 1) * per_rank_bs
            ]
        ),
        "expert_scales": expert_scales[ep_rank_id],
        "scales": case["scales_list"][tp_rank_id],
        "padded_scales": case["padded_scales_list"][tp_rank_id],
        "expand_x": (
            case["expand_x_list"][dispatch_index][ep_rank_id]
            if config.is_expand
            else case["expand_x_list"][dispatch_index][ep_rank_id]
        ),
        "expand_idx": case["expand_idx_list"][dispatch_index][ep_rank_id],
        "ep_recv_counts": case["ep_recv_count_list"][dispatch_index][ep_rank_id],
        "initial_expand_x": initial_expand_x,
        "initial_expand_idx": initial_expand_idx,
        "initial_ep_recv_counts": initial_ep_recv_counts,
        "tokens": case["tokens_list"][dispatch_index][ep_rank_id],
        "golden_x": case["golden_x"][rank],
        "residual_x": (
            case["residual_x"][rank].view(-1, 1, config.hidden_size)
            if config.is_add_rms_norm
            else torch.empty(0, 1, config.hidden_size, dtype=_dtype(config))
        ),
        "shared_expert_x": (
            case["shared_expert_x_list"][tp_rank_id][ep_rank_id]
            if ep_rank_id < len(case["shared_expert_x_list"][tp_rank_id])
            else torch.empty(0, config.hidden_size, dtype=_dtype(config))
        ),
        "ori_x": ori_x,
        "elastic_info": case["elastic_info_list"][tp_rank_id],
        "const_expert_alpha_1": case["const_expert_alpha_1_list"][tp_rank_id],
        "const_expert_alpha_2": case["const_expert_alpha_2_list"][tp_rank_id],
        "const_expert_v": case["const_expert_v_list"][tp_rank_id],
    }

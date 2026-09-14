#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MoeUpdateExpert correlated inputs ported from test_eplb.py."""

import random

import numpy as np
import torch


def _fill_tensor(tensor, value):
    if tensor is not None:
        tensor.copy_(torch.as_tensor(value, dtype=tensor.dtype))


def customize_inputs(
    expert_ids,
    eplb_table,
    expert_scales=None,
    pruning_threshold=None,
    active_mask=None,
    local_rank_id=0,
    world_size=1,
    balance_mode=0,
    balanced_expert_ids=None,
    balanced_active_mask=None,
    **kwargs,
):
    """Generate the same expert table and pruning tensors as the EPLB script."""
    seed = kwargs.get("seed", 1)
    world_size = int(world_size)
    local_rank_id = int(local_rank_id)
    balance_mode = int(balance_mode)
    place_per_rank = kwargs.get("placePerRank", kwargs.get("place_per_rank", 1))
    expert_num = kwargs.get("moeExpertNum", kwargs.get("moe_expert_num", 1))
    is_pruning = kwargs.get("isPruning", kwargs.get("is_pruning", False))
    pruning_dims = kwargs.get("pruningDims", kwargs.get("pruning_dims", 1))
    is_active_mask = kwargs.get("isActiveMask", kwargs.get("is_active_mask", False))

    table_width = random.Random(seed).randint(2, world_size + 1)
    logical_expert_num = expert_num
    physical_expert_num = world_size * place_per_rank

    np.random.seed(seed)
    expert_table = np.zeros((logical_expert_num, table_width - 1), dtype=np.int32)
    all_ranks = np.arange(physical_expert_num)
    count_column = np.random.randint(1, table_width, size=(logical_expert_num, 1))
    for expert_idx in range(logical_expert_num):
        np.random.shuffle(all_ranks)
        count = int(count_column[expert_idx, 0])
        expert_table[expert_idx, :count] = all_ranks[:count]
    _fill_tensor(eplb_table, np.hstack((count_column.astype(np.int32), expert_table)))

    expert_ids_array = np.random.randint(low=0, high=logical_expert_num, size=expert_ids.shape)
    _fill_tensor(expert_ids, expert_ids_array)

    topk = expert_ids.shape[-1]
    if is_pruning:
        _fill_tensor(
            expert_scales,
            -np.sort(
            -np.random.uniform(low=0.0, high=0.25, size=expert_ids.shape), axis=1
            ),
        )
        threshold_shape = (1, topk) if pruning_dims == 2 else (topk,)
        _fill_tensor(
            pruning_threshold,
            np.random.uniform(
            low=0.0, high=0.15, size=threshold_shape
            ),
        )
    if is_active_mask:
        true_count = int(np.random.randint(0, expert_ids.shape[0] + 1))
        _fill_tensor(
            active_mask,
            np.concatenate(
                (
                    np.ones(true_count, dtype=bool),
                    np.zeros(expert_ids.shape[0] - true_count, dtype=bool),
                )
            ),
        )
    return (
        expert_ids,
        eplb_table,
        expert_scales,
        pruning_threshold,
        active_mask,
    )

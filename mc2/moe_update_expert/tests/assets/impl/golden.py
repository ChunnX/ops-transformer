#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Golden for aclnnMoeUpdateExpert."""

import math


def moe_update_expert_golden(
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
    """Port test_eplb.exec_cpu and return balanced expert IDs and mask."""
    import torch

    is_pruning = expert_scales is not None and pruning_threshold is not None
    batch_size, topk = expert_ids.shape
    balanced_expert_ids = torch.zeros_like(expert_ids)
    balanced_active_mask = torch.zeros(
        expert_ids.shape, dtype=torch.bool, device=expert_ids.device
    )

    for token_idx in range(batch_size):
        for k_idx in range(topk):
            logical_expert_id = int(expert_ids[token_idx, k_idx])
            count = int(eplb_table[logical_expert_id, 0])
            if balance_mode == 0:
                modulus = math.ceil(world_size / count)
                physical_expert_id = eplb_table[
                    logical_expert_id, (local_rank_id // modulus) + 1
                ]
            else:
                physical_expert_id = eplb_table[
                    logical_expert_id, (token_idx % count) + 1
                ]
            balanced_expert_ids[token_idx, k_idx] = physical_expert_id

            if is_pruning:
                if pruning_threshold.dim() == 2:
                    threshold = pruning_threshold[0, k_idx]
                else:
                    threshold = pruning_threshold[k_idx]
                if active_mask is not None and not bool(active_mask[token_idx]):
                    balanced_active_mask[token_idx, k_idx] = False
                else:
                    balanced_active_mask[token_idx, k_idx] = bool(
                        expert_scales[token_idx, k_idx]
                        >= threshold * expert_scales[token_idx].sum()
                    )

    if is_pruning:
        return balanced_expert_ids, balanced_active_mask
    return balanced_expert_ids, torch.ones_like(balanced_active_mask)

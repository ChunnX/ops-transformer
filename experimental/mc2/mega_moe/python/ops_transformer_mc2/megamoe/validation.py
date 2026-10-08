# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit host-side preflight, never a hidden scan in the kernel hot path."""

import math
from ops_transformer_mc2._store import StoreGroup


def check_routing(indices, weights, num_experts, num_topk):
    """Validate CPU lists obtained from the actual input tensors."""
    if len(indices) != len(weights):
        raise ValueError("Routing IDs and weights have different BS")
    for token, (ids, probs) in enumerate(zip(indices, weights)):
        if len(ids) != num_topk or len(probs) != num_topk:
            raise ValueError(
                f"token={token}: routing width must equal top-k={num_topk}"
            )
        if any(
            type(expert) is not int or not 0 <= expert < num_experts for expert in ids
        ):
            raise ValueError(
                f"token={token}: expert IDs must be in [0,{num_experts}); -1 masking is unsupported"
            )
        if len(set(ids)) != num_topk:
            raise ValueError(f"token={token}: duplicate expert IDs are unsupported")
        if any(not math.isfinite(value) for value in probs):
            raise ValueError(f"token={token}: route weights must be finite")


def collective_check(group, stage, error, signature=None):
    if isinstance(group, StoreGroup):
        peers = group.all_gather((error, signature), f"check:{stage}")
    else:
        import torch.distributed as dist

        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, (error, signature), group=group)
    failures = [
        f"rank={rank}: {value[0]}" for rank, value in enumerate(peers) if value[0]
    ]
    if failures:
        raise ValueError(f"MegaMoE {stage} failed: " + "; ".join(failures))
    if any(value[1] != peers[0][1] for value in peers):
        raise ValueError(
            f"MegaMoE {stage}: ranks disagree: "
            + "; ".join(f"rank={rank}: {value[1]}" for rank, value in enumerate(peers))
        )

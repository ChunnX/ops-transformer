"""TTK TestSpec for the legacy ACLNN MoeFusedTopk ATK regression matrix."""

from __future__ import annotations

import json
import ctypes
from functools import lru_cache
from pathlib import Path


__spec__ = {"aclnnMoeFusedTopk": "AclnnMoeFusedTopkTestSpec"}


def _enable_ttk_uint32_t_support() -> None:
    """Bridge the current TTK ACLNN ctypes map to the standard CANN type.

    MoeFusedTopk correctly exposes its unsigned attributes as ``uint32_t``.
    TTK 3.0's Phase-1 builder maps the neighbouring ``int32_t`` and
    ``uint64_t`` types but omits ``uint32_t``.  Keep this compatibility shim in
    the test plugin instead of weakening the public ACLNN header just for a
    test runner.
    """
    try:
        from ttk.core_modules.npu.op_api.profiling import Phase1ParamBuilder
    except ImportError:
        return
    Phase1ParamBuilder.CBaseTypeMap.setdefault("uint32_t", ctypes.c_uint32)


_enable_ttk_uint32_t_support()


@lru_cache(maxsize=1)
def _special_inputs() -> dict[str, dict[str, str]]:
    """Load NaN/Inf intent from the retained legacy ATK source matrix."""
    source = Path(__file__).with_name("atk_aclnnMoeFusedTopk.json")
    cases = json.loads(source.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for case in cases:
        values = {item["name"]: item["range_values"] for item in case["inputs"]}
        special = {
            name: value[0]
            for name, value in values.items()
            if name in {"x", "add_num"}
            and isinstance(value, list)
            and len(value) == 1
            and value[0] in {"nan", "inf", "-inf"}
        }
        if special:
            result[f"moe_fused_topk_atk_{case['id']:03d}"] = special
    return result


class AclnnMoeFusedTopkTestSpec:
    """Python Golden equivalent to ``executor_aclnnMoeFusedTopk.py``."""

    def customize_inputs(
        x,
        addNum,
        mappingNum,
        mappingTable,
        groupNum,
        groupTopk,
        topN,
        topK,
        activateType,
        isNorm,
        scale,
        enableExpertMapping,
        y,
        indices,
        *,
        testcase_name=None,
        **_kwargs,
    ):
        """Restore the legacy matrix's NaN/Inf inputs after TTK random generation."""
        special = _special_inputs().get(testcase_name, {})
        for tensor_name, tensor in (("x", x), ("add_num", addNum)):
            if tensor_name in special:
                tensor.fill_(float(special[tensor_name]))

    def golden(
        x,
        addNum,
        mappingNum,
        mappingTable,
        groupNum,
        groupTopk,
        topN,
        topK,
        activateType,
        isNorm,
        scale,
        enableExpertMapping,
        y,
        indices,
        **_kwargs,
    ):
        """Calculate the historical ATK Golden with stable group and expert ordering."""
        import torch

        del activateType
        x_float = x.to(torch.float32)
        add_num_float = addNum.to(torch.float32)
        sigmoid = torch.sigmoid(x_float)
        scores = sigmoid + add_num_float

        token_count, expert_count = scores.shape
        group_num = int(groupNum)
        group_topk = int(groupTopk)
        top_n = int(topN)
        top_k = int(topK)
        grouped = scores.reshape(token_count, group_num, expert_count // group_num)
        masked = grouped.clone()
        group_scores = torch.topk(grouped, top_n, dim=-1).values.sum(dim=-1)
        group_order = torch.argsort(group_scores, dim=-1, descending=True, stable=True)

        if group_topk < group_num:
            deselected = group_order[:, group_topk:]
            row_indices = torch.arange(token_count, device=grouped.device).unsqueeze(1)
            row_indices = row_indices.expand_as(deselected)
            masked[row_indices.reshape(-1), deselected.reshape(-1)] = 0.0

        sorted_scores, sorted_indices = torch.sort(
            masked.reshape(token_count, expert_count), dim=-1, descending=True, stable=True
        )
        del sorted_scores
        selected_indices = sorted_indices[:, :top_k]
        result = torch.gather(sigmoid, dim=-1, index=selected_indices)
        if bool(isNorm):
            result = result / result.sum(dim=-1, keepdim=True)
            result = result * float(scale)

        output_indices = selected_indices.to(torch.int32)
        if bool(enableExpertMapping):
            output_indices = output_indices.clone()
            for batch_index in range(token_count):
                for topk_index in range(top_k):
                    expert_id = int(selected_indices[batch_index, topk_index].item())
                    redundant_offset = batch_index % max(1, int(mappingNum[expert_id].item()))
                    output_indices[batch_index, topk_index] = mappingTable[expert_id, redundant_offset]

        return [result.to(dtype=y.dtype), output_indices.to(dtype=indices.dtype)]

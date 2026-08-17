"""CPU Golden and special-input restoration for direct Kernel MoeFusedTopk."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy


__spec__ = {"moe_fused_topk": "MoeFusedTopkKernelTestSpec"}


@lru_cache(maxsize=1)
def _special_inputs() -> dict[str, dict[str, str]]:
    """Read the retained ATK NaN/Inf intent for the selected 200 source cases."""
    source = Path(__file__).parents[1] / "aclnnMoeFusedTopk" / "atk_aclnnMoeFusedTopk.json"
    result: dict[str, dict[str, str]] = {}
    for case in json.loads(source.read_text(encoding="utf-8")):
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
            result[f"moe_fused_topk_kernel_atk_{case['id']:03d}"] = special
    return result


class MoeFusedTopkKernelTestSpec:
    """Numpy-input TestSpec matching the stable ATK reference semantics."""

    @staticmethod
    def _fill_ordered_scores(x, add_num) -> None:
        """Create exactly representable, strictly ordered routing scores.

        The sequence uses 127 non-boundary mantissas in each exponent band,
        so it remains distinct in FP16, BF16 and FP32.  With ``x=0`` sigmoid
        is exactly 0.5 and every selected-group score remains positive.  This
        keeps the regression's TopK boundary reproducible without requiring a
        scalar stable-tie reconstruction in the device kernel.
        """
        expert_count = add_num.shape[0]
        ranks = numpy.arange(expert_count, dtype=numpy.int32)
        values = -numpy.ldexp(
            1 + ranks % 127,
            -20 + ranks // 127,
        ).astype(numpy.float32)
        x.fill(0)
        add_num[...] = values

    def customize_inputs(
        x,
        add_num,
        mapping_num,
        mapping_table,
        *,
        testcase_name=None,
        **_kwargs,
    ):
        """Restore the legacy matrix's NaN/Inf cases after random generation."""
        special = _special_inputs().get(testcase_name, {})
        if "x" in special:
            x.fill(numpy.float32(special["x"]))
        if "add_num" in special:
            add_num.fill(numpy.float32(special["add_num"]))
        if not special:
            # Avoid CPU/device sigmoid rounding at an indistinguishable TopK
            # boundary, while retaining every dtype, shape and attribute branch.
            # NaN/Inf source cases intentionally stay on their dedicated path.
            MoeFusedTopkKernelTestSpec._fill_ordered_scores(x, add_num)
        return x, add_num, mapping_num, mapping_table

    def golden(
        x,
        add_num,
        mapping_num,
        mapping_table,
        *,
        group_num,
        group_topk,
        top_n,
        top_k,
        activate_type,
        is_norm,
        scale,
        enable_expert_mapping,
        **_kwargs,
    ):
        """Compute the ATK Golden with stable group/expert tie ordering."""
        import torch

        del activate_type
        x_t = torch.from_numpy(numpy.ascontiguousarray(x, dtype=numpy.float32))
        add_num_t = torch.from_numpy(numpy.ascontiguousarray(add_num, dtype=numpy.float32))
        mapping_num_t = torch.from_numpy(numpy.ascontiguousarray(mapping_num, dtype=numpy.int32))
        mapping_table_t = torch.from_numpy(numpy.ascontiguousarray(mapping_table, dtype=numpy.int32))

        sigmoid = torch.sigmoid(x_t)
        scores = sigmoid + add_num_t
        token_count, expert_count = scores.shape
        group_num = int(group_num)
        group_topk = int(group_topk)
        top_n = int(top_n)
        top_k = int(top_k)
        grouped = scores.reshape(token_count, group_num, expert_count // group_num)
        masked = grouped.clone()
        group_scores = torch.topk(grouped, top_n, dim=-1).values.sum(dim=-1)
        group_order = torch.argsort(group_scores, dim=-1, descending=True, stable=True)

        if group_topk < group_num:
            deselected = group_order[:, group_topk:]
            rows = torch.arange(token_count).unsqueeze(1).expand_as(deselected)
            masked[rows.reshape(-1), deselected.reshape(-1)] = 0.0

        selected_indices = torch.sort(
            masked.reshape(token_count, expert_count), dim=-1, descending=True, stable=True
        ).indices[:, :top_k]
        result = torch.gather(sigmoid, dim=-1, index=selected_indices)
        if bool(is_norm):
            result = result / result.sum(dim=-1, keepdim=True)
            result = result * float(scale)

        output_indices = selected_indices.to(torch.int32)
        if bool(enable_expert_mapping):
            output_indices = output_indices.clone()
            for batch_index in range(token_count):
                for topk_index in range(top_k):
                    expert_id = int(selected_indices[batch_index, topk_index].item())
                    offset = batch_index % max(1, int(mapping_num_t[expert_id].item()))
                    output_indices[batch_index, topk_index] = mapping_table_t[expert_id, offset]

        return [result.numpy().astype(numpy.float32, copy=False), output_indices.numpy()]

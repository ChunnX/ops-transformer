"""Performance-only single execution, input controls, and append-only results."""

import csv
import itertools
import json
import os
import random
from pathlib import Path

STATUS_OK = "Executed (accuracy skipped)"
RESULT_FIELDS = (
    "op", "case_id", "variant", "quant_mode", "suite", "phase", "template_mode",
    "layout_q", "layout_kv", "B", "N1", "S1", "S2", "CKV", "K1", "K",
    "cmp_ratio", "residual", "ori_mask_mode", "cmp_mask_mode",
    "ori_win_left", "ori_win_right", "block_size", "page_order",
    "index_distribution", "actual_k1", "actual_k", "return_softmax_lse",
    "fd_build", "fd_triggered", "status", "correctness", "reason",
    "warmup", "repeat", "completed_calls", "parameters",
)


def expand_params(enabled):
    for axes in enabled:
        for values in itertools.product(*axes.values()):
            yield dict(zip(axes, values))


def write_result(operator, params, status, reason="", completed=0):
    path = Path(os.environ["FD_RESULT_CSV"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: params.get(key, "") for key in RESULT_FIELDS}
    row.update(op=operator, status=status, reason=reason, correctness="NOT_CHECKED",
               completed_calls=completed, warmup=int(os.environ.get("FD_WARMUP", "2")),
               repeat=int(os.environ.get("FD_REPEAT", "10")),
               fd_build=os.environ.get("FD_BUILD_TAG", "installed"),
               fd_triggered="UNKNOWN", parameters=json.dumps(params, default=str, sort_keys=True))
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def sparse_values(left, right, count, distribution, page_size, rng):
    count = min(count, max(0, right - left))
    if distribution == "contiguous":
        return list(range(left, left + count))
    if distribution == "random":
        return rng.sample(range(left, right), count)
    if distribution != "cross_page":
        raise ValueError(f"Unknown index distribution: {distribution}")
    if count == 0:
        return []
    pages = list(range(left // page_size, (right + page_size - 1) // page_size))
    rng.shuffle(pages)
    values = []
    for offset in range(page_size):
        for page in pages:
            value = page * page_size + offset
            if left <= value < right:
                values.append(value)
                if len(values) == count:
                    return values
    return values


def configure_inputs(data, params):
    """Apply identical logical sparse indices and page mappings to each format."""
    import torch

    inputs = data.get("op_input", data.get("input"))
    metadata = data["metadata_input"]
    batch, query, kv = (params[name] for name in ("B", "S1", "S2"))
    for branch, capacity, actual, length in (
        ("ori", params["K1"], params["actual_k1"], kv),
        ("cmp", params["K"], params["actual_k"], params["CKV"]),
    ):
        indices = inputs.get(f"{branch}_sparse_indices")
        if capacity and indices is not None:
            mask = params[f"{branch}_mask_mode"]
            rows = torch.full((query, 1, capacity), -1, dtype=torch.int32)
            lengths = torch.zeros((query, 1), dtype=torch.int32)
            rng = random.Random(42)
            for q_index in range(query):
                threshold = kv - query + q_index + 1
                left, right = 0, length
                if mask == 3:
                    right = min(length, threshold // (params["cmp_ratio"] if branch == "cmp" else 1))
                elif mask == 4:
                    win_l, win_r = params["ori_win_left"], params["ori_win_right"]
                    left = 0 if win_l == -1 else max(0, threshold - win_l - 1)
                    right = length if win_r == -1 else min(length, threshold + win_r)
                values = sparse_values(left, right, actual, params["index_distribution"],
                                       params["block_size"], rng)
                rows[q_index, 0, :len(values)] = torch.tensor(values, dtype=torch.int32)
                lengths[q_index, 0] = len(values)
            rows = rows.unsqueeze(0).expand(batch, -1, -1, -1).clone()
            lengths = lengths.unsqueeze(0).expand(batch, -1, -1).clone()
            if params["layout_q"] == "TND":
                rows = rows.reshape(batch * query, 1, capacity)
                lengths = lengths.reshape(batch * query, 1)
            inputs[f"{branch}_sparse_indices"] = rows
            inputs[f"{branch}_topk_length"] = lengths if mask == 0 else None
            metadata[f"{branch}_topk_length"] = lengths if mask == 0 else None
        if params["layout_kv"] == "PA_BBND":
            table, tensor = inputs.get(f"{branch}_block_table"), inputs.get(f"{branch}_kv")
            if table is not None and tensor is not None:
                # Gather old logical pages, then scatter through a fixed permutation.
                ids = table.reshape(-1).to(device=tensor.device, dtype=torch.long)
                logical_pages = tensor.index_select(0, ids)
                count = len(ids)
                order = list(range(count))
                if params["page_order"] == "random":
                    random.Random(42).shuffle(order)
                order = torch.tensor(order, dtype=torch.long, device=tensor.device)
                physical = torch.empty_like(logical_pages)
                physical.index_copy_(0, order, logical_pages)
                inputs[f"{branch}_kv"] = physical
                inputs[f"{branch}_block_table"] = order.reshape(table.shape).to(torch.int32)
    inputs["sinks"] = torch.zeros(params["N1"], dtype=torch.float32)
    for key in ("q_descale", "ori_kv_descale", "cmp_kv_descale"):
        if inputs.get(key) is not None:
            inputs[key].fill_(1.0)
    data.pop("golden_state", None)
    data.pop("cpu_output", None)
    data.pop("cpu_lse", None)


def run_single(operator, params, generate, execute):
    import numpy as np
    import pytest
    import torch
    import torch_npu

    if params.get("unsupported_reason"):
        write_result(operator, params, "UNSUPPORTED", params["unsupported_reason"])
        pytest.skip(params["unsupported_reason"])
    completed = 0
    warmup, repeat = int(os.environ.get("FD_WARMUP", "2")), int(os.environ.get("FD_REPEAT", "10"))
    if warmup < 0 or repeat < 1:
        raise ValueError("FD_WARMUP must be >= 0 and FD_REPEAT must be >= 1")
    try:
        params = dict(params, batch_consistency=False)
        torch_npu.npu.set_device(int(os.environ.get("FD_DEVICE_ID", "0")))
        torch_npu.npu.set_deterministic_level(0)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        data = generate(params, generate_golden=False)
        configure_inputs(data, params)
        for _ in range(warmup + repeat):
            execute(data)
            torch_npu.npu.synchronize()
            completed += 1
        write_result(operator, params, STATUS_OK, completed=completed)
    except Exception as error:
        write_result(operator, params, "NPU ERROR", str(error), completed=completed)
        raise

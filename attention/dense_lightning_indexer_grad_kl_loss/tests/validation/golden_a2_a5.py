"""CPU/NPU reference for DenseLightningIndexerGradKLLoss migration validation.

This file was derived from DSA_LOSS/golden.py. The original file is intentionally
left unchanged. This version fixes multi-batch TND generation, preserves FP32
weight gradients, and computes the four softmax statistics from the real inputs.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch


TENSOR_INPUTS = (
    "query",
    "key",
    "query_index",
    "key_index",
    "weight",
    "softmax_max",
    "softmax_sum",
    "softmax_max_index",
    "softmax_sum_index",
    "query_rope",
    "key_rope",
)
OUTPUTS = ("d_query_index", "d_key_index", "d_weight", "loss")


def torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("fp16", "float16"):
        return torch.float16
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp32", "float32", "float"):
        return torch.float32
    if normalized in ("int64",):
        return torch.int64
    raise ValueError(f"unsupported dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int64: "int64",
    }
    if dtype not in names:
        raise ValueError(f"unsupported dtype: {dtype}")
    return names[dtype]


def cumulative(lengths: Sequence[int]) -> List[int]:
    total = 0
    result = []
    for length in lengths:
        total += int(length)
        result.append(total)
    return result


def _balanced_lengths(batch: int, maximum: int, rng: random.Random) -> List[int]:
    values = [rng.randint(1, maximum) for _ in range(batch)]
    values[rng.randrange(batch)] = maximum
    return values


def resolve_lengths(case: Mapping[str, object], seed: int) -> Tuple[List[int], List[int]]:
    batch = int(case["batch"])
    max_q = int(case["s1"])
    max_k = int(case["s2"])
    if case["layout"] == "BSND":
        return [max_q] * batch, [max_k] * batch

    if "q_lengths" in case:
        q_lengths = [int(value) for value in case["q_lengths"]]
        k_lengths = [int(value) for value in case["k_lengths"]]
    else:
        rng = random.Random(seed ^ 0x5EED5EED)
        q_lengths = _balanced_lengths(batch, max_q, rng)
        k_lengths = _balanced_lengths(batch, max_k, rng)
        zero_position = case.get("zero_position")
        if zero_position is not None:
            positions = {
                "first": 0,
                "middle": batch // 2,
                "last": batch - 1,
            }
            position = positions[str(zero_position)]
            q_lengths[position] = 0
            k_lengths[position] = 0

    if len(q_lengths) != batch or len(k_lengths) != batch:
        raise ValueError(f"{case['id']}: TND lengths must match batch={batch}")
    if any(q < 0 or k < 0 or q > max_q or k > max_k for q, k in zip(q_lengths, k_lengths)):
        raise ValueError(f"{case['id']}: TND length is outside declared maximum")
    if any(q > k for q, k in zip(q_lengths, k_lengths)):
        raise ValueError(f"{case['id']}: every TND batch must satisfy q_length <= k_length")
    if sum(q_lengths) == 0 or sum(k_lengths) == 0:
        raise ValueError(f"{case['id']}: total TND length must be non-zero")
    return q_lengths, k_lengths


def _randn(shape: Sequence[int], dtype: torch.dtype, generator: torch.Generator, scale: float) -> torch.Tensor:
    return torch.randn(tuple(shape), dtype=dtype, generator=generator) * scale


def make_inputs(case: Mapping[str, object], seed: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    generator = torch.Generator().manual_seed(seed)
    layout = str(case["layout"])
    batch = int(case["batch"])
    n1 = int(case["n1"])
    n1_index = int(case["n1_index"])
    d = int(case.get("d", 128))
    d_index = int(case.get("d_index", 128))
    dr = int(case.get("dr", 64))
    q_dtype = torch_dtype(str(case["q_dtype"]))
    weight_dtype = torch_dtype(str(case["weight_dtype"]))
    q_lengths, k_lengths = resolve_lengths(case, seed)

    if layout == "BSND":
        s1, s2 = q_lengths[0], k_lengths[0]
        q_shape = (batch, s1, n1, d)
        k_shape = (batch, s2, n1, d)
        qi_shape = (batch, s1, n1_index, d_index)
        ki_shape = (batch, s2, 1, d_index)
        weight_shape = (batch, s1, n1_index)
        rope_q_shape = (batch, s1, n1, dr)
        rope_k_shape = (batch, s2, n1, dr)
    elif layout == "TND":
        t1, t2 = sum(q_lengths), sum(k_lengths)
        q_shape = (t1, n1, d)
        k_shape = (t2, n1, d)
        qi_shape = (t1, n1_index, d_index)
        ki_shape = (t2, 1, d_index)
        weight_shape = (t1, n1_index)
        rope_q_shape = (t1, n1, dr)
        rope_k_shape = (t2, n1, dr)
    else:
        raise ValueError(f"unsupported layout: {layout}")

    tensors = {
        "query": _randn(q_shape, q_dtype, generator, 0.2),
        "key": _randn(k_shape, q_dtype, generator, 0.2),
        "query_index": _randn(qi_shape, q_dtype, generator, 0.2),
        "key_index": _randn(ki_shape, q_dtype, generator, 0.2),
        "weight": _randn(weight_shape, weight_dtype, generator, 0.1),
    }
    if bool(case["rope"]):
        tensors["query_rope"] = _randn(rope_q_shape, q_dtype, generator, 0.2)
        tensors["key_rope"] = _randn(rope_k_shape, q_dtype, generator, 0.2)

    metadata = {
        "q_lengths": q_lengths,
        "k_lengths": k_lengths,
        "actual_seq_qlen": cumulative(q_lengths),
        "actual_seq_klen": cumulative(k_lengths),
    }
    return tensors, metadata


def _batch_slice(
    tensor: torch.Tensor,
    layout: str,
    batch_index: int,
    starts: Sequence[int],
    ends: Sequence[int],
) -> torch.Tensor:
    if layout == "BSND":
        return tensor[batch_index]
    return tensor[starts[batch_index] : ends[batch_index]]


def _chunk_size(n1: int, n1_index: int, s2: int) -> int:
    # Keep the two largest temporary score tensors below roughly 384 MiB.
    denominator = max(n1, n1_index) * max(s2, 1)
    return max(1, min(512, (96 * 1024 * 1024) // max(denominator, 1)))


def compute_reference(
    tensors: MutableMapping[str, torch.Tensor],
    case: Mapping[str, object],
    metadata: Mapping[str, object],
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    layout = str(case["layout"])
    batch = int(case["batch"])
    n1 = int(case["n1"])
    n1_index = int(case["n1_index"])
    scale = float(case.get("scale", 1.0 / math.sqrt(int(case.get("d", 128)))))
    q_lengths = [int(value) for value in metadata["q_lengths"]]
    k_lengths = [int(value) for value in metadata["k_lengths"]]
    q_ends = cumulative(q_lengths)
    k_ends = cumulative(k_lengths)
    q_starts = [0] + q_ends[:-1]
    k_starts = [0] + k_ends[:-1]
    target = torch.device(device)

    working = {name: tensor.to(target) for name, tensor in tensors.items()}
    q_dtype = working["query"].dtype
    weight_dtype = working["weight"].dtype
    if bool(case["rope"]):
        query_total = torch.cat((working["query"], working["query_rope"]), dim=-1)
        key_total = torch.cat((working["key"], working["key_rope"]), dim=-1)
    else:
        query_total = working["query"]
        key_total = working["key"]

    d_query_index = torch.zeros_like(working["query_index"])
    d_key_index = torch.zeros_like(working["key_index"])
    d_weight = torch.zeros(working["weight"].shape, dtype=weight_dtype, device=target)
    loss = torch.zeros((1,), dtype=torch.float32, device=target)

    if layout == "BSND":
        softmax_max = torch.empty((batch, n1, q_lengths[0], 1), dtype=torch.float32, device=target)
        softmax_sum = torch.empty_like(softmax_max)
        softmax_max_index = torch.empty((batch, 1, q_lengths[0]), dtype=torch.float32, device=target)
        softmax_sum_index = torch.empty_like(softmax_max_index)
    else:
        softmax_max = torch.empty((n1, sum(q_lengths), 1), dtype=torch.float32, device=target)
        softmax_sum = torch.empty_like(softmax_max)
        softmax_max_index = torch.empty((1, sum(q_lengths)), dtype=torch.float32, device=target)
        softmax_sum_index = torch.empty_like(softmax_max_index)

    for batch_index in range(batch):
        q_len, k_len = q_lengths[batch_index], k_lengths[batch_index]
        if q_len == 0:
            continue
        q_batch = _batch_slice(query_total, layout, batch_index, q_starts, q_ends)
        k_batch = _batch_slice(key_total, layout, batch_index, k_starts, k_ends)
        qi_batch = _batch_slice(working["query_index"], layout, batch_index, q_starts, q_ends)
        ki_batch = _batch_slice(working["key_index"], layout, batch_index, k_starts, k_ends)
        w_batch = _batch_slice(working["weight"], layout, batch_index, q_starts, q_ends)
        chunk = _chunk_size(n1, n1_index, k_len)

        for q_start in range(0, q_len, chunk):
            q_end = min(q_start + chunk, q_len)
            q_slice = q_batch[q_start:q_end]
            qi_slice = qi_batch[q_start:q_end]
            w_slice = w_batch[q_start:q_end]
            row_ids = torch.arange(q_start, q_end, device=target).unsqueeze(-1)
            col_ids = torch.arange(k_len, device=target).unsqueeze(0)
            valid = col_ids <= (k_len - q_len + row_ids)

            p_logits = torch.einsum("snd,tnd->nst", q_slice.float(), k_batch.float()) * scale
            p_logits = p_logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
            p_max = p_logits.max(dim=-1).values
            p_exp = torch.exp(p_logits - p_max.unsqueeze(-1))
            p_sum = p_exp.sum(dim=-1)
            p_reduce = (p_exp / p_sum.unsqueeze(-1)).sum(dim=0) / float(n1)

            score = torch.einsum("snd,tnd->nst", qi_slice.float(), ki_batch.float()).relu()
            weighted = score * w_slice.transpose(0, 1).unsqueeze(-1).float()
            sy_logits = weighted.sum(dim=0).masked_fill(~valid, float("-inf"))
            sy_max = sy_logits.max(dim=-1).values
            sy_exp = torch.exp(sy_logits - sy_max.unsqueeze(-1))
            sy_sum = sy_exp.sum(dim=-1)
            sy = sy_exp / sy_sum.unsqueeze(-1)

            p_clip = torch.clamp(p_reduce, min=1e-8)
            sy_clip = torch.clamp(sy, min=1e-8)
            loss += (p_reduce * (torch.log(p_clip) - torch.log(sy_clip))).sum()

            d_i = sy - p_reduce
            d_weight_slice = (d_i.unsqueeze(1) * score.transpose(0, 1)).sum(dim=-1)
            relu_grad = (
                d_i.unsqueeze(0)
                * w_slice.transpose(0, 1).unsqueeze(-1).float()
                * (score > 0).float()
            ).to(q_dtype)
            repeated_key = ki_batch.repeat(1, n1_index, 1)
            dq = torch.einsum("nst,tnd->snd", relu_grad.float(), repeated_key.float()).to(q_dtype)
            dk = torch.einsum("nst,snd->tnd", relu_grad.float(), qi_slice.float())
            dk = dk.sum(dim=1, keepdim=True).to(q_dtype)

            if layout == "BSND":
                d_query_index[batch_index, q_start:q_end] = dq
                d_key_index[batch_index] += dk
                d_weight[batch_index, q_start:q_end] = d_weight_slice.to(weight_dtype)
                softmax_max[batch_index, :, q_start:q_end, 0] = p_max
                softmax_sum[batch_index, :, q_start:q_end, 0] = p_sum
                softmax_max_index[batch_index, 0, q_start:q_end] = sy_max
                softmax_sum_index[batch_index, 0, q_start:q_end] = sy_sum
            else:
                global_q_start = q_starts[batch_index] + q_start
                global_q_end = q_starts[batch_index] + q_end
                d_query_index[global_q_start:global_q_end] = dq
                d_key_index[k_starts[batch_index] : k_ends[batch_index]] += dk
                d_weight[global_q_start:global_q_end] = d_weight_slice.to(weight_dtype)
                softmax_max[:, global_q_start:global_q_end, 0] = p_max
                softmax_sum[:, global_q_start:global_q_end, 0] = p_sum
                softmax_max_index[0, global_q_start:global_q_end] = sy_max
                softmax_sum_index[0, global_q_start:global_q_end] = sy_sum

    stats = {
        "softmax_max": softmax_max.cpu(),
        "softmax_sum": softmax_sum.cpu(),
        "softmax_max_index": softmax_max_index.cpu(),
        "softmax_sum_index": softmax_sum_index.cpu(),
    }
    outputs = {
        "d_query_index": d_query_index.cpu(),
        "d_key_index": d_key_index.cpu(),
        "d_weight": d_weight.cpu(),
        "loss": loss.cpu(),
    }
    for name, tensor in {**stats, **outputs}.items():
        if not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"{case['id']}: {name} contains NaN or Inf")
    return stats, outputs


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        return contiguous.view(torch.uint16).numpy().tobytes()
    return contiguous.numpy().tobytes()


def save_tensor(path: Path, tensor: torch.Tensor) -> str:
    data = tensor_bytes(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_tensor(path: Path, shape: Sequence[int], dtype: str) -> torch.Tensor:
    target_dtype = torch_dtype(dtype)
    if target_dtype == torch.bfloat16:
        raw = np.fromfile(path, dtype=np.uint16)
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(tuple(shape))
    numpy_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int64: np.int64,
    }[target_dtype]
    return torch.from_numpy(np.fromfile(path, dtype=numpy_dtype).copy()).reshape(tuple(shape))

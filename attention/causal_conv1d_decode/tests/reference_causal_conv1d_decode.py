#!/usr/bin/env python3
"""CPU reference for CausalConv1dDecode update/decode semantics.

The reference covers non-MTP decode x=[B,D], fixed-batch decode x=[B,S,D],
and vLLM-style flattened update x=[T,D] with query_start_loc=[B+1]. It
intentionally keeps index tensors int32 for this project even though vLLM's
OpDef uses int64.
"""

from __future__ import annotations

import numpy as np


def bf16_round_f32(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x00008000)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _ranges_from_x(x: np.ndarray, query_start_loc: np.ndarray | None) -> tuple[list[tuple[int, int]], np.ndarray]:
    if x.ndim == 3:
        batch, seq_len, dim = x.shape
        flat_x = x.reshape(batch * seq_len, dim)
        ranges = [(b * seq_len, (b + 1) * seq_len) for b in range(batch)]
        return ranges, flat_x
    if x.ndim == 2 and query_start_loc is None:
        batch, _ = x.shape
        ranges = [(b, b + 1) for b in range(batch)]
        return ranges, x
    if x.ndim == 2 and query_start_loc is not None:
        q = query_start_loc.astype(np.int64)
        ranges = [(int(q[b]), int(q[b + 1])) for b in range(len(q) - 1)]
        return ranges, x
    raise ValueError("x must be [B,D], [B,S,D], or [T,D] with query_start_loc")


def causal_conv1d_decode_ref(
    x: np.ndarray,
    weight: np.ndarray,
    conv_states: np.ndarray,
    query_start_loc: np.ndarray | None = None,
    cache_indices: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    num_accepted_tokens: np.ndarray | None = None,
    activation_mode: int = 0,
    pad_slot_id: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    width, dim = weight.shape
    if width != 4:
        raise ValueError("width must be 4")
    state_len = conv_states.shape[1]
    ranges, flat_x = _ranges_from_x(x, query_start_loc)
    y = np.zeros_like(flat_x, dtype=np.float32)
    out_state = conv_states.copy().astype(np.float32)

    for b, (start, end) in enumerate(ranges):
        slot = b if cache_indices is None else int(cache_indices[b])
        if slot == pad_slot_id:
            continue
        token_count = end - start
        accepted = token_count if num_accepted_tokens is None else int(num_accepted_tokens[b])
        accepted = max(0, min(accepted, token_count))
        hist_base = 0
        if num_accepted_tokens is not None:
            hist_base = max(0, min(accepted - 1, state_len - (width - 1)))
        for d in range(dim):
            for s in range(token_count):
                acc = 0.0 if bias is None else float(bias[d])
                for k in range(width):
                    rel = s - (width - 1 - k)
                    val = flat_x[start + rel, d] if rel >= 0 else conv_states[slot, hist_base + rel + width - 1, d]
                    acc += float(val) * float(weight[k, d])
                if activation_mode == 1:
                    acc = acc / (1.0 + np.exp(-acc))
                y[start + s, d] = acc

        if num_accepted_tokens is not None and state_len >= token_count + 2:
            out_state[slot, 0:2, :] = conv_states[slot, hist_base + 1 : hist_base + 3, :]
            if token_count > 0:
                out_state[slot, 2 : 2 + token_count, :] = flat_x[start:end, :]
            continue

        hist = conv_states[slot, hist_base : hist_base + width - 1, :]
        if accepted > 0:
            hist = np.concatenate([hist, flat_x[start : start + accepted, :]], axis=0)
        out_state[slot, 0 : width - 1, :] = hist[-(width - 1) :, :]

    y = bf16_round_f32(y)
    if x.ndim == 3:
        y = y.reshape(x.shape)
    return y, bf16_round_f32(out_state)


def smoke() -> None:
    rng = np.random.default_rng(20260614)
    batch, seq_len, dim = 2, 4, 16
    width = 4
    state_len = width - 1 + seq_len
    x3 = rng.normal(size=(batch, seq_len, dim)).astype(np.float32)
    weight = (rng.normal(size=(width, dim)) * 0.5).astype(np.float32)
    states = rng.normal(size=(8, state_len, dim)).astype(np.float32)
    cache = np.array([5, 7], dtype=np.int32)
    bias = (rng.normal(size=(dim,)) * 0.1).astype(np.float32)
    accepted = np.array([seq_len, 2], dtype=np.int32)
    y3, new_state3 = causal_conv1d_decode_ref(x3, weight, states, None, cache, bias, accepted)
    x1 = rng.normal(size=(batch, dim)).astype(np.float32)
    states1 = rng.normal(size=(8, state_len, dim)).astype(np.float32)
    y1, new_state1 = causal_conv1d_decode_ref(x1, weight, states1, None, cache, bias, None)
    q = np.array([0, 2, 3], dtype=np.int32)
    x2 = rng.normal(size=(3, dim)).astype(np.float32)
    states2 = rng.normal(size=(8, state_len, dim)).astype(np.float32)
    accepted2 = np.array([1, 1], dtype=np.int32)
    y2, new_state2 = causal_conv1d_decode_ref(x2, weight, states2, q, cache, bias, accepted2)
    print(
        "width",
        width,
        "3d",
        y3.shape,
        float(y3.sum() + new_state3.sum()),
        "bd",
        y1.shape,
        float(y1.sum() + new_state1.sum()),
        "2d",
        y2.shape,
        float(y2.sum() + new_state2.sum()),
    )


if __name__ == "__main__":
    smoke()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import math
import numpy as np


def _window_for_head(window_size, head):
    values = list(window_size)
    if len(values) == 3:
        return values
    return values[head * 3 : head * 3 + 3]


def _tile_window_indices(seq_shape, query_seq, window_size):
    tile_shape = (6, 8, 8)
    tile_tokens = math.prod(tile_shape)
    tile_grid = tuple(a // b for a, b in zip(seq_shape, tile_shape))
    tile_id = query_seq // tile_tokens
    tile_t = tile_id // (tile_grid[1] * tile_grid[2])
    rem = tile_id % (tile_grid[1] * tile_grid[2])
    tile_h = rem // tile_grid[2]
    tile_w = rem % tile_grid[2]

    starts = []
    for center, win, limit in zip((tile_t, tile_h, tile_w), window_size, tile_grid):
        if limit <= win:
            starts.append(0)
        else:
            starts.append(min(max(center - win // 2, 0), limit - win))

    indices = []
    for tt in range(starts[0], min(starts[0] + window_size[0], tile_grid[0])):
        for hh in range(starts[1], min(starts[1] + window_size[1], tile_grid[1])):
            for ww in range(starts[2], min(starts[2] + window_size[2], tile_grid[2])):
                base = ((tt * tile_grid[1] + hh) * tile_grid[2] + ww) * tile_tokens
                indices.extend(range(base, base + tile_tokens))
    return indices


def sliding_tile_attention_reference(q, k, v, window_size, text_length=0, has_text=True, seq_shape=(6, 8, 8)):
    batch, heads, seq_len, dim = q.shape
    image_len = math.prod(seq_shape)
    real_text = text_length if has_text else 0
    valid_len = image_len + real_text
    out = np.zeros_like(q, dtype=np.float32)
    scale = 1.0 / math.sqrt(dim)
    for b in range(batch):
        for h in range(heads):
            head_window = _window_for_head(window_size, h)
            for s in range(valid_len):
                if has_text and s >= image_len:
                    kv_indices = list(range(valid_len))
                else:
                    kv_indices = [idx for idx in _tile_window_indices(seq_shape, s, head_window) if idx < image_len]
                    if has_text:
                        kv_indices.extend(range(image_len, valid_len))
                scores = (q[b, h, s] @ k[b, h, kv_indices].T).astype(np.float32) * scale
                probs = np.exp(scores - np.max(scores))
                probs = probs / probs.sum()
                out[b, h, s] = probs @ v[b, h, kv_indices].astype(np.float32)
    return out.astype(q.dtype)


def test_reference_image_only_smoke():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 2, 384, 16)).astype(np.float32)
    k = rng.normal(size=(1, 2, 384, 16)).astype(np.float32)
    v = rng.normal(size=(1, 2, 384, 16)).astype(np.float32)
    y = sliding_tile_attention_reference(q, k, v, [1, 1, 1], has_text=False, seq_shape=(6, 8, 8))
    assert y.shape == q.shape
    assert np.isfinite(y).all()


def test_reference_text_and_per_head_windows_smoke():
    rng = np.random.default_rng(13)
    q = rng.normal(size=(1, 2, 386, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 386, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 386, 8)).astype(np.float32)
    y = sliding_tile_attention_reference(q, k, v, [1, 1, 1, 1, 1, 1], text_length=2, has_text=True,
                                         seq_shape=(6, 8, 8))
    assert y.shape == q.shape
    assert np.isfinite(y[:, :, :386]).all()

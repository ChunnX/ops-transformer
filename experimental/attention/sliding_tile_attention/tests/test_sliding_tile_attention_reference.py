# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import math
import os
import numpy as np
import pytest


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
                q_vec = q[b, h, s].astype(np.float32)
                k_mat = k[b, h, kv_indices].astype(np.float32)
                scores = (q_vec @ k_mat.T) * scale
                probs = np.exp(scores - np.max(scores))
                probs = probs / probs.sum()
                out[b, h, s] = probs @ v[b, h, kv_indices].astype(np.float32)
    return out.astype(q.dtype)


CASES = [
    {
        "name": "image_only_single_tile",
        "shape": (1, 2, 384, 16),
        "window_size": [1, 1, 1],
        "text_length": 0,
        "has_text": False,
        "seq_shape": (6, 8, 8),
        "seed": 7,
    },
    {
        "name": "text_and_per_head_windows",
        "shape": (1, 2, 770, 8),
        "window_size": [1, 1, 1, 1, 2, 1],
        "text_length": 2,
        "has_text": True,
        "seq_shape": (6, 16, 8),
        "seed": 13,
    },
]


def _make_inputs(case, dtype=np.float32):
    rng = np.random.default_rng(case["seed"])
    q = rng.normal(loc=0.0, scale=0.2, size=case["shape"]).astype(dtype)
    k = rng.normal(loc=0.0, scale=0.2, size=case["shape"]).astype(dtype)
    v = rng.normal(loc=0.0, scale=0.2, size=case["shape"]).astype(dtype)
    return q, k, v


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_reference_smoke(case):
    q, k, v = _make_inputs(case)
    y = sliding_tile_attention_reference(
        q,
        k,
        v,
        case["window_size"],
        text_length=case["text_length"],
        has_text=case["has_text"],
        seq_shape=case["seq_shape"],
    )
    assert y.shape == q.shape
    assert np.isfinite(y).all()


def _import_torch_npu_or_skip():
    try:
        import torch
        import torch_npu
    except ImportError as exc:
        pytest.skip(f"torch/torch_npu is required for the NPU comparison: {exc}")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU device is not available in this environment")

    torch_npu.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = False
    return torch


def _lookup_attr(root, dotted_path):
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    schemas = getattr(obj, "_schemas", None)
    if schemas == {}:
        raise AttributeError(f"{dotted_path} has no registered schemas")
    return obj


def _candidate_npu_ops(torch):
    explicit = os.getenv("SLIDING_TILE_ATTENTION_NPU_OP")
    if explicit:
        return [explicit]
    return [
        "torch_npu.npu_sliding_tile_attention",
        "torch_npu._npu_sliding_tile_attention",
        "torch.ops.npu_ops_transformer.sliding_tile_attention",
        "torch.ops.npu_ops_transformer_ext.sliding_tile_attention",
        "torch.ops.npu.SlidingTileAttention",
        "torch.ops.npu.sliding_tile_attention",
    ]


def _run_sliding_tile_attention_npu(torch, q, k, v, case):
    roots = {"torch": torch}
    try:
        import torch_npu

        roots["torch_npu"] = torch_npu
    except ImportError:
        pass

    lookup_errors = []
    call_errors = []
    seq_shape = "x".join(str(dim) for dim in case["seq_shape"])
    for path in _candidate_npu_ops(torch):
        root_name, _, attr_path = path.partition(".")
        try:
            op = _lookup_attr(roots[root_name], attr_path)
        except (AttributeError, KeyError, RuntimeError) as exc:
            lookup_errors.append(f"{path}: {exc}")
            continue

        call_forms = [
            ((q, k, v, case["window_size"], case["text_length"], case["has_text"], seq_shape), {}),
            (
                (q, k, v),
                {
                    "window_size": case["window_size"],
                    "text_length": case["text_length"],
                    "has_text": case["has_text"],
                    "seq_shape": seq_shape,
                },
            ),
        ]
        for args, kwargs in call_forms:
            try:
                out = op(*args, **kwargs)
                return out[0] if isinstance(out, tuple) else out
            except TypeError as exc:
                call_errors.append(f"{path}: {exc}")
                continue
            except RuntimeError as exc:
                if "does not exist" in str(exc) or "No such operator" in str(exc):
                    call_errors.append(f"{path}: {exc}")
                    continue
                raise

    reason = "No callable SlidingTileAttention NPU entry was found."
    if lookup_errors or call_errors:
        reason += " Tried: " + " | ".join(lookup_errors + call_errors)
    pytest.skip(reason)


@pytest.mark.npu
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_npu_matches_cpu_reference(case):
    torch = _import_torch_npu_or_skip()
    q_np, k_np, v_np = _make_inputs(case, dtype=np.float16)
    expected = sliding_tile_attention_reference(
        q_np,
        k_np,
        v_np,
        case["window_size"],
        text_length=case["text_length"],
        has_text=case["has_text"],
        seq_shape=case["seq_shape"],
    ).astype(np.float32)

    q = torch.from_numpy(q_np).npu()
    k = torch.from_numpy(k_np).npu()
    v = torch.from_numpy(v_np).npu()
    actual = _run_sliding_tile_attention_npu(torch, q, k, v, case)
    torch.npu.synchronize()

    actual_np = actual.cpu().float().numpy()
    np.testing.assert_allclose(actual_np, expected, rtol=5e-2, atol=5e-2)

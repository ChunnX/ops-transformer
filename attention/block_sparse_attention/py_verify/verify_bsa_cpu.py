#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block Sparse Attention CPU 参考实现，并与 NPU 测试落盘结果对比。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "bsa_output"


def kv_head_for_q_head(q_head: int, num_heads: int, num_kv_heads: int) -> int:
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    group_size = num_heads // num_kv_heads
    return q_head // group_size


def softmax_scores(scores: np.ndarray, inner_precise: int) -> np.ndarray:
    if inner_precise in (1, 4):
        return scores.astype(np.float16).astype(np.float32)
    return scores.astype(np.float32)


def online_softmax_row(
    q_vec: np.ndarray,
    kv_slices: Iterable[tuple[np.ndarray, np.ndarray]],
    scale: float,
    inner_precise: int,
) -> np.ndarray:
    """对单个 Q token 在激活 KV 块上做 Online Softmax + PV。"""
    gm = -np.inf
    gl = 0.0
    o_acc = np.zeros(q_vec.shape[0], dtype=np.float32)
    initialized = False

    for k_block, v_block in kv_slices:
        scores = (q_vec.astype(np.float32) @ k_block.astype(np.float32).T) * scale
        scores_softmax = softmax_scores(scores, inner_precise)

        lm = float(np.max(scores_softmax))
        if not initialized:
            gm = lm
            exp_scores = np.exp(scores_softmax - gm)
            gl = float(np.sum(exp_scores))
            o_acc = exp_scores @ v_block.astype(np.float32)
            initialized = True
            continue

        hm = max(gm, lm)
        dm = float(np.exp(gm - hm))
        gm = hm
        exp_scores = np.exp(scores_softmax - gm)
        ll = float(np.sum(exp_scores))
        pv = exp_scores @ v_block.astype(np.float32)
        o_acc = o_acc * dm + pv
        gl = dm * gl + ll

    if not initialized or gl == 0.0:
        return np.zeros(q_vec.shape[0], dtype=np.float16)
    return (o_acc / gl).astype(np.float16)


def block_sparse_attention_cpu(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    block_sparse_mask: np.ndarray,
    batch: int,
    q_seqlen: int,
    kv_seqlen: int,
    block_shape_x: int,
    block_shape_y: int,
    scale: float,
    inner_precise: int,
) -> np.ndarray:
    total_q_tokens, num_heads, head_dim = query.shape
    _, num_kv_heads, _ = key.shape
    q_block_num = (q_seqlen + block_shape_x - 1) // block_shape_x
    kv_block_num = (kv_seqlen + block_shape_y - 1) // block_shape_y
    output = np.zeros((total_q_tokens, num_heads, head_dim), dtype=np.float16)

    for b in range(batch):
        q_base = b * q_seqlen
        kv_base = b * kv_seqlen
        for h in range(num_heads):
            kv_h = kv_head_for_q_head(h, num_heads, num_kv_heads)
            for q_local in range(q_seqlen):
                q_global = q_base + q_local
                q_vec = query[q_global, h]
                q_block = q_local // block_shape_x

                kv_slices: list[tuple[np.ndarray, np.ndarray]] = []
                for kv_block in range(kv_block_num):
                    if block_sparse_mask[b, h, q_block, kv_block] == 0:
                        continue
                    kv_start = kv_block * block_shape_y
                    kv_end = min(kv_start + block_shape_y, kv_seqlen)
                    k_block = key[kv_base + kv_start : kv_base + kv_end, kv_h]
                    v_block = value[kv_base + kv_start : kv_base + kv_end, kv_h]
                    kv_slices.append((k_block, v_block))

                output[q_global, h] = online_softmax_row(q_vec, kv_slices, scale, inner_precise)
    return output


def _load_binary_tensor(
    output_dir: Path,
    metadata: dict,
    file_key: str,
    shape_key: str,
    dtype: np.dtype,
) -> np.ndarray:
    if file_key not in metadata or shape_key not in metadata:
        raise KeyError(f"metadata missing {file_key}/{shape_key}")
    bin_path = output_dir / metadata[file_key]
    if not bin_path.is_file():
        raise FileNotFoundError(f"tensor file not found: {bin_path}")
    shape = tuple(int(x) for x in metadata[shape_key])
    data = np.fromfile(bin_path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{metadata[file_key]} size mismatch: file={data.size}, metadata={expected}")
    return data.reshape(shape)


def load_inputs(output_dir: Path, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_dtype = np.float32 if metadata.get("input_dtype", "float32") == "float32" else np.float16
    mask_dtype = np.int8 if metadata.get("block_sparse_mask_dtype", "int8") == "int8" else np.int8

    query = _load_binary_tensor(output_dir, metadata, "query_file", "query_shape", input_dtype)
    key = _load_binary_tensor(output_dir, metadata, "key_file", "key_shape", input_dtype)
    value = _load_binary_tensor(output_dir, metadata, "value_file", "value_shape", input_dtype)
    block_sparse_mask = _load_binary_tensor(
        output_dir, metadata, "block_sparse_mask_file", "block_sparse_mask_shape", mask_dtype
    )

    # NPU 算子输入为 fp16，CPU 参考按相同精度计算
    return (
        query.astype(np.float16),
        key.astype(np.float16),
        value.astype(np.float16),
        block_sparse_mask,
    )


def load_npu_output(output_dir: Path, metadata: dict) -> np.ndarray:
    return _load_binary_tensor(
        output_dir,
        metadata,
        "output_file",
        "output_shape",
        np.float32 if metadata.get("output_dtype", "float32") == "float32" else np.float16,
    )


def compare_outputs(
    cpu_out: np.ndarray,
    npu_out: np.ndarray,
    inner_precise: int,
) -> tuple[bool, dict]:
    cpu_f = cpu_out.astype(np.float32)
    npu_f = npu_out.astype(np.float32)
    diff = np.abs(cpu_f - npu_f)

    if inner_precise == 0:
        atol, rtol = 1e-3, 1e-3
    elif inner_precise == 1:
        atol, rtol = 2e-2, 2e-2
    else:
        atol, rtol = 5e-2, 5e-2

    close = np.isclose(cpu_f, npu_f, atol=atol, rtol=rtol)
    stats = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "mismatch_count": int(np.size(diff) - np.count_nonzero(close)),
        "total_count": int(diff.size),
        "atol": atol,
        "rtol": rtol,
    }
    return bool(np.all(close)), stats


def run_verify(output_dir: Path) -> int:
    meta_path = output_dir / "metadata.json"
    if not meta_path.is_file():
        print(f"metadata not found: {meta_path}", file=sys.stderr)
        return 1

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    batch = int(meta["batch"])
    q_seqlen = int(meta["q_seqlen"])
    kv_seqlen = int(meta["kv_seqlen"])
    num_heads = int(meta["num_heads"])
    head_dim = int(meta["head_dim"])
    block_shape_x = int(meta["block_shape_x"])
    block_shape_y = int(meta["block_shape_y"])
    sparsity_mode = int(meta["sparsity_mode"])
    scale = float(meta["scale_value"])
    inner_precise = int(meta["inner_precise"])

    try:
        query, key, value, block_sparse_mask = load_inputs(output_dir, meta)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"failed to load input tensors: {exc}", file=sys.stderr)
        return 1

    print(f"Running CPU block_sparse_attention: batch={batch}, q={q_seqlen}, kv={kv_seqlen}, "
          f"heads={num_heads}, dim={head_dim}, sparsity_mode={sparsity_mode}, inner_precise={inner_precise}")

    cpu_out = block_sparse_attention_cpu(
        query,
        key,
        value,
        block_sparse_mask,
        batch,
        q_seqlen,
        kv_seqlen,
        block_shape_x,
        block_shape_y,
        scale,
        inner_precise,
    )

    try:
        npu_out = load_npu_output(output_dir, meta)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"failed to load NPU output: {exc}", file=sys.stderr)
        return 1

    passed, stats = compare_outputs(cpu_out, npu_out, inner_precise)
    print(f"max_abs_diff={stats['max_abs_diff']:.6g}, mean_abs_diff={stats['mean_abs_diff']:.6g}, "
          f"mismatch={stats['mismatch_count']}/{stats['total_count']}, "
          f"tol(atol={stats['atol']}, rtol={stats['rtol']})")

    if passed:
        print("PASS: CPU reference matches NPU output.")
        return 0

    bad_idx = np.where(~np.isclose(cpu_out.astype(np.float32), npu_out.astype(np.float32),
                                   atol=stats["atol"], rtol=stats["rtol"]))[0][:10]
    print("FAIL: first mismatched flat indices:", bad_idx.tolist())
    for idx in bad_idx[:5]:
        print(f"  idx={idx}: cpu={float(cpu_out.reshape(-1)[idx]):.6g}, "
              f"npu={float(npu_out.reshape(-1)[idx]):.6g}")
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify block_sparse_attention NPU output on CPU.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing metadata.json and dumped tensors from C++ test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_verify(args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

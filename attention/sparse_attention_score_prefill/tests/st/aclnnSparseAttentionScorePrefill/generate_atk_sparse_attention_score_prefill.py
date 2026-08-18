#!/usr/bin/env python3
"""Generate a small, deterministic ATK case file for SparseAttentionScorePrefill.

The executor replaces the randomly generated tensor contents with the same
deterministic tensors on CPU and NPU, while this file describes the ATK shapes,
dtypes, attributes, and precision variants.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CASE_NAME = "sparse_attention_score_prefill_lse"
Q_TOKENS = 16
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 128
TOP_K = 1


def _tensor(name: str, dtype: str, shape: list[int], value_range: list[float | int]) -> dict:
    return {
        "name": name,
        "type": "tensor",
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value_range,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def _attr(name: str, dtype: str, value) -> dict:
    return {
        "name": name,
        "type": "attr",
        "required": True,
        "dtype": dtype,
        "shape": None,
        "range_values": value,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def build_case(case_id: int, inner_precise: int, continuous_kv: bool = False) -> dict:
    scale = 1.0 / math.sqrt(HEAD_DIM)
    case_name = CASE_NAME + ("_continuous" if continuous_kv else "")
    key_shape = (
        [BLOCK_SIZE, KV_HEADS, HEAD_DIM]
        if continuous_kv else [1, BLOCK_SIZE, KV_HEADS, HEAD_DIM]
    )
    inputs = [
        _tensor("query", "bf16", [Q_TOKENS, Q_HEADS, HEAD_DIM], [-0.5, 0.5]),
        _tensor("key", "bf16", key_shape, [-0.5, 0.5]),
        _tensor("value", "bf16", key_shape, [-0.5, 0.5]),
        _tensor("blockTable", "int32", [1, TOP_K], [0, 0]),
        _tensor("k2qRowPtr", "int32", [KV_HEADS, 2], [0, Q_TOKENS]),
        _tensor("k2qQIndices", "int32", [KV_HEADS, Q_TOKENS], [0, Q_TOKENS - 1]),
        _tensor("k2qSlotIndices", "int32", [KV_HEADS, Q_TOKENS], [0, 0]),
        _tensor("actualSeqLengths", "int32", [1], [Q_TOKENS, Q_TOKENS]),
        _tensor("actualSeqLengthsKv", "int32", [1], [BLOCK_SIZE, BLOCK_SIZE]),
        _attr("numKeyValueHeads", "int", KV_HEADS),
        _attr("scaleValue", "double", scale),
        _attr("blockSize", "int", 0 if continuous_kv else BLOCK_SIZE),
        _attr("topK", "int", TOP_K),
        _attr("innerPrecise", "int", inner_precise),
        _attr("softmaxLseFlag", "bool", True),
    ]
    return {
        "id": case_id,
        "name": case_name,
        "aclnn_name": "SparseAttentionScorePrefill",
        "api_type": "ascend_sparse_attention_score_prefill",
        "aclnn_api_type": "ascend_sparse_attention_score_prefill_npu",
        "standard": {
            "acc": {"single_bm": {"type": "high_precision"}},
            "perf": "not_key",
        },
        "inputs": inputs,
        "is_boundary": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("atk_aclnnSparseAttentionScorePrefill.json"),
        help="output ATK JSON case file",
    )
    parser.add_argument(
        "--inner-precise",
        type=int,
        choices=(0, 1, 4),
        nargs="+",
        default=(0, 1, 4),
        help="precision variants to emit (default: 0 1 4)",
    )
    parser.add_argument(
        "--continuous-kv",
        action="store_true",
        help="emit the blockSize=0 concatenated-KV case instead of paged KV",
    )
    args = parser.parse_args()
    cases = [
        build_case(case_id, mode, args.continuous_kv)
        for case_id, mode in enumerate(args.inner_precise)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"generated {len(cases)} ATK cases -> {args.output}")


if __name__ == "__main__":
    main()

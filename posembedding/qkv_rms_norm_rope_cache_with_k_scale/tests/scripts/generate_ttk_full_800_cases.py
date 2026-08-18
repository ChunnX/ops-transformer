#!/usr/bin/env python3
"""Generate the deterministic full-800 TTK matrix with random cache strides.

The semantic matrix mirrors ``/home/r00957730/qkv_test/run_qkv_large_cases.py``
but is intentionally self-contained.  Only cache strides are randomized; no
tensor-view offset field is emitted.  Random padding is reproducible through a
fixed generator seed recorded in each testcase name and remark.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import random


HEAD_DIM = 128
BLOCK_SIZE = 16
STRIDE_SEED = 2026081701
CASE_SEED_BASE = 2026081400
HEAD_CONFIGS = tuple((n_q, n_q // 8, n_q // 8) for n_q in range(8, 65, 8))
CONFIGS = (
    ("rope", "NTD", "NTD"),
    ("rope", "TND", "TND"),
    ("rope", "TND", "NTD"),
    ("mrope", "TND", "TND"),
)
TOKEN_SIZES = (
    1, 2, 15, 16, 17,
    127, 128, 129, 255, 256, 257,
    1023, 1024, 1025,
    4095, 4096, 4097,
    16383, 16384, 16385,
    65535, 131071, 196607, 262142, 262143,
)
SLOT_PATTERNS = ("linear", "reverse", "random", "affine")
BATCH_COUNTS = (1, 2, 4, 8, 16)
MROPE_SECTIONS = ((11, 20, 20), (0, 21, 0), (0, 0, 21), (22, 21, 21), (0, 21, 21))

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "st" / "arch35"
ACLNN_CSV = CASE_DIR / "ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_full_800_random_stride.csv"
PTA_CSV = CASE_DIR / "ttk_pta_qkv_rms_norm_rope_cache_with_k_scale_full_800_random_stride.csv"

ACLNN_FIELDS = (
    "testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
    "tensor_formats", "tensor_storage_shapes", "tensor_view_strides",
    "attributes", "input_data_ranges", "output_tensor_indexes", "output_inplace_indexes",
    "is_enabled", "remark", "soc_series",
)
PTA_FIELDS = (
    "testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
    "tensor_formats", "tensor_storage_shapes", "tensor_view_strides",
    "attributes", "input_data_ranges", "inplace_input_indexes", "is_enabled", "remark",
    "soc_series",
)


def split_batches(token_num: int, batch_count: int) -> tuple[int, ...]:
    base, remainder = divmod(token_num, batch_count)
    return tuple(base + (1 if index < remainder else 0) for index in range(batch_count))


def rope_table_rows(token_num: int, batch_count: int, history_offset: bool, seed: int) -> int:
    lengths = split_batches(token_num, batch_count)
    if history_offset:
        offsets = tuple(257 + 113 * index + seed % 97 for index in range(batch_count))
    else:
        offsets = (0,) * batch_count
    return max(length + offset for length, offset in zip(lengths, offsets)) + 1


def cache_layout(shape: tuple[int, int, int, int], pad: int):
    blocks, heads, block_size, dim = shape
    token_stride = dim + pad
    strides = (
        heads * block_size * token_stride,
        block_size * token_stride,
        token_stride,
        1,
    )
    storage_shape = (blocks, heads, block_size, token_stride)
    return storage_shape, strides


def matrix():
    case_index = 0
    for head_index, (n_q, n_k, n_v) in enumerate(HEAD_CONFIGS):
        for config_index, (scene, layout_qkv, layout_q_out) in enumerate(CONFIGS):
            for token_index, token_num in enumerate(TOKEN_SIZES):
                rotation_index = head_index + config_index + token_index
                slot_pattern = SLOT_PATTERNS[rotation_index % len(SLOT_PATTERNS)]
                batch_count = min(BATCH_COUNTS[rotation_index % len(BATCH_COUNTS)], token_num)
                history_offset = scene == "rope" and rotation_index % 2 != 0
                section = MROPE_SECTIONS[(head_index + token_index) % len(MROPE_SECTIONS)] if scene == "mrope" else ()
                seed = CASE_SEED_BASE + case_index
                stride_rng = random.Random(STRIDE_SEED + case_index)
                # Positive padding guarantees that all three cache tensors are
                # non-contiguous.  K and V must share one descriptor stride.
                # Cache descriptors accepted by the operator require the
                # padded row width to retain an 8-element alignment for K/V.
                # Randomize among legal aligned paddings; scale uses the same
                # family as the already-supported 1+8 layout.
                legal_pads = (8, 16, 24, 32)
                kv_pad = stride_rng.choice(legal_pads)
                pads = (kv_pad, kv_pad, stride_rng.choice(legal_pads))
                source_name = (
                    f"full_{scene}_nq{n_q}_nk{n_k}_nv{n_v}_t{token_num}_"
                    f"{layout_qkv.lower()}_to_{layout_q_out.lower()}_{slot_pattern}"
                )
                meta = (
                    f"__meta_b{batch_count}_h{int(history_offset)}_"
                    f"s{seed}_p{slot_pattern}_rk{pads[0]}_rv{pads[1]}_rs{pads[2]}"
                )
                yield {
                    "index": case_index,
                    "source_name": source_name,
                    "meta": meta,
                    "scene": scene,
                    "layout_qkv": layout_qkv,
                    "layout_q_out": layout_q_out,
                    "token_num": token_num,
                    "n_q": n_q,
                    "n_k": n_k,
                    "n_v": n_v,
                    "batch_count": batch_count,
                    "history_offset": history_offset,
                    "mrope_section": section,
                    "slot_pattern": slot_pattern,
                    "seed": seed,
                    "pads": pads,
                }
                case_index += 1


def common_case(case):
    token_num = case["token_num"]
    n_q, n_k, n_v = case["n_q"], case["n_k"], case["n_v"]
    total_heads = n_q + n_k + n_v
    layout_qkv, layout_q_out = case["layout_qkv"], case["layout_q_out"]
    qkv_shape = (
        (token_num, total_heads, HEAD_DIM)
        if layout_qkv == "TND"
        else (total_heads, token_num, HEAD_DIM)
    )
    q_shape = (
        (token_num, n_q, HEAD_DIM)
        if layout_q_out == "TND"
        else (n_q, token_num, HEAD_DIM)
    )
    capacity = math.ceil((token_num + 32) / BLOCK_SIZE) * BLOCK_SIZE
    block_num = capacity // BLOCK_SIZE
    k_shape = (block_num, n_k, BLOCK_SIZE, HEAD_DIM)
    v_shape = (block_num, n_v, BLOCK_SIZE, HEAD_DIM)
    ks_shape = (block_num, n_k, BLOCK_SIZE, 1)
    k_storage, k_stride = cache_layout(k_shape, case["pads"][0])
    v_storage, v_stride = cache_layout(v_shape, case["pads"][1])
    ks_storage, ks_stride = cache_layout(ks_shape, case["pads"][2])

    if case["scene"] == "rope":
        cos_rows = rope_table_rows(
            token_num, case["batch_count"], case["history_offset"], case["seed"]
        )
        query_shape = (case["batch_count"] + 1,)
        seq_shape = (case["batch_count"],)
        mrope_shape = None
        v_scale_shape = (n_v,)
        k_dtype = "float8_e4m3fn"
        q_dtype = "float8_e4m3fn"
        q_scale_shape = q_shape[:-1]
        q_quant_mode = "PerTokenPerHead"
        q_out_dtype = 36
    else:
        cos_rows = 8192
        query_shape = None
        seq_shape = None
        mrope_shape = (token_num, 3)
        v_scale_shape = (n_v, HEAD_DIM)
        k_dtype = "int8"
        q_dtype = "bfloat16"
        q_scale_shape = None
        q_quant_mode = "NoQuant"
        q_out_dtype = 27

    input_shapes = (
        qkv_shape, (HEAD_DIM,), (HEAD_DIM,), (cos_rows, HEAD_DIM),
        (token_num,), k_shape, v_shape, ks_shape, query_shape, seq_shape,
        (HEAD_DIM, HEAD_DIM), v_scale_shape, mrope_shape,
    )
    input_dtypes = (
        "bfloat16", "float32", "float32", "float32", "int32", k_dtype,
        "float8_e4m3fn", "float32", "int32" if query_shape else None,
        "int32" if seq_shape else None, "bfloat16", "float32", "int32" if mrope_shape else None,
    )
    storage_shapes = (
        None, None, None, None, None, k_storage, v_storage, ks_storage,
        None, None, None, None, None,
    )
    view_strides = (
        None, None, None, None, None, k_stride, v_stride, ks_stride,
        None, None, None, None, None,
    )
    return {
        "input_shapes": input_shapes,
        "input_dtypes": input_dtypes,
        "storage_shapes": storage_shapes,
        "view_strides": view_strides,
        "q_shape": q_shape,
        "q_scale_shape": q_scale_shape,
        "q_dtype": q_dtype,
        "q_quant_mode": q_quant_mode,
        "q_out_dtype": q_out_dtype,
    }


def aclnn_row(case):
    data = common_case(case)
    shapes = data["input_shapes"] + (data["q_shape"], data["q_scale_shape"])
    dtypes = data["input_dtypes"] + (data["q_dtype"], "float32" if data["q_scale_shape"] else None)
    storage_shapes = data["storage_shapes"] + (None, None)
    view_strides = data["view_strides"] + (None, None)
    attrs = {
        "headNums": [case["n_q"], case["n_k"], case["n_v"]],
        "layoutQkv": case["layout_qkv"],
        "layoutQOut": case["layout_q_out"],
        "epsilon": 1e-6,
        "mropeSectionOptional": list(case["mrope_section"]),
        "qQuantMode": data["q_quant_mode"],
    }
    output_indexes = (13, 5, 6, 7) if case["scene"] == "mrope" else (13, 14, 5, 6, 7)
    return {
        "testcase_name": f"aclnn_{case['source_name']}{case['meta']}",
        "api_name": "aclnnQkvRmsNormRopeCacheWithKScale",
        "tensor_view_shapes": repr(shapes),
        "tensor_dtypes": repr(dtypes),
        "tensor_formats": repr(("ND",)),
        "tensor_storage_shapes": repr(storage_shapes),
        "tensor_view_strides": repr(view_strides),
        "attributes": repr(attrs),
        "input_data_ranges": repr(((0, 0),) * len(shapes)),
        "output_tensor_indexes": repr(output_indexes),
        "output_inplace_indexes": repr((5, 6, 7)),
        "is_enabled": "True",
        "remark": (
            f"source={case['source_name']}; stride_seed={STRIDE_SEED}; "
            f"pads(k,v,scale)={case['pads']}"
        ),
        "soc_series": repr(("Ascend950",)),
    }


def pta_row(case):
    data = common_case(case)
    # E2E input generation materializes every non-optional storage explicitly
    # once tensor_storage_shapes is present.  Preserve logical shapes for the
    # ordinary tensors and override only the three cache storages.
    storage_shapes = tuple(
        view_shape if storage_shape is None and view_shape is not None else storage_shape
        for view_shape, storage_shape in zip(data["input_shapes"], data["storage_shapes"])
    )
    attrs = {
        "head_nums": [case["n_q"], case["n_k"], case["n_v"]],
        "layout_qkv": case["layout_qkv"],
        "layout_q_out": case["layout_q_out"],
        "epsilon": 1e-6,
        "mrope_section": list(case["mrope_section"]),
        "q_quant_mode": data["q_quant_mode"],
        "q_out_dtype": data["q_out_dtype"],
    }
    return {
        "testcase_name": f"pta_{case['source_name']}{case['meta']}",
        "api_name": "qkv_ttk_wrapper.pta_inplace",
        "tensor_view_shapes": repr(data["input_shapes"]),
        "tensor_dtypes": repr(data["input_dtypes"]),
        "tensor_formats": repr(("ND",)),
        "tensor_storage_shapes": repr(storage_shapes),
        "tensor_view_strides": repr(data["view_strides"]),
        "attributes": repr(attrs),
        "input_data_ranges": repr(((0, 0),) * len(data["input_shapes"])),
        "inplace_input_indexes": repr((5, 6, 7)),
        "is_enabled": "True",
        "remark": (
            f"source={case['source_name']}; stride_seed={STRIDE_SEED}; "
            f"pads(k,v,scale)={case['pads']}"
        ),
        "soc_series": repr(("Ascend950",)),
    }


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate(cases, aclnn_rows, pta_rows) -> None:
    if len(cases) != 800 or len(aclnn_rows) != 800 or len(pta_rows) != 800:
        raise AssertionError("full matrix must contain exactly 800 semantic cases per CSV")
    if len({case["source_name"] for case in cases}) != 800:
        raise AssertionError("source case names must be unique")
    if {case["token_num"] for case in cases} != set(TOKEN_SIZES):
        raise AssertionError("token matrix mismatch")
    for head in HEAD_CONFIGS:
        selected = [case for case in cases if (case["n_q"], case["n_k"], case["n_v"]) == head]
        if len(selected) != 100:
            raise AssertionError(f"head {head} does not own 100 cases")
    for case in cases:
        data = common_case(case)
        for index, logical_dim in ((5, HEAD_DIM), (6, HEAD_DIM), (7, 1)):
            stride = data["view_strides"][index]
            if stride[-1] != 1 or stride[-2] <= logical_dim:
                raise AssertionError(f"case {case['source_name']} cache {index} is contiguous")
        if min(case["pads"]) <= 0:
            raise AssertionError("all cache pads must be positive")
    if "tensor_view_offsets" in ACLNN_FIELDS or "tensor_view_offsets" in PTA_FIELDS:
        raise AssertionError("offset must not be emitted in this iteration")


def main() -> None:
    cases = list(matrix())
    aclnn_rows = [aclnn_row(case) for case in cases]
    pta_rows = [pta_row(case) for case in cases]
    validate(cases, aclnn_rows, pta_rows)
    write_csv(ACLNN_CSV, ACLNN_FIELDS, aclnn_rows)
    write_csv(PTA_CSV, PTA_FIELDS, pta_rows)
    print(f"generated {len(aclnn_rows)} ACLNN cases: {ACLNN_CSV}")
    print(f"generated {len(pta_rows)} PTA in-place cases: {PTA_CSV}")
    print(f"stride_seed={STRIDE_SEED}; stride_only=true")


if __name__ == "__main__":
    main()

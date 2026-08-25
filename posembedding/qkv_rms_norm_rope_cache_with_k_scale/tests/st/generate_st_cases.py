#!/usr/bin/env python3
"""Generate the single 1200-case ST source for QKV RMSNorm RoPE cache.

The files in ``design/`` are the editable contract.  The five CSVs in
``arch35/`` and ``07_因子值.csv`` are generated products; do not edit them by
hand.  The generator uses only the Python standard library so ``check`` is an
offline gate and does not probe an NPU.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
OP_ROOT = HERE.parents[1]
DESIGN = HERE / "design"
ARCH35 = HERE / "arch35"

API = "aclnnQkvRmsNormRopeCacheWithKScale"
OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale"
TORCH_API = "torch.ops.cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale"
TORCH_INPLACE_API = TORCH_API + "_"

OUTPUTS = {
    "aclnn_l0": ARCH35 / "ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l0_st.csv",
    "aclnn_l1": ARCH35 / "ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l1_st.csv",
    "aclnn_l2": ARCH35 / "negative_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l2_manifest.csv",
    "kernel": ARCH35 / "ttk_kernel_qkv_rms_norm_rope_cache_with_k_scale_st.csv",
    "torch": ARCH35 / "ttk_e2e_qkv_rms_norm_rope_cache_with_k_scale_st.csv",
}
FACTOR_OUTPUT = HERE / "07_因子值.csv"

ACLNN_FIELDS = (
    "testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
    "tensor_formats", "attributes", "output_tensor_indexes",
    "output_inplace_indexes", "input_data_ranges", "soc_series", "priority",
    "is_enabled", "remark", "tensor_storage_shapes", "tensor_view_offsets",
    "tensor_view_strides",
)
KERNEL_FIELDS = (
    "testcase_name", "op_name", "input_shapes", "input_dtypes", "input_formats",
    "output_shapes", "output_dtypes", "output_formats", "attributes",
    "output_inplace_indexes", "input_data_ranges", "soc_series", "priority",
    "is_enabled", "remark",
)
TORCH_FIELDS = (
    "testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
    "tensor_formats", "attributes", "inplace_input_indexes", "input_data_ranges",
    "soc_series", "priority", "is_enabled", "remark", "tensor_storage_shapes",
    "tensor_view_offsets", "tensor_view_strides",
)
L2_FIELDS = (
    "testcase_name", "base_case_id", "scene", "constraint_id", "mutated_field",
    "mutated_value", "expected_stage", "expected_status", "error_contains",
    "priority", "remark",
)
FACTOR_FIELDS = (
    "case_id", "route", "scene", "T", "Nq", "Nk", "Nv", "D",
    "layout_qkv", "layout_q_out", "mrope_section", "block_size", "cache_layout",
    "numeric_profile", "slot_profile", "batch_profile", "variant", "priority",
    "constraint_id", "expected_status", "q_route", "k_route", "whitebox_tags",
    "source_anchor",
)


def load_json(name: str):
    with (DESIGN / name).open(encoding="utf-8") as handle:
        return json.load(handle)


PARAMS = load_json("03_参数定义.yaml")
FACTORS = load_json("04_测试因子.yaml")
CONSTRAINTS = load_json("05_约束定义.yaml")
SOLVER = load_json("06_求解配置.yaml")
WHITEBOX = load_json("08_白盒分支覆盖矩阵.yaml")
SEED = int(SOLVER["seed"])


def stable(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sanitize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def contiguous_strides(shape):
    if shape is None:
        return None
    stride = 1
    result = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= int(dimension)
    return tuple(reversed(result))


def q_route(nq: int) -> str:
    if (nq & 15) == 0:
        return "VECTOR"
    full_batches = nq // 16
    return "Q_TAIL_GLOBAL" if full_batches in (1, 3) else "Q_TAIL_PAIR"


def k_route(nk: int) -> str:
    return {2: "K_BATCH8", 4: "K_BATCH4", 8: "K_BATCH2"}.get(nk, "K_GENERIC")


def default_spec(scene: str) -> dict:
    base = {
        "scene": scene,
        "T": 128,
        "heads": (8, 1, 1),
        "D": 128,
        "layout_qkv": "TND",
        "layout_q_out": "TND",
        "section": (22, 12, 10),
        "block_size": 16,
        "cache_layout": "dense",
        "numeric_profile": "normal",
        "slot_profile": "shuffled",
        "batch_profile": "none",
        "epsilon": 1e-6,
        "variant": "functional",
        "anchor_id": "",
    }
    if scene == "rope":
        base.update({"batch_profile": "batch2", "section": None})
    return base


def normalize_spec(raw: dict) -> dict:
    spec = default_spec(raw["scene"])
    spec.update(raw)
    spec["heads"] = tuple(int(x) for x in spec["heads"])
    if spec.get("section") is not None:
        spec["section"] = tuple(int(x) for x in spec["section"])
    spec["T"] = int(spec["T"])
    spec["block_size"] = int(spec["block_size"])
    spec["epsilon"] = float(spec["epsilon"])
    return spec


def valid(spec: dict) -> bool:
    if spec["T"] <= 0:
        return False
    if spec["numeric_profile"] == "position_1024" and spec["T"] < 3:
        return False
    if spec["slot_profile"] == "slot_cross_block" and spec["T"] < 2:
        return False
    if spec["batch_profile"] == "batch_many" and spec["T"] < 4:
        return False
    if spec["numeric_profile"] in ("scale_edge_qk", "max_finite_qk") and spec["epsilon"] != 0.0:
        return False
    if spec["scene"] != "rope" and (spec["layout_qkv"], spec["layout_q_out"]) != ("TND", "TND"):
        return False
    return True


def factor_values(scene: str) -> dict:
    common = FACTORS["common"]
    values = {
        "T": FACTORS[scene]["T"],
        "heads": [tuple(x) for x in common["heads"]],
        "block_size": common["block_sizes"],
        "cache_layout": common["cache_layouts"],
        "slot_profile": common["slot_profiles"],
        "epsilon": common["epsilons"],
        "numeric_profile": FACTORS[scene]["numeric_profiles"],
    }
    if scene == "rope":
        values["layout_pair"] = [tuple(x) for x in FACTORS[scene]["layouts"]]
        values["batch_profile"] = FACTORS[scene]["batch_profiles"]
    else:
        values["section"] = [tuple(x) for x in FACTORS[scene]["sections"]]
    return values


def set_factor(spec: dict, key: str, value):
    result = dict(spec)
    if key == "layout_pair":
        result["layout_qkv"], result["layout_q_out"] = value
    else:
        result[key] = value
    return normalize_spec(result)


def spec_key(spec: dict) -> str:
    fields = (
        "scene", "T", "heads", "layout_qkv", "layout_q_out", "section",
        "block_size", "cache_layout", "numeric_profile", "slot_profile",
        "batch_profile", "epsilon", "variant",
    )
    return stable([spec.get(field) for field in fields])


def coverage_pairs(spec: dict):
    values = {
        "T": spec["T"],
        "heads": spec["heads"],
        "block_size": spec["block_size"],
        "cache_layout": spec["cache_layout"],
        "slot_profile": spec["slot_profile"],
        "epsilon": spec["epsilon"],
        "numeric_profile": spec["numeric_profile"],
    }
    if spec["scene"] == "rope":
        values["layout_pair"] = (spec["layout_qkv"], spec["layout_q_out"])
        values["batch_profile"] = spec["batch_profile"]
    else:
        values["section"] = spec["section"]
    result = set()
    keys = sorted(values)
    for left, right in itertools.combinations(keys, 2):
        result.add((left, stable(values[left]), right, stable(values[right])))
    return result


def candidate_pool(scene: str) -> list[dict]:
    cached = _CANDIDATE_CACHE.get(scene)
    if cached is not None:
        return [dict(item) for item in cached]
    base = default_spec(scene)
    values = factor_values(scene)
    # The public upper-bound case is intentionally independent.  Pairing a
    # 262144-token allocation with every other factor would turn one resource
    # stress gate into dozens of multi-gigabyte functional cases.
    values["T"] = [value for value in values["T"] if value != 262144]
    candidates = [normalize_spec(base)]
    keys = list(values)
    for key in keys:
        for value in values[key]:
            candidate = set_factor(base, key, value)
            if valid(candidate):
                candidates.append(candidate)
    for left, right in itertools.combinations(keys, 2):
        for left_value in values[left]:
            for right_value in values[right]:
                candidate = set_factor(set_factor(base, left, left_value), right, right_value)
                if valid(candidate):
                    candidates.append(candidate)
    rng = random.Random(SEED + {"mrope_mx": 1, "rope": 2, "mrope": 3}[scene])
    for _ in range(1000):
        candidate = dict(base)
        for key, choices in values.items():
            candidate = set_factor(candidate, key, rng.choice(choices))
        if valid(candidate):
            candidates.append(candidate)
    unique = {}
    for candidate in candidates:
        unique.setdefault(spec_key(candidate), candidate)
    result = [unique[key] for key in sorted(unique)]
    _CANDIDATE_CACHE[scene] = tuple(dict(item) for item in result)
    return result


_CANDIDATE_CACHE = {}


def explicit_specs(scene: str, route: str) -> list[dict]:
    result = []
    if scene == "mrope_mx":
        if route in ("aclnn_l0", "kernel", "aclnn_l1"):
            for anchor in WHITEBOX["anchors"]:
                result.append(normalize_spec({
                    "scene": scene,
                    "T": anchor["T"],
                    "heads": anchor["heads"],
                    "anchor_id": anchor["id"],
                }))
        result.extend(normalize_spec(item) for item in (
            {"scene": scene, "T": 1, "heads": (8, 1, 1)},
            {"scene": scene, "T": 37, "heads": (8, 1, 1)},
            {"scene": scene, "T": 3073, "heads": (8, 1, 1)},
            {"scene": scene, "T": 3, "numeric_profile": "position_1024"},
            {"scene": scene, "T": 2, "slot_profile": "slot_cross_block"},
            {"scene": scene, "T": 128, "cache_layout": "padded_token"},
            {"scene": scene, "T": 128, "cache_layout": "padded_head"},
            {"scene": scene, "T": 128, "cache_layout": "nonzero_offset"},
            {"scene": scene, "T": 128, "cache_layout": "kscale_padded_token"},
            {"scene": scene, "T": 128, "cache_layout": "kscale_padded_head"},
            {"scene": scene, "T": 128, "cache_layout": "mixed_stride"},
            {"scene": scene, "T": 8, "numeric_profile": "zero_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "scale_order"},
            {"scene": scene, "T": 8, "numeric_profile": "tiny_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "saturation_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "nan_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "inf_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "scale_edge_qk", "epsilon": 0.0},
            {"scene": scene, "T": 8, "heads": (16, 2, 2),
             "numeric_profile": "scale_edge_qk", "epsilon": 0.0},
            {"scene": scene, "T": 8, "heads": (24, 3, 3),
             "numeric_profile": "scale_edge_qk", "epsilon": 0.0},
            {"scene": scene, "T": 8, "numeric_profile": "max_finite_qk", "epsilon": 0.0},
            {"scene": scene, "T": 8, "heads": (16, 2, 2),
             "numeric_profile": "max_finite_qk", "epsilon": 0.0},
            {"scene": scene, "T": 8, "heads": (24, 3, 3),
             "numeric_profile": "max_finite_qk", "epsilon": 0.0},
        ))
        if route == "aclnn_l1":
            result.append(normalize_spec({"scene": scene, "T": 262144, "heads": (64, 8, 8)}))
        if route == "torch":
            result = []
            for anchor_id, t, heads in (
                ("WB-QPAIR-ODD-H8", 129, (8, 1, 1)),
                ("WB-KB8-RESIDUAL-H16", 513, (16, 2, 2)),
                ("WB-QGLOBAL-ODD-H24", 129, (24, 3, 3)),
            ):
                for variant in ("functional", "inplace"):
                    result.append(normalize_spec({
                        "scene": scene, "T": t, "heads": heads,
                        "anchor_id": anchor_id, "variant": variant,
                    }))
    elif scene == "rope":
        result.extend(normalize_spec(item) for item in (
            {"scene": scene, "T": 1, "batch_profile": "batch1"},
            {"scene": scene, "T": 8, "batch_profile": "batch2"},
            {"scene": scene, "T": 8, "batch_profile": "batch_many"},
            {"scene": scene, "T": 3073, "batch_profile": "batch2"},
            {"scene": scene, "T": 8, "numeric_profile": "nan_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "inf_qk"},
        ))
        if route == "torch":
            result.extend(normalize_spec({"scene": scene, "T": 8, "variant": variant})
                          for variant in ("functional", "inplace"))
    else:
        result.extend(normalize_spec(item) for item in (
            {"scene": scene, "T": 1},
            {"scene": scene, "T": 3073},
            {"scene": scene, "T": 8, "numeric_profile": "nan_qk"},
            {"scene": scene, "T": 8, "numeric_profile": "inf_qk"},
        ))
        if route == "torch":
            result.extend(normalize_spec({"scene": scene, "T": 8, "variant": variant})
                          for variant in ("functional", "inplace"))
    return [item for item in result if valid(item)]


def choose_specs(scene: str, route: str, count: int) -> list[dict]:
    explicit = explicit_specs(scene, route)
    selected = []
    seen = set()
    for item in explicit:
        key = spec_key(item)
        if key not in seen:
            selected.append(item)
            seen.add(key)
    raw_pool = candidate_pool(scene)
    if route == "torch":
        raw_pool = [dict(item, variant=variant) for item in raw_pool
                    for variant in ("functional", "inplace")]
    pool = [item for item in raw_pool if spec_key(item) not in seen]
    signatures = {spec_key(item): coverage_pairs(item) for item in pool}
    uncovered = set().union(*signatures.values()) if signatures else set()
    for item in selected:
        uncovered.difference_update(coverage_pairs(item))
    # Twenty-four incremental choices establish the high-value pairwise
    # frontier after the explicit branch anchors.  Repeatedly rescanning thousands of candidates for all 400 L1
    # rows adds no new contract value, so the remaining rows use a seeded hash
    # order from the same valid pairwise/random candidate pool.
    greedy_limit = min(count, len(selected) + 24)
    while len(selected) < greedy_limit and pool:
        best_index, best_score = 0, -1
        for index, item in enumerate(pool):
            score = len(signatures[spec_key(item)] & uncovered)
            if score > best_score:
                best_index, best_score = index, score
        item = pool.pop(best_index)
        selected.append(item)
        seen.add(spec_key(item))
        uncovered.difference_update(signatures[spec_key(item)])
    if len(selected) < count:
        pool.sort(key=lambda item: hashlib.sha256(
            f"{SEED}:{route}:{scene}:{spec_key(item)}".encode()).hexdigest())
        selected.extend(pool[:count - len(selected)])
    if len(selected) < count:
        raise RuntimeError(f"insufficient unique {scene}/{route} cases: {len(selected)} < {count}")
    return selected[:count]


def assign_priorities(specs: list[dict], p0: int, p2: int) -> list[dict]:
    if p0 + p2 > len(specs):
        raise ValueError("priority quota exceeds route size")
    result = [dict(item, priority=1) for item in specs]
    p0_order = sorted(range(len(result)), key=lambda index: (
        0 if result[index].get("anchor_id") else 1,
        0 if result[index]["numeric_profile"] in ("scale_edge_qk", "max_finite_qk") else 1,
        index,
    ))
    p0_indexes = set(p0_order[:p0])
    for index in p0_indexes:
        result[index]["priority"] = 0
    p2_order = sorted(
        (index for index in range(len(result)) if index not in p0_indexes),
        key=lambda index: (
            0 if result[index]["T"] == 262144 else 1,
            0 if result[index]["T"] >= 2048 else 1,
            0 if result[index]["numeric_profile"] in ("nan_qk", "inf_qk", "max_finite_qk") else 1,
            -result[index]["T"],
            index,
        ),
    )
    for index in p2_order[:p2]:
        result[index]["priority"] = 2
    return result


def build_positive_specs() -> dict[str, list[dict]]:
    result = {}
    for route in ("aclnn_l0", "aclnn_l1", "kernel", "torch"):
        config = SOLVER["routes"][route]
        all_specs = []
        for scene in ("mrope_mx", "rope", "mrope"):
            all_specs.extend(choose_specs(scene, route, int(config["scenes"][scene])))
        priorities = config["priorities"]
        result[route] = assign_priorities(all_specs, int(priorities["P0"]), int(priorities["P2"]))
    return result


def position_count(spec: dict) -> int:
    if spec["numeric_profile"] == "position_1024":
        return 2048
    return max(2048, int(spec["T"]))


def cache_layout(shape, layout: str, scale: bool = False):
    """Return storage shape, logical strides and offset for one cache view."""
    if shape is None:
        return None, None, None
    storage = list(shape)
    offset = 0
    data_padding = layout in ("padded_token", "padded_head", "mixed_stride") and not scale
    scale_padding = layout in ("kscale_padded_token", "kscale_padded_head", "mixed_stride") and scale
    if data_padding:
        if layout == "padded_head" or layout == "mixed_stride":
            storage[1] += 1
        else:
            storage[2] += 1
    elif scale_padding:
        if layout == "kscale_padded_head":
            storage[1] += 1
        else:
            storage[2] += 1
    elif layout == "nonzero_offset":
        storage[-1] += 1
        offset = 1
        # Non-zero base with compact logical pitch catches runners that flatten
        # the logical view instead of preserving its physical allocation.
        return tuple(storage), contiguous_strides(shape), offset
    return tuple(storage), contiguous_strides(tuple(storage)), offset


def contract(spec: dict):
    t = spec["T"]
    nq, nk, nv = spec["heads"]
    total = nq + nk + nv
    block_size = spec["block_size"]
    block_num = max(4, math.ceil(t / block_size) + 1)
    pos_count = position_count(spec)
    layout_qkv, layout_q_out = spec["layout_qkv"], spec["layout_q_out"]
    qkv_shape = (total, t, 128) if layout_qkv == "NTD" else (t, total, 128)
    q_shape = (nq, t, 128) if layout_q_out == "NTD" else (t, nq, 128)

    is_mx = spec["scene"] == "mrope_mx"
    is_rope = spec["scene"] == "rope"
    q_quant = "MxQuant" if is_mx else ("PerTokenPerHead" if is_rope else "NoQuant")
    k_quant = "MxQuant" if is_mx else "PerTokenPerHead"
    q_dtype = "float8_e4m3fn" if is_mx or is_rope else "bfloat16"
    k_dtype = "float8_e4m3fn" if is_mx or is_rope else "int8"
    scale_dtype = "float8_e8m0" if is_mx else "float32"
    k_scale_shape = ((block_num, nk, block_size, 4) if is_mx else
                     (block_num, nk, block_size, 1))
    q_scale_shape = ((nq, t, 4) if layout_q_out == "NTD" else (t, nq, 4)) if is_mx else (
        ((nq, t) if layout_q_out == "NTD" else (t, nq)) if is_rope else None)

    if is_rope:
        batch = {"batch1": 1, "batch2": 2, "batch_many": 4}[spec["batch_profile"]]
        query_shape, seq_shape = (batch + 1,), (batch,)
        mrope_shape = None
    else:
        query_shape = seq_shape = None
        mrope_shape = (t, 3)
    rotation_shape = None if is_mx else (128, 128)
    v_scale_shape = (nv,) if is_rope else (nv, 128)

    inputs = (
        qkv_shape, (128,), (128,), (pos_count, 128), (t,),
        (block_num, nk, block_size, 128), (block_num, nv, block_size, 128),
        k_scale_shape, query_shape, seq_shape, rotation_shape, v_scale_shape,
        mrope_shape,
    )
    input_dtypes = (
        "bfloat16", "float32", "float32", "float32", "int32", k_dtype,
        "float8_e4m3fn", scale_dtype,
        "int32" if query_shape else None, "int32" if seq_shape else None,
        "bfloat16" if rotation_shape else None, "float32", "int32" if mrope_shape else None,
    )
    outputs = (
        q_shape, q_scale_shape, inputs[5], inputs[6], inputs[7],
    )
    output_dtypes = (q_dtype, "float8_e8m0" if is_mx else ("float32" if is_rope else "float32"),
                     k_dtype, "float8_e4m3fn", scale_dtype)

    storage_shapes = list(inputs)
    view_strides = [contiguous_strides(shape) for shape in inputs]
    view_offsets = [0 if shape is not None else None for shape in inputs]
    for index, scale in ((5, False), (6, False), (7, True)):
        storage_shapes[index], view_strides[index], view_offsets[index] = cache_layout(
            inputs[index], spec["cache_layout"], scale=scale)

    ranges = []
    capacity = block_num * block_size
    for index, shape in enumerate(inputs):
        if shape is None:
            ranges.append((None, None))
        elif index == 4:
            ranges.append((0, capacity - 1))
        elif index in (8, 9, 12):
            ranges.append((0, pos_count - 1))
        else:
            ranges.append((-1, 1))

    return {
        "inputs": inputs,
        "input_dtypes": input_dtypes,
        "outputs": outputs,
        "output_dtypes": output_dtypes,
        "storage_shapes": tuple(storage_shapes),
        "view_strides": tuple(view_strides),
        "view_offsets": tuple(view_offsets),
        "ranges": tuple(ranges),
        "q_quant": q_quant,
        "k_quant": k_quant,
        "q_dtype": q_dtype,
        "block_num": block_num,
    }


def case_name(route: str, scene: str, ordinal: int, spec: dict) -> str:
    prefix = {"aclnn_l0": "al0", "aclnn_l1": "al1", "kernel": "ker", "torch": "tor"}[route]
    anchor = sanitize(spec["anchor_id"]) if spec.get("anchor_id") else ""
    head = spec["heads"][0]
    tokens = [prefix, scene, f"{ordinal:04d}", f"t{spec['T']}", f"h{head}"]
    if anchor:
        tokens.append(anchor)
    tokens.extend((spec["layout_qkv"].lower(), spec["layout_q_out"].lower(),
                   spec["cache_layout"], spec["numeric_profile"], spec["slot_profile"]))
    if spec["scene"] == "rope":
        tokens.append(spec["batch_profile"])
    if route == "torch":
        tokens.append(spec["variant"])
    return "_".join(tokens)


def common_attributes(spec: dict, style: str):
    c = contract(spec)
    section = list(spec["section"]) if spec["section"] is not None else None
    if style == "aclnn":
        return {
            "headNums": list(spec["heads"]), "layoutQkv": spec["layout_qkv"],
            "layoutQOut": spec["layout_q_out"], "epsilon": spec["epsilon"],
            "mropeSectionOptional": section, "qQuantMode": c["q_quant"],
            "kQuantMode": c["k_quant"],
        }
    attrs = {
        "head_nums": list(spec["heads"]), "layout_qkv": spec["layout_qkv"],
        "layout_q_out": spec["layout_q_out"], "epsilon": spec["epsilon"],
        "mrope_section": section, "q_quant_mode": c["q_quant"],
        "k_quant_mode": c["k_quant"],
    }
    if style == "kernel":
        attrs["q_out_dtype"] = 36 if c["q_dtype"] == "float8_e4m3fn" else 27
    else:
        attrs["q_out_dtype"] = c["q_dtype"]
    return attrs


def whitebox_tags(spec: dict) -> str:
    tags = []
    if spec["scene"] == "mrope_mx":
        tags.extend((q_route(spec["heads"][0]), k_route(spec["heads"][1])))
        if spec["T"] & 1:
            tags.append("ODD_TOKEN_RESIDUAL")
        if spec["T"] == 262144:
            tags.append("T_UPPER_BOUND")
        if spec["cache_layout"] != "dense":
            tags.append("PHYSICAL_STORAGE_VIEW")
        if spec["numeric_profile"] != "normal":
            tags.append("NUMERIC_" + spec["numeric_profile"].upper())
    return ";".join(tags)


def render_aclnn(name: str, spec: dict) -> dict:
    c = contract(spec)
    shapes = c["inputs"] + (c["outputs"][0], c["outputs"][1])
    dtypes = c["input_dtypes"] + (c["output_dtypes"][0],
                                   None if c["outputs"][1] is None else c["output_dtypes"][1])
    formats = tuple("ND" if shape is not None else None for shape in shapes)
    storage = c["storage_shapes"] + (c["outputs"][0], c["outputs"][1])
    offsets = c["view_offsets"] + (0, 0 if c["outputs"][1] is not None else None)
    strides = c["view_strides"] + (contiguous_strides(c["outputs"][0]),
                                    contiguous_strides(c["outputs"][1]))
    output_indexes = (13, 14, 5, 6, 7) if c["outputs"][1] is not None else (13, 5, 6, 7)
    return {
        "testcase_name": name, "api_name": API,
        "tensor_view_shapes": repr(shapes), "tensor_dtypes": repr(dtypes),
        "tensor_formats": repr(formats), "attributes": repr(common_attributes(spec, "aclnn")),
        "output_tensor_indexes": repr(output_indexes), "output_inplace_indexes": repr((5, 6, 7)),
        "input_data_ranges": repr(c["ranges"] + ((-1, 1), (-1, 1) if c["outputs"][1] else (None, None))),
        "soc_series": repr(("Ascend950",)), "priority": spec["priority"], "is_enabled": True,
        "remark": f"{spec['scene']} {whitebox_tags(spec)}".strip(),
        "tensor_storage_shapes": repr(storage), "tensor_view_offsets": repr(offsets),
        "tensor_view_strides": repr(strides),
    }


def render_kernel(name: str, spec: dict) -> dict:
    c = contract(spec)
    output_dtypes = tuple(
        dtype if shape is not None else None
        for shape, dtype in zip(c["outputs"], c["output_dtypes"])
    )
    output_formats = tuple("ND" if shape is not None else None for shape in c["outputs"])
    return {
        "testcase_name": name, "op_name": OP_NAME,
        "input_shapes": repr(c["inputs"]), "input_dtypes": repr(c["input_dtypes"]),
        "input_formats": repr(tuple("ND" if x is not None else None for x in c["inputs"])),
        "output_shapes": repr(c["outputs"]), "output_dtypes": repr(output_dtypes),
        "output_formats": repr(output_formats),
        "attributes": repr(common_attributes(spec, "kernel")),
        "output_inplace_indexes": repr((None, None, 5, 6, 7)), "input_data_ranges": repr(c["ranges"]),
        "soc_series": repr(("Ascend950",)), "priority": spec["priority"], "is_enabled": True,
        "remark": f"{spec['scene']} {whitebox_tags(spec)}; direct Kernel uses logical dense tensors".strip(),
    }


def render_torch(name: str, spec: dict) -> dict:
    c = contract(spec)
    api = TORCH_INPLACE_API if spec["variant"] == "inplace" else TORCH_API
    inplace_input_indexes = (5, 6, 7) if spec["variant"] == "inplace" else ()
    return {
        "testcase_name": name, "api_name": api,
        "tensor_view_shapes": repr(c["inputs"]), "tensor_dtypes": repr(c["input_dtypes"]),
        "tensor_formats": repr(tuple("ND" if x is not None else None for x in c["inputs"])),
        "attributes": repr(common_attributes(spec, "torch")),
        "inplace_input_indexes": repr(inplace_input_indexes), "input_data_ranges": repr(c["ranges"]),
        "soc_series": repr(("Ascend950",)), "priority": spec["priority"], "is_enabled": True,
        "remark": f"{spec['variant']} {spec['scene']} {whitebox_tags(spec)}".strip(),
        "tensor_storage_shapes": repr(c["storage_shapes"]),
        "tensor_view_offsets": repr(c["view_offsets"]),
        "tensor_view_strides": repr(c["view_strides"]),
    }


def build_l2(positive: dict[str, list[dict]]):
    rows = []
    base_by_scene = {}
    scene_ordinals = Counter()
    for scene in ("mrope_mx", "rope", "mrope"):
        scene_ordinals.clear()
        for spec in positive["aclnn_l0"]:
            scene_ordinals[spec["scene"]] += 1
            if spec["scene"] == scene:
                base_by_scene[scene] = case_name(
                    "aclnn_l0", scene, scene_ordinals[scene], spec)
                break
    ordinal = 1
    for scene in ("mrope_mx", "rope", "mrope"):
        mutations = CONSTRAINTS["negative_mutations"][scene]
        expected = int(SOLVER["routes"]["aclnn_l2"]["scenes"][scene])
        if len(mutations) != expected:
            raise AssertionError(f"{scene} L2 mutation count {len(mutations)} != {expected}")
        for mutation in mutations:
            rows.append({
                "testcase_name": f"neg_{scene}_{ordinal:03d}_{sanitize(mutation['constraint_id'])}",
                "base_case_id": base_by_scene[scene], "scene": scene,
                "constraint_id": mutation["constraint_id"], "mutated_field": mutation["field"],
                "mutated_value": mutation["value"], "expected_stage": "aclnn_parameter_check",
                "expected_status": mutation["status"], "error_contains": mutation["error"],
                "priority": 0, "remark": "one mutation only; no tiling or kernel launch",
            })
            ordinal += 1
    return rows


def factor_row(name: str, route: str, spec: dict, constraint_id="", expected_status="SUCCESS"):
    nq, nk, nv = spec["heads"]
    return {
        "case_id": name, "route": route, "scene": spec["scene"], "T": spec["T"],
        "Nq": nq, "Nk": nk, "Nv": nv, "D": 128, "layout_qkv": spec["layout_qkv"],
        "layout_q_out": spec["layout_q_out"], "mrope_section": repr(spec["section"]),
        "block_size": spec["block_size"], "cache_layout": spec["cache_layout"],
        "numeric_profile": spec["numeric_profile"], "slot_profile": spec["slot_profile"],
        "batch_profile": spec["batch_profile"], "variant": spec["variant"],
        "priority": spec.get("priority", 0), "constraint_id": constraint_id,
        "expected_status": expected_status,
        "q_route": q_route(nq) if spec["scene"] == "mrope_mx" else "NON_MX",
        "k_route": k_route(nk) if spec["scene"] == "mrope_mx" else "NON_MX",
        "whitebox_tags": whitebox_tags(spec), "source_anchor": spec.get("anchor_id", ""),
    }


def build_products():
    positive = build_positive_specs()
    rows = {}
    factors = []
    for route, specs in positive.items():
        rendered = []
        scene_ordinals = Counter()
        for spec in specs:
            scene_ordinals[spec["scene"]] += 1
            name = case_name(route, spec["scene"], scene_ordinals[spec["scene"]], spec)
            if route.startswith("aclnn"):
                rendered.append(render_aclnn(name, spec))
            elif route == "kernel":
                rendered.append(render_kernel(name, spec))
            else:
                rendered.append(render_torch(name, spec))
            factors.append(factor_row(name, route, spec))
        rows[route] = rendered
    rows["aclnn_l2"] = build_l2(positive)
    for row in rows["aclnn_l2"]:
        base_factor = next(item for item in factors if item["case_id"] == row["base_case_id"])
        spec = normalize_spec({
            "scene": row["scene"], "T": int(base_factor["T"]),
            "heads": (int(base_factor["Nq"]), int(base_factor["Nk"]), int(base_factor["Nv"])),
            "block_size": int(base_factor["block_size"]), "cache_layout": base_factor["cache_layout"],
            "numeric_profile": base_factor["numeric_profile"], "slot_profile": base_factor["slot_profile"],
            "batch_profile": base_factor["batch_profile"],
        })
        spec["priority"] = 0
        factors.append(factor_row(row["testcase_name"], "aclnn_l2", spec,
                                  row["constraint_id"], row["expected_status"]))
    return rows, factors


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate():
    rows, factors = build_products()
    write_csv(OUTPUTS["aclnn_l0"], ACLNN_FIELDS, rows["aclnn_l0"])
    write_csv(OUTPUTS["aclnn_l1"], ACLNN_FIELDS, rows["aclnn_l1"])
    write_csv(OUTPUTS["aclnn_l2"], L2_FIELDS, rows["aclnn_l2"])
    write_csv(OUTPUTS["kernel"], KERNEL_FIELDS, rows["kernel"])
    write_csv(OUTPUTS["torch"], TORCH_FIELDS, rows["torch"])
    write_csv(FACTOR_OUTPUT, FACTOR_FIELDS, factors)
    return rows, factors


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_repr_tuple(row: dict, field: str, length: int | None = None):
    value = ast.literal_eval(row[field])
    if not isinstance(value, tuple):
        raise AssertionError(f"{row['testcase_name']} {field} is not a tuple")
    if length is not None and len(value) != length:
        raise AssertionError(f"{row['testcase_name']} {field} length {len(value)} != {length}")
    return value


def source_gate():
    files_and_tokens = {
        OP_ROOT / "op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h":
            ("CheckParams", "requiredPointers", "qScaleOptional", "kQuantMode"),
        OP_ROOT / "op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp":
            ("SelectMropeMxTokenTile", "TryMakeMropeMxQGlobalTileWaveUbLayout"),
        OP_ROOT / "op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_controller.h":
            ("MakeCoreTokenRange", "FillTile"),
        OP_ROOT / "op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vec.h":
            ("tokensPerBatch", "ProcessQGlobalTileWave",
             "QRmsNormMropeMxD128GlobalTileWaveVfImpl"),
        OP_ROOT / "op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vf.h":
            ("MxQuantCublasScaleRowBatch16DeferredRms", "ProcessKMropeMxD128RowBatch16",
             "QRmsNormMropeMxD128GlobalTileWaveVfImpl"),
    }
    for path, tokens in files_and_tokens.items():
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                raise AssertionError(f"source predicate {token!r} missing from {path}")


def validate(rows, factors):
    config = SOLVER["routes"]
    all_names = []
    for route, path in OUTPUTS.items():
        physical = read_csv(path)
        expected = int(config[route]["total"])
        if len(physical) != expected or len(rows[route]) != expected:
            raise AssertionError(f"{route}: expected {expected}, got {len(physical)}/{len(rows[route])}")
        all_names.extend(row["testcase_name"] for row in physical)
        scene_counts = Counter(row["scene"] for row in factors if row["route"] == route)
        if dict(scene_counts) != {key: int(value) for key, value in config[route]["scenes"].items()}:
            raise AssertionError(f"{route} scene quota mismatch: {scene_counts}")
        priority_counter = Counter(f"P{row['priority']}" for row in factors if row["route"] == route)
        priority_counts = {key: priority_counter.get(key, 0) for key in ("P0", "P1", "P2")}
        if priority_counts != {key: int(value) for key, value in config[route]["priorities"].items()}:
            raise AssertionError(f"{route} priority quota mismatch: {priority_counts}")
    if len(all_names) != len(set(all_names)):
        raise AssertionError("testcase_name must be globally unique")
    if len(factors) != 1200 or len(read_csv(FACTOR_OUTPUT)) != 1200:
        raise AssertionError("factor table must contain exactly 1200 physical cases")

    factor_by_name = {row["case_id"]: row for row in factors}
    for route in ("aclnn_l0", "aclnn_l1", "kernel", "torch"):
        shape_field = "input_shapes" if route == "kernel" else "tensor_view_shapes"
        for row in rows[route]:
            factor = factor_by_name[row["testcase_name"]]
            shapes = ast.literal_eval(row[shape_field])
            nv = int(factor["Nv"])
            expected_v_scale = (nv,) if factor["scene"] == "rope" else (nv, 128)
            if tuple(shapes[11]) != expected_v_scale:
                raise AssertionError(
                    f"{row['testcase_name']} vScale shape {shapes[11]} != {expected_v_scale}"
                )

    for route in ("aclnn_l0", "aclnn_l1"):
        for row in rows[route]:
            if row["api_name"] != API or "V2" in stable(row):
                raise AssertionError(f"{row['testcase_name']} does not use ACLNN V1")
            assert_repr_tuple(row, "tensor_view_shapes", 15)
            assert_repr_tuple(row, "tensor_storage_shapes", 15)
            attrs = ast.literal_eval(row["attributes"])
            if "kQuantMode" not in attrs:
                raise AssertionError(f"{row['testcase_name']} misses V1 kQuantMode")

    for row in rows["kernel"]:
        shapes = assert_repr_tuple(row, "output_shapes", 5)
        dtypes = assert_repr_tuple(row, "output_dtypes", 5)
        formats = assert_repr_tuple(row, "output_formats", 5)
        inplace = assert_repr_tuple(row, "output_inplace_indexes", 5)
        if inplace != (None, None, 5, 6, 7):
            raise AssertionError(f"{row['testcase_name']} output_inplace_indexes is not output-aligned")
        for index, shape in enumerate(shapes):
            if shape is None and (dtypes[index] is not None or formats[index] is not None):
                raise AssertionError(
                    f"{row['testcase_name']} absent output {index} must also use None dtype/format"
                )

    l2 = rows["aclnn_l2"]
    if len({row["testcase_name"] for row in l2}) != 50:
        raise AssertionError("L2 names are not unique")
    if any(not row["constraint_id"] or not row["mutated_field"] for row in l2):
        raise AssertionError("every L2 row must carry one mutation")
    mx_l2_fields = {row["mutated_field"] for row in l2 if row["scene"] == "mrope_mx"}
    for field in ("qQuantMode", "kQuantMode"):
        values = {row["mutated_value"] for row in l2 if row["mutated_field"] == field}
        if not {"<null>", "<empty>", "Unknown"}.issubset(values):
            raise AssertionError(f"L2 misses {field} null/empty/unknown")
    if not {"qQuantMode", "kQuantMode"}.issubset(mx_l2_fields):
        raise AssertionError("L2 misses direct V1 quant-mode mutations")

    anchors = {anchor["id"] for anchor in WHITEBOX["anchors"]}
    for route in ("aclnn_l0", "kernel"):
        present = {row["source_anchor"] for row in factors if row["route"] == route}
        missing = anchors - present
        if missing:
            raise AssertionError(f"{route} misses white-box anchors: {sorted(missing)}")
    mx_positive = [row for row in factors if row["scene"] == "mrope_mx" and row["route"] != "aclnn_l2"]
    if {int(row["Nq"]) for row in mx_positive} != set(range(8, 65, 8)):
        raise AssertionError("MX head matrix must cover Nq=8..64 step8")
    if {row["q_route"] for row in mx_positive} != {"VECTOR", "Q_TAIL_PAIR", "Q_TAIL_GLOBAL"}:
        raise AssertionError("MX Q route matrix incomplete")
    if {row["k_route"] for row in mx_positive} != {"K_GENERIC", "K_BATCH8", "K_BATCH4", "K_BATCH2"}:
        raise AssertionError("MX K route matrix incomplete")

    numeric = {row["numeric_profile"] for row in mx_positive}
    required_numeric = set(FACTORS["mrope_mx"]["numeric_profiles"])
    if not required_numeric.issubset(numeric):
        raise AssertionError(f"MX numeric profiles missing {sorted(required_numeric - numeric)}")
    layouts = {row["cache_layout"] for row in mx_positive if row["route"] in ("aclnn_l0", "aclnn_l1", "torch")}
    required_layouts = set(FACTORS["common"]["cache_layouts"])
    if not required_layouts.issubset(layouts):
        raise AssertionError(f"MX physical storage layouts missing {sorted(required_layouts - layouts)}")
    if not any(int(row["T"]) == 262144 for row in mx_positive):
        raise AssertionError("MX T=262144 stress case missing")
    stress = [row for row in mx_positive if int(row["T"]) == 262144]
    if len(stress) != 1 or stress[0]["route"] != "aclnn_l1" or int(stress[0]["priority"]) != 2:
        raise AssertionError("MX T=262144 must be one independent ACLNN L1 P2 case")

    for route in ("aclnn_l0", "kernel"):
        route_rows = [row for row in mx_positive if row["route"] == route]
        for profile in ("scale_edge_qk", "max_finite_qk"):
            profile_routes = {row["q_route"] for row in route_rows if row["numeric_profile"] == profile}
            if profile_routes != {"VECTOR", "Q_TAIL_PAIR", "Q_TAIL_GLOBAL"}:
                raise AssertionError(f"{route} {profile} misses Q route coverage: {sorted(profile_routes)}")

    parity_matrix = {(row["q_route"], int(row["T"]) & 1) for row in mx_positive}
    expected_parity = {(route, parity) for route in ("VECTOR", "Q_TAIL_PAIR", "Q_TAIL_GLOBAL")
                       for parity in (0, 1)}
    if not expected_parity.issubset(parity_matrix):
        raise AssertionError(f"MX Q route/token parity matrix missing {sorted(expected_parity - parity_matrix)}")

    torch_mx = [row for row in mx_positive if row["route"] == "torch"]
    matrix = {(row["q_route"], row["variant"]) for row in torch_mx}
    required_matrix = {
        ("VECTOR", "functional"), ("VECTOR", "inplace"),
        ("Q_TAIL_PAIR", "functional"), ("Q_TAIL_PAIR", "inplace"),
        ("Q_TAIL_GLOBAL", "functional"), ("Q_TAIL_GLOBAL", "inplace"),
    }
    if not required_matrix.issubset(matrix):
        raise AssertionError(f"Torch Q tail API matrix missing {sorted(required_matrix - matrix)}")

    for row in rows["torch"]:
        inplace_indexes = ast.literal_eval(row["inplace_input_indexes"])
        is_inplace_api = row["api_name"] == TORCH_INPLACE_API
        expected_indexes = (5, 6, 7) if is_inplace_api else ()
        if inplace_indexes != expected_indexes:
            raise AssertionError(
                f"{row['testcase_name']} inplace_input_indexes {inplace_indexes} "
                f"does not match {'in-place' if is_inplace_api else 'functional'} API semantics"
            )

    golden_text = (OP_ROOT / "tests/assets/qkv_rms_norm_rope_cache_with_k_scale_golden.py").read_text(encoding="utf-8")
    for profile in required_numeric - {"normal"}:
        if profile not in golden_text:
            raise AssertionError(f"Golden profile {profile!r} is not implemented")
    source_gate()

    # Rendered products must equal a fresh in-memory render; this detects hand edits.
    field_map = {"aclnn_l0": ACLNN_FIELDS, "aclnn_l1": ACLNN_FIELDS,
                 "aclnn_l2": L2_FIELDS, "kernel": KERNEL_FIELDS, "torch": TORCH_FIELDS}
    for route, path in OUTPUTS.items():
        physical = read_csv(path)
        if physical != [{key: str(row.get(key, "")) for key in field_map[route]} for row in rows[route]]:
            raise AssertionError(f"{path} is not the current generator output")
    physical_factors = read_csv(FACTOR_OUTPUT)
    if physical_factors != [{key: str(row.get(key, "")) for key in FACTOR_FIELDS} for row in factors]:
        raise AssertionError(f"{FACTOR_OUTPUT} is not the current generator output")

    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (*OUTPUTS.values(), FACTOR_OUTPUT)}
    print("ST validation passed: 1200 rows, ACLNN V1 only, current SIMD Q/K routes covered")
    for route in ("aclnn_l0", "aclnn_l1", "aclnn_l2", "kernel", "torch"):
        print(f"  {route}: {len(rows[route])}")
    for name, digest in sorted(hashes.items()):
        print(f"  {digest}  {name}")


def check():
    rows, factors = build_products()
    validate(rows, factors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "check"), nargs="?", default="check")
    args = parser.parse_args()
    if args.command == "generate":
        rows, factors = generate()
        validate(rows, factors)
    else:
        check()


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, RuntimeError) as error:
        print(f"ST validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

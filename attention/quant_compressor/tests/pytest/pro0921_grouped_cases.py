#!/usr/bin/env python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pro0921 formal QuantCompressor batch accuracy entry for the latent grouped path.

Mirrors ``attention/compressor/tests/pytest/pro0921_grouped_cases.py`` for the
quantised operator.  The quant carriage is HiFloat8 (a uint8 byte carrier with
``ACL_HIFLOAT8``); x / wkv / wgate / x_cache hold HiFloat8 codes and the three
descale inputs sit after ``ape`` in the quant ABI order.  The case set is a
plain JSON file so a case can be added without editing Excel.

Commands:
    python3 pro0921_grouped_cases.py --emit /tmp/pro0921_quant_cases
    python3 pro0921_grouped_cases.py --run /tmp/pro0921_quant_cases/cases.json
    python3 pro0921_grouped_cases.py --plan
    python3 pro0921_grouped_cases.py --emit-pt /tmp/pro0921_quant_pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


CONTRACT = "pro0921-grouped-quant-compressor-v1"
OP = "quant_compressor"
VALID_PATHS = ("normal", "grouped")
GROUPED_REQUIRED = ("x_cache_update",)
GROUPED_PAYLOAD = ("x_cache", "x_cache_slot_mapping")

QUANT_MODE = 1
RATIO = 128
HEAD = 128
HIDDEN = 1024
BLOCK_SIZE = 128
BLOCKS = 2

PYTEST_DIR = Path(__file__).resolve().parent
# attention/quant_compressor/tests/pytest -> attention/compressor/tests/pro0921_latent_smoke
SMOKE_DIR = PYTEST_DIR.parents[2] / "compressor" / "tests" / "pro0921_latent_smoke"

QUANT_INPUTS = (
    "x", "wkv", "wgate", "state_cache", "ape",
    "x_descale", "wkv_descale", "wgate_descale",
    "state_block_table", "cu_seqlens", "seqused", "start_pos",
    "x_cache", "x_cache_slot_mapping",
)
OUTPUT_NAMES = ("cmp_kv", "state_cache")
CSV_COLUMNS = [
    "testcase_name", "network_name", "op_name", "input_shapes", "input_dtypes",
    "input_formats", "output_shapes", "output_dtypes", "output_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision", "output_inplace_indexes", "is_enabled", "remark",
]
INPUT_DTYPES = {
    "x": "hifloat8", "wkv": "hifloat8", "wgate": "hifloat8", "state_cache": "float32",
    "ape": "float32", "x_descale": "float32", "wkv_descale": "float32",
    "wgate_descale": "float32", "state_block_table": "int32", "cu_seqlens": "int32",
    "seqused": "int32", "start_pos": "int32", "x_cache": "hifloat8",
    "x_cache_slot_mapping": "int32",
}
TOLERANCE = (0.005, 0.005)
INPLACE = (None, 3)


class CaseContractError(Exception):
    """Raised when a case does not satisfy the explicit pro0921 quant contract."""


def _smoke():
    if str(SMOKE_DIR) not in sys.path:
        sys.path.insert(0, str(SMOKE_DIR))
    import latent_ref  # noqa: E402

    return latent_ref


REQUIRED_CASE_FIELDS = (
    "name", "path", "builder", "cmp_ratio", "coff", "cache_mode",
    "hidden_size", "head_dim", "batch", "seq",
)
BUILDERS = ("quant_boundary", "quant_update_only", "quant_normal")


def validate_case(case):
    if not isinstance(case, dict):
        raise CaseContractError("case entry must be an object")
    missing = [key for key in REQUIRED_CASE_FIELDS if key not in case]
    if missing:
        raise CaseContractError("case %r missing fields: %s" % (case.get("name"), missing))
    if case["path"] not in VALID_PATHS:
        raise CaseContractError(
            "case %r has path=%r, expected one of %s"
            % (case["name"], case["path"], list(VALID_PATHS)))
    if case["builder"] not in BUILDERS:
        raise CaseContractError(
            "case %r uses unknown builder %r, expected %s"
            % (case["name"], case["builder"], list(BUILDERS)))
    if case["cmp_ratio"] != RATIO or case["head_dim"] != HEAD or case["hidden_size"] != HIDDEN:
        raise CaseContractError(
            "case %r must use the frozen small shape R=%d head=%d hidden=%d"
            % (case["name"], RATIO, HEAD, HIDDEN))
    if case["path"] == "grouped":
        for key in GROUPED_REQUIRED:
            if key not in case:
                raise CaseContractError(
                    "grouped case %r is missing the grouped field %r; a normal case must "
                    "declare path=\"normal\" explicitly" % (case["name"], key))
        if not isinstance(case["x_cache_update"], bool):
            raise CaseContractError(
                "grouped case %r x_cache_update must be a boolean" % case["name"])
    return case


def validate_case_set(document):
    if not isinstance(document, dict):
        raise CaseContractError("case set must be a JSON object")
    if document.get("contract") != CONTRACT:
        raise CaseContractError(
            "case set contract must be %r, got %r" % (CONTRACT, document.get("contract")))
    if document.get("op") != OP:
        raise CaseContractError("case set op must be %r, got %r" % (OP, document.get("op")))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CaseContractError("case set must carry a non-empty 'cases' list")
    seen = set()
    for case in cases:
        validate_case(case)
        if case["name"] in seen:
            raise CaseContractError("duplicate case name %r" % case["name"])
        seen.add(case["name"])
    return document


def load_case_set(path):
    return validate_case_set(json.loads(Path(path).read_text(encoding="utf-8")))


def default_case_set():
    return {
        "contract": CONTRACT,
        "op": OP,
        "seed": 20260928,
        "cases": [
            {
                "name": "quant_normal_r128", "path": "normal", "builder": "quant_normal",
                "cmp_ratio": RATIO, "coff": 1, "cache_mode": 1,
                "hidden_size": HIDDEN, "head_dim": HEAD, "batch": 1, "seq": RATIO,
            },
            {
                "name": "quant_grouped_boundary", "path": "grouped",
                "builder": "quant_boundary", "cmp_ratio": RATIO, "coff": 1, "cache_mode": 1,
                "hidden_size": HIDDEN, "head_dim": HEAD, "batch": 1, "seq": 1,
                "x_cache_update": True,
            },
            {
                "name": "quant_grouped_update_only", "path": "grouped",
                "builder": "quant_update_only", "cmp_ratio": RATIO, "coff": 1, "cache_mode": 1,
                "hidden_size": HIDDEN, "head_dim": HEAD, "batch": 1, "seq": 1,
                "x_cache_update": True,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------
def _weights(rng, latent_ref, rows, hidden):
    return latent_ref.encode_hif8(rng.standard_normal((rows, hidden)).astype(np.float32) * 0.02)


def _quant_static(rng, latent_ref):
    wkv = _weights(rng, latent_ref, HEAD, HIDDEN)
    wgate = _weights(rng, latent_ref, HEAD, HIDDEN)
    x_descale = np.array([0.5], dtype=np.float32)
    wkv_descale = (0.6 + 0.2 * rng.random(HEAD)).astype(np.float32)
    wgate_descale = (0.6 + 0.2 * rng.random(HEAD)).astype(np.float32)
    ape = (rng.standard_normal((RATIO, HEAD)) * 0.01).astype(np.float32)
    state = np.zeros((BLOCKS, BLOCK_SIZE, 2 * HEAD), dtype=np.float32)
    block_table = np.array([[1]], dtype=np.int32)
    return wkv, wgate, x_descale, wkv_descale, wgate_descale, ape, state, block_table


def _build_quant_grouped(rng, latent_ref, name, update_only):
    wkv, wgate, xds, wkds, wgds, ape, state, block_table = _quant_static(rng, latent_ref)
    window_values = rng.standard_normal((RATIO, HIDDEN)).astype(np.float32) * 0.02
    window_codes = latent_ref.encode_hif8(window_values)
    x_codes = latent_ref.encode_hif8(window_values[RATIO - 1: RATIO])
    sentinel = latent_ref.encode_hif8(np.array([[0.1]], dtype=np.float32))
    x_cache = np.zeros((BLOCKS, RATIO, HIDDEN), dtype=np.uint8)
    x_cache[1, : RATIO - 1] = window_codes[: RATIO - 1]
    x_cache[1, RATIO - 1] = sentinel[0, 0]
    start_pos = 64 if update_only else RATIO - 1
    slot = 1 * RATIO + start_pos
    golden = latent_ref.quant_grouped_case_golden(
        x_codes, wkv, wgate, xds, wkds, wgds, state, ape, block_table, x_cache, slot,
        RATIO, 1, 1, start_pos=np.array([start_pos]), seqused=np.array([1]))
    if update_only:
        if golden["active"]:
            raise CaseContractError("update_only fixture completed a group unexpectedly")
    elif not (golden["active"] and golden["window_start"] == 0):
        raise CaseContractError("boundary fixture did not complete the first group")
    return {
        "name": name,
        "payload": {
            "x": x_codes.reshape(1, 1, HIDDEN),
            "wkv": wkv, "wgate": wgate, "state_cache": state, "ape": ape,
            "x_descale": xds, "wkv_descale": wkds, "wgate_descale": wgds,
            "state_block_table": block_table, "cu_seqlens": None,
            "seqused": np.array([1], dtype=np.int32),
            "start_pos": np.array([start_pos], dtype=np.int32),
            "x_cache": x_cache,
            "x_cache_slot_mapping": np.array([slot], dtype=np.int32),
        },
        "outputs": (golden["cmp_kv"], golden["state_cache"]),
        "attrs": {
            "quant_mode": QUANT_MODE, "cmp_ratio": RATIO, "coff": 1, "cache_mode": 1,
            "state_cache_stride_dim0": BLOCK_SIZE * (2 * HEAD),
            "x_cache_update": True,
        },
        "active": bool(golden["active"]),
    }


def _build_quant_normal(rng, latent_ref, name):
    wkv, wgate, xds, wkds, wgds, ape, state, block_table = _quant_static(rng, latent_ref)
    window_values = rng.standard_normal((RATIO, HIDDEN)).astype(np.float32) * 0.02
    x_codes = latent_ref.encode_hif8(window_values).reshape(1, RATIO, HIDDEN)
    cmp_kv, _, _, out_state = latent_ref.quant_compress_golden(
        x_codes, wkv, wgate, xds, wkds, wgds, state, ape, block_table, RATIO, 1, 1,
        start_pos=[0], seqused=[RATIO])
    return {
        "name": name,
        "payload": {
            "x": x_codes, "wkv": wkv, "wgate": wgate, "state_cache": state, "ape": ape,
            "x_descale": xds, "wkv_descale": wkds, "wgate_descale": wgds,
            "state_block_table": block_table, "cu_seqlens": None,
            "seqused": np.array([RATIO], dtype=np.int32),
            "start_pos": np.array([0], dtype=np.int32),
            "x_cache": None, "x_cache_slot_mapping": None,
        },
        "outputs": (cmp_kv, out_state),
        "attrs": {
            "quant_mode": QUANT_MODE, "cmp_ratio": RATIO, "coff": 1, "cache_mode": 1,
            "state_cache_stride_dim0": BLOCK_SIZE * (2 * HEAD),
            "x_cache_update": True,
        },
        "active": True,
    }


def build_case(case):
    latent_ref = _smoke()
    rng = np.random.default_rng(case.get("seed", 20260928))
    if case["builder"] == "quant_boundary":
        built = _build_quant_grouped(rng, latent_ref, case["name"], update_only=False)
    elif case["builder"] == "quant_update_only":
        built = _build_quant_grouped(rng, latent_ref, case["name"], update_only=True)
    elif case["builder"] == "quant_normal":
        built = _build_quant_normal(rng, latent_ref, case["name"])
    else:  # pragma: no cover
        raise CaseContractError("unknown builder %r" % case["builder"])
    payload = built["payload"]
    attrs = built["attrs"]
    if (attrs["cmp_ratio"] != case["cmp_ratio"]) or (attrs["coff"] != case["coff"]):
        raise CaseContractError(
            "builder %r produced cmp_ratio=%d coff=%d but the case spec asked for "
            "cmp_ratio=%d coff=%d" % (case["builder"], attrs["cmp_ratio"], attrs["coff"],
                                      case["cmp_ratio"], case["coff"]))
    if case["path"] == "grouped":
        for key in GROUPED_PAYLOAD:
            if payload.get(key) is None:
                raise CaseContractError("builder %r did not provide grouped payload %r"
                                        % (case["builder"], key))
    else:
        for key in GROUPED_PAYLOAD:
            if payload.get(key) is not None:
                raise CaseContractError("normal case %r must not carry grouped payload %r"
                                        % (case["name"], key))
    return built


def _shapes_dtypes(payload):
    shapes, dtypes, formats = [], [], []
    for key in QUANT_INPUTS:
        value = payload[key]
        if value is None:
            shapes.append(None)
            dtypes.append(None)
            formats.append(None)
        else:
            shapes.append(tuple(int(v) for v in value.shape))
            dtypes.append(INPUT_DTYPES[key])
            formats.append("ND")
    return shapes, dtypes, formats


def emit_case(out_root, built, case):
    out_root = Path(out_root)
    dest = out_root / built["name"]
    dest.mkdir(parents=True, exist_ok=True)
    payload = built["payload"]
    for key, value in payload.items():
        if value is not None:
            np.save(dest / (key + ".npy"), value)
    files, dtypes = [], []
    for index, original in enumerate(built["outputs"]):
        if original is None:
            files.append(None)
            dtypes.append(None)
            continue
        name = "golden_%d.npy" % index
        np.save(dest / name, np.asarray(original, dtype=np.float32))
        files.append(name)
        dtypes.append("float32")
    (dest / "golden.json").write_text(json.dumps(
        {"files": files, "dtypes": dtypes, "output_names": list(OUTPUT_NAMES)}, indent=2),
        encoding="utf-8")
    expected = {
        "checked_outputs": ["cmp_kv", "state_cache"] if built["outputs"][0] is not None
        else ["state_cache"],
        "suppressed_outputs": [] if built["outputs"][0] is not None else ["cmp_kv"],
        "path": case["path"],
        "quant_carrier": "x/wkv/wgate/x_cache are hifloat8 uint8 codes (ACL_HIFLOAT8)",
        "note": ("quant_mode=1 (A8W8 A-per-tensor / W-per-channel); descale inputs sit after "
                 "ape in the quant ABI order"),
    }
    (dest / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (dest / "readback_spec.json").write_text(json.dumps(
        [{"index": QUANT_INPUTS.index("x_cache"), "name": "x_cache", "dtype": "hifloat8",
          "shape": [int(v) for v in payload["x_cache"].shape]}] if payload["x_cache"] is not None
        else [], indent=2), encoding="utf-8")
    shapes, dtype_names, formats = _shapes_dtypes(payload)
    attributes = dict(built["attrs"])
    for key in ("state_block_table", "seqused", "start_pos", "x_cache_slot_mapping"):
        if payload.get(key) is not None:
            attributes[key] = payload[key].tolist()
    state = payload["state_cache"]
    sr = (payload["x"].shape[1] + RATIO - 1) // RATIO
    row = {
        "testcase_name": built["name"],
        "network_name": "deepseek_v4",
        "op_name": "quant_compressor",
        "input_shapes": repr(tuple(shapes)),
        "input_dtypes": repr(tuple(dtype_names)),
        "input_formats": repr(tuple(formats)),
        "output_shapes": repr(((1, sr, HEAD), tuple(int(v) for v in state.shape))),
        "output_dtypes": repr(("bfloat16", "float32")),
        "output_formats": repr(("ND", "ND")),
        "attributes": repr(attributes),
        "input_data_ranges": repr(tuple((None if s is None else (-0.1, 0.1)) for s in shapes)),
        "precision_tolerances": repr((TOLERANCE, TOLERANCE)),
        "absolute_precision": 0.001,
        "output_inplace_indexes": repr(INPLACE),
        "is_enabled": True,
        "remark": ("Quant %s R%d coff=%d; hif8 codes + descale; independent CPU golden"
                   % (case["path"], RATIO, case["coff"])),
    }
    with (dest / "case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return dest


def materialize(case_set, out_root):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"contract": CONTRACT, "op": OP, "root": str(out_root), "cases": []}
    for case in case_set["cases"]:
        built = build_case(case)
        dest = emit_case(out_root, built, case)
        payload = built["payload"]
        manifest["cases"].append({
            "name": built["name"],
            "path": case["path"],
            "dir": str(dest),
            "builder": case["builder"],
            "seed": case.get("seed", case_set.get("seed", 20260928)),
            "cmp_ratio": RATIO,
            "coff": case["coff"],
            "cache_mode": case["cache_mode"],
            "hidden_size": HIDDEN,
            "head_dim": HEAD,
            "batch": case["batch"],
            "seq": case["seq"],
            "quant_mode": QUANT_MODE,
            "x_cache_update": bool(case.get("x_cache_update", True)),
            "start_pos": _tolist(payload["start_pos"]),
            "seqused": _tolist(payload["seqused"]),
            "state_block_table": _tolist(payload["state_block_table"]),
            "x_cache_slot_mapping": _tolist(payload["x_cache_slot_mapping"]),
            "readback": ["x_cache"] if payload["x_cache"] is not None else [],
            "expected": json.loads((dest / "expected.json").read_text(encoding="utf-8")),
        })
    (out_root / "cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _tolist(value):
    return None if value is None else [int(v) for v in np.asarray(value).reshape(-1)]


def load_manifest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("contract") != CONTRACT:
        raise CaseContractError(
            "manifest contract must be %r, got %r" % (CONTRACT, document.get("contract")))
    for entry in document.get("cases", []):
        if entry.get("path") not in VALID_PATHS:
            raise CaseContractError(
                "manifest case %r has path=%r" % (entry.get("name"), entry.get("path")))
        if entry["path"] == "grouped" and "x_cache_update" not in entry:
            raise CaseContractError(
                "manifest grouped case %r is missing x_cache_update" % entry.get("name"))
    return document


# ---------------------------------------------------------------------------
# Independent numeric recompute (CPU)
# ---------------------------------------------------------------------------
def _load(case_dir, name):
    path = Path(case_dir) / (name + ".npy")
    return np.load(path) if path.exists() else None


def _softmax_cols(z):
    z = np.asarray(z, dtype=np.float32)
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z).astype(np.float32)
    return (e / np.sum(e, axis=0, keepdims=True)).astype(np.float32)


def independent_contract_check(case_dir, entry):
    checks = []
    latent_ref = _smoke()
    golden_json = json.loads((Path(case_dir) / "golden.json").read_text(encoding="utf-8"))
    goldens = {n: (None if f is None else np.load(Path(case_dir) / f))
               for n, f in zip(golden_json["output_names"], golden_json["files"])}
    x_cache = _load(case_dir, "x_cache")
    slots = _load(case_dir, "x_cache_slot_mapping")
    if entry["path"] == "grouped":
        rows = x_cache.shape[1]
        slot = int(slots.reshape(-1)[0])
        refreshed = x_cache.copy()
        expect = x_cache.copy()
        update = bool(entry["x_cache_update"])
        if update and slot >= 0:
            token = _load(case_dir, "x").reshape(-1, x_cache.shape[-1])[
                int(np.asarray(entry["seqused"]).reshape(-1)[0]) - 1]
            expect[slot // rows, slot % rows] = token
            refreshed[slot // rows, slot % rows] = token
        checks.append((entry["name"] + ":x_cache_ring_refresh",
                       bool(np.array_equal(refreshed, expect)), 0.0))
        start_pos = int(np.asarray(entry["start_pos"]).reshape(-1)[0])
        seqused = int(np.asarray(entry["seqused"]).reshape(-1)[0])
        group_offset = start_pos % RATIO
        active = seqused >= RATIO - group_offset
        if not active:
            checks.append((entry["name"] + ":update_only_state_unchanged",
                           bool(np.array_equal(goldens["state_cache"],
                                               _load(case_dir, "state_cache"))), 0.0))
            return checks
        window_start = ((start_pos + seqused) // RATIO) * RATIO - RATIO
        row_idx = [(window_start + k) % rows for k in range(RATIO)]
        window = refreshed[slot // rows, row_idx].astype(np.uint8)
    else:
        window = _load(case_dir, "x").reshape(RATIO, HIDDEN).astype(np.uint8)

    x_fp32 = latent_ref.decode_hif8(window)
    wkv_fp32 = latent_ref.decode_hif8(_load(case_dir, "wkv"))
    wgate_fp32 = latent_ref.decode_hif8(_load(case_dir, "wgate"))
    xds = _load(case_dir, "x_descale").astype(np.float32).reshape(-1)
    wkds = _load(case_dir, "wkv_descale").astype(np.float32).reshape(-1)
    wgds = _load(case_dir, "wgate_descale").astype(np.float32).reshape(-1)
    ape = _load(case_dir, "ape").astype(np.float32)
    kv = np.matmul(x_fp32, wkv_fp32.T, dtype=np.float32) * (xds[0] * wkds)
    score = np.matmul(x_fp32, wgate_fp32.T, dtype=np.float32) * (xds[0] * wgds)
    score = score + ape[: RATIO]
    weights = _softmax_cols(score)
    independent = (weights * kv).sum(axis=0, keepdims=True).reshape(1, 1, HEAD)
    if goldens["cmp_kv"] is not None:
        delta = float(np.abs(independent - goldens["cmp_kv"]).max())
        checks.append((entry["name"] + ":quant_cmp_kv_independent_recompute",
                       delta < 2e-2, delta))
    checks.append((entry["name"] + ":state_window_rows_written",
                   not np.array_equal(goldens["state_cache"], _load(case_dir, "state_cache")),
                   0.0))
    return checks


# ---------------------------------------------------------------------------
# NPU accuracy execution
# ---------------------------------------------------------------------------
def _load_npu_runtime():
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError(
            "no NPU device is available; run with --plan to print the plan instead. "
            "This runner never fabricates results without a device.")
    try:
        from cann_ops_transformer.ops import quant_compressor
    except Exception as exc:  # pragma: no cover - depends on the on-board install
        raise RuntimeError(
            "the formal provider cann_ops_transformer.ops.quant_compressor is not "
            "importable; install the current dual-repo operators and wheel first, using "
            "the torch -> torch_npu -> formal provider import order (%r)" % exc)

    return torch, quant_compressor


def _run_one_accuracy(torch, quant_compressor, entry, device_id):
    case_dir = Path(entry["dir"])
    device = "npu:%d" % device_id
    payload = {name: _load(case_dir, name) for name in QUANT_INPUTS}

    def dev(name):
        value = payload[name]
        if value is None:
            return None
        if name in ("start_pos", "seqused", "state_block_table", "cu_seqlens",
                    "x_cache_slot_mapping"):
            return torch.from_numpy(np.ascontiguousarray(value.astype(np.int32))).to(
                dtype=torch.int32, device=device)
        if name in ("state_cache", "ape", "x_descale", "wkv_descale", "wgate_descale"):
            return torch.from_numpy(np.ascontiguousarray(value.astype(np.float32))).to(
                dtype=torch.float32, device=device)
        return torch.from_numpy(np.ascontiguousarray(value.astype(np.uint8))).to(
            dtype=torch.uint8, device=device)

    state_cache = dev("state_cache")
    x_cache = dev("x_cache")
    kwargs = dict(
        x_descale=dev("x_descale"), wkv_descale=dev("wkv_descale"),
        wgate_descale=dev("wgate_descale"), state_block_table=dev("state_block_table"),
        cu_seqlens=dev("cu_seqlens"), seqused=dev("seqused"), start_pos=dev("start_pos"),
        quant_mode=entry["quant_mode"], cmp_ratio=entry["cmp_ratio"],
        coff=entry["coff"], cache_mode=entry["cache_mode"],
    )
    if entry["path"] == "grouped":
        kwargs["x_cache"] = x_cache
        kwargs["x_cache_slot_mapping"] = dev("x_cache_slot_mapping")
        kwargs["x_cache_update"] = bool(entry["x_cache_update"])
    cmp_kv = quant_compressor(dev("x"), dev("wkv"), dev("wgate"), state_cache, dev("ape"),
                              **kwargs)
    torch.npu.synchronize()
    cmp_kv_cpu = cmp_kv.detach().cpu().to(torch.float32).numpy()
    state_cpu = state_cache.detach().cpu().to(torch.float32).numpy()
    x_cache_cpu = None if x_cache is None else x_cache.detach().cpu().numpy()
    golden_json = json.loads((case_dir / "golden.json").read_text(encoding="utf-8"))
    goldens = {n: (None if f is None else np.load(case_dir / f))
               for n, f in zip(golden_json["output_names"], golden_json["files"])}
    checks = []
    if goldens["cmp_kv"] is not None:
        checks.append((entry["name"] + ":cmp_kv", _compare(goldens["cmp_kv"], cmp_kv_cpu)))
    if goldens["state_cache"] is not None:
        checks.append((entry["name"] + ":state_cache",
                       _compare_state(goldens["state_cache"], state_cpu,
                                      payload["state_cache"])))
    if entry["path"] == "grouped":
        rows = payload["x_cache"].shape[1]
        slot = int(payload["x_cache_slot_mapping"].reshape(-1)[0])
        expect = payload["x_cache"].copy()
        if entry["x_cache_update"] and slot >= 0:
            token = payload["x"].reshape(-1, payload["x"].shape[-1])[
                int(payload["seqused"].reshape(-1)[0]) - 1]
            expect[slot // rows, slot % rows] = token
        checks.append((entry["name"] + ":x_cache_side_effect",
                       {"pass": bool(np.array_equal(x_cache_cpu, expect)),
                        "modified": bool(not np.array_equal(x_cache_cpu, payload["x_cache"])),
                        "expected_modified": bool(not np.array_equal(expect, payload["x_cache"]))}))
    return {
        "name": entry["name"], "path": entry["path"], "quant_mode": entry["quant_mode"],
        "x_cache_update": entry["x_cache_update"],
        "checks": [{"check": name, "ok": bool(value["pass"]), "metric": value}
                   for name, value in checks],
    }


def _compare_state(expect, actual, pristine, rtol=0.0078125, atol=1e-4):
    """State-cache compare that never dilutes an untouched element into a mean.

    ``state_cache`` is an in-place input/output: the operator legitimately
    rewrites only the rows its group completes and must leave every other row
    bit-for-bit untouched.  A whole-tensor percentage (``_compare``) can hide a
    single corrupted untouched element behind >=99.5% of unchanged values, so
    this check splits the tensor with the ``expect == pristine`` mask:

    * untouched positions (``expect == pristine``) must match the golden in the
      exact float32 bit pattern, and
    * updated positions must satisfy ``np.isclose`` with the given rtol/atol.

    NaN/Inf anywhere (golden, readback, or pristine) fails outright, and a
    shape mismatch is a hard error.  Returns ``pass`` plus reason/counts.
    """
    expect = np.ascontiguousarray(np.asarray(expect, dtype=np.float32))
    actual = np.ascontiguousarray(np.asarray(actual, dtype=np.float32))
    pristine = np.ascontiguousarray(np.asarray(pristine, dtype=np.float32))
    if expect.shape != actual.shape or expect.shape != pristine.shape:
        return {"pass": False, "reason": "shape mismatch",
                "expect_shape": list(expect.shape),
                "actual_shape": list(actual.shape),
                "pristine_shape": list(pristine.shape)}
    if expect.size == 0:
        return {"pass": True, "rtol": rtol, "atol": atol}
    nonfinite = int((~np.isfinite(expect)).sum() + (~np.isfinite(actual)).sum()
                    + (~np.isfinite(pristine)).sum())
    if nonfinite:
        return {"pass": False, "reason": "non-finite value present",
                "expect_nonfinite": int((~np.isfinite(expect)).sum()),
                "actual_nonfinite": int((~np.isfinite(actual)).sum()),
                "pristine_nonfinite": int((~np.isfinite(pristine)).sum()),
                "rtol": rtol, "atol": atol}
    untouched = expect == pristine
    updated = ~untouched
    untouched_mismatch = int((actual.view(np.uint32)[untouched]
                              != expect.view(np.uint32)[untouched]).sum())
    updated_mismatch = int((~np.isclose(actual[updated], expect[updated],
                                        rtol=rtol, atol=atol)).sum())
    return {
        "pass": bool(untouched_mismatch == 0 and updated_mismatch == 0),
        "untouched_total": int(untouched.sum()),
        "updated_total": int(updated.sum()),
        "untouched_mismatch": untouched_mismatch,
        "updated_mismatch": updated_mismatch,
        "max_abs_untouched": float(np.abs(actual[untouched] - expect[untouched]).max())
        if untouched.any() else 0.0,
        "max_abs_updated": float(np.abs(actual[updated] - expect[updated]).max())
        if updated.any() else 0.0,
        "rtol": rtol, "atol": atol,
    }


def _compare(expect, actual, rtol=0.0078125, atol=1e-4):
    """Numeric compare that never masks NaN/Inf (``equal_nan=True`` is not used)."""
    expect = np.asarray(expect, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if expect.size != actual.size:
        return {"pass": False, "reason": "shape mismatch",
                "expect_size": int(expect.size), "actual_size": int(actual.size)}
    if expect.size == 0:
        return {"pass": True, "fulfill_percent": 100.0}
    nonfinite = int((~np.isfinite(expect)).sum() + (~np.isfinite(actual)).sum())
    if nonfinite:
        return {"pass": False, "reason": "non-finite value present",
                "expect_nonfinite": int((~np.isfinite(expect)).sum()),
                "actual_nonfinite": int((~np.isfinite(actual)).sum()),
                "rtol": rtol, "atol": atol}
    isclose = np.isclose(actual, expect, rtol=rtol, atol=atol)
    fulfill = float(isclose.mean() * 100.0)
    return {"pass": bool(fulfill >= 99.5), "fulfill_percent": fulfill,
            "max_abs": float(np.abs(actual - expect).max()), "rtol": rtol, "atol": atol}


def run_accuracy(manifest, device_id=0, out_json=None):
    torch, quant_compressor = _load_npu_runtime()
    torch.npu.set_device(int(device_id))
    results = [_run_one_accuracy(torch, quant_compressor, entry, device_id)
               for entry in manifest["cases"]]
    report = {"contract": CONTRACT, "device": int(device_id),
              "provider": {"module": "cann_ops_transformer.ops.quant_compressor",
                           "torch": torch.__version__},
              "cases": results}
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["failed"] = [c["name"] for c in results if not all(chk["ok"] for chk in c["checks"])]
    return report


# ---------------------------------------------------------------------------
# Batch .pt emission
# ---------------------------------------------------------------------------
def emit_pt(entry, pt_path):
    import torch

    case_dir = Path(entry["dir"])
    built = build_case({
        "name": entry["name"], "path": entry["path"], "builder": entry["builder"],
        "cmp_ratio": entry["cmp_ratio"], "coff": entry["coff"],
        "cache_mode": entry["cache_mode"], "hidden_size": entry["hidden_size"],
        "head_dim": entry["head_dim"], "batch": entry["batch"], "seq": entry["seq"],
        "x_cache_update": entry.get("x_cache_update", True),
        "seed": entry.get("seed", 20260928),
    })
    payload = built["payload"]
    golden_json = json.loads((case_dir / "golden.json").read_text(encoding="utf-8"))
    goldens = {n: (None if f is None else np.load(case_dir / f))
               for n, f in zip(golden_json["output_names"], golden_json["files"])}

    def tensor(array, dtype=None):
        if array is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(array))
        return t.to(dtype) if dtype is not None else t

    cmp_kv = goldens["cmp_kv"]
    if cmp_kv is None:
        cmp_kv = np.zeros((entry["batch"], (entry["seq"] + entry["cmp_ratio"] - 1)
                           // entry["cmp_ratio"], entry["head_dim"]), dtype=np.float32)
        mask = np.zeros_like(cmp_kv, dtype=bool)
    else:
        mask = np.ones_like(cmp_kv, dtype=bool)
    state_out = goldens["state_cache"]
    half = state_out.shape[2] // 2
    updated = state_out != payload["state_cache"]
    output = {
        "params": _params_tuple(entry),
        "path": entry["path"],
        "x_cache_update": bool(entry.get("x_cache_update", True)),
        "quant_mode": entry["quant_mode"],
        "cpu_result": tensor(cmp_kv, torch.bfloat16),
        "kv_mask_result": torch.from_numpy(mask),
        "update_kv": torch.from_numpy(updated[:, :, :half]),
        "update_score": torch.from_numpy(updated[:, :, half:]),
        "cpu_kv_state": tensor(state_out[:, :, :half]),
        "cpu_score_state": tensor(state_out[:, :, half:]),
        "x": tensor(payload["x"]),
        "wkv": tensor(payload["wkv"]),
        "wgate": tensor(payload["wgate"]),
        "x_descale": tensor(payload["x_descale"]),
        "wkv_descale": tensor(payload["wkv_descale"]),
        "wgate_descale": tensor(payload["wgate_descale"]),
        "kv_state": tensor(payload["state_cache"][:, :, :half]),
        "score_state": tensor(payload["state_cache"][:, :, half:]),
        "ape": tensor(payload["ape"]),
        "block_table": tensor(payload["state_block_table"], torch.int32),
        "cu_seqlens": tensor(payload["cu_seqlens"], torch.int32),
        "seqused": tensor(payload["seqused"], torch.int32),
        "start_pos": entry["start_pos"],
        "cmp_ratio": entry["cmp_ratio"],
        "coff": entry["coff"],
        "cache_mode": entry["cache_mode"],
    }
    if entry["path"] == "grouped":
        output["x_cache"] = tensor(payload["x_cache"])
        output["x_cache_slot_mapping"] = tensor(payload["x_cache_slot_mapping"], torch.int32)
        output["expected_x_cache"] = tensor(_expected_grouped_ring(entry, payload))
    Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, pt_path)
    return output


def _expected_grouped_ring(entry, payload):
    """Full ring after one grouped call, from the independent numpy golden.

    ``x_cache_update=False`` (LI read-only shared cache) leaves the ring
    untouched, so the expected ring is the input ring itself.  The ring holds
    HiFloat8 codes, so it is compared bit-for-bit (uint8) by the loader.
    """
    latent_ref = _smoke()
    out = latent_ref.quant_grouped_case_golden(
        np.asarray(payload["x"]),
        np.asarray(payload["wkv"]),
        np.asarray(payload["wgate"]),
        np.asarray(payload["x_descale"], dtype=np.float32),
        np.asarray(payload["wkv_descale"], dtype=np.float32),
        np.asarray(payload["wgate_descale"], dtype=np.float32),
        np.asarray(payload["state_cache"], dtype=np.float32),
        np.asarray(payload["ape"], dtype=np.float32),
        np.asarray(payload["state_block_table"]),
        np.asarray(payload["x_cache"]),
        np.asarray(payload["x_cache_slot_mapping"]).reshape(-1)[0],
        entry["cmp_ratio"], entry["coff"], entry["cache_mode"],
        start_pos=np.asarray(payload["start_pos"]),
        seqused=np.asarray(payload["seqused"]),
        x_cache_update=bool(entry.get("x_cache_update", True)),
    )
    return np.asarray(out["x_cache"])


def _params_tuple(entry):
    return [
        entry["batch"], entry["hidden_size"], entry["seq"], entry["head_dim"],
        BLOCK_SIZE, entry["cmp_ratio"], entry["coff"], entry["start_pos"][0],
        entry["cache_mode"], "BSH", "BF16", None,
        str(entry["seqused"]), str(entry["start_pos"]),
        "[-10,10]", "[-10,10]", "[-10,10]", "[-10,10]", "[-10,10]", "[-10,10]",
        "[0.001,1.0]", "[0.001,1.0]", "[0.001,1.0]", entry["quant_mode"],
    ]


def pt_grouped_args(test_data):
    """Explicit op kwargs selector for ``batch/quant_compressor_pt_loadprocess.py``.

    ``path`` is mandatory: a ``.pt`` emitted by :func:`emit_pt` always declares
    ``path`` (``"normal"`` or ``"grouped"``).  A file without ``path`` is a
    pre-grouped ABI and is rejected here; it must be regenerated with the
    current writer.  A grouped case must carry the x_cache fields or this raises
    instead of silently running the normal path.
    """
    path = test_data.get("path")
    if path is None:
        raise CaseContractError(
            "pt case has no 'path' field; regenerate it with the current writer "
            "(normal/grouped must be explicit, no legacy default)")
    if path not in VALID_PATHS:
        raise CaseContractError(
            "pt case declares path=%r, expected one of %s" % (path, list(VALID_PATHS)))
    if path == "grouped":
        for key in ("x_cache", "x_cache_slot_mapping", "x_cache_update"):
            if key not in test_data:
                raise CaseContractError(
                    "grouped pt is missing the grouped field %r; refusing to run it as a "
                    "normal case" % key)
        return {
            "path": "grouped",
            "x_cache": test_data["x_cache"],
            "x_cache_slot_mapping": test_data["x_cache_slot_mapping"],
            "x_cache_update": bool(test_data["x_cache_update"]),
        }
    return {"path": "normal"}


def emit_pt_from_spec(spec, pt_path):
    """Validate a single case spec, materialize it, and emit one ``.pt``."""
    validate_case(spec)
    import tempfile

    manifest = materialize({"contract": CONTRACT, "op": OP, "cases": [spec]},
                           Path(tempfile.mkdtemp(prefix="pro0921_quant_pt_")))
    return emit_pt(manifest["cases"][0], pt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _plan(manifest):
    print("contract=%s op=%s cases=%d" % (CONTRACT, OP, len(manifest["cases"])))
    for entry in manifest["cases"]:
        print("  %-30s path=%-7s R=%d coff=%d update=%-5s quant_mode=%d"
              % (entry["name"], entry["path"], entry["cmp_ratio"], entry["coff"],
                 entry["x_cache_update"], entry.get("quant_mode", QUANT_MODE)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pro0921 grouped QuantCompressor accuracy entry")
    parser.add_argument("--cases", type=Path, help="input case-set JSON (default: frozen set)")
    parser.add_argument("--emit", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--emit-pt", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)

    try:
        case_set = validate_case_set(load_case_set(args.cases)) if args.cases else default_case_set()
    except CaseContractError as exc:
        print("PRO0921_QUANT_CASE_CONTRACT=FAIL %s" % exc)
        return 1
    if args.plan:
        manifest = materialize(case_set, args.emit) if args.emit else _plan_only_manifest(case_set)
        _plan(manifest)
        return 0
    if args.emit:
        manifest = materialize(case_set, args.emit)
        print("materialized %d cases -> %s" % (len(manifest["cases"]), args.emit / "cases.json"))
        if not (args.run or args.emit_pt):
            return 0
    else:
        manifest = None
    if args.emit_pt:
        if manifest is None:
            import tempfile

            manifest = materialize(case_set,
                                   Path(tempfile.mkdtemp(prefix="pro0921_quant_pt_")))
        for entry in manifest["cases"]:
            dest = args.emit_pt / (entry["name"] + ".pt")
            emit_pt(entry, dest)
            print("pt -> %s" % dest)
    if args.run:
        try:
            target = load_manifest(args.run)
        except CaseContractError as exc:
            print("PRO0921_QUANT_CASE_CONTRACT=FAIL %s" % exc)
            return 1
    else:
        target = manifest if args.emit else None
    if target is not None:
        try:
            report = run_accuracy(target, device_id=args.device, out_json=args.out_json)
        except RuntimeError as exc:
            print("PRO0921_QUANT_ACCURACY=FAIL %s" % exc)
            return 2
        for case in report["cases"]:
            print("  %-30s %s" % (case["name"],
                                  "PASS" if all(c["ok"] for c in case["checks"]) else "FAIL"))
        if report["failed"]:
            print("PRO0921_QUANT_ACCURACY=FAIL cases=%s" % ",".join(report["failed"]))
            return 1
        print("PRO0921_QUANT_ACCURACY=PASS cases=%d" % len(report["cases"]))
    return 0


def _plan_only_manifest(case_set):
    cases = []
    for case in case_set["cases"]:
        cases.append({
            "name": case["name"], "path": case["path"], "cmp_ratio": case["cmp_ratio"],
            "coff": case["coff"], "x_cache_update": bool(case.get("x_cache_update", True)),
            "quant_mode": QUANT_MODE,
        })
    return {"contract": CONTRACT, "op": OP, "cases": cases}


if __name__ == "__main__":
    raise SystemExit(main())

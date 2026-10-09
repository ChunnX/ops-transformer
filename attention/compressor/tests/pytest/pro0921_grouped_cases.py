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
"""Pro0921 formal Compressor batch accuracy entry for the latent grouped path.

This module is the single explicit entry for the Pro0921 latent *grouped*
Compressor batch accuracy path.  The legacy raw CLI in ``compressor_golden.py``
still drives the normal path; the two paths are selected explicitly:

* ``path == "normal"``  -> grouped switch off, the operator is driven without
  ``x_cache`` (the pre-existing behaviour).
* ``path == "grouped"`` -> the case must carry the grouped contract
  (``x_cache_update``) plus the ``x_cache`` / ``x_cache_slot_mapping`` payload
  the fixture builder produces.  A missing grouped field is a hard error; it is
  never silently downgraded to a normal case.

The case set is a plain JSON file so a case can be added without editing Excel.
``build_tiny_cases``/``latent_ref`` from ``../pro0921_latent_smoke`` provide the
deterministic fixtures and the independent numpy golden; this module re-derives
the compressed result a second way (``independent_contract_check``) so a golden
bug cannot pass unnoticed.

Commands:
    # materialize the frozen B1 set into fixtures + cases.json manifest
    python3 pro0921_grouped_cases.py --emit /tmp/pro0921_compressor_cases

    # run the materialized set on NPU (exit != 0 means a case failed)
    python3 pro0921_grouped_cases.py --run /tmp/pro0921_compressor_cases/cases.json

    # print the plan without touching a device (never fabricates results)
    python3 pro0921_grouped_cases.py --plan

    # emit batch .pt files consumable by test_compressor_batch.py
    python3 pro0921_grouped_cases.py --emit-pt /tmp/pro0921_compressor_pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


CONTRACT = "pro0921-grouped-compressor-v1"
OP = "compressor"
VALID_PATHS = ("normal", "grouped")
GROUPED_REQUIRED = ("x_cache_update",)
GROUPED_PAYLOAD = ("x_cache", "x_cache_slot_mapping")

PYTEST_DIR = Path(__file__).resolve().parent
SMOKE_DIR = PYTEST_DIR.parent / "pro0921_latent_smoke"


class CaseContractError(Exception):
    """Raised when a case does not satisfy the explicit pro0921 contract."""


def _smoke():
    """Import the pure-numpy fixture builders and the independent golden."""
    if str(SMOKE_DIR) not in sys.path:
        sys.path.insert(0, str(SMOKE_DIR))
    import build_tiny_cases  # noqa: E402
    import latent_ref  # noqa: E402

    return build_tiny_cases, latent_ref


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
REQUIRED_CASE_FIELDS = (
    "name",
    "path",
    "builder",
    "cmp_ratio",
    "coff",
    "cache_mode",
    "hidden_size",
    "head_dim",
    "batch",
    "seq",
    "dtype",
)

BUILDERS = ("hca_boundary", "hca_update_only", "csa_coff2", "normal")


def validate_case(case):
    """Validate one case entry. Missing ``path`` is an error, never a default."""
    if not isinstance(case, dict):
        raise CaseContractError("case entry must be an object, got %r" % type(case).__name__)
    missing = [key for key in REQUIRED_CASE_FIELDS if key not in case]
    if missing:
        raise CaseContractError("case %r missing fields: %s" % (case.get("name"), missing))
    if case["path"] not in VALID_PATHS:
        raise CaseContractError(
            "case %r has path=%r, expected one of %s (the grouped/normal path must be "
            "explicit)" % (case["name"], case["path"], list(VALID_PATHS))
        )
    if case["builder"] not in BUILDERS:
        raise CaseContractError(
            "case %r uses unknown builder %r, expected one of %s"
            % (case["name"], case["builder"], list(BUILDERS))
        )
    if case["dtype"] not in ("bfloat16", "float16"):
        raise CaseContractError("case %r dtype must be bfloat16 or float16" % case["name"])
    if case["cache_mode"] not in (1, 2):
        raise CaseContractError("case %r cache_mode must be 1 or 2" % case["name"])
    if case["path"] == "grouped":
        for key in GROUPED_REQUIRED:
            if key not in case:
                raise CaseContractError(
                    "grouped case %r is missing the grouped field %r; a normal case must "
                    "declare path=\"normal\" explicitly" % (case["name"], key)
                )
        if not isinstance(case["x_cache_update"], bool):
            raise CaseContractError(
                "grouped case %r x_cache_update must be a boolean" % case["name"]
            )
    return case


def validate_case_set(document):
    if not isinstance(document, dict):
        raise CaseContractError("case set must be a JSON object")
    if document.get("contract") != CONTRACT:
        raise CaseContractError(
            "case set contract must be %r, got %r" % (CONTRACT, document.get("contract"))
        )
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
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_case_set(document)


def default_case_set():
    """The frozen B1 coverage set: normal, grouped boundary, update-only, coff2, LI."""
    return {
        "contract": CONTRACT,
        "op": OP,
        "seed": 20260927,
        "cases": [
            {
                "name": "hca_normal_r128",
                "path": "normal",
                "builder": "normal",
                "cmp_ratio": 128,
                "coff": 1,
                "cache_mode": 1,
                "hidden_size": 1024,
                "head_dim": 128,
                "batch": 1,
                "seq": 128,
                "dtype": "bfloat16",
            },
            {
                "name": "hca_grouped_boundary",
                "path": "grouped",
                "builder": "hca_boundary",
                "cmp_ratio": 128,
                "coff": 1,
                "cache_mode": 1,
                "hidden_size": 1024,
                "head_dim": 128,
                "batch": 1,
                "seq": 1,
                "dtype": "bfloat16",
                "x_cache_update": True,
                "params": {"grad_enabled": False},
            },
            {
                "name": "hca_grouped_update_only",
                "path": "grouped",
                "builder": "hca_update_only",
                "cmp_ratio": 128,
                "coff": 1,
                "cache_mode": 1,
                "hidden_size": 1024,
                "head_dim": 128,
                "batch": 1,
                "seq": 1,
                "dtype": "bfloat16",
                "x_cache_update": True,
            },
            {
                "name": "csa_grouped_coff2_boundary",
                "path": "grouped",
                "builder": "csa_coff2",
                "cmp_ratio": 4,
                "coff": 2,
                "cache_mode": 1,
                "hidden_size": 1024,
                "head_dim": 512,
                "batch": 1,
                "seq": 1,
                "dtype": "bfloat16",
                "x_cache_update": True,
                "params": {"li": False},
            },
            {
                "name": "csa_grouped_li_read_shared",
                "path": "grouped",
                "builder": "csa_coff2",
                "cmp_ratio": 4,
                "coff": 2,
                "cache_mode": 1,
                "hidden_size": 1024,
                "head_dim": 128,
                "batch": 1,
                "seq": 1,
                "dtype": "bfloat16",
                "x_cache_update": False,
                "params": {"li": True},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------
def _build_normal(rng, case):
    """Normal-path (grouped switch off) B1 case: no x_cache, one full group."""
    build_tiny_cases, latent_ref = _smoke()
    hidden = case["hidden_size"]
    head = case["head_dim"]
    ratio = case["cmp_ratio"]
    seq = case["seq"]
    dtype = case["dtype"]
    if case["coff"] != 1:
        raise CaseContractError("normal builder only supports coff=1")
    if dtype not in ("bfloat16", "float16"):
        raise CaseContractError("normal builder only supports bfloat16/float16")

    def round_op_dtype(array):
        if dtype == "bfloat16":
            return latent_ref.to_bf16(array)
        return array.astype(np.float16).astype(np.float32)

    x = round_op_dtype(rng.standard_normal((1, seq, hidden)).astype(np.float32) * 0.02)
    wkv = round_op_dtype(rng.standard_normal((head, hidden)).astype(np.float32) * 0.02)
    wgate = round_op_dtype(rng.standard_normal((head, hidden)).astype(np.float32) * 0.02)
    ape = (rng.standard_normal((ratio, head)) * 0.01).astype(np.float32)
    state = np.zeros(build_tiny_cases.state_shape(case["coff"], head), dtype=np.float32)
    block_table = np.array([[1]], dtype=np.int32)
    cmp_kv, _, _, out_state = latent_ref.compress_golden(
        x, wkv, wgate, state, ape, block_table, ratio, case["coff"], case["cache_mode"],
        start_pos=[0], seqused=[seq])
    return {
        "name": case["name"],
        "kind": "compressor",
        "payload": {
            "x": x,
            "wkv": wkv,
            "wgate": wgate,
            "state_cache": state,
            "ape": ape,
            "state_block_table": block_table,
            "cu_seqlens": None,
            "seqused": np.array([seq], dtype=np.int32),
            "start_pos": np.array([0], dtype=np.int32),
            "x_cache": None,
            "x_cache_slot_mapping": None,
        },
        "outputs": (cmp_kv, out_state, None, None),
        "attrs": {
            "cmp_ratio": ratio, "coff": case["coff"], "cache_mode": case["cache_mode"],
            "state_cache_stride_dim0": build_tiny_cases.BLOCK_SIZE * (2 * case["coff"] * head),
            "grad_enabled": False, "x_cache_update": True,
        },
        "expected": {
            "checked_outputs": ["cmp_kv", "state_cache"],
            "suppressed_outputs": ["softmax_score", "kv"],
            "path": "normal",
            "note": ("grouped switch off: x_cache is not passed, the operator runs the "
                     "pre-existing full-compress path"),
        },
        "remark": ("HCA normal B1 S%d hidden%d head%d R%d; grouped off; independent CPU "
                   "golden" % (seq, hidden, head, ratio)),
        "readback": [],
    }


def build_case(case):
    build_tiny_cases, latent_ref = _smoke()
    rng = np.random.default_rng(case.get("seed", 20260927))
    params = case.get("params", {})
    if case["builder"] == "normal":
        built = _build_normal(rng, case)
    elif case["builder"] == "hca_boundary":
        built = build_tiny_cases.build_hca_boundary(rng, bool(params.get("grad_enabled", False)))
    elif case["builder"] == "hca_update_only":
        built = build_tiny_cases.build_hca_update_only(rng)
    elif case["builder"] == "csa_coff2":
        built = build_tiny_cases.build_csa_coff2(rng, bool(params.get("li", False)))
    else:  # pragma: no cover - validate_case already rejects this
        raise CaseContractError("unknown builder %r" % case["builder"])
    _assert_case_matches_spec(built, case)
    built["name"] = case["name"]
    return built


def _assert_case_matches_spec(built, case):
    attrs = built["attrs"]
    checks = (
        ("cmp_ratio", attrs["cmp_ratio"], case["cmp_ratio"]),
        ("coff", attrs["coff"], case["coff"]),
        ("cache_mode", attrs["cache_mode"], case["cache_mode"]),
    )
    for label, have, want in checks:
        if have != want:
            raise CaseContractError(
                "builder %r produced %s=%r but the case spec asked for %r"
                % (case["builder"], label, have, want)
            )
    payload = built["payload"]
    built_hidden = int(payload["x"].shape[-1])
    built_head = int(payload["wkv"].shape[0] // case["coff"])
    if built_hidden != case["hidden_size"]:
        raise CaseContractError(
            "builder %r produced hidden=%d but the case spec asked for %d"
            % (case["builder"], built_hidden, case["hidden_size"]))
    if built_head != case["head_dim"]:
        raise CaseContractError(
            "builder %r produced head_dim=%d but the case spec asked for %d"
            % (case["builder"], built_head, case["head_dim"]))
    if case["path"] == "grouped":
        for key in GROUPED_PAYLOAD:
            if payload.get(key) is None:
                raise CaseContractError(
                    "builder %r did not provide grouped payload %r" % (case["builder"], key)
                )
    else:
        for key in GROUPED_PAYLOAD:
            if payload.get(key) is not None:
                raise CaseContractError(
                    "normal case %r must not carry grouped payload %r" % (case["name"], key)
                )


def materialize(case_set, out_root):
    """Build the fixtures + golden for every case and write ``cases.json``."""
    build_tiny_cases, _ = _smoke()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"contract": CONTRACT, "op": OP, "root": str(out_root), "cases": []}
    for case in case_set["cases"]:
        built = build_case(case)
        x_dtype = build_tiny_cases.INPUT_DTYPES["x"]
        dest = build_tiny_cases.emit_case(out_root, built, x_dtype)
        payload = built["payload"]
        entry = {
            "name": built["name"],
            "path": case["path"],
            "dir": str(dest),
            "builder": case["builder"],
            "params": case.get("params", {}),
            "seed": case.get("seed", case_set.get("seed", 20260927)),
            "dtype": case["dtype"],
            "cmp_ratio": case["cmp_ratio"],
            "coff": case["coff"],
            "cache_mode": case["cache_mode"],
            "hidden_size": case["hidden_size"],
            "head_dim": case["head_dim"],
            "batch": case["batch"],
            "seq": case["seq"],
            "x_cache_update": bool(case.get("x_cache_update", True)),
            "start_pos": _tolist(payload["start_pos"]),
            "seqused": _tolist(payload["seqused"]),
            "state_block_table": _tolist(payload["state_block_table"]),
            "x_cache_slot_mapping": _tolist(payload["x_cache_slot_mapping"]),
            "grad_enabled": bool(built["attrs"].get("grad_enabled", False)),
            "readback": list(built.get("readback", [])),
            "expected": built["expected"],
        }
        manifest["cases"].append(entry)
    (out_root / "cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _tolist(value):
    return None if value is None else [int(v) for v in np.asarray(value).reshape(-1)]


def load_manifest(path):
    """Load a materialized manifest (used by ``--run``)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("contract") != CONTRACT:
        raise CaseContractError(
            "manifest contract must be %r, got %r" % (CONTRACT, document.get("contract"))
        )
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CaseContractError("manifest must carry a non-empty 'cases' list")
    for entry in cases:
        if entry.get("path") not in VALID_PATHS:
            raise CaseContractError(
                "manifest case %r has path=%r" % (entry.get("name"), entry.get("path"))
            )
        if entry["path"] == "grouped" and "x_cache_update" not in entry:
            raise CaseContractError(
                "manifest grouped case %r is missing x_cache_update" % entry.get("name")
            )
    return document


# ---------------------------------------------------------------------------
# Independent numeric recompute (CPU, no device) -- second derivation
# ---------------------------------------------------------------------------
def _load(case_dir, name):
    path = Path(case_dir) / (name + ".npy")
    return np.load(path) if path.exists() else None


def _attributes(case_dir):
    import ast

    row = next(csv.DictReader((Path(case_dir) / "case.csv").open(encoding="utf-8")))
    return ast.literal_eval(row["attributes"])


def _softmax_cols(z):
    z = np.asarray(z, dtype=np.float32)
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z).astype(np.float32)
    return (e / np.sum(e, axis=0, keepdims=True)).astype(np.float32)


def _grouped_window(entry, x, x_cache, slots, ratio):
    rows = x_cache.shape[1]
    bank = int(slots[0]) // rows
    start_pos = int(np.asarray(entry["start_pos"]).reshape(-1)[0])
    seqused = int(np.asarray(entry["seqused"]).reshape(-1)[0])
    update = bool(entry["x_cache_update"])
    x_cache = x_cache.copy()
    if update and int(slots[0]) >= 0:
        flat = x.reshape(-1, x.shape[-1])
        x_cache[int(slots[0]) // rows, int(slots[0]) % rows] = flat[seqused - 1]
    group_offset = start_pos % ratio
    if seqused < ratio - group_offset:
        return None, None, x_cache
    group_end = ((start_pos + seqused) // ratio) * ratio
    window_start = group_end - ratio
    row_idx = [(window_start + k) % rows for k in range(ratio)]
    return x_cache[bank, row_idx].copy(), window_start, x_cache


def independent_contract_check(case_dir, entry):
    """Re-derive cmp_kv/state with a plain-numpy path distinct from the golden.

    Returns a list of ``(check_name, ok, value)`` tuples.
    """
    checks = []
    golden_json = json.loads((Path(case_dir) / "golden.json").read_text(encoding="utf-8"))
    goldens = {
        n: (None if f is None else np.load(Path(case_dir) / f))
        for n, f in zip(golden_json["output_names"], golden_json["files"])
    }
    attrs = _attributes(case_dir)
    ratio = int(attrs["cmp_ratio"])
    coff = int(attrs["coff"])
    x = _load(case_dir, "x").astype(np.float32)
    wkv = _load(case_dir, "wkv").astype(np.float32)
    wgate = _load(case_dir, "wgate").astype(np.float32)
    ape = _load(case_dir, "ape").astype(np.float32)
    state_in = _load(case_dir, "state_cache").astype(np.float32)
    head_dim = wkv.shape[0] // coff

    if entry["path"] == "grouped":
        x_cache = _load(case_dir, "x_cache").astype(np.float32)
        slots = _load(case_dir, "x_cache_slot_mapping").reshape(-1)
        window, window_start, refreshed = _grouped_window(entry, x, x_cache, slots, ratio)
        if bool(entry["x_cache_update"]) and int(slots[0]) >= 0:
            rows = x_cache.shape[1]
            expect_cache = x_cache.copy()
            expect_cache[int(slots[0]) // rows, int(slots[0]) % rows] = \
                x.reshape(-1, x.shape[-1])[int(np.asarray(entry["seqused"]).reshape(-1)[0]) - 1]
            checks.append((entry["name"] + ":x_cache_ring_refresh",
                           bool(np.array_equal(refreshed, expect_cache)), 0.0))
        if window is None:
            checks.append((entry["name"] + ":update_only_state_unchanged",
                           bool(np.array_equal(goldens["state_cache"], state_in)), 0.0))
            return checks
    else:
        window = x.reshape(-1, x.shape[-1])
        window_start = int(np.asarray(entry["start_pos"]).reshape(-1)[0])

    kv_full = np.matmul(window, wkv.T, dtype=np.float32)
    score_full = np.matmul(window, wgate.T, dtype=np.float32)
    logical = window.shape[0]
    score_full = score_full + ape[:logical]
    if coff == 2:
        # coff=2 overlap: the previous group is the constant slice-0 state rows
        # [window_start - ratio, window_start); the current group is slice 1.
        prev = state_in[1, window_start - ratio:window_start]
        prev_kv = prev[:, 0:head_dim]
        prev_score = prev[:, coff * head_dim: coff * head_dim + head_dim]
        sc_kv = np.concatenate([prev_kv, kv_full[:, head_dim:2 * head_dim]], axis=0)
        sc_score = np.concatenate([prev_score, score_full[:, head_dim:2 * head_dim]], axis=0)
    else:
        sc_kv, sc_score = kv_full, score_full

    weights = _softmax_cols(sc_score)
    independent = (weights * sc_kv).sum(axis=0, keepdims=True).reshape(1, 1, head_dim)
    if goldens["cmp_kv"] is not None:
        delta = float(np.abs(independent - goldens["cmp_kv"]).max())
        checks.append((entry["name"] + ":cmp_kv_independent_recompute", delta < 1e-4, delta))

    state_out = goldens["state_cache"]
    if coff == 2:
        checks.append((entry["name"] + ":coff2_previous_rows_untouched",
                       bool(np.array_equal(state_out[1, :ratio], state_in[1, :ratio])), 0.0))
    checks.append((entry["name"] + ":state_window_rows_written",
                   not np.array_equal(state_out[1], state_in[1]), 0.0))
    checks.append((entry["name"] + ":unused_block_stays_zero",
                   float(np.abs(state_out[0]).max()) == 0.0, float(np.abs(state_out[0]).max())))
    return checks


# ---------------------------------------------------------------------------
# Normal <-> grouped equivalence of a completed group (CPU)
# ---------------------------------------------------------------------------
def equivalence_check(seed=20260927):
    """A completed group's grouped projection must equal the normal compression.

    Independent of the device: the grouped path projects one window at the
    boundary, the normal path compresses the same window in one shot.  With the
    same initial state and logical history the compiled cmp_kv must match.
    """
    _, latent_ref = _smoke()
    rng = np.random.default_rng(seed)
    ratio = 4
    hidden = 64
    head = 32
    tokens = latent_ref.to_bf16(rng.standard_normal((ratio, hidden)).astype(np.float32) * 0.02)
    wkv = latent_ref.to_bf16(rng.standard_normal((head, hidden)).astype(np.float32) * 0.02)
    wgate = latent_ref.to_bf16(rng.standard_normal((head, hidden)).astype(np.float32) * 0.02)
    ape = (rng.standard_normal((ratio, head)) * 0.01).astype(np.float32)
    state = np.zeros((2, 128, 2 * head), dtype=np.float32)
    block = np.array([[1]], dtype=np.int32)
    normal, _, _, normal_state = latent_ref.compress_golden(
        tokens[None, :, :], wkv, wgate, state, ape, block, ratio, 1, 1,
        start_pos=[0], seqused=[ratio])
    x_cache = np.zeros((2, ratio, hidden), dtype=np.float32)
    x_cache[1] = tokens
    grouped = latent_ref.grouped_case_golden(
        tokens[ratio - 1: ratio][None, :, :], wkv, wgate, state, ape, block, x_cache,
        np.array([1 * ratio + ratio - 1], dtype=np.int32), ratio, 1, 1,
        start_pos=np.array([ratio - 1]), seqused=np.array([1]))
    cmp_delta = float(np.abs(normal - grouped["cmp_kv"]).max())
    state_delta = float(np.abs(normal_state - grouped["state_cache"]).max())
    return {
        "cmp_kv_max_abs": cmp_delta,
        "state_max_abs": state_delta,
        "pass": cmp_delta < 1e-5 and state_delta < 1e-5,
    }


def cycle_check(ratio=4, coff=1, seed=20260927, hidden=64, head=32):
    """Drive a full compression cycle token by token through the grouped path.

    The ring holds exactly ``ratio`` rows, so the previous group is evicted and
    (for coff=2) must come from the original state.  Every step resets the
    mutable ring/state from the pristine fixture, so a stale ring cannot leak
    across trials.  Returns the grouped/normal deltas against a hand-rolled
    numeric reference (not ``compress_golden``), plus reset determinism.
    """
    _, latent_ref = _smoke()
    rng = np.random.default_rng(seed)

    def tok(count):
        return latent_ref.to_bf16(
            rng.standard_normal((count, hidden)).astype(np.float32) * 0.02)

    prev = tok(ratio)
    cur = tok(ratio)
    wkv = latent_ref.to_bf16(rng.standard_normal((coff * head, hidden)).astype(np.float32) * 0.02)
    wgate = latent_ref.to_bf16(rng.standard_normal((coff * head, hidden)).astype(np.float32) * 0.02)
    ape = (rng.standard_normal((ratio, coff * head)) * 0.01).astype(np.float32)
    # The cycle walks absolute positions [ratio, 2*ratio); with block_size=128 a
    # ratio-128 cycle reaches block index 1, so the table needs two entries.
    block = np.array([[1, 2]], dtype=np.int32)
    state0 = np.zeros((3, 128, 2 * coff * head), dtype=np.float32)
    _, _, _, state_prev = latent_ref.compress_golden(
        prev[None, :, :], wkv, wgate, state0, ape, block, ratio, coff, 1,
        start_pos=[0], seqused=[ratio])
    ring0 = np.zeros((2, ratio, hidden), dtype=np.float32)

    def run_cycle():
        state = state_prev.copy()
        ring = ring0.copy()
        best = None
        state_before_last = None
        for step in range(ratio):
            state_before_last = state.copy()
            golden = latent_ref.grouped_case_golden(
                cur[step: step + 1][None, :, :], wkv, wgate, state, ape, block, ring,
                np.array([1 * ratio + step], dtype=np.int32), ratio, coff, 1,
                start_pos=np.array([ratio + step]), seqused=np.array([1]))
            state = golden["state_cache"]
            ring = golden["x_cache"]
            if golden["active"]:
                best = golden["cmp_kv"]
        return best, state_before_last, ring

    grouped, state_before_last, ring = run_cycle()
    grouped_again, _, ring_again = run_cycle()
    reset_stable = bool(np.array_equal(ring, ring_again)) and grouped is not None and \
        bool(np.array_equal(grouped, grouped_again))
    if grouped is None:
        raise AssertionError("the cycle never completed a group for ratio=%d" % ratio)

    # Hand-rolled numeric reference for the completed group only.
    prev_kv = state_before_last[1, :ratio, 0:head]
    prev_score = state_before_last[1, :ratio, coff * head: coff * head + head]
    cur_score = np.matmul(cur, wgate.T, dtype=np.float32)[:, coff * head - head: coff * head]
    cur_score = cur_score + ape[:, coff * head - head: coff * head]
    cur_kv = np.matmul(cur, wkv.T, dtype=np.float32)[:, coff * head - head: coff * head]
    if coff == 2:
        sc_kv = np.concatenate([prev_kv, cur_kv], axis=0)
        sc_score = np.concatenate([prev_score, cur_score], axis=0)
    else:
        sc_kv, sc_score = cur_kv, cur_score
    weights = _softmax_cols(sc_score)
    reference = (weights * sc_kv).sum(axis=0, keepdims=True).reshape(1, 1, head)
    grouped_delta = float(np.abs(grouped - reference).max())

    # Normal path over the same window with the same history.
    normal, _, _, normal_state = latent_ref.compress_golden(
        cur[None, :, :], wkv, wgate, state_before_last, ape, block, ratio, coff, 1,
        start_pos=[ratio], seqused=[ratio])
    normal_delta = float(np.abs(normal - reference).max())
    return {
        "ratio": ratio,
        "coff": coff,
        "grouped_vs_reference_max_abs": grouped_delta,
        "normal_vs_reference_max_abs": normal_delta,
        "reset_stable": reset_stable,
        "pass": grouped_delta < 1e-4 and normal_delta < 1e-4 and reset_stable,
    }


# ---------------------------------------------------------------------------
# NPU accuracy execution
# ---------------------------------------------------------------------------
def _load_npu_runtime():
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError(
            "no NPU device is available; run with --plan to print the plan instead. "
            "This runner never fabricates results without a device."
        )
    try:
        from cann_ops_transformer.ops import compressor
    except Exception as exc:  # pragma: no cover - depends on the on-board install
        raise RuntimeError(
            "the formal provider cann_ops_transformer.ops.compressor is not importable; "
            "install the current dual-repo operators and wheel first, using the "
            "torch -> torch_npu -> formal provider import order (%r)" % exc
        )

    return torch, compressor


def _round_to_op_dtype(array, dtype):
    """Round a float array to the exact values the operator's buffer holds.

    numpy has no bfloat16, so the fixtures are stored as float32, but the device
    buffer is created with the op dtype.  A value that the op dtype cannot
    represent (the 0.1 sentinel the builders leave in the ring, read back as
    0.10009765625 in bf16) is therefore stored rounded, and the expected ring and
    its ``modified`` baseline must be rounded the same way before the strict
    ``np.array_equal``.  This is a dtype alignment, never a tolerance compare.
    """
    if array is None:
        return None
    if dtype == "bfloat16":
        _, latent_ref = _smoke()
        return latent_ref.to_bf16(array)
    return np.asarray(array, dtype=np.float32).astype(np.float16).astype(np.float32)


def _to_device(torch, array, dtype, device):
    if array is None:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(dtype=dtype, device=device)


_INT32 = ("start_pos", "seqused", "state_block_table", "cu_seqlens", "x_cache_slot_mapping")
_FP32 = ("state_cache", "ape")
_OP_INPUT_DTYPES = {"bfloat16": None, "float16": None}


def _run_one_accuracy(torch, compressor, entry, device_id):
    case_dir = Path(entry["dir"])
    device = "npu:%d" % device_id
    op_dtype = torch.bfloat16 if entry["dtype"] == "bfloat16" else torch.float16
    payload = {name: _load(case_dir, name) for name in (
        "x", "wkv", "wgate", "state_cache", "ape", "state_block_table",
        "cu_seqlens", "seqused", "start_pos", "x_cache", "x_cache_slot_mapping")}
    payload["x"] = payload["x"].astype(np.float32)

    def dev(name):
        value = payload[name]
        if value is None:
            return None
        if name in _INT32:
            return _to_device(torch, value.astype(np.int32), torch.int32, device)
        if name in _FP32:
            return _to_device(torch, value.astype(np.float32), torch.float32, device)
        return _to_device(torch, value, op_dtype, device)

    state_cache = dev("state_cache")
    x_cache = dev("x_cache")
    kwargs = dict(
        cmp_ratio=entry["cmp_ratio"],
        state_block_table=dev("state_block_table"),
        cu_seqlens=dev("cu_seqlens"),
        seqused=dev("seqused"),
        start_pos=dev("start_pos"),
        coff=entry["coff"],
        cache_mode=entry["cache_mode"],
    )
    if entry["path"] == "grouped":
        kwargs["x_cache"] = x_cache
        kwargs["x_cache_slot_mapping"] = dev("x_cache_slot_mapping")
        kwargs["x_cache_update"] = bool(entry["x_cache_update"])
    cmp_kv = compressor(dev("x"), dev("wkv"), dev("wgate"), state_cache, dev("ape"), **kwargs)
    torch.npu.synchronize()
    # numpy has no bfloat16: widen the device tensor to float32 before .numpy()
    # (a raw .numpy() on a bf16 tensor raises).  The readback is therefore the
    # on-device op-dtype values widened to float32, so the ring check below must
    # round its expectation to the op dtype instead of comparing raw float32.
    cmp_kv_cpu = cmp_kv.detach().cpu().to(torch.float32).numpy()
    state_cpu = state_cache.detach().cpu().to(torch.float32).numpy()
    x_cache_cpu = None if x_cache is None else x_cache.detach().cpu().to(torch.float32).numpy()

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
        # The device buffer holds the op dtype, so a fixture row the operator
        # never touches still reads back rounded; the unchanged baseline and the
        # written token are rounded the same way before the exact compare.
        ring_baseline = _round_to_op_dtype(payload["x_cache"], entry["dtype"])
        if entry["x_cache_update"] and int(payload["x_cache_slot_mapping"].reshape(-1)[0]) >= 0:
            rows = payload["x_cache"].shape[1]
            slot = int(payload["x_cache_slot_mapping"].reshape(-1)[0])
            token = _round_to_op_dtype(
                payload["x"].reshape(-1, payload["x"].shape[-1])[
                    int(payload["seqused"].reshape(-1)[0]) - 1], entry["dtype"])
            expect_cache = ring_baseline.copy()
            expect_cache[slot // rows, slot % rows] = token
        else:
            expect_cache = ring_baseline
        same = np.array_equal(x_cache_cpu, expect_cache)
        checks.append((entry["name"] + ":x_cache_side_effect",
                       {"pass": bool(same), "modified": bool(not np.array_equal(
                           x_cache_cpu, ring_baseline)), "expected_modified": bool(
                           not np.array_equal(expect_cache, ring_baseline))}))
    return {
        "name": entry["name"],
        "path": entry["path"],
        "cmp_ratio": entry["cmp_ratio"],
        "coff": entry["coff"],
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
    """Numeric compare that never masks NaN/Inf.

    ``equal_nan`` is deliberately not used: a NaN or Inf in either the golden or
    the device readback is an operator/reference fault, not a tolerance hit.  A
    non-finite value fails the check outright.
    """
    expect = np.asarray(expect, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if expect.size != actual.size:
        return {"pass": False, "reason": "shape mismatch",
                "expect_size": int(expect.size), "actual_size": int(actual.size)}
    if expect.size == 0:
        return {"pass": True, "fulfill_percent": 100.0}
    nonfinite = int((~np.isfinite(expect)).sum() + (~np.isfinite(actual)).sum())
    if nonfinite:
        return {
            "pass": False, "reason": "non-finite value present",
            "expect_nonfinite": int((~np.isfinite(expect)).sum()),
            "actual_nonfinite": int((~np.isfinite(actual)).sum()),
            "rtol": rtol, "atol": atol,
        }
    isclose = np.isclose(actual, expect, rtol=rtol, atol=atol)
    fulfill = float(isclose.mean() * 100.0)
    return {
        "pass": bool(fulfill >= 99.5),
        "fulfill_percent": fulfill,
        "max_abs": float(np.abs(actual - expect).max()),
        "rtol": rtol,
        "atol": atol,
    }


def run_accuracy(manifest, device_id=0, out_json=None, provider_note=None):
    torch, compressor = _load_npu_runtime()
    torch.npu.set_device(int(device_id))
    results = [_run_one_accuracy(torch, compressor, entry, device_id) for entry in manifest["cases"]]
    report = {
        "contract": CONTRACT,
        "device": int(device_id),
        "provider": provider_note or _provider_note(torch),
        "cases": results,
    }
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [c["name"] for c in results if not all(chk["ok"] for chk in c["checks"])]
    report["failed"] = failed
    return report


def _provider_note(torch):
    import cann_ops_transformer

    return {
        "module": "cann_ops_transformer.ops.compressor",
        "torch": torch.__version__,
        "cann_ops_transformer": getattr(cann_ops_transformer, "__version__", "unknown"),
    }


# ---------------------------------------------------------------------------
# Batch .pt emission (keeps the existing test_compressor_batch compare chain)
# ---------------------------------------------------------------------------
def emit_pt(entry, pt_path):
    """Emit a ``.pt`` that ``batch/compressor_pt_loadprocess.py`` can run.

    Normal and grouped cases both carry ``path``; grouped cases additionally
    carry ``x_cache``/``x_cache_slot_mapping``/``x_cache_update`` plus the full
    ``expected_x_cache`` ring the batch loader checks the operator's in-place
    read-back against.  The golden fields reuse the numpy grouped golden so the
    existing compare chain stays meaningful.
    """
    import torch

    case_dir = Path(entry["dir"])
    built = _load_case_built(entry, case_dir)
    payload = built["payload"]
    golden_json = json.loads((case_dir / "golden.json").read_text(encoding="utf-8"))
    goldens = {n: (None if f is None else np.load(case_dir / f))
               for n, f in zip(golden_json["output_names"], golden_json["files"])}

    def tensor(array, dtype=None):
        if array is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(array))
        return t.to(dtype) if dtype is not None else t

    # The operator ABI expects x/wkv/wgate/x_cache in the op dtype and ape /
    # state in float32; pass the real dtypes instead of whatever numpy produced
    # (a float32 x_cache would be rejected by the kernel).
    op_dtype = torch.bfloat16 if entry["dtype"] == "bfloat16" else torch.float16
    cmp_kv = goldens["cmp_kv"]
    if cmp_kv is None:
        cmp_kv = np.zeros((entry["batch"], (entry["seq"] + entry["cmp_ratio"] - 1)
                           // entry["cmp_ratio"], entry["head_dim"]), dtype=np.float32)
        mask = np.zeros_like(cmp_kv, dtype=bool)
    else:
        mask = np.ones_like(cmp_kv, dtype=bool)
    state_out = goldens["state_cache"] if goldens["state_cache"] is not None else payload["state_cache"]
    kv_half = state_out.shape[2] // 2
    updated = state_out != payload["state_cache"]
    output = {
        "params": _params_tuple(entry),
        "path": entry["path"],
        "x_cache_update": bool(entry.get("x_cache_update", True)),
        "cpu_result": tensor(cmp_kv, torch.bfloat16),
        "kv_mask_result": torch.from_numpy(mask),
        "update_kv": torch.from_numpy(updated[:, :, :kv_half]),
        "update_score": torch.from_numpy(updated[:, :, kv_half:]),
        "cpu_kv_state": tensor(state_out[:, :, :kv_half]),
        "cpu_score_state": tensor(state_out[:, :, kv_half:]),
        "x": tensor(payload["x"], op_dtype),
        "wkv": tensor(payload["wkv"], op_dtype),
        "wgate": tensor(payload["wgate"], op_dtype),
        "kv_state": tensor(payload["state_cache"][:, :, :kv_half]),
        "score_state": tensor(payload["state_cache"][:, :, kv_half:]),
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
        output["x_cache"] = tensor(payload["x_cache"], op_dtype)
        output["x_cache_slot_mapping"] = tensor(payload["x_cache_slot_mapping"], torch.int32)
        output["expected_x_cache"] = tensor(
            _expected_grouped_ring(entry, payload), op_dtype)
    Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, pt_path)
    return output


def _expected_grouped_ring(entry, payload):
    """Full ring after one grouped call, from the independent numpy golden.

    ``x_cache_update=False`` (LI read-only shared cache) leaves the ring
    untouched, so the expected ring is the input ring itself.
    """
    _, latent_ref = _smoke()
    out = latent_ref.grouped_case_golden(
        np.asarray(payload["x"], dtype=np.float32),
        np.asarray(payload["wkv"], dtype=np.float32),
        np.asarray(payload["wgate"], dtype=np.float32),
        np.asarray(payload["state_cache"], dtype=np.float32),
        np.asarray(payload["ape"], dtype=np.float32),
        np.asarray(payload["state_block_table"]),
        np.asarray(payload["x_cache"], dtype=np.float32),
        np.asarray(payload["x_cache_slot_mapping"]),
        entry["cmp_ratio"], entry["coff"], entry["cache_mode"],
        start_pos=np.asarray(payload["start_pos"]),
        seqused=np.asarray(payload["seqused"]),
        x_cache_update=bool(entry.get("x_cache_update", True)),
    )
    return np.asarray(out["x_cache"], dtype=np.float32)


def _params_tuple(entry):
    return [
        entry["batch"], entry["hidden_size"], entry["seq"], entry["head_dim"],
        128, entry["cmp_ratio"], entry["coff"], entry["start_pos"][0],
        entry["cache_mode"], "BSH", entry["dtype"].upper(), None,
        str(entry["seqused"]), str(entry["start_pos"]),
        "[-10,10]", "[-10,10]", "[-10,10]", "[-10,10]", "[-10,10]", "[-10,10]",
    ]


def emit_pt_from_spec(spec, pt_path):
    """Validate a single case spec, materialize it, and emit one ``.pt``.

    Used by ``batch/compressor_pt_save.py`` so an explicitly grouped Excel row
    is routed through the same contract instead of the normal builder.
    """
    validate_case(spec)
    import tempfile

    manifest = materialize({"contract": CONTRACT, "op": OP, "cases": [spec]},
                           Path(tempfile.mkdtemp(prefix="pro0921_pt_")))
    return emit_pt(manifest["cases"][0], pt_path)


def _load_case_built(entry, case_dir):
    """Reconstruct the built case (payload/golden) from a materialized manifest."""
    spec = {
        "name": entry["name"],
        "path": entry["path"],
        "builder": entry.get("builder", _builder_for(entry["name"])),
        "cmp_ratio": entry["cmp_ratio"],
        "coff": entry["coff"],
        "cache_mode": entry["cache_mode"],
        "hidden_size": entry["hidden_size"],
        "head_dim": entry["head_dim"],
        "batch": entry["batch"],
        "seq": entry["seq"],
        "dtype": entry["dtype"],
        "x_cache_update": entry.get("x_cache_update", True),
        "seed": entry.get("seed", 20260927),
        "params": entry.get("params", {}),
    }
    return build_case(spec)


def _builder_for(name):
    return {
        "hca_grouped_boundary": "hca_boundary",
        "hca_grouped_update_only": "hca_update_only",
        "csa_grouped_coff2_boundary": "csa_coff2",
        "csa_grouped_li_read_shared": "csa_coff2",
        "hca_normal_r128": "normal",
    }.get(name, "normal")


def pt_grouped_args(test_data):
    """Explicit op kwargs selector for ``batch/compressor_pt_loadprocess.py``.

    ``path`` is mandatory: a ``.pt`` emitted by :func:`emit_pt` always declares
    ``path`` (``"normal"`` or ``"grouped"``).  A file without ``path`` is a
    pre-grouped ABI and is rejected here; it must be regenerated with the
    current writer.  A grouped case must also carry the x_cache fields or this
    raises instead of silently running the normal path.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _plan(manifest):
    print("contract=%s op=%s cases=%d" % (CONTRACT, OP, len(manifest["cases"])))
    for entry in manifest["cases"]:
        print("  %-30s path=%-7s R=%-4d coff=%d update=%-5s shape=B%dS%dH%d/head%d"
              % (entry["name"], entry["path"], entry["cmp_ratio"], entry["coff"],
                 entry["x_cache_update"], entry["batch"], entry["seq"],
                 entry["hidden_size"], entry["head_dim"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pro0921 grouped Compressor accuracy entry")
    parser.add_argument("--cases", type=Path, help="input case-set JSON (default: frozen B1 set)")
    parser.add_argument("--emit", type=Path, help="materialize fixtures + cases.json here")
    parser.add_argument("--run", type=Path, help="run a materialized cases.json on NPU")
    parser.add_argument("--emit-pt", type=Path, help="emit .pt files for test_compressor_batch.py")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--plan", action="store_true", help="print the plan only, never a device")
    args = parser.parse_args(argv)

    try:
        case_set = validate_case_set(load_case_set(args.cases)) if args.cases else default_case_set()
    except CaseContractError as exc:
        print("PRO0921_CASE_CONTRACT=FAIL %s" % exc)
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

            manifest = materialize(case_set, Path(tempfile.mkdtemp(prefix="pro0921_compressor_pt_")))
        for entry in manifest["cases"]:
            dest = args.emit_pt / (entry["name"] + ".pt")
            emit_pt(entry, dest)
            print("pt -> %s" % dest)

    if args.run:
        try:
            target = load_manifest(args.run)
        except CaseContractError as exc:
            print("PRO0921_CASE_CONTRACT=FAIL %s" % exc)
            return 1
    elif args.emit and not args.emit_pt:
        target = manifest
    else:
        target = None
    if target is not None:
        try:
            report = run_accuracy(target, device_id=args.device, out_json=args.out_json)
        except RuntimeError as exc:
            print("PRO0921_ACCURACY=FAIL %s" % exc)
            return 2
        for case in report["cases"]:
            status = "PASS" if all(chk["ok"] for chk in case["checks"]) else "FAIL"
            print("  %-30s %s" % (case["name"], status))
        if report["failed"]:
            print("PRO0921_ACCURACY=FAIL cases=%s" % ",".join(report["failed"]))
            return 1
        print("PRO0921_ACCURACY=PASS cases=%d" % len(report["cases"]))
    return 0


def _plan_only_manifest(case_set):
    cases = []
    for case in case_set["cases"]:
        cases.append({
            "name": case["name"], "path": case["path"], "dir": "<unbuilt>",
            "cmp_ratio": case["cmp_ratio"], "coff": case["coff"],
            "cache_mode": case["cache_mode"], "hidden_size": case["hidden_size"],
            "head_dim": case["head_dim"], "batch": case["batch"], "seq": case["seq"],
            "x_cache_update": bool(case.get("x_cache_update", True)),
        })
    return {"contract": CONTRACT, "op": OP, "cases": cases}


if __name__ == "__main__":
    raise SystemExit(main())

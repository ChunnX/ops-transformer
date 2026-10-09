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
"""Paired on-board accuracy / performance runner for the Pro0921 Compressor.

Both paths of every scenario are derived from **one shared fixture**: the same
``x`` tokens, ``wkv``/``wgate``, ``ape``, initial state, logical previous group
and seed.  A ``boundary`` scenario is exactly one S1 operator call on the last
token of a group on *both* paths: the grouped path reads the completed window
from ``x_cache``, the normal path reads it from the state rows that were warmed
with the first ``ratio - 1`` tokens outside the timing window.  A
``nonboundary`` scenario is one S1 call on a token that does not finish a group
on both paths.  A ``cycle`` scenario walks the whole group token by token on
both paths.  Batch is fixed to B1 and each operator call is S1; an unsupported
batch/seq is rejected instead of silently claimed as covered.

Matrix (12 scenarios): for CSA (cmp_ratio=4, coff=2) and HCA (cmp_ratio=128,
coff=1) in both normal and grouped, the boundary token, a non-boundary token and
a full compression cycle.

Rules that keep a number honest:

* the timing window never contains the ring/state reset nor the host-to-device
  upload; each trial resets from a pristine device copy and synchronises first;
* every prepared scenario is numerically verified on the *same device run*
  against the independent numpy golden before any timing is recorded, so a
  functional failure can never produce a performance PASS;
* only equal-scope pairs are divided (normal vs grouped for the same geometry
  and the same logical work); non-boundary ``cmp_kv`` is undefined and is not
  compared;
* samples must be finite and strictly positive; raw samples plus min/median are
  preserved.

Commands:
    python3 pro0921_accuracy_perf.py --mode plan
    python3 pro0921_accuracy_perf.py --mode accuracy --accuracy-out /tmp/pro0921_accuracy.json
    python3 pro0921_accuracy_perf.py --mode perf --perf-out /tmp/pro0921_perf.json --warmup 5 --repeats 20
    python3 pro0921_accuracy_perf.py --mode all
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

import pro0921_grouped_cases as cases


CONTRACT = cases.CONTRACT
OP = cases.OP

BLOCK_SIZE = 128
BLOCKS = 2
SENTINEL = np.float32(0.1)

# Explicit geometry: head_dim is stated, never derived from hidden_size.  CSA
# keeps a real previous-group projection in state_cache; HCA's first group has
# no previous group (zero state).
GEOMETRIES = (
    {"key": "csa", "ratio": 4, "coff": 2, "head_dim": 512, "hidden": 1024, "prev_group": True},
    {"key": "hca", "ratio": 128, "coff": 1, "head_dim": 128, "hidden": 1024, "prev_group": False},
)
GEOM_BY_KEY = {g["key"]: g for g in GEOMETRIES}

KINDS = ("boundary", "nonboundary", "cycle")
PATHS = ("normal", "grouped")


def _make_scenarios():
    scenarios = []
    for geom in GEOMETRIES:
        for path in PATHS:
            for kind in KINDS:
                if kind == "cycle":
                    steps = geom["ratio"]
                else:
                    # boundary and non-boundary are a single S1 call on *both*
                    # paths, so a normal/grouped pair never mixes a full cycle
                    # with one token.
                    steps = 1
                scenarios.append({
                    "name": "%s_%s_%s" % (geom["key"], path, kind),
                    "geom": geom["key"], "path": path, "kind": kind,
                    "ratio": geom["ratio"], "coff": geom["coff"],
                    "head": geom["head_dim"], "hidden": geom["hidden"],
                    "steps": steps,
                    "note": _note(path, kind),
                })
    return tuple(scenarios)


def _note(path, kind):
    if kind == "boundary":
        return ("grouped reads the completed window from x_cache in one S1 boundary call"
                if path == "grouped" else
                "normal projects the final group token against a state warmed with the "
                "first ratio-1 tokens (one S1 call)")
    if kind == "nonboundary":
        return ("grouped non-boundary token: ring refresh only, cmp_kv undefined"
                if path == "grouped" else
                "normal non-boundary token: state accumulation only, cmp_kv undefined")
    return ("grouped full cycle: ratio single-token steps, projection at the boundary"
            if path == "grouped" else
            "normal full cycle: ratio single-token steps")


SCENARIOS = _make_scenarios()


# ---------------------------------------------------------------------------
# Shared fixture -> per-path scenario
# ---------------------------------------------------------------------------
def _token(ref, rng, count, hidden):
    return ref.to_bf16(rng.standard_normal((count, hidden)).astype(np.float32) * 0.02)


def _shared_fixture(geom, seed):
    """One deterministic fixture per geometry; both paths derive from it."""
    _, ref = cases._smoke()
    rng = np.random.default_rng(seed)
    ratio, coff, head, hidden = geom["ratio"], geom["coff"], geom["head_dim"], geom["hidden"]
    wkv = ref.to_bf16(rng.standard_normal((coff * head, hidden)).astype(np.float32) * 0.02)
    wgate = ref.to_bf16(rng.standard_normal((coff * head, hidden)).astype(np.float32) * 0.02)
    ape = (rng.standard_normal((ratio, coff * head)) * 0.01).astype(np.float32)
    block_table = np.array([[1]], dtype=np.int32)
    state0 = np.zeros((BLOCKS, BLOCK_SIZE, 2 * coff * head), dtype=np.float32)
    prev_cmp = None
    if geom["prev_group"]:
        prev_tokens = _token(ref, rng, ratio, hidden)
        prev_cmp, _, _, state0 = ref.compress_golden(
            prev_tokens[None, :, :], wkv, wgate, state0, ape, block_table, ratio, coff, 1,
            start_pos=[0], seqused=[ratio])
        window_start = ratio
    else:
        window_start = 0
    tokens = _token(ref, rng, ratio, hidden)
    # The operator's contract golden: the whole window projected as one group.
    golden_cmp, _, _, golden_state = ref.compress_golden(
        tokens[None, :, :], wkv, wgate, state0, ape, block_table, ratio, coff, 1,
        start_pos=[window_start], seqused=[ratio])
    return {
        "geom": geom, "ratio": ratio, "coff": coff, "head": head, "hidden": hidden,
        "window_start": window_start, "wkv": wkv, "wgate": wgate, "ape": ape,
        "block_table": block_table, "state0": state0, "tokens": tokens,
        "prev_cmp": prev_cmp, "golden_cmp": golden_cmp, "golden_state": golden_state,
    }


def _slot(abs_pos, rows):
    return int(1 * rows + (abs_pos % rows))


def _ring_window(shared, kind):
    """Grouped ring pre-state carrying only the history that already exists.

    The ring never pre-fills future tokens.  For a boundary call that is the
    first ``ratio - 1`` tokens of the group; for a non-boundary call it is the
    prefix strictly before the current token.  The trailing rows hold a sentinel
    that the operator's in-place write must overwrite (boundary) or leave
    untouched (non-boundary).
    """
    rows, hidden = shared["ratio"], shared["hidden"]
    ring = np.zeros((BLOCKS, rows, hidden), dtype=np.float32)
    if kind == "nonboundary":
        # history is only the prefix before the current token (index ratio-2);
        # the current slot and the future row stay sentinels.
        ring[1, : rows - 2] = shared["tokens"][: rows - 2]
        ring[1, rows - 2:] = SENTINEL
    else:
        ring[1, : rows - 1] = shared["tokens"][: rows - 1]
        ring[1, rows - 1] = SENTINEL
    return ring


def _fold_prefix(ref, shared, count):
    """Pristine normal-path state: fold the first ``count`` group tokens in.

    The boundary / non-boundary call is a single S1 step, so the tokens before
    its ``start_pos`` are projected into the state here, outside the timing
    window, from the same logical history the grouped ring represents.  For
    coff=2 ``shared["state0"]`` already holds the real previous group, so the
    fold only appends the current group's prefix.
    """
    state = shared["state0"].copy()
    ratio, coff = shared["ratio"], shared["coff"]
    for k in range(count):
        _, _, _, state = ref.compress_golden(
            shared["tokens"][k: k + 1][None, :, :], shared["wkv"], shared["wgate"],
            state, shared["ape"], shared["block_table"], ratio, coff, 1,
            start_pos=[shared["window_start"] + k], seqused=[1])
    return state


def _normal_steps(shared, kind):
    ratio, window_start = shared["ratio"], shared["window_start"]
    if kind == "boundary":
        pos = window_start + ratio - 1
        return [{"x": shared["tokens"][ratio - 1: ratio][None, :, :],
                 "start_pos": pos, "slot": None}]
    if kind == "nonboundary":
        pos = window_start + ratio - 2
        return [{"x": shared["tokens"][ratio - 2: ratio - 1][None, :, :],
                 "start_pos": pos, "slot": None}]
    return [{"x": shared["tokens"][k: k + 1][None, :, :],
             "start_pos": window_start + k, "slot": None} for k in range(ratio)]


def _grouped_steps(shared, kind):
    ratio, rows, window_start = shared["ratio"], shared["ratio"], shared["window_start"]
    if kind == "nonboundary":
        pos = window_start + ratio - 2
        return [{"x": shared["tokens"][ratio - 2: ratio - 1][None, :, :],
                 "start_pos": pos, "slot": _slot(pos, rows)}]
    if kind == "boundary":
        pos = window_start + ratio - 1
        return [{"x": shared["tokens"][ratio - 1: ratio][None, :, :],
                 "start_pos": pos, "slot": _slot(pos, rows)}]
    return [{"x": shared["tokens"][k: k + 1][None, :, :],
             "start_pos": window_start + k, "slot": _slot(window_start + k, rows)}
            for k in range(ratio)]


def build_scenario(scn, seed):
    """Pure-numpy fixture for one scenario (shared arrays + path-specific steps)."""
    _, ref = cases._smoke()
    geom = GEOM_BY_KEY[scn["geom"]]
    shared = _shared_fixture(geom, seed)
    ratio, coff = shared["ratio"], shared["coff"]
    wkv, wgate, ape = shared["wkv"], shared["wgate"], shared["ape"]
    block_table = shared["block_table"]
    cmp_step = None

    if scn["path"] == "normal":
        steps = _normal_steps(shared, scn["kind"])
        if scn["kind"] == "cycle":
            pristine_state = shared["state0"].copy()
        else:
            count = ratio - 1 if scn["kind"] == "boundary" else ratio - 2
            pristine_state = _fold_prefix(ref, shared, count)
        ring0 = None
        expected_ring = None
        state = pristine_state
        for index, step in enumerate(steps):
            cmp_kv, _, _, state = ref.compress_golden(
                step["x"], wkv, wgate, state, ape, block_table, ratio, coff, 1,
                start_pos=[step["start_pos"]], seqused=[1])
            if scn["kind"] != "nonboundary":
                cmp_step = index
        expected_cmp = cmp_kv if scn["kind"] != "nonboundary" else None
        expected_state = state
    else:
        steps = _grouped_steps(shared, scn["kind"])
        ring0 = _ring_window(shared, scn["kind"]) if scn["kind"] != "cycle" \
            else np.zeros((BLOCKS, ratio, shared["hidden"]), dtype=np.float32)
        # The grouped path projects the whole window in one call, so its state
        # only holds the previous group (identical logical history, different
        # representation from the normal path's warmed prefix).
        pristine_state = shared["state0"].copy()
        state = pristine_state
        ring = ring0.copy()
        cmp_step = None
        expected_cmp = None
        for index, step in enumerate(steps):
            out = ref.grouped_case_golden(
                step["x"], wkv, wgate, state, ape, block_table, ring,
                np.array([step["slot"]], dtype=np.int32), ratio, coff, 1,
                start_pos=np.array([step["start_pos"]]), seqused=np.array([1]),
                x_cache_update=True)
            state = out["state_cache"]
            ring = out["x_cache"]
            if out["active"]:
                expected_cmp = out["cmp_kv"]
                cmp_step = index
        expected_state = state
        expected_ring = ring

    return {
        "scenario": scn["name"], "geom": scn["geom"], "path": scn["path"], "kind": scn["kind"],
        "ratio": ratio, "coff": coff, "head": shared["head"], "hidden": shared["hidden"],
        "window_start": shared["window_start"], "state0": pristine_state,
        "wkv": wkv, "wgate": wgate, "ape": ape, "block_table": block_table,
        "steps": steps, "ring0": ring0, "update": True, "cmp_step": cmp_step,
        "expected": {"cmp_kv": expected_cmp, "state": expected_state, "ring": expected_ring},
        "contract_golden": {"cmp_kv": shared["golden_cmp"], "state": shared["golden_state"]},
    }


# ---------------------------------------------------------------------------
# CPU self-check: the fixtures themselves follow the operator contract
# ---------------------------------------------------------------------------
def cpu_self_check(seed=20260927):
    """Verify each scenario fixture on the CPU against the contract golden.

    For a completed group both the normal fold and the grouped fold must equal
    the single-shot ``compress_golden`` projection of the same window; a
    non-boundary scenario must produce no group and must leave/reset the right
    side effects.  This runs on the *same* fixtures the perf run uses.
    """
    results = []
    for scn in SCENARIOS:
        fx = build_scenario(scn, seed)
        checks = []
        if scn["kind"] != "nonboundary":
            expected = fx["contract_golden"]["cmp_kv"]
            got = fx["expected"]["cmp_kv"]
            checks.append(("cmp_kv_matches_contract",
                           None if got is None else float(np.abs(got - expected).max())))
            checks.append(("state_matches_contract",
                           float(np.abs(fx["expected"]["state"]
                                        - fx["contract_golden"]["state"]).max())))
        else:
            checks.append(("no_group_completed", fx["expected"]["cmp_kv"] is None))
        ok = all(math.isfinite(v) and v < 1e-6 for _, v in checks if isinstance(v, float)) \
            and all(v is True for _, v in checks if isinstance(v, bool))
        results.append({"scenario": scn["name"], "ok": bool(ok),
                        "checks": [{"check": n, "value": v} for n, v in checks]})
    return {"pass": all(r["ok"] for r in results), "scenarios": results}


# ---------------------------------------------------------------------------
# Timing aggregation
# ---------------------------------------------------------------------------
def aggregate_samples(samples):
    values = [float(s) for s in samples]
    if not values:
        raise ValueError("no timing samples to aggregate")
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("timing samples must be finite and positive: %r" % values)
    return {
        "count": len(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": sum(values) / len(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


# ---------------------------------------------------------------------------
# Device execution
# ---------------------------------------------------------------------------
def _dev(torch, array, dtype, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype, device=device)


def _is_npu(device):
    return str(device).startswith("npu")


def _sync(torch, device):
    if _is_npu(device):
        torch.npu.synchronize()


def _npu_timer(torch):
    def timed(fn):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end)

    return timed


def _cpu_timer():
    import time

    def timed(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0

    return timed


def _to_np(torch, tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().to(torch.float32).numpy()


def _prepare(torch, scn, seed, device):
    fx = build_scenario(scn, seed)
    op_dtype = torch.bfloat16
    state = _dev(torch, fx["state0"], torch.float32, device)
    ring = None if fx["ring0"] is None else _dev(torch, fx["ring0"], op_dtype, device)
    steps = [{
        "x": _dev(torch, step["x"], op_dtype, device),
        "start_pos": _dev(torch, np.array([step["start_pos"]], np.int32), torch.int32, device),
        "slot": None if step["slot"] is None
        else _dev(torch, np.array([step["slot"]], np.int32), torch.int32, device),
    } for step in fx["steps"]]
    return {
        "scenario": scn["name"], "geom": scn["geom"], "path": scn["path"], "kind": scn["kind"],
        "ratio": fx["ratio"], "coff": fx["coff"], "head": fx["head"],
        "hidden": fx["hidden"], "seed": seed,
        "device": device,
        "wkv": _dev(torch, fx["wkv"], op_dtype, device),
        "wgate": _dev(torch, fx["wgate"], op_dtype, device),
        "ape": _dev(torch, fx["ape"], torch.float32, device),
        "block_table": _dev(torch, fx["block_table"], torch.int32, device),
        "seqused": _dev(torch, np.array([1], np.int32), torch.int32, device),
        "state": state, "ring": ring, "steps": steps, "update": fx["update"],
        "cmp_step": fx["cmp_step"], "expected": fx["expected"],
        "pristine": {"state": state.clone(), "ring": None if ring is None else ring.clone()},
    }


def reset_prepared(torch, prepared):
    pristine = prepared["pristine"]
    prepared["state"].copy_(pristine["state"])
    if prepared["ring"] is not None:
        prepared["ring"].copy_(pristine["ring"])


def _run_prepared(torch, compressor, prepared):
    outputs = []
    for step in prepared["steps"]:
        kwargs = dict(
            cmp_ratio=prepared["ratio"], state_block_table=prepared["block_table"],
            cu_seqlens=None, seqused=prepared["seqused"], start_pos=step["start_pos"],
            coff=prepared["coff"], cache_mode=1,
        )
        if prepared["path"] == "grouped":
            kwargs["x_cache"] = prepared["ring"]
            kwargs["x_cache_slot_mapping"] = step["slot"]
            kwargs["x_cache_update"] = prepared["update"]
        outputs.append(compressor(step["x"], prepared["wkv"], prepared["wgate"],
                                  prepared["state"], prepared["ape"], **kwargs))
    return outputs


STATE_RTOL = 0.0078125
STATE_ATOL = 1e-4


def _round_to_dtype(torch, array, dtype):
    src = torch.from_numpy(np.ascontiguousarray(np.asarray(array, dtype=np.float32)))
    return src.to(dtype).to(torch.float32).numpy()


def _compare_x_cache(torch, expected_np, ring_tensor):
    """Full element-wise exact check of the grouped ``x_cache`` side effect.

    The cache is a bf16 buffer on device, so the golden is first rounded to the
    buffer dtype (the operator can store nothing finer there) and the two fp32
    readbacks are then compared bit for bit.  This never averages over a
    percentage of the tensor: a single corrupted element, a whole token row
    that was never written, or a spurious write into a read-only cache fails.
    """
    got = _to_np(torch, ring_tensor)
    expected = _round_to_dtype(torch, expected_np, ring_tensor.dtype)
    if got is None or got.shape != expected.shape:
        return {"pass": False, "reason": "shape mismatch",
                "expect_shape": tuple(np.shape(expected)),
                "actual_shape": None if got is None else tuple(got.shape)}
    if not (np.isfinite(got).all() and np.isfinite(expected).all()):
        return {"pass": False, "reason": "non-finite value present"}
    mismatch = int((got.view(np.uint32) != expected.view(np.uint32)).sum())
    return {
        "pass": mismatch == 0,
        "mismatch_elements": mismatch,
        "total_elements": int(got.size),
        "max_abs": float(np.abs(got - expected).max()) if got.size else 0.0,
        "cache_dtype": str(ring_tensor.dtype),
        "comparison": "element-wise exact at cache dtype",
    }


def _compare_state_cache(expected_np, pristine_np, got_np):
    """Element-wise side-effect check of ``state_cache``.

    The golden decides which elements were updated: an element the golden left
    equal to the pristine state must still read back bit-equal to pristine (so a
    stray write, an overwritten row, or a damaged element fails), and an element
    the golden updated is compared element-wise with ``isclose`` (rtol/atol) so a
    partial or missing update cannot be averaged away by a percentage tolerance.
    """
    expected = np.asarray(expected_np, dtype=np.float32)
    pristine = np.asarray(pristine_np, dtype=np.float32)
    got = np.asarray(got_np, dtype=np.float32)
    if not (expected.shape == pristine.shape == got.shape):
        return {"pass": False, "reason": "shape mismatch",
                "expect_shape": tuple(expected.shape),
                "pristine_shape": tuple(pristine.shape),
                "actual_shape": tuple(got.shape)}
    if not (np.isfinite(expected).all() and np.isfinite(got).all()):
        return {"pass": False, "reason": "non-finite value present"}
    updated = expected != pristine
    untouched = ~updated
    untouched_mismatch = int((got[untouched] != pristine[untouched]).sum())
    updated_mismatch = 0
    if updated.any():
        close = np.isclose(got[updated], expected[updated],
                           rtol=STATE_RTOL, atol=STATE_ATOL)
        updated_mismatch = int((~close).sum())
    return {
        "pass": untouched_mismatch == 0 and updated_mismatch == 0,
        "updated_elements": int(updated.sum()),
        "untouched_mismatch": untouched_mismatch,
        "updated_mismatch": updated_mismatch,
        "max_abs": float(np.abs(got - expected).max()) if got.size else 0.0,
        "rtol": STATE_RTOL, "atol": STATE_ATOL,
        "comparison": "untouched region exact, updated region element-wise isclose",
    }


def verify_prepared(torch, compressor, prepared):
    """Run the prepared scenario once and compare readbacks to the golden.

    Side effects are checked element-wise, never with the percentage tolerance
    used for the output projection: ``x_cache`` must be exact at the cache dtype,
    and ``state_cache`` is split with the pristine state so untouched elements
    must be bit-equal to pristine and updated elements must each pass ``isclose``.
    A single wrong element or a missing token/state write therefore fails the
    scenario before any timing is recorded.
    """
    reset_prepared(torch, prepared)
    outputs = _run_prepared(torch, compressor, prepared)
    _sync(torch, prepared["device"])
    checks = []
    expected = prepared["expected"]
    if expected["cmp_kv"] is not None:
        got = _to_np(torch, outputs[prepared["cmp_step"]])
        checks.append(("cmp_kv", cases._compare(expected["cmp_kv"], got)))
    checks.append(("state_cache", _compare_state_cache(
        expected["state"], _to_np(torch, prepared["pristine"]["state"]),
        _to_np(torch, prepared["state"]))))
    if prepared["ring"] is not None:
        checks.append(("x_cache", _compare_x_cache(
            torch, expected["ring"], prepared["ring"])))
    return checks


def measure_scenario(torch, compressor, prepared, scn, warmup, repeats, timer=None):
    if timer is None:
        timer = _npu_timer(torch)
    device = prepared["device"]
    for _ in range(warmup):
        reset_prepared(torch, prepared)
        _sync(torch, device)
        _run_prepared(torch, compressor, prepared)
        _sync(torch, device)
    samples = []
    for _ in range(repeats):
        reset_prepared(torch, prepared)
        _sync(torch, device)
        samples.append(timer(lambda: _run_prepared(torch, compressor, prepared)))
        _sync(torch, device)
    stats = aggregate_samples(samples)
    stats.update({
        "scenario": scn["name"], "geom": scn["geom"], "kind": scn["kind"],
        "path": scn["path"], "ratio": scn["ratio"], "coff": scn["coff"],
        "head": scn["head"], "hidden": scn["hidden"], "seed": prepared["seed"],
        "steps_per_trial": len(prepared["steps"]), "note": scn.get("note", ""),
        "method": "NPU Event elapsed_time around the op sequence; ring/state reset and "
                  "H2D upload happen before start.record(); torch.npu.synchronize() per trial",
    })
    return stats


_SCOPE_KEYS = ("kind", "steps_per_trial", "ratio", "coff", "head", "hidden", "seed")


def _pair_speedups(scenarios, seed=None):
    by_key = {}
    for stat in scenarios:
        by_key.setdefault((stat["geom"], stat["kind"]), {})[stat["path"]] = stat
    pairs = []
    for (geom, kind), paths in sorted(by_key.items()):
        if "normal" not in paths or "grouped" not in paths:
            continue
        normal, grouped = paths["normal"], paths["grouped"]
        # Honesty guard: divide only equal-scope pairs.  A boundary pair is one
        # S1 call on each side, a cycle pair is ratio calls on each side; the
        # shape and seed must match too, not just the geom/kind label.
        for key in _SCOPE_KEYS:
            if normal.get(key) != grouped.get(key):
                raise AssertionError(
                    "normal/grouped scope mismatch for %s/%s on %r: %r vs %r"
                    % (geom, kind, key, normal.get(key), grouped.get(key)))
        if seed is not None and normal.get("seed") != seed:
            raise AssertionError("scenario %s built with seed %r, expected %r"
                                 % (normal.get("scenario"), normal.get("seed"), seed))
        pairs.append({
            "geom": geom, "kind": kind,
            "normal_median_ms": normal["median_ms"],
            "grouped_median_ms": grouped["median_ms"],
            "speedup_median": normal["median_ms"] / grouped["median_ms"],
            "normal_min_ms": normal["min_ms"], "grouped_min_ms": grouped["min_ms"],
            "speedup_min": normal["min_ms"] / grouped["min_ms"],
            "tokens_per_trial": normal.get("steps_per_trial"),
            "scope": "same kind, same calls/tokens, same shape and seed (normal vs grouped)",
        })
    return pairs


def _provider_note(torch):
    note = {"torch": torch.__version__}
    try:
        import cann_ops_transformer

        note["cann_ops_transformer"] = getattr(cann_ops_transformer, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - install dependent
        note["cann_ops_transformer"] = "unavailable: %r" % exc
    try:
        note["device_name"] = torch.npu.get_device_name(0)
    except Exception:  # pragma: no cover - device dependent
        note["device_name"] = "unknown"
    return note


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_accuracy(out_json, seed):
    report = {"contract": CONTRACT, "op": OP, "seed": seed,
              "cpu_self_check": cpu_self_check(seed), "scenarios": [], "npu": None}
    try:
        torch, compressor = cases._load_npu_runtime()
    except RuntimeError as exc:
        report["npu_error"] = str(exc)
        report["passed"] = False
        if out_json:
            Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    torch.npu.set_device(0)
    for scn in SCENARIOS:
        prepared = _prepare(torch, scn, seed, "npu:0")
        checks = verify_prepared(torch, compressor, prepared)
        report["scenarios"].append({
            "scenario": scn["name"], "ok": all(c[1]["pass"] for c in checks),
            "checks": [{"check": n, "metric": v} for n, v in checks]})
    report["passed"] = (report["cpu_self_check"]["pass"]
                        and all(s["ok"] for s in report["scenarios"]))
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_perf(seed, warmup, repeats, out_json):
    torch, compressor = cases._load_npu_runtime()
    torch.npu.set_device(0)
    device = "npu:0"
    timer = _npu_timer(torch)
    functional = []
    stats = []
    for scn in SCENARIOS:
        prepared = _prepare(torch, scn, seed, device)
        checks = verify_prepared(torch, compressor, prepared)
        ok = all(c[1]["pass"] for c in checks)
        functional.append({"scenario": scn["name"], "ok": ok,
                           "checks": [{"check": n, "metric": v} for n, v in checks]})
        if not ok:
            raise RuntimeError(
                "functional verification failed for %s; refusing to record a performance "
                "result for a broken scenario" % scn["name"])
        stats.append(measure_scenario(torch, compressor, prepared, scn, warmup, repeats, timer))
    report = {
        "contract": CONTRACT, "op": OP, "provider": _provider_note(torch),
        "dtype": "bfloat16", "seed": seed, "warmup": warmup, "repeats": repeats,
        "functional": functional, "scenarios": stats, "pairs": _pair_speedups(stats, seed),
    }
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _plan_matrix(seed, warmup, repeats):
    print("scenarios=%d pairs=%d seed=%d warmup=%d repeats=%d (B1/S1 per call)" % (
        len(SCENARIOS), len(_pair_speedups([_stub(s, seed) for s in SCENARIOS], seed)),
        seed, warmup, repeats))
    for scn in SCENARIOS:
        print("  %-24s geom=%-4s path=%-7s kind=%-11s R=%-4d coff=%d steps=%d"
              % (scn["name"], scn["geom"], scn["path"], scn["kind"], scn["ratio"],
                 scn["coff"], scn["steps"]))


def _stub(scn, seed):
    return {"geom": scn["geom"], "kind": scn["kind"], "path": scn["path"],
            "median_ms": 1.0, "min_ms": 1.0, "scenario": scn["name"],
            "ratio": scn["ratio"], "coff": scn["coff"], "head": scn["head"],
            "hidden": scn["hidden"], "steps_per_trial": scn["steps"], "seed": seed}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pro0921 Compressor accuracy/perf runner")
    parser.add_argument("--mode", choices=["plan", "accuracy", "perf", "all"], default="plan")
    parser.add_argument("--accuracy-out", type=Path, default=None)
    parser.add_argument("--perf-out", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--batch", type=int, default=1, help="must be 1 (only B1 is supported)")
    parser.add_argument("--seq", type=int, default=1, help="must be 1 (S1 per op call)")
    args = parser.parse_args(argv)

    if args.batch != 1 or args.seq != 1:
        print("PRO0921_ACCURACY=FAIL unsupported batch/seq: only B1/S1 is supported "
              "(got batch=%d seq=%d)" % (args.batch, args.seq))
        return 1

    if args.mode == "plan":
        _plan_matrix(args.seed, args.warmup, args.repeats)
        print("--mode plan touches no device and produces no timing value")
        return 0

    rc = 0
    if args.mode in ("accuracy", "all"):
        try:
            report = run_accuracy(args.accuracy_out, args.seed)
        except RuntimeError as exc:
            print("PRO0921_ACCURACY=FAIL %s" % exc)
            return 2
        print("PRO0921_ACCURACY=%s scenarios=%d" % (
            "PASS" if report["passed"] else "FAIL", len(report["scenarios"])))
        rc = rc or (0 if report["passed"] else 1)
    if args.mode in ("perf", "all"):
        try:
            report = run_perf(args.seed, args.warmup, args.repeats, args.perf_out)
        except RuntimeError as exc:
            print("PRO0921_PERF=FAIL %s" % exc)
            return 2
        for stat in report["scenarios"]:
            print("  %-24s median=%.4f ms min=%.4f ms n=%d"
                  % (stat["scenario"], stat["median_ms"], stat["min_ms"], stat["count"]))
        for pair in report["pairs"]:
            print("  pair %-4s %-11s speedup(median)=%.2fx" % (
                pair["geom"], pair["kind"], pair["speedup_median"]))
        print("PRO0921_PERF=OK scenarios=%d pairs=%d" % (
            len(report["scenarios"]), len(report["pairs"])))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Executed contract checks for the built cases. Pure Python/numpy: no compile,
# no device. It re-derives the operator's documented Vec1 contract with plain
# numpy, independently of latent_ref's grouping loop, so a golden bug cannot
# pass unnoticed.
#
# Per compressor case it checks:
#   * cmp_kv == sum(softmax(window @ Wgate^T + ape) * (window @ Wkv^T)) over the
#     group axis (independent recompute);
#   * softmax_score columns sum to 1 (grad_enabled=true cases);
#   * kv == the raw window projection (grad_enabled=true cases);
#   * state_cache: window rows written, previous group's rows untouched
#     (coff=2), unused block rows still zero;
#   * update-only: state_cache byte-identical to the fixture;
#   * the ring-write discrimination delta stays well above tolerance.
#
# Run:
#   python3 golden_contract_check.py --cases /tmp/pro0921_latent_cases

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

import latent_ref as ref


def _load(case_dir, name):
    path = case_dir / (name + ".npy")
    return np.load(path) if path.exists() else None


def _attributes(case_dir):
    row = next(csv.DictReader((case_dir / "case.csv").open(encoding="utf-8")))
    return ast.literal_eval(row["attributes"])


def _window(case_dir, x, x_cache, slots, ratio, update):
    rows = x_cache.shape[1]
    bank = int(slots[0]) // rows
    start_pos = int(_load(case_dir, "start_pos").reshape(-1)[0])
    active, logical, window_start, _ = ref.derive_grouped_window(start_pos, 1, ratio, ratio)
    if not active:
        return None, None
    row_idx = [(window_start + k) % rows for k in range(logical)]
    window = x_cache[bank, row_idx].copy()
    if update and int(slots[0]) >= 0:
        # UpdateXCache stores this step's token in the mapped ring row; the
        # mapped row is always the last row of the completed window here.
        window[-1] = ref.to_bf16(x[0, 0])
    return window, logical


def check_compressor(case_dir, name, checks):
    golden_json = json.loads((case_dir / "golden.json").read_text(encoding="utf-8"))
    goldens = {n: (None if f is None else np.load(case_dir / f))
               for n, f in zip(golden_json["output_names"], golden_json["files"])}
    attrs = _attributes(case_dir)
    ratio, coff = int(attrs["cmp_ratio"]), int(attrs["coff"])
    update = bool(attrs.get("x_cache_update", True))

    x = _load(case_dir, "x").astype(np.float32)
    wkv = _load(case_dir, "wkv").astype(np.float32)
    wgate = _load(case_dir, "wgate").astype(np.float32)
    ape = _load(case_dir, "ape").astype(np.float32)
    x_cache = _load(case_dir, "x_cache").astype(np.float32)
    slots = _load(case_dir, "x_cache_slot_mapping").reshape(-1)
    state_in = _load(case_dir, "state_cache").astype(np.float32)

    window, logical = _window(case_dir, x, x_cache, slots, ratio, update)
    if window is None:
        checks.append((name, "update_only_state_unchanged",
                       np.array_equal(goldens["state_cache"], state_in), 0.0))
        return

    head_dim = wkv.shape[0] // coff
    kv_full = window @ wkv.T
    score_full = window @ wgate.T + ape[:logical]
    if coff == 1:
        sc_kv, sc_score = kv_full, score_full
    else:
        # coff=2 overlap: slice 0 of state_cache holds the previous group's kv_a
        # / score_a, the current window contributes kv_b / score_b (slice 1).
        start_pos = int(_load(case_dir, "start_pos").reshape(-1)[0])
        _, _, window_start, _ = ref.derive_grouped_window(start_pos, 1, ratio, ratio)
        prev_rows = state_in[1, window_start - ratio:window_start]
        prev_kv = prev_rows[:, 0:head_dim]
        prev_score = prev_rows[:, coff * head_dim: coff * head_dim + head_dim]
        sc_kv = np.concatenate([prev_kv, kv_full[:, head_dim:2 * head_dim]], axis=0)
        sc_score = np.concatenate([prev_score, score_full[:, head_dim:2 * head_dim]], axis=0)
        assert window_start == ratio, "checker assumes the second group boundary"
    weights = ref.softmax_columns(sc_score)
    independent = (weights * sc_kv).sum(axis=0, keepdims=True)[None, :, :]
    delta = float(np.abs(independent - goldens["cmp_kv"]).max())
    checks.append((name, "cmp_kv_independent_recompute", delta < 1e-5, delta))

    if goldens["softmax_score"] is not None:
        col_sums = goldens["softmax_score"].sum(axis=2)
        dev = float(np.abs(col_sums - 1.0).max())
        checks.append((name, "softmax_columns_sum_to_one", dev < 1e-4, dev))
    if goldens["kv"] is not None:
        delta = float(np.abs(goldens["kv"][0, 0] - sc_kv).max())
        checks.append((name, "kv_output_is_raw_projection", delta < 1e-5, delta))

    state_out = goldens["state_cache"]
    if coff == 2:
        checks.append((name, "coff2_previous_group_rows_untouched",
                       np.array_equal(state_out[1, :ratio], state_in[1, :ratio]), 0.0))
    checks.append((name, "state_window_rows_written",
                   not np.array_equal(state_out[1], state_in[1]), 0.0))
    checks.append((name, "unused_block_stays_zero",
                   float(np.abs(state_out[0]).max()) == 0.0,
                   float(np.abs(state_out[0]).max())))


def check_quant(case_dir, name, checks):
    """Independent recompute of the quant golden: decode codes, matmul, then
    multiply by (x_descale * wkv_descale) as a separate step (the reference
    does not fold the descale into the weights)."""
    goldens = {n: (None if f is None else np.load(case_dir / f))
               for n, f in zip(
                   json.loads((case_dir / "golden.json").read_text(encoding="utf-8"))["output_names"],
                   json.loads((case_dir / "golden.json").read_text(encoding="utf-8"))["files"])}
    attrs = _attributes(case_dir)
    ratio = int(attrs["cmp_ratio"])
    codec = ref.hif8_utils()
    x_cache = np.load(case_dir / "x_cache.npy")
    rows = x_cache.shape[1]
    slots = np.load(case_dir / "x_cache_slot_mapping.npy").reshape(-1)
    update = bool(attrs.get("x_cache_update", True))
    if update and int(slots[0]) >= 0:
        x_cache = x_cache.copy()
        x_cache[int(slots[0]) // rows, int(slots[0]) % rows] = np.load(case_dir / "x.npy").reshape(-1, x_cache.shape[-1])[0]
    bank = int(slots[0]) // rows
    window = x_cache[bank]
    x_fp32 = codec.hif8_to_fp32_lut(window)
    wkv_fp32 = codec.hif8_to_fp32_lut(np.load(case_dir / "wkv.npy"))
    wgate_fp32 = codec.hif8_to_fp32_lut(np.load(case_dir / "wgate.npy"))
    ape = np.load(case_dir / "ape.npy").astype(np.float32)
    x_descale = np.load(case_dir / "x_descale.npy").astype(np.float32).reshape(-1)
    wkv_descale = np.load(case_dir / "wkv_descale.npy").astype(np.float32).reshape(-1)
    wgate_descale = np.load(case_dir / "wgate_descale.npy").astype(np.float32).reshape(-1)
    kv = np.matmul(x_fp32, wkv_fp32.T, dtype=np.float32) * (x_descale * wkv_descale)
    score = np.matmul(x_fp32, wgate_fp32.T, dtype=np.float32) * (x_descale * wgate_descale)
    score = score + ape
    weights = ref.softmax_columns(score)
    independent = (weights * kv).sum(axis=0, keepdims=True)[None, :, :]
    delta = float(np.abs(independent - goldens["cmp_kv"]).max())
    checks.append((name, "quant_cmp_kv_independent_recompute", delta < 1e-5, delta))
    decoded = codec.hif8_to_fp32_lut(np.load(case_dir / "x.npy"))
    rel = np.abs(codec.hif8_to_fp32_lut(codec.fp32_to_hif8_numpy(decoded)) - decoded)
    checks.append((name, "quant_hif8_roundtrip_note", True, float(rel.max())))


def check_ring_discrimination(case_dir, name, checks):
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    if "ring_write_delta_rel" not in expected:
        return
    rel = expected["ring_write_delta_rel"]
    checks.append((name, "ring_write_discrimination_rel", rel > 0.05, rel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("/tmp/pro0921_latent_cases"))
    args = parser.parse_args()
    manifest = json.loads((args.cases / "cases.json").read_text(encoding="utf-8"))
    checks = []
    for entry in manifest:
        case_dir = Path(entry["dir"])
        check_ring_discrimination(case_dir, entry["name"], checks)
        check_compressor(case_dir, entry["name"], checks)
    quant_manifest = args.cases / "quant_cases.json"
    if quant_manifest.exists():
        for entry in json.loads(quant_manifest.read_text(encoding="utf-8")):
            check_quant(Path(entry["dir"]), entry["name"], checks)
    failures = [(n, c, v) for n, c, ok, v in checks if not ok]
    for name, check, value in failures:
        print("FAIL %-26s %-40s value=%r" % (name, check, value))
    if failures:
        print("PRO0921_GOLDEN_CONTRACT=FAIL failures=%d checks=%d"
              % (len(failures), len(checks)))
        return 1
    print("PRO0921_GOLDEN_CONTRACT=PASS checks=%d" % len(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

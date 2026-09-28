# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Small legal QuantCompressor case + independent golden.
#
# The QuantCompressor ABI differs from Compressor in three ways that the old
# plugin ignored:
#   * three descale inputs (x_descale, wkv_descale, wgate_descale) sit between
#     ape and state_block_table;
#   * x / wkv / wgate / x_cache are HIFLOAT8 (a uint8 byte carrier);
#   * only two outputs are declared (cmp_kv, state_cache).
# A B1 S1 hidden1024 head128 cmp_ratio128 fixture is built here, with the
# HiFloat8 conversion done by the operator's own reference codec
# (quant_compressor/tests/pytest/hif8_utils.py): the golden decodes the exact
# codes the kernel consumes instead of substituting a BF16 path.
#
# Run:
#   python3 build_quant_cases.py --out /tmp/pro0921_latent_cases

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import latent_ref as ref


HIDDEN = 1024
HEAD = 128
RATIO = 128
BLOCK_SIZE = 128
BLOCKS = 2

CSV_COLUMNS = [
    "testcase_name", "network_name", "op_name", "input_shapes", "input_dtypes",
    "input_formats", "output_shapes", "output_dtypes", "output_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision", "output_inplace_indexes", "is_enabled", "remark",
]

QUANT_INPUTS = (
    "x", "wkv", "wgate", "state_cache", "ape",
    "x_descale", "wkv_descale", "wgate_descale",
    "state_block_table", "cu_seqlens", "seqused", "start_pos",
    "x_cache", "x_cache_slot_mapping",
)
OUTPUT_NAMES = ("cmp_kv", "state_cache")
INPLACE = (None, 3)

QUANT_INPUT_DTYPES = {
    "x": "hifloat8",
    "wkv": "hifloat8",
    "wgate": "hifloat8",
    "state_cache": "float32",
    "ape": "float32",
    "x_descale": "float32",
    "wkv_descale": "float32",
    "wgate_descale": "float32",
    "state_block_table": "int32",
    "cu_seqlens": "int32",
    "seqused": "int32",
    "start_pos": "int32",
    "x_cache": "hifloat8",
    "x_cache_slot_mapping": "int32",
}

TOLERANCE = (0.005, 0.005)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/pro0921_latent_cases"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260928)
    # HiFloat8 codes (uint8 carrier) for the window, the token and the weights.
    window_values = rng.standard_normal((RATIO, HIDDEN)).astype(np.float32) * 0.02
    token_values = window_values[RATIO - 1: RATIO]
    x_codes = ref.encode_hif8(token_values)
    sentinel_codes = ref.encode_hif8(np.array([[0.1]], dtype=np.float32))
    window_codes = ref.encode_hif8(window_values)
    x_cache = np.zeros((BLOCKS, RATIO, HIDDEN), dtype=np.uint8)
    x_cache[1, : RATIO - 1] = window_codes[: RATIO - 1]
    x_cache[1, RATIO - 1] = sentinel_codes[0, 0]
    wkv_codes = ref.encode_hif8(rng.standard_normal((HEAD, HIDDEN)).astype(np.float32) * 0.02)
    wgate_codes = ref.encode_hif8(rng.standard_normal((HEAD, HIDDEN)).astype(np.float32) * 0.02)
    x_descale = np.array([0.5], dtype=np.float32)
    wkv_descale = (0.6 + 0.2 * rng.random(HEAD)).astype(np.float32)
    wgate_descale = (0.6 + 0.2 * rng.random(HEAD)).astype(np.float32)
    ape = (rng.standard_normal((RATIO, HEAD)) * 0.01).astype(np.float32)
    state = np.zeros((BLOCKS, BLOCK_SIZE, 2 * HEAD), dtype=np.float32)
    slot = 1 * RATIO + (RATIO - 1)
    block_table = np.array([[1]], dtype=np.int32)

    golden = ref.quant_grouped_case_golden(
        x_codes, wkv_codes, wgate_codes, x_descale, wkv_descale, wgate_descale,
        state, ape, block_table, x_cache, slot, RATIO, 1,
        start_pos=np.array([RATIO - 1]), seqused=np.array([1]))
    assert golden["active"] and golden["window_start"] == 0
    cmp_kv = golden["cmp_kv"]
    assert cmp_kv.shape == (1, 1, HEAD), cmp_kv.shape

    # Decode round-trip sanity: the codes the golden decodes must match the
    # codes stored in the fixture (guards against an accidental float path).
    decoded = ref.decode_hif8(window_codes)
    assert np.array_equal(ref.encode_hif8(decoded), window_codes), \
        "hif8 decode/encode round trip is not stable for the fixture"

    name = "quant_b1_boundary"
    dest = args.out / name
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": x_codes.reshape(1, 1, HIDDEN),
        "wkv": wkv_codes,
        "wgate": wgate_codes,
        "state_cache": state,
        "ape": ape,
        "x_descale": x_descale,
        "wkv_descale": wkv_descale,
        "wgate_descale": wgate_descale,
        "state_block_table": block_table,
        "cu_seqlens": None,
        "seqused": np.array([1], dtype=np.int32),
        "start_pos": np.array([RATIO - 1], dtype=np.int32),
        "x_cache": x_cache,
        "x_cache_slot_mapping": np.array([slot], dtype=np.int32),
    }
    for key, value in payload.items():
        if value is not None:
            np.save(dest / (key + ".npy"), value)
    np.save(dest / "golden_0.npy", np.asarray(cmp_kv, dtype=np.float32))
    np.save(dest / "golden_1.npy", np.asarray(golden["state_cache"], dtype=np.float32))
    (dest / "golden.json").write_text(json.dumps({
        "files": ["golden_0.npy", "golden_1.npy"],
        "dtypes": ["float32", "float32"],
        "output_names": list(OUTPUT_NAMES),
    }, indent=2), encoding="utf-8")
    expected = {
        "checked_outputs": ["cmp_kv", "state_cache"],
        "suppressed_outputs": [],
        "hif8_carrier": ("x/wkv/wgate/x_cache are hifloat8 uint8 codes encoded and decoded by "
                         "quant_compressor/tests/pytest/hif8_utils.py"),
        "ring_write_proof": "cmp_kv depends on the in-place ring write of the current token",
        "note": ("quant_mode=1 (A8W8 A-per-tensor / W-per-channel); descale inputs are declared "
                 "in the quant ABI order after ape; cmp_kv and state_cache are compared"),
    }
    (dest / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (dest / "readback_spec.json").write_text(json.dumps(
        [{"index": QUANT_INPUTS.index("x_cache"), "name": "x_cache",
          "dtype": "hifloat8", "shape": [int(v) for v in x_cache.shape]}], indent=2),
        encoding="utf-8")

    shapes, dtypes, formats = [], [], []
    for key in QUANT_INPUTS:
        value = payload[key]
        if value is None:
            shapes.append(None)
            dtypes.append(None)
            formats.append(None)
            continue
        shapes.append(tuple(int(v) for v in value.shape))
        dtypes.append(QUANT_INPUT_DTYPES[key])
        formats.append("ND")
    attributes = {
        "quant_mode": 1, "cmp_ratio": RATIO, "coff": 1, "cache_mode": 1,
        "state_cache_stride_dim0": BLOCK_SIZE * (2 * HEAD), "x_cache_update": True,
    }
    for key in ("state_block_table", "seqused", "start_pos", "x_cache_slot_mapping"):
        attributes[key] = payload[key].tolist()
    row = {
        "testcase_name": name,
        "network_name": "deepseek_v4",
        "op_name": "quant_compressor",
        "input_shapes": repr(tuple(shapes)),
        "input_dtypes": repr(tuple(dtypes)),
        "input_formats": repr(tuple(formats)),
        "output_shapes": repr(((1, 1, HEAD), tuple(int(v) for v in state.shape))),
        "output_dtypes": repr(("bfloat16", "float32")),
        "output_formats": repr(("ND", "ND")),
        "attributes": repr(attributes),
        "input_data_ranges": repr(tuple((None if s is None else (-0.1, 0.1)) for s in shapes)),
        "precision_tolerances": repr((TOLERANCE, TOLERANCE)),
        "absolute_precision": 0.001,
        "output_inplace_indexes": repr(INPLACE),
        "is_enabled": True,
        "remark": "Quant B1 S1 hidden1024 head128 R128 boundary; hif8 codes + descale; independent CPU golden",
    }
    with (dest / "case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    manifest = [{
        "name": name, "dir": str(dest), "kind": "quant_compressor",
        "cmp_ratio": RATIO, "coff": 1, "x_cache_update": True,
        "expected": expected,
    }]
    (args.out / "quant_cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("built %s ratio=%d coff=1 -> %s" % (name, RATIO, dest))
    print("cases under %s" % args.out)


if __name__ == "__main__":
    main()

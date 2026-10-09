# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Build the smallest legal formal-compressor cases plus an independent CPU
# golden. Pure Python/numpy; no native compile, no simulation.
#
# The formal Compressor op-def declares FOUR outputs in this order:
#   cmp_kv (bf16) | state_cache (fp32) | softmax_score (fp32) | kv (fp32)
# and the state_cache output overwrites input index 3 (in-place).  Every case
# here declares all four so the CSV/plugin match the operator definition.
#
# softmax_score / kv are only *valid* when grad_enabled=true (see
# compressor/README.md and the `if constexpr (COMP::gradEnabled ==
# GRAD_ENABLED::ENABLE)` guard around CopyOutMidResToOutput).  For
# grad_enabled=false cases they are still declared with the operator's own
# infer shape, but their golden is None so TTK reports them SUPPRESSED instead
# of comparing an output the op never writes.  `hca_b1_boundary_grad` runs the
# same fixture with grad_enabled=true and does check all four outputs.
#
# Case order matters: the first case is the single HCA boundary probe, so the
# first native run confirms the grouped path end to end before the schedule
# variants are added.
#
# Run:
#   python3 build_tiny_cases.py --out /tmp/pro0921_latent_cases
#   python3 build_tiny_cases.py --out /tmp/pro0921_latent_cases --first-only

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import latent_ref as ref


HIDDEN = 1024
HEAD = 512          # CSA KV compressor head dim
HCA_HEAD = 128      # minimum legal HCA smoke head dim
LI_HEAD = 128       # same-layer LI compressor head dim
BLOCK_SIZE = 128
BLOCKS = 2          # 1 null block + 1 request-owned block

CSV_COLUMNS = [
    "testcase_name", "network_name", "op_name", "input_shapes", "input_dtypes",
    "input_formats", "output_shapes", "output_dtypes", "output_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision", "output_inplace_indexes", "is_enabled", "remark",
]

COMPRESSOR_INPUTS = (
    "x", "wkv", "wgate", "state_cache", "ape", "state_block_table",
    "cu_seqlens", "seqused", "start_pos", "x_cache", "x_cache_slot_mapping",
)
OUTPUT_NAMES = ("cmp_kv", "state_cache", "softmax_score", "kv")

# Output index 1 (state_cache) overwrites input index 3 (the state_cache input).
INPLACE = (None, 3, None, None)

INPUT_DTYPES = {
    "x": "bfloat16",
    "wkv": "bfloat16",
    "wgate": "bfloat16",
    "state_cache": "float32",
    "ape": "float32",
    "state_block_table": "int32",
    "cu_seqlens": "int32",
    "seqused": "int32",
    "start_pos": "int32",
    "x_cache": "bfloat16",
    "x_cache_slot_mapping": "int32",
}

TOLERANCE = (0.005, 0.005)


def state_shape(coff, head_dim):
    return (BLOCKS, BLOCK_SIZE, 2 * coff * head_dim)


def _tokens(rng, count, hidden):
    return ref.to_bf16(rng.standard_normal((count, hidden)).astype(np.float32) * 0.02)


def _weights(rng, rows, hidden):
    return ref.to_bf16(rng.standard_normal((rows, hidden)).astype(np.float32) * 0.02)


def _window_from(x_cache, bank, row, row_value):
    window = x_cache[bank].copy()
    window[row] = row_value
    return window[None, :, :]


def ring_write_discrimination(case):
    """Prove the cmp_kv golden depends on the in-place ring write.

    `hca_b1_boundary` leaves a sentinel in the ring row the operator must
    overwrite.  If the op skipped the write, cmp_kv would be computed from the
    sentinel instead of the token.  Build both variants and return the delta so
    a precision PASS on the executed case is real evidence of the write plus
    read order (this is an executed numeric check, not a boolean flag).
    """
    payload = case["payload"]
    ratio = case["attrs"]["cmp_ratio"]
    common = dict(wkv=payload["wkv"], wgate=payload["wgate"],
                  state_cache=payload["state_cache"], ape=payload["ape"],
                  block_table=payload["state_block_table"], cmp_ratio=ratio,
                  coff=case["attrs"]["coff"], cache_mode=case["attrs"]["cache_mode"])
    token = payload["x"][0, 0]
    sentinel = payload["_sentinel"]
    with_write, _, _, _ = ref.compress_golden(
        _window_from(payload["x_cache"], 1, ratio - 1, token),
        start_pos=[0], seqused=[ratio], **common)
    without_write, _, _, _ = ref.compress_golden(
        _window_from(payload["x_cache"], 1, ratio - 1, sentinel),
        start_pos=[0], seqused=[ratio], **common)
    delta = float(np.abs(with_write - without_write).max())
    scale = float(np.abs(with_write).max()) or 1.0
    return delta, scale


def build_hca_boundary(rng, grad_enabled):
    ratio = 128
    sentinel = np.float32(0.1)
    window_tokens = _tokens(rng, ratio, HIDDEN)
    x_cache = np.zeros((BLOCKS, ratio, HIDDEN), dtype=np.float32)
    x_cache[1, : ratio - 1] = window_tokens[: ratio - 1]
    x_cache[1, ratio - 1] = sentinel            # the op must overwrite this row first
    token = window_tokens[ratio - 1: ratio]
    wkv = _weights(rng, HCA_HEAD, HIDDEN)
    wgate = _weights(rng, HCA_HEAD, HIDDEN)
    ape = (rng.standard_normal((ratio, HCA_HEAD)) * 0.01).astype(np.float32)
    state = np.zeros(state_shape(1, HCA_HEAD), dtype=np.float32)
    slot = 1 * ratio + (ratio - 1)
    golden = ref.grouped_case_golden(
        token[None, :, :], wkv, wgate, state, ape, np.array([[1]]), x_cache,
        np.array([slot], dtype=np.int32), ratio, 1, 1,
        start_pos=np.array([ratio - 1]), seqused=np.array([1]),
        grad_enabled=grad_enabled)
    assert golden["active"] and golden["window_start"] == 0
    softmax = golden["softmax"] if grad_enabled else None
    kv = golden["kv"] if grad_enabled else None
    name = "hca_b1_boundary_grad" if grad_enabled else "hca_b1_boundary"
    return {
        "name": name,
        "kind": "compressor",
        "payload": {
            "x": token[None, :, :],
            "wkv": wkv,
            "wgate": wgate,
            "state_cache": state,
            "ape": ape,
            "state_block_table": np.array([[1]], dtype=np.int32),
            "cu_seqlens": None,
            "seqused": np.array([1], dtype=np.int32),
            "start_pos": np.array([ratio - 1], dtype=np.int32),
            "x_cache": x_cache,
            "x_cache_slot_mapping": np.array([slot], dtype=np.int32),
            "_sentinel": sentinel,
        },
        "outputs": (golden["cmp_kv"], golden["state_cache"], softmax, kv),
        "attrs": {"cmp_ratio": ratio, "coff": 1, "cache_mode": 1,
                  "state_cache_stride_dim0": BLOCK_SIZE * (2 * 1 * HCA_HEAD),
                  "grad_enabled": grad_enabled, "x_cache_update": True},
        "expected": {
            "checked_outputs": list(OUTPUT_NAMES) if grad_enabled
            else ["cmp_kv", "state_cache"],
            "suppressed_outputs": [] if grad_enabled
            else ["softmax_score", "kv"],
            "ring_write_proof": "cmp_kv depends on the in-place ring write of the current token",
            "note": ("grad_enabled=true: all four outputs are written by the op and compared"
                     if grad_enabled else
                     "grad_enabled=false: softmax_score/kv are declared with the op infer "
                     "shape but never written, so their golden is None (SUPPRESSED)"),
        },
        "remark": ("HCA B1 S1 hidden1024 head128 R128 boundary; independent CPU golden"
                   + ("; grad_enabled=true" if grad_enabled else "")),
        "readback": ["x_cache"],
    }


def build_hca_update_only(rng):
    ratio = 128
    sentinel = np.float32(0.1)
    x_cache = np.zeros((BLOCKS, ratio, HIDDEN), dtype=np.float32)
    x_cache[1, : ratio - 1] = _tokens(rng, ratio - 1, HIDDEN)
    x_cache[1, ratio - 1] = sentinel
    token = _tokens(rng, 1, HIDDEN)
    wkv = _weights(rng, HCA_HEAD, HIDDEN)
    wgate = _weights(rng, HCA_HEAD, HIDDEN)
    ape = (rng.standard_normal((ratio, HCA_HEAD)) * 0.01).astype(np.float32)
    state = np.zeros(state_shape(1, HCA_HEAD), dtype=np.float32)
    start_pos = 64          # 128 - 64 = 64 > seqused(1) -> no group completes
    slot = 1 * ratio + start_pos
    golden = ref.grouped_case_golden(
        token[None, :, :], wkv, wgate, state, ape, np.array([[1]]), x_cache,
        np.array([slot], dtype=np.int32), ratio, 1, 1,
        start_pos=np.array([start_pos]), seqused=np.array([1]))
    assert golden["active"] is False, "update-only fixture must not complete a group"
    return {
        "name": "hca_b1_update_only",
        "kind": "compressor",
        "payload": {
            "x": token[None, :, :],
            "wkv": wkv,
            "wgate": wgate,
            "state_cache": state,
            "ape": ape,
            "state_block_table": np.array([[1]], dtype=np.int32),
            "cu_seqlens": None,
            "seqused": np.array([1], dtype=np.int32),
            "start_pos": np.array([start_pos], dtype=np.int32),
            "x_cache": x_cache,
            "x_cache_slot_mapping": np.array([slot], dtype=np.int32),
            "_sentinel": sentinel,
        },
        "outputs": (None, golden["state_cache"], None, None),
        "attrs": {"cmp_ratio": ratio, "coff": 1, "cache_mode": 1,
                  "state_cache_stride_dim0": BLOCK_SIZE * (2 * 1 * HCA_HEAD),
                  "grad_enabled": False, "x_cache_update": True},
        "expected": {
            "checked_outputs": ["state_cache"],
            "suppressed_outputs": ["cmp_kv", "softmax_score", "kv"],
            "ring_write_proof": ("verified by the optional x_cache read-back hook "
                                 "(PRO0921_READBACK_XCACHE=1); cmp_kv is not written so the "
                                 "proof cannot be taken from this case's outputs"),
            "state_cache_unchanged": True,
            "note": ("no group completes: the grouped path must exit without a mismatched "
                     "cross-core barrier and leave state_cache untouched; state_cache is the "
                     "checked output, cmp_kv/softmax_score/kv are not written (SUPPRESSED)"),
        },
        "remark": "HCA B1 S1 update-only (no boundary); barrier/exit + untouched-state check",
        "readback": ["x_cache"],
    }


def build_csa_coff2(rng, li):
    """coff=2 CSA / LI grouped step (previous group state + current group).

    The step sits on the second compression boundary (absolute position
    2*cmp_ratio-1).  The ring holds cmp_ratio rows, so the previous group is no
    longer resident and must come from state_cache; the current group's rows sit
    in the ring rows [0, cmp_ratio) because windowStart % ring_size == 0.

    The fixture is built from the previous group so the state rows
    [windowStart-ratio, windowStart) really hold that group's kv_a/kv_b -- the
    "previous group state + current group" golden the kernel's Vec1 ops encode.
    """
    head_dim = LI_HEAD if li else HEAD
    ratio = 4
    wkv = _weights(rng, head_dim * 2, HIDDEN)
    wgate = _weights(rng, head_dim * 2, HIDDEN)
    ape = (rng.standard_normal((ratio, 2 * head_dim)) * 0.01).astype(np.float32)
    state = np.zeros(state_shape(2, head_dim), dtype=np.float32)

    prev_tokens = _tokens(rng, ratio, HIDDEN)
    _, _, _, state = ref.compress_golden(
        prev_tokens[None, :, :], wkv, wgate, state, ape, np.array([[1]]), ratio, 2, 1,
        start_pos=[0], seqused=[ratio])

    cur_tokens = _tokens(rng, ratio, HIDDEN)
    ring_rows = ratio
    x_cache = np.zeros((BLOCKS, ring_rows, HIDDEN), dtype=np.float32)
    x_cache[1] = cur_tokens
    sentinel = np.float32(0.1)
    token = cur_tokens[ratio - 1: ratio]
    if li:
        # read-only consumer: the same-layer KV compressor already refreshed the
        # shared ring, so the current token is resident and this call must not
        # rewrite it.
        x_cache[1, ratio - 1] = token[0]
    else:
        # refresh path: leave a sentinel the op must overwrite before reading.
        x_cache[1, ratio - 1] = sentinel
    start_pos = 2 * ratio - 1     # completes the group [ratio, 2*ratio)
    slot = 1 * ring_rows + (ratio - 1)

    golden, _, _, golden_state = ref.compress_golden(
        _window_from(x_cache, 1, ratio - 1, token[0]),
        wkv, wgate, state, ape, np.array([[1]]), ratio, 2, 1,
        start_pos=[ratio], seqused=[ratio])
    assert np.array_equal(golden_state[1, :ratio], state[1, :ratio]), \
        "the current step must leave the previous group's state rows intact"
    assert golden.shape == (1, 1, head_dim), golden.shape

    name = "csa_li_read_shared" if li else "csa_b1_coff2_boundary"
    return {
        "name": name,
        "kind": "compressor",
        "payload": {
            "x": token[None, :, :],
            "wkv": wkv,
            "wgate": wgate,
            "state_cache": state,
            "ape": ape,
            "state_block_table": np.array([[1]], dtype=np.int32),
            "cu_seqlens": None,
            "seqused": np.array([1], dtype=np.int32),
            "start_pos": np.array([start_pos], dtype=np.int32),
            "x_cache": x_cache,
            "x_cache_slot_mapping": np.array([slot], dtype=np.int32),
            "_sentinel": sentinel,
        },
        "outputs": (golden, golden_state, None, None),
        "attrs": {"cmp_ratio": ratio, "coff": 2, "cache_mode": 1,
                  "state_cache_stride_dim0": BLOCK_SIZE * (2 * 2 * head_dim),
                  "grad_enabled": False, "x_cache_update": not li},
        "expected": {
            "checked_outputs": ["cmp_kv", "state_cache"],
            "suppressed_outputs": ["softmax_score", "kv"],
            "previous_group_state": True,
            "ring_write_proof": ("cmp_kv uses the previous group state slice and the current "
                                 "group ring rows" if not li else
                                 "read-only consumer: cmp_kv uses the shared ring window"),
            "note": ("coff=2 overlap golden: previous group state (rows [0, ratio)) + current "
                     "group; cmp_kv and state_cache are compared"
                     + ("; LI consumer (x_cache_update=false) must not rewrite the ring" if li
                        else "")),
        },
        "remark": ("CSA %s B1 S1 R%d coff2 previous-group state + current group; independent "
                   "CPU golden" % ("LI read-only" if li else "KV", ratio)),
        "readback": ["x_cache"],
    }


def infer_output_shapes(case):
    """The operator's own infer shapes for the four outputs."""
    ratio = case["attrs"]["cmp_ratio"]
    coff = case["attrs"]["coff"]
    x = case["payload"]["x"]
    wkv = case["payload"]["wkv"]
    state = case["payload"]["state_cache"]
    b, s, _ = x.shape
    head_dim = wkv.shape[0] // coff
    sr = (s - 1) // ratio + 1
    return (
        (b, sr, head_dim),
        tuple(int(v) for v in state.shape),
        (b, sr, coff * ratio, head_dim),
        (b, sr, coff * ratio, head_dim),
    )


def _csv_row(case, x_dtype):
    payload = case["payload"]
    shapes, dtypes, formats = [], [], []
    for name in COMPRESSOR_INPUTS:
        value = payload[name]
        if value is None:
            shapes.append(None)
            dtypes.append(None)
            formats.append(None)
            continue
        shapes.append(tuple(int(v) for v in value.shape))
        dtypes.append(INPUT_DTYPES[name])
        formats.append("ND")
    checked = set(case["expected"]["checked_outputs"])
    tolerances = tuple(TOLERANCE if OUTPUT_NAMES[i] in checked else (1.0, 1.0)
                       for i in range(4))
    attributes = dict(case["attrs"])
    for key in ("state_block_table", "seqused", "start_pos", "x_cache_slot_mapping",
                "cu_seqlens"):
        value = payload[key]
        if value is not None:
            attributes[key] = value.tolist()
    return {
        "testcase_name": case["name"],
        "network_name": "deepseek_v4",
        "op_name": "compressor",
        "input_shapes": repr(tuple(shapes)),
        "input_dtypes": repr(tuple(dtypes)),
        "input_formats": repr(tuple(formats)),
        "output_shapes": repr(tuple(
            tuple(int(v) for v in out) for out in case["output_shapes"])),
        "output_dtypes": repr((x_dtype, "float32", "float32", "float32")),
        "output_formats": repr(("ND", "ND", "ND", "ND")),
        "attributes": repr(attributes),
        "input_data_ranges": repr(tuple((None if s is None else (-0.1, 0.1)) for s in shapes)),
        "precision_tolerances": repr(tolerances),
        "absolute_precision": 0.001,
        "output_inplace_indexes": repr(INPLACE),
        "is_enabled": True,
        "remark": case["remark"],
    }


def emit_case(out_root, case, x_dtype):
    case["output_shapes"] = infer_output_shapes(case)
    dest = out_root / case["name"]
    dest.mkdir(parents=True, exist_ok=True)
    for name in COMPRESSOR_INPUTS:
        value = case["payload"][name]
        if value is None:
            continue
        np.save(dest / (name + ".npy"), value)
    files, dtypes = [], []
    for index, original in enumerate(case["outputs"]):
        name = "golden_%d.npy" % index
        if original is None:
            files.append(None)
            dtypes.append(None)
            continue
        np.save(dest / name, np.asarray(original, dtype=np.float32))
        files.append(name)
        dtypes.append("float32")
    (dest / "golden.json").write_text(
        json.dumps({"files": files, "dtypes": dtypes,
                    "output_names": list(OUTPUT_NAMES)}, indent=2), encoding="utf-8")
    (dest / "expected.json").write_text(
        json.dumps(case["expected"], indent=2), encoding="utf-8")
    spec = []
    for name in case.get("readback", []):
        index = COMPRESSOR_INPUTS.index(name)
        value = case["payload"][name]
        spec.append({"index": index, "name": name, "dtype": INPUT_DTYPES[name],
                     "shape": [int(v) for v in value.shape]})
    (dest / "readback_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    with (dest / "case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(_csv_row(case, x_dtype))
    return dest


def self_check(case):
    coff = case["attrs"]["coff"]
    head_dim = case["payload"]["wkv"].shape[0] // coff
    state = case["payload"]["state_cache"]
    assert state.shape[2] == 2 * coff * head_dim, (
        "state_cache last dim must be 2*coff*head_dim, got %d for coff=%d head_dim=%d"
        % (state.shape[2], coff, head_dim))
    assert case["payload"]["x"].shape[1] == 1, "tiny decode case must be S=1"
    cmp_kv = case["outputs"][0]
    if cmp_kv is not None:
        assert cmp_kv.shape == (1, 1, head_dim), (cmp_kv.shape, head_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/pro0921_latent_cases"))
    parser.add_argument("--only", default="", help="build a single case by name")
    parser.add_argument("--first-only", action="store_true",
                        help="build only the first (HCA boundary) case")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260927)
    cases = [
        build_hca_boundary(rng, grad_enabled=False),
        build_hca_boundary(rng, grad_enabled=True),
        build_hca_update_only(rng),
        build_csa_coff2(rng, li=False),
        build_csa_coff2(rng, li=True),
    ]
    if args.first_only:
        cases = cases[:1]
    elif args.only:
        cases = [c for c in cases if c["name"] == args.only]
        if not cases:
            raise SystemExit("FAIL: no case named %r" % args.only)

    manifest = []
    for case in cases:
        self_check(case)
        if case["name"] == "hca_b1_boundary":
            delta, scale = ring_write_discrimination(case)
            case["expected"]["ring_write_delta"] = delta
            case["expected"]["ring_write_delta_rel"] = delta / scale
            assert delta > 50 * TOLERANCE[0] * scale, (
                "sentinel-vs-token cmp_kv delta too small to prove the write: %g" % delta)
        dest = emit_case(args.out, case, INPUT_DTYPES["x"])
        manifest.append({
            "name": case["name"], "dir": str(dest), "kind": case["kind"],
            "cmp_ratio": case["attrs"]["cmp_ratio"], "coff": case["attrs"]["coff"],
            "x_cache_update": case["attrs"]["x_cache_update"],
            "expected": case["expected"],
        })
        print("built %-26s ratio=%-3d coff=%d -> %s"
              % (case["name"], case["attrs"]["cmp_ratio"], case["attrs"]["coff"], dest))
    (args.out / "cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("cases under %s" % args.out)


if __name__ == "__main__":
    main()

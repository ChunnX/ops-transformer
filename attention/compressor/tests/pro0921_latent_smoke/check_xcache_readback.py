# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Validate the optional x_cache read-back produced by run_case_ttk.py's local
# hook (PRO0921_READBACK_XCACHE=1).  TTK only returns outputs, so a mutated
# *input* cannot be observed from result.csv; this script compares the raw
# device bytes of the declared inputs against the documented ring-refresh rule.
#
# Verdict: "PASS:<detail>" or "FAIL:<reason>".  Exit status mirrors the verdict.
#
# Run:
#   python3 check_xcache_readback.py <case-dir>

import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

import latent_ref as ref


def _attrs(case_dir):
    row = next(csv.DictReader((case_dir / "case.csv").open(encoding="utf-8")))
    return ast.literal_eval(row["attributes"])


def _expected_bytes(case_dir, attrs, name, dtype):
    fixture = np.load(case_dir / (name + ".npy"))
    X_CACHE_ROWS = fixture.shape[1]
    x = np.load(case_dir / "x.npy")
    slots = np.load(case_dir / "x_cache_slot_mapping.npy").reshape(-1)
    update = bool(attrs.get("x_cache_update", True))
    expected = fixture.copy()
    if update:
        slot = int(slots[0])
        if slot >= 0:
            token = x.reshape(-1, x.shape[-1])[int(np.load(case_dir / "seqused.npy").reshape(-1)[0]) - 1]
            expected[slot // X_CACHE_ROWS, slot % X_CACHE_ROWS] = token
    if "bfloat16" in dtype or "float16" in dtype:
        # The plugin hands bf16 (or fp16) to TTK; for bf16 the fixture carrier is
        # already bf16-representable, so the expected bits are well defined.
        if "bfloat16" in dtype:
            return (ref.to_bf16(expected).view(np.uint32) >> 16).astype(np.uint16), True
        return expected.astype(np.float16).view(np.uint16), True
    if "uint8" in dtype or "hifloat8" in dtype:
        return expected.astype(np.uint8), False
    raise SystemExit("FAIL: unsupported read-back dtype %r" % dtype)


def main(case_dir):
    case = Path(case_dir)
    spec_path = case / "readback_spec.json"
    if not spec_path.exists():
        return "FAIL:no readback_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    attrs = _attrs(case)
    update = bool(attrs.get("x_cache_update", True))
    checked = 0
    for entry in spec:
        name = entry["name"]
        binary = case / "readback" / (name + ".bin")
        meta_path = case / "readback" / (name + ".meta.json")
        if not binary.exists() or not meta_path.exists():
            return "FAIL:missing read-back for %s (run with PRO0921_READBACK_XCACHE=1)" % name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected, as_uint16 = _expected_bytes(case, attrs, name, meta["dtype"])
        actual = np.frombuffer(binary.read_bytes(), dtype=np.uint16 if as_uint16 else np.uint8)
        if actual.size != expected.size:
            return "FAIL:%s size %d, expected %d" % (name, actual.size, expected.size)
        actual = actual.reshape(expected.shape)
        if np.array_equal(actual, expected):
            checked += 1
            continue
        diff = int(np.count_nonzero(actual != expected))
        # Distinguish "the expected refresh happened" from "the buffer moved":
        fixture_bytes = np.load(case / (name + ".npy"))
        if "bfloat16" in meta["dtype"]:
            pre = (ref.to_bf16(fixture_bytes).view(np.uint32) >> 16).astype(np.uint16)
        elif "uint8" in meta["dtype"] or "hifloat8" in meta["dtype"]:
            pre = fixture_bytes.astype(np.uint8)
        else:
            pre = fixture_bytes.view(np.uint16)
        moved = int(np.count_nonzero(actual != pre))
        if update and moved > 0:
            return ("FAIL:%s differs from the expected post-refresh ring in %d elements "
                    "(refresh happened but the written value/shape rule is wrong)" % (name, diff))
        if not update and moved > 0:
            return "FAIL:%s was modified although x_cache_update=false (%d elements)" % (name, moved)
        return "FAIL:%s no elements changed but the refresh was expected" % name
    return "PASS:readback verified for %d input(s) update=%s" % (checked, update)


if __name__ == "__main__":
    verdict = main(sys.argv[1])
    print(verdict)
    raise SystemExit(0 if verdict.startswith("PASS:") else 1)

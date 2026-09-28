# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Per-output verdict for one TTK case directory.
#
# The previous round accepted "PASS" purely from dyn_perf_us > 0, which marks
# *any* executed kernel as a pass even when the precision comparison was
# suppressed or never ran.  This checker reads the per-output precision string
# (the `dyn_precision` column: TTK joins one token per flattened output) and
# requires:
#   * a synchronized native kernel and device-output readback, and
#   * precision_status == PASS, and
#   * every output listed in expected.json checked_outputs carries a *numeric*
#     comparison token (a real isclose ran), and
#   * every output listed in suppressed_outputs is reported SUPPRESSED.
#
# Outputs:
#   "PASS:<detail>" or "FAIL:<reason>"

import csv
import json
import math
import sys
from pathlib import Path


SUPPRESSED_TOKENS = {"", "suppressed", "none", "null", "dyn_off", "stc_off",
                     "bin_off", "unsupported", "golden_failure", "golden_timeout"}


def _tokens(value):
    return [t.strip() for t in (value or "").split(",")]


def _is_numeric(token):
    try:
        return math.isfinite(float(token.rstrip("%")))
    except (TypeError, ValueError):
        return False


def main(case_dir):
    case = Path(case_dir)
    expected = json.loads((case / "expected.json").read_text(encoding="utf-8"))
    output_names = None
    golden_path = case / "golden.json"
    if golden_path.exists():
        output_names = json.loads(golden_path.read_text(encoding="utf-8")).get("output_names")

    result = case / "result.csv"
    if not result.exists():
        return "FAIL:no result.csv"
    rows = list(csv.DictReader(result.open(encoding="utf-8")))
    if len(rows) != 1:
        return "FAIL:expected one result row, got %d" % len(rows)
    row = rows[0]

    perf = row.get("dyn_perf_us")
    receipt_path = case / "native_execution.json"
    if not receipt_path.exists():
        return "FAIL:no native stream synchronization/readback receipt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not (receipt.get("stream_synchronized") is True and
            receipt.get("outputs_copied_from_device") is True and
            int(receipt.get("block_dim", 0)) > 0):
        return "FAIL:incomplete native execution receipt"

    if row.get("precision_status") != "PASS":
        return "FAIL:precision_status=%s (dyn_precision=%s)" % (
            row.get("precision_status"), row.get("dyn_precision"))

    tokens = _tokens(row.get("dyn_precision"))
    checked = set(expected.get("checked_outputs", []))
    suppressed = set(expected.get("suppressed_outputs", []))
    if output_names:
        index_of = {name: i for i, name in enumerate(output_names)}
    else:
        index_of = {}

    for name in sorted(checked):
        index = index_of.get(name)
        if index is None:
            return "FAIL:expected.json names %r but golden.json has no output_names" % name
        if index >= len(tokens):
            return "FAIL:%s has no precision token (dyn_precision=%s)" % (name, row.get("dyn_precision"))
        token = tokens[index]
        if token.lower() in SUPPRESSED_TOKENS:
            return "FAIL:%s reported %r, expected a real comparison" % (name, token)
        if not _is_numeric(token):
            return "FAIL:%s precision token %r is not numeric" % (name, token)

    for name in sorted(suppressed):
        index = index_of.get(name)
        if index is None or index >= len(tokens):
            continue
        if tokens[index].lower() not in SUPPRESSED_TOKENS:
            return "FAIL:%s should be SUPPRESSED but reported %r" % (name, tokens[index])

    detail = "native_readback=complete dyn_perf_us=%s checked=%s suppressed=%s" % (
        perf, ",".join(sorted(checked)) or "-", ",".join(sorted(suppressed)) or "-")
    if expected.get("ring_write_delta_rel") is not None:
        detail += " ring_write_delta_rel=%.4g" % expected["ring_write_delta_rel"]
    return "PASS:%s" % detail


if __name__ == "__main__":
    print(main(sys.argv[1]))

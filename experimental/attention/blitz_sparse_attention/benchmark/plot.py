#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Plot speedup vs sparsity from `benchmark.py` stdout.

Reads the benchmark table (one row per (block_shape, B, H, S, D, sparsity)
config) from stdin, then draws one line per distinct (sequence_length,
block_shape) pair. y = `npu_fusion_attention` (dense reference) latency at
sparsity 0 divided by our_latency at the corresponding sparsity.

Usage
-----
    python benchmark.py | python plot.py
    python plot.py < bench.log
    python plot.py --output bench.png < bench.log

A horizontal y = 1 line marks parity with the dense reference. The reference
is measured only at sparsity 0 in `benchmark.py` (one Ref_Latency cell per
block_shape × seq_len pair), so the same reference baseline is used for every
sparsity of the matching (seq_len, block_shape) line.
"""

import argparse
import re
import sys
from collections import defaultdict


# Result row format produced by `benchmark.py`:
#   block_shape   H   B    s_q   s_kv    D  frame(L,R,T,B)  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
#       128x128   3   1 118806 118806  128               -      0.00             yes           160225.13          185475.31
#       128x256   3   1 118806 118806  128     (29,1,29,1)      0.05             N/A                N/A          164822.39
# The frame column is "-" when sparsity == 0 (dense, frame irrelevant) or
# "(L,R,T,B)" otherwise; matched as a single non-whitespace token and discarded.
ROW_RE = re.compile(
    r"^\s*"
    r"(?P<bsq>\d+)x(?P<bskv>\d+)\s+"
    r"(?P<h>\d+)\s+"
    r"(?P<b>\d+)\s+"
    r"(?P<sq>\d+)\s+"
    r"(?P<skv>\d+)\s+"
    r"(?P<d>\d+)\s+"
    r"(?P<frame>\S+)\s+"
    r"(?P<sp>[\d.]+)\s+"
    r"(?P<eq>\S+)\s+"
    r"(?P<ref>\S+)\s+"
    r"(?P<ours>\S+)\s*$"
)


def _to_float(token: str):
    if token == "N/A":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_rows(stream):
    """Yield dicts for each parsable result row.  Header / divider lines are
    ignored automatically by the regex."""
    for line in stream:
        m = ROW_RE.match(line.rstrip("\n"))
        if not m:
            continue
        yield {
            "block_shape": (int(m["bsq"]), int(m["bskv"])),
            "b": int(m["b"]),
            "h": int(m["h"]),
            "s_q": int(m["sq"]),
            "s_kv": int(m["skv"]),
            "d": int(m["d"]),
            "sparsity": float(m["sp"]),
            "ref": _to_float(m["ref"]),
            "ours": _to_float(m["ours"]),
        }


def _fmt_s(s: int) -> str:
    """Sequence-length label: full integer, no thousands separator (118806 -> '118806')."""
    return str(s)


def main():
    parser = argparse.ArgumentParser(
        description="Plot blitz_sparse_attention speedup vs sparsity "
                    "from benchmark.py stdout.",
    )
    parser.add_argument("--output", "-o", default="benchmark.png",
                        help="output PNG path (default: benchmark.png)")
    parser.add_argument("--title", default="blitz_sparse_attention speedup vs sparsity")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    rows = list(parse_rows(sys.stdin))
    if not rows:
        sys.exit("plot.py: no benchmark rows parsed from stdin "
                 "(pipe `benchmark.py` stdout into this script).")

    # Group by (B, H, S, D, block_shape).
    groups = defaultdict(list)
    for r in rows:
        groups[(r["b"], r["h"], r["s_q"], r["d"], r["block_shape"])].append(r)

    # Promote any (B, H, S, D) that is constant across every line into the
    # figure title — keeps per-line labels short and focuses the legend on
    # what actually varies (block_shape, in the common single-shape sweep).
    bs_vals  = {k[0] for k in groups}
    hs_vals  = {k[1] for k in groups}
    sqs_vals = {k[2] for k in groups}
    ds_vals  = {k[3] for k in groups}
    common = {
        "B": next(iter(bs_vals))  if len(bs_vals)  == 1 else None,
        "H": next(iter(hs_vals))  if len(hs_vals)  == 1 else None,
        "S": next(iter(sqs_vals)) if len(sqs_vals) == 1 else None,
        "D": next(iter(ds_vals))  if len(ds_vals)  == 1 else None,
    }
    title_bits = []
    if all(v is not None for v in common.values()):
        title_bits.append(
            f"BNSD=({common['B']},{common['H']},{_fmt_s(common['S'])},{common['D']})"
        )

    # matplotlib is imported lazily so `--help` works on hosts without it.
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no DISPLAY on NPU hosts.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    # Colour each line by BLOCK_SIZE_KV (the dominant lever for speedup at high
    # sparsity per the README sweep); linestyle separates the four
    # BLOCK_SIZE_Q values; markers separate sequence lengths (typically a
    # single S in one run, so all markers look identical).
    KV_PALETTE = {
        128:  "#d62728",   # tab:red
        256:  "#ff7f0e",   # tab:orange
        512:  "#2ca02c",   # tab:green
        1024: "#1f77b4",   # tab:blue
    }
    LINESTYLE_BY_Q = {128: "-", 256: "--", 512: "-.", 1024: ":"}
    seq_lens = sorted({k[2] for k in groups})
    MARKER_BY_SEQ = dict(zip(seq_lens, ("o", "s", "^", "D", "v", "P", "X", "*")))

    lines_drawn = 0
    skipped = []
    # Sort lines so the legend is colour-grouped: by KV first (groups all
    # red/orange/green/blue lines together), then Q, then any varying B/H/S/D.
    def _sort_key(key):
        b, h, s_q, d, (bsq, bskv) = key
        return (bskv, bsq, s_q, h, b, d)

    for key in sorted(groups, key=_sort_key):
        b, h, s_q, d, bs = key
        bsq, bskv = bs
        grows = groups[key]

        # Reference baseline: any row where Ref_Latency was emitted.
        # benchmark.py prints it only at sparsity 0, so take the first non-None.
        ref_baseline = next((r["ref"] for r in grows if r["ref"] is not None), None)
        if ref_baseline is None:
            skipped.append((s_q, bs, "no Ref_Latency baseline (run sparsity=0 row)"))
            continue

        grows = sorted(grows, key=lambda r: r["sparsity"])
        xs, ys = [], []
        for r in grows:
            if r["ours"] is None or r["ours"] == 0.0:
                continue
            xs.append(r["sparsity"])
            ys.append(ref_baseline / r["ours"])
        if not xs:
            skipped.append((s_q, bs, "no Our_Latency cells"))
            continue

        # Per-line label: just block_shape if (B,H,S,D) is shared by all lines;
        # otherwise prepend the differing BNSD tuple.
        if all(v is not None for v in common.values()):
            label = f"block_shape=({bsq},{bskv})"
        else:
            label = f"BNSD=({b},{h},{_fmt_s(s_q)},{d}), block_shape=({bsq},{bskv})"

        ax.plot(
            xs, ys,
            color=KV_PALETTE.get(bskv, "#666666"),
            linestyle=LINESTYLE_BY_Q.get(bsq, "-"),
            marker=MARKER_BY_SEQ.get(s_q, "o"),
            markersize=5, linewidth=1.5,
            label=label,
        )
        lines_drawn += 1

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("sparsity (fraction of blocks zeroed)", fontsize=11)
    ax.set_ylabel("speedup  =  Ref_Latency(sparsity=0)  /  Our_Latency", fontsize=11)
    title = args.title
    if title_bits:
        title = f"{title}\n{'   '.join(title_bits)}"
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    # One column keeps the colour-grouped ordering intact (matplotlib's
    # 2-column flow interleaves by row); legend can run tall vertically.
    ax.legend(loc="upper left", fontsize=10, ncol=1, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"plot.py: wrote {args.output}  ({lines_drawn} lines, {len(rows)} rows parsed)",
          file=sys.stderr)
    for s_q, bs, why in skipped:
        print(f"plot.py: skipped S={s_q} {bs[0]}x{bs[1]}  ({why})", file=sys.stderr)


if __name__ == "__main__":
    main()

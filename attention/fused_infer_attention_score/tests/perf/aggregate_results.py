#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Aggregate grid_results.csv into a Bound-grouped Markdown summary.

Usage:
    python3 aggregate_results.py <run_dir>

Reads:
    <run_dir>/grid_results.csv
Writes:
    <run_dir>/grid_summary.md
"""

import argparse
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict


BOUND_ORDER = [
    "MTE2 BOUND", "CUBE BOUND", "FIXP BOUND", "MTE3 BOUND", "VEC BOUND",
    "SCALAR BOUND", "无 bound", "UNKNOWN", "ERROR",
]


def safe_float(s, default=0.0):
    if s is None or str(s).strip() == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def case_signature(row):
    extras = []
    if row.get("scenario"):
        extras.append(row["scenario"])
    if row.get("antiquant_mode"):
        extras.append(f"aq_mode={row['antiquant_mode']}")
    if row.get("mask_subtype"):
        extras.append(row["mask_subtype"])
    if row.get("out_dtype"):
        extras.append(f"out={row['out_dtype']}")
    extra_str = (" [" + " ".join(extras) + "]") if extras else ""
    return (f"B={row.get('batch_size','?')} N_q={row.get('q_head_num','?')} "
            f"N_kv={row.get('kv_head_num','?')} q_seq={row.get('q_seq','?')} "
            f"kv_seq={row.get('kv_seq','?')} D={row.get('head_dim','?')} "
            f"{row.get('in_layout','?')} {row.get('dtype','?')} "
            f"block={row.get('block_size','?')}{extra_str}")


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def regime(row):
    qs = safe_float(row.get("q_seq"), 0.0)
    return "decode" if qs <= 1 else "prefill"


def render_summary(rows):
    lines = []
    total = len(rows)
    if total == 0:
        return "# FIA 性能网格 Sweep 汇总\n\n_No cases in CSV._\n"

    bound_groups = defaultdict(list)
    for r in rows:
        bt = (r.get("bound_type") or "UNKNOWN").strip() or "UNKNOWN"
        bound_groups[bt].append(r)

    lines.append("# FIA 性能网格 Sweep 汇总")
    lines.append("")
    lines.append(f"- 总 case 数：**{total}**")
    ok_count = sum(1 for r in rows if r.get("status") == "OK")
    err_count = total - ok_count
    lines.append(f"- 成功：{ok_count}  失败：{err_count}")
    lines.append("")

    # Overview
    lines.append("## 1. Bound 类型总览")
    lines.append("")
    lines.append("| Bound 类型 | Case 数 | 占比 |")
    lines.append("|---|---|---|")
    for bt in BOUND_ORDER:
        n = len(bound_groups.get(bt, []))
        if n == 0:
            continue
        lines.append(f"| {bt} | {n} | {n / total * 100:.1f}% |")
    # Any unexpected bound types not in BOUND_ORDER
    for bt in sorted(bound_groups.keys()):
        if bt in BOUND_ORDER:
            continue
        n = len(bound_groups[bt])
        lines.append(f"| {bt} | {n} | {n / total * 100:.1f}% |")
    lines.append("")

    # Phase 2 scenario coverage matrix
    scenario_groups = defaultdict(list)
    for r in rows:
        sc = (r.get("scenario") or "no_quant").strip() or "no_quant"
        scenario_groups[sc].append(r)
    if scenario_groups:
        lines.append("## 1.5. Phase 2 场景覆盖矩阵")
        lines.append("")
        lines.append("| scenario | total | OK | ERROR | bound 分布 |")
        lines.append("|---|---|---|---|---|")
        for sc in sorted(scenario_groups):
            grp = scenario_groups[sc]
            ok = sum(1 for r in grp if r.get("status") == "OK")
            err = len(grp) - ok
            bnd_count = Counter((r.get("bound_type") or "?").strip() or "?" for r in grp)
            bnd_str = "; ".join(f"{k}:{v}" for k, v in bnd_count.most_common())
            lines.append(f"| {sc} | {len(grp)} | {ok} | {err} | {bnd_str} |")
        lines.append("")

    # SCALAR subtype split
    scalar_rows = bound_groups.get("SCALAR BOUND", [])
    if scalar_rows:
        sub_counter = Counter((r.get("scalar_subtype") or "UNKNOWN").strip() or "UNKNOWN"
                              for r in scalar_rows)
        lines.append("## 2. SCALAR BOUND 子类拆分")
        lines.append("")
        lines.append("| 子类 | Case 数 |")
        lines.append("|---|---|")
        for sub, n in sub_counter.most_common():
            lines.append(f"| {sub} | {n} |")
        lines.append("")

    # Per-bound top-N tables
    section_no = 3 if scalar_rows else 2
    lines.append(f"## {section_no}. 按 Bound 分组的 Top-10 最慢 Case")
    lines.append("")
    for bt in [*BOUND_ORDER, *sorted(set(bound_groups.keys()) - set(BOUND_ORDER))]:
        group = bound_groups.get(bt, [])
        if not group:
            continue
        lines.append(f"### {bt}（{len(group)} cases）")
        lines.append("")
        lines.append("| case_id | duration(us) | SCALAR% | SCALARLDST% | MTE2% | MTE3% | CUBE% | 参数 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sorted_group = sorted(group, key=lambda r: -safe_float(r.get("task_duration_us")))
        for r in sorted_group[:10]:
            lines.append(
                f"| {r.get('case_id','')} | {r.get('task_duration_us','')} | "
                f"{r.get('scalar_pct','')} | {r.get('scalarldst_pct','')} | "
                f"{r.get('mte2_pct','')} | {r.get('mte3_pct','')} | {r.get('cube_pct','')} | "
                f"{case_signature(r)} |"
            )
        lines.append("")

    # (scenario × bound_type) double-grouping
    section_no += 1
    lines.append(f"## {section_no}. (scenario, bound_type) 双重分组矩阵")
    lines.append("")
    pairs = defaultdict(list)
    for r in rows:
        if r.get("status") != "OK":
            continue
        sc = (r.get("scenario") or "no_quant").strip() or "no_quant"
        bt = (r.get("bound_type") or "?").strip() or "?"
        pairs[(sc, bt)].append(r)
    if pairs:
        all_scs = sorted({k[0] for k in pairs})
        all_bts = [bt for bt in BOUND_ORDER if any(k[1] == bt for k in pairs)]
        lines.append("| scenario \\ bound | " + " | ".join(all_bts) + " |")
        lines.append("|---|" + "---|" * len(all_bts))
        for sc in all_scs:
            cells = [str(len(pairs.get((sc, bt), []))) for bt in all_bts]
            lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    else:
        lines.append("_no successful cases to cross-tab._")
    lines.append("")

    # Outliers within each (layout, dtype, regime) cohort
    section_no += 1
    lines.append(f"## {section_no}. 离群 Case（同 cohort 95 分位以上）")
    lines.append("")
    lines.append("cohort = (in_layout, dtype, q_seq_regime)；regime = q_seq<=1 ? decode : prefill")
    lines.append("")
    lines.append("| case_id | cohort | duration(us) | 95p(us) | bound_type | 参数 |")
    lines.append("|---|---|---|---|---|---|")

    cohorts = defaultdict(list)
    for r in rows:
        if r.get("status") != "OK":
            continue
        key = (r.get("in_layout", ""), r.get("dtype", ""), regime(r))
        cohorts[key].append(r)

    outlier_lines = []
    for key, group in cohorts.items():
        if len(group) < 4:
            continue
        durations = [safe_float(r.get("task_duration_us")) for r in group]
        p95 = percentile(durations, 95)
        for r in group:
            d = safe_float(r.get("task_duration_us"))
            if d >= p95 and d > 0:
                outlier_lines.append(
                    (d, f"| {r.get('case_id','')} | {key[0]}/{key[1]}/{key[2]} | "
                         f"{d:.2f} | {p95:.2f} | {r.get('bound_type','')} | "
                         f"{case_signature(r)} |")
                )
    outlier_lines.sort(key=lambda t: -t[0])
    for _, line in outlier_lines[:20]:
        lines.append(line)
    if not outlier_lines:
        lines.append("| _no outliers_ |  |  |  |  |  |")
    lines.append("")

    # Footer with stats
    section_no += 1
    lines.append(f"## {section_no}. 全局统计")
    lines.append("")
    durations = [safe_float(r.get("task_duration_us")) for r in rows
                 if r.get("status") == "OK" and safe_float(r.get("task_duration_us")) > 0]
    if durations:
        lines.append(f"- duration min / median / max / p95: "
                     f"{min(durations):.2f} / {statistics.median(durations):.2f} / "
                     f"{max(durations):.2f} / {percentile(durations, 95):.2f} us")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate FIA grid sweep CSV → Markdown summary.")
    parser.add_argument("run_dir", help="Sweep run directory containing grid_results.csv")
    parser.add_argument("-o", "--output", help="Output markdown path (default: <run_dir>/grid_summary.md)")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "grid_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(2)

    rows = load_rows(csv_path)
    md = render_summary(rows)

    out_path = args.output or os.path.join(args.run_dir, "grid_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        if not md.endswith("\n"):
            f.write("\n")
    print(f"Wrote {out_path}  ({len(rows)} cases)")


if __name__ == "__main__":
    main()

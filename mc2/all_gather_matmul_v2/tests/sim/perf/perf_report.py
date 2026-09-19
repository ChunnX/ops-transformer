#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Perf report: bottleneck-dimension decomposition + multi-round statistics.

输入：collect_perf.sh 的 perf_raw/<case>/branch_{v1|apace}/round_<i>/ 目录
（4.1 双分支布局，键记为 case[branch]；旧布局 perf_raw/<case>/round_<i>/ 兼容）。
每轮可含：
  * npusim_*/report/summary.json —— cannsim 仿真报告（pipe 级利用率与周期）
  * stdout.log —— example 计时打印（AGM_V2_TIMING_US 行，参数化 example
    契约；每 launch 一行）
  * prof/ —— msprof 归档（有卡；深度 pipe 级分析走 ops-profiling 的
    msprof_perf_summary.py，本脚本记录 kernel 时长中位数/离散度入口待
    有卡数据接入）

输出（--raw-root 下）：
  * perf_summary.json —— 逐用例：各维度占比/耗时 + 多轮中位数（median）、
    最小/最大、MAD（median absolute deviation）与 CV（离散系数）
  * perf_report.md    —— 人读报告（瓶颈维度排序）

瓶颈维度分解（计算 / 搬入 / 搬出 / 标量）：
  cannsim summary.json 的 pipe 利用率映射——
    compute  = AIC_CUBE
    copy_in  = AIC_MTE2 + AIC_MTE1        （GM/L1 → UB/L0 载入侧）
    copy_out = AIC_FIXPIPE                （L0C → GM 写出侧）
    scalar   = AIC_SCALAR
  占比 = 各 pipe busy 利用率归一化；绝对量以 kernel_total_clocks 记录。
  （维度口径与 ops-simulator performance-metrics-reference 一致；AIV 侧
  pipe 若出现（AIVx_*）单独列出，不并入四维。）

Usage:
  python3 perf_report.py --raw-root <perf_raw dir> [--out-dir DIR]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# 维度 → summary.json pipe_utilization 键（ops-simulator 口径）
DIMENSION_PIPES = {
    "compute": ["AIC_CUBE"],
    "copy_in": ["AIC_MTE2", "AIC_MTE1"],
    "copy_out": ["AIC_FIXPIPE"],
    "scalar": ["AIC_SCALAR"],
}

EXAMPLE_TIMING_PATTERN = "AGM_V2_TIMING_US:"


def _iter_rounds(case_dir):
    for round_dir in sorted(Path(case_dir).glob("round_*")):
        if round_dir.is_dir():
            yield round_dir


def _find_sim_summary(round_dir):
    """Locate npusim_*/report/summary.json in one round dir."""
    for cand in round_dir.glob("npusim_*/report/summary.json"):
        return cand
    for cand in round_dir.glob("cannsim_*/report/summary.json"):
        return cand
    return None


def parse_round(round_dir):
    """Parse one round → metrics dict (may be partial)."""
    metrics = {}
    summary_path = _find_sim_summary(round_dir)
    if summary_path is not None:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = None
        if summary:
            pipes = summary.get("pipe_utilization", {}) or {}
            pipe_means = {}
            for pipe, val in pipes.items():
                if isinstance(val, dict) and "mean" in val:
                    pipe_means[pipe] = float(val["mean"])
                elif isinstance(val, (int, float)):
                    pipe_means[pipe] = float(val)
            dims = {}
            for dim, pipe_keys in DIMENSION_PIPES.items():
                vals = [pipe_means[p] for p in pipe_keys if p in pipe_means]
                dims[dim] = sum(vals) if vals else None
            known = [v for v in dims.values() if v is not None]
            total = sum(known)
            metrics["dimensions_util"] = dims
            metrics["dimensions_share"] = (
                {k: (v / total if total > 0 else 0.0)
                 for k, v in dims.items() if v is not None}
                if known else None)
            metrics["kernel_total_clocks"] = summary.get(
                "kernel_total_clocks", summary.get("kernel_info", {}).get(
                    "kernel_total_clocks")
                if isinstance(summary.get("kernel_info"), dict) else None)
            diag = summary.get("top_level_diagnosis", {}) or {}
            metrics["dominant_pipeline"] = diag.get("dominant_pipeline")
            metrics["imbalance_ratio"] = diag.get("imbalance_ratio")
    # example 计时打印（参数化 example 契约）
    stdout = round_dir / "stdout.log"
    if stdout.exists():
        us = []
        for line in stdout.read_text(errors="ignore").splitlines():
            if EXAMPLE_TIMING_PATTERN in line:
                try:
                    us.append(float(line.split(EXAMPLE_TIMING_PATTERN)[1]
                                    .strip().split()[0]))
                except (ValueError, IndexError):
                    pass
        if us:
            metrics["example_timing_us"] = us
    if (round_dir / "prof").exists():
        metrics["msprof_prof_dir"] = str(round_dir / "prof")
    return metrics


def aggregate(case_dir):
    rounds = [parse_round(r) for r in _iter_rounds(case_dir)]
    rounds = [r for r in rounds if r]
    if not rounds:
        return None

    def _series(key):
        vals = []
        for r in rounds:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def _stats(vals):
        if not vals:
            return None
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        mean = statistics.fmean(vals)
        cv = (statistics.pstdev(vals) / mean) if mean else None
        return {
            "rounds": len(vals),
            "median": med,
            "min": min(vals),
            "max": max(vals),
            "mad": mad,
            "cv": cv,
        }

    agg = {"rounds_collected": len(rounds)}
    # 维度占比：逐轮归一化后取中位数
    shares = {}
    for dim in DIMENSION_PIPES:
        series = [r.get("dimensions_share", {}).get(dim)
                  for r in rounds
                  if r.get("dimensions_share")]
        series = [s for s in series if s is not None]
        if series:
            shares[dim] = {
                "median_share": round(statistics.median(series), 6),
                "min_share": round(min(series), 6),
                "max_share": round(max(series), 6),
            }
    agg["dimension_share"] = shares or None
    # 各维度利用率绝对值（mean utilization）
    utils = {}
    for dim in DIMENSION_PIPES:
        series = [r.get("dimensions_util", {}).get(dim)
                  for r in rounds if r.get("dimensions_util")]
        series = [s for s in series if s is not None]
        if series:
            utils[dim] = round(statistics.median(series), 6)
    agg["dimension_util_median"] = utils or None
    # 周期与计时
    agg["kernel_total_clocks"] = _stats(_series("kernel_total_clocks"))
    timing = [us for r in rounds for us in r.get("example_timing_us", [])]
    agg["example_timing_us"] = _stats(timing)
    # 诊断
    dominants = [r.get("dominant_pipeline") for r in rounds
                 if r.get("dominant_pipeline")]
    if dominants:
        agg["dominant_pipeline_rounds"] = {
            str(d): dominants.count(d) for d in set(dominants)}
    if any("msprof_prof_dir" in r for r in rounds):
        agg["msprof_note"] = (
            "msprof 归档存在：pipe 级深度分析请用 ops-profiling 的 "
            "msprof_perf_summary.py 解析 PROF_GROUP（MC2 多 rank 必须 "
            "msprof，禁 msprof op）")
    # 瓶颈维度判定（按占比中位数排序）
    if shares:
        ranked = sorted(shares.items(),
                        key=lambda kv: kv[1]["median_share"], reverse=True)
        agg["bottleneck_ranking"] = [
            {"dimension": k, **v} for k, v in ranked]
        agg["bottleneck_dimension"] = ranked[0][0]
    return agg


def render_markdown(summary, out_path):
    lines = ["# AllGatherMatmulV2 perf report (A8 备用框架)", "",
             "> 瓶颈维度口径：compute=AIC_CUBE / copy_in=AIC_MTE2+MTE1 / "
             "copy_out=AIC_FIXPIPE / scalar=AIC_SCALAR（cannsim "
             "summary.json pipe 利用率归一化占比）", ""]
    for case, agg in sorted(summary.items()):
        if agg is None:
            lines += [f"## {case}", "", "- （无有效轮次数据）", ""]
            continue
        lines += [f"## {case}", "",
                  f"- rounds collected: {agg['rounds_collected']}"]
        if agg.get("bottleneck_ranking"):
            lines += ["- 瓶颈维度分解（占比中位数）：", "",
                      "| 维度 | median | min | max |", "|---|---|---|---|"]
            for item in agg["bottleneck_ranking"]:
                lines.append(
                    f"| {item['dimension']} | {item['median_share']:.4f} | "
                    f"{item['min_share']:.4f} | {item['max_share']:.4f} |")
            lines += [f"- bottleneck dimension: "
                      f"{agg.get('bottleneck_dimension')}"]
        if agg.get("dominant_pipeline_rounds"):
            lines += [f"- dominant pipeline (per round): "
                      f"{agg['dominant_pipeline_rounds']}"]
        if agg.get("kernel_total_clocks"):
            s = agg["kernel_total_clocks"]
            lines += [f"- kernel_total_clocks: median={s['median']} "
                      f"min={s['min']} max={s['max']} mad={s['mad']} "
                      f"cv={s['cv']}"]
        if agg.get("example_timing_us"):
            s = agg["example_timing_us"]
            lines += [f"- example timing (us): median={s['median']} "
                      f"min={s['min']} max={s['max']} mad={s['mad']} "
                      f"cv={s['cv']}（n={s['rounds']}）"]
        if agg.get("msprof_note"):
            lines += [f"- note: {agg['msprof_note']}"]
        lines += [""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True,
                        help="collect_perf.sh 的 perf_raw 目录")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录（默认 raw-root）")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir) if args.out_dir else raw_root
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for case_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        # 分支层发现（4.1 双分支布局）：case/branch_{v1|apace}/round_* → 键 case[branch]；
        # 无分支层时按旧布局（case/round_*）聚合，键为 case。
        branch_dirs = [b for b in sorted(case_dir.glob("branch_*"))
                       if b.is_dir() and any(b.glob("round_*"))]
        if branch_dirs:
            for b in branch_dirs:
                key = "{}[{}]".format(case_dir.name, b.name[len("branch_"):])
                summary[key] = aggregate(b)
        else:
            summary[case_dir.name] = aggregate(case_dir)

    with open(out_dir / "perf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    render_markdown(summary, out_dir / "perf_report.md")
    print(f"[perf_report] {len(summary)} cases -> "
          f"{out_dir / 'perf_summary.json'} / {out_dir / 'perf_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

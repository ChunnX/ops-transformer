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
    契约；每 launch 一行）+ msprof 透传输出
  * flush.log —— 轮前 L2 flush 证据（l2_flush.py FLUSH_OK 行，msprof 模式）
  * prof/ —— msprof 归档（有卡；B9⑤-5 起本脚本直接解析 op_summary_*.csv）

输出（--raw-root 下）：
  * perf_summary.json —— 逐用例：各维度占比/耗时 + 多轮中位数（median）、
    最小/最大、MAD（median absolute deviation）与 CV（离散系数）
  * perf_report.md    —— 人读报告（瓶颈维度排序）

瓶颈维度分解（计算 / 搬入 / 搬出 / 标量）：
  cannsim summary.json 的 pipe 利用率映射——
    compute  = AIC_CUBE；copy_in = AIC_MTE2 + AIC_MTE1；
    copy_out = AIC_FIXPIPE；scalar = AIC_SCALAR
  msprof op_summary（--aic-metrics=PipeUtilization，950 实测列名）——
    compute  = aic_mac_time(us)（mac = Cube 乘累加）
    copy_in  = aic_mte1_time(us) + aic_mte2_time(us)
    copy_out = aic_mte3_time(us) + aic_fixpipe_time(us)
    scalar   = aic_scalar_time(us)（含核内同步等待的标量轮询）
  AIV 侧（aiv_vec/scalar/mte2/mte3）单独记录不并入四维；命中列登记于
  输出 msprof.dimension_columns（提取方法留痕）。

msprof 主 kernel 识别（950 实测：op_summary 的 Op Name 为算子注册名）：
  * 主 kernel（MIX_AIC 计算核）：Op Name 归一化（lower + 去下划线）含
    `allgathermatmul` 且 Task Type ∈ {MIX_AIC, AI_CORE, AIV, MIX}——
    实测名 `AllGatherMatmulV2`（op_kernel/arch35 单入口
    all_gather_matmul_v2，v1/APACE 由 tiling USING_APACE_IMPL 位核内
    分发，两分支同名）；
  * AI_CPU 通信侧 kernel（信息性单独记录）：同前缀且 Task Type=AI_CPU
    （实测名 `AllGatherMatmulV2Mc2AicpuKernel`，HCCL server 侧）；
  * HCCL 建链类（RunAicpuCommInit 等）只进 op_counts 噪声清单。

msprof 多轮统计口径（B9⑤-5 任务拍板 + ops-profiling msprof-guide「MC²
多 rank 算子采集」官方口径）：
  * 跳 warmup：丢弃 round_1（每轮为独立进程，首轮含进程冷启动/频率爬升）；
  * 去离群：|v - median| > 3×MAD 剔除（MAD=0 时全保留），剔除数登记；
  * 中位数为准：per-rank median + min/max/MAD/CV；
  * 跨 rank 聚合（官方口径，木桶效应——最慢 rank 决定整体）：主口径 =
    max(per-rank median)；另记 mean(per-rank median)（卡均值）与官方字面
    口径 per-rank last-5 mean → max 作参考。

Usage:
  python3 perf_report.py --raw-root <perf_raw dir> [--out-dir DIR]
"""

import argparse
import csv
import json
import re
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

# ---- msprof op_summary 解析（B9⑤-5 有卡数据接入，列名经 950 实测校准）----
# 主 kernel 匹配：Op Name 归一化（lower+去下划线）含 allgathermatmul——
# 同时覆盖算子注册名 AllGatherMatmulV2 与 kernel 函数名 all_gather_matmul_v2
MAIN_KERNEL_TOKEN = "allgathermatmul"
# 主 kernel 合法 Task Type（AI_CPU 为通信侧 kernel，单独归类）
MAIN_TASK_TYPES = {"mix_aic", "ai_core", "aiv", "mix", ""}

# 多轮统计口径常量（详见模块 docstring）
WARMUP_DROP_ROUNDS = 1
OUTLIER_MAD_K = 3.0
OFFICIAL_LAST_N = 5

# 维度 → op_summary 时间列（绝对 us，跨维度可比；950 实测列名）
MSPROF_DIMENSION_COLS = {
    "compute": [r"aic_mac_time\(us\)"],
    "copy_in": [r"aic_mte1_time\(us\)", r"aic_mte2_time\(us\)"],
    "copy_out": [r"aic_mte3_time\(us\)", r"aic_fixpipe_time\(us\)"],
    "scalar": [r"aic_scalar_time\(us\)"],
}
# AIV 侧信息列（单独记录，不并入四维）
MSPROF_AIV_COLS = [
    r"aiv_vec_time\(us\)",
    r"aiv_scalar_time\(us\)",
    r"aiv_mte2_time\(us\)",
    r"aiv_mte3_time\(us\)",
]
# 附加信息列
MSPROF_EXTRA_COLS = [
    r"Task Wait Time\(us\)",
    r"aicore_time\(us\)",
    r"aiv_time\(us\)",
    r"cube_utilization\(%\)",
]


def _iter_rounds(case_dir):
    """轮目录按轮号数值序（round_2 < round_10；字典序会把 round_10 排前，
    10 轮口径下 warmup-drop/last-N 取值将错序——B9⑤-5 修复）。"""

    def _round_key(p):
        tail = p.name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    return sorted(
        (p for p in Path(case_dir).glob("round_*") if p.is_dir()), key=_round_key
    )


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
                {
                    k: (v / total if total > 0 else 0.0)
                    for k, v in dims.items()
                    if v is not None
                }
                if known
                else None
            )
            metrics["kernel_total_clocks"] = summary.get(
                "kernel_total_clocks",
                summary.get("kernel_info", {}).get("kernel_total_clocks")
                if isinstance(summary.get("kernel_info"), dict)
                else None,
            )
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
                    us.append(
                        float(line.split(EXAMPLE_TIMING_PATTERN)[1].strip().split()[0])
                    )
                except (ValueError, IndexError):
                    pass
        if us:
            metrics["example_timing_us"] = us
    # 轮前 L2 flush 证据（msprof 模式，l2_flush.py）
    flush_log = round_dir / "flush.log"
    if flush_log.exists():
        for line in flush_log.read_text(errors="ignore").splitlines():
            if "FLUSH_OK" in line:
                metrics["l2_flush"] = line.strip()
                break
    if (round_dir / "prof").exists():
        metrics["msprof_prof_dir"] = str(round_dir / "prof")
        msprof = parse_msprof_round(round_dir)
        if msprof:
            metrics["msprof"] = msprof
    return metrics


# ---------------- msprof op_summary 解析（B9⑤-5） ----------------


def _to_float(v):
    try:
        s = str(v).strip()
        return float(s) if s else None
    except (TypeError, ValueError):
        return None


def _find_op_summaries(round_dir):
    """prof/ 下全部 op_summary_*.csv（950 实测：单 PROF_* 目录内含
    device_0/device_1 子目录，op_summary 单文件覆盖全部设备、行级
    Device_id 列区分 rank）。"""
    prof_root = round_dir / "prof"
    if not prof_root.exists():
        return []
    return sorted(prof_root.glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))


def _pick_col(fieldnames, *patterns):
    for pat in patterns:
        for c in fieldnames:
            if re.fullmatch(pat, c.strip()):
                return c
    return None


def _pick_cols(fieldnames, patterns):
    cols = []
    for pat in patterns:
        for c in fieldnames:
            if re.fullmatch(pat, c.strip()) and c not in cols:
                cols.append(c)
    return cols


def _norm_name(name):
    return name.lower().replace("_", "")


def parse_msprof_round(round_dir):
    """解析一轮 msprof 归档 → 主 kernel / AI_CPU 通信 kernel 逐 rank 记录。

    返回 {"main": [...], "aicpu_comm": [...], "columns": [...],
    "op_counts": {...}, "dimension_columns": {...}}；无主 kernel 记录返回
    None。rank 归属：Device_id 列（950 实测列名），缺失时 slot{i} 回退。
    """
    csvs = _find_op_summaries(round_dir)
    if not csvs:
        return None
    main, comm = [], []
    columns = None
    op_counts = {}
    dim_columns = None
    for slot, path in enumerate(csvs):
        try:
            f = open(path, newline="", encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            if columns is None:
                columns = list(reader.fieldnames)
                dim_columns = {}
                for dim, pats in MSPROF_DIMENSION_COLS.items():
                    dim_columns[dim] = _pick_cols(reader.fieldnames, pats)
                dim_columns["_aiv"] = _pick_cols(reader.fieldnames, MSPROF_AIV_COLS)
                dim_columns["_extra"] = _pick_cols(reader.fieldnames, MSPROF_EXTRA_COLS)
            dur_col = _pick_col(
                reader.fieldnames,
                r"Task Duration\(us\)",
                r"(?i).*duration.*us.*",
                r"(?i).*duration.*",
            )
            dev_col = _pick_col(reader.fieldnames, r"(?i)device[_ ]?id", r"(?i)device")
            name_col = _pick_col(reader.fieldnames, r"Op Name", r"Kernel Name", r"Name")
            type_col = _pick_col(reader.fieldnames, r"Task Type")
            for row in reader:
                name = (row.get(name_col) or "").strip() if name_col else ""
                if name:
                    op_counts[name] = op_counts.get(name, 0) + 1
                norm = _norm_name(name)
                if MAIN_KERNEL_TOKEN not in norm:
                    continue
                task_type = (
                    (row.get(type_col) or "").strip() if type_col else ""
                ).lower()
                dur = _to_float(row.get(dur_col)) if dur_col else None
                dev = (row.get(dev_col) or "").strip() if dev_col else ""
                rec = {
                    "rank_slot": slot,
                    "device": dev if dev and dev != "N/A" else f"slot{slot}",
                    "op_name": name,
                    "task_type": task_type,
                    "duration_us": dur,
                }
                dims = {}
                if dim_columns:
                    for dim, cols in dim_columns.items():
                        vals = [_to_float(row.get(c)) for c in cols]
                        vals = [v for v in vals if v is not None]
                        if vals:
                            dims[dim] = sum(vals)
                rec["dimensions"] = dims
                if task_type in MAIN_TASK_TYPES:
                    main.append(rec)
                elif task_type == "ai_cpu":
                    comm.append(rec)
    if not main:
        return None
    return {
        "main": main,
        "aicpu_comm": comm,
        "columns": columns,
        "op_counts": op_counts,
        "dimension_columns": dim_columns,
    }


def _robust_stats(vals):
    """去离群（|v-median|>K×MAD）后 median/min/max/MAD/CV；剔除数登记。"""
    if not vals:
        return None
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    kept = (
        vals if mad == 0 else [v for v in vals if abs(v - med) <= OUTLIER_MAD_K * mad]
    )
    if not kept:
        kept = vals
    kmed = statistics.median(kept)
    kmean = statistics.fmean(kept)
    return {
        "rounds_total": len(vals),
        "rounds_kept": len(kept),
        "outliers_dropped": len(vals) - len(kept),
        "median": kmed,
        "min": min(kept),
        "max": max(kept),
        "mad": statistics.median([abs(v - kmed) for v in kept]),
        "cv": (statistics.pstdev(kept) / kmean) if kmean else None,
    }


def _series_stats(series_map, warmup_drop=True):
    """rank→[逐轮时长] 的 per-rank 统计（去 warmup + 去离群 + last-N）。"""
    per_rank = {}
    for dev, series in series_map.items():
        eff = (
            series[WARMUP_DROP_ROUNDS:]
            if warmup_drop and len(series) > WARMUP_DROP_ROUNDS
            else series
        )
        st = _robust_stats(eff)
        if st is None:
            continue
        lastn = eff[-OFFICIAL_LAST_N:] if len(eff) >= OFFICIAL_LAST_N else eff
        st[f"last{OFFICIAL_LAST_N}_mean"] = statistics.fmean(lastn) if lastn else None
        per_rank[dev] = st
    return per_rank


def _aggregate_msprof(rounds):
    """跨轮聚合 msprof 主 kernel / AI_CPU 通信 kernel 数据（口径见
    模块 docstring）。"""
    msprof_rounds = [r for r in rounds if r.get("msprof")]
    if not msprof_rounds:
        return None
    out = {
        "rounds_with_msprof": len(msprof_rounds),
        "caliber": {
            "warmup_drop_rounds": WARMUP_DROP_ROUNDS,
            "outlier_rule": f"|v-median|>{OUTLIER_MAD_K}xMAD (MAD=0 keep all)",
            "per_rank_stat": "median",
            "cross_rank_primary": "max(per-rank median)  # 官方口径：木桶效应",
            "cross_rank_refs": [
                "mean(per-rank median)  # 卡均值",
                "max(per-rank last-%d mean)  # msprof-guide 官方字面口径"
                % OFFICIAL_LAST_N,
            ],
        },
    }
    flush_lines = [r.get("l2_flush") for r in msprof_rounds if r.get("l2_flush")]
    out["l2_flush_rounds"] = len(flush_lines)

    # 主 kernel 逐 rank 序列 + 每轮 max（回退口径）
    rank_series = {}
    dim_series = {}
    round_max_series = []
    dim_columns = None
    for r in msprof_rounds:
        recs = r["msprof"]["main"]
        dim_columns = r["msprof"]["dimension_columns"]
        durs = [rec["duration_us"] for rec in recs if rec["duration_us"] is not None]
        if durs:
            round_max_series.append(max(durs))
        for rec in recs:
            if rec["duration_us"] is None:
                continue
            rank_series.setdefault(rec["device"], []).append(rec["duration_us"])
            if any(v is not None for v in rec["dimensions"].values()):
                dim_series.setdefault(rec["device"], []).append(rec["dimensions"])

    out["rank_attribution"] = len(rank_series) > 1 or None
    per_rank = _series_stats(rank_series)
    out["per_rank"] = per_rank or None
    if per_rank:
        medians = [v["median"] for v in per_rank.values()]
        out["overall_max_of_medians_us"] = max(medians)
        out["overall_mean_of_medians_us"] = statistics.fmean(medians)
        lastns = [
            v[f"last{OFFICIAL_LAST_N}_mean"]
            for v in per_rank.values()
            if v.get(f"last{OFFICIAL_LAST_N}_mean") is not None
        ]
        if lastns:
            out["overall_max_of_lastN_means_us"] = max(lastns)
        # 最慢 rank（整体性能决定者）
        out["slowest_rank"] = max(per_rank, key=lambda d: per_rank[d]["median"])
    if round_max_series:
        eff = (
            round_max_series[WARMUP_DROP_ROUNDS:]
            if len(round_max_series) > WARMUP_DROP_ROUNDS
            else round_max_series
        )
        out["round_max_series_stats"] = _robust_stats(eff)

    # AI_CPU 通信侧 kernel（信息性：HCCL server 执行时长）
    comm_series = {}
    for r in msprof_rounds:
        for rec in r["msprof"]["aicpu_comm"]:
            if rec["duration_us"] is None:
                continue
            comm_series.setdefault(rec["device"], []).append(rec["duration_us"])
    if comm_series:
        out["aicpu_comm_per_rank"] = _series_stats(comm_series) or None

    # 维度分解：per-rank 各维中位数（us）→ per-rank 归一占比；
    # 汇总口径 = 最慢 rank 的占比（整体性能决定者的瓶颈结构）
    if dim_columns:
        out["dimension_columns"] = dim_columns
        if any(cols for k, cols in dim_columns.items() if not k.startswith("_")):
            dims_merged = {}
            for dev, series in dim_series.items():
                per_dim = {}
                for dim in MSPROF_DIMENSION_COLS:
                    vals = [s.get(dim) for s in series if s.get(dim) is not None]
                    if vals:
                        per_dim[dim] = statistics.median(vals)
                if per_dim:
                    dims_merged[dev] = per_dim
            if dims_merged:
                out["dimension_util_median"] = dims_merged
                for dev, dims in dims_merged.items():
                    total = sum(dims.values())
                    if total > 0:
                        dims_merged[dev] = dict(dims)
                        dims_merged[dev]["_share"] = {
                            k: round(v / total, 6) for k, v in dims.items()
                        }
                slow = out.get("slowest_rank")
                if slow and slow in dims_merged and "_share" in dims_merged[slow]:
                    shares = dims_merged[slow]["_share"]
                    out["dimension_share"] = shares
                    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
                    out["bottleneck_ranking"] = [
                        {"dimension": k, "share": v} for k, v in ranked
                    ]
                    out["bottleneck_dimension"] = ranked[0][0]
    # op 名清单（提取方法留痕：主 kernel 过滤与非主 kernel 噪声一览）
    op_counts = {}
    for r in msprof_rounds:
        for name, cnt in r["msprof"]["op_counts"].items():
            op_counts[name] = op_counts.get(name, 0) + cnt
    out["op_name_counts"] = op_counts
    out["main_kernel_filter"] = (
        f"normalized(Op Name) contains '{MAIN_KERNEL_TOKEN}' and "
        f"Task Type in {sorted(t for t in MAIN_TASK_TYPES if t)}"
    )
    return out


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
    # 维度占比：逐轮归一化后取中位数（cannsim 路径）
    shares = {}
    for dim in DIMENSION_PIPES:
        series = [
            r.get("dimensions_share", {}).get(dim)
            for r in rounds
            if r.get("dimensions_share")
        ]
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
        series = [
            r.get("dimensions_util", {}).get(dim)
            for r in rounds
            if r.get("dimensions_util")
        ]
        series = [s for s in series if s is not None]
        if series:
            utils[dim] = round(statistics.median(series), 6)
    agg["dimension_util_median"] = utils or None
    # 周期与计时
    agg["kernel_total_clocks"] = _stats(_series("kernel_total_clocks"))
    timing = [us for r in rounds for us in r.get("example_timing_us", [])]
    agg["example_timing_us"] = _stats(timing)
    # msprof 聚合（B9⑤-5）
    msprof_agg = _aggregate_msprof(rounds)
    if msprof_agg:
        agg["msprof"] = msprof_agg
    # 诊断
    dominants = [
        r.get("dominant_pipeline") for r in rounds if r.get("dominant_pipeline")
    ]
    if dominants:
        agg["dominant_pipeline_rounds"] = {
            str(d): dominants.count(d) for d in set(dominants)
        }
    # 瓶颈维度判定（cannsim 路径，按占比中位数排序）
    if shares:
        ranked = sorted(
            shares.items(), key=lambda kv: kv[1]["median_share"], reverse=True
        )
        agg["bottleneck_ranking_sim"] = [{"dimension": k, **v} for k, v in ranked]
        if "bottleneck_dimension" not in agg:
            agg["bottleneck_dimension"] = ranked[0][0]
    return agg


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def render_markdown(summary, out_path):
    lines = [
        "# AllGatherMatmulV2 perf report",
        "",
        "> 瓶颈维度口径：compute=aic_mac / copy_in=aic_mte2+mte1 / "
        "copy_out=aic_mte3+fixpipe / scalar=aic_scalar（msprof "
        "op_summary PipeUtilization 时间列，950 实测列名；cannsim 路径 "
        "AIC_CUBE/MTE2+MTE1/FIXPIPE/SCALAR）；AIV 侧单独记录",
        "",
        "> msprof 多轮口径：drop round_1(warmup) → 3×MAD 去离群 → "
        "per-rank median → 跨 rank max（官方口径，木桶效应）；"
        "卡均值与 last-5 mean 作参考列",
        "",
    ]
    for case, agg in sorted(summary.items()):
        if agg is None:
            lines += [f"## {case}", "", "- （无有效轮次数据）", ""]
            continue
        lines += [f"## {case}", "", f"- rounds collected: {agg['rounds_collected']}"]
        if agg.get("bottleneck_ranking"):
            lines += [
                "- 瓶颈维度分解·cannsim（占比中位数）：",
                "",
                "| 维度 | median | min | max |",
                "|---|---|---|---|",
            ]
            for item in agg["bottleneck_ranking"]:
                lines.append(
                    f"| {item['dimension']} | {item['median_share']:.4f} | "
                    f"{item['min_share']:.4f} | {item['max_share']:.4f} |"
                )
            lines += [
                f"- bottleneck dimension (sim): {agg.get('bottleneck_dimension')}"
            ]
        if agg.get("dominant_pipeline_rounds"):
            lines += [
                f"- dominant pipeline (per round): {agg['dominant_pipeline_rounds']}"
            ]
        if agg.get("kernel_total_clocks"):
            s = agg["kernel_total_clocks"]
            lines += [
                f"- kernel_total_clocks: median={s['median']} "
                f"min={s['min']} max={s['max']} mad={s['mad']} "
                f"cv={_fmt(s['cv'])}"
            ]
        if agg.get("example_timing_us"):
            s = agg["example_timing_us"]
            lines += [
                f"- example timing (us): median={_fmt(s['median'])} "
                f"min={_fmt(s['min'])} max={_fmt(s['max'])} "
                f"mad={_fmt(s['mad'])} cv={_fmt(s['cv'])}"
                f"（n={s['rounds']}）"
            ]
        msprof = agg.get("msprof")
        if msprof:
            lines += [
                "",
                "**msprof 主 kernel（B9⑤-5）**",
                f"- rounds with msprof: {msprof['rounds_with_msprof']}"
                f"（L2 flush 生效轮数: {msprof.get('l2_flush_rounds', 0)}）",
                f"- main kernel filter: `{msprof['main_kernel_filter']}`",
            ]
            if msprof.get("per_rank"):
                lines += [
                    "",
                    "| rank | median(us) | min | max | MAD | CV | "
                    "kept/total | last-5 mean |",
                    "|---|---|---|---|---|---|---|---|",
                ]
                for dev, st in sorted(msprof["per_rank"].items()):
                    lines.append(
                        f"| {dev} | {_fmt(st['median'])} | {_fmt(st['min'])} "
                        f"| {_fmt(st['max'])} | {_fmt(st['mad'])} | "
                        f"{_fmt(st['cv'])} | {st['rounds_kept']}/"
                        f"{st['rounds_total']} | "
                        f"{_fmt(st.get('last5_mean'))} |"
                    )
            if msprof.get("overall_max_of_medians_us") is not None:
                lines += [
                    f"- **overall (max of per-rank medians): "
                    f"{_fmt(msprof['overall_max_of_medians_us'])} us"
                    f"（官方口径·主口径）**",
                    f"- overall (mean of per-rank medians / 卡均值): "
                    f"{_fmt(msprof.get('overall_mean_of_medians_us'))} us",
                    f"- overall (max of per-rank last-5 means / 官方字面): "
                    f"{_fmt(msprof.get('overall_max_of_lastN_means_us'))} us",
                ]
            if msprof.get("aicpu_comm_per_rank"):
                lines += [
                    "- AI_CPU 通信侧 kernel (AllGatherMatmulV2Mc2AicpuKernel, 信息性)："
                ]
                for dev, st in sorted(msprof["aicpu_comm_per_rank"].items()):
                    lines.append(
                        f"  - {dev}: median={_fmt(st['median'])} us "
                        f"(min={_fmt(st['min'])} max={_fmt(st['max'])})"
                    )
            if msprof.get("round_max_series_stats"):
                st = msprof["round_max_series_stats"]
                lines += [
                    f"- round-max fallback: median={_fmt(st['median'])} "
                    f"min={_fmt(st['min'])} max={_fmt(st['max'])} "
                    f"(kept {st['rounds_kept']}/{st['rounds_total']})"
                ]
            if msprof.get("bottleneck_ranking"):
                lines += [
                    "",
                    f"- 瓶颈维度分解·msprof（最慢 rank "
                    f"{msprof.get('slowest_rank')} 归一占比）：",
                    "",
                    "| 维度 | share |",
                    "|---|---|",
                ]
                for item in msprof["bottleneck_ranking"]:
                    lines.append(f"| {item['dimension']} | {item['share']:.4f} |")
                lines += [
                    f"- bottleneck dimension (msprof): "
                    f"{msprof.get('bottleneck_dimension')}"
                ]
            if msprof.get("dimension_util_median"):
                lines += ["- 维度原始值（per-rank 中位数, us）："]
                for dev, dims in sorted(msprof["dimension_util_median"].items()):
                    d = ", ".join(
                        f"{k}={_fmt(v)}"
                        for k, v in sorted(dims.items())
                        if not k.startswith("_")
                    )
                    lines.append(f"  - {dev}: {d}")
        lines += [""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", required=True, help="collect_perf.sh 的 perf_raw 目录"
    )
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 raw-root）")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir) if args.out_dir else raw_root
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for case_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if case_dir.name == "unused_dump":
            continue  # collect_perf.sh 的 sim_cases dump-root 占位目录（非用例）
        # 分支层发现（4.1 双分支布局）：case/branch_{v1|apace}/round_* → 键 case[branch]；
        # 无分支层时按旧布局（case/round_*）聚合，键为 case。
        branch_dirs = [
            b
            for b in sorted(case_dir.glob("branch_*"))
            if b.is_dir() and any(b.glob("round_*"))
        ]
        if branch_dirs:
            for b in branch_dirs:
                key = "{}[{}]".format(case_dir.name, b.name[len("branch_") :])
                summary[key] = aggregate(b)
        else:
            summary[case_dir.name] = aggregate(case_dir)

    with open(out_dir / "perf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    render_markdown(summary, out_dir / "perf_report.md")
    print(
        f"[perf_report] {len(summary)} cases -> "
        f"{out_dir / 'perf_summary.json'} / {out_dir / 'perf_report.md'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

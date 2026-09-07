from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .common import (
        find_column,
        iter_named_files,
        parse_number,
        percentile,
        read_csv_rows,
        resolve_frequency_mhz,
        write_csv,
    )
except ImportError:  # direct script execution
    from common import (
        find_column,
        iter_named_files,
        parse_number,
        percentile,
        read_csv_rows,
        resolve_frequency_mhz,
        write_csv,
    )


MOE_OPS = (
    "MoeEpDispatch",
    "MoeEpDispatchEpilogue",
    "MoeEpCombine",
    "MoeEpCombineEpilogue",
)
PIPELINES = ("mte1", "mte2", "mte3", "vector", "cube", "scalar", "fixpipe")


def _is_moe_row(row: Mapping[str, str]) -> bool:
    key = find_column(row.keys(), ["Op Type", "OP Type", "Op Name", "Name"])
    if key is None:
        return True
    value = row.get(key, "")
    return any(value.startswith(op) for op in MOE_OPS)


def _op_name(row: Mapping[str, str], source: Path) -> str:
    key = find_column(row.keys(), ["Op Type", "OP Type", "Op Name", "Name"])
    if key is not None and row.get(key):
        value = row[key]
        for op in sorted(MOE_OPS, key=len, reverse=True):
            if value.startswith(op):
                return op
        return value
    parent_text = str(source.parent)
    for op in MOE_OPS:
        if op.lower() in parent_text.lower():
            return op
    return "unknown"


def _find_metric(
    row: Mapping[str, str], role: str, pipeline: str, suffix: str
) -> float | None:
    """Read exact PipeUtilization fields and avoid similarly named wait/conflict ratios."""
    pipeline_name = "vec" if pipeline == "vector" else pipeline
    expected = f"{role}{pipeline_name}{suffix}"
    alternatives = {expected}
    if suffix == "activebwgbps":
        alternatives.add(f"{role}{pipeline_name}activebwgbs")
    for key, raw_value in row.items():
        normalized = "".join(ch for ch in key.lower() if ch.isalnum())
        if normalized in alternatives:
            return parse_number(raw_value)
    return None


def collect_metrics(input_root: Path) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    pipe_files = list(iter_named_files(input_root, ["PipeUtilization"]))
    for source in pipe_files:
        try:
            rows = read_csv_rows(source)
        except (OSError, UnicodeError, csv.Error):
            continue
        for row_index, row in enumerate(rows):
            if not _is_moe_row(row):
                continue
            block_key = find_column(row.keys(), ["block_id", "Block ID", "Block Id"])
            sub_key = find_column(
                row.keys(), ["sub_block_id", "Sub Block ID", "Core Name"]
            )
            base = {
                "source_file": str(source),
                "source_row": row_index + 2,
                "op": _op_name(row, source),
                "block_id": row.get(block_key, "") if block_key else "",
                "sub_block_id": row.get(sub_key, "") if sub_key else "",
            }
            for role in ("aiv", "aic"):
                for pipeline in PIPELINES:
                    output = dict(base)
                    output["role"] = role.upper()
                    output["pipeline"] = pipeline
                    output["time_us"] = _find_metric(row, role, pipeline, "timeus")
                    output["ratio"] = _find_metric(row, role, pipeline, "ratio")
                    output["active_bw_gbps"] = _find_metric(
                        row, role, pipeline, "activebwgbps"
                    )
                    if (
                        output["time_us"] is not None
                        or output["ratio"] is not None
                        or output["active_bw_gbps"] is not None
                    ):
                        rows_out.append(output)
    return rows_out


def summarize_metrics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["op"]), str(row["role"]), str(row["pipeline"]))].append(row)
    summary: list[dict[str, object]] = []
    for (op, role, pipeline), items in sorted(grouped.items()):
        times = [
            float(item["time_us"]) for item in items if item.get("time_us") is not None
        ]
        ratios = [
            float(item["ratio"]) for item in items if item.get("ratio") is not None
        ]
        bandwidths = [
            float(item["active_bw_gbps"])
            for item in items
            if item.get("active_bw_gbps") is not None
        ]
        summary.append(
            {
                "op": op,
                "role": role,
                "pipeline": pipeline,
                "samples": len(items),
                "time_us_min": min(times) if times else "",
                "time_us_p50": percentile(times, 0.5) if times else "",
                "time_us_mean": statistics.fmean(times) if times else "",
                "time_us_max": max(times) if times else "",
                "ratio_p50": percentile(ratios, 0.5) if ratios else "",
                "ratio_mean": statistics.fmean(ratios) if ratios else "",
                "active_bw_gbps_p50": percentile(bandwidths, 0.5) if bandwidths else "",
            }
        )
    return summary


def archive_raw(input_root: Path, raw_dir: Path) -> list[str]:
    allowed_names = {
        "OpBasicInfo.csv",
        "PipeUtilization.csv",
        "ArithmeticUtilization.csv",
        "Memory.csv",
        "MemoryL0.csv",
        "MemoryUB.csv",
        "L2Cache.csv",
        "ResourceConflictRatio.csv",
        "visualize_data.bin",
    }
    raw_dir = raw_dir.resolve()
    sources = [
        source
        for source in input_root.rglob("*")
        if source.is_file()
        and source.name in allowed_names
        and raw_dir not in source.resolve().parents
    ]
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sources:
        relative = source.relative_to(input_root)
        target = raw_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def _write_svg(path: Path, summary: Sequence[Mapping[str, object]]) -> None:
    rows = [row for row in summary if row.get("ratio_p50") not in (None, "")]
    width, left, row_h = 1100, 260, 28
    height = max(100, 45 + row_h * len(rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font:12px sans-serif}.label{fill:#222}.bar{fill:#377eb8}.grid{stroke:#ddd}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in range(0, 101, 20):
        x = left + tick * 8
        parts.append(
            f'<line class="grid" x1="{x}" y1="20" x2="{x}" y2="{height - 10}"/>'
        )
        parts.append(f'<text x="{x}" y="15" text-anchor="middle">{tick}%</text>')
    for index, row in enumerate(rows):
        y = 30 + index * row_h
        ratio = max(0.0, min(100.0, float(row["ratio_p50"])))
        label = html.escape(f"{row['op']} / {row['role']} / {row['pipeline']}")
        parts.append(f'<text class="label" x="8" y="{y + 15}">{label}</text>')
        parts.append(
            f'<rect class="bar" x="{left}" y="{y + 3}" width="{ratio * 8:.2f}" height="17"/>'
        )
        parts.append(
            f'<text x="{left + ratio * 8 + 5:.2f}" y="{y + 16}">{ratio:.2f}%</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_report(
    path: Path, summary: Sequence[Mapping[str, object]], frequency: Mapping[str, object]
) -> None:
    lines = [
        "# MoeEp msprof 流水报告",
        "",
        f"频率：{float(frequency['mhz']):.3f} MHz；来源：`{frequency['source']}`。",
        "",
        "该报告的 MTE2/MTE3/Vector/Scalar 数据来自硬件 msprof CSV，不使用 GetSystemCycle 区间推断。",
        "",
        "## 各算子流水中位占比",
        "",
        "| 算子 | 物理角色 | 流水 | p50 ratio | p50 time(us) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        ratio = row.get("ratio_p50", "")
        time_us = row.get("time_us_p50", "")
        ratio_text = "-" if ratio == "" else f"{float(ratio):.3f}%"
        time_text = "-" if time_us == "" else f"{float(time_us):.3f}"
        lines.append(
            f"| {row['op']} | {row['role']} | {row['pipeline']} | {ratio_text} | {time_text} |"
        )
    dispatch_rows = [
        row
        for row in summary
        if row["op"] == "MoeEpDispatch"
        and row["pipeline"] in {"mte2", "mte3", "vector", "scalar"}
    ]
    lines.extend(["", "## SendPhase 重点", ""])
    if dispatch_rows:
        lines.append(
            "`SendPhase` 位于 MoeEpDispatch 主 kernel 内。msprof 的硬件流水计数是 kernel/core 粒度；下列数值是 Dispatch 主 kernel 的真实流水数据，精确的 SendPhase 函数边界需与 cycle-profiler 的 `SEND_PHASE_BEGIN/END` 区间联合查看："
        )
        lines.append("")
        for row in dispatch_rows:
            ratio = row.get("ratio_p50", "")
            ratio_text = "N/A" if ratio == "" else f"{float(ratio):.3f}%"
            lines.append(f"- {row['role']}/{row['pipeline']}: p50 ratio {ratio_text}")
    else:
        lines.append(
            "未在 PipeUtilization.csv 中识别到 MoeEpDispatch 行；请保留原始 CSV 并检查该 CANN 版本的列名。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "不同 rank 的 GetSystemCycle 计数器不保证同一 epoch；跨 rank 图仅使用各 rank 自身相对时间，不宣称绝对对齐。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    input_root: Path,
    output_dir: Path,
    frequency_mhz_override: float | None = None,
) -> dict[str, object]:
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(input_root)
    summary = summarize_metrics(metrics)
    if frequency_mhz_override is not None and frequency_mhz_override > 0:
        frequency = {
            "mhz": float(frequency_mhz_override),
            "source": "cli --frequency-mhz override",
            "source_files": [],
            "observed_mhz": [float(frequency_mhz_override)],
        }
    else:
        frequency = resolve_frequency_mhz(input_root)
    copied = archive_raw(input_root, output_dir / "raw_msprof")
    metric_fields = [
        "source_file",
        "source_row",
        "op",
        "block_id",
        "sub_block_id",
        "role",
        "pipeline",
        "time_us",
        "ratio",
        "active_bw_gbps",
    ]
    summary_fields = [
        "op",
        "role",
        "pipeline",
        "samples",
        "time_us_min",
        "time_us_p50",
        "time_us_mean",
        "time_us_max",
        "ratio_p50",
        "ratio_mean",
        "active_bw_gbps_p50",
    ]
    write_csv(output_dir / "pipeline_metrics.csv", metric_fields, metrics)
    write_csv(output_dir / "msprof_summary.csv", summary_fields, summary)
    _write_svg(output_dir / "msprof_pipeline.svg", summary)
    _write_report(output_dir / "msprof_report.md", summary, frequency)
    manifest = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "frequency": frequency,
        "raw_files": copied,
        "metrics_rows": len(metrics),
    }
    (output_dir / "msprof_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive and analyze MoeEp msprof op output"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="OPPROF directory or parent containing OPPROF directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="analysis directory (default: <input>/moe_ep_analysis)",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        help="override frequency in MHz; skips OpBasicInfo/platform_config/fallback",
    )
    args = parser.parse_args(argv)
    output = args.output or args.input / "moe_ep_analysis"
    manifest = analyze(args.input, output, args.frequency_mhz)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

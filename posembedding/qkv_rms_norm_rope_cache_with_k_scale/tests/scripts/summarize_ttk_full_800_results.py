#!/usr/bin/env python3
"""Summarize one or more TTK result CSVs into a reproducible Markdown report."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
from pathlib import Path
import re


METRIC = re.compile(r"([a-zA-Z0-9_]+)=([0-9]+(?:\.[0-9]+)?)%")
PADS = re.compile(r"pads\(k,v,scale\)=\((\d+),\s*(\d+),\s*(\d+)\)")


def read_rows(paths):
    rows = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows[row["testcase_name"]] = row
    return list(rows.values())


def scenario(row):
    attrs = ast.literal_eval(row.get("attributes") or "{}")
    mode = "M-RoPE" if attrs.get("qQuantMode", attrs.get("q_quant_mode")) == "NoQuant" else "RoPE"
    qkv = attrs.get("layoutQkv", attrs.get("layout_qkv"))
    qout = attrs.get("layoutQOut", attrs.get("layout_q_out"))
    return f"{mode} {qkv}->{qout}"


def precision_text(row):
    return row.get("precision") or row.get("eager_precision") or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.cases.open(newline="", encoding="utf-8") as stream:
        case_reader = csv.DictReader(stream)
        fieldnames = case_reader.fieldnames or []
        cases = list(case_reader)
    rows = read_rows(args.results)
    status = Counter(row.get("precision_status") or "UNKNOWN" for row in rows)
    scenarios = Counter(scenario(row) for row in rows)
    pads = Counter()
    for case in cases:
        match = PADS.search(case.get("remark", ""))
        if match:
            pads[tuple(map(int, match.groups()))] += 1

    minima = {}
    minimum_cases = {}
    for row in rows:
        for name, value in METRIC.findall(precision_text(row)):
            number = float(value)
            if name not in minima or number < minima[name]:
                minima[name] = number
                minimum_cases[name] = row["testcase_name"]

    complete = len(rows) == len(cases)
    lines = [
        "# QkvRmsNormRopeCacheWithKScale TTK 随机 stride 精度报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 结论",
        "",
        f"- CSV 语义用例：{len(cases)} 条。",
        f"- 已执行并去重：{len(rows)} 条；PASS={status['PASS']}，FAIL={status['FAIL']}。",
        f"- 回归状态：{'全量完成' if complete else '阶段性结果，尚未覆盖全部 CSV 用例'}。",
        f"- offset：{'错误：CSV 含 offset 字段' if 'tensor_view_offsets' in fieldnames else '未配置，仅测试 stride'}。",
        "- K/V 使用相同的随机行 padding，取值集合为 {8,16,24,32}；scale 独立随机。",
        "",
        "## 精度契约",
        "",
        "- BF16：单点 `rtol=0.001`，超差点比例不超过 `0.001`。",
        "- FP8：单点 `rtol=0.01`，超差点比例不超过 `0.01`。",
        "- INT8：所有点最大绝对误差 `<=1`。",
        "- Scale FP32：单点 `rtol=0.001`，超差点比例不超过 `0.001`。",
        "- 未命中 cache slot：逐字节不变。",
        "",
        "## 已执行场景",
        "",
        "| 场景 | 数量 |",
        "|:--|--:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(scenarios.items()))
    lines += [
        "",
        "## 已执行最低命中率",
        "",
        "| 指标 | 最低命中率 | 用例 |",
        "|:--|--:|:--|",
    ]
    lines.extend(
        f"| `{name}` | {value:.6f}% | `{minimum_cases[name]}` |"
        for name, value in sorted(minima.items())
    )
    failures = [row for row in rows if row.get("precision_status") != "PASS"]
    lines += ["", "## 失败项", ""]
    if failures:
        for row in failures:
            lines.append(f"- `{row['testcase_name']}`：{precision_text(row)}")
    else:
        lines.append("无。")
    lines += [
        "",
        "## CSV stride 分布",
        "",
        "| (K pad, V pad, scale pad) | 用例数 |",
        "|:--|--:|",
    ]
    lines.extend(f"| `{key}` | {count} |" for key, count in sorted(pads.items()))
    lines += [
        "",
        "## 原始结果",
        "",
    ]
    lines.extend(f"- `{path}`" for path in args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

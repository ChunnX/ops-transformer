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
# Description: run_sim_cases.sh 逐例汇总辅助（B9⑤ P2 处置 2026-09-20）。
#  - 从 compare_dump.py 报告提取关键指标（maxAbsErr / matched_ratio / gatherOut
#    bitwise / 确定性）与失败原因摘要；
#  - 合并 example 退出状态与权威比对结论，产出每例一行 driver_summary.jsonl
#    （失败例 summary 补全：用例号 / 判定 / 关键指标 / 失败原因摘要）；
#  - informational 用例（sim_cases.py INFO_CASES，AGM_V2_INFORMATIONAL=1 透传）
#    判定标记 INFO_ 前缀，不计入驱动级 PASS/FAIL 门禁。
# Usage: driver_case_summary.py --case-id L0-01 --informational 0 \
#           --example-rc 0 --compare-rc 0 \
#           --report <report.json> [--summary-out <driver_summary.jsonl>]
#   compare-rc: 0=PASS 1=FAIL 2=未比对（dump 不完整/崩溃未落盘）
# 输出（stdout 单行，供驱动日志展示）：<case> | <VERDICT> | <metrics> | <reason>
# ----------------------------------------------------------------------------
import argparse
import json
from pathlib import Path


def brief_from_report(report):
    """(metrics_brief, failure_reasons) from a compare_dump report dict."""
    metrics = []
    reasons = []
    for t in report.get("tensors", []):
        if t.get("role") == "crosscheck":
            continue  # 交叉核对不作通过依据，不进失败原因
        m = t.get("measured", {}) or {}
        rank = t.get("rank", "-")
        if t.get("tensor") == "output" and t.get("metric") == "mixed_tolerance":
            metrics.append(
                f"output[r{rank}]:maxAbsErr={m.get('max_abs_error')},"
                f"matched={m.get('matched_ratio')}"
            )
            if t.get("conclusion") == "FAIL":
                th = t.get("threshold", {}) or {}
                if (m.get("max_abs_error") or 0.0) > th.get(
                    "max_abs_error_limit", float("inf")
                ):
                    reasons.append(
                        f"output[r{rank}] max_abs_error {m.get('max_abs_error')}"
                        f" > {th.get('max_abs_error_limit')}"
                    )
                if (m.get("matched_ratio") or 0.0) < th.get("matched_ratio_min", 1.0):
                    reasons.append(
                        f"output[r{rank}] matched_ratio {m.get('matched_ratio')}"
                        f" < {th.get('matched_ratio_min')}"
                    )
        elif t.get("tensor") == "gather_out":
            metrics.append(
                f"gather[r{rank}]:{t.get('conclusion')}({m.get('mismatched_bytes')}B)"
            )
            if t.get("conclusion") == "FAIL":
                reasons.append(
                    f"gather_out[r{rank}] mismatched_bytes "
                    f"{m.get('mismatched_bytes')} > 0"
                )
        elif t.get("metric") == "determinism_bitwise":
            metrics.append(f"determinism:{t.get('conclusion')}")
            if t.get("conclusion") == "FAIL":
                reasons.append(f"determinism {json.dumps(m, ensure_ascii=False)}")
        elif t.get("metric") == "special_value_propagation":
            metrics.append(f"special:{t.get('conclusion')}")
    return ("; ".join(metrics) if metrics else "metrics=n/a", "; ".join(reasons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--informational", required=True, choices=["0", "1"])
    parser.add_argument("--example-rc", required=True, choices=["0", "1"])
    parser.add_argument("--compare-rc", required=True, choices=["0", "1", "2"])
    parser.add_argument(
        "--report",
        default=None,
        help="compare_dump 报告 json 路径（compare-rc<2 时存在）",
    )
    parser.add_argument(
        "--summary-out", default=None, help="driver_summary.jsonl 追加路径"
    )
    args = parser.parse_args()

    informational = args.informational == "1"
    example_ok = args.example_rc == "0"
    compare_rc = int(args.compare_rc)

    report = None
    if args.report and Path(args.report).exists():
        try:
            with open(args.report, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as exc:
            metrics_brief, reasons = "metrics=n/a", f"report unreadable: {exc}"
        else:
            metrics_brief, reasons = brief_from_report(report)
    else:
        metrics_brief, reasons = "metrics=n/a", ""

    if compare_rc == 2:
        verdict = "FAIL"
        reason = (
            "example exited non-zero without complete dump "
            "(crash/launch failure before kernel output)"
        )
    elif compare_rc == 0:
        verdict = "PASS"
        reason = reasons
        if not example_ok:
            reason = (
                "example_internal_check_failed but authoritative "
                "compare_dump PASS (two-golden divergence, see run log)"
                + (f"; {reasons}" if reasons else "")
            )
    else:
        verdict = "FAIL"
        reason = reasons or "compare_dump FAIL (see report)"

    final_verdict = ("INFO_" + verdict) if informational else verdict

    line = {
        "case_id": args.case_id,
        "informational": informational,
        "example_exit_ok": example_ok,
        "compare_rc": compare_rc,
        "compare_pass": (compare_rc == 0),
        "verdict": final_verdict,
        "reason": reason,
        "metrics": metrics_brief,
    }
    if args.summary_out:
        with open(args.summary_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(
        f"[case-summary] {args.case_id} | {final_verdict} | {metrics_brief} | {reason}"
    )
    # 驱动按 verdict 前缀计门禁：INFO_* 不计入 PASS/FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

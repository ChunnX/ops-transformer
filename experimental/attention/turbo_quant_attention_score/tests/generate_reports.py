# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build reviewable XLSX reports and charts from saved hardware test results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, PatternFill

BASELINES = {
    "gqa_decode_kv4k": 1501.634,
    "gqa_decode_kv32k": 9278.507,
    "gqa_decode_kv128k": 35681.690,
    "gqa_prefill_kv4k": 75984.253,
}
TEMPLATE = "https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2"


def sheet(wb, name, headers, rows):
    ws = wb.create_sheet(name)
    ws.append(headers)
    for row in rows:
        ws.append(row)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="19384F")
        cell.font = Font(color="FFFFFF", bold=True)
    for column in ws.columns:
        letter = column[0].column_letter
        ws.column_dimensions[letter].width = min(
            65, max(16, max(len(str(c.value or "")) for c in column) + 2)
        )
        for cell in column:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for row in range(2, ws.max_row + 1):
        ws.row_dimensions[row].height = 55
    return ws


def common(wb):
    wb.remove(wb.active)
    sheet(
        wb,
        "说明",
        ["项目", "内容"],
        [
            ["算子", "TurboQuantAttentionScore"],
            ["环境", "Ascend 910B3 / CANN 9.1.0 / torch 2.9.0 / torch_npu 2.9.0.post6"],
            ["模板来源", TEMPLATE],
            [
                "模板形式",
                "按在线模板的 shape、dtype、精度是否通过、TBE截图、Ascend截图、TBE算子性能(us)、Ascend C算子性能(us)、解释说明 字段重建；增加详细数据页。",
            ],
            [
                "精度结论",
                "仓库FP32标准：atol=2^-16，rtol=2^-10，通过率>=0.99，最大绝对误差<=1e-2。任务附件纯相对解释另列，专项比较器待发布者确认。",
            ],
            [
                "仓库精度标准",
                "opbase/docs/zh/ops_precision_standard/experimental_standard.md，opbase commit 0a83b915a6f6be49a03b907713d6f000b3dea1ea",
            ],
            [
                "性能口径",
                "与任务书给定 Paged Full-QK baseline 比较，非本机重新测得的 FlashAttention；device 为NPU Event，wall为规划/提交/同步耗时。",
            ],
            [
                "显存口径",
                "3.765倍仅指默认Key cache编码；输出和进程整体显存不享有该比例。",
            ],
            [
                "联测",
                "未完成KL/PPL：缺编码器、算子2、模型及数据集；不得据此报告端到端验收通过。",
            ],
            [
                "可复现性",
                "见自验证步骤、测试源码和JSON原始样本；图像由真实结果渲染，不是另行手填的测量值。",
            ],
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--edge", required=True)
    parser.add_argument("--invalid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    full = json.loads(Path(args.full).read_text())
    edge = json.loads(Path(args.edge).read_text())
    invalid = json.loads(Path(args.invalid).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if len(full) != 5 or any(r.get("accuracy_skipped") for r in full):
        raise ValueError("Reports require all five accuracy-tested cases")
    labels = [r["case"].replace("gqa_", "") for r in full]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(labels, [r["device_us_median"] / 1000 for r in full], color="#137C8B")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NPU Event latency (ms, log scale)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, [r["task_golden_max_abs_error"] for r in full], color="#CF7944")
    axes[1].set_ylabel("Maximum absolute error vs supplied golden")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("TurboQuantAttentionScore | Ascend 910B3 | actual aclnn measurements")
    fig.savefig(out / "results.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis("off")
    data = [
        [
            r["case"],
            f"{r['task_golden_max_abs_error']:.3e}",
            str(r["task_golden_pure_relative_exceedances"]),
            f"{r['device_us_median']:.2f}",
            str(r["workspace_bytes"]),
        ]
        for r in full
    ]
    table = ax.table(
        cellText=data,
        colLabels=[
            "Case",
            "max_abs",
            "pure-relative failures",
            "device_us",
            "workspace_B",
        ],
        loc="center",
        cellLoc="left",
        colWidths=[0.27, 0.15, 0.23, 0.17, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title(
        "Recorded test results (rendered from JSON; pure-relative failures remain visible)"
    )
    fig.savefig(out / "test_evidence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    headers = [
        "shape",
        "dtype",
        "精度是否通过",
        "TBE截图",
        "Ascend截图",
        "TBE算子性能(us)",
        "Ascend C算子性能(us)",
        "解释说明",
    ]
    template_rows = []
    for r in full:
        template_rows.append(
            [
                f"T={r['tokens']}, Hq=32, Hkv=8, D=128, L={r['kv_len']}, bits=3",
                "BF16/UINT8/INT32 -> FP32",
                "仓库标准通过；附件口径待确认",
                "无TBE截图，使用任务书Full-QK基线",
                "test_evidence.png",
                BASELINES.get(r["case"], "未提供"),
                r["device_us_median"],
                f"{r['case']}; 原golden纯相对超限 {r['task_golden_pure_relative_exceedances']}/{r['elements']}; 详见精度明细。",
            ]
        )

    wb = Workbook()
    common(wb)
    sheet(wb, "模板字段", headers, template_rows)
    keys = [
        "case",
        "elements",
        "max_abs_error",
        "relative_l2_error",
        "bad_elements",
        "pure_relative_1e3_exceedances",
        "task_golden_max_abs_error",
        "task_golden_mixed_exceedances",
        "task_golden_pure_relative_exceedances",
    ]
    sheet(wb, "精度明细", keys, [[r.get(k, "") for k in keys] for r in full + edge])
    sheet(
        wb,
        "仓库标准判定",
        [
            "case",
            "FP32 atol",
            "FP32 rtol",
            "实际最大绝对误差",
            "已证明的通过率下界",
            "是否通过",
        ],
        [
            [
                r["case"],
                2**-16,
                2**-10,
                r["max_abs_error"],
                1.0 if r["max_abs_error"] <= 2**-16 else None,
                r["max_abs_error"] <= 2**-16,
            ]
            for r in full + edge
        ],
    )
    sheet(
        wb,
        "非法输入",
        ["case", "rejected", "error"],
        [[r[k] for k in ("case", "rejected", "error")] for r in invalid],
    )
    wb.create_sheet("数据可视化").add_image(Image(str(out / "test_evidence.png")), "A1")
    wb.save(out / "2.1 精度自验证报告.xlsx")

    wb = Workbook()
    common(wb)
    sheet(wb, "模板字段", headers, template_rows)
    rows = []
    for r in full:
        baseline = BASELINES.get(r["case"])
        rows.append(
            [
                r["case"],
                r["device_us_median"],
                r["wall_us_median"],
                baseline,
                r["device_us_median"] / baseline if baseline else None,
                "满足给定Decode基线的1.2倍"
                if "decode" in r["case"] and r["device_us_median"] <= 1.2 * baseline
                else "仅记录，非Decode约束",
                json.dumps(r["device_us_samples"]),
            ]
        )
    sheet(
        wb,
        "性能明细",
        [
            "case",
            "device中位数us",
            "wall中位数us",
            "任务书baseline_us",
            "device/baseline",
            "判定",
            "原始样本us",
        ],
        rows,
    )
    wb.create_sheet("数据可视化").add_image(Image(str(out / "results.png")), "A1")
    wb.save(out / "3.1 性能自验证报告.xlsx")

    wb = Workbook()
    common(wb)
    keys = [
        "case",
        "packed_key_bytes",
        "fp16_key_bytes",
        "output_bytes",
        "workspace_bytes",
        "npu_peak_call_bytes",
    ]
    sheet(
        wb,
        "内存明细",
        keys + ["Key压缩比", "Key加输出总内存比"],
        [
            [r.get(k) for k in keys]
            + [
                r["fp16_key_bytes"] / r["packed_key_bytes"],
                (r["fp16_key_bytes"] + r["output_bytes"])
                / (r["packed_key_bytes"] + r["output_bytes"]),
            ]
            for r in full
        ],
    )
    wb.save(out / "4.1 内存自验证报告.xlsx")
    print(f"Generated 3 reports and 2 evidence images in {out}")


if __name__ == "__main__":
    main()

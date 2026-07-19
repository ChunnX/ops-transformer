#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Excel -> SMLA CPU golden -> baseline/derived PT offline acceptance."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

import sparse_flash_mla_golden
import utils
from batch.consistency.offline import mode_options_from_mapping, prepare_saved_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", required=True)
    parser.add_argument("--sheet", default="SMLA")
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["SMLA_CPU_ONLY"] = "1"

    rows = utils.load_excel_test_cases(args.excel, args.sheet)
    combinations = utils.generate_param_combinations(rows, is_save_pt=True)
    baseline_dir = Path(args.pt_dir) / "baseline"
    report_root = Path(args.report_dir)
    reports = []
    for index, combination in enumerate(combinations):
        params = utils.generate_case_with_default_param(combination)
        baseline = sparse_flash_mla_golden.gen_data(params, params.get("template_mode"))
        baseline_path = sparse_flash_mla_golden.save_test_case(baseline, str(baseline_dir))
        case_dir = report_root / f"case_{index:03d}"
        reports.append(
            prepare_saved_baseline(
                baseline_path,
                str(case_dir),
                args.excel,
                args.sheet,
                seed=args.seed,
                **mode_options_from_mapping(rows[index]),
            )
        )

    aggregate = {
        "operator": "smla",
        "scope": "CPU-only offline acceptance; no NPU PASS claim",
        "case_count": len(reports),
        "cases": reports,
        "pass": bool(reports) and all(item["pass"] for item in reports),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = report_root / "smla_consistency_prepare_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"pass": aggregate["pass"], "report": str(aggregate_path)}, ensure_ascii=False))
    return 0 if aggregate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

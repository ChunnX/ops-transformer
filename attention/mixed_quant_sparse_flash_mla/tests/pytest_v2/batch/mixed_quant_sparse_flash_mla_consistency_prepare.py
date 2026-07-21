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

"""Excel -> MQSMLA CPU golden -> baseline/derived PT offline acceptance."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import check_valid_param
import mixed_quant_sparse_flash_mla_golden
import utils
from batch.consistency.offline import mode_options_from_mapping, prepare_saved_baseline
from batch.consistency.relation_manifest import merge_relation_manifests
from batch.consistency.reporting import stable_json_digest


DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.uint8": torch.uint8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", required=True)
    parser.add_argument("--sheet", default="MQSMLA")
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def _normalize(row):
    normalized = dict(row)
    for name in ("q_type", "ori_kv_type", "cmp_kv_type"):
        normalized[name] = DTYPES.get(normalized.get(name), normalized.get(name))
    return utils.fill_none_params(normalized)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = utils.load_excel_test_cases(args.excel, args.sheet)
    baseline_dir = Path(args.pt_dir) / "baseline"
    report_root = Path(args.report_dir)
    reports = []
    for index, row in enumerate(rows):
        params = _normalize(row)
        check_valid_param.check_valid_param(params)
        baseline = mixed_quant_sparse_flash_mla_golden.generate_and_save_testdata(params)
        baseline_path = mixed_quant_sparse_flash_mla_golden.save_test_case(
            baseline, str(baseline_dir)
        )
        case_dir = report_root / f"case_{index:03d}"
        reports.append(
            prepare_saved_baseline(
                baseline_path,
                str(case_dir),
                args.excel,
                args.sheet,
                seed=args.seed,
                **mode_options_from_mapping(row),
            )
        )

    report_root.mkdir(parents=True, exist_ok=True)
    relation_manifest = (
        merge_relation_manifests(item["relation_manifest"] for item in reports)
        if reports
        else None
    )
    relation_manifest_path = report_root / "mqsmla_consistency_relation_manifest.json"
    if relation_manifest is not None:
        with relation_manifest_path.open("w", encoding="utf-8") as file:
            json.dump(relation_manifest, file, ensure_ascii=False, indent=2, sort_keys=True)

    aggregate = {
        "operator": "mqsmla",
        "scope": "CPU-only offline acceptance; no NPU PASS claim",
        "case_count": len(reports),
        "cases": reports,
        "relation_manifest_path": (
            str(relation_manifest_path.resolve()) if relation_manifest is not None else None
        ),
        "relation_manifest_sha256": (
            stable_json_digest(relation_manifest) if relation_manifest is not None else None
        ),
        "relation_coverage": (
            relation_manifest["coverage"] if relation_manifest is not None else None
        ),
        "pass": (
            bool(reports)
            and all(item["pass"] for item in reports)
            and relation_manifest is not None
            and relation_manifest["precheck"]["status"] == "PASS"
        ),
    }
    aggregate_path = report_root / "mqsmla_consistency_prepare_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"pass": aggregate["pass"], "report": str(aggregate_path)}, ensure_ascii=False))
    return 0 if aggregate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

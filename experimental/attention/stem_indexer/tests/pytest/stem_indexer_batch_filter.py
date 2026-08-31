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

import csv
import glob
import os


def load_csv_case_ids(path_pattern):
    csv_files = sorted(glob.glob(path_pattern))
    if not csv_files and os.path.isfile(path_pattern):
        csv_files = [path_pattern]
    csv_files = [path for path in csv_files if path.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV testcase file found from path: {path_pattern}"
        )

    case_ids = set()
    for csv_file in csv_files:
        with open(csv_file, newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            if not reader.fieldnames or "case_id" not in reader.fieldnames:
                raise ValueError(f"{csv_file} missing column: case_id")
            case_ids.update(
                row["case_id"].strip()
                for row in reader
                if row.get("case_id", "").strip()
            )
    return case_ids


def filter_testcase_files_by_csv(testcase_files, csv_path_pattern, load_case):
    csv_case_ids = load_csv_case_ids(csv_path_pattern)
    filtered = []
    for filepath in testcase_files:
        try:
            case = load_case(filepath)
        except Exception as err:
            print(f"Warning: skip unreadable StemIndexer pt file {filepath}: {err}")
            continue
        if case.get("case_id", "") in csv_case_ids:
            filtered.append(filepath)
    return filtered

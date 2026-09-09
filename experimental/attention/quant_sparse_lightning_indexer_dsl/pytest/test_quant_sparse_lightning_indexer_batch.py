"""ASC-style PT batch execution and result comparison for QSLI."""

import os
from pathlib import Path

import pytest
import torch

from qsli_test_utils import QsliCaseSelector, compare_qsli_case, run_qsli_case

PT_DIR = os.environ.get("QSLI_TESTCASE_DIR", "pt_path")
PT_LIST = os.environ.get("QSLI_PT_FILE_LIST", "")
CASE_NAMES = os.environ.get("QSLI_CASE_NAMES", "")
CASE_INDEXES = os.environ.get("QSLI_CASE_INDEXES", "")
SINGLE_PATH = os.environ.get("QSLI_TESTCASE_PATH", "")

try:
    TESTCASE_FILES = QsliCaseSelector.resolve(
        PT_DIR, explicit_files=SINGLE_PATH or PT_LIST,
        case_names=CASE_NAMES, case_indexes=CASE_INDEXES,
    )
except ValueError:
    if any((SINGLE_PATH, PT_LIST, CASE_NAMES, CASE_INDEXES, os.environ.get("QSLI_TESTCASE_DIR"))):
        raise
    TESTCASE_FILES = []


@pytest.mark.ci
@pytest.mark.npu
@pytest.mark.parametrize("testcase_file", TESTCASE_FILES, ids=lambda x: Path(x).stem)
def test_qsli_batch(testcase_file):
    print(f"执行用例：{testcase_file}")
    case_data = torch.load(testcase_file, map_location="cpu", weights_only=False)
    actual = run_qsli_case(case_data, int(os.environ.get("QSLI_DEVICE_ID", "0")))
    result, fulfill = compare_qsli_case(case_data, actual)
    print(f"result: {result}")
    print(f"fulfill_percent: {fulfill}")
    print("result_return_value: Pass")

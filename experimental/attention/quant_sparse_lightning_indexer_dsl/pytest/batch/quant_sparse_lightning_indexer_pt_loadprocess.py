"""Load one QSLI PT fixture, execute NPU, and apply result_compare."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsli_test_utils import compare_qsli_case, run_qsli_case


def test_qsli_process(filepath, device_id=0):
    case_data = torch.load(filepath, map_location="cpu", weights_only=False)
    actual = run_qsli_case(case_data, device_id)
    return compare_qsli_case(case_data, actual)

"""ASC-style paramset-driven single-case entry for QSLI."""

import itertools
import os
from pathlib import Path

import pytest
import torch

from qsli_test_utils import compare_qsli_case, generate_qsli_test_data, run_qsli_case
from test_quant_sparse_lightning_indexer_paramset import ENABLED_PARAMSETS

PARAM_NAMES = (
    "batch_size", "query_length", "sequence_length", "cmp_ratio",
    "storage_mode", "mask_mode", "topk", "candidate_block_length",
    "output_idx_offset", "seed",
)
REQUESTED = {x.strip() for x in os.environ.get("QSLI_CASE_NAMES", "").split(",") if x.strip()}
SAVE_PT_DIR = os.environ.get("QSLI_SINGLE_SAVE_PT_DIR", "").strip()

PARAMS = []
matched = set()
for paramset_name, values in ENABLED_PARAMSETS:
    names = PARAM_NAMES + tuple(name for name in values if name not in PARAM_NAMES)
    combinations = itertools.product(*(values[name] for name in names))
    for index, combination in enumerate(combinations, 1):
        case_name = paramset_name if index == 1 else f"{paramset_name}_{index:03d}"
        if REQUESTED and paramset_name not in REQUESTED and case_name not in REQUESTED:
            continue
        matched.update((paramset_name, case_name))
        PARAMS.append(pytest.param(dict(zip(names, combination)), case_name, id=case_name))
if REQUESTED - matched:
    raise ValueError(f"unknown case(s): {sorted(REQUESTED - matched)}")


@pytest.mark.ci
@pytest.mark.npu
@pytest.mark.parametrize("params,case_name", PARAMS)
def test_qsli_single(params, case_name):
    case_data = generate_qsli_test_data(params)
    if SAVE_PT_DIR:
        target = Path(SAVE_PT_DIR) / f"{case_name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case_data, target)
        print(f"saved: {target}")
    actual = run_qsli_case(case_data, int(os.environ.get("QSLI_DEVICE_ID", "0")))
    result, fulfill = compare_qsli_case(case_data, actual)
    print(f"result={result}, fulfill_percent={fulfill}")

"""ASC-compatible paramset entry for CPU Golden versus fused DSL NPU QLI."""

import importlib
import itertools
import os
from pathlib import Path

import pytest
import torch
import torch_npu

import quant_lightning_indexer_v2_golden
import result_compare_method
from qliv2_test_utils import PARAM_NAMES, QliV2ResultWriter, ensure_comparison_passed


paramset = os.environ.get("QLIV2_PARAMSET", "default")
if paramset != "default":
    raise ValueError(f"unsupported paramset: {paramset}")
ENABLED_PARAMSETS = importlib.import_module(
    "test_quant_lightning_indexer_v2_paramset"
).ENABLED_PARAMSETS
requested_names = {
    name.strip() for name in os.environ.get("QLIV2_CASE_NAMES", "").split(",")
    if name.strip()
}
matched_names = set()
SAVE_PT_DIR = os.environ.get("QLIV2_SINGLE_SAVE_PT_DIR", "").strip()
RESULT_PATH = os.environ.get("QLIV2_SINGLE_RESULT_PATH", "").strip()


param_names = list(PARAM_NAMES) + ["run_mode"]
param_combinations = []
for paramset_name, params in ENABLED_PARAMSETS:
    missing = [name for name in param_names if name not in params]
    if missing:
        raise ValueError(f"{paramset_name} is missing parameters: {missing}")
    combinations = itertools.product(*(params[name] for name in param_names))
    combinations = list(combinations)
    for combo_index, combo in enumerate(combinations, start=1):
        case_name = (
            paramset_name if len(combinations) == 1
            else f"{paramset_name}_{combo_index:03d}"
        )
        if requested_names and not (
            paramset_name in requested_names or case_name in requested_names
        ):
            continue
        matched_names.update((paramset_name, case_name))
        values = dict(zip(param_names, combo))
        values["case_name"] = case_name
        param_combinations.append(pytest.param(values, id=case_name))

unknown_names = sorted(requested_names - matched_names)
if unknown_names:
    raise ValueError(f"unknown case(s): {unknown_names}")


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_qliv2(param_combinations):
    torch_npu.npu.set_device(0)
    run_mode = os.environ.get(
        "QLIV2_RUN_MODE", param_combinations["run_mode"]
    ).strip().lower()
    if run_mode != "eager":
        raise ValueError("DSL QLI transfer suite currently supports eager mode only")
    test_data = tuple(param_combinations[name] for name in PARAM_NAMES)
    case_name = QliV2ResultWriter.case_name(
        test_data, explicit_name=param_combinations["case_name"]
    )

    if SAVE_PT_DIR:
        case_data = quant_lightning_indexer_v2_golden.generate_qliv2_test_data(
            test_data
        )
        case_path = Path(SAVE_PT_DIR) / f"{case_name}.pt"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case_data, case_path)
        from batch.quant_lightning_indexer_v2_pt_loadprocess import test_qliv2_process
        outputs = test_qliv2_process(case_path, device_id=0)
        cpu_result, npu_result, topk_value, cpu_value, npu_value, _, _ = outputs
    else:
        cpu_result, npu_result, topk_value, cpu_value, npu_value = (
            quant_lightning_indexer_v2_golden.qliv2_output_single(test_data)
        )

    result, fulfill_percent = result_compare_method.check_result(
        cpu_result, npu_result, topk_value,
        param_combinations["output_idx_offset"], test_data,
        cpu_value, npu_value,
    )
    value_result, value_percent = "N/A", 0
    if bool(param_combinations["return_value"]):
        value_result, value_percent = result_compare_method.check_result_return_value(
            cpu_value, npu_value, test_data, cpu_result, npu_result,
            topk_value, param_combinations["output_idx_offset"],
        )
    if RESULT_PATH:
        QliV2ResultWriter.append(
            RESULT_PATH,
            QliV2ResultWriter.row(
                case_name, test_data, result, fulfill_percent,
                value_result, value_percent,
            ),
        )
    ensure_comparison_passed(
        case_name, result, fulfill_percent, value_result, value_percent
    )


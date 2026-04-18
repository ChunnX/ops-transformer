import copy
import math

import pytest
import torch

import check_valid_param
import kv_quant_sparse_flash_attention_paramset
import utils


def _make_param_combination(**overrides):
    base = {
        "Testcase_Prefix": "blackbox",
        "Testcase_Number": 7,
        "layout_query": "BSND",
        "layout_kv": "PA_BSND",
        "q_type": torch.bfloat16,
        "kv_dtype": None,
        "B": 2,
        "S1": 4,
        "S2": 32,
        "N1": 8,
        "N2": 1,
        "D": 256,
        "K": 16,
        "scale_value": 1.0 / 24.0,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "sparse_block_size": 1,
        "tile_size": 128,
        "rope_head_dim": 64,
        "sparse_mode": 3,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "block_size": 16,
        "block_num": None,
        "actual_seq_q": [3, 4],
        "actual_seq_kv": [17, 32],
    }
    base.update(overrides)
    return base


def _make_param_tuple(**overrides):
    combination = _make_param_combination(**overrides)
    converted = utils.convert_param_combination_to_cs_format(combination)
    return (
        converted["case_name"],
        combination["layout_query"],
        combination["layout_kv"],
        combination["q_type"],
        combination["kv_dtype"],
        combination["B"],
        combination["S1"],
        combination["S2"],
        combination["N1"],
        combination["N2"],
        combination["D"],
        combination["K"],
        combination["scale_value"],
        combination["key_quant_mode"],
        combination["value_quant_mode"],
        combination["sparse_block_size"],
        combination["tile_size"],
        combination["rope_head_dim"],
        combination["sparse_mode"],
        combination["attention_mode"],
        combination["quant_scale_repo_mode"],
        combination["block_size"],
        math.ceil(max(combination["actual_seq_kv"]) / combination["block_size"]) * combination["B"]
        if combination["layout_kv"] == "PA_BSND" and combination["block_num"] is None
        else combination["block_num"],
        combination["actual_seq_q"],
        combination["actual_seq_kv"],
    )


def test_check_valid_param_accepts_representative_pa_case():
    check_valid_param.check_valid_param(_make_param_tuple())


def test_check_valid_param_rejects_non_pa_layout_mismatch():
    with pytest.raises(ValueError, match="layout_query and layout_kv must match"):
        check_valid_param.check_valid_param(
            _make_param_tuple(layout_query="BSND", layout_kv="TND")
        )


def test_check_valid_param_rejects_actual_seq_length_mismatch():
    with pytest.raises(ValueError, match="actual_seq_q length should equal B=2"):
        check_valid_param.check_valid_param(
            _make_param_tuple(actual_seq_q=[4])
        )


def test_convert_param_combination_to_cs_format_keeps_case_name_and_auto_block_num():
    param_combination = _make_param_combination()

    params = utils.convert_param_combination_to_cs_format(param_combination)

    assert params["case_name"] == "blackbox_BSND_PA_BSND_BF16_2_8_1_4_32_256_16_000007"
    assert params["shape_input"][1][0] == 4
    assert params["shape_input"][4] == [2, 4]
    assert params["dtype_output"] == ["bf16"]


def test_combin_params_expands_cartesian_product_without_changing_values():
    enabled_params = [copy.deepcopy(_make_param_combination())]
    enabled_params[0]["layout_query"] = ["BSND", "TND"]
    enabled_params[0]["layout_kv"] = ["PA_BSND"]
    enabled_params[0]["q_type"] = [torch.bfloat16]
    enabled_params[0]["kv_dtype"] = [None, "hifloat8"]
    for key, value in list(enabled_params[0].items()):
        if not isinstance(value, list):
            enabled_params[0][key] = [value]
    enabled_params[0]["actual_seq_q"] = [[3, 4]]
    enabled_params[0]["actual_seq_kv"] = [[17, 32]]

    param_combinations = utils.combin_params(enabled_params)

    assert len(param_combinations) == 4
    assert {case["layout_query"] for case in param_combinations} == {"BSND", "TND"}
    assert {case["kv_dtype"] for case in param_combinations} == {None, "hifloat8"}


def test_baseline_md5_covers_all_enabled_single_cases():
    param_combinations = utils.combin_params(kv_quant_sparse_flash_attention_paramset.ENABLED_PARAMS)
    baseline_data = utils.load_baseline_md5()
    expected_case_names = {
        utils.convert_param_combination_to_cs_format(case)["case_name"]
        for case in param_combinations
    }

    assert len(expected_case_names) == len(param_combinations)
    assert expected_case_names.issubset(set(baseline_data))

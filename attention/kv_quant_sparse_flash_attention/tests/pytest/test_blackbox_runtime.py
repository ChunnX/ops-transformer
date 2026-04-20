from pathlib import Path

import pytest

import utils
import test_kv_quant_sparse_flash_attention_single as single


def test_qsfa_run_npu_records_skipcompare_when_cpu_golden_is_unavailable(monkeypatch, tmp_path):
    captured = {}

    def fake_save_result(params, result, fulfill_percent, result_path):
        captured["params"] = params
        captured["result"] = result
        captured["fulfill_percent"] = fulfill_percent
        captured["result_path"] = result_path

    monkeypatch.setattr(utils, "save_result", fake_save_result)

    result_path = tmp_path / "result.xlsx"
    test_data = {
        "Testcase_Name": "runtime_skipcompare",
        "params": ("runtime_skipcompare", "BSND", "PA_BSND"),
        "cpu_output": None,
    }

    single.qsfa_run_npu(test_data, device_id=0, result_path=result_path)

    assert captured["params"] == ("runtime_skipcompare", "BSND", "PA_BSND")
    assert captured["result"] == "SkipCompare"
    assert captured["fulfill_percent"] == 100.0
    assert captured["result_path"] == Path(result_path)


def test_compare_output_bin_md5_detects_match_and_mismatch(tmp_path):
    baseline_file = tmp_path / "baseline.bin"
    same_file = tmp_path / "same.bin"
    different_file = tmp_path / "different.bin"

    baseline_file.write_bytes(b"cpu-output")
    same_file.write_bytes(b"cpu-output")
    different_file.write_bytes(b"cpu-output-v2")

    match_result = utils.compare_output_bin_md5(baseline_file, same_file)
    mismatch_result = utils.compare_output_bin_md5(baseline_file, different_file)

    assert match_result["match"] is True
    assert match_result["expected_md5"] == match_result["actual_md5"]
    assert mismatch_result["match"] is False
    assert mismatch_result["expected_md5"] != mismatch_result["actual_md5"]


def test_qsfa_run_cpu_and_save_md5_returns_cpu_bin_metadata(monkeypatch, tmp_path):
    def fake_convert_param_combination_to_cs_format(param_combination):
        return {
            "case_path": str(tmp_path),
            "case_name": "cpu_md5_case",
            "dtype_output": ["fp16"],
        }

    def fake_sub_process_cpu(params):
        output_file = tmp_path / "cpu_md5_case_output_0.bin"
        output_file.write_bytes(b"cpu-binary-output")
        params["cpu_output_bin_path"] = str(output_file)
        params["cpu_output_md5"] = utils.calculate_file_md5(output_file)
        return {
            "testcase_name": params["case_name"],
            "params": params,
            "input_tensors": {},
            "input_attrs": {},
            "cpu_output": "cpu_result",
        }

    monkeypatch.setattr(utils, "convert_param_combination_to_cs_format", fake_convert_param_combination_to_cs_format)
    monkeypatch.setattr(single, "sub_process_cpu", fake_sub_process_cpu)

    result = single.qsfa_run_cpu_and_save_md5({"layout_query": "BSND"})

    assert result["cpu_output"] == "cpu_result"
    assert result["params"]["cpu_output_bin_path"] == str(tmp_path / "cpu_md5_case_output_0.bin")
    assert result["params"]["cpu_output_md5"] == utils.calculate_file_md5(tmp_path / "cpu_md5_case_output_0.bin")


def test_is_baseline_update_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("UPDATE_BASELINE", raising=False)

    assert utils.is_baseline_update_enabled() is False


def test_is_baseline_update_enabled_accepts_explicit_switch(monkeypatch):
    monkeypatch.setenv("UPDATE_BASELINE", "1")

    assert utils.is_baseline_update_enabled() is True


def test_verify_md5_with_baseline_reports_missing_baseline(monkeypatch):
    monkeypatch.setattr(utils, "load_baseline_md5", lambda: {})

    result = utils.verify_md5_with_baseline("case_without_baseline", "abc123")

    assert result["status"] == "no_baseline"
    assert result["match"] is True
    assert "No baseline MD5" in result["message"]

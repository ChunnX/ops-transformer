import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.st_accuracy import run_st_accuracy
from scripts.st_accuracy import coverage as coverage_module
from scripts.st_accuracy import st_cases as st_cases_module
from scripts.st_accuracy import taxonomy as taxonomy_module
from scripts.st_accuracy.adapters import get_adapter
from scripts.st_accuracy.inventory import discover_operators
from scripts.st_accuracy.inventory import Operator
from scripts.st_accuracy.native import NativeRunner
from scripts.st_accuracy.runner import CaseSelector
from scripts.st_accuracy.st_cases import StCase, load_cases


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _assert_timing_keys(testcase: unittest.TestCase, timings: dict, expected_keys: set[str]) -> None:
    testcase.assertIsInstance(timings, dict)
    testcase.assertTrue(expected_keys <= set(timings), f"missing timing keys: {sorted(expected_keys - set(timings))}")
    for key in expected_keys:
        testcase.assertIsInstance(timings[key], (int, float), key)
        testcase.assertGreaterEqual(timings[key], 0.0, key)


class TransformerStAccuracyTest(unittest.TestCase):
    def test_discover_defaults_to_attention_ascendc_operators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "attention/lightning_indexer/CMakeLists.txt", "")
            _write(root / "attention/lightning_indexer/op_host/lightning_indexer_def.cpp", "")
            _write(root / "attention/lightning_indexer/op_kernel/lightning_indexer.cpp", "#include \"kernel_operator.h\"\n")

            _write(root / "ffn/ffn/CMakeLists.txt", "")
            _write(root / "ffn/ffn/op_host/ffn_def.cpp", "")
            _write(root / "ffn/ffn/op_kernel/ffn.cpp", "#include \"kernel_operator.h\"\n")

            operators = discover_operators(root)

        self.assertEqual([operator.name for operator in operators], ["lightning_indexer"])
        self.assertEqual(operators[0].category, "attention")
        self.assertEqual(operators[0].path.as_posix(), "attention/lightning_indexer")

    def test_case_selector_accepts_two_attention_aclnn_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for op_name, api_name in (
                ("lightning_indexer", "LightningIndexer"),
                ("recurrent_gated_delta_rule", "RecurrentGatedDeltaRule"),
            ):
                _write(root / f"attention/{op_name}/CMakeLists.txt", "")
                _write(root / f"attention/{op_name}/op_host/{op_name}_def.cpp", "")
                _write(root / f"attention/{op_name}/op_kernel/{op_name}.cpp", "__aicore__ void kernel() {}\n")
                _write_json(
                    root / f"attention/{op_name}/tests/st/aclnn{api_name}/all_aclnn{api_name}.json",
                    [
                        {
                            "id": 7,
                            "name": api_name,
                            "aclnn_name": api_name,
                            "inputs": [
                                {
                                    "name": "query",
                                    "type": "tensor",
                                    "dtype": "fp16",
                                    "shape": [1, 1, 1, 128],
                                    "range_values": [-1.0, 1.0],
                                }
                            ],
                        }
                    ],
                )

            operators = discover_operators(root)
            cases = load_cases(root, operators)
            selected = CaseSelector(
                ops={"lightning_indexer", "recurrent_gated_delta_rule"},
                apis={"LightningIndexer", "RecurrentGatedDeltaRule"},
            ).apply(cases)

        self.assertEqual(len(selected), 2)
        self.assertEqual({case.aclnn_name for case in selected}, {"LightningIndexer", "RecurrentGatedDeltaRule"})
        self.assertEqual({case.case_id for case in selected}, {7})

    def test_discovery_report_marks_missing_and_zero_case_requested_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "attention/no_st_cases/CMakeLists.txt", "")
            _write(root / "attention/no_st_cases/op_host/no_st_cases_def.cpp", "")
            _write(root / "attention/no_st_cases/op_kernel/no_st_cases.cpp", "__aicore__ void kernel() {}\n")

            operators = discover_operators(root)
            self.assertTrue(hasattr(st_cases_module, "load_cases_with_diagnostics"))
            load_result = st_cases_module.load_cases_with_diagnostics(root, operators)
            self.assertTrue(hasattr(run_st_accuracy, "_build_discovery_report"))
            report = run_st_accuracy._build_discovery_report(
                operators,
                load_result,
                requested_ops={"no_st_cases", "missing_op"},
            )

        self.assertEqual(report["missing_ops"], ["missing_op"])
        self.assertEqual(report["zero_case_ops"], ["no_st_cases"])
        self.assertEqual(
            report["skipped_assets"],
            [
                {
                    "case_index": None,
                    "category": "attention",
                    "json_path": "attention/no_st_cases/tests/st",
                    "message": "operator has no tests/st directory",
                    "op": "no_st_cases",
                    "path": "attention/no_st_cases",
                    "reason": "missing_st_dir",
                }
            ],
        )

    def test_load_cases_with_diagnostics_reports_skipped_st_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "attention/bad_st/CMakeLists.txt", "")
            _write(root / "attention/bad_st/op_host/bad_st_def.cpp", "")
            _write(root / "attention/bad_st/op_kernel/bad_st.cpp", "__aicore__ void kernel() {}\n")
            _write(root / "attention/bad_st/tests/st/aclnnBadInvalidJson/all_invalid_json.json", "{")
            _write_json(root / "attention/bad_st/tests/st/aclnnBadNonList/all_non_list.json", {"id": 1})
            _write_json(
                root / "attention/bad_st/tests/st/aclnnBadMixed/all_mixed.json",
                [
                    "not a dict",
                    {"id": "not-int", "aclnn_name": "Bad", "inputs": []},
                    {"id": 3, "aclnn_name": "Bad", "inputs": []},
                ],
            )

            operators = discover_operators(root)
            self.assertTrue(hasattr(st_cases_module, "load_cases_with_diagnostics"))
            load_result = st_cases_module.load_cases_with_diagnostics(root, operators)

        self.assertEqual([case.case_id for case in load_result.cases], [3])
        self.assertEqual(
            [(asset.reason, asset.case_index) for asset in load_result.skipped_assets],
            [
                ("invalid_json", None),
                ("case_not_object", 0),
                ("invalid_case_id", 1),
                ("payload_not_list", None),
            ],
        )

    def test_cli_list_fails_for_missing_requested_op(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/st_accuracy/run_st_accuracy.py",
                "list",
                "--ops",
                "definitely_missing_op",
                "--format",
                "json",
            ],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["missing_ops"], ["definitely_missing_op"])

    def test_cli_run_fails_before_writing_results_for_missing_requested_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "missing_op_run"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/st_accuracy/run_st_accuracy.py",
                    "run",
                    "--ops",
                    "definitely_missing_op",
                    "--no-execute",
                    "--out",
                    str(out_dir),
                ],
                cwd=Path.cwd(),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("definitely_missing_op", result.stdout)
            self.assertFalse((out_dir / "summary.json").exists())

    def test_cli_list_json_includes_discovery_timings(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/st_accuracy/run_st_accuracy.py",
                "list",
                "--ops",
                "lightning_indexer,recurrent_gated_delta_rule",
                "--format",
                "json",
            ],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(result.stdout)
        _assert_timing_keys(
            self,
            payload["timings"],
            {"total_ms", "inventory_discovery_ms", "case_load_ms", "discovery_report_ms"},
        )

    def test_cli_run_no_execute_summary_includes_python_stage_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "no_execute"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/st_accuracy/run_st_accuracy.py",
                    "run",
                    "--ops",
                    "lightning_indexer,recurrent_gated_delta_rule",
                    "--no-execute",
                    "--out",
                    str(out_dir),
                ],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

        _assert_timing_keys(
            self,
            summary["timings"],
            {
                "total_ms",
                "inventory_discovery_ms",
                "case_load_ms",
                "discovery_report_ms",
                "case_select_ms",
                "support_gate_ms",
                "report_write_ms",
            },
        )

    def test_cli_coverage_and_taxonomy_summaries_include_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage_out = root / "coverage"
            taxonomy_out = root / "taxonomy"
            for command, out_dir in (("coverage", coverage_out), ("taxonomy", taxonomy_out)):
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/st_accuracy/run_st_accuracy.py",
                        command,
                        "--ops",
                        "lightning_indexer,recurrent_gated_delta_rule",
                        "--out",
                        str(out_dir),
                    ],
                    cwd=Path.cwd(),
                    check=True,
                    capture_output=True,
                    text=True,
                )

            coverage_summary = json.loads((coverage_out / "coverage_summary.json").read_text(encoding="utf-8"))
            taxonomy_summary = json.loads((taxonomy_out / "taxonomy_summary.json").read_text(encoding="utf-8"))

        expected = {
            "total_ms",
            "inventory_discovery_ms",
            "case_load_ms",
            "discovery_report_ms",
            "support_gate_ms",
            "report_write_ms",
        }
        _assert_timing_keys(self, coverage_summary["timings"], expected)
        _assert_timing_keys(self, taxonomy_summary["timings"], expected)

    def test_st_case_reuses_flattened_inputs(self):
        case = StCase(
            op_name="cached_inputs",
            category="attention",
            op_path=Path("attention/cached_inputs"),
            api_dir="aclnnCachedInputs",
            aclnn_name="CachedInputs",
            json_path=Path("attention/cached_inputs/tests/st/aclnnCachedInputs/all.json"),
            case_id=0,
            raw={
                "inputs": [
                    [
                        {
                            "name": "x",
                            "type": "tensor",
                            "dtype": "fp16",
                            "shape": [1],
                            "range_values": [0.0, 1.0],
                        }
                    ],
                    {"name": "dim", "type": "attr", "dtype": "int64", "range_values": 0},
                ]
            },
        )
        calls = 0
        original = st_cases_module.flatten_input_items

        def counting_flatten(values):
            nonlocal calls
            calls += 1
            return original(values)

        st_cases_module.flatten_input_items = counting_flatten
        try:
            self.assertEqual(len(case.flat_inputs), 2)
            self.assertEqual(len(case.flat_inputs), 2)
            self.assertEqual(case.tensor_dtypes, ["fp16"])
            self.assertEqual(case.all_dtypes, ["fp16", "int64"])
        finally:
            st_cases_module.flatten_input_items = original

        self.assertEqual(calls, 1)

    def test_coverage_reuses_support_result_per_case(self):
        case = StCase(
            op_name="cached_support",
            category="attention",
            op_path=Path("attention/cached_support"),
            api_dir="aclnnCachedSupport",
            aclnn_name="CachedSupport",
            json_path=Path("attention/cached_support/tests/st/aclnnCachedSupport/all.json"),
            case_id=0,
            raw={"inputs": []},
        )
        operator = Operator(name="cached_support", category="attention", path=Path("attention/cached_support"))
        calls = 0
        original = coverage_module._case_support

        def counting_support(current_case):
            nonlocal calls
            calls += 1
            self.assertIs(current_case, case)
            return True, ""

        coverage_module._case_support = counting_support
        try:
            report = coverage_module.build_coverage_report([operator], [case])
        finally:
            coverage_module._case_support = original

        self.assertEqual(report["summary"]["executable_case_count"], 1)
        self.assertEqual(calls, 1)

    def test_taxonomy_reuses_support_result_per_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = StCase(
                op_name="cached_support",
                category="attention",
                op_path=Path("attention/cached_support"),
                api_dir="aclnnCachedSupport",
                aclnn_name="CachedSupport",
                json_path=Path("attention/cached_support/tests/st/aclnnCachedSupport/all.json"),
                case_id=0,
                raw={"inputs": []},
            )
            operator = Operator(name="cached_support", category="attention", path=Path("attention/cached_support"))
            calls = 0
            original = taxonomy_module._case_support

            def counting_support(current_case):
                nonlocal calls
                calls += 1
                self.assertIs(current_case, case)
                return True, ""

            taxonomy_module._case_support = counting_support
            try:
                report = taxonomy_module.build_taxonomy_report(root, [operator], [case])
            finally:
                taxonomy_module._case_support = original

        self.assertEqual(report["summary"]["executable_case_count"], 1)
        self.assertEqual(calls, 1)

    def test_native_runner_falls_back_to_transformer_vendor_opapi_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = NativeRunner(repo_root=Path(tmp), out_dir=Path(tmp) / "out", vendor="custom", soc="ascend910b", device=0)

            opapi_lib = runner._vendor_opapi_lib()

        self.assertTrue(opapi_lib.as_posix().endswith("/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so"))

    def test_lightning_indexer_adapter_supports_case_and_serializes_batch_fields(self):
        case = StCase(
            op_name="lightning_indexer",
            category="attention",
            op_path=Path("attention/lightning_indexer"),
            api_dir="aclnnLightningIndexer",
            aclnn_name="LightningIndexer",
            json_path=Path("attention/lightning_indexer/tests/st/aclnnLightningIndexer/all_aclnnLightningIndexer.json"),
            case_id=0,
            raw={
                "id": 0,
                "aclnn_name": "LightningIndexer",
                "inputs": [
                    {"name": "query", "type": "tensor", "dtype": "fp16", "shape": [1, 2, 1, 128], "range_values": [-1.0, 1.0]},
                    {"name": "key", "type": "tensor", "dtype": "fp16", "shape": [1, 4, 1, 128], "range_values": [-1.0, 1.0]},
                    {"name": "weights", "type": "tensor", "dtype": "fp16", "shape": [1, 2, 1], "range_values": [0.25, 1.0]},
                    {"name": "layoutQuery", "type": "attr", "dtype": "str", "range_values": "BSND"},
                    {"name": "layoutKey", "type": "attr", "dtype": "str", "range_values": "BSND"},
                    {"name": "sparseCount", "type": "attr", "dtype": "int64", "range_values": 2},
                    {"name": "sparseMode", "type": "attr", "dtype": "int64", "range_values": 0},
                    {"name": "returnValues", "type": "attr", "dtype": "bool", "range_values": True},
                ],
            },
        )
        adapter = get_adapter("aclnnLightningIndexer")
        runner = NativeRunner(repo_root=Path("."), out_dir=Path("build/st_accuracy/test"), vendor="custom", soc="ascend910b", device=0)

        self.assertIsNotNone(adapter)
        supported, reason = adapter.supports(case)
        batch_line = runner._case_batch_line(3, case, adapter)

        self.assertTrue(supported, reason)
        self.assertIn("case_index=3", batch_line)
        self.assertIn("api=LightningIndexer", batch_line)
        self.assertIn("layout_query=BSND", batch_line)
        self.assertIn("layout_key=BSND", batch_line)
        self.assertIn("sparse_count=2", batch_line)
        self.assertIn("return_values=1", batch_line)

    def test_recurrent_gated_delta_rule_adapter_supports_case_and_serializes_batch_fields(self):
        case = StCase(
            op_name="recurrent_gated_delta_rule",
            category="attention",
            op_path=Path("attention/recurrent_gated_delta_rule"),
            api_dir="aclnnRecurrentGatedDeltaRule",
            aclnn_name="RecurrentGatedDeltaRule",
            json_path=Path(
                "attention/recurrent_gated_delta_rule/tests/st/aclnnRecurrentGatedDeltaRule/all_aclnnRecurrentGatedDeltaRule.json"
            ),
            case_id=0,
            raw={
                "id": 0,
                "aclnn_name": "RecurrentGatedDeltaRule",
                "inputs": [
                    {"name": "query", "type": "tensor", "dtype": "bf16", "shape": [4, 4, 32], "range_values": [-0.5, 0.5]},
                    {"name": "key", "type": "tensor", "dtype": "bf16", "shape": [4, 4, 32], "range_values": [-0.5, 0.5]},
                    {"name": "value", "type": "tensor", "dtype": "bf16", "shape": [4, 8, 32], "range_values": [-0.5, 0.5]},
                    {"name": "beta", "type": "tensor", "dtype": "bf16", "shape": [4, 8], "range_values": [0.0, 0.5]},
                    {"name": "stateRef", "type": "tensor", "dtype": "bf16", "shape": [4, 8, 32, 32], "range_values": [-0.25, 0.25]},
                    {"name": "actualSeqLengths", "type": "tensor", "dtype": "int32", "shape": [2], "value": [2, 2]},
                    {"name": "ssmStateIndices", "type": "tensor", "dtype": "int32", "shape": [4], "value": [0, 1, 2, 3]},
                    {"name": "g", "type": "tensor", "dtype": "fp32", "shape": [4, 8], "range_values": [-0.1, 0.1]},
                    {"name": "gk", "type": "tensor", "dtype": "fp32", "shape": [4, 8, 32], "range_values": [-0.1, 0.1]},
                    {"name": "numAcceptedTokens", "type": "tensor", "dtype": "int32", "shape": [2], "value": [1, 1]},
                    {"name": "scaleValue", "type": "attr", "dtype": "float", "range_values": 1.0},
                ],
            },
        )
        adapter = get_adapter("aclnnRecurrentGatedDeltaRule")
        runner = NativeRunner(repo_root=Path("."), out_dir=Path("build/st_accuracy/test"), vendor="custom", soc="ascend910b", device=0)

        self.assertIsNotNone(adapter)
        supported, reason = adapter.supports(case)
        batch_line = runner._case_batch_line(4, case, adapter)

        self.assertTrue(supported, reason)
        self.assertIn("case_index=4", batch_line)
        self.assertIn("api=RecurrentGatedDeltaRule", batch_line)
        self.assertIn("query_shape=4,4,32", batch_line)
        self.assertIn("value_shape=4,8,32", batch_line)
        self.assertIn("actual_seq_lengths=2,2", batch_line)
        self.assertIn("g_present=1", batch_line)
        self.assertIn("gk_present=1", batch_line)

    def test_recurrent_gated_delta_rule_st_cases_use_unique_state_indices(self):
        cases = json.loads(
            Path(
                "attention/recurrent_gated_delta_rule/tests/st/aclnnRecurrentGatedDeltaRule/"
                "all_aclnnRecurrentGatedDeltaRule.json"
            ).read_text(encoding="utf-8")
        )

        for case in cases:
            tensors = {
                input_desc["name"]: input_desc
                for input_desc in case["inputs"]
                if input_desc.get("type") == "tensor"
            }
            values = tensors["ssmStateIndices"]["value"]
            self.assertEqual(
                len(values),
                len(set(values)),
                f"case {case['id']} should keep ssmStateIndices unique for deterministic ST comparison",
            )

    def test_attention_native_runner_declares_golden_comparison_paths(self):
        source = Path("scripts/st_accuracy/native/st_accuracy_aclnn_runner.cpp").read_text(encoding="utf-8")

        self.assertIn("GoldenLightningIndexer", source)
        self.assertIn("CompareLightningIndexerResult", source)
        self.assertIn("GoldenRecurrentGatedDeltaRule", source)
        self.assertIn("CompareRecurrentGatedDeltaRuleResult", source)
        self.assertIn("CompareLightningIndexerResult(actualIndices", source)
        self.assertIn("CompareRecurrentGatedDeltaRuleResult(outHost", source)

    def test_lightning_indexer_native_runner_uses_empty_values_shape_when_values_are_disabled(self):
        source = Path("scripts/st_accuracy/native/st_accuracy_aclnn_runner.cpp").read_text(encoding="utf-8")

        self.assertIn("valuesShape = args.returnValues ? outShape : std::vector<int64_t>{0}", source)
        self.assertIn("CreateGeneratedTensorByDtype(args.dtype, valuesShape", source)

    def test_attention_native_runner_reports_internal_stage_timings(self):
        source = Path("scripts/st_accuracy/native/st_accuracy_aclnn_runner.cpp").read_text(encoding="utf-8")

        self.assertIn("NativeStageTimings", source)
        self.assertIn("PrintStageTimingsJson", source)
        for timing_key in (
            "parse_ms",
            "input_generate_h2d_ms",
            "golden_ms",
            "workspace_ms",
            "launch_sync_ms",
            "d2h_ms",
            "compare_ms",
        ):
            self.assertIn(timing_key, source)


if __name__ == "__main__":
    unittest.main()

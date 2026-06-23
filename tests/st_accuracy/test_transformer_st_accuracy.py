import json
import tempfile
import unittest
from pathlib import Path

from scripts.st_accuracy.adapters import get_adapter
from scripts.st_accuracy.inventory import discover_operators
from scripts.st_accuracy.native import NativeRunner
from scripts.st_accuracy.runner import CaseSelector
from scripts.st_accuracy.st_cases import StCase, load_cases


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only contract, adapter, decoder, order, and HTML tests."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mega_moe_trace_contract import (
    ContractError,
    expected_header_from_metadata,
    load_json,
    resolve_record_words,
    validate_capture_contract,
    validate_timestamp_order,
)
from pack_normalized_trace import COMPACT_FORMAT, pack_trace_file
from run_critical_path_trace import (
    FORCED_OVERRIDES,
    adapt_test_source,
    decode_artifacts,
    replace_unique_top_level_assignment,
    validate_artifact_destination,
    validate_trace_source_defaults,
)


REPO_ROOT = HERE.parents[3]
SPEC = HERE / "trace-spec.json"
TRACE_TOOL = HERE / "trace_tool.py"
AUDIT = HERE / "audit-implemented/trace_point_audit.json"
MOE_TEST = Path("/home/mace/workspace/moe_test/mega_moe/mega_moe.py")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_synthetic_capture(
    directory: Path, spec_path: Path = SPEC, dispatch_group_ready: int = 0
) -> tuple[Path, Path]:
    spec = load_json(spec_path)
    protocol = spec["trace"]["protocol"]
    metadata = {
        "adapter_version": 9,
        "dispatch_group_ready": dispatch_group_ready,
        "rank": 0,
        "bs": 24,
        "h": 6144,
        "n": 12288,
        "k": 8,
        "local_moe_expert_num": 8,
        "shared_expert_num": 0,
        "world_size": 2,
        "global_world_size": 2,
        "ep_world_size": 2,
        "tp_world_size": 1,
        "aic_core_num": 2,
        "dispatch_quant_out_type": 24,
        "comm_quant_mode": 0,
        "topk_weights_type": False,
        "is_interleaved": False,
        "quant_mode": 4,
        "quant_dtype": "float8_e4m3fn",
        "swiglu_quant_dtype": "float8_e4m3fn",
        "weight_dtype": "float8_e4m3fn",
        "graph_type": 0,
        "output_shape": [24, 6144],
        "output_dtype": "torch.bfloat16",
        "output_nbytes": 24 * 6144 * 2,
        "test_source_sha256": "synthetic-fixture",
    }
    expected = expected_header_from_metadata(metadata, metadata["output_nbytes"], spec)
    record_words = resolve_record_words(protocol, expected["capture_capacity"])
    scratch_record_words = 36 + 20 * expected["capture_capacity"]
    metadata["capacity_preflight"] = {
        "protocol_version": int(protocol["version"]),
        "traced_ai_blocks": metadata["aic_core_num"],
        "aic_core_num": metadata["aic_core_num"],
        "local_moe_expert_num": metadata["local_moe_expert_num"],
        "m_groups_per_wave": expected["m_groups_per_wave"],
        "wave_capacity": expected["wave_capacity"],
        "required_segment_capacity": expected["required_segment_capacity"],
        "scratch_capacity_limit": expected["scratch_capacity_limit"],
        "segment_capacity": expected["capture_capacity"],
        "record_words": record_words,
        "record_count": expected["record_count"],
        "descriptor_required_y_bytes": 4
        * (int(protocol["header_words"]) + expected["record_count"] * record_words),
        "available_y_bytes": metadata["output_nbytes"],
        "scratch_record_words": scratch_record_words,
        "scratch_required_bytes": 4 * expected["record_count"] * scratch_record_words,
        "scratch_reserve_bytes": 50 * 1024 * 1024,
    }
    word_count = metadata["output_nbytes"] // 4
    words = [0] * word_count
    header_values = {
        **expected,
        "magic": int(protocol["magic"], 0),
        "version": int(protocol["version"]),
        "schema_id": int(protocol["schema_id"], 0),
        "header_words": int(protocol["header_words"]),
        "record_words": record_words,
        "repetition_count": 3,
        "truncated": 0,
        "role_counts": [2, 2, 2],
    }
    for field in protocol["header_fields"]:
        value = header_values[field["name"]]
        if isinstance(value, list):
            for index, item in enumerate(value):
                words[int(field["word"]) + index] = item
        else:
            words[int(field["word"])] = value

    # Role-major records. Every stage contains
    # (wave, expert, begin_delta, end_delta) tuples.
    # Multiple active tasks in one Wave deliberately have blank gaps; the AIC
    # diagnostic stages classify selected GMM1 gaps without extending MMAD spans.
    sparse_records = [
        (
            [(0, 0, 200, 420), (0, 1, 520, 720)],
            [(0, 0, 900, 1120), (1, 2, 1300, 1480)],
            [(0, 0, 100, 180), (0, 1, 430, 500)],
            [(0, 0, 180, 190), (0, 1, 500, 510)],
            [(0, 0, 190, 200), (0, 1, 510, 520)],
        ),
        (
            [(0, 0, 230, 390)],
            [(0, 0, 930, 1080)],
            [(0, 0, 120, 210)],
            [(0, 0, 210, 220)],
            [(0, 0, 220, 230)],
        ),
        ([(0, 0, 440, 500), (0, 1, 740, 820)], [], [], [], []),
        ([(0, 0, 410, 470)], [], [], [], []),
        (
            [(0, 0, 80, 160), (0, 1, 260, 350), (1, 2, 1180, 1260)],
            [(0, 0, 1140, 1210), (1, 2, 1510, 1580)],
            [(0, 1, 165, 255)],
            [],
            [],
        ),
        (
            [(0, 0, 100, 175), (0, 1, 300, 365)],
            [(0, 0, 1110, 1180)],
            [(0, 1, 180, 295)],
            [],
            [],
        ),
    ]
    header_words = int(protocol["header_words"])
    payload_cursor = header_words + expected["record_count"] * record_words
    for record_index, stages in enumerate(sparse_records):
        core = record_index % metadata["aic_core_num"]
        role = record_index // metadata["aic_core_num"]
        group_base = (0xFFFFF000 + core * 0x20000) & 0xFFFFFFFF
        process_begin = (group_base + role * 3) & 0xFFFFFFFF
        base = header_words + record_index * record_words
        words[base + 0] = process_begin
        words[base + 1] = (group_base + 3000 + role * 3) & 0xFFFFFFFF
        fixed_deltas = {
            "input_buffer_init_begin_S": 1,
            "input_buffer_init_end_S": 3,
            "input_quantize_begin_S": 4,
            "input_quantize_end_S": 10,
            "route_index_begin_S": 11,
            "route_index_end_S": 20,
            "workspace_reset_begin_S": 21,
            "workspace_reset_end_S": 30,
            "local_prepare_sync_begin_S": 31,
            "local_prepare_sync_end_S": 40,
            "cross_rank_sync_begin_S": 41,
            "cross_rank_sync_end_S": 50,
            "expert_buffer_init_begin_S": 51,
            "expert_buffer_init_end_S": 60,
            "final_combine_drain_begin_S": 1600,
            "final_combine_drain_end_S": 1610,
            "moe_epilogue_begin_S": 1611,
            "moe_epilogue_end_S": 1640,
            "moe_tail_sync_begin_S": 1641,
            "moe_tail_sync_end_S": 1700,
            "shared_expert_gmm2_begin_S": 1701,
            "shared_expert_gmm2_end_S": 1702,
            "output_cross_rank_sync_begin_S": 1703,
            "output_cross_rank_sync_end_S": 1800,
            "unpermute_init_begin_S": 1801,
            "unpermute_init_end_S": 1810,
            "unpermute_begin_S": 1811,
            "unpermute_end_S": 2500,
        }
        role_name = spec["trace"]["roles"][role]
        for field in protocol["record_fields"]:
            name = field["name"]
            if name in fixed_deltas and role_name in field["roles"]:
                words[base + int(field["word"])] = (
                    process_begin + fixed_deltas[name]
                ) & 0xFFFFFFFF
        for stage_index, segments in enumerate(stages):
            stage_contract = protocol["sparse_segments"]["stages"][stage_index]
            descriptor = base + int(stage_contract["offset_record_word"])
            words[descriptor] = payload_cursor
            words[descriptor + 1] = len(segments)
            for wave, expert, begin, end in segments:
                words[payload_cursor : payload_cursor + 4] = [
                    wave,
                    expert,
                    (group_base + begin) & 0xFFFFFFFF,
                    (group_base + end) & 0xFFFFFFFF,
                ]
                payload_cursor += 4

    raw = b"".join(word.to_bytes(4, "little") for word in words)
    metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    raw_path = directory / "rank0.raw"
    metadata_path = directory / "rank0.runtime.json"
    raw_path.write_bytes(raw)
    _write_json(metadata_path, metadata)
    return raw_path, metadata_path


class TraceContractTests(unittest.TestCase):
    def test_directory_assignment_adapter_accepts_rhs_format_changes(self) -> None:
        replacement = "golden_cache_dir = '/tmp/trace-golden'"
        variants = (
            "golden_cache_dir = os.path.join(script_dir, 'golden_cache')\n",
            "golden_cache_dir: str = str(Path(script_dir) / 'golden_cache')\n",
            "golden_cache_dir = os.path.join(\n    script_dir,\n    'golden_cache',\n)\n",
        )
        for source in variants:
            with self.subTest(source=source):
                self.assertEqual(
                    replace_unique_top_level_assignment(
                        source,
                        "golden_cache_dir",
                        replacement,
                        "golden cache directory",
                    ),
                    replacement + "\n",
                )

    def test_directory_assignment_adapter_remains_fail_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "found 0"):
            replace_unique_top_level_assignment(
                "cache_dir = '/tmp/cache'\n",
                "golden_cache_dir",
                "golden_cache_dir = '/tmp/golden'",
                "golden cache directory",
            )
        with self.assertRaisesRegex(ContractError, "found 2"):
            replace_unique_top_level_assignment(
                "golden_cache_dir = 'a'\ngolden_cache_dir = 'b'\n",
                "golden_cache_dir",
                "golden_cache_dir = '/tmp/golden'",
                "golden cache directory",
            )

    def test_trace_switch_defaults_to_diagnostic_mode(self) -> None:
        source_defaults = validate_trace_source_defaults(REPO_ROOT)
        self.assertEqual(
            source_defaults["defaults"],
            {"MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE": 1},
        )

    def test_run_artifacts_must_stay_outside_source_repo(self) -> None:
        with self.assertRaisesRegex(ContractError, "outside the source repository"):
            validate_artifact_destination(
                REPO_ROOT / "build" / "trace-output",
                REPO_ROOT,
                decode_only=False,
            )
        validate_artifact_destination(
            REPO_ROOT / "mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic",
            REPO_ROOT,
            decode_only=True,
        )
        validate_artifact_destination(
            Path("/tmp/mega-moe-trace-output"),
            REPO_ROOT,
            decode_only=False,
        )

    def test_full_decode_order_and_html_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            make_synthetic_capture(directory)
            results = decode_artifacts(
                artifact_dir=directory,
                spec_path=SPEC,
                audit_path=AUDIT,
                trace_tool=TRACE_TOOL,
                python=sys.executable,
                allow_truncated=False,
                expected_source_sha="synthetic-fixture",
            )
            self.assertEqual(len(results), 1)
            self.assertTrue((directory / "rank0.html").is_file())
            rendered_html = (directory / "rank0.html").read_text(encoding="utf-8")
            self.assertIn("打点位置与语义说明", rendered_html)
            self.assertIn("什么时候打", rendered_html)
            self.assertIn("为什么打点", rendered_html)
            block0_aic = rendered_html.index("AI Block 00 / AIC")
            block0_aiv0 = rendered_html.index("AI Block 00 / AIV0")
            block0_aiv1 = rendered_html.index("AI Block 00 / AIV1")
            self.assertLess(block0_aic, block0_aiv0)
            self.assertLess(block0_aiv0, block0_aiv1)
            block1_aic = rendered_html.index("AI Block 01 / AIC")
            block1_aiv0 = rendered_html.index("AI Block 01 / AIV0")
            block1_aiv1 = rendered_html.index("AI Block 01 / AIV1")
            self.assertLess(block0_aiv1, block1_aic)
            self.assertLess(block1_aic, block1_aiv0)
            self.assertLess(block1_aiv0, block1_aiv1)
            self.assertNotIn("Trace-point placement audit", rendered_html)
            normalized = load_json(directory / "rank0.normalized.json")
            self.assertEqual(normalized["protocol_version"], 14)
            self.assertEqual(normalized["schema_id"], "0x0aaba4b5")
            self.assertEqual(normalized["metadata"]["resolved_record_words"], 40)
            self.assertEqual(normalized["metadata"]["header"]["role_counts"], [2, 2, 2])
            self.assertFalse(normalized["metadata"]["trace_truncated"])
            compact_path = directory / "rank0.compact.json.gz"
            pack_stats = pack_trace_file(
                directory / "rank0.normalized.json", compact_path
            )
            compact = json.loads(gzip.decompress(compact_path.read_bytes()))
            self.assertEqual(compact["format"], COMPACT_FORMAT)
            self.assertEqual(compact["protocol"], [14, "0x0aaba4b5", 32])
            self.assertEqual(compact["config"]["role_counts"], [2, 2, 2])
            self.assertEqual(len(compact["records"]), 6)
            self.assertEqual(compact["records"][0][0:2], ["AIC", 0])
            self.assertIn("ls", {phase[0] for phase in compact["records"][0][4]})
            self.assertIn("g1", {task[0] for task in compact["records"][0][5]})
            self.assertIn("gs", {task[0] for task in compact["records"][0][5]})
            compact_aiv1_core0 = next(
                record for record in compact["records"] if record[0:2] == ["AIV1", 0]
            )
            self.assertIn("ds", {task[0] for task in compact_aiv1_core0[5]})
            self.assertLess(pack_stats["output_bytes"], pack_stats["input_bytes"])
            self.assertIn("Dispatch（本核有效专家行）", rendered_html)
            self.assertIn("Dispatch 间 Scalar 规划/游标推进", rendered_html)
            self.assertIn("Combine（逐 256-token group，MTE3 完成边界）", rendered_html)
            self.assertIn("Wave 0 / 专家 1 / 本核任务 1", rendered_html)
            order = load_json(directory / "rank0.order-check.json")
            self.assertEqual(order["status"], "passed")
            self.assertEqual(order["span_summary"]["process"]["count"], 6)
            self.assertEqual(order["span_summary"]["dispatch_scalar_gap"]["count"], 2)

    def test_bad_magic_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            raw = bytearray(raw_path.read_bytes())
            raw[0:4] = (0).to_bytes(4, "little")
            raw_path.write_bytes(raw)
            metadata = load_json(metadata_path)
            metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ContractError, "magic"):
                validate_capture_contract(
                    raw_path=raw_path,
                    metadata_path=metadata_path,
                    spec_path=SPEC,
                    allow_truncated=False,
                )

    def test_dispatch_group_ready_build_flag_is_part_of_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(
                directory, dispatch_group_ready=1
            )
            contract, expected = validate_capture_contract(
                raw_path=raw_path,
                metadata_path=metadata_path,
                spec_path=SPEC,
                allow_truncated=False,
            )
            self.assertEqual(expected["flags"], 0x3FFF)
            self.assertEqual(contract["header"]["flags"], 0x3FFF)

    def test_runtime_config_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            raw = bytearray(raw_path.read_bytes())
            config_hash_word = 15
            offset = config_hash_word * 4
            current = int.from_bytes(raw[offset : offset + 4], "little")
            raw[offset : offset + 4] = (current ^ 1).to_bytes(4, "little")
            raw_path.write_bytes(raw)
            metadata = load_json(metadata_path)
            metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ContractError, "runtime_config_hash"):
                validate_capture_contract(
                    raw_path=raw_path,
                    metadata_path=metadata_path,
                    spec_path=SPEC,
                    allow_truncated=False,
                )

    def test_sparse_offset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            raw = bytearray(raw_path.read_bytes())
            spec = load_json(SPEC)
            stage0_offset_word = 30 + int(
                spec["trace"]["protocol"]["sparse_segments"]["stages"][0][
                    "offset_record_word"
                ]
            )
            offset = stage0_offset_word * 4
            current = int.from_bytes(raw[offset : offset + 4], "little")
            raw[offset : offset + 4] = (current + 1).to_bytes(4, "little")
            raw_path.write_bytes(raw)
            metadata = load_json(metadata_path)
            metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ContractError, "expected compact cursor"):
                validate_capture_contract(
                    raw_path=raw_path,
                    metadata_path=metadata_path,
                    spec_path=SPEC,
                    allow_truncated=False,
                )

    def test_truncated_capture_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            raw = bytearray(raw_path.read_bytes())
            raw[12 * 4 : 13 * 4] = (1).to_bytes(4, "little")
            raw_path.write_bytes(raw)
            metadata = load_json(metadata_path)
            metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ContractError, "trace is truncated"):
                validate_capture_contract(
                    raw_path=raw_path,
                    metadata_path=metadata_path,
                    spec_path=SPEC,
                    allow_truncated=False,
                )

    def test_one_sided_stage_boundary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            make_synthetic_capture(directory)
            normalized = directory / "rank0.normalized.json"
            subprocess.run(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "decode",
                    "--spec",
                    str(SPEC),
                    "--raw",
                    str(directory / "rank0.raw"),
                    "--output",
                    str(normalized),
                    "--reference-role",
                    "AIC",
                    "--reference-core",
                    "0",
                    "--reference-point",
                    "process_begin_S",
                ],
                check=True,
            )
            value = load_json(normalized)
            value["records"][0]["timestamps"]["gmm1_problem_end_S"][0] = 0
            _write_json(normalized, value)
            with self.assertRaisesRegex(ContractError, "missing boundary"):
                validate_timestamp_order(normalized, SPEC)

    def test_non_monotonic_wave_intervals_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            make_synthetic_capture(directory)
            normalized = directory / "rank0.normalized.json"
            subprocess.run(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "decode",
                    "--spec",
                    str(SPEC),
                    "--raw",
                    str(directory / "rank0.raw"),
                    "--output",
                    str(normalized),
                    "--reference-role",
                    "AIC",
                    "--reference-core",
                    "0",
                    "--reference-point",
                    "process_begin_S",
                ],
                check=True,
            )
            value = load_json(normalized)
            aiv1_core0 = next(
                record
                for record in value["records"]
                if record["role"] == "AIV1" and record["core"] == 0
            )
            aiv1_core0["timestamps"]["dispatch_task_begin_S"][1] = (
                aiv1_core0["timestamps"]["dispatch_task_begin_S"][0] - 1
            ) & 0xFFFFFFFF
            _write_json(normalized, value)
            with self.assertRaisesRegex(ContractError, "not time-monotonic"):
                validate_timestamp_order(normalized, SPEC)

    def test_dynamic_wave_capacity_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            make_synthetic_capture(directory)
            results = decode_artifacts(
                artifact_dir=directory,
                spec_path=SPEC,
                audit_path=AUDIT,
                trace_tool=TRACE_TOOL,
                python=sys.executable,
                allow_truncated=False,
                expected_source_sha="synthetic-fixture",
            )
            self.assertEqual(len(results), 1)
            normalized = load_json(directory / "rank0.normalized.json")
            self.assertEqual(
                len(normalized["records"][0]["timestamps"]["gmm1_problem_begin_S"]),
                2,
            )
            self.assertEqual(normalized["metadata"]["required_trace_bytes"], 1528)
            order = load_json(directory / "rank0.order-check.json")
            self.assertEqual(order["fine_truncated_records"], [])

    def test_capacity_preflight_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            metadata = load_json(metadata_path)
            metadata["capacity_preflight"]["descriptor_required_y_bytes"] += 4
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ContractError, "capacity_preflight mismatch"):
                validate_capture_contract(
                    raw_path=raw_path,
                    metadata_path=metadata_path,
                    spec_path=SPEC,
                    allow_truncated=False,
                )

    @unittest.skipUnless(
        MOE_TEST.is_file(), "audited moe_test checkout is not available"
    )
    def test_adapter_contains_pre_operator_dynamic_capacity_check(self) -> None:
        source = MOE_TEST.read_text(encoding="utf-8")
        transformed = adapt_test_source(
            source,
            original_script_dir=MOE_TEST.parent,
            overrides=FORCED_OVERRIDES,
        )
        call = "_critical_path_trace_validate_capacity()"
        self.assertIn("record_words = 40", transformed)
        self.assertIn(call, transformed)
        self.assertLess(
            transformed.index(call, transformed.index("out_dtype =")),
            transformed.index("def run_cascade_npu"),
        )

        tree = ast.parse(transformed)
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_critical_path_trace_validate_capacity"
        )

        class _Scalar:
            @staticmethod
            def element_size() -> int:
                return 2

        class _TorchStub:
            bfloat16 = object()

            @staticmethod
            def empty(*_args, **_kwargs):
                return _Scalar()

        namespace = {
            "torch": _TorchStub,
            "out_dtype": _TorchStub.bfloat16,
            "local_moe_expert_num": 8,
            "aic_core_num": 36,
            "bs": 24,
            "h": 6144,
            "n": 12288,
            "k": 8,
            "ep_world_size": 2,
            "_critical_path_trace_capacity": {},
        }
        helper_module = ast.fix_missing_locations(
            ast.Module(body=[helper], type_ignores=[])
        )
        exec(compile(helper_module, "adapter_capacity_helper.py", "exec"), namespace)
        namespace["_critical_path_trace_validate_capacity"]()
        capacity = namespace["_critical_path_trace_capacity"]
        self.assertEqual(capacity["record_words"], 40)
        self.assertEqual(capacity["record_count"], 108)
        self.assertEqual(capacity["descriptor_required_y_bytes"], 17400)
        self.assertEqual(capacity["available_y_bytes"], 294912)
        self.assertEqual(capacity["required_segment_capacity"], 9)
        self.assertEqual(capacity["scratch_required_bytes"], 93312)
        self.assertEqual(capacity["scratch_reserve_bytes"], 50 * 1024 * 1024)

    @unittest.skipUnless(
        MOE_TEST.is_file(), "audited moe_test checkout is not available"
    )
    def test_current_moe_test_adapter_is_syntax_valid(self) -> None:
        source = MOE_TEST.read_text(encoding="utf-8")
        transformed = adapt_test_source(
            source,
            original_script_dir=MOE_TEST.parent,
            overrides=FORCED_OVERRIDES,
        )
        self.assertIn("_critical_path_trace_capture_y(y, rank)", transformed)
        self.assertIn("loop_acc_time = 100  # critical-path trace adapter", transformed)
        self.assertIn('"capture_policy": "last_iteration_only"', transformed)
        self.assertIn('"captured_iteration": int(loop_acc_time)', transformed)
        self.assertIn("'runtime-cache', 'torch_extensions'", transformed)
        self.assertIn("'runtime-cache', 'golden_cache'", transformed)
        self.assertIn("accuracy comparison is intentionally skipped", transformed)
        self.assertIn("capacity preflight passed", transformed)
        compile(transformed, "adapted_mega_moe.py", "exec")


if __name__ == "__main__":
    unittest.main()

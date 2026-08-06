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
from run_critical_path_trace import (
    FORCED_OVERRIDES,
    adapt_test_source,
    decode_artifacts,
    validate_artifact_destination,
    validate_trace_source_defaults,
)


REPO_ROOT = HERE.parents[3]
SPEC = HERE / "trace-spec.json"
COARSE_SPEC = HERE / "trace-spec-coarse-v5.json"
A8W4_SPEC = HERE / "trace-spec-a8w4.json"
TRACE_TOOL = HERE / "trace_tool.py"
AUDIT = HERE / "audit-implemented/trace_point_audit.json"
COARSE_AUDIT = HERE / "audit-implemented-coarse-v5/trace_point_audit.json"
A8W4_AUDIT = HERE / "audit-implemented-a8w4/trace_point_audit.json"
MOE_TEST = Path("/home/mace/workspace/moe_test/mega_moe/mega_moe.py")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_synthetic_capture(
    directory: Path,
    spec_path: Path = SPEC,
    *,
    configured_aic_count: int = 36,
    runtime_aic_count: int | None = None,
) -> tuple[Path, Path]:
    spec = load_json(spec_path)
    protocol = spec["trace"]["protocol"]
    fine_grained = isinstance(protocol["record_words"], dict)
    a8w4 = spec["trace"]["name"] == "mega-moe-a8w4-critical-path"
    metadata = {
        "adapter_version": 4,
        "rank": 0,
        "bs": 24,
        "h": 6144,
        "n": 12288,
        "k": 8,
        "local_moe_expert_num": 8,
        "max_output_size": 384,
        "shared_expert_num": 0,
        "world_size": 2,
        "aic_core_num": configured_aic_count,
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
    actual_aic_count = (
        configured_aic_count if runtime_aic_count is None else runtime_aic_count
    )
    expected = expected_header_from_metadata(
        metadata,
        metadata["output_nbytes"],
        spec,
        runtime_aic_count=actual_aic_count,
    )
    record_words = resolve_record_words(protocol, expected)
    if fine_grained:
        preflight_record_header = {"moe_experts_per_rank": metadata["local_moe_expert_num"]}
        if a8w4:
            max_gmm2_tiles = ((metadata["max_output_size"] + 255) // 256) * (
                (metadata["h"] + 255) // 256
            )
            preflight_tiles_per_expert = (
                max_gmm2_tiles + configured_aic_count - 1
            ) // configured_aic_count
            preflight_record_header["combine_tile_capacity"] = (
                metadata["local_moe_expert_num"] * preflight_tiles_per_expert
            )
        preflight_record_words = resolve_record_words(protocol, preflight_record_header)
        metadata["capacity_preflight"] = {
            "protocol_version": int(protocol["version"]),
            "aic_core_num": configured_aic_count,
            "local_moe_expert_num": metadata["local_moe_expert_num"],
            "record_words": preflight_record_words,
            "required_y_bytes": 4
            * (
                int(protocol["header_words"])
                + 3 * configured_aic_count * preflight_record_words
            ),
            "available_y_bytes": metadata["output_nbytes"],
            "scratch_required_bytes": 4 * 3 * configured_aic_count * (
                (3 if a8w4 else 2) * 3 * 8
                + 4 * preflight_record_header.get("combine_tile_capacity", 0)
            ),
            "scratch_reserve_bytes": 50 * 1024 * 1024,
        }
        if a8w4:
            metadata["capacity_preflight"].update(
                {
                    "max_output_size": metadata["max_output_size"],
                    "combine_tiles_per_expert_per_core": preflight_tiles_per_expert,
                    "combine_tile_capacity": preflight_record_header[
                        "combine_tile_capacity"
                    ],
                }
            )
    word_count = metadata["output_nbytes"] // 4
    words = [0] * word_count
    header_values = {
        **expected,
        "magic": int(protocol["magic"], 0),
        "version": int(protocol["version"]),
        "schema_id": int(protocol["schema_id"], 0),
        "header_words": int(protocol["header_words"]),
        "record_words": record_words,
        "role_counts": [expected["aic_count"]] * 3,
    }
    for field in protocol["header_fields"]:
        value = header_values[field["name"]]
        if isinstance(value, list):
            for index, item in enumerate(value):
                words[int(field["word"]) + index] = item
        else:
            words[int(field["word"])] = value

    header_words = int(protocol["header_words"])
    aic_count = expected["aic_count"]
    for record_index in range(expected["record_count"]):
        role_index = record_index // aic_count
        core = record_index % aic_count
        base = header_words + record_index * record_words
        process_begin = (0xFFFFF000 + core * 3 + role_index) & 0xFFFFFFFF
        words[base + 0] = process_begin
        words[base + 1] = (process_begin + 6000) & 0xFFFFFFFF
        if fine_grained:
            if a8w4:
                active_stage_count = 2 if role_index != 2 else 3
                if role_index == 2 or core < 2:
                    for expert in range(8):
                        for stage in range(active_stage_count):
                            begin = 100 + 1800 * stage + 180 * expert
                            end = begin + 100
                            if role_index == 2 and stage == 2:
                                begin = 3700 + 220 * expert
                                end = begin + 180
                            words[base + 2 + 6 * expert + 2 * stage] = (
                                process_begin + begin
                            ) & 0xFFFFFFFF
                            words[base + 3 + 6 * expert + 2 * stage] = (
                                process_begin + end
                            ) & 0xFFFFFFFF
                    if role_index == 2:
                        tile_record_base = base + 2 + 6 * 8
                        tiles_per_expert = expected[
                            "combine_tiles_per_expert_per_core"
                        ]
                        for expert in range(8):
                            aggregate_begin = 3700 + 220 * expert
                            for tile_ordinal in range(2):
                                slot = expert * tiles_per_expert + tile_ordinal
                                wait_begin = aggregate_begin + tile_ordinal * 100 - (
                                    30 if tile_ordinal == 0 else 40
                                )
                                tile_begin = aggregate_begin + tile_ordinal * 100
                                tile_end = tile_begin + (60 if tile_ordinal == 0 else 80)
                                entry = tile_record_base + slot * 4
                                words[entry + 0] = (process_begin + wait_begin) & 0xFFFFFFFF
                                words[entry + 1] = (process_begin + tile_begin) & 0xFFFFFFFF
                                words[entry + 2] = (process_begin + tile_begin) & 0xFFFFFFFF
                                words[entry + 3] = (process_begin + tile_end) & 0xFFFFFFFF
            elif role_index == 0 and core < 2:
                for expert in range(8):
                    words[base + 2 + 4 * expert] = (
                        process_begin + 200 + 300 * expert
                    ) & 0xFFFFFFFF
                    words[base + 3 + 4 * expert] = (
                        process_begin + 300 + 300 * expert
                    ) & 0xFFFFFFFF
                    words[base + 4 + 4 * expert] = (
                        process_begin + 2800 + 300 * expert
                    ) & 0xFFFFFFFF
                    words[base + 5 + 4 * expert] = (
                        process_begin + 2950 + 300 * expert
                    ) & 0xFFFFFFFF
            elif role_index == 1 and core < 2:
                for expert in range(8):
                    words[base + 2 + 4 * expert] = (
                        process_begin + 350 + 300 * expert
                    ) & 0xFFFFFFFF
                    words[base + 3 + 4 * expert] = (
                        process_begin + 500 + 300 * expert
                    ) & 0xFFFFFFFF
            elif role_index == 2:
                for expert, begin, end in ((0, 100, 900), (4, 950, 1750)):
                    words[base + 2 + 4 * expert] = (process_begin + begin) & 0xFFFFFFFF
                    words[base + 3 + 4 * expert] = (process_begin + end) & 0xFFFFFFFF
                if core < 6:
                    for expert, begin, end in ((0, 3600, 4300), (4, 4400, 5200)):
                        words[base + 4 + 4 * expert] = (
                            process_begin + begin
                        ) & 0xFFFFFFFF
                        words[base + 5 + 4 * expert] = (
                            process_begin + end
                        ) & 0xFFFFFFFF
        elif role_index == 0:
            for wave, begin in enumerate((300, 1600)):
                words[base + 2 + 4 * wave] = (process_begin + begin) & 0xFFFFFFFF
                words[base + 3 + 4 * wave] = (process_begin + begin + 500) & 0xFFFFFFFF
                words[base + 4 + 4 * wave] = (process_begin + begin + 700) & 0xFFFFFFFF
                words[base + 5 + 4 * wave] = (process_begin + begin + 1100) & 0xFFFFFFFF
        elif role_index == 1:
            for wave, begin in enumerate((650, 1950)):
                words[base + 2 + 4 * wave] = (process_begin + begin) & 0xFFFFFFFF
                words[base + 3 + 4 * wave] = (process_begin + begin + 450) & 0xFFFFFFFF
        elif role_index == 2:
            for wave, begin in enumerate((100, 1200)):
                words[base + 2 + 4 * wave] = (process_begin + begin) & 0xFFFFFFFF
                words[base + 3 + 4 * wave] = (process_begin + begin + 700) & 0xFFFFFFFF
                words[base + 4 + 4 * wave] = (
                    process_begin + 3000 + 900 * wave
                ) & 0xFFFFFFFF
                words[base + 5 + 4 * wave] = (
                    process_begin + 3600 + 900 * wave
                ) & 0xFFFFFFFF

    raw = b"".join(word.to_bytes(4, "little") for word in words)
    metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    raw_path = directory / "rank0.raw"
    metadata_path = directory / "rank0.runtime.json"
    raw_path.write_bytes(raw)
    _write_json(metadata_path, metadata)
    return raw_path, metadata_path


class TraceContractTests(unittest.TestCase):
    def test_a8w4_spec_exposes_per_tile_combine_wait_and_work(self) -> None:
        spec = load_json(A8W4_SPEC)
        point_ids = {point["id"] for point in spec["points"]}
        span_ids = {span["id"] for span in spec["spans"]}
        self.assertTrue(
            {
                "a8w4_combine_tile_wait_begin_S",
                "a8w4_combine_tile_wait_end_S",
                "a8w4_combine_tile_begin_S",
                "a8w4_combine_tile_end_S",
            }.issubset(point_ids)
        )
        self.assertTrue(
            {"a8w4_combine_tile_wait", "a8w4_combine_tile_work"}.issubset(
                span_ids
            )
        )

    def test_trace_switches_default_to_fine_diagnostic_mode(self) -> None:
        source_defaults = validate_trace_source_defaults(REPO_ROOT)
        self.assertEqual(
            source_defaults["defaults"],
            {
                "MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE": 1,
                "MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE": 1,
            },
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
                coarse_spec_path=COARSE_SPEC,
                coarse_audit_path=COARSE_AUDIT,
            )
            self.assertEqual(len(results), 1)
            self.assertTrue((directory / "rank0.html").is_file())
            rendered_html = (directory / "rank0.html").read_text(encoding="utf-8")
            self.assertIn("打点位置与语义说明", rendered_html)
            self.assertIn("什么时候打", rendered_html)
            self.assertIn("为什么打点", rendered_html)
            self.assertIn("不同 AI Block", rendered_html)
            block0_aic = rendered_html.index("AI Block 00 / AIC")
            block0_aiv0 = rendered_html.index("AI Block 00 / AIV0")
            block0_aiv1 = rendered_html.index("AI Block 00 / AIV1")
            block1_aic = rendered_html.index("AI Block 01 / AIC")
            self.assertLess(block0_aic, block0_aiv0)
            self.assertLess(block0_aiv0, block0_aiv1)
            self.assertLess(block0_aiv1, block1_aic)
            self.assertNotIn("Trace-point placement audit", rendered_html)
            normalized = load_json(directory / "rank0.normalized.json")
            self.assertEqual(normalized["protocol_version"], 7)
            self.assertEqual(normalized["schema_id"], "0x3eb1c6b3")
            self.assertEqual(normalized["metadata"]["resolved_record_words"], 34)
            self.assertEqual(normalized["metadata"]["header"]["capture_capacity"], 8)
            self.assertFalse(normalized["metadata"]["trace_truncated"])
            self.assertIn("SwiGLU 等待/调度间隙", rendered_html)
            self.assertIn("推断间隙", rendered_html)
            self.assertIn("专家 7", rendered_html)
            self.assertNotIn("最多四槽", rendered_html)
            order = load_json(directory / "rank0.order-check.json")
            self.assertEqual(order["status"], "passed")
            self.assertEqual(order["span_summary"]["process"]["count"], 108)

    def test_a8w4_decode_order_and_html_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            make_synthetic_capture(directory, A8W4_SPEC)
            results = decode_artifacts(
                artifact_dir=directory,
                spec_path=SPEC,
                audit_path=AUDIT,
                trace_tool=TRACE_TOOL,
                python=sys.executable,
                allow_truncated=False,
                expected_source_sha="synthetic-fixture",
                coarse_spec_path=COARSE_SPEC,
                coarse_audit_path=COARSE_AUDIT,
                a8w4_spec_path=A8W4_SPEC,
                a8w4_audit_path=A8W4_AUDIT,
            )
            self.assertEqual(results[0]["protocol_version"], 2)
            normalized = load_json(directory / "rank0.normalized.json")
            self.assertEqual(normalized["schema_id"], "0xe9b9e5a3")
            self.assertEqual(normalized["metadata"]["resolved_record_words"], 114)
            order = load_json(directory / "rank0.order-check.json")
            self.assertEqual(order["span_summary"]["a8w4_process"]["count"], 108)
            self.assertEqual(
                order["span_summary"]["a8w4_combine_tile_wait"]["count"], 576
            )
            self.assertEqual(
                order["span_summary"]["a8w4_combine_tile_work"]["count"], 576
            )
            rendered_html = (directory / "rank0.html").read_text(encoding="utf-8")
            self.assertIn("GMM1 权重 prologue", rendered_html)
            self.assertIn("逐专家 Combine", rendered_html)
            self.assertIn("Combine tile wait", rendered_html)

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

    def test_fine_switch_off_decodes_as_unchanged_coarse_v5(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            make_synthetic_capture(directory, COARSE_SPEC)
            results = decode_artifacts(
                artifact_dir=directory,
                spec_path=SPEC,
                audit_path=AUDIT,
                trace_tool=TRACE_TOOL,
                python=sys.executable,
                allow_truncated=False,
                expected_source_sha="synthetic-fixture",
                coarse_spec_path=COARSE_SPEC,
                coarse_audit_path=COARSE_AUDIT,
            )
            self.assertEqual(results[0]["protocol_version"], 5)
            normalized = load_json(directory / "rank0.normalized.json")
            self.assertEqual(normalized["protocol_version"], 5)
            self.assertEqual(normalized["schema_id"], "0xd32d8a49")
            rendered_html = (directory / "rank0.html").read_text(encoding="utf-8")
            self.assertNotIn("细粒度边界", rendered_html)

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

    def test_runtime_header_core_count_overrides_conservative_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(
                directory, configured_aic_count=36, runtime_aic_count=32
            )
            contract, expected = validate_capture_contract(
                raw_path=raw_path,
                metadata_path=metadata_path,
                spec_path=SPEC,
                allow_truncated=False,
            )
            self.assertEqual(expected["aic_count"], 32)
            self.assertEqual(expected["record_count"], 96)
            self.assertEqual(contract["header"]["role_counts"], [32, 32, 32])
            self.assertEqual(contract["metadata"]["aic_core_num"], 36)

    def test_a8w4_runtime_core_count_recomputes_tile_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(
                directory,
                A8W4_SPEC,
                configured_aic_count=36,
                runtime_aic_count=32,
            )
            contract, expected = validate_capture_contract(
                raw_path=raw_path,
                metadata_path=metadata_path,
                spec_path=A8W4_SPEC,
                allow_truncated=False,
            )
            self.assertEqual(expected["aic_count"], 32)
            self.assertEqual(expected["record_count"], 96)
            self.assertEqual(expected["combine_tiles_per_expert_per_core"], 2)
            self.assertEqual(expected["combine_tile_capacity"], 16)
            self.assertEqual(contract["header"]["role_counts"], [32, 32, 32])
            self.assertEqual(
                contract["metadata"]["capacity_preflight"]["aic_core_num"], 36
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
            value["records"][0]["timestamps"]["gmm1_expert_end_S"][0] = 0
            _write_json(normalized, value)
            with self.assertRaisesRegex(ContractError, "only one boundary"):
                validate_timestamp_order(normalized, SPEC)

    def test_missing_second_dispatch_wave_fails_closed(self) -> None:
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
            aiv1_core0["timestamps"]["dispatch_begin_S"][4] = 0
            aiv1_core0["timestamps"]["dispatch_end_S"][4] = 0
            _write_json(normalized, value)
            with self.assertRaisesRegex(ContractError, "dispatch slot 4"):
                validate_timestamp_order(normalized, SPEC)

    def test_dynamic_eight_expert_layout_has_no_fixed_slot_limit(self) -> None:
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
                coarse_spec_path=COARSE_SPEC,
                coarse_audit_path=COARSE_AUDIT,
            )
            self.assertEqual(len(results), 1)
            normalized = load_json(directory / "rank0.normalized.json")
            self.assertEqual(
                len(normalized["records"][0]["timestamps"]["gmm1_expert_begin_S"]),
                8,
            )
            self.assertEqual(normalized["metadata"]["required_trace_bytes"], 14808)
            order = load_json(directory / "rank0.order-check.json")
            self.assertEqual(order["fine_truncated_records"], [])

    def test_capacity_preflight_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            metadata = load_json(metadata_path)
            metadata["capacity_preflight"]["required_y_bytes"] += 4
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
        self.assertIn("record_words = 2 + 4 * expert_count", transformed)
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
            "_critical_path_trace_capacity": {},
        }
        helper_module = ast.fix_missing_locations(
            ast.Module(body=[helper], type_ignores=[])
        )
        exec(compile(helper_module, "adapter_capacity_helper.py", "exec"), namespace)
        namespace["_critical_path_trace_validate_capacity"]()
        capacity = namespace["_critical_path_trace_capacity"]
        self.assertEqual(capacity["record_words"], 34)
        self.assertEqual(capacity["required_y_bytes"], 14808)
        self.assertEqual(capacity["available_y_bytes"], 294912)
        self.assertEqual(capacity["scratch_required_bytes"], 20736)
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

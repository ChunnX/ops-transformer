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
TRACE_TOOL = HERE / "trace_tool.py"
AUDIT = HERE / "audit-implemented/trace_point_audit.json"
COARSE_AUDIT = HERE / "audit-implemented-coarse-v5/trace_point_audit.json"
MOE_TEST = Path("/home/mace/workspace/moe_test/mega_moe/mega_moe.py")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_synthetic_capture(
    directory: Path, spec_path: Path = SPEC
) -> tuple[Path, Path]:
    spec = load_json(spec_path)
    protocol = spec["trace"]["protocol"]
    fine_grained = int(protocol["version"]) == 6
    metadata = {
        "adapter_version": 1,
        "rank": 0,
        "bs": 1,
        "h": 4096,
        "n": 1024,
        "k": 6,
        "local_moe_expert_num": 4,
        "shared_expert_num": 0,
        "world_size": 2,
        "aic_core_num": 36,
        "dispatch_quant_out_type": 24,
        "comm_quant_mode": 0,
        "topk_weights_type": False,
        "is_interleaved": False,
        "quant_mode": 4,
        "quant_dtype": "float8_e4m3fn",
        "swiglu_quant_dtype": "float8_e4m3fn",
        "weight_dtype": "float8_e4m3fn",
        "graph_type": 0,
        "output_shape": [1, 4096],
        "output_dtype": "torch.bfloat16",
        "output_nbytes": 8192,
        "test_source_sha256": "synthetic-fixture",
    }
    expected = expected_header_from_metadata(metadata, metadata["output_nbytes"], spec)
    word_count = metadata["output_nbytes"] // 4
    words = [0] * word_count
    header_values = {
        **expected,
        "magic": int(protocol["magic"], 0),
        "version": int(protocol["version"]),
        "schema_id": int(protocol["schema_id"], 0),
        "header_words": int(protocol["header_words"]),
        "record_words": int(protocol["record_words"]),
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
    record_words = int(protocol["record_words"])
    aic_count = expected["aic_count"]
    for record_index in range(expected["record_count"]):
        role_index = record_index // aic_count
        core = record_index % aic_count
        base = header_words + record_index * record_words
        process_begin = (0xFFFFF000 + core * 3 + role_index) & 0xFFFFFFFF
        words[base + 0] = process_begin
        words[base + 1] = (process_begin + 5000) & 0xFFFFFFFF
        if role_index == 0:
            if core < 8:
                words[base + 2] = (process_begin + 300) & 0xFFFFFFFF
                words[base + 3] = (process_begin + 500) & 0xFFFFFFFF
                if fine_grained:
                    words[base + 6] = (process_begin + 800) & 0xFFFFFFFF
                    words[base + 7] = (process_begin + 1000) & 0xFFFFFFFF
            words[base + 4] = (process_begin + 1300) & 0xFFFFFFFF
            words[base + 5] = (process_begin + 1600) & 0xFFFFFFFF
            if fine_grained and core < 8:
                words[base + 8] = (process_begin + 1700) & 0xFFFFFFFF
                words[base + 9] = (process_begin + 2000) & 0xFFFFFFFF
        elif role_index == 1 and core < 8:
            words[base + 2] = (process_begin + 650) & 0xFFFFFFFF
            words[base + 3] = (process_begin + 850) & 0xFFFFFFFF
            if fine_grained:
                words[base + 6] = (process_begin + 1050) & 0xFFFFFFFF
                words[base + 7] = (process_begin + 1250) & 0xFFFFFFFF
        elif role_index == 2:
            words[base + 2] = (process_begin + 100) & 0xFFFFFFFF
            words[base + 3] = (process_begin + 600) & 0xFFFFFFFF
            if core < 6:
                words[base + 4] = (process_begin + 2300) & 0xFFFFFFFF
                words[base + 5] = (process_begin + 3600) & 0xFFFFFFFF

    fine_tail = protocol.get("fine_metadata_tail")
    if fine_tail is not None:
        tail_base = header_words + expected["record_count"] * record_words
        stride = int(fine_tail["core_stride_words"])
        for core in range(aic_count):
            gmm1_count = 2 if core < 8 else 0
            gmm2_count = 2 if core < 8 else 1
            aic_meta = (gmm1_count << 16) | (gmm2_count << 19) | (1 << 24)
            swiglu_count = 2 if core < 8 else 0
            aiv0_meta = (swiglu_count << 8) | (1 << 12)
            words[tail_base + core * stride] = aic_meta
            words[tail_base + core * stride + 1] = aiv0_meta

    raw = b"".join(word.to_bytes(4, "little") for word in words)
    metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    raw_path = directory / "rank0.raw"
    metadata_path = directory / "rank0.runtime.json"
    raw_path.write_bytes(raw)
    _write_json(metadata_path, metadata)
    return raw_path, metadata_path


class TraceContractTests(unittest.TestCase):
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
            self.assertEqual(normalized["protocol_version"], 6)
            self.assertEqual(normalized["schema_id"], "0x4c23837d")
            self.assertIn("SwiGLU 等待/调度间隙", rendered_html)
            self.assertIn("推断间隙", rendered_html)
            order = load_json(directory / "rank0.order-check.json")
            self.assertEqual(order["status"], "passed")
            self.assertEqual(order["span_summary"]["process"]["count"], 108)

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
            with self.assertRaisesRegex(ContractError, "missing inside captured_count"):
                validate_timestamp_order(normalized, SPEC)

    def test_fine_expert_slot_truncation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw_path, metadata_path = make_synthetic_capture(directory)
            spec = load_json(SPEC)
            protocol = spec["trace"]["protocol"]
            tail_word = int(protocol["header_words"]) + 108 * int(
                protocol["record_words"]
            )
            raw = bytearray(raw_path.read_bytes())
            offset = tail_word * 4
            meta = int.from_bytes(raw[offset : offset + 4], "little")
            raw[offset : offset + 4] = (meta | (1 << 22)).to_bytes(4, "little")
            raw_path.write_bytes(raw)
            metadata = load_json(metadata_path)
            metadata["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(metadata_path, metadata)

            with self.assertRaisesRegex(
                ContractError, "exceeded its 4 active-expert slots"
            ):
                decode_artifacts(
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

            results = decode_artifacts(
                artifact_dir=directory,
                spec_path=SPEC,
                audit_path=AUDIT,
                trace_tool=TRACE_TOOL,
                python=sys.executable,
                allow_truncated=True,
                expected_source_sha="synthetic-fixture",
                coarse_spec_path=COARSE_SPEC,
                coarse_audit_path=COARSE_AUDIT,
            )
            self.assertEqual(len(results), 1)
            order = load_json(directory / "rank0.order-check.json")
            self.assertIn("AIC-00/gmm1_expert", order["fine_truncated_records"])

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
        compile(transformed, "adapted_mega_moe.py", "exec")


if __name__ == "__main__":
    unittest.main()

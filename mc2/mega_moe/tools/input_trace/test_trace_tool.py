# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic checks for protocol safety and the reused report renderer."""

import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import trace_tool as trace
from critical_path_report import write_report_bundle


def capture_words(wrap=False, truncated=False):
    capacity, cores = 4, 2
    rw = trace.RECORD_HEADER_WORDS + capacity * trace.EVENT_WORDS
    words = [
        trace.MAGIC,
        trace.VERSION,
        trace.HEADER_WORDS,
        rw,
        trace.EVENT_WORDS,
        cores * 3,
        cores,
        capacity,
        72,
        5120,
        8,
        1,
        0,
        1,
        0,
        0,
    ]
    for role in range(3):
        for core in range(cores):
            begin = (0xFFFFFF80 if wrap else 1000) + role * 8 + core * 2
            at = lambda x: (begin + x) & 0xFFFFFFFF
            events = [
                [12, 0, 0, 0, 0, 0, 1, at(20), at(40)],
                [1, 0, 0, 0, 0, 0, 0, at(10), at(80)],
                [11, 0, 0, 0, 0, 0, 0, at(100), at(100)],
            ]
            record = [role, core, len(events), int(truncated), at(0), at(300), 0, 0]
            for event in events:
                record.extend(event)
            record.extend([0] * (rw - len(record)))
            words.extend(record)
    return words


class TraceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "rank0.raw"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, words, path=None):
        (path or self.raw).write_bytes(struct.pack(f"<{len(words)}I", *words))

    def test_kernel_schema(self):
        kernel = (
            Path(__file__).resolve().parents[2]
            / "op_kernel/arch35/common/mega_moe_input_trace.h"
        )
        ids = {
            int(value)
            for value in re.findall(r"^    [A-Z_]+ = (\d+),$", kernel.read_text(), re.M)
        }
        self.assertEqual(ids, set(trace.EVENTS))
        self.assertEqual(ids, set(trace.POINTS))

    def test_moe_count_ready_wait(self):
        words = capture_words()
        words[16 + 8] = 15
        words[16 + 8 + 6] = 0
        self.write(words)
        event = trace.parse_capture(self.raw)["records"][0]["events"][0]
        self.assertTrue(event["is_wait_event"])
        self.assertEqual(event["duration_cycles"], 20)

    def test_parse_and_wrap(self):
        for wrap in (False, True):
            self.write(capture_words(wrap=wrap))
            parsed = trace.parse_capture(self.raw)
            self.assertEqual(len(parsed["records"]), 6)
            self.assertEqual(parsed["records"][0]["process_duration_cycles"], 300)
            self.assertEqual(parsed["records"][0]["events"][0]["duration_cycles"], 20)
            self.assertIsNone(parsed["records"][0]["events"][1]["wait_cycles"])

    def test_invalid_headers(self):
        for field, value in (
            (0, 0),
            (1, 2),
            (2, 8),
            (3, 4),
            (4, 8),
            (5, 5),
            (6, 0),
            (7, 0),
            (7, 33),
            (11, 0),
        ):
            words = capture_words()
            words[field] = value
            self.write(words)
            with self.assertRaises(ValueError):
                trace.parse_capture(self.raw)

    def test_bad_records(self):
        for field, value in ((0, 2), (1, 9), (2, 5), (3, 3)):
            words = capture_words()
            words[16 + field] = value
            self.write(words)
            with self.assertRaises(ValueError):
                trace.parse_capture(self.raw)

    def test_truncated_storage_and_header(self):
        for words in (capture_words()[:-1], [0] * 4):
            self.write(words)
            with self.assertRaises(ValueError):
                trace.parse_capture(self.raw)

    def test_parent_validation(self):
        words = capture_words()
        words[16 + 8 + 8] = 1090  # wait ends after parent
        self.write(words)
        with self.assertRaises(ValueError):
            trace.parse_capture(self.raw)

    def test_truncation_flag(self):
        self.write(capture_words(truncated=True))
        self.assertTrue(
            all(r["truncated"] for r in trace.parse_capture(self.raw)["records"])
        )

    def test_reports_and_multirank(self):
        self.write(capture_words())
        self.write(capture_words(wrap=True), self.root / "rank1.raw")
        self.assertEqual(len(trace.discover([self.root])), 2)
        output = self.root / "report"
        subprocess.run(
            [
                sys.executable,
                str(Path(trace.__file__)),
                str(self.root),
                "--output-dir",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        self.assertTrue((output / "run-summary.json").exists())
        for folder in (output / "000_rank0", output / "001_rank1"):
            for name in ("trace.json", "events.csv", "timeline.svg", "index.html"):
                self.assertTrue((folder / name).exists())
            ElementTree.parse(folder / "timeline.svg")
            page = (folder / "index.html").read_text()
            for token in ("<canvas", "Worker", "Blob", "CountTableCheckWait"):
                self.assertIn(token, page)
            node = shutil.which("node")
            if node:
                for index, (attrs, script) in enumerate(
                    re.findall(r"<script([^>]*)>(.*?)</script>", page, re.S)
                ):
                    if 'type="application/json"' in attrs:
                        json.loads(script)
                        continue
                    source = self.root / f"script{index}.js"
                    source.write_text(script)
                    subprocess.run(
                        [node, "--check", str(source)], check=True, capture_output=True
                    )

    def test_mode2_fence(self):
        words = capture_words()
        words[11] = 2
        words[16 + 8] = 9
        words[16 + 8 + 6] = 0
        self.write(words)
        self.assertEqual(
            trace.parse_capture(self.raw)["records"][0]["events"][0]["event_id"], 9
        )


if __name__ == "__main__":
    unittest.main()

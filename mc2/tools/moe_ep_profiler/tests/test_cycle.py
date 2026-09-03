from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

from mc2.tools.moe_ep_profiler.analyze_cycle import (
    HEADER_BYTES,
    MAGIC,
    VERSION,
    analyze,
)


def _event(op: int, engine: int, stage: int, boundary: int) -> int:
    return (op << 28) | (1 << 24) | (engine << 20) | (stage << 1) | boundary


def _write_raw(path: Path, op: int, invocations: int, slots: int = 8) -> None:
    max_cores = 2
    stride = 64 + slots * 16
    data = bytearray(HEADER_BYTES + max_cores * stride)
    struct.pack_into(
        "<16I",
        data,
        0,
        MAGIC,
        VERSION,
        64,
        op,
        1,
        max_cores,
        slots,
        16,
        64,
        stride,
        1,
        32,
        0,
        0,
        0,
        0,
    )
    records: list[tuple[int, int, int]] = []
    for invocation in range(invocations):
        base = 10_000 + invocation * 1_000
        records.extend(
            [
                (_event(op, 1, 0, 0), invocation, base),
                (_event(op, 2, 1, 0), invocation, base + 100),
                (_event(op, 2, 1, 1), invocation, base + 300),
                (_event(op, 1, 0, 1), invocation, base + 500),
            ]
        )
    control = HEADER_BYTES
    write_count = len(records)
    flags = 1 if write_count > slots else 0
    struct.pack_into(
        "<QIIQQI",
        data,
        control,
        write_count,
        write_count % slots,
        flags,
        0,
        invocations,
        4,
    )
    records_base = control + 64
    for index, record in enumerate(records):
        struct.pack_into("<IIQ", data, records_base + (index % slots) * 16, *record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class CycleAnalysisTest(unittest.TestCase):
    def test_wrapped_multi_rank_tail_and_visualizations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            _write_raw(run / "raw" / "rank_000" / "moe_ep_dispatch.profbin", 1, 3)
            _write_raw(
                run / "raw" / "rank_001" / "moe_ep_dispatch_epilogue.profbin", 2, 3
            )
            (run / "raw" / "rank_000" / "moe_ep_combine.profbin").write_bytes(bytes(64))
            with (run / "OpBasicInfo.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=["Op Type", "Current Freq"])
                writer.writeheader()
                writer.writerow(
                    {"Op Type": "MoeEpDispatch", "Current Freq": "1600 MHz"}
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                manifest = analyze(run, run / "analysis", warmup=0, tail_count=1)

            self.assertEqual(
                manifest["frequency"]["source"], "msprof.OpBasicInfo.Current Freq"
            )
            self.assertEqual(manifest["frequency"]["mhz"], 1600.0)
            self.assertEqual(len(manifest["unused_raw_files"]), 1)
            self.assertTrue(
                any("ring wrapped" in item for item in manifest["warnings"])
            )
            analysis = run / "analysis"
            for name in (
                "events.csv",
                "intervals.csv",
                "function_summary.csv",
                "tail_summary.csv",
                "function_summary.svg",
                "functions_rank000.trace.json",
                "logical_pipeline_rank000.svg",
                "logical_pipeline_rank000.trace.json",
            ):
                self.assertTrue((analysis / name).is_file(), name)
            trace = json.loads(
                (analysis / "logical_pipeline_rank000.trace.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(trace["traceEvents"]), 1)
            self.assertTrue(trace["traceEvents"][0]["args"]["relative_time_only"])
            with (analysis / "tail_summary.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["scope"] for row in rows}, {"last_1"})
            self.assertEqual({row["count"] for row in rows}, {"1"})


if __name__ == "__main__":
    unittest.main()

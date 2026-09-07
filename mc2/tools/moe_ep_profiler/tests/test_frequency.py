from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mc2.tools.moe_ep_profiler import analyze_cycle, analyze_msprof, run_msprof
from mc2.tools.moe_ep_profiler.tests.test_cycle import _write_raw


class FrequencyOverrideTest(unittest.TestCase):
    def test_cycle_override_skips_detection_and_converts_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _write_raw(run / "raw/rank_000/moe_ep_combine.profbin", 3, 1)
            with mock.patch.object(analyze_cycle, "resolve_frequency_mhz") as detect:
                manifest = analyze_cycle.analyze(
                    run,
                    run / "analysis",
                    warmup=0,
                    tail_count=1,
                    frequency_mhz_override=1000,
                )
            detect.assert_not_called()
            self.assertEqual(manifest["frequency"]["mhz"], 1000.0)
            self.assertEqual(
                manifest["frequency"]["source"], "cli --frequency-mhz override"
            )
            with (run / "analysis/intervals.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            kernel = next(row for row in rows if row["function"] == "KERNEL")
            self.assertEqual(float(kernel["duration_cycles"]), 500)
            self.assertEqual(float(kernel["duration_us"]), 0.5)

    def test_msprof_override_does_not_rescale_existing_microseconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "OpBasicInfo.csv").write_text(
                "Op Type,Current Freq\nMoeEpCombine,1650\n", encoding="utf-8"
            )
            (root / "PipeUtilization.csv").write_text(
                "Op Type,block_id,sub_block_id,aiv_mte2_time(us),aiv_mte2_ratio\n"
                "MoeEpCombine,0,0,2.5,30\n",
                encoding="utf-8",
            )
            with mock.patch.object(analyze_msprof, "resolve_frequency_mhz") as detect:
                manifest = analyze_msprof.analyze(root, root / "analysis", 1000)
            detect.assert_not_called()
            self.assertEqual(manifest["frequency"]["mhz"], 1000.0)
            with (root / "analysis/pipeline_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["time_us"]), 2.5)

    def test_all_cli_entry_points_forward_frequency(self) -> None:
        for module in (analyze_cycle, analyze_msprof):
            with (
                self.subTest(module=module.__name__),
                mock.patch.object(module, "analyze", return_value={}) as analyze,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(module.main(["input", "--frequency-mhz", "1000"]), 0)
                self.assertEqual(analyze.call_args.args[-1], 1000.0)
        with mock.patch.object(run_msprof, "run_msprof", return_value=0) as run:
            self.assertEqual(
                run_msprof.main(["--frequency-mhz", "1000", "--", "python", "test.py"]),
                0,
            )
            self.assertEqual(run.call_args.args[-1], 1000.0)

    def test_msprof_runner_forwards_frequency_and_disables_cycle(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                run_msprof.subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as command,
            mock.patch.object(run_msprof, "analyze") as analyze,
        ):
            self.assertEqual(
                run_msprof.run_msprof(
                    ["python", "test.py"], Path(directory), frequency_mhz_override=1000
                ),
                0,
            )
            self.assertEqual(analyze.call_args.args[-1], 1000)
            self.assertEqual(
                command.call_args.kwargs["env"]["MOE_EP_CYCLE_PROFILER"], "0"
            )

    def test_reference_patch_passes_explicit_system_cycle_frequency(self) -> None:
        patch = (
            Path(__file__).parents[1] / "patches/moe_train_test_cycle_profiler.patch"
        )
        text = patch.read_text(encoding="utf-8")
        self.assertIn('str(cycle_profiler_run_dir), "--frequency-mhz", "1000"', text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from mc2.tools.moe_ep_profiler.analyze_msprof import analyze


class MsprofAnalysisTest(unittest.TestCase):
    def test_exact_pipeline_columns_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "collection"
            op_dir = root / "OPPROF_1"
            op_dir.mkdir(parents=True)
            with (op_dir / "OpBasicInfo.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=["Op Type", "Current Freq"])
                writer.writeheader()
                writer.writerow({"Op Type": "MoeEpDispatch", "Current Freq": "1550"})
            fields = [
                "Op Type",
                "block_id",
                "sub_block_id",
                "aiv_mte2_time(us)",
                "aiv_mte2_ratio",
                "aiv_mte2_wait_ratio",
                "aiv_mte2_active_bw(GB/s)",
                "aiv_vec_time(us)",
                "aiv_vec_ratio",
            ]
            with (op_dir / "PipeUtilization.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "Op Type": "MoeEpDispatch",
                        "block_id": 0,
                        "sub_block_id": 0,
                        "aiv_mte2_time(us)": 2.5,
                        "aiv_mte2_ratio": 30,
                        "aiv_mte2_wait_ratio": 99,
                        "aiv_mte2_active_bw(GB/s)": 123,
                        "aiv_vec_time(us)": 4,
                        "aiv_vec_ratio": 60,
                    }
                )

            output = root / "analysis"
            manifest = analyze(root, output)
            self.assertEqual(manifest["frequency"]["mhz"], 1550.0)
            self.assertEqual(manifest["metrics_rows"], 2)
            self.assertEqual(len(manifest["raw_files"]), 2)
            with (output / "pipeline_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            by_pipeline = {row["pipeline"]: row for row in rows}
            self.assertEqual(by_pipeline["mte2"]["role"], "AIV")
            self.assertEqual(by_pipeline["mte2"]["ratio"], "30.0")
            self.assertEqual(by_pipeline["mte2"]["active_bw_gbps"], "123.0")
            self.assertTrue(
                (output / "raw_msprof" / "OPPROF_1" / "PipeUtilization.csv").is_file()
            )
            self.assertTrue((output / "msprof_pipeline.svg").is_file())


if __name__ == "__main__":
    unittest.main()

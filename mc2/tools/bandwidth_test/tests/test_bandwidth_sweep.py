"""CPU-only sweep/report regression tests; no CANN or Torch required."""

import ast
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bandwidth_sweep as sweep
from config import CONFIG


class SweepTests(unittest.TestCase):
    def test_driver_payload_formula(self):
        # Execute the actual reporting helper without importing the NPU driver module.
        source = Path(sweep.__file__).with_name("bandwidth_test.py").read_text(encoding="utf-8")
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "win_stream_rank_data_bytes")
        namespace = {"CONFIG": CONFIG, "mode": 3, "effective_shared_jetty_num": 15}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "payload_helper", "exec"), namespace)
        for peers in (0, 1, 2, 4, 8, 15):
            namespace["CONFIG"] = replace(CONFIG, model=replace(CONFIG.model, peers_per_jetty=peers))
            self.assertEqual(namespace["win_stream_rank_data_bytes"](32, 8192, 16),
                             15 * (peers or 15) * 32 * 8192)
        namespace["mode"] = 4
        self.assertEqual(namespace["win_stream_rank_data_bytes"](32, 8192, 16), 15 * 32 * 8192)

    def test_matrix(self):
        self.assertEqual(sweep.peer_counts(2), [1])
        self.assertEqual(sweep.peer_counts(9), [1, 2, 4, 8])
        self.assertEqual(sweep.peer_counts(16), [1, 2, 4, 8, 15])
        self.assertEqual(sweep.peer_counts(32), [1, 2, 4, 8, 16, 31])
        cases = sweep.sweep_cases(16)
        self.assertEqual(len(cases), 12)
        self.assertEqual([case[2] for case in cases[:6]], [1, 2, 4, 8, 16, 32])
        self.assertTrue(all(case[2] == 15 for case in cases[6:]))
        self.assertEqual(cases[-1], ("mode4_baseline", 4, 15, 1))
        with self.assertRaises(ValueError):
            sweep.peer_counts(1)

    def test_child_inheritance(self):
        case = {"mode": 3, "jettys": 15, "peers_attr": 4, "master_port": 50123,
                "profiling_path": "/tmp/sweep_profile"}
        code = ("from config import CONFIG; import json; "
                "print(json.dumps([CONFIG.model.mode, CONFIG.model.shared_jetty_num, "
                "CONFIG.model.peers_per_jetty, CONFIG.run.sweep_enabled, CONFIG.parallel.master_port]))")
        env = dict(os.environ, BANDWIDTH_TEST_CASE=json.dumps(case))
        output = subprocess.check_output([sys.executable, "-c", code], env=env,
                                         cwd=Path(sweep.__file__).parent, text=True)
        self.assertEqual(json.loads(output), [3, 15, 4, False, 50123])

    def fake_run(self, command, env, check):
        case = json.loads(env["BANDWIDTH_TEST_CASE"])
        self.assertTrue(check)
        self.assertEqual(Path(command[1]).name, "bandwidth_test.py")
        self.seen.append(case)
        model = replace(CONFIG.model, mode=case["mode"], shared_jetty_num=case["jettys"])
        rank_bytes = case["jettys"] * case["peers"] * 1024 * 2
        result = {"shared_jetty_batch_tokens": 1, "sampled_ranks": list(range(8)),
                  "config_index": 1, "bs": 1, "data_size": 1024, "single_rank_data_kb": rank_bytes / 1024,
                  "mean_time_us": 8.0, "min_time_us": 7.0, "max_time_us": 9.0,
                  "mean_bandwidth_gbps": rank_bytes / (8e-6 * 2**30),
                  "max_bandwidth_gbps": rank_bytes / (7e-6 * 2**30)}
        with patch.dict(os.environ, {"BANDWIDTH_TEST_CASE": json.dumps(case)}):
            sweep.save_case_results(replace(CONFIG, model=model), [result], 0, 16, 64)

    def test_orchestration_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(CONFIG, parallel=replace(CONFIG.parallel, server_num=2, rank_per_dev=8,
                                                      server_index=0, is_cloud=False, aiv_num=64),
                             shape=replace(CONFIG.shape, test_configs=[{"bs": 1, "data_size": 1024, "max_bs": 1}]),
                             run=replace(CONFIG.run, sweep_output_path=directory))
            self.seen = []
            with patch.object(sweep.subprocess, "run", side_effect=self.fake_run):
                sweep.run_sweep(config)
            self.assertEqual(len(self.seen), 12)
            self.assertEqual(len({case["master_port"] for case in self.seen}), 12)
            report = next(Path(directory).rglob("results.csv"))
            with report.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["mean_time_us"], "8.0")  # No second subtraction in reporting.
            self.assertEqual(int(rows[0]["rank_data_bytes"]), 15 * 2048)
            self.assertEqual(int(rows[6]["rank_data_bytes"]), 15 * 2048)
            self.assertEqual(int(rows[-1]["rank_data_bytes"]), 15 * 2048)
            self.assertNotIn("raw_time", report.read_text(encoding="utf-8-sig"))
            self.assertIn("## mode4_baseline", report.with_suffix(".md").read_text(encoding="utf-8"))
            manifest = json.loads((report.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")

    def test_failure_preserves_completed_group(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(CONFIG, parallel=replace(CONFIG.parallel, server_num=2, rank_per_dev=8,
                                                      server_index=0, is_cloud=False, aiv_num=64),
                             shape=replace(CONFIG.shape, test_configs=[{"bs": 1, "data_size": 1024, "max_bs": 1}]),
                             run=replace(CONFIG.run, sweep_output_path=directory))
            self.seen = []

            def fail_second(*args, **kwargs):
                if self.seen:
                    raise subprocess.CalledProcessError(1, args[0])
                return self.fake_run(*args, **kwargs)

            with patch.object(sweep.subprocess, "run", side_effect=fail_second):
                with self.assertRaises(subprocess.CalledProcessError):
                    sweep.run_sweep(config)
            report = next(Path(directory).rglob("results.csv"))
            manifest = json.loads((report.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_cases"], 1)
            self.assertTrue(report.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()

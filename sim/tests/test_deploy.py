# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise deployment orchestration with stub commands, without running npusim."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Deploy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mega deploy ")
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo with spaces"
        self.repo.mkdir()
        shutil.copy2(ROOT / "mega-sim-deploy", self.repo / "mega-sim-deploy")
        config_dir = self.repo / "sim/configs"
        config_dir.mkdir(parents=True)
        (config_dir / "bs72.json").write_text("{}")
        self.trace = self.repo / "trace.jsonl"
        self.build_env = self.repo / "build env.sh"
        self.build_env.write_text(
            "export MEGA_TEST_TOOLKIT=compiler\nexport MEGA_TEST_BUILD_ONLY=1\n"
        )
        self.sim_env = self.repo / "sim env.sh"
        self.sim_env.write_text(
            "export MEGA_TEST_TOOLKIT=simulator\nexport PATH="
            + shlex.quote(str(self.repo))
            + ':"$PATH"\n'
        )
        self.env = dict(os.environ, MEGA_TEST_TRACE=str(self.trace))
        for key in ("MEGA_TEST_TOOLKIT", "MEGA_TEST_BUILD_ONLY", "MEGA_TEST_FAIL"):
            self.env.pop(key, None)
        common = """#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ['MEGA_TEST_TRACE'], 'a') as f:
    f.write(json.dumps({'command': pathlib.Path(sys.argv[0]).name,
        'args': sys.argv[1:], 'cwd': os.getcwd(),
        'toolkit': os.environ.get('MEGA_TEST_TOOLKIT'),
        'build_only': os.environ.get('MEGA_TEST_BUILD_ONLY')}) + '\\n')
"""
        stub = self.repo / "mega-sim"
        stub.write_text(
            common
            + """
if sys.argv[1] == os.environ.get('MEGA_TEST_FAIL'):
    sys.exit(7)
if sys.argv[1] == 'prepare':
    pathlib.Path(sys.argv[sys.argv.index('--case') + 1]).mkdir()
"""
        )
        stub.chmod(0o755)
        stub = self.repo / "npusim"
        stub.write_text(common)
        stub.chmod(0o755)

    def launch(self, *args):
        result = subprocess.run(
            [
                str(self.repo / "mega-sim-deploy"),
                "--build-env",
                str(self.build_env),
                "--sim-env",
                str(self.sim_env),
                *map(str, args),
            ],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
        )
        records = [json.loads(line) for line in self.trace.read_text().splitlines()]
        return result, records

    def test_new_case_separate_environments_and_simulator_cwd(self):
        case = self.repo / "case with spaces"
        result, records = self.launch("--case", case)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [r["args"][0] for r in records], ["test", "prepare", "validate", "-c"]
        )
        self.assertTrue(all(r["toolkit"] == "compiler" for r in records[:-1]))
        self.assertEqual(records[-1]["toolkit"], "simulator")
        self.assertIsNone(records[-1]["build_only"])
        self.assertEqual(records[-1]["cwd"], str(case))
        self.assertEqual(records[-1]["args"], ["-c", str(case)])

    def test_transferred_inputs_use_their_own_config(self):
        case = self.repo / "transferred inputs"
        case.mkdir()
        (case / "config.json").write_text('{"custom": true}')
        (case / "top.pending.json").write_text("{}")
        result, records = self.launch("--inputs", case)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [r["args"][0] for r in records],
            ["test", "relocate", "build", "attach", "validate", "-c"],
        )
        self.assertIn("--inputs-only", records[1]["args"])
        self.assertEqual(
            records[2]["args"], ["build", "--config", str(case / "config.json")]
        )

    def test_failed_validation_never_starts_simulator(self):
        self.env["MEGA_TEST_FAIL"] = "validate"
        result, records = self.launch("--case", self.repo / "bad case")
        self.assertEqual(result.returncode, 7)
        self.assertTrue(all(r["command"] != "npusim" for r in records))


if __name__ == "__main__":
    unittest.main()

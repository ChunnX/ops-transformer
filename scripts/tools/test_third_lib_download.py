#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Ethan Zou.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host CLI regressions for third-party Git clone failure reporting."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("third_lib_download.py")
# Execute the real script's __main__. Only public HTTPS sources are redirected
# to local fixtures; Git clone, its exit status, and files are real.
DRIVER = r"""
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

script, source, mode = sys.argv[1:]
source = Path(source)
real_run = subprocess.run

def download(url, destination):
    Path(destination).write_bytes(b'Host download fixture')
    return destination, None

def clone(args, **kwargs):
    repo_name = args[2].rsplit('/', 1)[-1][:-4]
    if mode == 'missing-git':
        raise FileNotFoundError('git unavailable in Host fixture')
    if mode == 'existing':
        raise AssertionError('An existing repository must not be cloned again')
    local_source = source / ('absent.git' if mode == repo_name else 'source.git')
    return real_run([args[0], args[1], str(local_source), args[3]], **kwargs)

with patch('urllib.request.urlretrieve', side_effect=download):
    with patch('subprocess.run', side_effect=clone):
        runpy.run_path(script, run_name='__main__')
"""


class OfflineCloneExitStatusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.git"
        subprocess.run(
            ["git", "init", "--bare", str(self.source)],
            check=True,
            capture_output=True,
        )
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.script = self.tools / SCRIPT.name
        self.script.write_bytes(SCRIPT.read_bytes())

    def run_downloader(self, mode):
        return subprocess.run(
            [sys.executable, "-c", DRIVER, str(self.script), str(self.root), mode],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_failed_opbase_clone_is_not_success(self):
        result = self.run_downloader("opbase")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("克隆失败", result.stdout)
        self.assertFalse((self.tools / "opbase").exists())
        self.assertTrue((self.tools / "ops-tensor" / ".git").is_dir())

    def test_failed_ops_tensor_clone_is_not_success(self):
        result = self.run_downloader("ops-tensor")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("克隆失败", result.stdout)
        self.assertTrue((self.tools / "opbase" / ".git").is_dir())
        self.assertFalse((self.tools / "ops-tensor").exists())

    def test_missing_git_is_not_success(self):
        result = self.run_downloader("missing-git")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("git 命令未找到", result.stdout)

    def assert_downloads_present(self):
        files = [path for path in self.tools.iterdir() if path.is_file()]
        self.assertEqual(len(files), 9)  # script plus eight downloaded files
        for path in files:
            if path != self.script:
                self.assertEqual(path.read_bytes(), b"Host download fixture")

    def test_successful_real_clones_remain_successful(self):
        result = self.run_downloader("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("opbase", "ops-tensor"):
            subprocess.run(
                ["git", "-C", str(self.tools / name), "rev-parse", "--git-dir"],
                check=True,
                capture_output=True,
            )
        self.assert_downloads_present()

    def test_existing_repositories_still_skip_clone(self):
        for name in ("opbase", "ops-tensor"):
            subprocess.run(
                ["git", "clone", str(self.source), str(self.tools / name)],
                check=True,
                capture_output=True,
            )
        result = self.run_downloader("existing")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("跳过克隆", result.stdout)
        self.assert_downloads_present()


if __name__ == "__main__":
    unittest.main()

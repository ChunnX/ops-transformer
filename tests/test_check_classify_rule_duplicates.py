# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).parents[1] / "scripts" / "check_classify_rule_duplicates.py"


class CheckClassifyRuleDuplicatesTest(unittest.TestCase):
    def run_checker(self, content: str) -> subprocess.CompletedProcess[str]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8"
        ) as rule_file:
            rule_file.write(content)
            rule_file.flush()
            return subprocess.run(
                [sys.executable, str(CHECKER), rule_file.name],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_unique_items_pass(self):
        result = self.run_checker(
            "owner:\n  release:\n    - path/to/dir\n    - path/to/dir/file.cpp\n"
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_duplicate_items_fail_with_both_locations(self):
        result = self.run_checker(
            "first:\n  - duplicate/path\nsecond:\n  - duplicate/path\n"
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn(":4:5: duplicate item: duplicate/path", result.stdout)
        self.assertIn(":2:5: first defined here", result.stdout)


if __name__ == "__main__":
    unittest.main()

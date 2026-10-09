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
"""Host CLI regressions using real shared libraries and GNU nm."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "get_opapi_abs_path.py"
SYMBOL = "aclnnExample"


class OpApiSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.builtin = self.root / "opp" / "lib64" / "libopapi.so"
        self._compile(self.builtin, f"void {SYMBOL}(void) {{}}")

    def _compile(self, library, source, versioned=False):
        library.parent.mkdir(parents=True, exist_ok=True)
        code = library.with_suffix(".c")
        code.write_text(source, encoding="utf-8")
        args = ["gcc", "-shared", "-fPIC", str(code), "-o", str(library)]
        if versioned:
            version = library.with_suffix(".map")
            version.write_text(f"EXAMPLE_1 {{ global: {SYMBOL}; local: *; }};")
            args.append(f"-Wl,--version-script={version}")
        subprocess.run(args, check=True, capture_output=True, text=True)

    def _select(self, source, spaced=False, versioned=False):
        custom = self.root / ("custom opp" if spaced else "custom")
        library = custom / "op_api" / "lib" / "libcust_opapi.so"
        self._compile(library, source, versioned)
        env = os.environ.copy()
        env.update(
            ASCEND_CUSTOM_OPP_PATH=str(custom),
            ASCEND_OPP_PATH=str(self.root / "opp"),
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--func", SYMBOL],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout), library

    def test_exact_custom_export_keeps_priority(self):
        selected, custom = self._select(f"void {SYMBOL}(void) {{}}")
        self.assertEqual(selected, custom)

    def test_prefix_only_export_falls_back_to_builtin(self):
        selected, _ = self._select(f"void {SYMBOL}Extra(void) {{}}")
        self.assertEqual(selected, self.builtin)

    def test_undefined_import_falls_back_to_builtin(self):
        selected, _ = self._select(
            f"extern void {SYMBOL}(void); void wrapper(void) {{ {SYMBOL}(); }}"
        )
        self.assertEqual(selected, self.builtin)

    def test_custom_path_with_spaces_is_one_nm_argument(self):
        selected, custom = self._select(f"void {SYMBOL}(void) {{}}", spaced=True)
        self.assertEqual(selected, custom)

    def test_default_versioned_export_keeps_priority(self):
        selected, custom = self._select(f"void {SYMBOL}(void) {{}}", versioned=True)
        self.assertEqual(selected, custom)


if __name__ == "__main__":
    unittest.main()

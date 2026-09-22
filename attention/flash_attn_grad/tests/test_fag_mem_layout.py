#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Host-only regression checks for the shared newDeter UB layout."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / "op_kernel"


def load_fag_mem():
    """Load the layout calculator without requiring a PyPTO installation."""
    fake_package = types.ModuleType("pypto_pro")
    fake_language = types.ModuleType("pypto_pro.language")
    fake_package.language = fake_language
    saved = {
        name: sys.modules.get(name)
        for name in ("pypto_pro", "pypto_pro.language", "fag_common")
    }
    old_path = list(sys.path)
    try:
        sys.modules["pypto_pro"] = fake_package
        sys.modules["pypto_pro.language"] = fake_language
        sys.modules.pop("fag_common", None)
        sys.path.insert(0, str(KERNEL_DIR))
        spec = spec_from_file_location("fag_mem_under_test", KERNEL_DIR / "fag_mem.py")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


MEM = load_fag_mem()


class TestDeterministicUbLayout(unittest.TestCase):
    def test_no_private_descriptor_or_merge_storage(self):
        for width in (64, 128, 192, 256, 768):
            with self.subTest(width=width):
                layout = MEM._UB[width]
                self.assertNotIn("deter_desc", layout)
                for arena in layout["_arenas"]:
                    labels = [record[0] for record in arena.map]
                    self.assertFalse(any("deter" in label for label in labels))
                    self.assertFalse(any("merge" in label for label in labels))

    def test_buffer_and_address_modules_have_no_epoch_storage(self):
        for filename in ("fag_buffers.py", "fag_mem.py"):
            source = (KERNEL_DIR / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertNotIn("DETER_EPOCH", source)
                self.assertNotIn("DETER_DESC", source)
                self.assertNotIn("deter_desc", source)
                self.assertNotIn("deter_merge", source)


if __name__ == "__main__":
    unittest.main()

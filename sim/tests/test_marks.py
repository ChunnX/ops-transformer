# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Keep debug IDs usable across kernel updates without changing the kernel ABI."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marks import KERNEL, catalog, check_sites, decode


class Marks(unittest.TestCase):
    def test_stable_tags_and_current_coverage(self):
        entries = catalog()
        self.assertEqual(entries[0x001]["name"], "KERNEL")
        self.assertEqual(entries[0x209]["name"], "WAIT_DISPATCH_DATA")
        self.assertIn("BEGIN WAIT_DISPATCH_DATA", decode(0x01000209, entries))
        self.assertIn("END   KERNEL", decode(33554433, entries))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            decode(0x03000001, entries)
        self.assertEqual(len(check_sites()), len(entries))

    def test_upstream_update_losing_wait_marks_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "arch35"
            shutil.copytree(KERNEL, root)
            path = root / "stage/mega_moe_gmm1_activation.h"
            path.write_text(
                path.read_text().replace("MEGA_MOE_MARK_END(WAIT_DISPATCH_DATA);", "")
            )
            with self.assertRaisesRegex(ValueError, "WAIT_DISPATCH_DATA"):
                check_sites(root)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "marks.h"
            header.write_text(
                "    FIRST = 0x001, // first\n    SECOND = 0x001, // second\n"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                catalog(header)

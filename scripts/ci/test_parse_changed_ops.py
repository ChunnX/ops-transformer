# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression coverage for CI operator selection after an experimental-to-standard move."""

import os
from pathlib import Path
import tempfile
import unittest

from parse_changed_ops import extract_operator_name, get_change_ops_list


class TestOperatorSelection(unittest.TestCase):
    def test_standard_operator_without_experimental_copy(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                op = Path("attention/turbo_quant_sparse_flash_attention/op_host")
                op.mkdir(parents=True)
                definition = op / "turbo_quant_sparse_flash_attention_def.cpp"
                definition.write_text(
                    'this->AICore().AddConfig("ascend910b");', encoding="utf-8"
                )
                Path("changed.txt").write_text(
                    definition.as_posix() + "\n", encoding="utf-8"
                )
                self.assertEqual(
                    get_change_ops_list("changed.txt", "FALSE", "ascend910b"),
                    "turbo_quant_sparse_flash_attention",
                )
                self.assertEqual(
                    extract_operator_name("attention/common/foo.h", "FALSE"),
                    "nsa_compress_attention_infer",
                )
                self.assertEqual(
                    extract_operator_name("attention/missing/op_host/def.cpp", "FALSE"),
                    "nsa_compress_attention_infer",
                )
                experimental = Path("experimental/attention/example/op_host")
                experimental.mkdir(parents=True)
                self.assertEqual(
                    extract_operator_name(
                        (experimental / "def.cpp").as_posix(), "TRUE"
                    ),
                    "example",
                )
                self.assertEqual(
                    extract_operator_name(
                        "experimental/attention/missing/def.cpp", "TRUE"
                    ),
                    "nsa_compress_attention_infer",
                )
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()

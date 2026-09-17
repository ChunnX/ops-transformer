#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only regression tests for the shared pytest/TTK OPP selection."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


_MODULE_PATH = Path(__file__).resolve().parents[2] / "pytest/common_utils/op_runtime.py"
_SPEC = importlib.util.spec_from_file_location("qbsa_op_runtime", _MODULE_PATH)
_RUNTIME = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNTIME)


class QbsaOpRuntimeTest(unittest.TestCase):
    def test_explicit_vendors_keep_order(self):
        first = Path("installed_first")
        second = Path("installed_second")
        selected = os.pathsep.join(map(str, (first, second)))
        first_lib = str(first / "op_api/lib")
        second_lib = str(second / "op_api/lib")
        env = {
            "ASCEND_CUSTOM_OPP_PATH": selected,
            "LD_LIBRARY_PATH": os.pathsep.join(("cann/lib64", second_lib, first_lib)),
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(Path, "is_dir", return_value=True),
        ):
            _RUNTIME.configure_custom_opp_paths()
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], selected)
            self.assertEqual(
                os.environ["LD_LIBRARY_PATH"].split(os.pathsep),
                [first_lib, second_lib, "cann/lib64"],
            )
            after_first_call = dict(os.environ)
            _RUNTIME.configure_custom_opp_paths()
            self.assertEqual(dict(os.environ), after_first_call)

    def test_no_custom_path_uses_installed_opp_even_if_build_exists(self):
        env = {"ASCEND_OPP_PATH": "installed/opp", "LD_LIBRARY_PATH": "cann/lib64"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(Path, "exists", return_value=True),
        ):
            _RUNTIME.configure_custom_opp_paths()
            self.assertEqual(dict(os.environ), env)

    def test_missing_vendor_does_not_fall_back_to_staging(self):
        env = {"ASCEND_CUSTOM_OPP_PATH": "missing/vendor"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(Path, "is_dir", return_value=False),
            patch.object(Path, "exists", return_value=True),
        ):
            _RUNTIME.configure_custom_opp_paths()
            self.assertEqual(dict(os.environ), env)


if __name__ == "__main__":
    unittest.main()

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

"""CPU-only checks for installed extension loading and registration."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


_QBSA_ROOT = Path(__file__).resolve().parents[3]
_LOADER_PATH = _QBSA_ROOT / "tests/pytest/common_utils/extension_loader.py"
_EXTENSION_INIT = _QBSA_ROOT / "torch_ops_extension/custom_ops/__init__.py"
_OP_NAMES = (
    "npu_quant_block_sparse_attn",
    "npu_quant_block_sparse_attn_metadata",
)


class QbsaExtensionLoaderTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "qbsa_extension_loader", _LOADER_PATH
        )
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)
        self.extension = ModuleType("custom_ops")
        self.extension.__file__ = "installed/custom_ops/__init__.py"
        self.torch = ModuleType("torch")
        self.torch.ops = SimpleNamespace(
            custom=SimpleNamespace(**{name: object() for name in _OP_NAMES})
        )

    def test_import_once_without_changing_environment_or_search_paths(self):
        env = {
            "ASCEND_CUSTOM_OPP_PATH": "installed/vendor",
            "LD_LIBRARY_PATH": "installed/lib",
            "QBSA_CUSTOM_OPS_PATH": "stale/build",
        }
        search_paths = list(sys.path)
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"torch": self.torch}),
            patch.object(
                self.loader.importlib, "import_module", return_value=self.extension
            ) as import_module,
        ):
            self.assertIs(self.loader.ensure_qbsa_extension(), self.extension)
            self.assertIs(self.loader.ensure_qbsa_extension(), self.extension)
            import_module.assert_called_once_with("custom_ops")
            self.assertEqual(dict(os.environ), env)
            self.assertEqual(sys.path, search_paths)

    def test_import_failure_preserves_cause_and_does_not_search_for_another_binary(
        self,
    ):
        for error in (
            ModuleNotFoundError("custom_ops"),
            OSError("missing ACLNN library"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(
                    self.loader.importlib, "import_module", side_effect=error
                ) as import_module,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "build_and_install.sh"
                ) as raised:
                    self.loader.ensure_qbsa_extension()
                self.assertIs(raised.exception.__cause__, error)
                import_module.assert_called_once_with("custom_ops")
                self.assertIsNone(self.loader._LOADED_EXTENSION)

    def test_wrong_extension_reports_missing_operator_and_package_path(self):
        for missing in _OP_NAMES:
            with self.subTest(missing=missing):
                self.torch.ops.custom = SimpleNamespace(
                    **{name: object() for name in _OP_NAMES if name != missing}
                )
                with (
                    patch.dict(sys.modules, {"torch": self.torch}),
                    patch.object(
                        self.loader.importlib,
                        "import_module",
                        return_value=self.extension,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, missing) as raised:
                        self.loader.ensure_qbsa_extension()
                    self.assertIn(self.extension.__file__, str(raised.exception))
                    self.assertIsNone(self.loader._LOADED_EXTENSION)

    def test_package_registers_both_converters_and_only_exports_qbsa(self):
        torch_npu = ModuleType("torch_npu")
        self.torch.ops.custom.npu_other_operator = object()
        binary = ModuleType("custom_ops.custom_ops_lib")
        converters = ModuleType("custom_ops.converter")
        converters.npu_quant_block_sparse_attn = ModuleType("attn_converter")
        converters.npu_quant_block_sparse_attn_metadata = ModuleType(
            "metadata_converter"
        )
        spec = importlib.util.spec_from_file_location(
            "custom_ops",
            _EXTENSION_INIT,
            submodule_search_locations=[str(_EXTENSION_INIT.parent)],
        )
        extension = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "torch_npu": torch_npu,
                "custom_ops": extension,
                "custom_ops.custom_ops_lib": binary,
                "custom_ops.converter": converters,
            },
        ):
            spec.loader.exec_module(extension)
            self.assertIs(extension.custom_ops_lib, binary)
            self.assertIs(
                extension._attn_converter, converters.npu_quant_block_sparse_attn
            )
            self.assertIs(
                extension._metadata_converter,
                converters.npu_quant_block_sparse_attn_metadata,
            )
            self.assertEqual(extension.__all__, list(_OP_NAMES))
            for name in _OP_NAMES:
                self.assertIs(
                    getattr(torch_npu, name), getattr(self.torch.ops.custom, name)
                )
            self.assertFalse(hasattr(torch_npu, "npu_other_operator"))


if __name__ == "__main__":
    unittest.main()

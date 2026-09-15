# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils.cpp_extension import load


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlashMlaMetadataTorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        repo = here.parents[4]
        builder = load_source(
            "metadata_test_builder",
            repo / "torch_extension/cann_ops_transformer/op_builder/builder.py",
        )
        # Load the real schema, registration and Python wrapper without importing
        # the package's unrelated operators or initializing a device runtime.
        torch_npu = types.ModuleType("torch_npu")
        torch_npu.__file__ = str(here / "torch_npu.py")
        with mock.patch.dict(
            sys.modules,
            {"cann_ops_transformer.op_builder": builder, "torch_npu": torch_npu},
        ):
            cls.module = load_source(
                "metadata_under_test",
                here.parents[2] / "torch_extension/flash_mla_metadata.py",
            )
        cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
        build = os.environ.get("FLASH_MLA_METADATA_UT_BUILD_DIR") or tempfile.mkdtemp(
            prefix="flash_mla_metadata_torch_ut_"
        )
        Path(build).mkdir(parents=True, exist_ok=True)
        cls.extension = load(
            name="flash_mla_metadata_torch_ut",
            sources=[str(here / "flash_mla_metadata_test_bindings.cpp")],
            extra_include_paths=[
                str(cann / "include"),
                str(repo / "torch_extension/cann_ops_transformer/common"),
            ],
            extra_cflags=["-O0", "-g0"],
            build_directory=build,
            verbose=False,
        )

    def test_schema_has_single_layout_and_integer_quant_mode(self):
        args = {
            arg.name: arg
            for arg in torch.ops.cann_ops_transformer.flash_mla_metadata.default._schema.arguments
        }
        self.assertEqual(str(args["input_layer"].type), "str")
        self.assertEqual(args["input_layer"].default_value, "BSH")
        self.assertEqual(str(args["quant_mode"].type), "int")
        self.assertEqual(args["quant_mode"].default_value, 0)
        for removed in ("layout_q", "layout_kv", "query_dtype"):
            self.assertNotIn(removed, args)
            with self.assertRaises(TypeError):
                self.module.flash_mla_metadata(8, 2, 1, 128, **{removed: 0})

    def test_meta_output_for_layouts_modes_and_empty_batch(self):
        layouts = (
            "BSH",
            "BSND",
            "BNSD",
            "TND",
            "BSH_NBSD",
            "BSND_NBSD",
            "BNSD_NBSD",
            "TND_NTD",
        )
        for layout in layouts:
            for mode in (0, 1):
                if mode == 1 and layout.startswith("BNSD"):
                    continue
                for batch in (0, 2):
                    with self.subTest(layout=layout, mode=mode, batch=batch):
                        result = self.module.flash_mla_metadata(
                            8,
                            batch,
                            1,
                            128,
                            input_layer=layout,
                            quant_mode=mode,
                            block_size=128,
                            device="meta",
                        )
                        self.assertEqual(result.shape, (480,))
                        self.assertEqual(result.dtype, torch.int32)
                        self.assertEqual(result.device.type, "meta")
                        self.assertEqual(result.storage_offset(), 0)
                        self.assertTrue(result.is_contiguous())

    def test_invalid_quant_modes_and_layouts(self):
        for mode in (-1, 2, 27, 2**63 - 1, 1.5, "1", torch.int8):
            with self.subTest(mode=mode), self.assertRaises((RuntimeError, TypeError)):
                self.module.flash_mla_metadata(
                    8, 2, 1, 128, quant_mode=mode, device="meta"
                )
        for layout in ("", "NZ", "bsh", "BSH_", "BSH_NTD", "TND_NBSD", "TND_NTD_extra"):
            with self.subTest(layout=layout), self.assertRaises(RuntimeError):
                self.module.flash_mla_metadata(
                    8, 2, 1, 128, input_layer=layout, device="meta"
                )

    def test_public_wrapper_forwards_new_arguments_in_cpp_order(self):
        seq_q = torch.tensor([1, 3], dtype=torch.int64)
        seq_kv = torch.tensor([128, 256], dtype=torch.int64)
        expected = torch.empty(480, dtype=torch.int32)
        extension = mock.Mock()
        extension.flash_mla_metadata.return_value = expected
        with mock.patch.object(
            self.module._flash_mla_metadata_op_builder, "load", return_value=extension
        ):
            result = self.module.flash_mla_metadata(
                8,
                2,
                3,
                512,
                actual_seq_lengths_q=seq_q,
                actual_seq_lengths_kv=seq_kv,
                input_layer="TND_NTD",
                block_size=128,
                sparse_mode=3,
                pre_tokens=17,
                next_tokens=19,
                has_atten_mask=True,
                quant_mode=1,
                device="cpu",
            )
            self.assertIs(result, expected)
            extension.flash_mla_metadata.assert_called_once_with(
                8,
                2,
                3,
                512,
                seq_q,
                seq_kv,
                "TND_NTD",
                128,
                3,
                17,
                19,
                True,
                1,
                torch.device("cpu"),
            )

    def test_dense_without_sequence_tensors_uses_default_arguments(self):
        extension = mock.Mock()
        extension.flash_mla_metadata.return_value = torch.empty(480, dtype=torch.int32)
        with mock.patch.object(
            self.module._flash_mla_metadata_op_builder, "load", return_value=extension
        ):
            self.module.flash_mla_metadata(8, 2, 1, 128)
            extension.flash_mla_metadata.assert_called_once_with(
                8,
                2,
                1,
                128,
                None,
                None,
                "BSH",
                0,
                0,
                2147483647,
                2147483647,
                False,
                0,
                None,
            )

    def test_fake_registration_uses_new_parameters_without_loading_extension(self):
        with (
            mock.patch.object(
                self.module._flash_mla_metadata_op_builder, "load"
            ) as load_extension,
            FakeTensorMode(),
        ):
            seq = torch.empty(2, dtype=torch.int64)
            for quant_mode in (0, 1):
                result = self.module.flash_mla_metadata(
                    8,
                    2,
                    2,
                    512,
                    actual_seq_lengths_q=seq,
                    actual_seq_lengths_kv=seq,
                    input_layer="TND_NTD",
                    block_size=128,
                    quant_mode=quant_mode,
                )
                self.assertIsInstance(result, FakeTensor)
                self.assertEqual(result.shape, (480,))
                self.assertEqual(result.dtype, torch.int32)
                self.assertEqual(result.device, seq.device)
            with self.assertRaisesRegex(RuntimeError, "quant_mode"):
                self.module.flash_mla_metadata(
                    8, 2, 2, 512, actual_seq_lengths_q=seq, quant_mode=2
                )
            load_extension.assert_not_called()

    def test_cpp_binding_rejects_invalid_mode_before_allocation(self):
        for mode in (-1, 2, 27):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(RuntimeError, "quant_mode"),
            ):
                self.extension.flash_mla_metadata(
                    8,
                    2,
                    1,
                    128,
                    None,
                    None,
                    "BSH",
                    0,
                    0,
                    0,
                    0,
                    False,
                    mode,
                    torch.device("cpu"),
                )
        for mode in (0, 1):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "NPU"):
                self.extension.flash_mla_metadata(
                    8,
                    2,
                    1,
                    128,
                    None,
                    None,
                    "BSH",
                    0,
                    0,
                    0,
                    0,
                    False,
                    mode,
                    torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()

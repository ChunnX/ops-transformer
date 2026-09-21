# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run after sourcing CANN set_env.sh: python -m unittest discover -s mc2/mega_moe/tests/torch_extension -v."""

import math
import os
from pathlib import Path
import unittest

import torch
import torch_npu
from torch.utils.cpp_extension import load


class TestInt4Wrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[4]
        cann = Path(os.environ["ASCEND_HOME_PATH"])
        npu = Path(torch_npu.__file__).parent
        cls.includes = [
            str(repo / "torch_extension/cann_ops_transformer/common"),
            str(repo / "mc2/mega_moe/torch_extension/csrc"),
            str(npu / "include"),
            str(npu / "include/third_party/acl/inc"),
            str(npu / "include/third_party/hccl/inc"),
            str(cann / "include"),
        ]
        cls.ldflags = [
            f"-L{cann}/lib64",
            "-lopapi",
            "-lascendcl",
            f"-L{npu}/lib",
            "-ltorch_npu",
        ]
        cls.bridge = load(
            name="mega_moe_int4_wrapper_test",
            sources=[str(Path(__file__).with_name("int4_wrapper_test.cpp"))],
            extra_include_paths=cls.includes,
            extra_cflags=["-O0"],
            extra_ldflags=cls.ldflags,
        )
        cls.repo = repo

    def test_int32_descriptor_and_offset(self):
        tensor = torch.arange(3 * 4 * 8, dtype=torch.int32).reshape(3, 4, 8)[1:]
        before = tensor.clone()
        desc = self.bridge.describe(tensor, self.bridge.INT4)
        self.assertEqual(desc["shape"], [2, 4, 64])
        self.assertEqual(desc["strides"], [256, 64, 1])
        self.assertEqual(desc["offset"], 256)
        self.assertEqual(
            math.prod(desc["storage_shape"]), tensor.untyped_storage().nbytes() * 2
        )
        self.assertEqual(desc["dtype"], self.bridge.INT4)
        self.assertEqual(desc["address"], tensor.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(before, tensor))

    def test_transposed_descriptor(self):
        tensor = torch.empty((2, 4, 8), dtype=torch.int32).transpose(-1, -2)
        desc = self.bridge.describe(tensor, self.bridge.INT4)
        self.assertEqual(desc["shape"], [2, 64, 4])
        self.assertEqual(desc["strides"], [256, 1, 64])

    def test_fp4_byte_carrier_unchanged(self):
        tensor = torch.empty((3, 4, 8), dtype=torch.int8)[1:]
        desc = self.bridge.describe(tensor, self.bridge.FP4)
        self.assertEqual(desc["shape"], [2, 4, 16])
        self.assertEqual(desc["strides"], [64, 16, 1])
        self.assertEqual(desc["offset"], 64)
        self.assertEqual(
            math.prod(desc["storage_shape"]), tensor.untyped_storage().nbytes() * 2
        )
        transposed = self.bridge.describe(tensor.transpose(-1, -2), self.bridge.FP4)
        self.assertEqual(transposed["shape"], [2, 16, 4])
        self.assertEqual(transposed["strides"], [64, 1, 16])
        self.assertEqual(
            self.bridge.resolve([tensor], self.bridge.DTYPE_FP4, False), self.bridge.FP4
        )

    def test_other_dtypes_and_empty_list(self):
        for dtype, acl_dtype in [
            (torch.int8, self.bridge.INT8),
            (torch.bfloat16, self.bridge.BF16),
        ]:
            tensor = torch.empty((4, 8), dtype=dtype)
            self.assertEqual(self.bridge.resolve([tensor], None, True), acl_dtype)
            desc = self.bridge.describe(tensor, acl_dtype)
            self.assertEqual(desc["shape"], [4, 8])
            self.assertEqual(desc["strides"], [8, 1])
        with self.assertRaisesRegex(RuntimeError, "must not be empty"):
            self.bridge.resolve([], None, True)
        self.bridge.check_shared([], self.bridge.INT4)
        tensor = torch.empty((4, 8), dtype=torch.int32)
        self.assertEqual(self.bridge.resolve([tensor], None, False), self.bridge.INT32)
        with self.assertRaisesRegex(RuntimeError, "FRACTAL_NZ"):
            self.bridge.resolve([tensor], None, True)

    @unittest.skipUnless(
        torch.npu.is_available(), "NPU required for NZ descriptor test"
    )
    def test_nz_int4_and_shared_weights(self):
        # Same preparation as the documented MegaMoE A8W4 example: pack bytes, NZ, view INT32.
        previous_internal_format = (
            torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        )
        torch.npu.config.allow_internal_format = True
        self.addCleanup(
            setattr, torch.npu.config, "allow_internal_format", previous_internal_format
        )
        packed_bytes = torch.arange(64 * 128, dtype=torch.int8).reshape(64, 128).npu()
        nz_bytes = torch_npu.npu_format_cast(packed_bytes, self.bridge.NZ)
        packed = nz_bytes.view(torch.int32)
        self.assertEqual(list(packed.shape), [64, 32])
        for dtype in (None, self.bridge.DTYPE_INT4, self.bridge.DTYPE_INT32):
            self.assertEqual(
                self.bridge.resolve([packed, packed], dtype, True), self.bridge.INT4
            )
        self.bridge.check_shared([packed], self.bridge.INT4)
        desc = self.bridge.describe(packed, self.bridge.INT4)
        self.assertEqual(desc["shape"], [64, 256])
        self.assertEqual(desc["strides"], [256, 1])
        self.assertEqual(desc["format"], self.bridge.NZ)
        self.assertEqual(
            math.prod(desc["storage_shape"]), packed.untyped_storage().nbytes() * 2
        )
        self.assertEqual(desc["address"], packed.untyped_storage().data_ptr())
        # The same packed NZ bytes must produce the same logical layout via the existing FP4 path.
        byte_desc = self.bridge.describe(nz_bytes, self.bridge.FP4)
        for key in ("shape", "strides", "storage_shape", "offset", "format", "address"):
            self.assertEqual(desc[key], byte_desc[key], key)
        # A tensor allocated as INT32 in NZ has INT32 storage units, unlike the byte dtype view above.
        native_int32_nz = torch_npu.npu_format_cast(
            torch.empty((64, 32), dtype=torch.int32, device="npu"), self.bridge.NZ
        )
        native_desc = self.bridge.describe(native_int32_nz, self.bridge.INT4)
        self.assertEqual(native_desc["shape"], [64, 256])
        self.assertEqual(
            math.prod(native_desc["storage_shape"]),
            native_int32_nz.untyped_storage().nbytes() * 2,
        )
        with self.assertRaisesRegex(RuntimeError, "logical dtype"):
            self.bridge.resolve([packed], self.bridge.DTYPE_FP4, True)
        with self.assertRaisesRegex(RuntimeError, "torch.int32"):
            self.bridge.resolve([packed, packed_bytes], None, True)
        with self.assertRaisesRegex(RuntimeError, "torch.int32"):
            self.bridge.check_shared([packed_bytes], self.bridge.INT4)
        with self.assertRaisesRegex(RuntimeError, "non-transposed"):
            self.bridge.resolve([packed.t()], None, True)

    def test_mega_moe_extension_build(self):
        # Compile the production entry point as well as the descriptor test bridge.
        module = load(
            name="mega_moe_int4_extension_build_test",
            sources=[str(self.repo / "mc2/mega_moe/torch_extension/csrc/mega_moe.cpp")],
            extra_include_paths=self.includes,
            extra_cflags=["-O0"],
            extra_ldflags=self.ldflags,
        )
        self.assertTrue(hasattr(module, "npu_mega_moe"))


if __name__ == "__main__":
    unittest.main()

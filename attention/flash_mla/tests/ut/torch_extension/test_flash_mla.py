# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.cpp_extension import load


class FlashMlaTorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        repo = here.parents[4]
        cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
        build = os.environ.get("FLASH_MLA_UT_BUILD_DIR") or tempfile.mkdtemp(
            prefix="flash_mla_torch_ut_"
        )
        Path(build).mkdir(parents=True, exist_ok=True)
        cls.extension = load(
            name="flash_mla_torch_ut",
            sources=[str(here / "flash_mla_test_bindings.cpp")],
            extra_include_paths=[
                str(cann / "include"),
                str(repo / "torch_extension/cann_ops_transformer/common"),
            ],
            extra_cflags=["-O0", "-g0"],
            build_directory=build,
            verbose=False,
        )

    def test_output_shapes_and_lse(self):
        cases = (
            ("BSH", (2, 3, 4096), [2, 3, 4096], [2, 8, 3, 1]),
            ("BSND", (2, 3, 8, 512), [2, 3, 8, 512], [2, 8, 3, 1]),
            ("BNSD", (2, 8, 3, 512), [2, 8, 3, 512], [2, 8, 3, 1]),
            ("TND", (3, 8, 512), [3, 8, 512], [3, 8, 1]),
            ("BSH_NBSD", (2, 3, 4096), [8, 2, 3, 512], [2, 8, 3, 1]),
            ("BSND_NBSD", (2, 3, 8, 512), [8, 2, 3, 512], [2, 8, 3, 1]),
            ("BNSD_NBSD", (2, 8, 3, 512), [8, 2, 3, 512], [2, 8, 3, 1]),
            ("TND_NTD", (3, 8, 512), [8, 3, 512], [3, 8, 1]),
        )
        for layout, shape, output, lse in cases:
            with self.subTest(layout=layout):
                query = torch.empty(shape, dtype=torch.float16)
                self.assertEqual(
                    self.extension.output_shapes(query, 8, layout, True), (output, lse)
                )
                self.assertEqual(
                    self.extension.output_shapes(query, 8, layout, False), (output, [0])
                )

    def test_invalid_shapes_are_rejected_before_allocation(self):
        cases = (
            ("BSH", (2, 1, 4096), 0),
            ("BSH", (2, 1, 4096), 129),
            ("BSH", (2, 1, 4095), 8),
            ("BSH", (2, 1, 8, 512), 8),
            ("BSH_NTD", (2, 1, 4096), 8),
            ("INVALID", (2, 1, 4096), 8),
            ("BSND", (2, 1, 4, 512), 8),
            ("TND", (2, 8, 511), 8),
        )
        for layout, shape, heads in cases:
            with (
                self.subTest(layout=layout, shape=shape, heads=heads),
                self.assertRaises(RuntimeError),
            ):
                self.extension.output_shapes(torch.empty(shape), heads, layout, False)

    def test_empty_output_shapes(self):
        cases = (
            ("BSH", (0, 1, 4096), [0, 1, 4096], [0, 8, 1, 1]),
            ("BSND_NBSD", (2, 0, 8, 512), [8, 2, 0, 512], [2, 8, 0, 1]),
            ("TND_NTD", (0, 8, 512), [8, 0, 512], [0, 8, 1]),
        )
        for layout, shape, output, lse in cases:
            with self.subTest(layout=layout):
                self.assertEqual(
                    self.extension.output_shapes(torch.empty(shape), 8, layout, True),
                    (output, lse),
                )

    def test_sequence_view_descriptor_preserves_address_and_offset(self):
        base = torch.arange(8)
        seq = base[2:6]
        desc = self.extension.describe(seq)
        self.assertEqual(desc["view"], [4])
        # The ND descriptor retains the full allocation behind the sliced view.
        self.assertEqual(desc["storage"], [8])
        self.assertEqual(desc["strides"], [1])
        self.assertEqual(desc["offset"], 2)
        self.assertEqual(desc["base"], base.data_ptr())
        self.assertEqual(seq.storage_offset(), 2)

    def test_cache_descriptors_preserve_views(self):
        base = torch.arange(30).reshape(5, 2, 3)
        cache = base[1::2]
        for tensor in (cache, cache.transpose(1, 2), base[:1].expand(2, 2, 3)):
            with self.subTest(strides=tensor.stride()):
                desc = self.extension.describe(tensor)
                self.assertEqual(desc["view"], list(tensor.shape))
                self.assertEqual(desc["storage"], [30])
                self.assertEqual(desc["strides"], list(tensor.stride()))
                self.assertEqual(desc["offset"], tensor.storage_offset())
                self.assertEqual(desc["base"], base.data_ptr())

    def test_cache_normalization_matches_fia_dim0_policy(self):
        base = torch.arange(120).reshape(5, 4, 6)
        dim0_view = base[1::2]
        for keep_dim0 in (False, True):
            for cache in (
                base,
                dim0_view,
                base.transpose(1, 2),
                base[:1].expand(2, 4, 6),
            ):
                with self.subTest(keep_dim0=keep_dim0, strides=cache.stride()):
                    result = self.extension.normalize_cache(cache, keep_dim0)
                    torch.testing.assert_close(result, cache)
                    if cache.is_contiguous() or (keep_dim0 and cache is dim0_view):
                        self.assertEqual(result.data_ptr(), cache.data_ptr())
                        self.assertEqual(result.stride(), cache.stride())
                        self.assertEqual(
                            result.storage_offset(), cache.storage_offset()
                        )
                    else:
                        self.assertTrue(result.is_contiguous())
                        self.assertNotEqual(result.data_ptr(), cache.data_ptr())

    def test_sequence_descriptor_keeps_slices_and_leaves_values_to_metadata(self):
        self.extension.check_sequence_descriptor(None, "seq")
        for seq in (
            torch.empty(0, dtype=torch.int64),
            torch.tensor([-1, 2**40]),
            torch.arange(8)[2:6],
        ):
            self.extension.check_sequence_descriptor(seq, "seq")
        for seq in (
            torch.arange(8)[::2],
            torch.empty(2, 1, dtype=torch.int64),
            torch.empty(2, dtype=torch.int32),
        ):
            with (
                self.subTest(shape=seq.shape, strides=seq.stride()),
                self.assertRaises(RuntimeError),
            ):
                self.extension.check_sequence_descriptor(seq, "seq")

    def test_metadata_descriptor_preserves_protocol_checks(self):
        self.extension.check_metadata_descriptor(torch.empty(480, dtype=torch.int32))
        invalid = (
            torch.empty(480, dtype=torch.int64),
            torch.empty(479, dtype=torch.int32),
            torch.empty(1, 480, dtype=torch.int32),
            torch.empty(960, dtype=torch.int32)[::2],
            torch.empty(496, dtype=torch.int32)[16:],
        )
        for metadata in invalid:
            with (
                self.subTest(shape=metadata.shape, offset=metadata.storage_offset()),
                self.assertRaises(RuntimeError),
            ):
                self.extension.check_metadata_descriptor(metadata)


if __name__ == "__main__":
    unittest.main()

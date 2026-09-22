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
"""Host-only checks for the newDeter public-workspace ABI."""

import ast
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
HOST = ROOT / "op_host"
KERNEL = ROOT / "op_kernel"


def _text(path):
    return path.read_text(encoding="utf-8")


def _kernel_tiling_fields():
    tree = ast.parse(_text(KERNEL / "fag_tiling.py"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FlashAttnGradTilingData":
            return [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
    raise AssertionError("FlashAttnGradTilingData not found")


def _host_tiling_fields():
    source = _text(HOST / "flash_attn_grad_tiling.h")
    body = source.split("BEGIN_TILING_DATA_DEF(FlashAttnGradTilingDataTmp)", 1)[1]
    body = body.split("END_TILING_DATA_DEF", 1)[0]
    return re.findall(r"TILING_DATA_FIELD_DEF\([^,]+,\s*(\w+)\)", body)


def _abi_size(fields):
    offset = 0
    max_align = 1
    for name in fields:
        size, align = (4, 4) if name == "scaleValue" else (8, 8)
        offset = (offset + align - 1) // align * align
        offset += size
        max_align = max(max_align, align)
    return (offset + max_align - 1) // max_align * max_align


def _workspace_bytes(b, s1, s2, n1, n2, d, dv, libapi):
    public_elems = b * s1 * n1 * d + b * s2 * n2 * d + b * s2 * n2 * dv
    guard_elems = 128 * max(d, dv)
    return libapi + 4 * (public_elems + guard_elems)


class TestNewDeterHostContract(unittest.TestCase):
    def test_tiling_fields_and_abi_match(self):
        host_fields = _host_tiling_fields()
        kernel_fields = _kernel_tiling_fields()
        self.assertEqual(host_fields, kernel_fields)
        self.assertEqual(_abi_size(kernel_fields), 200)
        self.assertEqual(kernel_fields[-1], "deterMaxRound")

        source = _text(HOST / "base" / "flash_attn_grad_tiling_regbase.cpp")
        self.assertIn("sizeof(FlashAttnGradTilingData) == 200", source)
        self.assertNotIn("deterUsedCube", source)

    def test_workspace_is_only_public_accumulators_and_guard(self):
        source = _text(HOST / "plan" / "flash_attn_grad_tiling_workspace.cpp")
        header = _text(HOST / "plan" / "flash_attn_grad_tiling_workspace.h")
        plan = _text(HOST / "plan" / "flash_attn_grad_tiling_plan.cpp")

        signature = "BuildWorkspacePlan(const FagParsedInfo &info)"
        self.assertIn(signature, source)
        self.assertIn(signature, header)
        self.assertIn("BuildWorkspacePlan(info);", plan)
        self.assertNotIn("info.isDeterministic", source)
        self.assertNotIn("FagKernelPlan", source)
        self.assertNotIn("FagSchedulePlan", source)

        for symbol in (
            "deterDq",
            "deterDk",
            "deterDv",
            "deterMeta",
            "DETER_EPOCH_ROUNDS",
            "DETER_DESC_WORDS",
        ):
            self.assertNotIn(symbol, source)

        # The result is independent of deterministic mode because no private
        # area participates in the formula. Cover D/Dv equality and mismatch.
        self.assertEqual(_workspace_bytes(1, 128, 128, 8, 8, 64, 64, 4096), 823296)
        self.assertEqual(_workspace_bytes(2, 256, 320, 16, 4, 192, 128, 0), 9666560)

    def test_epoch_workspace_symbols_are_absent_from_host_contract(self):
        sources = "\n".join(
            _text(path)
            for path in HOST.rglob("*")
            if path.is_file() and path.suffix in {".h", ".cpp"}
        )
        for symbol in ("deterUsedCube", "deterDq", "deterDk", "deterDv", "deterMeta"):
            self.assertNotIn(symbol, sources)
        self.assertNotIn("deterUsedCube", _text(KERNEL / "fag_tiling.py"))

    def test_kernel_has_direct_fix_chain_without_epoch_reduction(self):
        kernel = "\n".join(
            _text(KERNEL / name)
            for name in (
                "fag_kernel.py",
                "fag_kernel_norm.py",
                "fag_kernel_deter.py",
            )
        )
        common = _text(KERNEL / "fag_common.py")
        cube = _text(KERNEL / "fag_block_cube.py")
        sources = "\n".join(
            _text(KERNEL / name)
            for name in (
                "fag_kernel.py",
                "fag_kernel_norm.py",
                "fag_kernel_deter.py",
                "fag_common.py",
                "fag_block_cube.py",
                "fag_block_vec.py",
                "fag_buffers.py",
                "fag_mem.py",
            )
        )
        for symbol in (
            "DETER_EPOCH",
            "deterSlot",
            "deter_desc",
            "deter_merge",
            "tensor_dq_deter",
            "tensor_dk_deter",
            "tensor_dv_deter",
            "process_deter_epoch",
        ):
            self.assertNotIn(symbol, sources)

        self.assertNotIn("bar_all()", kernel)
        # preload3 pre/post, preload2 BN2 pre, preload2 GS1S2 pre/post only.
        self.assertEqual(
            _text(KERNEL / "fag_kernel.py").count("pl.system.sync_all()"), 5
        )
        self.assertIn("SYNC_DQ_DETER_FIX_FLAG = 6", common)
        self.assertIn("SYNC_DK_DETER_FIX_FLAG = 7", common)
        self.assertIn("SYNC_DV_DETER_FIX_FLAG = 14", common)
        for flag in (
            "SYNC_DQ_DETER_FIX_FLAG",
            "SYNC_DK_DETER_FIX_FLAG",
            "SYNC_DV_DETER_FIX_FLAG",
        ):
            token_call = f"event_id={flag}"
            paired = re.findall(
                rf"{re.escape(token_call)}\s*,\s*sync_mode=pl\.CrossCoreSyncMode\.INTER_BLOCK",
                kernel,
            )
            self.assertGreater(kernel.count(token_call), 0)
            self.assertEqual(kernel.count(token_call), len(paired), msg=flag)
        self.assertIn("def flush_dk(", cube)
        self.assertIn("def flush_dv(", cube)
        self.assertIn("flush_dk(", kernel)
        self.assertIn("flush_dv(", kernel)


if __name__ == "__main__":
    unittest.main()

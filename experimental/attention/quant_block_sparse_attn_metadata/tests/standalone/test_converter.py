# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU tests of production Python call boundaries with recording backend doubles.

These execute the actual converter/golden function bodies without importing the
unavailable NPU runtime. They do not validate GE execution or NPU numerics.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import unittest
from unittest.mock import patch

ATTENTION = Path(__file__).resolve().parents[3]
OPERATOR = ATTENTION / "quant_block_sparse_attn"
CONVERTER = (
    OPERATOR
    / "torch_ops_extension/custom_ops/converter/npu_quant_block_sparse_attn_metadata.py"
)
GOLDEN = OPERATOR / "tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py"


def load_function(path, name, bindings):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    bindings = dict(bindings)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        bindings,
    )
    return bindings[name]


class BoundaryTests(unittest.TestCase):
    def test_converter_preserves_layout_and_none(self):
        calls = []
        converter = load_function(
            CONVERTER,
            "convert_npu_quant_block_sparse_attn_metadata",
            {
                "Tensor": object,
                "TensorSpec": object,
                "List": List,
                "Optional": Optional,
                "attr": SimpleNamespace(Int=int, Str=str),
                "torch": SimpleNamespace(
                    npu=SimpleNamespace(
                        current_stream=lambda: None,
                        get_stream_limit=lambda _: {
                            "cube_core_num": 36,
                            "vector_core_num": 72,
                        },
                        get_device_properties=lambda: SimpleNamespace(name="Ascend950"),
                    )
                ),
                "torchair": SimpleNamespace(
                    ge=SimpleNamespace(custom_op=lambda *a, **kw: calls.append((a, kw)))
                ),
            },
        )
        for layout in ("BSND", "BNSD"):
            converter(object(), 4, 2, 128, batch_size=2, quant_mode=2, layout_q=layout)
            self.assertEqual(calls[-1][1]["attrs"]["layout_q"], layout)
            self.assertIsNone(calls[-1][1]["inputs"]["cu_seqlens_q"])
            self.assertIsNone(calls[-1][1]["inputs"]["seqused_q"])
            for invalid in (
                {"quant_mode": 1},
                {"cu_seqlens_q": object()},
                {"seqused_q": object()},
                {"layout_kv": "BSND"},
            ):
                kwargs = dict(quant_mode=2, layout_q=layout)
                kwargs.update(invalid)
                with self.assertRaises(ValueError):
                    converter(object(), 4, 2, 128, **kwargs)
        self.assertEqual(len(calls), 2)  # Invalid calls never reach GE.

    def test_golden_builds_metadata_for_each_checked_layout(self):
        metadata_calls, main_calls = [], []

        class Output:
            def detach(self):
                return self

            def cpu(self):
                return self

        def prepare(
            cu_q,
            cu_kv,
            used_q,
            used_kv,
            table,
            nq,
            nk,
            layout,
            kv_layout,
            indices,
            sparse,
        ):
            metadata_calls.append(layout)
            return (
                cu_q,
                cu_kv,
                used_q,
                used_kv,
                indices,
                sparse,
                table,
                "metadata:" + layout,
            )

        def main_op(*args, **kwargs):
            main_calls.append(kwargs)
            return Output(), Output()

        def check(invoke, q, scale, *args):
            for layout in ("BSND", "BNSD"):
                invoke(q, scale, layout, layout)

        invoke = load_function(
            GOLDEN,
            "_call_npu_fa_op",
            {
                "os": os,
                "DEVICE_ID": 0,
                "SPARSE_BLOCK_SIZE": 128,
                "SPARSE_INDICES_LAYOUT": "B_N_Qb_Kb",
                "QUANT_MODE_MXFP8": 2,
                "MASK_MODE": 0,
                "ENABLE_LSE": True,
                "OUT_LAYOUT": "TND",
                "_ensure_custom_ops_registered": lambda: None,
                "_prepare_npu_metadata": prepare,
                "_check_padded_layouts": check,
                "torch_npu": SimpleNamespace(
                    npu=SimpleNamespace(
                        set_device=lambda _: None, synchronize=lambda: None
                    )
                ),
                "torch": SimpleNamespace(
                    ops=SimpleNamespace(
                        custom=SimpleNamespace(npu_quant_block_sparse_attn=main_op)
                    )
                ),
            },
        )
        with patch.dict(os.environ, QBSA_CHECK_PADDED_LAYOUTS="1"):
            invoke(
                "q",
                "k",
                "v",
                None,
                "prefix",
                None,
                None,
                "kv_lengths",
                "qs",
                "ks",
                "vs",
                None,
                "table",
                4,
                2,
                0.125,
                "TND",
                "PA_BNBD",
                "indices",
                "sparse",
            )
        self.assertEqual(metadata_calls, ["TND", "BSND", "BNSD"])
        for call in main_calls:
            self.assertEqual(call["metadata"], "metadata:" + call["layout_q"])
            self.assertEqual(
                call["cu_seqlens_q"], "prefix" if call["layout_q"] == "TND" else None
            )


if __name__ == "__main__":
    unittest.main()

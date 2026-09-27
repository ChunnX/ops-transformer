# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dependency-free tests of the production Python validator/Meta function.
Only tensor metadata and torch._check/new_empty are emulated. This does not test
PyTorch dispatcher registration, JIT compilation or NPU execution.
"""

import ast
import math
from pathlib import Path
import types
import unittest


class TensorMetadata:
    """A tensor view sufficient for shape/type inference, without numerical storage."""

    def __init__(self, shape, dtype="float16", device="meta"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def dim(self):
        return len(self.shape)

    def new_empty(self, shape, dtype=None):
        return TensorMetadata(shape, dtype or self.dtype, self.device)


def check(condition, message):
    """Match torch._check's false-condition behavior for concrete metadata."""
    if not condition:
        raise RuntimeError(message())


def load_production_functions():
    """Execute the unchanged validator/Meta AST bodies without importing torch_npu."""
    source = Path(__file__).parents[1] / "torch_extension/mega_mhc.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_check_inputs",
            "mega_mhc_meta",
        ):
            node.decorator_list = []
            functions.append(node)
    namespace = {
        "math": math,
        "torch": types.SimpleNamespace(
            float16="float16", bfloat16="bfloat16", float32="float32", _check=check
        ),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["mega_mhc_meta"]


def make_inputs(leading=(33,), d=2560, dtype="float16", mid=False):
    """Create the public nine required tensor shapes, plus optional gamma2."""
    shapes = [
        (*leading, d),
        (*leading, 4, d),
        (*leading, 4),
        (*leading, 4, 4),
        (24, 4 * d),
        (3,),
        (24,),
        (d,),
        (d,),
    ]
    values = [
        TensorMetadata(shape, dtype if i < 2 else "float32")
        for i, shape in enumerate(shapes)
    ]
    return values, {"gamma_2": TensorMetadata((4 * d,), "float32") if mid else None}


class TestTorchContract(unittest.TestCase):
    meta = staticmethod(load_production_functions())

    def test_all_template_families_and_ranks(self):
        """Eight four/five x FP32/HF32 x input-type combinations, both D and ranks."""
        for leading in ((1,), (33,), (2, 17)):
            for d in (2560, 2816, 5120, 6144, 7168):
                for dtype in ("float16", "bfloat16"):
                    for mid in (False, True):
                        for mode in (0, 1):
                            with self.subTest(
                                leading=leading, d=d, dtype=dtype, mid=mid, mode=mode
                            ):
                                args, kwargs = make_inputs(leading, d, dtype, mid)
                                outputs = self.meta(*args, **kwargs, compute_mode=mode)
                                self.assertEqual(
                                    [x.shape for x in outputs],
                                    [args[i].shape for i in (0, 2, 3, 1)],
                                )
                                self.assertEqual(
                                    [x.dtype for x in outputs],
                                    [dtype, "float32", "float32", dtype],
                                )

    def test_required_gamma0_and_optional_gamma2(self):
        args, kwargs = make_inputs()
        args[-1] = None
        with self.assertRaisesRegex(RuntimeError, "gamma_0 is required"):
            self.meta(*args, **kwargs)
        args, kwargs = make_inputs(mid=True)
        kwargs["gamma_2"].shape = (4, 2560)
        with self.assertRaisesRegex(RuntimeError, "gamma_2"):
            self.meta(*args, **kwargs)

    def test_each_tensor_shape_dtype_and_device(self):
        for i in range(9):
            for attr, bad in (
                ("dtype", "float64"),
                ("device", "other"),
                ("shape", (7,)),
            ):
                with self.subTest(tensor=i, attr=attr):
                    args, kwargs = make_inputs()
                    setattr(args[i], attr, bad)
                    with self.assertRaises(RuntimeError):
                        self.meta(*args, **kwargs)

    def test_invalid_shapes_and_attributes(self):
        for leading, d in [
            ((0,), 2560),
            ((2**32,), 2560),
            ((65536, 65536), 2560),
            ((1,), 128),
            ((1,), 7169),
            ((1,), 7424),
            ((1,), 2561),
        ]:
            args, kwargs = make_inputs(leading=leading, d=d)
            with self.assertRaises(RuntimeError):
                self.meta(*args, **kwargs)
        for key, invalids in {
            "compute_mode": (-1, 2),
            "norm_eps": (0, -1, math.inf, math.nan, 1e-50, 1e40),
            "hc_eps": (-1, math.inf, math.nan, 1e40),
        }.items():
            for value in invalids:
                with self.subTest(key=key, value=value):
                    args, kwargs = make_inputs()
                    with self.assertRaises(RuntimeError):
                        self.meta(*args, **kwargs, **{key: value})

    def test_zero_hc_epsilon_and_fp32_subnormal(self):
        args, kwargs = make_inputs()
        self.meta(*args, **kwargs, hc_eps=0, norm_eps=1e-45)


if __name__ == "__main__":
    unittest.main()

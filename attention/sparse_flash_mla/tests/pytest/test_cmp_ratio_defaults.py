"""Exercise Python dispatch defaults without requiring an installed NPU runtime."""

import ast
from pathlib import Path
from typing import Optional
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


class CmpRatioDefaultsTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "torch_extension/sparse_flash_mla.py"
        tree = ast.parse(source.read_text())
        names = {"sparse_flash_mla", "sparse_flash_mla_metadata"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in functions:
            node.decorator_list = []
        self.module = Mock()
        self.namespace = {
            "Optional": Optional,
            "torch": SimpleNamespace(Tensor=object),
            "sparse_flash_mla_op_builder": SimpleNamespace(load=lambda: self.module),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), self.namespace)

    def test_main_dispatch(self):
        for present in (False, True):
            for ratio in (None, 0, 1, 2, 4, 128):
                with self.subTest(present=present, ratio=ratio):
                    kwargs = {} if ratio is None else {"cmp_ratio": ratio}
                    self.namespace["sparse_flash_mla"](object(), cmp_kv=object() if present else None, **kwargs)
                    expected = int(present) if ratio is None else ratio
                    self.assertEqual(self.module.sparse_flash_mla.call_args.args[19], expected)

    def test_metadata_dispatch(self):
        for present in (None, False, True):
            for ratio in (None, 0, 1, 2, 4, 128):
                with self.subTest(present=present, ratio=ratio):
                    self.namespace["sparse_flash_mla_metadata"](
                        1, 1, 512, has_cmp_kv=present, cmp_ratio=ratio
                    )
                    expected = int(present is not False) if ratio is None else ratio
                    self.assertEqual(self.module.sparse_flash_mla_metadata.call_args.args[18], expected)


if __name__ == "__main__":
    unittest.main()

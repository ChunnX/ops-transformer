"""CPU-only checks for the default ROCE launch; no Torch or NPU imports."""

import argparse
import ast
from dataclasses import replace
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
DRIVER = BASE / "bandwidth_test.py"


class RoceConfigurationTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.config = runpy.run_path(str(BASE / "config.py"))["CONFIG"]
        self.tree = ast.parse(DRIVER.read_text(encoding="utf-8"))

    def test_argument_free_startup_selects_one_roce_case(self):
        begin = next(i for i, node in enumerate(self.tree.body)
                     if isinstance(node, ast.FunctionDef) and node.name == "configure_roce_functional_test")
        end = next(i for i, node in enumerate(self.tree.body)
                   if isinstance(node, ast.Import) and node.names[0].name == "torch")
        namespace = dict(CONFIG=self.config, argparse=argparse, os=os, sys=sys, replace=replace,
                         __name__="__main__", __package__="")
        sweep = Mock(side_effect=AssertionError("Default launch must not start a sweep"))
        with patch.object(sys, "argv", [str(DRIVER)]), patch.dict(os.environ, {}, clear=True), \
                patch.dict(sys.modules, {"bandwidth_sweep": SimpleNamespace(run_sweep=sweep)}):
            exec(compile(ast.Module(body=self.tree.body[begin:end], type_ignores=[]), str(DRIVER), "exec"), namespace)
        sweep.assert_not_called()
        self.assertTrue(namespace["ROCE_FUNCTIONAL"])
        config = namespace["CONFIG"]
        selection = next(node for node in self.tree.body if isinstance(node, ast.Assign) and
                         any(isinstance(target, ast.Name) and target.id == "test_configs" for target in node.targets))
        exec(compile(ast.Module(body=[selection], type_ignores=[]), str(DRIVER), "exec"), namespace)
        self.assertEqual(namespace["test_configs"], [dict(name="scattered", bs=512, max_bs=512,
                                                        data_size=8192, route="striped")])
        self.assertTrue(config.run.is_single_test)
        self.assertEqual((namespace["roce_iterations"], namespace["roce_timeout_seconds"]), (1, 600))
        self.assertEqual((config.parallel.server_num, config.parallel.rank_per_dev,
                          config.parallel.server_index, config.parallel.master_ip), (1, 2, 0, "127.0.0.1"))

    def test_config_itself_is_roce_precision(self):
        model, run = self.config.model, self.config.run
        self.assertEqual((model.comm_alg, model.mode, model.graph_type), ("roce", 0, 0))
        self.assertFalse(model.shared_jetty)
        self.assertEqual((model.shared_jetty_num, model.batch_tokens, model.peers_per_jetty), (1, 0, 0))
        self.assertTrue(run.is_precision_test)
        self.assertFalse(run.sweep_enabled)
        self.assertFalse(run.is_bandwidth_test)

    def test_insufficient_devices_fail_before_worker_creation(self):
        nodes = [node for node in self.tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in ("check_local_device_count", "run_roce_functional_test")]
        manager = Mock(side_effect=AssertionError("Must not create workers"))
        namespace = dict(CONFIG=self.config, Manager=manager,
                         torch_npu=SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 1)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DRIVER), "exec"), namespace)
        with self.assertRaisesRegex(ValueError, "rank_per_dev=2, but only 1"):
            namespace["run_roce_functional_test"]()
        manager.assert_not_called()
        namespace["torch_npu"].npu.device_count = lambda: 2
        namespace["check_local_device_count"]()


if __name__ == "__main__":
    unittest.main()

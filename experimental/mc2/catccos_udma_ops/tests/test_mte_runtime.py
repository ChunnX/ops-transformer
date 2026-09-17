# Copyright (c) 2026 Huawei Technologies Co., Ltd.
import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


class MteRuntimeTests(unittest.TestCase):
    def test_native_workspace_launch_and_lifecycle_share_lock_order(self):
        """Keep RecordStream + enqueue atomic with respect to another call/free."""
        native = PROJECT / "csrc" / "mte"
        runtime = (native / "mte_runtime.cpp").read_text(encoding="utf-8")
        self.assertRegex(runtime, r"AcquireLaunchGuard\(\)\s*\{\s*"
                         r"return std::unique_lock<std::mutex>\(launchMutex_\);")
        for method in ("Initialize", "Finalize"):
            with self.subTest(method=method):
                self.assertRegex(runtime,
                    r"void MteRuntime::" + method + r"\([^{}]+\)\s*\{\s*"
                    r"auto launchGuard = AcquireLaunchGuard\(\);\s*"
                    r"std::lock_guard<std::mutex> lock\(mutex_\);")
        for name in ("all_to_allv_grouped_mat_mul_mte.cpp", "grouped_mat_mul_all_to_allv_mte.cpp"):
            with self.subTest(operator=name):
                source = (native / name).read_text(encoding="utf-8")
                # Equal indentation places the RAII owner in the operator's
                # outer scope, spanning every variant's enqueue until return.
                self.assertRegex(source,
                    r"\n    MteRuntime &runtime = MteRuntime::Instance\(\);\n"
                    r"    auto launchGuard = runtime.AcquireLaunchGuard\(\);\n"
                    r"    runtime.EnsureInitialized\(\);")
                start = source.index("auto launchGuard = runtime.AcquireLaunchGuard();")
                end = source.index("\n    return output;", start)
                protected = source[start:end]
                self.assertIn("runtime.RecordStream(stream);", protected)
                self.assertIn("launchWithConfig", protected)
                self.assertNotRegex(protected, r"launchGuard\s*\.\s*(unlock|release|swap)\s*\(")

    def setUp(self):
        self.runtime = importlib.import_module("ops_transformer_catccos_mte.runtime")
        self.runtime._anchor = self.runtime._config = self.runtime._pending_override = None

    def tearDown(self):
        self.runtime._anchor = self.runtime._config = self.runtime._pending_override = None

    def test_cpu_import_does_not_import_torch_or_extension(self):
        program = """
import sys
class BlockNpu:
    def find_spec(self, fullname, *args):
        if fullname in ('torch', 'torch_npu') or fullname.endswith('._C'):
            raise AssertionError('unexpected NPU dependency: ' + fullname)
sys.meta_path.insert(0, BlockNpu())
import ops_transformer_catccos_mte
import ops_transformer_catccos_mte.runtime
import ops_transformer_catccos_mte.validation.case_schema
"""
        result = subprocess.run([sys.executable, "-c", program], cwd=PROJECT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mathematical_api_matches_existing_facade(self):
        def signatures(package):
            tree = ast.parse((PROJECT / package / "ops.py").read_text(encoding="utf-8"))
            return {n.name: ast.dump(n.args, include_attributes=False) for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in
                    ("all_to_allv_grouped_mat_mul", "grouped_mat_mul_all_to_allv")}
        self.assertEqual(signatures("ops_transformer_catccos_mte"), signatures("ops_transformer_catccos_udma"))

    def test_production_override_rejected_without_loading_npu(self):
        with mock.patch.object(self.runtime._build_versions, "WITH_TUNING", False):
            with self.assertRaisesRegex(RuntimeError, "--with-tuning"):
                self.runtime._set_tuning_override((7, 1, 65536, 2))

    def test_override_queue_applied_after_init(self):
        calls = []
        ops = types.SimpleNamespace(
            _configure_mte=lambda *args: calls.append("init"),
            _set_mte_tuning_override=lambda anchor, config: calls.append(tuple(config)),
            _finalize_mte=lambda anchor: calls.append("finalize"))
        torch = types.SimpleNamespace(uint8="uint8", empty=lambda *a, **k: object(),
                                      ops=types.SimpleNamespace(ops_transformer_catccos_mte=ops))
        npu = types.SimpleNamespace(npu=types.SimpleNamespace(set_device=lambda device: None))
        with mock.patch.object(self.runtime._build_versions, "WITH_TUNING", True), \
             mock.patch.object(self.runtime, "load_extension"), \
             mock.patch.dict(sys.modules, {"torch": torch, "torch_npu": npu}):
            self.runtime._set_tuning_override((7, 1, 65536, 2))
            self.assertEqual(calls, [])
            self.runtime.init_mte_group(rank=0, world_size=8, local_rank=0, init_method="tcp://127.0.0.1:29800")
            self.assertEqual(calls, ["init", (7, 1, 65536, 2)])
            self.runtime.destroy_mte_group()
            self.assertEqual(calls[-1], "finalize")
            self.assertIsNone(self.runtime._pending_override)

    def test_ub_budget_includes_reserves(self):
        with mock.patch.object(self.runtime._build_versions, "WITH_TUNING", True):
            with self.assertRaisesRegex(ValueError, "UB"):
                self.runtime._set_tuning_override((7, 1, 31 * 2048 * 2, 2))
            self.runtime._set_tuning_override((7, 1, 65536, 2))

    def test_failed_override_cleans_up_native_init(self):
        finalize = mock.Mock()
        ops = types.SimpleNamespace(_configure_mte=mock.Mock(), _finalize_mte=finalize,
            _set_mte_tuning_override=mock.Mock(side_effect=RuntimeError("unsupported AIV pool")))
        torch = types.SimpleNamespace(uint8="uint8", empty=lambda *a, **k: object(),
            ops=types.SimpleNamespace(ops_transformer_catccos_mte=ops))
        npu = types.SimpleNamespace(npu=types.SimpleNamespace(set_device=lambda device: None))
        with mock.patch.object(self.runtime._build_versions, "WITH_TUNING", True), \
             mock.patch.object(self.runtime, "load_extension"), \
             mock.patch.dict(sys.modules, {"torch": torch, "torch_npu": npu}):
            self.runtime._set_tuning_override((7, 1, 65536, 2))
            with self.assertRaisesRegex(RuntimeError, "AIV"):
                self.runtime.init_mte_group(rank=0, world_size=8, init_method="tcp://127.0.0.1:29800")
            finalize.assert_called_once()
            self.assertIsNone(self.runtime._anchor)
            self.assertIsNone(self.runtime._config)

    def test_second_backend_load_rejected_before_import(self):
        loader = importlib.import_module("ops_transformer_catccos_mte._loader")
        with mock.patch.object(loader, "_loaded", False), \
             mock.patch.object(sys, "_ops_transformer_catccos_backend", "udma", create=True), \
             mock.patch.object(loader.importlib, "import_module") as imports:
            with self.assertRaisesRegex(RuntimeError, "separate torchrun"):
                loader.load_extension()
            imports.assert_not_called()

    def test_invalid_heap_and_rank_fail_before_npu_load(self):
        with mock.patch.object(self.runtime, "load_extension") as load:
            with self.assertRaises(ValueError):
                self.runtime.init_mte_group(world_size=9)
            with self.assertRaisesRegex(ValueError, "workspace"):
                self.runtime.init_mte_group(heap_size=400 << 20)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import types
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]


def load_setup(backend, tuning="0"):
    """Capture setup metadata without setuptools or any NPU dependencies."""
    captured = {}
    setuptools = types.ModuleType("setuptools")
    setuptools.Command = type("Command", (), {})
    setuptools.Distribution = type("Distribution", (), {})
    def find_packages(include):
        root = include[0]
        if root == "validation":
            return ["validation"]
        packages = [root, root + ".validation", root + ".tuning"]
        return packages if backend == "mte" else [root]
    setuptools.find_packages = find_packages
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    build_py = types.ModuleType("setuptools.command.build_py")
    build_py.build_py = type("build_py", (), {"run": lambda self: None})
    wheel = types.ModuleType("wheel.bdist_wheel")
    wheel.bdist_wheel = type("bdist_wheel", (), {})
    with mock.patch.dict(sys.modules, {"setuptools": setuptools,
            "setuptools.command": types.ModuleType("setuptools.command"),
            "setuptools.command.build_py": build_py,
            "wheel": types.ModuleType("wheel"), "wheel.bdist_wheel": wheel}), \
         mock.patch.dict(os.environ, {"CATCCOS_BACKEND": backend, "CATCCOS_MTE_WITH_TUNING": tuning}):
        namespace = runpy.run_path(str(PROJECT / "setup.py"))
    return captured, namespace


class MteBuildTests(unittest.TestCase):
    def test_mte_bootstrap_export_is_required_before_build(self):
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("CMake is required for the SHMEM export check")
        cases = (
            ("0000000000123400 B g_boot_handle", 0, True),
            ("0000000000123400 B g_boot_handle_private", 0, False),
            ("                 U g_boot_handle", 0, False),
            ("0000000000123400 B g_boot_handle", 1, False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "libshmem.so"
            library.write_bytes(b"fake-library-for-export-check")
            for symbols, status, accepted in cases:
                with self.subTest(symbols=symbols, status=status):
                    if os.name == "nt":
                        fake_nm = root / "fake-nm.cmd"
                        fake_nm.write_text("@echo off\necho " + symbols +
                                           "\nexit /b " + str(status) + "\n", encoding="utf-8")
                    else:
                        fake_nm = root / "fake-nm"
                        fake_nm.write_text("#!/bin/sh\nprintf '%s\\n' '" + symbols +
                                           "'\nexit " + str(status) + "\n", encoding="utf-8")
                        fake_nm.chmod(0o755)
                    result = subprocess.run([cmake,
                        "-DCATCCOS_MTE_SHMEM_LIBRARY=" + str(library),
                        "-DCATCCOS_MTE_NM=" + str(fake_nm), "-P",
                        str(PROJECT / "cmake" / "check_mte_shmem_exports.cmake")],
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    self.assertEqual(result.returncode == 0, accepted, result.stdout)

    def test_wheels_own_disjoint_packages_and_output_paths(self):
        mte, _ = load_setup("mte")
        udma, _ = load_setup("udma")
        self.assertTrue(set(mte["packages"]).isdisjoint(udma["packages"]))
        self.assertNotIn("validation", mte["packages"])
        self.assertIn("validation", udma["packages"])
        self.assertEqual(mte["name"], "ops_transformer_catccos_mte")
        self.assertTrue(mte["options"]["bdist_wheel"]["dist_dir"].endswith(os.path.join("dist", "mte")))
        self.assertTrue(udma["options"]["bdist_wheel"]["dist_dir"].endswith(os.sep + "dist"))
        self.assertIn("ops-transformer-udma-accuracy=validation.run_accuracy:main",
                      udma["entry_points"]["console_scripts"])

    def test_tuning_package_is_opt_in(self):
        production, _ = load_setup("mte")
        tuning, _ = load_setup("mte", "1")
        self.assertNotIn("ops_transformer_catccos_mte.tuning", production["packages"])
        self.assertIn("ops_transformer_catccos_mte.tuning", tuning["packages"])

    def test_invalid_backend_fails_during_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "CATCCOS_BACKEND"):
            load_setup("auto")

    def test_mte_build_selects_only_mte_target(self):
        _, namespace = load_setup("mte")
        command = namespace["CMakeBuild"]()
        fake_torch = types.SimpleNamespace(utils=types.SimpleNamespace(cmake_prefix_path="/torch"))
        fake_npu = types.SimpleNamespace(__file__="/torch_npu/__init__.py")
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "torch_npu": fake_npu}), \
             mock.patch.dict(os.environ, {"CATCCOS_MTE_WITH_TUNING": "1"}), \
             mock.patch.dict(command.run.__globals__, {"find_cmake": lambda: "cmake"}), \
             mock.patch("shutil.rmtree"), \
             mock.patch("subprocess.check_call") as call:
            command.run()
        configure, build = [entry.args[0] for entry in call.call_args_list]
        self.assertIn("-DCATCCOS_BACKEND=mte", configure)
        self.assertIn("-DCATCCOS_MTE_WITH_TUNING=ON", configure)
        self.assertEqual(build[build.index("--target") + 1], "catccos_mte_extension")
        self.assertNotIn("catccos_udma_extension", build)

    def test_new_runtime_staging_removes_obsolete_source_libraries(self):
        _, namespace = load_setup("mte", "1")
        command = namespace["BackendBuildPy"]()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            (stage / ".libs").mkdir(parents=True)
            (stage / "_C.abi3.so").write_bytes(b"selected-extension")
            (stage / ".libs" / "libshmem.so").write_bytes(b"selected-runtime")
            command.build_lib = str(root / "lib")
            output = Path(command.build_lib) / "ops_transformer_catccos_mte"
            (output / ".libs").mkdir(parents=True)
            (output / ".libs" / "obsolete.so").write_bytes(b"old-runtime")
            (output / "_build_versions.py").write_text("WITH_TUNING = False\n", encoding="utf-8")
            with mock.patch.dict(command.run.__globals__, {"PROFILE_STAGE_DIR": str(stage)}), \
                 mock.patch.dict(os.environ, {"CATCCOS_MTE_WITH_TUNING": "1"}):
                command.run()
            self.assertFalse((output / ".libs" / "obsolete.so").exists())
            self.assertEqual((output / ".libs" / "libshmem.so").read_bytes(), b"selected-runtime")
            self.assertIn("WITH_TUNING = True", (output / "_build_versions.py").read_text(encoding="utf-8"))

    def test_udma_clean_preserves_mte_wheel(self):
        _, namespace = load_setup("udma")
        command = namespace["Clean"]()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dist = root / "dist"
            (dist / "mte").mkdir(parents=True)
            udma_wheel = dist / "ops_transformer_catccos_udma-1.0.0-cp38.whl"
            mte_wheel = dist / "mte" / "ops_transformer_catccos_mte-1.0.0-cp38.whl"
            udma_wheel.write_bytes(b"udma")
            mte_wheel.write_bytes(b"mte")
            with mock.patch.dict(command.run.__globals__, {"SOURCE_DIR": str(root),
                    "PROFILE_DIST_DIR": str(dist), "PROFILE_BUILD_DIR": str(root / "build" / "udma")}):
                command.run()
            self.assertFalse(udma_wheel.exists())
            self.assertEqual(mte_wheel.read_bytes(), b"mte")


if __name__ == "__main__":
    unittest.main()

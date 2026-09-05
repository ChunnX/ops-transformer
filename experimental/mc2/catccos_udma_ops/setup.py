#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os
import re
import runpy
import shutil
import subprocess

from setuptools import Command, Distribution, find_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BACKEND = os.environ.get("CATCCOS_BACKEND", "udma").strip().lower()
if BACKEND not in {"udma", "mte"}:
    raise RuntimeError("CATCCOS_BACKEND must be udma or mte")
PACKAGE_NAME = "ops_transformer_catccos_" + BACKEND
SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROFILE_BUILD_DIR = os.path.join(SOURCE_DIR, "build", BACKEND)
PROFILE_STAGE_DIR = os.path.join(PROFILE_BUILD_DIR, "cmake", "stage", PACKAGE_NAME)
PROFILE_DIST_DIR = os.path.join(SOURCE_DIR, "dist", "mte") if BACKEND == "mte" else os.path.join(SOURCE_DIR, "dist")
VERSION = runpy.run_path(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "_version.py")
)["__version__"]
MINIMUM_CMAKE_VERSION = (3, 18, 4)
_TRUE_ENV_VALUES = {"1", "on", "true", "yes"}
_FALSE_ENV_VALUES = {"", "0", "off", "false", "no"}


def find_cmake():
    cmake = shutil.which("cmake")
    required = ".".join(str(part) for part in MINIMUM_CMAKE_VERSION)
    if cmake is None:
        raise RuntimeError(
            f"CMake >= {required} is required. Install it in the active Conda "
            "environment or add it to PATH."
        )

    try:
        output = subprocess.check_output(
            [cmake, "--version"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Failed to execute CMake at {cmake}: {error}") from error

    match = re.search(r"^cmake version (\d+)\.(\d+)\.(\d+)", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"Unable to determine the version of CMake at {cmake}: {output.strip()}"
        )

    actual = tuple(int(part) for part in match.groups())
    if actual < MINIMUM_CMAKE_VERSION:
        actual_text = ".".join(str(part) for part in actual)
        raise RuntimeError(
            f"CMake >= {required} is required, but {cmake} reports {actual_text}."
        )
    return cmake


def cmake_boolean_from_env(name, default="0"):
    value = os.environ.get(name, default).strip().lower()
    if value in _TRUE_ENV_VALUES:
        return "ON"
    if value in _FALSE_ENV_VALUES:
        return "OFF"
    raise RuntimeError(f"{name} must be a boolean value")


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class CMakeBuild(Command):
    description = "Build the CatCCOS UDMA PyTorch extension with CMake"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmake = find_cmake()

        import torch
        import torch_npu

        source_dir = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(source_dir, "../../.."))
        build_dir = os.path.join(PROFILE_BUILD_DIR, "cmake")
        torch_dir = os.path.join(torch.utils.cmake_prefix_path, "Torch")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        third_party_dir = os.path.join(repo_root, "3rdparty")
        jobs = str(os.cpu_count() or 2)
        # Backend-specific staging prevents stale opposite-backend objects from
        # entering the Wheel, while keeping source packages free of binaries.
        if os.path.isdir(PROFILE_STAGE_DIR):
            shutil.rmtree(PROFILE_STAGE_DIR)

        config = [
            cmake, "-S", source_dir, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCATCCOS_BACKEND={BACKEND}",
            "-DCATCCOS_MTE_WITH_TUNING={}".format(cmake_boolean_from_env("CATCCOS_MTE_WITH_TUNING")),
            f"-DTorch_DIR={torch_dir}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            "-DNPU_SOC_VERSION=ascend950",
            f"-DCANN_3RD_LIB_PATH={third_party_dir}",
        ]
        if BACKEND == "udma":
            config.extend([
                "-DCATCCOS_UDMA_MSSANITIZER={}".format(cmake_boolean_from_env("CATCCOS_UDMA_MSSANITIZER")),
                "-DCATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER={}".format(
                    cmake_boolean_from_env("CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER")),
            ])
        if os.environ.get("CATCCOS_SOURCE_PATH"):
            config.append(f"-DCATCCOS_SOURCE_PATH={os.environ['CATCCOS_SOURCE_PATH']}")
        if os.environ.get("SHMEM_SOURCE_PATH"):
            config.append(f"-DSHMEM_SOURCE_PATH={os.environ['SHMEM_SOURCE_PATH']}")
        if os.environ.get("SHMEM_HOME_PATH"):
            config.append(f"-DSHMEM_HOME_PATH={os.environ['SHMEM_HOME_PATH']}")

        subprocess.check_call(config, cwd=source_dir)
        subprocess.check_call(
            [cmake, "--build", build_dir, "--config", "Release", "--target", f"catccos_{BACKEND}_extension", "--parallel", jobs],
            cwd=source_dir,
        )


class BackendBuildPy(build_py):
    def run(self):
        super().run()
        destination = os.path.join(self.build_lib, PACKAGE_NAME)
        if not os.path.isfile(os.path.join(PROFILE_STAGE_DIR, "_C.abi3.so")):
            raise RuntimeError("Missing selected-backend extension; run bdist_wheel or cmake_build first")
        # Older builds staged binaries in the source package. build_py may have
        # copied those package_data files, so replace the binary subtree rather
        # than merging it with a differently pinned previous runtime.
        for name in ("_C.abi3.so", ".libs"):
            artifact = os.path.join(destination, name)
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            elif os.path.isfile(artifact):
                os.remove(artifact)
        shutil.copytree(PROFILE_STAGE_DIR, destination, dirs_exist_ok=True)
        if BACKEND == "mte":
            versions_file = os.path.join(destination, "_build_versions.py")
            with open(versions_file, encoding="utf-8") as source:
                versions = source.read()
            with open(versions_file, "w", encoding="utf-8") as output:
                output.write(re.sub(r"WITH_TUNING = (True|False)",
                    "WITH_TUNING = " + str(cmake_boolean_from_env("CATCCOS_MTE_WITH_TUNING") == "ON"), versions))


class ABI3Wheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "cp38", "abi3", platform

    def run(self):
        self.run_command("cmake_build")
        super().run()


class Clean(Command):
    description = "Remove local build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Never delete another backend's build, Wheel or source staging.
        paths = [PROFILE_BUILD_DIR, os.path.join(SOURCE_DIR, f"{PACKAGE_NAME}.egg-info")]
        if BACKEND == "mte":
            paths.append(PROFILE_DIST_DIR)
        elif os.path.isdir(PROFILE_DIST_DIR):
            # Preserve dist/mte and any other backend's products while retaining
            # the established UDMA dist/*.whl installation command.
            from pathlib import Path
            for wheel_file in Path(PROFILE_DIST_DIR).glob(PACKAGE_NAME + "-*.whl"):
                wheel_file.unlink()
        for path in paths:
            if os.path.isdir(path):
                shutil.rmtree(path)


def selected_packages():
    packages = find_packages(include=[PACKAGE_NAME, PACKAGE_NAME + ".*"])
    if BACKEND == "udma":
        # Preserve the original installed validation import/CLI entry points.
        # MTE validation is namespaced and never installs this top-level name.
        packages.extend(find_packages(include=["validation", "validation.*"]))
    if cmake_boolean_from_env("CATCCOS_MTE_WITH_TUNING") == "OFF":
        packages = [name for name in packages if not name.startswith(PACKAGE_NAME + ".tuning")]
    return packages


VALIDATION_PACKAGE = PACKAGE_NAME + ".validation" if BACKEND == "mte" else "validation"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=f"Ascend950 CatCCOS {BACKEND.upper()} fused operators for PyTorch",
    packages=selected_packages(),
    package_data={
        PACKAGE_NAME: ["*.so", ".libs/*.so", ".libs/*.so.*"],
        (PACKAGE_NAME + ".validation" if BACKEND == "mte" else "validation"): ["cases/*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["torch", "torch_npu"],
    extras_require={
        "validation": ["PyYAML>=6.0", "matplotlib>=3.5", "numpy>=1.21"]
    },
    entry_points={
        "console_scripts": [
            f"ops-transformer-{BACKEND}-accuracy={VALIDATION_PACKAGE}.run_accuracy:main",
            f"ops-transformer-{BACKEND}-performance={VALIDATION_PACKAGE}.run_performance:main",
            f"ops-transformer-{BACKEND}-import-baseline={VALIDATION_PACKAGE}.import_msprof_baseline:main",
        ] + ([
            f"ops-transformer-udma-stress={VALIDATION_PACKAGE}.run_stress:main",
            f"ops-transformer-udma-analyze-dumps={VALIDATION_PACKAGE}.model_dump_analysis:main",
        ] if BACKEND == "udma" else []),
    },
    options={
        "build": {"build_base": PROFILE_BUILD_DIR},
        "bdist_wheel": {"dist_dir": PROFILE_DIST_DIR,
                        "bdist_dir": os.path.join(PROFILE_BUILD_DIR, "bdist")},
    },
    distclass=BinaryDistribution,
    cmdclass={"cmake_build": CMakeBuild, "bdist_wheel": ABI3Wheel, "clean": Clean, "build_py": BackendBuildPy},
)

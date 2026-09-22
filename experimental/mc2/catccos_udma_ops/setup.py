#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os
import re
import runpy
import shutil
import subprocess

from setuptools import Command, Distribution, find_packages, setup
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PACKAGE_NAME = "ops_transformer_catccos_udma"
VERSION = runpy.run_path(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "_version.py")
)["__version__"]
MINIMUM_CMAKE_VERSION = (3, 18, 4)


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
        build_dir = os.path.join(source_dir, "build")
        torch_dir = os.path.join(torch.utils.cmake_prefix_path, "Torch")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        third_party_dir = os.path.join(repo_root, "3rdparty")
        jobs = str(os.cpu_count() or 2)
        package_dir = os.path.join(source_dir, PACKAGE_NAME)
        staged_extension = os.path.join(package_dir, "_C.abi3.so")
        staged_libs = os.path.join(package_dir, ".libs")
        if os.path.isfile(staged_extension):
            os.remove(staged_extension)
        if os.path.isdir(staged_libs):
            shutil.rmtree(staged_libs)

        config = [
            cmake, "-S", source_dir, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTorch_DIR={torch_dir}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            "-DNPU_SOC_VERSION=ascend950",
            f"-DCANN_3RD_LIB_PATH={third_party_dir}",
        ]
        if os.environ.get("CATCCOS_SOURCE_PATH"):
            config.append(f"-DCATCCOS_SOURCE_PATH={os.environ['CATCCOS_SOURCE_PATH']}")
        if os.environ.get("SHMEM_SOURCE_PATH"):
            config.append(f"-DSHMEM_SOURCE_PATH={os.environ['SHMEM_SOURCE_PATH']}")
        if os.environ.get("SHMEM_HOME_PATH"):
            config.append(f"-DSHMEM_HOME_PATH={os.environ['SHMEM_HOME_PATH']}")

        subprocess.check_call(config, cwd=source_dir)
        subprocess.check_call(
            [cmake, "--build", build_dir, "--config", "Release", "--parallel", jobs],
            cwd=source_dir,
        )


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
        source_dir = os.path.abspath(os.path.dirname(__file__))
        for name in ("build", "dist", f"{PACKAGE_NAME}.egg-info"):
            path = os.path.join(source_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
        package_dir = os.path.join(source_dir, PACKAGE_NAME)
        extension = os.path.join(package_dir, "_C.abi3.so")
        if os.path.isfile(extension):
            os.remove(extension)
        libs = os.path.join(package_dir, ".libs")
        if os.path.isdir(libs):
            shutil.rmtree(libs)


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="Ascend950 CatCCOS UDMA fused operators for PyTorch",
    packages=find_packages(),
    package_data={
        PACKAGE_NAME: ["*.so", ".libs/*.so", ".libs/*.so.*"],
        "validation": ["cases/*.yaml", "cases/*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["torch", "torch_npu"],
    extras_require={"validation": ["PyYAML>=6.0", "matplotlib>=3.5"]},
    entry_points={
        "console_scripts": [
            "ops-transformer-udma-accuracy=validation.run_accuracy:main",
            "ops-transformer-udma-performance=validation.run_performance:main",
            "ops-transformer-udma-import-baseline=validation.import_msprof_baseline:main",
            "ops-transformer-udma-search-tiling=validation.search_tiling:main",
        ]
    },
    distclass=BinaryDistribution,
    cmdclass={"cmake_build": CMakeBuild, "bdist_wheel": ABI3Wheel, "clean": Clean},
)

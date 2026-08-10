#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os
import shutil
import subprocess

from setuptools import Command, Distribution, find_packages, setup
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PACKAGE_NAME = "ops_transformer_catccos_udma"
VERSION = "0.1.0"


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
            "cmake", "-S", source_dir, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTorch_DIR={torch_dir}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            "-DNPU_SOC_VERSION=ascend950",
            f"-DCANN_3RD_LIB_PATH={third_party_dir}",
        ]
        if os.environ.get("CATCCOS_SOURCE_PATH"):
            config.append(f"-DCATCCOS_SOURCE_PATH={os.environ['CATCCOS_SOURCE_PATH']}")
        if os.environ.get("SHMEM_HOME_PATH"):
            config.append(f"-DSHMEM_HOME_PATH={os.environ['SHMEM_HOME_PATH']}")

        subprocess.check_call(config, cwd=source_dir)
        subprocess.check_call(
            ["cmake", "--build", build_dir, "--config", "Release", "--parallel", jobs],
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
    package_data={PACKAGE_NAME: ["*.so", ".libs/*.so", ".libs/*.so.*"]},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["torch", "torch_npu"],
    distclass=BinaryDistribution,
    cmdclass={"cmake_build": CMakeBuild, "bdist_wheel": ABI3Wheel, "clean": Clean},
)

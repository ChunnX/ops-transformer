#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import logging
import glob
import shutil
import subprocess
import sysconfig
import shutil
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools import Command

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CleanCommand(Command):
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        # 删除构建目录
        if os.path.exists('build'):
            shutil.rmtree('build')
            logger.info("Removed build/")
        
        # 删除dist目录
        if os.path.exists('dist'):
            shutil.rmtree('dist')
            logger.info("Removed dist/")
        
        # 删除egg-info目录
        egg_info_dir = f"{self.distribution.get_name().replace('-', '_')}.egg-info"
        if os.path.exists(egg_info_dir):
            shutil.rmtree(egg_info_dir)
            logger.info(f"Removed {egg_info_dir}/")
        
        # 删除.pyc文件和__pycache__目录
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pyc'):
                    os.remove(os.path.join(root, file))
                    logger.info(f"Removed {os.path.join(root, file)}")
            
            for item in dirs:
                if item == '__pycache__':
                    shutil.rmtree(os.path.join(root, item))
                    logger.info(f"Removed {os.path.join(root, item)}/")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        cmake_path = shutil.which("cmake")
        if not cmake_path:
            raise RuntimeError("cmake is not found in PATH. Please install cmake or specify full path.")
        try:
            subprocess.check_output([cmake_path, "--version"])
        except OSError as e:
            raise RuntimeError("CMake must be installed to build the extensions") from e

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        python_include = sysconfig.get_path('include')
        python_libs = sysconfig.get_config_var('LIBDIR')

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DBUILD_TORCH_OPS=ON",
            f"-DPYTHON_INCLUDE_DIR={python_include}",
            f"-DPYTHON_LIBRARIES={python_libs}",
        ]

        build_type = "Debug" if self.debug else "Release"
        build_args = ["--config", build_type]

        cpu_count = os.cpu_count() or 1
        parallel_jobs = cpu_count // 2
        build_args += ["--", f"-j{parallel_jobs}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


setup(
    name='ascend_ops',
    version='0.0.1',
    packages=['ascend_ops'],
    package_dir={'ascend_ops' : 'src'},
    ext_modules=[CMakeExtension("ascend_ops._C", sourcedir=".")],
    cmdclass={'build_ext': CMakeBuild, 'clean': CleanCommand},
    zip_safe=False,
    install_requires=["torch"],
    description="Example of PyTorch C++ and Ascend extensions (with CMake)",
)

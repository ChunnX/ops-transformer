#!/usr/bin/env python3

import os
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent

# 1. 搜集 CANN 与 NPU 头文件包含路径
include_dirs = [
    str(HERE / "python" / "sageattention2" / "csrc"),
    str(HERE / "op_api"),
]

ascend_home = os.environ.get(
    "ASCEND_HOME_PATH",
    os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"),
)
if os.path.exists(os.path.join(ascend_home, "include")):
    include_dirs.append(os.path.join(ascend_home, "include"))

try:
    import torch_npu
    torch_npu_root = os.path.dirname(os.path.abspath(torch_npu.__file__))
    for cand in [
        os.path.join(torch_npu_root, "include"),
        os.path.join(torch_npu_root, "include", "third_party", "acl", "inc"),
    ]:
        if os.path.exists(cand):
            include_dirs.append(cand)
except ImportError:
    pass

csrc_source = str(HERE / "python" / "sageattention2" / "csrc" / "attention.cpp")

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
    if os.environ.get("BUILD_EXT", "1") != "0" and os.path.exists(csrc_source):
        extra_compile_args = {
            "cxx": ["-O2", "-std=c++17", "-fPIC", "-Wno-sign-compare"]
        }
        ext_modules.append(
            CppExtension(
                name="sageattention2_ext",
                sources=[csrc_source],
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
            )
        )
        cmdclass["build_ext"] = BuildExtension
except ImportError:
    # PyTorch 未安装时仍允许 setuptools 生成 sdist / egg-info
    pass

setup(
    name="sageattention2",
    version="0.1.0",
    description="SageAttention2 for Ascend 950PR: INT8 QK^T + FP8 PV Attention Operator",
    author="Huawei Technologies Co., Ltd.",
    license="CANN Open Software License Agreement Version 2.0",
    package_dir={"": "python"},
    packages=find_packages("python"),
    package_data={
        "sageattention2": ["csrc/*.cpp", "*.py"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

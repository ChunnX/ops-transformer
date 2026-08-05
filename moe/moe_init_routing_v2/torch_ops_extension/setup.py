# copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv("USE_NINJA") == "1"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

CANN_OPS_COMMON_INC = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "..",
    "torch_extension",
    "cann_ops_transformer",
    "common",
    "inc",
)

source_files = glob.glob(
    os.path.join(BASE_DIR, "moe_init_routing_ops/csrc", "*.cpp"), recursive=True
)

exts = []
ext = NpuExtension(
    name="moe_init_routing_ops.moe_init_routing_lib",
    sources=source_files,
    extra_compile_args=[
        "-I" + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        "-I" + CANN_OPS_COMMON_INC,
    ],
    extra_link_args=["-Wl,--allow-multiple-definition"],
)
exts.append(ext)

setup(
    name="moe_init_routing_ops",
    version="1.0",
    keywords="moe_init_routing_ops",
    ext_modules=exts,
    package_data={
        "moe_init_routing_ops": ["*.py", "*.so"],
        "moe_init_routing_ops.ops": ["*.py", "*.so"],
    },
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)

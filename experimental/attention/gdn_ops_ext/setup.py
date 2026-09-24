# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


_HERE = os.path.dirname(os.path.realpath(__file__))
_ATTENTION_DIR = os.path.dirname(_HERE)
_FUSED_CSRC_DIR = os.path.join(
    _ATTENTION_DIR,
    "fused_gdn_gating",
    "torch_ops_extension",
    "custom_ops",
    "csrc",
)
_CHUNK_CSRC_DIR = os.path.join(
    _ATTENTION_DIR,
    "chunk_gated_delta_rule_compute_wy",
    "torch_ops_extension",
    "custom_ops",
    "csrc",
)
_EXT_CSRC_DIR = os.path.join(_HERE, "gdn_ops_ext", "csrc")

_TORCH_NPU_ROOT = os.path.dirname(os.path.abspath(torch_npu.__file__))
_ACL_INC_DIR = os.path.join(_TORCH_NPU_ROOT, "include", "third_party", "acl", "inc")
_NINJA_ENABLED = os.environ.get("USE_NINJA") == "1"

_SOURCES = [
    os.path.join(_EXT_CSRC_DIR, "ops_def_registration.cpp"),
    os.path.join(_FUSED_CSRC_DIR, "ops_common.cpp"),
    os.path.join(_FUSED_CSRC_DIR, "npu_fused_gdn_gating.cpp"),
    os.path.join(_CHUNK_CSRC_DIR, "npu_chunk_gated_delta_rule_compute_wy.cpp"),
]

_gdn_ops_ext = NpuExtension(
    name="gdn_ops_ext._C",
    sources=_SOURCES,
    include_dirs=[_FUSED_CSRC_DIR, _CHUNK_CSRC_DIR],
    extra_compile_args=["-I" + _ACL_INC_DIR, "-DBUILD_GDN_OPS_EXT"],
)

setup(
    name="gdn_ops_ext",
    version="1.0",
    keywords="gdn_ops_ext",
    packages=find_packages(),
    ext_modules=[_gdn_ops_ext],
    package_data={"gdn_ops_ext": ["*.py", "*.so"]},
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=_NINJA_ENABLED)},
)

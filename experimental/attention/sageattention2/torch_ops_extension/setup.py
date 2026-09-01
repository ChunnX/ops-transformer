# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TORCH_NPU_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
sources = glob.glob(os.path.join(BASE_DIR, "sageattention2", "csrc", "*.cpp"))
# Reuse the repository's common aclnn dynamic-loader implementation. The header is included directly by the
# SageAttention2 wrapper; compiling this source supplies its loader symbols.
sources.append(os.path.join(
    BASE_DIR,
    "..",
    "..",
    "quant_block_sparse_attn",
    "torch_ops_extension",
    "custom_ops",
    "csrc",
    "ops_common.cpp",
))

# torch_npu 2.7 exposes the PyTorch float8 scalar types even when it is paired
# with a CANN 9.0 ACL header that predates the corresponding aclDataType names.
# The ABI values are already reserved (35--37), so provide the names only for
# that older header.  CANN 9.2 and newer continue to use their native enums.
acl_include_dir = os.path.join(
    TORCH_NPU_PATH, "include", "third_party", "acl", "inc", "acl"
)
define_macros = []
# Newer torch_npu packages split aclDataType into acl_base_rt.h while acl_base.h only includes it.
# Inspect both files before adding compatibility macros, otherwise native enum names are macro-expanded
# inside their own declaration and compilation fails.
acl_base_contents = ""
for header_name in ("acl_base.h", "acl_base_rt.h"):
    try:
        with open(os.path.join(acl_include_dir, header_name), encoding="utf-8") as header_file:
            acl_base_contents += header_file.read()
    except OSError:
        pass
if "ACL_FLOAT8_E5M2" not in acl_base_contents:
    define_macros.extend([
        ("ACL_FLOAT8_E5M2", "((aclDataType)35)"),
        ("ACL_FLOAT8_E4M3FN", "((aclDataType)36)"),
        ("ACL_FLOAT8_E8M0", "((aclDataType)37)"),
    ])

extension = NpuExtension(
    name="sageattention2._C",
    sources=sources,
    define_macros=define_macros,
    extra_compile_args=["-I" + os.path.join(TORCH_NPU_PATH, "include", "third_party", "acl", "inc")],
)

setup(
    name="sageattention2-ascend",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=os.getenv("USE_NINJA") == "1")},
)

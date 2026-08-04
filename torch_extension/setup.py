# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


__description__ = "CannOpsTransformer"
__version__ = "1.0.0"
__package_name__ = "cann_ops_transformer"

import os
import shutil
import setuptools


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), __package_name__)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 算子目录下的 torch_extension 代码（拆分目录）复制进包内对应位置
_OP_TORCH_EXTENSION_DIRS = {
    "mhc/mhc_post_backward": "ops",
}


def copy_op_torch_extension():
    """将拆分到算子目录下的 torch_extension 文件同步到包内。"""
    for op_dir, pkg_subdir in _OP_TORCH_EXTENSION_DIRS.items():
        src_te = os.path.join(repo_root, op_dir, "torch_extension")
        if not os.path.isdir(src_te):
            continue
        dst_te = os.path.join(src_path, pkg_subdir)
        os.makedirs(dst_te, exist_ok=True)
        for path, dirnames, filenames in os.walk(src_te):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            rel = os.path.relpath(path, src_te)
            dst_dir = os.path.join(dst_te, rel) if rel != "." else dst_te
            os.makedirs(dst_dir, exist_ok=True)
            for filename in filenames:
                if filename.endswith(".pyc"):
                    continue
                shutil.copy2(
                    os.path.join(path, filename), os.path.join(dst_dir, filename)
                )


copy_op_torch_extension()

setuptools.setup(
    name=__package_name__,
    version=__version__,
    author="CANN",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_package_data=True,
    package_data={"": package_files(src_path)},
    zip_safe=False,
)

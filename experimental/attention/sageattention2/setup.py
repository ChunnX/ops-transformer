# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from setuptools import find_packages, setup

setup(
    name="sageattention2-npu",
    version="0.1.0",
    description="SageAttention2 for Ascend 950 with a standalone PyTorch C++ bridge",
    package_dir={"": "python"},
    packages=find_packages("python"),
    package_data={"sageattention2": ["csrc/*.cpp"]},
    install_requires=["torch", "torch_npu", "ninja"],
    python_requires=">=3.9",
)

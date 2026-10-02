# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Setup script for sageattention2-npu package."""

import os
from setuptools import find_packages, setup

extra_compile_args = ["-O3", "-fvisibility=hidden", "-Wno-deprecated-declarations", "-DHAS_TORCH_NPU=1"]
ext_modules = []

# Optional AOT compilation if BUILD_NPU_EXT=1
if os.environ.get("BUILD_NPU_EXT", "0") == "1":
    try:
        import torch_npu
        from torch_npu.utils.cpp_extension import NpuExtension
        from pathlib import Path

        npu = Path(torch_npu.__file__).resolve().parent
        cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
        if not cann.exists():
            cann = Path("/usr/local/Ascend/cann")

        ext = NpuExtension(
            name="sageattention2.sageattention2_npu_ext",
            sources=["python/sageattention2/csrc/attention.cpp"],
            include_dirs=[
                str(npu / "include"),
                str(npu / "include" / "third_party" / "acl" / "inc"),
                str(npu / "include" / "third_party" / "hccl" / "inc"),
                str(npu / "include" / "third_party" / "op-plugin"),
                str(cann / "include"),
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=[
                "-L" + str(npu / "lib"),
                "-ltorch_npu",
                "-Wl,-rpath," + str(npu / "lib"),
                "-L" + str(cann / "lib64"),
                "-lascendcl",
            ],
        )
        ext_modules.append(ext)
    except Exception as e:
        print(f"Notice: AOT build not triggered, fallback to JIT compilation ({e})")

setup(
    name="sageattention2-npu",
    version="0.1.0",
    description="SageAttention2 for Ascend 950 with PyTorch 2.x C++ bridge and FakeTensor/Meta support",
    package_dir={"": "python"},
    packages=find_packages("python"),
    package_data={"sageattention2": ["csrc/*.cpp"]},
    ext_modules=ext_modules,
    install_requires=["torch"],
    python_requires=">=3.9",
)

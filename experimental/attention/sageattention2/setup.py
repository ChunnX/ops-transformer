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
        # ASCEND_HOME_PATH 由镜像 ENV 导出（与 _extension.py / run.sh 容器侧同一契约），
        # 不做多候选探测
        cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))

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

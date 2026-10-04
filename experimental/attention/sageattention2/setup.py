"""Setup script for sageattention2-npu package."""

import importlib.util
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

        root = Path(__file__).resolve().parent
        npu = Path(torch_npu.__file__).resolve().parent
        cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
        if not cann.exists():
            cann = Path("/usr/local/Ascend/cann")

        # csrc 经 SAGE_EXEC_NPU_DIRECT 链接期直引 aclnnSageAttention2 两段式符号
        # （HAS_TORCH_NPU=1 下恒为直连形态），-lcust_opapi 是必需链接面，缺它则
        # AOT 产物 import 即 undefined symbol。链接参数唯一定义点在
        # python/sageattention2/_extension.py（与 JIT 形态同源）；此处独立加载该
        # 模块共享之，不经包 __init__（构建脚本不触发 torch.library 注册面）。
        _spec = importlib.util.spec_from_file_location(
            "sageattention2_aot_ldflags", root / "python" / "sageattention2" / "_extension.py")
        _ldflags = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ldflags)

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
                # vendor 两候选命中其一即补 -L/-lcust_opapi/-rpath（JIT 同源同序）
                *_ldflags.cust_opapi_ldflags(root, cann),
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

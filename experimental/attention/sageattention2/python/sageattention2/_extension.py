"""Load SageAttention2 PyTorch C++ bridge extension with compilation caching."""

from __future__ import annotations

import functools
import os
from pathlib import Path


@functools.lru_cache(None)
def load_extension():
    """Load or JIT-compile the SageAttention2 NPU C++ extension."""
    try:
        import torch_npu
    except ImportError as err:
        raise RuntimeError(
            "torch_npu is required to build/load the SageAttention2 NPU C++ extension."
        ) from err

    from torch.utils.cpp_extension import load

    package = Path(__file__).resolve().parent
    npu = Path(torch_npu.__file__).resolve().parent
    # ASCEND_HOME_PATH 由镜像 ENV 导出（与 run.sh 容器侧同一契约），不做多候选探测
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))

    csrc_file = package / "csrc" / "attention.cpp"
    if not csrc_file.exists():
        raise FileNotFoundError(f"Cannot find SageAttention2 C++ source at {csrc_file}")

    # 算子自身 aclnn proto 头与本地 OPP 安装产物（存在性守卫，幂等追加）——
    # 根包 python/ 的同名扩展经 shim 共享本实现，编译参数唯一定义点在此。
    op_root = package.parent.parent  # .../experimental/attention/sageattention2

    extra_include_paths = [
        str(npu / "include"),
        str(npu / "include" / "third_party" / "acl" / "inc"),
        str(npu / "include" / "third_party" / "hccl" / "inc"),
        str(npu / "include" / "third_party" / "op-plugin"),
        str(cann / "include"),
    ]
    if (op_root / "aclnn").exists():
        extra_include_paths.append(str(op_root / "aclnn"))

    extra_ldflags = [
        "-L" + str(npu / "lib"),
        "-ltorch_npu",
        "-Wl,-rpath," + str(npu / "lib"),
        "-L" + str(cann / "lib64"),
        "-lascendcl",
    ]

    for opp_vendor in (
        cann / "opp" / "vendors" / "custom_transformer",
        cann / "opp" / "vendors" / "customize",
        op_root / "build" / "opp_installed" / "vendors" / "customize",
    ):
        v_inc = opp_vendor / "op_api" / "include"
        if v_inc.exists() and str(v_inc) not in extra_include_paths:
            extra_include_paths.append(str(v_inc))
        v_lib = opp_vendor / "op_api" / "lib"
        if v_lib.exists():
            extra_ldflags.extend(["-L" + str(v_lib), "-lcust_opapi", "-Wl,-rpath," + str(v_lib)])
            break

    return load(
        name="sageattention2_npu_ext",
        sources=[str(csrc_file)],
        extra_include_paths=extra_include_paths,
        extra_cflags=["-O3", "-fvisibility=hidden", "-Wno-deprecated-declarations", "-DHAS_TORCH_NPU=1"],
        extra_ldflags=extra_ldflags,
        verbose=os.environ.get("SAGEATTN_BUILD_VERBOSE", "0") == "1",
    )

"""Load SageAttention2 PyTorch C++ bridge extension with compilation caching."""

from __future__ import annotations

import functools
import os
from pathlib import Path


def cust_opapi_ldflags(op_root: Path, cann: Path) -> list[str]:
    """算子 vendor 库 libcust_opapi.so 的链接参数（桥接扩展唯一定义点，JIT/AOT 共享）。

    Vendor 名唯一定义点（P2-14）：本算子自建 OPP 树的 vendor 目录固定为
    `customize`（build_opp.sh 出包 build/opp/vendors/customize，install_opp.sh
    默认部署 build/opp_installed/vendors/customize，config.ini load_priority=customize）。
    仓库根 build.sh 整包 run 包是另一交付物，其 vendor 目录为 ${vendor_name}_transformer
    （默认 --vendor_name=custom → custom_transformer，见 cmake/custom_build.cmake），
    与本算子本地树并存合法、互不同名；整包 opapi 由 CANN 按 ASCEND_OPP_PATH 装载、
    不经本扩展消费——桥接为链接期直引两段式符号（csrc SAGE_EXEC_NPU_DIRECT），
    -lcust_opapi 仅来自下方 customize 两候选，无需（也不应）按整包 vendor 名探测。
    头文件不经此注入：csrc 经相对路径直引仓库 op_api/aclnn_sage_attention2.h，
    aclnn/ 与 vendor include 均无可解析头。两候选均未命中时返回空列表：无 OPP 产物
    的环境下链接期 undefined symbol 显式暴露，不代拟第三候选（无向下兼容探测）。
    """
    for opp_vendor in (
        op_root / "build" / "opp_installed" / "vendors" / "customize",
        cann / "opp" / "vendors" / "customize",
    ):
        v_lib = opp_vendor / "op_api" / "lib"
        if v_lib.exists():
            return ["-L" + str(v_lib), "-lcust_opapi", "-Wl,-rpath," + str(v_lib)]
    return []


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

    # 本地 OPP 安装产物链接面（存在性守卫，幂等追加）——根包 python/ 的同名扩展经
    # shim 共享本实现，链接参数唯一定义点在本模块 cust_opapi_ldflags（setup.py 的
    # BUILD_NPU_EXT=1 AOT 形态同源复用）。
    op_root = package.parent.parent  # .../experimental/attention/sageattention2

    extra_include_paths = [
        str(npu / "include"),
        str(npu / "include" / "third_party" / "acl" / "inc"),
        str(npu / "include" / "third_party" / "hccl" / "inc"),
        str(npu / "include" / "third_party" / "op-plugin"),
        str(cann / "include"),
    ]

    extra_ldflags = [
        "-L" + str(npu / "lib"),
        "-ltorch_npu",
        "-Wl,-rpath," + str(npu / "lib"),
        "-L" + str(cann / "lib64"),
        "-lascendcl",
    ]

    extra_ldflags.extend(cust_opapi_ldflags(op_root, cann))

    return load(
        name="sageattention2_npu_ext",
        sources=[str(csrc_file)],
        extra_include_paths=extra_include_paths,
        extra_cflags=["-O3", "-fvisibility=hidden", "-Wno-deprecated-declarations", "-DHAS_TORCH_NPU=1"],
        extra_ldflags=extra_ldflags,
        verbose=os.environ.get("SAGEATTN_BUILD_VERBOSE", "0") == "1",
    )

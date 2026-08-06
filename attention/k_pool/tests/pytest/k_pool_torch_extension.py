from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch_npu
from torch.utils.cpp_extension import load


_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    cann_path = os.environ.get("ASCEND_HOME_PATH")
    if not cann_path:
        raise RuntimeError("ASCEND_HOME_PATH is not set")

    repo_root = Path(__file__).resolve().parents[4]
    package_root = repo_root / "torch_extension" / "cann_ops_transformer"
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    cxx_flags = [
        "-fstack-protector-all",
        "-Wl,-z,relro,-z,now,-z,noexecstack",
        "-fPIC",
        "-pie",
        "-s",
        "-fvisibility=hidden",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-w",
    ]
    cxx_flags.append(
        "-D_GLIBCXX_USE_CXX11_ABI="
        + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0")
    )

    _MODULE = load(
        name="k_pool_validation",
        sources=[str(package_root / "ops" / "csrc" / "k_pool.cpp")],
        extra_include_paths=[
            str(torch_npu_root / "include"),
            str(torch_npu_root / "include" / "third_party" / "hccl" / "inc"),
            str(torch_npu_root / "include" / "third_party" / "acl" / "inc"),
            str(Path(cann_path) / "include"),
            str(package_root / "common" / "inc"),
        ],
        extra_cflags=cxx_flags,
        extra_ldflags=[
            "-L" + str(Path(cann_path) / "lib64"),
            "-lascendcl",
            "-L" + str(torch_npu_root / "lib"),
            "-ltorch_npu",
        ],
        verbose=True,
    )
    return _MODULE


def k_pool(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int = 4,
    *,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    coff: int = 1,
    cache_mode: int = 1,
) -> torch.Tensor:
    module = _load_module()
    return module.k_pool(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        coff,
        cache_mode,
    )

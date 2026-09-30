# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Dynamic C++ bridge loader for SageAttention2 Ascend NPU backend."""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any


class AscendExtensionRegistry:
    """Manages compilation, caching, and symbol discovery for the NPU C++ bridge."""

    EXTENSION_MODULE_NAME: str = "sageattention2_npu_ext"

    @classmethod
    def resolve_cann_root(cls) -> Path:
        """Determines the active CANN installation path from environment variables."""
        env_path = os.environ.get("ASCEND_HOME_PATH")
        if env_path and Path(env_path).is_dir():
            return Path(env_path)
        default_base = Path("/usr/local/Ascend/cann-9.2.0-beta.1")
        if default_base.is_dir():
            return default_base
        return Path("/usr/local/Ascend/cann")

    @classmethod
    def build_and_load(cls) -> Any:
        """Invokes PyTorch JIT C++ extension builder with optimized compiler flags."""
        import torch_npu
        from torch.utils.cpp_extension import load

        module_dir = Path(__file__).resolve().parent
        source_file = module_dir / "csrc" / "attention.cpp"
        npu_root = Path(torch_npu.__file__).resolve().parent
        cann_root = cls.resolve_cann_root()

        include_dirs = [
            str(npu_root / "include"),
            str(npu_root / "include" / "third_party" / "acl" / "inc"),
            str(npu_root / "include" / "third_party" / "hccl" / "inc"),
            str(npu_root / "include" / "third_party" / "op-plugin"),
            str(cann_root / "include"),
        ]

        compiler_flags = [
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
            "-Wno-deprecated-declarations",
        ]
        if sys.platform == "linux":
            compiler_flags.append("-D_GLIBCXX_USE_CXX11_ABI=0")

        linker_flags = [
            f"-L{npu_root}/lib",
            "-ltorch_npu",
            f"-Wl,-rpath,{npu_root}/lib",
            f"-L{cann_root}/lib64",
            "-lascendcl",
        ]

        is_verbose = os.environ.get("SAGEATTN_BUILD_VERBOSE", "0") in ("1", "true", "True")

        return load(
            name=cls.EXTENSION_MODULE_NAME,
            sources=[str(source_file)],
            extra_include_paths=include_dirs,
            extra_cflags=compiler_flags,
            extra_ldflags=linker_flags,
            verbose=is_verbose,
        )


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    """Returns the loaded native NPU acceleration module (cached)."""
    try:
        import sageattention2_npu_ext
        return sageattention2_npu_ext
    except ImportError:
        pass

    return AscendExtensionRegistry.build_and_load()

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
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    if not cann.exists():
        cann_alt = Path("/usr/local/Ascend/cann")
        if cann_alt.exists():
            cann = cann_alt

    csrc_file = package / "csrc" / "attention.cpp"
    if not csrc_file.exists():
        raise FileNotFoundError(f"Cannot find SageAttention2 C++ source at {csrc_file}")

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

    return load(
        name="sageattention2_npu_ext",
        sources=[str(csrc_file)],
        extra_include_paths=extra_include_paths,
        extra_cflags=["-O3", "-fvisibility=hidden", "-Wno-deprecated-declarations", "-DHAS_TORCH_NPU=1"],
        extra_ldflags=extra_ldflags,
        verbose=os.environ.get("SAGEATTN_BUILD_VERBOSE", "0") == "1",
    )

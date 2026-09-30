#!/usr/bin/env python3
"""setup.py for compressor_grad AscendC kernel.

Ensures the kernel is built (cmake + make) BEFORE setuptools starts, then
packages a Python module `compressor_grad` whose __init__.py loads the
built .so through torch.ops.load_library (the .so registers via
TORCH_LIBRARY and has no PyInit_ symbol, so it cannot be imported directly).
"""

import glob
import os
import subprocess
import sysconfig
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension

OP_NAME = "compressor_grad"
EXT_NAME = f"{OP_NAME}_ext"
HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"
PKG_DIR = BUILD_DIR / OP_NAME


def ensure_kernel_built():
    so_files = sorted(glob.glob(str(BUILD_DIR / f"{EXT_NAME}*.so")))
    if not so_files:
        os.makedirs(BUILD_DIR, exist_ok=True)
        soc_ver = os.environ.get("SOC_VERSION", "ascend910_9382")
        ascend = os.environ.get("ASCEND_HOME_PATH", "")
        subprocess.check_call(
            ["cmake", str(HERE),
             f"-DSOC_VERSION={soc_ver}",
             f"-DASCEND_CANN_PACKAGE_PATH={ascend}",
             "-DCMAKE_BUILD_TYPE=Release"],
            cwd=BUILD_DIR)
        subprocess.check_call(
            ["make", f"-j{os.cpu_count()}"], cwd=BUILD_DIR)
        so_files = sorted(glob.glob(str(BUILD_DIR / f"{EXT_NAME}*.so")))
    if not so_files:
        raise RuntimeError(f"kernel .so not found under {BUILD_DIR}")

    PKG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    dst_so = PKG_DIR / f"_kernel{suffix}"
    if dst_so.exists():
        dst_so.unlink()
    subprocess.check_call(["cp", so_files[0], str(dst_so)])
    (PKG_DIR / "__init__.py").write_text(
        "import glob as _glob\n"
        "import os as _os\n"
        "import torch as _torch\n"
        "\n"
        "_here = _os.path.dirname(_os.path.abspath(__file__))\n"
        "_libs = _glob.glob(_os.path.join(_here, '_kernel*.so'))\n"
        "if _libs:\n"
        "    _torch.ops.load_library(_libs[0])\n"
    )


class _BuildExt(build_ext):
    def run(self):
        # kernel + package dir already staged by ensure_kernel_built()
        super().run()


ensure_kernel_built()

setuptools.setup(
    name=OP_NAME,
    version="0.1.0",
    description=f"{OP_NAME} AscendC kernel",
    ext_modules=[NpuExtension(EXT_NAME, sources=[])],
    cmdclass={"build_ext": _BuildExt},
    packages=[OP_NAME],
    package_dir={OP_NAME: str(PKG_DIR)},
    package_data={OP_NAME: ["*.so"]},
    license="BSD 3-Clause",
    python_requires=">=3.8",
)

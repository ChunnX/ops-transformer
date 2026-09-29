# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build one dtype/format signature and one tiling key via the upstream compiler wrapper."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from common import ROOT, cann_env, read, run, sha, source_hash, write
from marks import check_sites


def build(cfg, dest, dependencies=None, host_only=False):
    mark_catalog = check_sites()
    dest = Path(dest).resolve()
    if not host_only:
        (dest / "manifest.json").unlink(missing_ok=True)
    native, logs = dest / "native", dest / "logs"
    env = cann_env()
    cann = Path(env["ASCEND_HOME_PATH"])
    deps = Path(dependencies or ROOT / "sim_deps").resolve()
    before = source_hash()
    run(
        [
            "cmake",
            "-S",
            ROOT,
            "-B",
            native,
            "-G",
            "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_CCACHE=OFF",
            "-DASCEND_COMPUTE_UNIT=ascend950",
            "-DASCEND_OP_NAME=mega_moe",
            "-DENABLE_BUILT_IN=OFF",
            "-DENABLE_OPS_HOST=ON",
            "-DENABLE_OPS_KERNEL=ON",
            "-DENABLE_BUILD_PKG=OFF",
            "-DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON",
            f"-DCUSTOM_ASCEND_CANN_PACKAGE_PATH={cann}",
            f"-DCANN_3RD_LIB_PATH={deps}",
            "-DOPS_COMPILE_OPTIONS=-DNOT_DYNAMIC_COMPILE",
        ],
        ROOT,
        logs / "configure.log",
        env,
    )
    # Upstream's prepare step is regenerated on configure. Cache its pinned CMake
    # sources in the supported offline location so repeated builds do not reclone.
    if not (deps / "cann-cmake").exists():
        for candidate in (
            native / "prepare_build/_deps/cann-cmake-src",
            native / "_deps/cann-cmake-src",
        ):
            if (candidate / "function/prepare.cmake").exists():
                shutil.copytree(
                    candidate,
                    deps / "cann-cmake",
                    ignore=shutil.ignore_patterns(".git"),
                )
                break
    run(
        [
            "cmake",
            "--build",
            native,
            "--target",
            "optiling_compat",
            "common_src_copy",
            "generate_compile_cmd",
            "generate_ops_info",
            "mega_moe_apt_ascend950_src_copy",
            "mega_moe_apt_ascend950_py_copy",
            "mega_moe_apt_ascend950_mkdir",
            "-j",
            cfg["build"]["jobs"],
        ],
        ROOT,
        logs / "native.log",
        env,
    )
    compile_helper(dest, cann, env)
    write(
        dest / "layout_build.json",
        {"source_sha256": before, "helper_sha256": sha(dest / "tiling_helper")},
    )
    if host_only:
        print(
            f"Host layout helper ready: {dest / 'tiling_helper'} (no runnable device kernel)"
        )
        return
    matches = []
    for param in sorted((native / "binary/ascend950/gen").glob("MegaMoe_*_param.json")):
        data = read(param)
        spec = data[0] if isinstance(data, list) else data["op_list"][0]

        def item(i):
            value = spec["inputs"][i]
            return value[0] if isinstance(value, list) else value

        expected = {
            1: ("float8_e4m3fn", "ND"),
            2: ("int32", "ND"),
            3: (cfg.get("topk_weights_dtype", "bfloat16"), "ND"),
            4: ("float4_e2m1", "FRACTAL_NZ_C0_32"),
            5: ("float4_e2m1", "FRACTAL_NZ_C0_32"),
            6: ("float8_e8m0", "ND"),
            7: ("float8_e8m0", "ND"),
            11: ("float8_e8m0", "ND"),
            12: ("float8_e4m3fn", "ND"),
            13: ("float8_e4m3fn", "ND"),
            14: ("float8_e8m0", "ND"),
            15: ("float8_e8m0", "ND"),
        }
        if (
            all(
                (item(i).get("dtype"), item(i).get("format")) == want
                for i, want in expected.items()
            )
            and spec["outputs"][0]["dtype"] == "bfloat16"
        ):
            matches.append(param)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one generated FP8/FP4-NZ/shared-FP8-ND signature, found {len(matches)}; "
            f"inspect {native / 'binary/ascend950/gen'}"
        )
    param = matches[0]
    wrapper = native / "binary/ascend950/src/mega_moe_apt/MegaMoe.py"
    code = wrapper.read_text()
    code = code.replace(
        'ascendc_src_file = "mega_moe_apt.cpp"',
        'ascendc_src_file = "arch35/mega_moe_apt.cpp"',
    )
    wrapper.write_text(code)
    # Never pick an old .o left over from another configuration.
    kernel = dest / "kernel"
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    opc = dest / "opc"
    opc.mkdir(exist_ok=True)
    env.update(
        ASCEND_CUSTOM_OPP_PATH=str(native / "custom"),
        HI_PYTHON=sys.executable,
        TILINGKEY_PAR_COMPILE="1",
        BIN_FILENAME_HASHED="1",
    )
    shutil.copy2(param, dest / "compiler_parameters.json")
    run(
        [
            os.environ.get("ASC_OPC", "asc_opc"),
            wrapper,
            "--main_func=mega_moe",
            f"--input_param={param}",
            f"--soc_version={cfg['build']['soc_version']}",
            f"--output={kernel}",
            "--impl_mode=high_performance,optional",
            "--simplified_key_mode=0",
            "--op_mode=dynamic",
            f"--tiling_key={cfg['model']['tiling_key']}",
        ],
        opc,
        logs / "kernel.log",
        env,
    )
    objects = list(kernel.glob("*.o"))
    if len(objects) != 1:
        raise RuntimeError(f"Expected exactly one kernel ELF, found {objects}")
    from elf import images

    bins, symbols = images(objects[0], cfg["model"]["tiling_key"])
    (dest / "bin").mkdir(exist_ok=True)
    for kind, data in bins.items():
        if len(data) > 0x1000000:
            raise RuntimeError("Kernel image exceeds the 16 MiB code reservation")
        (dest / "bin" / f"{kind}.bin").write_bytes(data)
    if source_hash() != before:
        raise RuntimeError("Source changed during build; run build again")
    revisions = {}
    for d in deps.iterdir():
        if (d / ".git").exists():
            revisions[d.name] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=d, text=True
            ).strip()
    write(
        dest / "manifest.json",
        {
            "source_sha256": before,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "topk_weights_dtype": cfg.get("topk_weights_dtype", "bfloat16"),
            "tiling_key": cfg["model"]["tiling_key"],
            "soc_version": cfg["build"]["soc_version"],
            "elf": str(objects[0].relative_to(dest)),
            "elf_sha256": sha(objects[0]),
            "helper_sha256": sha(dest / "tiling_helper"),
            "dependencies": revisions,
            "marks": mark_catalog,
            "cann": str(cann),
            "signature": read(dest / "compiler_parameters.json"),
            "elf_symbols": symbols,
        },
    )
    print(f"Built key {cfg['model']['tiling_key']}: {objects[0]}")


def compile_helper(dest, cann, env):
    host = ROOT / "mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp"
    source = host.read_text()
    marker = "IMPL_OP_OPTILING(MegaMoe)"
    if source.count(marker) != 1:
        raise RuntimeError("Host registration changed; update tiling helper adapter")
    helper = dest / "tiling_helper.cpp"
    helper.write_text(
        source.split(marker)[0] + "\n}\n" + (ROOT / "sim/tiling_driver.cpp").read_text()
    )
    flags = (
        dest
        / "native/mc2/mega_moe/op_host/CMakeFiles/ophost_transformer_tiling_obj.dir/flags.make"
    ).read_text()
    args = []
    for key in ("CXX_DEFINES", "CXX_INCLUDES"):
        args += shlex.split(
            next(
                line.split(" = ", 1)[1]
                for line in flags.splitlines()
                if line.startswith(key + " = ")
            )
        )
    run(
        [
            os.environ.get("CXX", "g++"),
            *args,
            "-I" + str(host.parent),
            "-std=c++17",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            helper,
            "-L" + str(cann / "lib64"),
            "-Wl,-rpath," + str(cann / "lib64"),
            "-lregister",
            "-lgraph",
            "-lunified_dlog",
            "-lmetadef",
            "-o",
            dest / "tiling_helper",
        ],
        dest,
        dest / "logs/tiling_compile.log",
        env,
    )


def check_build(cfg, dest):
    dest = Path(dest).resolve()
    manifest = read(dest / "manifest.json")
    if manifest["source_sha256"] != source_hash():
        raise RuntimeError(
            "Source/scripts changed since compilation; run ./mega-sim build first"
        )
    if (
        manifest["tiling_key"] != cfg["model"]["tiling_key"]
        or manifest["soc_version"] != cfg["build"]["soc_version"]
    ):
        raise RuntimeError("Build key/SoC does not match case config; rebuild")
    if manifest.get("topk_weights_dtype", "bfloat16") != cfg.get(
        "topk_weights_dtype", "bfloat16"
    ):
        raise RuntimeError(
            "Build topk weight dtype does not match case config; rebuild"
        )
    for path, expected in (
        (dest / manifest["elf"], manifest["elf_sha256"]),
        (dest / "tiling_helper", manifest["helper_sha256"]),
    ):
        if sha(path) != expected:
            raise RuntimeError(f"Build artifact changed: {path}")
    return manifest

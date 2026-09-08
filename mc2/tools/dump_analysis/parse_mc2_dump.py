#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unified MC2 DFX dump parser.

Only the operator name is required when this script is run in a dump directory.
The script discovers and dispatches tiling_data, args_info, workspace segments,
CCU xn/cke address tables, and URMA context files automatically.  Each
source dump gets an independent readable text report; no combined report or
NumPy output is generated.
"""

import argparse
import importlib
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

PROFILES = {
    "allgathermatmulv2": {
        "name": "AllGatherMatmulV2",
        "backend": "parse_all_gather_matmul_v2_tiling.py",
        "module": "parse_all_gather_matmul_v2_tiling",
        "path": "CCU",
    },
    "allgathermatmulv3": {
        "name": "AllGatherMatmulV3",
        "backend": "parse_all_gather_matmul_v3_tiling.py",
        "module": "parse_all_gather_matmul_v3_tiling",
        "path": "AIV + URMA",
    },
    "alltoallmatmul": {
        "name": "AlltoAllMatmul",
        "backend": "parse_allto_all_matmul_tiling.py",
        "module": "parse_allto_all_matmul_tiling",
        "path": "mainline CCU / Apace CCU (auto-detect)",
    },
    "alltoallmatmulv2": {
        "name": "AlltoAllMatmulV2",
        "backend": "parse_allto_all_matmul_v2_tiling.py",
        "module": "parse_allto_all_matmul_v2_tiling",
        "path": "AIV + URMA",
    },
    "matmulreducescatterv2": {
        "name": "MatmulReduceScatterV2",
        "backend": "parse_matmul_reduce_scatter_v2_tiling.py",
        "module": "parse_matmul_reduce_scatter_v2_tiling",
        "path": "CCU",
    },
}

ALIASES = {
    "agmmv2": "allgathermatmulv2",
    "agmmv3": "allgathermatmulv3",
    "a2amm": "alltoallmatmul",
    "a2ammv2": "alltoallmatmulv2",
    "alltoallmatmulapace": "alltoallmatmul",
    "alltoallmatmulmainline": "alltoallmatmul",
    "matmulreducescatter": "matmulreducescatterv2",
    "mmrs": "matmulreducescatterv2",
    "mmrsv2": "matmulreducescatterv2",
}

FILE_KINDS = (
    ("tiling", re.compile(r"^tiling_data_", re.IGNORECASE)),
    ("args", re.compile(r"^args_info_", re.IGNORECASE)),
    ("xn", re.compile(r"^xn_addr_info_", re.IGNORECASE)),
    ("cke", re.compile(r"^cke_addr_info_", re.IGNORECASE)),
    ("comm_context", re.compile(r"^comm_context_", re.IGNORECASE)),
    ("workspace", re.compile(r"^workspace_seg\d+_type\d+_", re.IGNORECASE)),
)

KIND_ORDER = {
    "tiling": 0,
    "args": 1,
    "xn": 2,
    "cke": 3,
    "comm_context": 4,
    "workspace": 5,
}

WORKSPACE_SEG_TYPE_NAMES = {
    0: "LIB_API", 1: "ND2NZ", 2: "GATHER", 3: "GATHER_SCALE1",
    4: "GATHER_SCALE", 5: "COMM_OUT", 6: "PERMUTE_OUT", 7: "BIAS",
    8: "MM_RESULT", 9: "RECV_BUF", 10: "COMM_INT8", 11: "DYNAMIC_QUANT",
    12: "RUNTIME_INFO", 13: "MM_WORKSPACE", 255: "RESERVED",
}

WORKSPACE_SEG_DESCRIPTIONS = {
    0: "系统库预留 Workspace，不属于算子用户数据",
    1: "ND 到 NZ 格式转换的中间缓冲区",
    2: "Gather 通信数据缓冲区",
    3: "Gather 的第一路 Scale 缓冲区",
    4: "Gather Scale 缓冲区",
    5: "通信输出缓冲区",
    6: "转置输出缓冲区",
    7: "Bias 缓冲区",
    8: "Matmul 结果；在部分融合通路中也作为通信发送缓冲区",
    9: "通信接收缓冲区",
    10: "INT8 通信临时缓冲区",
    11: "动态量化临时缓冲区",
    12: "每核 RuntimeInfo，记录执行位置和通信 commit/wait 情况",
    13: "Matmul 内部临时 Workspace，不是 Matmul 结果，元素类型和逻辑形状不固定",
    255: "对齐或预留空间",
}

WORKSPACE_PREVIEW_SIZE = 64

def normalize(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_profile(operator_name):
    key = normalize(operator_name)
    key = ALIASES.get(key, key)
    if key not in PROFILES:
        choices = ", ".join(profile["name"] for profile in PROFILES.values())
        raise ValueError(f"不支持的算子名 {operator_name!r}。支持: {choices}; 别名: mmrs/a2amm/agmmv2/agmmv3")
    return key, PROFILES[key]


def classify_file(path):
    for kind, pattern in FILE_KINDS:
        if pattern.match(path.name):
            return kind
    return None


def matches_operator(path, profile_key):
    normalized_name = normalize(path.name)
    if profile_key == "alltoallmatmul":
        return "alltoallmatmul" in normalized_name and "alltoallmatmulv2" not in normalized_name
    if profile_key == "matmulreducescatterv2":
        return "matmulreducescatter" in normalized_name
    return profile_key in normalized_name


def discover_files(dump_dir, output_dir, profile_key):
    all_files = []
    for path in dump_dir.rglob("*"):
        if not path.is_file() or output_dir in path.parents:
            continue
        # Skip analysis outputs from any previous run, even one rooted at a
        # different dump_dir, so stale TXT reports are never re-parsed as dumps.
        if any(parent.name == output_dir.name for parent in path.parents):
            continue
        all_files.append(path)
    files = []
    for path in all_files:
        kind = classify_file(path)
        if kind is not None and matches_operator(path, profile_key):
            files.append((kind, path))
    return sorted(files, key=lambda item: (KIND_ORDER[item[0]], str(item[1]).lower()))


def read_data(path):
    with path.open("rb") as file_handle:
        return file_handle.read()


def read_preview(path, size=256):
    with path.open("rb") as file_handle:
        return file_handle.read(size)


def emit_decimal_byte_preview(lines, data):
    """Print an unambiguous decimal preview when the segment element type is unknown."""
    emit(lines, "previewType: uint8 raw bytes")
    emit(lines, f"previewRange: byte[0:{len(data)}]")
    emit(lines, f"previewAllZero: {not any(data)}")
    emit(lines, "decimalPreview (each value is one raw uint8 byte):")
    for offset in range(0, len(data), 16):
        row = data[offset:offset + 16]
        values = " ".join(f"{value:3d}" for value in row)
        emit(lines, f"  byte[{offset:04d}:{offset + len(row):04d}] : {values}")


def segment_identity(path, prefix):
    match = re.match(rf"^{prefix}_seg(\d+)_type(\d+)_", path.name, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def infer_aic_core_num(profile, tiling_file):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = importlib.import_module(profile["module"])
        data = read_data(tiling_file)
        name = profile["name"]
        if name == "AllGatherMatmulV3":
            return module.parse_tiling_data(data).get("mmTile", {}).get("usedCoreNum")
        if name == "AlltoAllMatmulV2":
            return module.parse_tiling_data(data).get("tileQbmmTilingData", {}).get("usedCoreNum")
        if name == "AlltoAllMatmul":
            tiling_format = module.detect_tiling_format(data)
            if tiling_format == "apace":
                result = module.parse_apace_tiling_data(data)
                return result.get("tileQbmmTilingData", {}).get("usedCoreNum")
            info = module.parse_allto_all_matmul_tiling_info(data, module.OFFSET_TILING_INFO)
            return info.get("aicCoreNum")
        if name == "AllGatherMatmulV2":
            layout = module.compute_layout()
        else:
            layout = module.compute_mrs_v2_layout()
        is_quant = module.detect_quant_mode(data, layout)
        return module.parse_tiling_data(data, layout, is_quant).get("param", {}).get("aicCoreNum")
    except Exception:
        return None
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def infer_apace_parents(profile, tiling_files):
    """alltoallmatmul: 返回 tiling 为 Apace CCU 格式的 rank 目录集合，用于 --apace 标签。"""
    if profile["name"] != "AlltoAllMatmul":
        return set()
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = importlib.import_module(profile["module"])
        parents = set()
        for tiling_file in tiling_files:
            try:
                if module.detect_tiling_format(read_data(tiling_file)) == "apace":
                    parents.add(tiling_file.parent)
            except (OSError, struct.error, ValueError, KeyError):
                continue
        return parents
    except ImportError:
        return set()
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def operator_suffix(path, profile):
    """Return the invocation suffix shared by tiling/workspace files."""
    candidates = [profile["name"]]
    if profile["name"] == "MatmulReduceScatterV2":
        candidates.append("MatmulReduceScatter")
    lowered = path.name.lower()
    for candidate in candidates:
        index = lowered.find(candidate.lower())
        if index >= 0:
            return lowered[index + len(candidate):]
    return lowered


def build_core_index(profile, tiling_files):
    exact = {}
    per_parent = {}
    values = set()
    for tiling_file in tiling_files:
        core_num = infer_aic_core_num(profile, tiling_file)
        if core_num is None:
            continue
        exact[(tiling_file.parent, operator_suffix(tiling_file, profile))] = core_num
        per_parent.setdefault(tiling_file.parent, set()).add(core_num)
        values.add(core_num)
    return exact, per_parent, values


def core_num_for_file(path, profile, core_index):
    exact, per_parent, values = core_index
    core_num = exact.get((path.parent, operator_suffix(path, profile)))
    if core_num is not None:
        return core_num
    parent_values = per_parent.get(path.parent, set())
    if len(parent_values) == 1:
        return next(iter(parent_values))
    if len(values) == 1:
        return next(iter(values))
    return None


def parse_address_table(input_path):
    """Parse a little-endian uint64 XN/CKE table into readable text lines."""
    byte_size = input_path.stat().st_size
    row_size = 64 * 8
    if byte_size % row_size != 0:
        raise ValueError(f"文件大小 {byte_size} 不是完整 512 字节行的整数倍")
    data = read_data(input_path)
    table = [struct.unpack_from("<64Q", data, offset) for offset in range(0, len(data), row_size)]
    nonzero = sum(1 for row in table for value in row if value != 0)
    lines = [
        f"source: {input_path.name}",
        "dtype: uint64 (little-endian)",
        f"shape: ({len(table)}, 64)",
        f"nonzero: {nonzero}",
    ]
    for index, row in enumerate(table):
        cells = " ".join(f"0x{value:016x}" for value in row)
        lines.append(f"row[{index:04d}] {cells}")
    return len(table), len(table) * 64, nonzero, lines


def backend_command(profile, kind, path, aic_core_num, use_apace=False):
    command = [sys.executable, str(SCRIPT_DIR / profile["backend"]), str(path)]
    if kind == "args":
        command.append("--args")
    elif kind == "workspace":
        command.append("--workspace")
        if aic_core_num is not None:
            command.extend(("--aic-core-num", str(aic_core_num)))
        if use_apace:
            command.append("--apace")
    elif kind == "comm_context":
        command.append("--comm-ctx")
    return command


def run_backend(profile, kind, path, aic_core_num, use_apace=False):
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        backend_command(profile, kind, path, aic_core_num, use_apace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        check=False,
    )
    return process.returncode, process.stdout.rstrip()


def emit(lines, value=""):
    print(value)
    lines.append(value)


def text_output_path(output_dir, relative_path):
    """Return the independent TXT path corresponding to one source dump."""
    output_path = output_dir / relative_path.parent / f"{relative_path.name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_text_result(output_path, profile, kind, relative_path, lines):
    """Save one source file's readable parsing result."""
    header = [
        f"operator: {profile['name']}",
        f"kind: {kind}",
        f"source: {relative_path}",
        "=" * 88,
    ]
    output_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="按算子名自动解析 MC2 tiling/workspace/xn/cke/URMA dump，并逐文件输出 TXT",
        epilog=(
            "在 dump 目录中只需执行: python3 parse_mc2_dump.py AlltoAllMatmul\n"
            "也可指定目录: python3 parse_mc2_dump.py AllGatherMatmulV3 /path/to/dump"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("operator", help="算子名或别名，例如 AlltoAllMatmul、AllGatherMatmulV3、mmrs")
    parser.add_argument("dump_dir", nargs="?", default=".", help="dump 根目录，默认当前目录")
    args = parser.parse_args(argv)

    try:
        profile_key, profile = resolve_profile(args.operator)
    except ValueError as error:
        parser.error(str(error))

    dump_dir = Path(args.dump_dir).expanduser().resolve()
    if not dump_dir.is_dir():
        parser.error(f"dump 目录不存在: {dump_dir}")
    output_dir = dump_dir / "mc2_dump_analysis" / profile["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Old versions wrote one combined report.  Remove it so users do not read
    # stale results after switching to per-source TXT output.
    (output_dir / "analysis.txt").unlink(missing_ok=True)

    files = discover_files(dump_dir, output_dir, profile_key)
    if not files:
        parser.error(f"在 {dump_dir} 中没有找到 {profile['name']} 的受支持 dump 文件")

    tiling_files = [path for kind, path in files if kind == "tiling"]
    core_index = build_core_index(profile, tiling_files)
    apace_parents = infer_apace_parents(profile, tiling_files)
    core_values = sorted(core_index[2])
    core_summary = core_values[0] if len(core_values) == 1 else (core_values or "unknown")
    console_lines = []
    emit(console_lines, "=" * 88)
    emit(console_lines, f"MC2 dump analysis: {profile['name']} | path: {profile['path']}")
    emit(console_lines, f"dump dir: {dump_dir}")
    emit(console_lines, f"matched files: {len(files)} | inferred AIC cores: {core_summary}")
    emit(console_lines, f"TXT output: {output_dir}")
    emit(console_lines, "=" * 88)

    failures = 0
    for kind, path in files:
        relative_path = path.relative_to(dump_dir)
        output_path = text_output_path(output_dir, relative_path)
        result_lines = []
        emit(console_lines)
        emit(console_lines, f"[{kind}] {relative_path}")
        emit(console_lines, "-" * 88)

        if kind in ("xn", "cke"):
            try:
                rows, values, nonzero, result_lines = parse_address_table(path)
                emit(
                    console_lines,
                    f"parsed: dtype=uint64 shape=({rows}, 64) values={values} nonzero={nonzero}",
                )
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                result_lines.append(f"ERROR: {error}")
                emit(console_lines, result_lines[-1])
            save_text_result(output_path, profile, kind, relative_path, result_lines)
            emit(console_lines, f"saved TXT: {output_path}")
            continue

        # Large data segments only need metadata and a short preview in their
        # own report. RuntimeInfo remains structurally parsed by the backend.
        if kind == "workspace":
            identity = segment_identity(path, "workspace")
            if identity is not None and identity[1] != 12:
                seg_index, seg_type = identity
                emit(result_lines, f"segIdx: {seg_index}")
                emit(result_lines, f"segType: {seg_type} ({WORKSPACE_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})")
                emit(result_lines, f"dataMeaning: {WORKSPACE_SEG_DESCRIPTIONS.get(seg_type, '未知 Workspace 段')}")
                emit(result_lines, f"fileSize: {path.stat().st_size} bytes")
                emit_decimal_byte_preview(result_lines, read_preview(path, WORKSPACE_PREVIEW_SIZE))
                emit(result_lines, "typedViewHint: 使用 parse_matrix.py 并指定 dtype/rows/cols 查看真实数值")
                save_text_result(output_path, profile, kind, relative_path, result_lines)
                emit(console_lines, f"saved TXT: {output_path}")
                continue

        aic_core_num = core_num_for_file(path, profile, core_index) if kind == "workspace" else None
        use_apace = kind == "workspace" and path.parent in apace_parents
        return_code, output = run_backend(profile, kind, path, aic_core_num, use_apace)
        if output:
            for line in output.splitlines():
                emit(result_lines, line)
        if kind == "comm_context":
            file_size = path.stat().st_size
            emit(result_lines, "")
            emit(result_lines, "rawDataHint: 如需按十进制查看 comm_context 原始字节，可使用 parse_matrix.py：")
            emit(
                result_lines,
                f'python3 "{SCRIPT_DIR / "parse_matrix.py"}" "{path}" '
                f'--dtype uint8 --rows 1 --cols {file_size} --row 0 --col-range 0 64',
            )
        if return_code != 0:
            failures += 1
            emit(result_lines, f"ERROR: backend exited with code {return_code}")
        save_text_result(output_path, profile, kind, relative_path, result_lines)
        emit(console_lines, f"saved TXT: {output_path}")

    print()
    print(f"解析完成: {len(files) - failures}/{len(files)} 个文件成功")
    print(f"逐文件 TXT 位于: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

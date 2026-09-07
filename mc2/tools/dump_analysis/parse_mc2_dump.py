#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unified MC2 exception-dump parser.

Only the operator name is required when this script is run in a dump directory.
The script discovers and dispatches tiling_data, args_info, workspace segments,
CCU xn/cke address tables, URMA context/peermem files, and exception_info
files automatically. Every source dump is exported to an independent NumPy NPY
file; no combined text report is generated.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
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
    ("peermem", re.compile(r"^peermem_(?:window|seg\d+_type\d+)_", re.IGNORECASE)),
    ("workspace", re.compile(r"^workspace_seg\d+_type\d+_", re.IGNORECASE)),
    ("exception", re.compile(r"^exception_info(?:\.|$)", re.IGNORECASE)),
)

KIND_ORDER = {
    "tiling": 0,
    "args": 1,
    "xn": 2,
    "cke": 3,
    "comm_context": 4,
    "peermem": 5,
    "workspace": 6,
    "exception": 7,
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

PEERMEM_SEG_TYPE_NAMES = {
    0: "DATA", 1: "SCALE", 2: "BIAS", 3: "FLAG", 255: "RESERVED",
}


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
            if kind == "exception" and not is_raw_exception(path):
                return None
            return kind
    return None


def is_raw_exception(path):
    """Return True only for the raw exception container, not msaicerr products."""
    lowered = path.name.lower()
    if not lowered.startswith("exception_info"):
        return False
    parsed_markers = (".input.", ".output.", ".workspace.", ".npy")
    return not any(marker in lowered for marker in parsed_markers)


def matches_operator(path, profile_key):
    normalized_name = normalize(path.name)
    if profile_key == "alltoallmatmul":
        return "alltoallmatmul" in normalized_name and "alltoallmatmulv2" not in normalized_name
    if profile_key == "matmulreducescatterv2":
        return "matmulreducescatter" in normalized_name
    return profile_key in normalized_name


def discover_files(dump_dir, output_dir, profile_key):
    all_files = [
        path for path in dump_dir.rglob("*")
        if path.is_file() and output_dir not in path.parents
    ]
    profile = PROFILES[profile_key]
    operator_files = [path for path in all_files if matches_operator(path, profile_key)]
    operator_dirs = {path.parent for path in operator_files}
    operator_invocations = {}
    for path in operator_files:
        invocation = operator_suffix(path, profile)
        if invocation is not None:
            operator_invocations.setdefault(path.parent, set()).add(invocation)
    raw_exceptions = [path for path in all_files if is_raw_exception(path)]
    # A raw exception filename does not contain the operator name. Associate it
    # with the operator through sibling dump/.o/.json files in the same rank dir.
    if not operator_dirs and raw_exceptions:
        operator_dirs = {path.parent for path in raw_exceptions}

    files = []
    for path in all_files:
        kind = classify_file(path)
        if kind == "exception" and path.parent in operator_dirs:
            invocation = exception_suffix(path)
            known_invocations = operator_invocations.get(path.parent, set())
            if invocation is None or not known_invocations or invocation in known_invocations:
                files.append((kind, path))
        elif kind is not None and matches_operator(path, profile_key):
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
        is_quant = module.detect_quant_mode(data)
        parser = module.parse_quant_tiling_data if is_quant else module.parse_non_quant_tiling_data
        return parser(data).get("param", {}).get("aicCoreNum")
    except (ImportError, OSError, struct.error, KeyError, TypeError, ValueError):
        return None
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def stream_task_suffix(value):
    """Normalize ``.streamId.taskId.timestamp`` to ``.streamId.taskId``."""
    match = re.match(r"^\.(\d+)\.(\d+)(?:\.|$)", value)
    return f".{match.group(1)}.{match.group(2)}" if match else None


def operator_suffix(path, profile):
    """Return the stream/task identity shared by files from one invocation."""
    candidates = [profile["name"]]
    if profile["name"] == "MatmulReduceScatterV2":
        candidates.append("MatmulReduceScatter")
    lowered = path.name.lower()
    for candidate in candidates:
        index = lowered.find(candidate.lower())
        if index >= 0:
            suffix = lowered[index + len(candidate):]
            return stream_task_suffix(suffix)
    return None


def exception_suffix(path):
    """Return the stream/task identity embedded in ``exception_info.*``."""
    lowered = path.name.lower()
    suffix = lowered[len("exception_info"):]
    return stream_task_suffix(suffix)


def build_core_index(profile, tiling_files):
    exact = {}
    per_parent = {}
    values = set()
    for tiling_file in tiling_files:
        core_num = infer_aic_core_num(profile, tiling_file)
        if core_num is None:
            continue
        invocation = operator_suffix(tiling_file, profile)
        if invocation is not None:
            exact[(tiling_file.parent, invocation)] = core_num
        per_parent.setdefault(tiling_file.parent, set()).add(core_num)
        values.add(core_num)
    return exact, per_parent, values


def core_num_for_file(path, profile, core_index):
    exact, per_parent, values = core_index
    invocation = operator_suffix(path, profile)
    core_num = exact.get((path.parent, invocation)) if invocation is not None else None
    if core_num is not None:
        return core_num
    parent_values = per_parent.get(path.parent, set())
    if len(parent_values) == 1:
        return next(iter(parent_values))
    if len(values) == 1:
        return next(iter(values))
    return None


def parse_address_table(input_path, output_path):
    """Parse a little-endian uint64 XN/CKE table and save a (rows, 64) NPY."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("解析 XN/CKE 并导出 NPY 需要 NumPy，请先安装 numpy") from error

    byte_size = input_path.stat().st_size
    row_size = 64 * 8
    if byte_size % row_size != 0:
        raise ValueError(f"文件大小 {byte_size} 不是完整 512 字节行的整数倍")
    values = np.fromfile(input_path, dtype="<u8")
    table = values.reshape((-1, 64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, table, allow_pickle=False)
    return table.shape[0], table.size, int(np.count_nonzero(table))


def require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("导出 NPY 需要 NumPy，请先安装 numpy") from error
    return np


def npy_output_path(output_dir, relative_path):
    """Return the independent NPY path for one source dump."""
    output_path = output_dir / relative_path.parent / f"{relative_path.name}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_structured_npy(output_path, parsed):
    """Save a JSON-like parser result as one object NPY."""
    np = require_numpy()
    np.save(output_path, parsed, allow_pickle=True)


def save_raw_npy(input_path, output_path):
    """Save an untyped data segment losslessly as a memory-mapped uint8 NPY."""
    np = require_numpy()
    if input_path.stat().st_size == 0:
        raw = np.empty((0,), dtype=np.uint8)
        np.save(output_path, raw, allow_pickle=False)
        return raw.shape, raw.dtype
    raw = np.memmap(input_path, dtype=np.uint8, mode="r")
    try:
        np.save(output_path, raw, allow_pickle=False)
        return raw.shape, raw.dtype
    finally:
        del raw


def find_msaicerr(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    for variable in ("MSAICERR_PATH", "ASCEND_TOOLKIT_HOME", "TOOL_PATH"):
        value = os.environ.get(variable)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if variable != "MSAICERR_PATH":
            candidate = candidate / "tools" / "msaicerr" / "msaicerr.py"
        candidates.append(candidate)
    candidates.extend((
        Path("/home/w00955510/Ascend0901/cann-9.2.0/tools/msaicerr/msaicerr.py"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/tools/msaicerr/msaicerr.py"),
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def run_command(command, cwd=None):
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        [str(item) for item in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
    )
    return process.returncode, process.stdout.rstrip()


def run_embedded_msaicerr(source, output_dir, msaicerr_path):
    """Call the parsing core used by msaicerr.py directly.

    ``msaicerr.py`` is only a CLI wrapper. Its ``-d`` branch ultimately calls
    ``DumpDataParser(...).parse()``; using that API here removes the extra
    manual command while keeping the implementation supplied by CANN.
    """
    added_path = None
    if msaicerr_path is not None:
        added_path = str(msaicerr_path.parent)
        if added_path not in sys.path:
            sys.path.insert(0, added_path)
        else:
            added_path = None

    capture = io.StringIO()
    try:
        try:
            from ms_interface.aic_error_info import AicErrorInfo
            from ms_interface.dump_data_parser import DumpDataParser
        except ImportError as error:
            return None, f"embedded msaicerr unavailable: {error}"

        try:
            with redirect_stdout(capture), redirect_stderr(capture):
                dump_info = AicErrorInfo()
                DumpDataParser(str(source), dump_info, "", str(output_dir)).parse()
            return 0, capture.getvalue().rstrip()
        except Exception as error:  # CANN parser uses several implementation-specific exception types.
            output = capture.getvalue().rstrip()
            message = f"embedded DumpDataParser failed: {type(error).__name__}: {error}"
            return 1, "\n".join(item for item in (output, message) if item)
    finally:
        if added_path is not None and sys.path and sys.path[0] == added_path:
            sys.path.pop(0)


def export_exception_npy(report_dir, npy_output, operator):
    sys.path.insert(0, str(SCRIPT_DIR))
    capture = io.StringIO()
    try:
        module = importlib.import_module("parse_mc2_exception_report")
        with redirect_stdout(capture), redirect_stderr(capture):
            module.export_report(report_dir, npy_output, operator)
        return 0, capture.getvalue().rstrip()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        output = capture.getvalue().rstrip()
        message = f"NPY export failed: {type(error).__name__}: {error}"
        return 1, "\n".join(item for item in (output, message) if item)
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def parse_exception_info(profile, source, relative_path, output_dir, msaicerr_path):
    invocation_dir = output_dir / "exception" / relative_path.parent / safe_path_component(source.name)
    msaicerr_output = invocation_dir / "msaicerr"
    npy_output = invocation_dir / "npy"
    msaicerr_output.mkdir(parents=True, exist_ok=True)
    details = [f"msaicerr output: {msaicerr_output}"]
    return_code, output = run_embedded_msaicerr(source, msaicerr_output, msaicerr_path)
    if return_code is None:
        details.append(output)
        if msaicerr_path is None:
            details.append(
                "ERROR: 未找到 CANN ms_interface；请确认 CANN 环境已加载，"
                "或使用 --msaicerr 指定 msaicerr.py"
            )
            return 1, details
        msaicerr_command = [
            sys.executable,
            str(msaicerr_path),
            "-d",
            str(source),
            "--output_path",
            str(msaicerr_output),
        ]
        # Compatibility fallback for CANN versions whose Python API cannot be
        # imported directly from this process.
        return_code, output = run_command(msaicerr_command, cwd=source.parent)
        details.append("msaicerr mode: command-line fallback")
    else:
        details.append("msaicerr mode: embedded DumpDataParser")
    if output:
        details.extend(output.splitlines())
    if return_code != 0:
        details.append(f"ERROR: msaicerr exited with code {return_code}")
        return return_code, details

    return_code, output = export_exception_npy(msaicerr_output, npy_output, profile["name"])
    details.append(f"NPY output: {npy_output}")
    if output:
        details.extend(output.splitlines())
    if return_code != 0:
        details.append(f"ERROR: exception report parser exited with code {return_code}")
    return return_code, details


def backend_command(profile, kind, path, aic_core_num):
    command = [sys.executable, str(SCRIPT_DIR / profile["backend"]), str(path)]
    if kind == "args":
        command.append("--args")
    elif kind == "workspace":
        command.append("--workspace")
        if aic_core_num is not None:
            command.extend(("--aic-core-num", str(aic_core_num)))
    elif kind == "comm_context":
        command.append("--comm-ctx")
    elif kind == "peermem":
        command.append("--peermem")
    command.append("--json")
    return command


def run_backend(profile, kind, path, aic_core_num):
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        backend_command(profile, kind, path, aic_core_num),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        check=False,
    )
    return process.returncode, process.stdout.rstrip(), process.stderr.rstrip()


def emit(lines, value=""):
    print(value)
    lines.append(value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="按算子名自动解析 MC2 tiling/workspace/xn/cke/URMA/exception dump",
        epilog=(
            "在 dump 目录中只需执行: python3 parse_mc2_dump.py AlltoAllMatmul\n"
            "也可指定目录: python3 parse_mc2_dump.py AllGatherMatmulV3 /path/to/dump"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("operator", help="算子名或别名，例如 AlltoAllMatmul、AllGatherMatmulV3、mmrs")
    parser.add_argument("dump_dir", nargs="?", default=".", help="dump 根目录，默认当前目录")
    parser.add_argument(
        "--msaicerr",
        help="msaicerr.py 路径；未指定时读取 MSAICERR_PATH/ASCEND_TOOLKIT_HOME/TOOL_PATH 并检查常用路径",
    )
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
    (output_dir / "analysis.txt").unlink(missing_ok=True)

    files = discover_files(dump_dir, output_dir, profile_key)
    if not files:
        parser.error(f"在 {dump_dir} 中没有找到 {profile['name']} 的受支持 dump 文件")

    tiling_files = [path for kind, path in files if kind == "tiling"]
    core_index = build_core_index(profile, tiling_files)
    core_values = sorted(core_index[2])
    core_summary = core_values[0] if len(core_values) == 1 else (core_values or "unknown")
    report_lines = []
    emit(report_lines, "=" * 88)
    emit(report_lines, f"MC2 dump analysis: {profile['name']} | path: {profile['path']}")
    emit(report_lines, f"dump dir: {dump_dir}")
    emit(report_lines, f"matched files: {len(files)} | inferred AIC cores: {core_summary}")
    emit(report_lines, "=" * 88)

    failures = 0
    exception_files = [path for kind, path in files if kind == "exception"]
    msaicerr_path = find_msaicerr(args.msaicerr) if exception_files else None
    for kind, path in files:
        relative_path = path.relative_to(dump_dir)
        emit(report_lines)
        emit(report_lines, f"[{kind}] {relative_path}")
        emit(report_lines, "-" * 88)
        if kind in ("xn", "cke"):
            relative_parent = relative_path.parent
            address_dir = output_dir / relative_parent
            address_dir.mkdir(parents=True, exist_ok=True)
            output_path = address_dir / f"{path.name}.npy"
            try:
                rows, values, nonzero = parse_address_table(path, output_path)
                message = (
                    f"saved NPY: dtype=uint64 shape=({rows}, 64) values={values} "
                    f"nonzero={nonzero} -> {output_path}"
                )
                emit(report_lines, message)
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                emit(report_lines, f"ERROR: {error}")
            continue

        if kind == "exception":
            emit(report_lines, f"msaicerr: {msaicerr_path or 'CANN ms_interface from Python environment'}")
            return_code, details = parse_exception_info(
                profile, path, relative_path, output_dir, msaicerr_path
            )
            for line in details:
                emit(report_lines, line)
            if return_code != 0:
                failures += 1
            continue

        # RuntimeInfo remains structurally parsed; other data segments are saved as raw uint8.
        if kind == "workspace":
            identity = segment_identity(path, "workspace")
            if identity is not None and identity[1] != 12:
                seg_index, seg_type = identity
                output_path = npy_output_path(output_dir, relative_path)
                try:
                    shape, dtype = save_raw_npy(path, output_path)
                    emit(report_lines, f"segIdx: {seg_index}")
                    emit(report_lines, f"segType: {seg_type} ({WORKSPACE_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})")
                    emit(report_lines, f"dataMeaning: {WORKSPACE_SEG_DESCRIPTIONS.get(seg_type, '未知 Workspace 段')}")
                    emit(report_lines, f"saved NPY: dtype={dtype} shape={shape} -> {output_path}")
                    emit_decimal_byte_preview(report_lines, read_preview(path, WORKSPACE_PREVIEW_SIZE))
                except (OSError, RuntimeError, ValueError) as error:
                    failures += 1
                    emit(report_lines, f"ERROR: {error}")
                continue

        if kind == "peermem":
            identity = segment_identity(path, "peermem")
            if identity is not None:
                seg_index, seg_type = identity
                emit(report_lines, f"segIdx: {seg_index}")
                emit(report_lines, f"segType: {seg_type} ({PEERMEM_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})")
            output_path = npy_output_path(output_dir, relative_path)
            try:
                shape, dtype = save_raw_npy(path, output_path)
                emit(report_lines, f"saved NPY: dtype={dtype} shape={shape} -> {output_path}")
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                emit(report_lines, f"ERROR: {error}")
            continue

        aic_core_num = core_num_for_file(path, profile, core_index) if kind == "workspace" else None
        return_code, output, backend_log = run_backend(profile, kind, path, aic_core_num)
        if return_code != 0:
            failures += 1
            for text in (backend_log, output):
                for line in text.splitlines():
                    emit(report_lines, line)
            emit(report_lines, f"ERROR: backend exited with code {return_code}")
            continue
        output_path = npy_output_path(output_dir, relative_path)
        try:
            parsed = json.loads(output)
            save_structured_npy(output_path, parsed)
            emit(report_lines, f"saved NPY: object -> {output_path}")
        except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
            failures += 1
            if backend_log:
                emit(report_lines, backend_log)
            emit(report_lines, f"ERROR: {error}")

    print()
    print(f"解析完成: {len(files) - failures}/{len(files)} 个文件成功")
    print(f"每个输入文件的独立 NPY 位于: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

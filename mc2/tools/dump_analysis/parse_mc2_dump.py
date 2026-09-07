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
import importlib
import json
import locale
import os
from pathlib import Path
import re
import shutil
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

WORKSPACE_PREVIEW_SIZE = 16

# HCCL dataType in MMRS TilingData -> NumPy storage information.  BF16 and
# FP8 have no portable NumPy arithmetic dtype, so their exact bit patterns are
# retained in uint16/uint8 storage.  This is still more useful than treating a
# BF16 matrix as a flat byte array because its logical matrix shape is kept.
HCCL_WORKSPACE_DTYPES = {
    0: ("int8", 1, "INT8", "typed"),
    1: ("<i2", 2, "INT16", "typed"),
    2: ("<i4", 4, "INT32", "typed"),
    3: ("<i8", 8, "INT64", "typed"),
    4: ("<f2", 2, "FP16", "typed"),
    5: ("<f4", 4, "FP32", "typed"),
    7: ("<f8", 8, "FP64", "typed"),
    8: ("<u2", 2, "BF16", "raw_bits"),
    10: ("uint8", 1, "UINT8", "typed"),
    11: ("<u2", 2, "UINT16", "typed"),
    12: ("<u4", 4, "UINT32", "typed"),
    13: ("<u8", 8, "UINT64", "typed"),
    14: ("uint8", 1, "FP8_E5M2", "raw_bits"),
    15: ("uint8", 1, "FP8_E4M3FN", "raw_bits"),
    16: ("<u2", 2, "BF16", "raw_bits"),
}

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
    """Print one compact decimal preview when the segment dtype is unknown."""
    emit(lines, f"rawBytePreview[0:{len(data)}]: {list(data)} | allZero={not any(data)}")


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


def dump_timestamp(path):
    """Return the numeric timestamp suffix used to pair files from one run."""
    match = re.search(r"\.(\d{14,})$", path.name)
    return int(match.group(1)) if match else None


def nearest_tiling_result(path, profile, parsed_tilings):
    """Find the parsed TilingData that is closest to one dump segment."""
    invocation = operator_suffix(path, profile)
    candidates = [
        (tiling_path, parsed)
        for tiling_path, parsed in parsed_tilings
        if tiling_path.parent == path.parent
        and (invocation is None or operator_suffix(tiling_path, profile) == invocation)
    ]
    if not candidates:
        return None
    timestamp = dump_timestamp(path)
    if timestamp is None:
        return candidates[0][1] if len(candidates) == 1 else None

    def distance(item):
        tiling_timestamp = dump_timestamp(item[0])
        return abs(timestamp - tiling_timestamp) if tiling_timestamp is not None else sys.maxsize

    return min(candidates, key=distance)[1]


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


def mmrs_workspace_spec(seg_type, tiling_result, byte_size):
    """Infer the safe storage dtype/shape of MMRS result/send/receive buffers."""
    if seg_type not in (8, 9) or not isinstance(tiling_result, dict):
        return None
    param = tiling_result.get("param", {})
    rank_m = param.get("rankM")
    rank_n = param.get("rankN")
    dtype_spec = HCCL_WORKSPACE_DTYPES.get(tiling_result.get("dataType"))
    if dtype_spec is None or not isinstance(rank_m, int) or not isinstance(rank_n, int):
        return None
    storage_dtype, item_size, logical_dtype, interpretation = dtype_spec
    if rank_m <= 0 or rank_n <= 0 or byte_size % item_size != 0:
        return None
    element_count = byte_size // item_size
    logical_count = rank_m * rank_n
    shape = (rank_m, rank_n) if logical_count == element_count else (element_count,)
    return {
        "storageDtype": storage_dtype,
        "itemSize": item_size,
        "logicalDtype": logical_dtype,
        "interpretation": interpretation,
        "shape": shape,
        "expectedShape": (rank_m, rank_n),
        "shapeMatched": logical_count == element_count,
    }


def save_typed_npy(input_path, output_path, storage_dtype, shape):
    """Save one typed segment without loading the complete buffer into RAM."""
    np = require_numpy()
    dtype = np.dtype(storage_dtype)
    source = np.memmap(input_path, dtype=dtype, mode="r", shape=shape)
    try:
        target = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)
        source_flat = source.reshape(-1)
        target_flat = target.reshape(-1)
        chunk_elements = max(1, (16 * 1024 * 1024) // dtype.itemsize)
        for begin in range(0, source_flat.size, chunk_elements):
            end = min(begin + chunk_elements, source_flat.size)
            target_flat[begin:end] = source_flat[begin:end]
        target.flush()
        del target
    finally:
        del source
    return str(dtype), shape


def typed_value_preview(input_path, spec, count=8):
    """Return a small numeric preview, decoding BF16 bits for readability."""
    np = require_numpy()
    values = np.fromfile(input_path, dtype=np.dtype(spec["storageDtype"]), count=count)
    if spec["logicalDtype"] == "BF16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
        label = "BF16 decoded as FP32"
    elif spec["interpretation"] == "raw_bits":
        label = f"{spec['logicalDtype']} raw bits"
    else:
        label = spec["logicalDtype"]
    return label, values.tolist()


def safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def parse_exception_info(profile, source, relative_path, output_dir):
    """Parse one raw exception container with the repository-owned decoder."""
    invocation_dir = output_dir / "exception" / relative_path.parent / safe_path_component(source.name)
    details = [f"NPY output: {invocation_dir}"]
    try:
        if invocation_dir.exists():
            shutil.rmtree(invocation_dir)
        invocation_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        details.append(f"ERROR: 无法准备异常解析目录: {error}")
        return 1, details
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = importlib.import_module("parse_mc2_exception_report")
        _, _, parser_lines = module.export_exception_info(
            source, invocation_dir, profile["name"]
        )
        details.extend(parser_lines)
        return 0, details
    except (ImportError, OSError, RuntimeError, ValueError, struct.error) as error:
        details.append(f"ERROR: exception_info 解析失败: {type(error).__name__}: {error}")
        return 1, details
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


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
        env=child_env,
        check=False,
    )
    return process.returncode, decode_child_output(process.stdout), decode_child_output(process.stderr)


def decode_child_output(data):
    """Decode backend output on both UTF-8 Linux and locale-encoded Windows."""
    if not data:
        return ""
    for encoding in ("utf-8", locale.getpreferredencoding(False)):
        try:
            return data.decode(encoding).rstrip()
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace").rstrip()


def run_backend_human(profile, kind, path, aic_core_num):
    """Run the backend's existing readable formatter for terminal display."""
    command = backend_command(profile, kind, path, aic_core_num)
    if command and command[-1] == "--json":
        command.pop()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        check=False,
    )
    return process.returncode, decode_child_output(process.stdout), decode_child_output(process.stderr)


def emit(lines, value=""):
    print(value)
    lines.append(value)


def short_output_path(path, output_dir):
    try:
        return path.relative_to(output_dir)
    except ValueError:
        return path


def emit_runtime_summary(lines, parsed):
    """Print the useful RuntimeInfo counters without expanding 128 raw slots."""
    runtime_info = parsed.get("runtimeInfo", {}) if isinstance(parsed, dict) else {}
    if not runtime_info:
        return
    slots = runtime_info.get("slots", [])
    active = runtime_info.get("activeSlots", 0)
    expected = runtime_info.get("expectedSlots", runtime_info.get("maxSlots", "?"))
    expected_slots = slots[:expected] if isinstance(expected, int) else slots
    valid_slots = [slot for slot in expected_slots if slot.get("magicValid")]
    commit_total = runtime_info.get(
        "commitTotal", sum(slot.get("commCommitCount", 0) for slot in valid_slots)
    )
    wait_total = runtime_info.get(
        "waitTotal", sum(slot.get("commWaitCount", 0) for slot in valid_slots)
    )
    pending_slots = [
        slot for slot in valid_slots
        if slot.get("commCommitCount", 0) != slot.get("commWaitCount", 0)
    ]
    pending_count = runtime_info.get("pendingCoreCount", len(pending_slots))
    emit(
        lines,
        "RuntimeInfo: "
        f"active={active}/{expected} | commit={commit_total} | "
        f"wait={wait_total} | pendingCores={pending_count}",
    )
    if active == 0:
        emit(lines, "WARNING: 没有有效 RuntimeInfo slot；请检查 Kernel 初始化、段起始地址和 Dump 时机")
    elif pending_slots:
        preview = ", ".join(
            f"slot{slot.get('slotIdx', slot.get('coreId', '?'))}({slot.get('coreType', '未知')}:"
            f"commit={slot.get('commCommitCount')},wait={slot.get('commWaitCount')})"
            for slot in pending_slots[:16]
        )
        suffix = " ..." if len(pending_slots) > 16 else ""
        emit(lines, f"unbalanced: {preview}{suffix}")
    emit(lines, "perCoreDetail: NPY -> runtimeInfo.slots[*].commCommitCount/commWaitCount")


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
    kind_counts = {}
    for kind, _ in files:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    file_summary = ", ".join(f"{kind}={count}" for kind, count in kind_counts.items())
    report_lines = []
    emit(report_lines, "=" * 88)
    emit(report_lines, f"MC2 dump analysis: {profile['name']} | path: {profile['path']}")
    emit(report_lines, f"dump dir: {dump_dir}")
    emit(report_lines, f"matched files: {len(files)} ({file_summary})")
    emit(report_lines, f"inferred AIC cores: {core_summary} | NPY output: {output_dir}")
    emit(report_lines, "=" * 88)

    failures = 0
    parsed_tilings = []
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
                    f"nonzero={nonzero} -> {short_output_path(output_path, output_dir)}"
                )
                emit(report_lines, message)
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                emit(report_lines, f"ERROR: {error}")
            continue

        if kind == "exception":
            return_code, details = parse_exception_info(profile, path, relative_path, output_dir)
            for line in details:
                emit(report_lines, line)
            if return_code != 0:
                failures += 1
            continue

        # RuntimeInfo remains structurally parsed.  MMRS result/send/receive
        # buffers use the output dtype and shape recorded in the paired tiling.
        if kind == "workspace":
            identity = segment_identity(path, "workspace")
            if identity is not None and identity[1] != 12:
                seg_index, seg_type = identity
                output_path = npy_output_path(output_dir, relative_path)
                try:
                    emit(report_lines, f"segIdx: {seg_index}")
                    emit(report_lines, f"segType: {seg_type} ({WORKSPACE_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})")
                    emit(report_lines, f"dataMeaning: {WORKSPACE_SEG_DESCRIPTIONS.get(seg_type, '未知 Workspace 段')}")
                    tiling_result = nearest_tiling_result(path, profile, parsed_tilings)
                    spec = (
                        mmrs_workspace_spec(seg_type, tiling_result, path.stat().st_size)
                        if profile_key == "matmulreducescatterv2" else None
                    )
                    if spec is not None:
                        dtype, shape = save_typed_npy(
                            path, output_path, spec["storageDtype"], spec["shape"]
                        )
                        emit(
                            report_lines,
                            f"saved NPY: logicalDtype={spec['logicalDtype']} storageDtype={dtype} "
                            f"shape={shape} interpretation={spec['interpretation']} -> "
                            f"{short_output_path(output_path, output_dir)}",
                        )
                        if not spec["shapeMatched"]:
                            emit(
                                report_lines,
                                f"WARNING: 文件大小与 TilingData shape={spec['expectedShape']} 不匹配，"
                                "NPY 保留为一维数组",
                            )
                        preview_label, preview_values = typed_value_preview(path, spec)
                        emit(report_lines, f"valuePreview[0:{len(preview_values)}] ({preview_label}): {preview_values}")
                    else:
                        shape, dtype = save_raw_npy(path, output_path)
                        emit(
                            report_lines,
                            f"saved NPY: dtype={dtype} shape={shape} interpretation=raw_bytes -> "
                            f"{short_output_path(output_path, output_dir)}",
                        )
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
                emit(
                    report_lines,
                    f"saved NPY: dtype={dtype} shape={shape} -> "
                    f"{short_output_path(output_path, output_dir)}",
                )
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
            if kind == "tiling":
                parsed_tilings.append((path, parsed))
            if kind == "workspace":
                identity = segment_identity(path, "workspace")
                if identity is not None:
                    seg_index, seg_type = identity
                    emit(report_lines, f"segIdx: {seg_index}")
                    emit(report_lines, f"segType: {seg_type} ({WORKSPACE_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})")
                    emit(report_lines, f"dataMeaning: {WORKSPACE_SEG_DESCRIPTIONS.get(seg_type, '未知 Workspace 段')}")
            emit(
                report_lines,
                f"saved NPY: object -> {short_output_path(output_path, output_dir)}",
            )
            if kind in ("tiling", "workspace"):
                display_code, display_output, display_log = run_backend_human(
                    profile, kind, path, aic_core_num
                )
                emit(report_lines, "parsedResult:")
                for text in (display_log, display_output):
                    for line in text.splitlines():
                        emit(report_lines, line)
                if display_code != 0:
                    emit(report_lines, f"WARNING: 可读结果格式化失败，退出码 {display_code}")
            if kind == "workspace":
                emit_runtime_summary(report_lines, parsed)
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

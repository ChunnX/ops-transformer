#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Parse MC2 ``exception_info.*`` files and export independent NumPy arrays.

The exception container consists of an eight-byte little-endian protobuf
header length, a ``DumpData`` protobuf header and consecutive input/output/
buffer/workspace payloads.  This module contains the small protobuf wire
decoder needed for that schema, so parsing does not import ``ms_interface`` or
call CANN's ``msaicerr.py``.

For compatibility, a directory containing already extracted ``*.bin``/NPY
tensors can still be passed to the standalone command.
"""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys


TENSOR_PATTERN = re.compile(r"(?:^|[._])(input|output|workspace)[._](\d+)(?:[._]|$)", re.IGNORECASE)

# Logical dtype -> (NumPy storage dtype, byte size, interpretation).
# NumPy has no portable BF16/FP8 representation, so these formats retain their
# exact bit patterns in an unsigned integer storage dtype.
DTYPE_SPECS = {
    "bool": ("bool", 1, "typed"),
    "int8": ("int8", 1, "typed"),
    "uint8": ("uint8", 1, "typed"),
    "int16": ("<i2", 2, "typed"),
    "uint16": ("<u2", 2, "typed"),
    "int32": ("<i4", 4, "typed"),
    "uint32": ("<u4", 4, "typed"),
    "int64": ("<i8", 8, "typed"),
    "uint64": ("<u8", 8, "typed"),
    "float16": ("<f2", 2, "typed"),
    "fp16": ("<f2", 2, "typed"),
    "float32": ("<f4", 4, "typed"),
    "fp32": ("<f4", 4, "typed"),
    "float64": ("<f8", 8, "typed"),
    "fp64": ("<f8", 8, "typed"),
    "bfloat16": ("<u2", 2, "raw_bits"),
    "bf16": ("<u2", 2, "raw_bits"),
    "float8e4m3fn": ("uint8", 1, "raw_bits"),
    "fp8e4m3": ("uint8", 1, "raw_bits"),
    "float8e5m2": ("uint8", 1, "raw_bits"),
    "fp8e5m2": ("uint8", 1, "raw_bits"),
    "float8e8m0": ("uint8", 1, "raw_bits"),
    "e8m0": ("uint8", 1, "raw_bits"),
}

# OutputDataType values from the DumpData protobuf schema.  Sub-byte and other
# NumPy-unsupported formats are intentionally retained as raw uint8 bits.
DATA_TYPE_NAMES = {
    0: "undefined", 1: "float32", 2: "float16", 3: "int8", 4: "uint8",
    5: "int16", 6: "uint16", 7: "int32", 8: "int64", 9: "uint32",
    10: "uint64", 11: "bool", 12: "float64", 13: "string",
    14: "dual_sub_int8", 15: "dual_sub_uint8", 16: "complex64",
    17: "complex128", 18: "qint8", 19: "qint16", 20: "qint32",
    21: "quint8", 22: "quint16", 23: "resource", 24: "string_ref",
    25: "dual", 26: "variant", 27: "bfloat16", 28: "int4",
    29: "uint1", 30: "int2", 31: "uint2", 32: "complex32",
    33: "hifloat8", 34: "float8_e5m2", 35: "float8_e4m3fn",
    36: "float8_e8m0", 37: "float6_e3m2", 38: "float6_e2m3",
    39: "float4_e2m1", 40: "float4_e1m2",
}

SIDECAR_DTYPE_ALIASES = {
    "dtfloat": "float32", "dtfloat16": "float16", "dtint8": "int8",
    "dtuint8": "uint8", "dtint16": "int16", "dtuint16": "uint16",
    "dtint32": "int32", "dtint64": "int64", "dtuint32": "uint32",
    "dtuint64": "uint64", "dtbool": "bool", "dtdouble": "float64",
    "dtbf16": "bfloat16", "dtfloat8e5m2": "float8_e5m2",
    "dtfloat8e4m3fn": "float8_e4m3fn", "dtfloat8e8m0": "float8_e8m0",
    "dtfloat4e2m1": "float4_e2m1",
}

FORMAT_NAMES = {
    0: "NCHW", 1: "NHWC", 2: "ND", 3: "NC1HWC0", 4: "FRACTAL_Z",
    29: "FRACTAL_NZ", 30: "NCDHW", 34: "CN", 35: "NC", 42: "NULL",
    47: "NCL", 255: "MAX",
}

PAYLOAD_ORDER = ("input", "output", "buffer", "space")
RAW_PREVIEW_COUNT = 8


def _read_varint(data, offset, limit=None):
    """Read one unsigned protobuf varint and return ``(value, next_offset)``."""
    end = len(data) if limit is None else limit
    value = 0
    shift = 0
    while offset < end and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("protobuf varint 截断或超过 64 位")


def _protobuf_fields(data):
    """Yield ``(field_number, wire_type, value)`` for one protobuf message."""
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x7
        if field_number == 0:
            raise ValueError("protobuf 字段号不能为 0")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            if offset + 8 > len(data):
                raise ValueError("protobuf fixed64 字段截断")
            value = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
        elif wire_type == 2:
            size, offset = _read_varint(data, offset)
            end = offset + size
            if end > len(data):
                raise ValueError("protobuf length-delimited 字段截断")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            if offset + 4 > len(data):
                raise ValueError("protobuf fixed32 字段截断")
            value = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        else:
            raise ValueError(f"不支持的 protobuf wire type: {wire_type}")
        yield field_number, wire_type, value


def _text(value):
    return bytes(value).decode("utf-8", errors="replace")


def _signed_int32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def _parse_shape(data):
    dims = []
    for field, wire_type, value in _protobuf_fields(data):
        if field != 1:
            continue
        if wire_type == 0:
            dims.append(int(value))
        elif wire_type == 2:
            offset = 0
            while offset < len(value):
                dim, offset = _read_varint(value, offset)
                dims.append(int(dim))
    return {"dim": dims}


def _parse_dim_range(data):
    result = {"dim_start": 0, "dim_end": 0}
    for field, wire_type, value in _protobuf_fields(data):
        if wire_type != 0:
            continue
        if field == 1:
            result["dim_start"] = int(value)
        elif field == 2:
            result["dim_end"] = int(value)
    return result


def _parse_original_op(data):
    result = {"name": "", "output_index": 0, "data_type": 0, "format": 0}
    for field, wire_type, value in _protobuf_fields(data):
        if field == 1 and wire_type == 2:
            result["name"] = _text(value)
        elif wire_type == 0 and field in (2, 3, 4):
            result[{2: "output_index", 3: "data_type", 4: "format"}[field]] = int(value)
    return result


def _parse_output(data):
    result = {
        "data_type": 0, "format": 0, "shape": {"dim": []}, "size": 0,
        "original_shape": {"dim": []}, "sub_format": 0, "address": 0,
        "offset": 0, "dim_range": [], "embedded_data_size": 0,
    }
    for field, wire_type, value in _protobuf_fields(data):
        if wire_type == 0:
            if field in (1, 2, 6, 9, 11):
                result[{1: "data_type", 2: "format", 6: "size", 9: "address", 11: "offset"}[field]] = int(value)
            elif field == 8:
                result["sub_format"] = _signed_int32(value)
        elif wire_type == 2:
            if field == 3:
                result["shape"] = _parse_shape(value)
            elif field == 4:
                result["original_op"] = _parse_original_op(value)
            elif field == 5:
                result["embedded_data_size"] = len(value)
            elif field == 7:
                result["original_shape"] = _parse_shape(value)
            elif field == 10:
                result["dim_range"].append(_parse_dim_range(value))
    return result


def _parse_input(data):
    result = {
        "data_type": 0, "format": 0, "shape": {"dim": []}, "size": 0,
        "original_shape": {"dim": []}, "sub_format": 0, "address": 0,
        "offset": 0, "arg_index": 0, "input_type": 0,
        "embedded_data_size": 0,
    }
    for field, wire_type, value in _protobuf_fields(data):
        if wire_type == 0:
            if field in (1, 2, 5, 8, 9, 10, 11):
                result[{1: "data_type", 2: "format", 5: "size", 8: "address",
                        9: "offset", 10: "arg_index", 11: "input_type"}[field]] = int(value)
            elif field == 7:
                result["sub_format"] = _signed_int32(value)
        elif wire_type == 2:
            if field == 3:
                result["shape"] = _parse_shape(value)
            elif field == 4:
                result["embedded_data_size"] = len(value)
            elif field == 6:
                result["original_shape"] = _parse_shape(value)
    return result


def _parse_buffer(data):
    result = {"buffer_type": 0, "size": 0, "embedded_data_size": 0}
    for field, wire_type, value in _protobuf_fields(data):
        if field == 1 and wire_type == 0:
            result["buffer_type"] = int(value)
        elif field == 2 and wire_type == 2:
            result["embedded_data_size"] = len(value)
        elif field == 3 and wire_type == 0:
            result["size"] = int(value)
    return result


def _parse_workspace(data):
    result = {"type": 0, "size": 0, "embedded_data_size": 0}
    for field, wire_type, value in _protobuf_fields(data):
        if field == 1 and wire_type == 0:
            result["type"] = int(value)
        elif field == 2 and wire_type == 2:
            result["embedded_data_size"] = len(value)
        elif field == 3 and wire_type == 0:
            result["size"] = int(value)
    return result


def _parse_attr(data):
    result = {"name": "", "value": ""}
    for field, wire_type, value in _protobuf_fields(data):
        if field == 1 and wire_type == 2:
            result["name"] = _text(value)
        elif field == 2 and wire_type == 2:
            result["value"] = _text(value)
    return result


def parse_dump_data_header(data):
    """Parse the ``DumpData`` protobuf header without generated protobuf code."""
    result = {
        "version": "", "dump_time": 0, "output": [], "input": [],
        "buffer": [], "op_name": "", "attr": [], "space": [],
        "dfx_message": "",
    }
    for field, wire_type, value in _protobuf_fields(data):
        if field == 1 and wire_type == 2:
            result["version"] = _text(value)
        elif field == 2 and wire_type == 0:
            result["dump_time"] = int(value)
        elif field == 3 and wire_type == 2:
            result["output"].append(_parse_output(value))
        elif field == 4 and wire_type == 2:
            result["input"].append(_parse_input(value))
        elif field == 5 and wire_type == 2:
            result["buffer"].append(_parse_buffer(value))
        elif field == 6 and wire_type == 2:
            result["op_name"] = _text(value)
        elif field == 7 and wire_type == 2:
            result["attr"].append(_parse_attr(value))
        elif field == 8 and wire_type == 2:
            result["space"].append(_parse_workspace(value))
        elif field == 9 and wire_type == 2:
            result["dfx_message"] = _text(value)
    return result


def normalize_dtype(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def tensor_identity(path):
    match = TENSOR_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1).lower(), int(match.group(2))


def infer_dtype(path):
    """Infer dtype only when the filename explicitly carries a known token."""
    tokens = re.split(r"[._-]+", path.name)
    for token in reversed(tokens):
        normalized = normalize_dtype(token)
        if normalized in DTYPE_SPECS:
            storage_dtype, item_size, interpretation = DTYPE_SPECS[normalized]
            return token, storage_dtype, item_size, interpretation
    normalized_name = normalize_dtype(path.name)
    for logical_dtype in sorted(DTYPE_SPECS, key=len, reverse=True):
        if logical_dtype in normalized_name:
            storage_dtype, item_size, interpretation = DTYPE_SPECS[logical_dtype]
            return logical_dtype, storage_dtype, item_size, interpretation
    return "unknown", "uint8", 1, "raw_bytes"


def unique_output_path(export_root, relative_path):
    output_path = export_root / relative_path
    if output_path.suffix.lower() in (".bin", ".npy"):
        output_path = output_path.with_suffix(".npy")
    else:
        output_path = output_path.with_name(output_path.name + ".npy")
    return output_path


def export_binary(source, output, storage_dtype, item_size):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("导出 NPY 需要 NumPy，请先安装 numpy") from error

    byte_size = source.stat().st_size
    if byte_size % item_size != 0:
        raise ValueError(
            f"文件大小 {byte_size} 不是 dtype 元素大小 {item_size} 的整数倍: {source}"
        )
    element_count = byte_size // item_size
    if element_count == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.empty((0,), dtype=np.dtype(storage_dtype)), allow_pickle=False)
        return 0
    source_array = np.memmap(source, dtype=np.dtype(storage_dtype), mode="r", shape=(element_count,))
    output.parent.mkdir(parents=True, exist_ok=True)
    target_array = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.dtype(storage_dtype), shape=(element_count,)
    )
    # Copy in bounded chunks so a large exception tensor is not loaded into RAM.
    chunk_elements = max(1, (16 * 1024 * 1024) // item_size)
    for begin in range(0, element_count, chunk_elements):
        end = min(begin + chunk_elements, element_count)
        target_array[begin:end] = source_array[begin:end]
    target_array.flush()
    del target_array
    del source_array
    return element_count


def _collect_json_dtypes(value, result):
    """Collect ``index``/``dtype`` pairs from nested operator JSON arrays."""
    if isinstance(value, dict):
        if "index" in value and "dtype" in value:
            try:
                result[int(value["index"])] = str(value["dtype"])
            except (TypeError, ValueError):
                pass
        for child in value.values():
            _collect_json_dtypes(child, result)
    elif isinstance(value, list):
        for child in value:
            _collect_json_dtypes(child, result)


def load_sidecar_dtypes(source):
    """Read input/output dtype fallbacks from the operator JSON beside a dump."""
    result = {"input": {}, "output": {}}
    candidates = sorted(
        (path for path in source.parent.glob("*.json") if path.name != source.name + ".json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        support = data.get("supportInfo", {}) if isinstance(data, dict) else {}
        local = {"input": {}, "output": {}}
        _collect_json_dtypes(support.get("inputs", []), local["input"])
        _collect_json_dtypes(support.get("outputs", []), local["output"])
        if local["input"] or local["output"]:
            return local
    return result


def dtype_spec(data_type, role, index, sidecar_dtypes):
    """Return logical dtype, storage dtype, item size and interpretation."""
    if role not in ("input", "output"):
        return "raw_bytes", "uint8", 1, "raw_bytes"
    logical_dtype = DATA_TYPE_NAMES.get(data_type, f"dtype_{data_type}")
    if logical_dtype == "undefined":
        logical_dtype = sidecar_dtypes.get(role, {}).get(index, logical_dtype)
    normalized = normalize_dtype(str(logical_dtype))
    logical_dtype = SIDECAR_DTYPE_ALIASES.get(normalized, logical_dtype)
    normalized = normalize_dtype(str(logical_dtype))
    spec = DTYPE_SPECS.get(normalized)
    if spec is None:
        return str(logical_dtype), "uint8", 1, "raw_bits"
    storage_dtype, item_size, interpretation = spec
    return str(logical_dtype), storage_dtype, item_size, interpretation


def safe_component(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return value or "unknown"


def export_binary_region(source, source_offset, byte_size, output, storage_dtype, item_size, shape):
    """Copy one payload region into a typed NPY without loading it all in RAM."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("导出 NPY 需要 NumPy，请先安装 numpy") from error

    if byte_size % item_size != 0:
        storage_dtype, item_size = "uint8", 1
        shape = []
    dtype = np.dtype(storage_dtype)
    element_count = byte_size // item_size
    output_shape = tuple(shape) if shape and math.prod(shape) == element_count else (element_count,)
    output.parent.mkdir(parents=True, exist_ok=True)
    if element_count == 0:
        np.save(output, np.empty(output_shape, dtype=dtype), allow_pickle=False)
        return str(dtype), output_shape, element_count
    target = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=output_shape)
    target_flat = target.reshape(-1)
    chunk_bytes = max(item_size, (16 * 1024 * 1024) // item_size * item_size)
    written = 0
    with source.open("rb") as source_file:
        source_file.seek(source_offset)
        remaining = byte_size
        while remaining:
            data = source_file.read(min(chunk_bytes, remaining))
            if not data:
                del target
                raise ValueError(f"payload 在 offset={source_offset + written * item_size} 处提前结束")
            values = np.frombuffer(data, dtype=dtype)
            target_flat[written:written + values.size] = values
            written += values.size
            remaining -= len(data)
    target.flush()
    del target
    return str(dtype), output_shape, element_count


def payload_preview(source, source_offset, byte_size, storage_dtype, item_size):
    """Return a small decimal value preview for terminal display."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("显示异常数据预览需要 NumPy，请先安装 numpy") from error

    readable_size = byte_size - byte_size % item_size
    count = min(RAW_PREVIEW_COUNT, readable_size // item_size)
    if count == 0:
        return []
    with source.open("rb") as source_file:
        source_file.seek(source_offset)
        data = source_file.read(count * item_size)
    values = np.frombuffer(data, dtype=np.dtype(storage_dtype))
    return values.tolist()


def export_exception_info(source, export_dir, operator="unknown"):
    """Parse one raw ``exception_info.*`` container using only repository code."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("解析 exception_info 并导出 NPY 需要 NumPy，请先安装 numpy") from error

    file_size = source.stat().st_size
    if file_size <= 8:
        raise ValueError(f"exception_info 文件过小: {file_size} bytes")
    with source.open("rb") as source_file:
        length_data = source_file.read(8)
        header_length = struct.unpack("<Q", length_data)[0]
        if header_length > file_size - 8:
            raise ValueError(
                f"无效 protobuf headerLength={header_length}，文件仅 {file_size} bytes"
            )
        header_data = source_file.read(header_length)
    header = parse_dump_data_header(header_data)
    sidecar_dtypes = load_sidecar_dtypes(source)
    export_dir.mkdir(parents=True, exist_ok=True)

    records = []
    lines = [
        "parser: repository built-in DumpData protobuf decoder (no CANN/msaicerr)",
        f"container: fileSize={file_size} headerLength={header_length} payloadOffset={8 + header_length}",
        f"header: opName={header['op_name'] or operator} version={header['version'] or '-'} "
        f"dumpTime={header['dump_time']} dfxMessage={header['dfx_message'] or '-'}",
    ]
    payload_offset = 8 + header_length
    for header_role in PAYLOAD_ORDER:
        output_role = "workspace" if header_role == "space" else header_role
        for index, item in enumerate(header[header_role]):
            byte_size = int(item.get("size", 0))
            if byte_size < 0 or payload_offset + byte_size > file_size:
                raise ValueError(
                    f"{output_role}[{index}] 越界: offset={payload_offset}, "
                    f"size={byte_size}, fileSize={file_size}"
                )
            logical_dtype, storage_dtype, item_size, interpretation = dtype_spec(
                int(item.get("data_type", 0)), output_role, index, sidecar_dtypes
            )
            if byte_size % item_size != 0:
                storage_dtype, item_size, interpretation = "uint8", 1, "raw_bytes_size_mismatch"
            logical_shape = [int(value) for value in item.get("shape", {}).get("dim", [])]
            output_name = (
                f"{output_role}.{index}.{safe_component(logical_dtype)}.npy"
            )
            output_path = export_dir / output_name
            numpy_dtype, output_shape, element_count = export_binary_region(
                source, payload_offset, byte_size, output_path,
                storage_dtype, item_size, logical_shape,
            )
            preview = payload_preview(source, payload_offset, byte_size, storage_dtype, item_size)
            format_value = item.get("format")
            format_name = FORMAT_NAMES.get(format_value, str(format_value)) if format_value is not None else "-"
            record = {
                "operator": header["op_name"] or operator,
                "role": output_role,
                "index": index,
                "source": str(source),
                "payload_offset": payload_offset,
                "byte_size": byte_size,
                "data_type": item.get("data_type"),
                "logical_dtype": logical_dtype,
                "numpy_dtype": numpy_dtype,
                "logical_shape": logical_shape,
                "npy_shape": list(output_shape),
                "element_count": element_count,
                "format": format_value,
                "format_name": format_name,
                "address": item.get("address"),
                "arg_index": item.get("arg_index"),
                "input_type": item.get("input_type"),
                "interpretation": interpretation,
                "output": str(output_path),
            }
            records.append(record)
            lines.append(
                f"{output_role}[{index}]: dtype={logical_dtype}/{numpy_dtype} "
                f"shape={logical_shape or '-'} npyShape={list(output_shape)} "
                f"format={format_name} size={byte_size} address={item.get('address', '-')} "
                f"preview={preview} -> {output_name}"
            )
            payload_offset += byte_size

    trailing_size = file_size - payload_offset
    trailing_path = None
    if trailing_size:
        trailing_path = export_dir / "trailing_bytes.npy"
        export_binary_region(
            source, payload_offset, trailing_size, trailing_path, "uint8", 1, []
        )
        lines.append(
            f"WARNING: header 未描述尾部 {trailing_size} bytes，已无损保存 -> {trailing_path.name}"
        )

    metadata = {
        "operator": header["op_name"] or operator,
        "source": str(source),
        "file_size": file_size,
        "header_length": header_length,
        "payload_offset": 8 + header_length,
        "payload_end": payload_offset,
        "trailing_size": trailing_size,
        "trailing_output": str(trailing_path) if trailing_path else None,
        "header": header,
        "tensors": records,
        "notes": (
            "每个 input/output/buffer/workspace payload 已保存为独立 NPY；"
            "raw_bits/raw_bytes 表示只保留位模式，没有猜测不可确定的数据类型。"
        ),
    }
    metadata_path = export_dir / "exception_metadata.npy"
    np.save(metadata_path, metadata, allow_pickle=True)
    lines.append(
        f"metadata: tensors={len(records)} consumed={payload_offset}/{file_size} -> {metadata_path.name}"
    )
    return records, metadata_path, lines


def export_existing_npy(source, output):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("校验 NPY 需要 NumPy，请先安装 numpy") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != output.resolve():
        shutil.copy2(source, output)
    array = np.load(output, mmap_mode="r", allow_pickle=False)
    return list(array.shape), str(array.dtype), int(array.size)


def discover_tensor_files(report_dir, export_dir):
    result = {}
    for path in report_dir.rglob("*"):
        if not path.is_file() or export_dir in path.parents:
            continue
        if path.suffix.lower() not in (".bin", ".npy"):
            continue
        if tensor_identity(path) is not None:
            relative_path = path.relative_to(report_dir)
            key = str(relative_path.with_suffix("")).lower()
            previous = result.get(key)
            if previous is None or path.suffix.lower() == ".npy":
                result[key] = path
    return sorted(result.values(), key=lambda item: str(item).lower())


def export_report(report_dir, export_dir, operator):
    files = discover_tensor_files(report_dir, export_dir)
    if not files:
        raise ValueError(f"在 {report_dir} 中没有找到 input/output/workspace 张量文件")

    records = []
    for source in files:
        identity = tensor_identity(source)
        role, index = identity
        relative_path = source.relative_to(report_dir)
        output = unique_output_path(export_dir, relative_path)
        if source.suffix.lower() == ".npy":
            shape, numpy_dtype, element_count = export_existing_npy(source, output)
            logical_dtype = numpy_dtype
            interpretation = "existing_npy"
        else:
            logical_dtype, storage_dtype, item_size, interpretation = infer_dtype(source)
            element_count = export_binary(source, output, storage_dtype, item_size)
            numpy_dtype = storage_dtype
            shape = [element_count]

        record = {
            "operator": operator,
            "role": role,
            "index": index,
            "source": str(source),
            "output": str(output),
            "logical_dtype": logical_dtype,
            "numpy_dtype": numpy_dtype,
            "shape": shape,
            "element_count": element_count,
            "byte_size": source.stat().st_size,
            "interpretation": interpretation,
        }
        records.append(record)
        print(
            f"[NPY] {role}[{index}] dtype={logical_dtype}/{numpy_dtype} "
            f"shape={shape} -> {output}"
        )

    manifest = {
        "operator": operator,
        "source_dir": str(report_dir),
        "export_dir": str(export_dir),
        "tensor_count": len(records),
        "notes": (
            "shape 为一维存储形状；真实逻辑 shape 需结合算子参数/TilingData。"
            "interpretation=raw_bytes/raw_bits 表示未做数值类型猜测，NPY 保存原始位模式。"
        ),
        "tensors": records,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] 导出完成: {len(records)} 个 NPY")
    print(f"[INFO] manifest: {manifest_path}")
    return records, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="直接解析 exception_info.*，或转换已解包目录中的异常张量"
    )
    parser.add_argument("input_path", help="原始 exception_info.* 文件或已解包张量目录")
    parser.add_argument("--operator", default="unknown", help="算子名，仅用于 manifest 标注")
    parser.add_argument("--export-npy", required=True, help="NPY 输出目录")
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser().resolve()
    export_dir = Path(args.export_npy).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"输入不存在: {input_path}")
    if input_path == export_dir:
        parser.error("--export-npy 不能与异常解析输入目录相同")

    try:
        if input_path.is_file():
            _, _, lines = export_exception_info(input_path, export_dir, args.operator)
            for line in lines:
                print(f"[INFO] {line}")
        else:
            export_report(input_path, export_dir, args.operator)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

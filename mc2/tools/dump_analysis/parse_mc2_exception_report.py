#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export tensor files produced by msaicerr to NumPy NPY files.

The msaicerr output filename identifies the tensor role and index, for example
``exception_info....input.0.float16.bin``.  When a supported dtype is present
in the filename it is retained.  If the dtype is absent or ``undefined``, the
payload is exported losslessly as uint8 instead of guessing a tensor type.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
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
    parser = argparse.ArgumentParser(description="将 msaicerr 异常报告中的张量导出为 NPY")
    parser.add_argument("report_dir", help="msaicerr 异常解析输出目录")
    parser.add_argument("--operator", default="unknown", help="算子名，仅用于 manifest 标注")
    parser.add_argument("--export-npy", required=True, help="NPY 输出目录")
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir).expanduser().resolve()
    export_dir = Path(args.export_npy).expanduser().resolve()
    if not report_dir.is_dir():
        parser.error(f"异常解析目录不存在: {report_dir}")
    if report_dir == export_dir:
        parser.error("--export-npy 不能与异常解析输入目录相同")

    try:
        export_report(report_dir, export_dir, args.operator)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

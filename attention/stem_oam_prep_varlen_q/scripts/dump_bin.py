#!/usr/bin/env python3
"""
Dump binary files to human-readable text.

Usage:
    python dump_bin.py                    # dump all default files to dump.txt
    python dump_bin.py -o result.txt      # dump to custom file
    python dump_bin.py path/to/file.bin -s 128 1 128 -d fp8  # dump single file
"""

import argparse
import sys
import os
import numpy as np
import torch

DTYPE_MAP = {
    "fp32":    (np.float32,  None,               None),
    "fp16":    (np.float16,  None,               None),
    "bf16":    (np.uint16,   torch.bfloat16,     "bfloat16"),
    "fp8":     (np.uint8,    torch.float8_e4m3fn, "float8_e4m3fn"),
    "int32":   (np.int32,    None,               None),
    "int64":   (np.int64,    None,               None),
    "uint8":   (np.uint8,    None,               None),
}

DEFAULT_FILES = [
    ("input/input_q.bin",            [128, 1, 128],    "fp8"),
    ("input/input_qscale.bin",       [1, 1, 128],      "fp32"),
    ("input/input_q_seq_lens.bin",   [1],              "int32"),
    ("input/input_cu_seqlens_q.bin", [2],              "int32"),
    ("output/golden.bin",            [1, 1, 1, 2048],  "bf16"),
]


def load_bin(path, shape, dtype_key):
    np_dtype, torch_dtype, torch_name = DTYPE_MAP[dtype_key]
    raw = np.fromfile(path, dtype=np_dtype)
    if torch_dtype is not None:
        tensor = torch.from_numpy(raw).view(torch_dtype).reshape(shape)
        return tensor
    else:
        return raw.reshape(shape)


def dump_file(path, shape, dtype_key, f):
    if not os.path.exists(path):
        f.write(f"[SKIP] {path} not found\n\n")
        return

    data = load_bin(path, shape, dtype_key)
    file_size = os.path.getsize(path)

    f.write(f"{'='*60}\n")
    f.write(f"File:  {path}\n")
    f.write(f"Shape: {shape}\n")
    f.write(f"Dtype: {dtype_key}\n")
    f.write(f"Size:  {file_size} bytes\n")
    f.write(f"{'='*60}\n")

    if isinstance(data, torch.Tensor):
        data_np = data.float().numpy()
    else:
        data_np = data.astype(np.float64)

    flat = data_np.flatten()
    total = len(flat)

    if total <= 256:
        for i in range(total):
            f.write(f"  [{i:6d}] {flat[i]:.8f}\n")
    else:
        for i in range(64):
            f.write(f"  [{i:6d}] {flat[i]:.8f}\n")
        f.write(f"  ... ({total - 128} elements omitted) ...\n")
        for i in range(total - 64, total):
            f.write(f"  [{i:6d}] {flat[i]:.8f}\n")

    f.write(f"\nStats: min={flat.min():.8f}, max={flat.max():.8f}, "
            f"mean={flat.mean():.8f}, std={flat.std():.8f}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Dump binary files to text")
    parser.add_argument("file", nargs="?", help="Single file to dump")
    parser.add_argument("-s", "--shape", nargs="+", type=int, help="Shape (space-separated)")
    parser.add_argument("-d", "--dtype", default="fp32", choices=DTYPE_MAP.keys(), help="Data type")
    parser.add_argument("-o", "--output", default="dump.txt", help="Output text file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    with open(args.output, "w") as f:
        if args.file:
            shape = args.shape if args.shape else [-1]
            dump_file(args.file, shape, args.dtype, f)
        else:
            for path, shape, dtype_key in DEFAULT_FILES:
                dump_file(path, shape, dtype_key, f)

    print(f"Dumped to {args.output}")


if __name__ == "__main__":
    main()

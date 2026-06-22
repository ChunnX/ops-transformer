#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskKComp Self-Test: Data Generation Script

Generates test input data according to aclnnBSASelectBlockMask interface constraints.
Accepts parameters from bash script, saves data as .bin files for NPU execution.

Parameters (from bash):
  keys = ["B", "N", "S1", "S2", "D", "bx", "by", "layout", "sparsity",
          "use_seqlen", "use_q_block_len", "use_k_block_len"]

Output files:
  input_query.bin        - FP16 tensor raw binary
  input_key.bin          - FP16 tensor raw binary
  input_block_shape.bin  - INT64 array [bx, by]
  input_seqlen_q.bin     - INT32 array (only if use_seqlen=1)
  input_seqlen_kv.bin    - INT32 array (only if use_seqlen=1)
  input_block_len_q.bin  - INT32 array (only if use_q_block_len=1)
  input_block_len_k.bin  - INT32 array (only if use_k_block_len=1)
  input_params.txt       - key=value metadata for C++ parsing
"""

import os
import shutil
import argparse
import torch
import numpy as np


def prepare_dir(dir_path):
    """Create dir if not exists, clear contents if exists"""
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            fpath = os.path.join(dir_path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
    else:
        os.makedirs(dir_path)

seed = 42
torch.manual_seed(seed)
import random
random.seed(seed)

DTYPE = torch.float16


def generate_seqlen_tnd(max_seqlen_q, max_seqlen_kv, batch_size):
    """Generate random seqlen for TND layout mode"""
    random.seed(seed)
    torch.manual_seed(seed)

    seqlen_q_list = [random.randint(1, max_seqlen_q) for _ in range(batch_size)]
    seqlen_kv_list = [random.randint(1, max_seqlen_kv) for _ in range(batch_size)]

    # Ensure at least one sample reaches max
    max_idx = max(range(batch_size), key=lambda i: seqlen_q_list[i])
    seqlen_q_list[max_idx] = max_seqlen_q
    max_idx = max(range(batch_size), key=lambda i: seqlen_kv_list[i])
    seqlen_kv_list[max_idx] = max_seqlen_kv

    total_seqlen_q = sum(seqlen_q_list)
    total_seqlen_kv = sum(seqlen_kv_list)

    return seqlen_q_list, seqlen_kv_list, total_seqlen_q, total_seqlen_kv


def generate_data(params):
    """Generate input tensors based on parameters"""
    random.seed(seed)
    torch.manual_seed(seed)

    layout = params["layout"]
    B = params["B"]
    N = params["N"]
    S1 = params["S1"]
    S2 = params["S2"]
    D = params["D"]
    bx = params["bx"]
    by = params["by"]

    q = None
    k = None
    q_seqlen_list = None
    k_seqlen_list = None
    q_block_len_list = None
    k_block_len_list = None

    block_x_nums = (S1 + bx - 1) // bx
    block_y_nums = (S2 + by - 1) // by

    if layout == "BNSD":
        q = torch.rand((B, N, S1, D), dtype=DTYPE)
        k = torch.rand((B, N, S2, D), dtype=DTYPE)

        if params["use_seqlen"]:
            new_S2 = random.randint(1, S2)
            q_seqlen_list = [S1] * B
            k_seqlen_list = [new_S2] * B

        if params["use_q_block_len"]:
            q_block_len_list = []
            for b in range(B):
                row = [random.randint(0, bx) for _ in range(block_x_nums)]
                q_block_len_list.append(row)

        if params["use_k_block_len"]:
            k_block_len_list = []
            for b in range(B):
                row = [random.randint(0, by) for _ in range(block_y_nums)]
                k_block_len_list.append(row)
    else:  # TND layout
        seqlen_q_list, seqlen_kv_list, total_seqlen_q, total_seqlen_kv = \
            generate_seqlen_tnd(S1, S2, B)
        q_seqlen_list = seqlen_q_list
        k_seqlen_list = seqlen_kv_list
        q = torch.rand((total_seqlen_q, N, D), dtype=DTYPE)
        k = torch.rand((total_seqlen_kv, N, D), dtype=DTYPE)

        if params["use_q_block_len"]:
            q_block_len_list = [random.randint(0, bx) for _ in range(block_x_nums * B)]

        if params["use_k_block_len"]:
            k_block_len_list = [random.randint(0, by) for _ in range(block_y_nums * B)]

    return q, k, q_seqlen_list, k_seqlen_list, q_block_len_list, k_block_len_list


def save_tensor_bin(tensor, filepath):
    """Save a torch tensor as raw binary file"""
    numpy_data = tensor.detach().cpu().contiguous().numpy()
    numpy_data.tofile(filepath)


def save_int_list_bin(int_list, filepath):
    """Save a Python list of ints as INT32 binary file"""
    arr = np.array(int_list, dtype=np.int32)
    arr.tofile(filepath)


def save_params_txt(params, shapes, output_dir):
    """Save parameters and shape metadata as key=value text file for C++ parsing"""
    filepath = os.path.join(output_dir, "input_params.txt")
    xBlocks = (params["S1"] + params["bx"] - 1) // params["bx"]
    yBlocks = (params["S2"] + params["by"] - 1) // params["by"]

    with open(filepath, "w") as f:
        # Scalar parameters
        f.write(f"B={params['B']}\n")
        f.write(f"N={params['N']}\n")
        f.write(f"S1={params['S1']}\n")
        f.write(f"S2={params['S2']}\n")
        f.write(f"D={params['D']}\n")
        f.write(f"bx={params['bx']}\n")
        f.write(f"by={params['by']}\n")
        f.write(f"layout={params['layout']}\n")
        f.write(f"sparsity={params['sparsity']}\n")
        f.write(f"use_seqlen={int(params['use_seqlen'])}\n")
        f.write(f"use_q_block_len={int(params['use_q_block_len'])}\n")
        f.write(f"use_k_block_len={int(params['use_k_block_len'])}\n")
        f.write(f"numKeyValueHeads={params['N']}\n")
        f.write(f"scaleValue={1.0 / (params['D'] ** 0.5)}\n")
        f.write(f"xBlocks={xBlocks}\n")
        f.write(f"yBlocks={yBlocks}\n")

        # Tensor shapes
        for name, shape in shapes.items():
            shape_str = ",".join(str(s) for s in shape)
            f.write(f"shape_{name}={shape_str}\n")

        # Optional array lengths
        if params["use_seqlen"]:
            f.write(f"seqlen_q_len={params['B']}\n")
            f.write(f"seqlen_kv_len={params['B']}\n")
        if params["use_q_block_len"]:
            if params["layout"] == "BNSD":
                f.write(f"block_len_q_len={params['B'] * xBlocks}\n")
            else:
                f.write(f"block_len_q_len={params['B'] * xBlocks}\n")
        if params["use_k_block_len"]:
            if params["layout"] == "BNSD":
                f.write(f"block_len_k_len={params['B'] * yBlocks}\n")
            else:
                f.write(f"block_len_k_len={params['B'] * yBlocks}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate test input data for MaskKComp")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--S1", type=int, default=128)
    parser.add_argument("--S2", type=int, default=128)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--bx", type=int, default=128)
    parser.add_argument("--by", type=int, default=128)
    parser.add_argument("--layout", type=str, default="BNSD")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--use_seqlen", type=int, default=0)
    parser.add_argument("--use_q_block_len", type=int, default=0)
    parser.add_argument("--use_k_block_len", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=".")

    args = parser.parse_args()

    params = {
        "B": args.B, "N": args.N, "S1": args.S1, "S2": args.S2, "D": args.D,
        "bx": args.bx, "by": args.by, "layout": args.layout,
        "sparsity": args.sparsity,
        "use_seqlen": bool(args.use_seqlen),
        "use_q_block_len": bool(args.use_q_block_len),
        "use_k_block_len": bool(args.use_k_block_len),
    }

    print(f"Parameters: {params}")

    q, k, q_seqlen_list, k_seqlen_list, q_block_len_list, k_block_len_list = \
        generate_data(params)

    output_dir = args.output_dir
    prepare_dir(output_dir)

    # Collect shapes for metadata
    shapes = {}
    shapes["query"] = list(q.shape)
    shapes["key"] = list(k.shape)

    # Save tensors
    save_tensor_bin(q, os.path.join(output_dir, "input_query.bin"))
    save_tensor_bin(k, os.path.join(output_dir, "input_key.bin"))

    block_shape = np.array([params["bx"], params["by"], ], dtype=np.int64)
    block_shape.tofile(os.path.join(output_dir, "input_block_shape.bin"))

    # Save optional arrays
    if q_seqlen_list is not None:
        save_int_list_bin(q_seqlen_list, os.path.join(output_dir, "input_seqlen_q.bin"))
        shapes["seqlen_q"] = [len(q_seqlen_list)]

    if k_seqlen_list is not None:
        save_int_list_bin(k_seqlen_list, os.path.join(output_dir, "input_seqlen_kv.bin"))
        shapes["seqlen_kv"] = [len(k_seqlen_list)]

    if q_block_len_list is not None:
        # Flatten 2D list if BNSD layout
        if params["layout"] == "BNSD":
            flat_list = []
            for row in q_block_len_list:
                flat_list.extend(row)
            save_int_list_bin(flat_list, os.path.join(output_dir, "input_block_len_q.bin"))
        else:
            save_int_list_bin(q_block_len_list, os.path.join(output_dir, "input_block_len_q.bin"))

    if k_block_len_list is not None:
        if params["layout"] == "BNSD":
            flat_list = []
            for row in k_block_len_list:
                flat_list.extend(row)
            save_int_list_bin(flat_list, os.path.join(output_dir, "input_block_len_k.bin"))
        else:
            save_int_list_bin(k_block_len_list, os.path.join(output_dir, "input_block_len_k.bin"))

    # Save params metadata
    save_params_txt(params, shapes, output_dir)

    print(f"Data saved to {output_dir}")
    print(f"  input_query.bin      shape={list(q.shape)} dtype=FP16")
    print(f"  input_key.bin        shape={list(k.shape)} dtype=FP16")
    print(f"  input_block_shape.bin shape=[{params['bx']},{params['by']}] dtype=INT64")
    if q_seqlen_list is not None:
        print(f"  input_seqlen_q.bin   len={len(q_seqlen_list)} dtype=INT32")
    if k_seqlen_list is not None:
        print(f"  input_seqlen_kv.bin  len={len(k_seqlen_list)} dtype=INT32")
    if q_block_len_list is not None:
        print(f"  input_block_len_q.bin dtype=INT32")
    if k_block_len_list is not None:
        print(f"  input_block_len_k.bin dtype=INT32")
    print(f"  input_params.txt     (metadata for C++)")


if __name__ == "__main__":
    main()
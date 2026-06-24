#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskKComp Self-Test: Golden Data Generation Script

Reads input bin files from gen_data.py output, computes golden reference:
  - q_comp (FP32): Q mean-pooling compression per [batch, head, Xblocks, D]
  - k_comp (FP32): K mean-pooling compression per [batch, head, Yblocks, D]
  - score   (FP32): attention score per [batch, head, Xblocks, Yblocks]
  - attn_score (FP16): softmax(score) per [batch, head, Xblocks, Yblocks]
  - mask     (INT8): block sparse mask per [B, N, Xblocks, Yblocks]

Output files:
  golden_q_comp.bin      - FP32 raw binary
  golden_k_comp.bin      - FP32 raw binary
  golden_score.bin       - FP32 raw binary
  golden_attn_score.bin  - FP16 raw binary
  golden_mask.bin        - INT8 raw binary
  golden_meta.txt        - shape/dtype metadata
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


def load_params_txt(input_dir):
    """Load key=value parameters from input_params.txt"""
    filepath = os.path.join(input_dir, "input_params.txt")
    params = {}
    shapes = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("shape_"):
                    name = key[len("shape_"):]
                    shapes[name] = [int(x) for x in value.split(",")]
                elif key in ("B", "N", "S1", "S2", "D", "bx", "by",
                             "use_seqlen", "use_q_block_len", "use_k_block_len",
                             "xBlocks", "yBlocks", "numKeyValueHeads",
                             "seqlen_q_len", "seqlen_kv_len",
                             "block_len_q_len", "block_len_k_len"):
                    params[key] = int(value)
                elif key in ("sparsity", "scaleValue"):
                    params[key] = float(value)
                elif key == "layout":
                    params[key] = value
    return params, shapes


def load_tensor_bin(filepath, shape, dtype):
    """Load a tensor from raw binary file"""
    np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }[dtype]
    data = np.fromfile(filepath, dtype=np_dtype)
    expected_size = 1
    for s in shape:
        expected_size *= s
    data = data.reshape(shape)
    return torch.from_numpy(data.copy())


def load_int_list_bin(filepath, length):
    """Load an int list from INT32 binary file"""
    data = np.fromfile(filepath, dtype=np.int32)
    return data.tolist()


FLOAT_BLOCK = 8
FP16_BLOCK = 16
NEG_INF = -65504.0

Q_CHUNK = 25
K_CHUNK = 128

def align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align

def split_q_rows(q_start: int, q_size: int, vec_idx: int):
    if vec_idx == 0:
        return q_start, q_size // 2 + q_size % 2
    return q_start + q_size // 2 + q_size % 2, q_size // 2


def kernel_like_single_vec(score: torch.Tensor) -> torch.Tensor:
    X, Y = score.shape
    out = torch.zeros((X, Y), dtype=torch.float16)

    for q_start in range(0, X, Q_CHUNK):
        q_size = min(Q_CHUNK, X - q_start)
        row_start, row_count = q_start, q_size
        if row_count == 0:
            continue

        global_max = torch.full((row_count,), float('-inf'), dtype=torch.float32)
        global_sum = torch.zeros((row_count,), dtype=torch.float32)

        for k_start in range(0, Y, K_CHUNK):
            k_size = min(K_CHUNK, Y - k_start)
            k_align = align_up(k_size, FLOAT_BLOCK)
            chunk = score[row_start:row_start + row_count, k_start:k_start + k_size].float()
            # if k_align > k_size:
            #     pad = torch.full((row_count, k_align - k_size), NEG_INF, dtype=torch.float32)
            #     chunk = torch.cat([chunk, pad], dim=1)

            # print("第一轮zzzzzzzzzzzzzz")

            local_max = chunk.max(dim=1).values
            new_max = torch.maximum(global_max, local_max)

            # print("score * scale")
            # print(chunk)
            # print("local_max")
            # print(local_max)

            t1 = global_max - new_max
            t2 = torch.exp(t1)
            t3 = global_sum * t2

            # print("t1 diff (global_max - new_max)")
            # print(t1)

            # print("exp(diff))")
            # print(t2)

            # print("global_sum * t2")
            # print(t3)

            t4 = chunk - new_max.unsqueeze(1)
            t5 = torch.exp(t4)
            t6 = t5.sum(dim=1)

            # print("score - max")
            # print(t4)

            # print("exp(score - max)")
            # print(t5)

            # print("exp(score - max).sum(dim=-1)")
            # print(t6)

            global_sum = t3 + t6

            # global_sum = global_sum * torch.exp(global_max - new_max) + torch.exp(chunk - new_max.unsqueeze(1)).sum(dim=1)
            global_max = new_max

            # print("global_max")
            # print(global_max)
            # print("global_sum")
            # print(global_sum)

        for k_start in range(0, Y, K_CHUNK):
            k_size = min(K_CHUNK, Y - k_start)
            k_align = max(align_up(k_size, FLOAT_BLOCK), align_up(k_size, FP16_BLOCK))
            chunk = score[row_start:row_start + row_count, k_start:k_start + k_size].float()
            # if k_align > k_size:
            #     pad = torch.full((row_count, k_align - k_size), NEG_INF, dtype=torch.float32)
            #     chunk = torch.cat([chunk, pad], dim=1)

            t1 = chunk - global_max.unsqueeze(1)

            # print("第二轮zzzzzzzzzzzzzzzzzzzzzzzz")
            # print("score - globl_max")
            # print(t1)

            t2 = torch.exp(t1)

            # print("exp(score - globl_max)")
            # print(t2)

            prob = t2 / global_sum.unsqueeze(1)

            # print("t2 / global_sum")
            # print(prob)

            # prob = torch.exp(chunk - global_max.unsqueeze(1)) / global_sum.unsqueeze(1)
            out[row_start:row_start + row_count, k_start:k_start + k_size] = prob[:, :k_size].half()

    return out


def kernel_like_softmax(score: torch.Tensor) -> torch.Tensor:
    return kernel_like_single_vec(score)

def verify_mask_k_comp(query, key, block_shape_x, block_shape_y, sparsity, scale,
                       actual_block_len_query=None, actual_block_len_key=None):
    """
    CPU reference implementation of MaskKComp.
    Returns: q_comp, k_comp, score, attn_score, mask
    """
    B, N, Sq, D = query.shape
    ori_dtype = query.dtype
    print("ori_dtype")
    print(ori_dtype)
    Skv = key.shape[2]

    X = (Sq + block_shape_x - 1) // block_shape_x
    Y = (Skv + block_shape_y - 1) // block_shape_y

    # Output tensors
    q_comp_all = torch.zeros(B, N, X, D, dtype=torch.float32)
    k_comp_all = torch.zeros(B, N, Y, D, dtype=torch.float32)
    score_all = torch.zeros(B, N, X, Y, dtype=torch.float32)
    attn_score_all = torch.zeros(B, N, X, Y, dtype=torch.float16)
    mask = torch.zeros(B, N, X, Y, dtype=torch.uint8)

    for b in range(B):
        for n in range(N):
            # Step1: Mean Pooling
            q_cmp = torch.zeros(X, D, dtype=torch.float32)
            k_cmp = torch.zeros(Y, D, dtype=torch.float32)

            for x in range(X):
                start = x * block_shape_x
                if actual_block_len_query is not None:
                    actual_len = actual_block_len_query[b][x]
                else:
                    actual_len = min(block_shape_x, Sq - start)
                if actual_len > 0:
                    end = min(start + actual_len, Sq)
                    q_cmp[x] = query[b, n, start:end].float().mean(dim=0)

            for y in range(Y):
                start = y * block_shape_y
                if actual_block_len_key is not None:
                    actual_len = actual_block_len_key[b][y]
                else:
                    actual_len = min(block_shape_y, Skv - start)
                if actual_len > 0:
                    # actual_len = 200
                    end = min(start + actual_len, Skv)
                    k_cmp[y] = key[b, n, start:end].float().mean(dim=0)

            q_comp_all[b, n] = q_cmp
            k_comp_all[b, n] = k_cmp

            # Step2: Matmul + Softmax (FP32)
            score =  (q_cmp.to(ori_dtype).to(torch.float32) @ k_cmp.to(ori_dtype).to(torch.float32).T)
            score_all[b, n] = score

            score = scale * score
            # attn_score = torch.softmax(score, dim=-1)
            attn_score = kernel_like_softmax(score)

            # Store per-batch/head results
            # score_all[b, n] = score
            attn_score_all[b, n] = attn_score.half()

            # # Step3: TopK/BottomK mask
            # attn_score_fp16 = attn_score.half()
            # flat_score = attn_score_fp16.flatten()

            # if sparsity <= 0.5:
            #     K = int(round(sparsity * X * Y))
            #     mask[b, n] = 0
            #     _, topk_indices = torch.topk(flat_score, K)
            #     for idx in topk_indices:
            #         qx, ky = idx.item() // Y, idx.item() % Y
            #         mask[b, n, qx, ky] = 1
            # else:
            #     K = int(round((1 - sparsity) * X * Y))
            #     mask[b, n] = 1
            #     _, bottomk_indices = torch.topk(-flat_score, K)
            #     for idx in bottomk_indices:
            #         qx, ky = idx.item() // Y, idx.item() % Y
            #         mask[b, n, qx, ky] = 0

    return q_comp_all, k_comp_all, score_all, attn_score_all, mask


def main():
    parser = argparse.ArgumentParser(description="Generate golden data for MaskKComp")
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    prepare_dir(output_dir)

    # Load parameters
    params, shapes = load_params_txt(input_dir)

    B = params["B"]
    N = params["N"]
    S1 = params["S1"]
    S2 = params["S2"]
    D = params["D"]
    bx = params["bx"]
    by = params["by"]
    sparsity = params["sparsity"]
    scale = params["scaleValue"]
    X = params["xBlocks"]
    Y = params["yBlocks"]

    # Load input tensors
    query = load_tensor_bin(os.path.join(input_dir, "input_query.bin"),
                            shapes["query"], torch.float16)
    key = load_tensor_bin(os.path.join(input_dir, "input_key.bin"),
                          shapes["key"], torch.float16)

    # Load optional arrays
    actual_block_len_query = None
    actual_block_len_key = None

    if params.get("use_q_block_len", 0):
        flat_q_blocks = load_int_list_bin(os.path.join(input_dir, "input_block_len_q.bin"),
                                          B * X)
        # Reshape to [B][X] list-of-lists
        actual_block_len_query = [flat_q_blocks[i * X:(i + 1) * X] for i in range(B)]

    if params.get("use_k_block_len", 0):
        flat_k_blocks = load_int_list_bin(os.path.join(input_dir, "input_block_len_k.bin"),
                                          B * Y)
        actual_block_len_key = [flat_k_blocks[i * Y:(i + 1) * Y] for i in range(B)]

    # Compute golden
    print(f"Computing golden: B={B}, N={N}, S1={S1}, S2={S2}, D={D}, bx={bx}, by={by}")
    print(f"  sparsity={sparsity}, scale={scale}, X={X}, Y={Y}")

    q_comp, k_comp, score, attn_score, mask = \
        verify_mask_k_comp(query, key, bx, by, sparsity, scale,
                          actual_block_len_query, actual_block_len_key)

    # Save golden results
    q_comp.cpu().numpy().tofile(os.path.join(output_dir, "golden_q_comp.bin"))
    k_comp.cpu().numpy().tofile(os.path.join(output_dir, "golden_k_comp.bin"))
    score.cpu().numpy().tofile(os.path.join(output_dir, "golden_score.bin"))
    attn_score.cpu().numpy().tofile(os.path.join(output_dir, "golden_attn_score.bin"))
    mask.cpu().numpy().astype(np.int8).tofile(os.path.join(output_dir, "golden_mask.bin"))

    # Save golden shapes metadata
    golden_shapes = {
        "q_comp": list(q_comp.shape),
        "k_comp": list(k_comp.shape),
        "score": list(score.shape),
        "attn_score": list(attn_score.shape),
        "mask": list(mask.shape),
    }

    with open(os.path.join(output_dir, "golden_meta.txt"), "w") as f:
        for name, shape in golden_shapes.items():
            shape_str = ",".join(str(s) for s in shape)
            f.write(f"shape_{name}={shape_str}\n")
        f.write("dtype_q_comp=FP32\n")
        f.write("dtype_k_comp=FP32\n")
        f.write("dtype_score=FP32\n")
        f.write("dtype_attn_score=FP16\n")
        f.write("dtype_mask=INT8\n")

    print(f"Golden data saved to {output_dir}")
    print(f"  golden_q_comp.bin     shape={list(q_comp.shape)} dtype=FP32")
    print(f"  golden_k_comp.bin     shape={list(k_comp.shape)} dtype=FP32")
    print(f"  golden_score.bin      shape={list(score.shape)} dtype=FP32")
    print(f"  golden_attn_score.bin shape={list(attn_score.shape)} dtype=FP16")
    print(f"  golden_mask.bin       shape={list(mask.shape)} dtype=INT8")


if __name__ == "__main__":
    main()
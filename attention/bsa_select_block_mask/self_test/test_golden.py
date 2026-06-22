

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

FLOAT_BLOCK = 8
FP16_BLOCK = 16
NEG_INF = -65504.0

# B, N, X, Y = 5, 10, 18, 17
SCALE = 1
Q_CHUNK = 128
K_CHUNK = 128
SEED = 2026


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
        # row_start, row_count = split_q_rows(q_start, q_size, vec_idx)
        row_start, row_count = q_start, q_size
        if row_count == 0:
            continue

        global_max = torch.full((row_count,), float('-inf'), dtype=torch.float32)
        global_sum = torch.zeros((row_count,), dtype=torch.float32)

        for k_start in range(0, Y, K_CHUNK):
            k_size = min(K_CHUNK, Y - k_start)
            k_align = align_up(k_size, FLOAT_BLOCK)
            chunk = score[row_start:row_start + row_count, k_start:k_start + k_size].float()
            if k_align > k_size:
                pad = torch.full((row_count, k_align - k_size), NEG_INF, dtype=torch.float32)
                chunk = torch.cat([chunk, pad], dim=1)

            local_max = chunk.max(dim=1).values
            new_max = torch.maximum(global_max, local_max)
            t1 = global_max - new_max
            t2 = torch.exp(t1)
            t3 = global_sum * t2

            t4 = chunk - new_max.unsqueeze(1)
            t5 = torch.exp(t4)
            t6 = t5.sum(dim=1)
            global_sum = t3 + t6

            # global_sum = global_sum * torch.exp(global_max - new_max) + torch.exp(chunk - new_max.unsqueeze(1)).sum(dim=1)

            global_max = new_max

        for k_start in range(0, Y, K_CHUNK):
            k_size = min(K_CHUNK, Y - k_start)
            k_align = max(align_up(k_size, FLOAT_BLOCK), align_up(k_size, FP16_BLOCK))
            chunk = score[row_start:row_start + row_count, k_start:k_start + k_size].float()
            if k_align > k_size:
                pad = torch.full((row_count, k_align - k_size), NEG_INF, dtype=torch.float32)
                chunk = torch.cat([chunk, pad], dim=1)
            
            t1 = chunk - global_max.unsqueeze(1)
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
            attn_score_ref = torch.softmax(score, dim=-1).half()
            attn_score_kernel = kernel_like_softmax(score)

            abs_err = (attn_score_kernel.float() - attn_score_ref.float()).abs()
            rel_err = abs_err / torch.clamp(attn_score_ref.float().abs(), min=1e-12)
            print(f"[b={b}, n={n}] softmax replace max_abs_err={abs_err.max().item():.8e} max_rel_err={rel_err.max().item():.8e} allclose={torch.allclose(attn_score_kernel, attn_score_ref, atol=1e-3, rtol=1e-3)}")

            attn_score_all[b, n] = attn_score_kernel

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
    torch.manual_seed(SEED)

    B, N, S1, S2, D = 10, 5, 1250, 1280, 128
    bx, by = 128, 128
    sparsity = 0.5
    scale = 0.125

    query = torch.randn(B, N, S1, D, dtype=torch.float16)
    key = torch.randn(B, N, S2, D, dtype=torch.float16)

    verify_mask_k_comp(query, key, bx, by, sparsity, scale)


if __name__ == "__main__":
    main()

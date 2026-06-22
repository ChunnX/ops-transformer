#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

FLOAT_BLOCK = 8
FP16_BLOCK = 16
NEG_INF = -65504.0

B, N, X, Y = 1, 1, 8, 17
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


def kernel_like_single_vec(score: torch.Tensor, vec_idx: int) -> torch.Tensor:
    out = torch.zeros_like(score, dtype=torch.float16)

    for b in range(B):
        for n in range(N):
            score_bn = score[b, n]
            for q_start in range(0, X, Q_CHUNK):
                q_size = min(Q_CHUNK, X - q_start)
                row_start, row_count = split_q_rows(q_start, q_size, vec_idx)
                if row_count == 0:
                    continue

                global_max = torch.full((row_count,), float('-inf'), dtype=torch.float32)
                global_sum = torch.zeros((row_count,), dtype=torch.float32)

                for k_start in range(0, Y, K_CHUNK):
                    k_size = min(K_CHUNK, Y - k_start)
                    k_align = align_up(k_size, FLOAT_BLOCK)
                    chunk = score_bn[row_start:row_start + row_count, k_start:k_start + k_size].float() * SCALE
                    if k_align > k_size:
                        pad = torch.full((row_count, k_align - k_size), NEG_INF, dtype=torch.float32)
                        chunk = torch.cat([chunk, pad], dim=1)

                    local_max = chunk.max(dim=1).values
                    new_max = torch.maximum(global_max, local_max)
                    global_sum = global_sum * torch.exp(global_max - new_max) + torch.exp(chunk - new_max.unsqueeze(1)).sum(dim=1)
                    global_max = new_max

                for k_start in range(0, Y, K_CHUNK):
                    k_size = min(K_CHUNK, Y - k_start)
                    k_align = max(align_up(k_size, FLOAT_BLOCK), align_up(k_size, FP16_BLOCK))
                    chunk = score_bn[row_start:row_start + row_count, k_start:k_start + k_size].float() * SCALE
                    if k_align > k_size:
                        pad = torch.full((row_count, k_align - k_size), NEG_INF, dtype=torch.float32)
                        chunk = torch.cat([chunk, pad], dim=1)

                    prob = torch.exp(chunk - global_max.unsqueeze(1)) / global_sum.unsqueeze(1)
                    out[b, n, row_start:row_start + row_count, k_start:k_start + k_size] = prob[:, :k_size].half()

    return out


def kernel_like_softmax(score: torch.Tensor) -> torch.Tensor:
    return kernel_like_single_vec(score, 0) + kernel_like_single_vec(score, 1)


def main():
    torch.manual_seed(SEED)
    score = torch.randn(B, N, X, Y, dtype=torch.float32)

    kernel_out = kernel_like_softmax(score)
    torch_out = torch.softmax(score * SCALE, dim=-1).half()

    abs_err = (kernel_out.float() - torch_out.float()).abs()
    rel_err = abs_err / torch.clamp(torch_out.float().abs(), min=1e-12)
    idx = abs_err.reshape(-1).argmax().item()

    print('=== Config ===')
    print(f'shape={[B, N, X, Y]}, scale={SCALE}, q_chunk={Q_CHUNK}, k_chunk={K_CHUNK}, seed={SEED}')
    print('=== Compare ===')
    print(f'max_abs_err={abs_err.max().item():.8e}')
    print(f'max_rel_err={rel_err.max().item():.8e}')
    print(f'worst_idx={idx}')
    print(f'torch={torch_out.reshape(-1)[idx].item():.8e}')
    print(f'kernel={kernel_out.reshape(-1)[idx].item():.8e}')
    print(f'allclose={torch.allclose(kernel_out, torch_out, atol=1e-3, rtol=1e-3)}')


if __name__ == '__main__':
    main()

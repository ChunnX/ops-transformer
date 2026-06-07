# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import math

import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from npu_ops_transformer.ops import npu_stem_indexer


DEVICE_ID = int(os.environ.get("DEVICE_ID", "3"))
torch_npu.npu.set_device(int(DEVICE_ID))


class TestCustomStemIndexer(TestCase):
    def _stem_indexer(self, qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens,
                      causal=True, alpha=1.0, stem_block_size=128, stem_stride=16,
                      initial_blocks=4, window_size=4):
        b = qflat.shape[0]
        q_heads = qflat.shape[1]
        kv_heads = kflat.shape[1]
        max_qb = qflat.shape[2]
        max_kb = kflat.shape[2]
        g = q_heads // kv_heads

        qflat = qflat.to(torch.float32)
        kflat = kflat.to(torch.float32)
        vbias = vbias.to(torch.float32)

        sparse_indices = torch.full((b, q_heads, max_qb, max_kb), -1, dtype=torch.int32)
        sparse_seq_len = torch.zeros((b, q_heads, max_qb), dtype=torch.int32)

        score_scale = 1.0 / ((stem_block_size // stem_stride) ** 2)
        q_lens = q_seq_lens.cpu().tolist()
        kv_lens = kv_seq_lens.cpu().tolist()
        prompt_prefix = num_prompt_tokens.cpu().tolist()

        prompt_lens = []
        last_prompt = 0
        for value in prompt_prefix:
            prompt_lens.append(value - last_prompt)
            last_prompt = value

        for batch_id in range(b):
            q_len = q_lens[batch_id]
            kv_len = kv_lens[batch_id]
            prompt_len = prompt_lens[batch_id]
            q_block_num = (q_len + stem_block_size - 1) // stem_block_size
            kv_block_num = (kv_len + stem_block_size - 1) // stem_block_size
            decode = q_len == 1 and prompt_len >= max(kv_len - 1, 0)

            if kv_len <= 16384:
                rate = 0.2
                bias = 30
            else:
                rate = 0.1
                bias = 30

            for kv_head_id in range(kv_heads):
                q_head_begin = kv_head_id * g
                q_head_end = q_head_begin + g
                q_group = qflat[batch_id, q_head_begin:q_head_end].reshape(g * max_qb, -1)
                k_group = kflat[batch_id, kv_head_id]
                score = torch.matmul(q_group, k_group.transpose(0, 1)) * score_scale
                score = score.reshape(g, max_qb, max_kb)
                score = score + vbias[batch_id, kv_head_id].reshape(1, 1, max_kb)

                for local_q_head_id, q_head_id in enumerate(range(q_head_begin, q_head_end)):
                    for q_block_id in range(q_block_num):
                        if causal and not decode:
                            s2_valid = min(kv_block_num, q_block_id + 1)
                        else:
                            s2_valid = kv_block_num
                        if s2_valid <= 0:
                            continue

                        k_start = rate * kv_block_num + bias
                        if decode:
                            position = max(kv_len - 1, 0)
                            denominator = max(kv_len, 1)
                        else:
                            position = q_block_id
                            denominator = max(q_block_num, 1)
                        topk_num = math.floor(k_start - (k_start * (1.0 - alpha) / denominator) * position)
                        topk_num = min(max(topk_num, 0), s2_valid)
                        if topk_num <= 0:
                            continue

                        sink = set(range(0, min(initial_blocks, s2_valid)))
                        window_begin = max(s2_valid - window_size, 0)
                        window = set(range(window_begin, s2_valid))
                        selected = sink.union(window)

                        remain_count = max(topk_num - len(selected), 0)
                        candidates = [idx for idx in range(s2_valid) if idx not in selected]
                        candidates = sorted(candidates, key=lambda idx: (-float(score[local_q_head_id, q_block_id, idx]), idx))
                        selected = sorted(selected.union(candidates[:remain_count]))

                        sparse_seq_len[batch_id, q_head_id, q_block_id] = len(selected)
                        sparse_indices[batch_id, q_head_id, q_block_id, :len(selected)] = torch.tensor(
                            selected, dtype=torch.int32
                        )

        return sparse_indices, sparse_seq_len

    def test_stem_indexer(self):
        b = 1
        q_heads = 32
        kv_heads = 8
        q_len = 512
        kv_len = 32768
        head_dim = 128
        stem_block_size = 128
        stem_stride = 16
        flatten_dim = stem_stride * head_dim
        max_qb = (q_len + stem_block_size - 1) // stem_block_size
        max_kb = (kv_len + stem_block_size - 1) // stem_block_size
        causal = False
        alpha = 1.0
        initial_blocks = 4
        window_size = 4

        np.random.seed(0)
        qflat = torch.tensor(np.random.uniform(-1, 1, (b, q_heads, max_qb, flatten_dim))).to(torch.bfloat16)
        kflat = torch.tensor(np.random.uniform(-1, 1, (b, kv_heads, max_kb, flatten_dim))).to(torch.bfloat16)
        vbias = torch.arange(max_kb, dtype=torch.float32).reshape(1, 1, max_kb)
        vbias = vbias.repeat(b, kv_heads, 1) * 1000.0
        q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
        kv_seq_lens = torch.tensor([kv_len], dtype=torch.int32)
        num_prompt_tokens = torch.tensor([q_len], dtype=torch.int32)
        metadata = torch.zeros((2048,), dtype=torch.int32)

        cpu_indices, cpu_seq_len = self._stem_indexer(
            qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens,
            causal=causal, alpha=alpha, stem_block_size=stem_block_size, stem_stride=stem_stride,
            initial_blocks=initial_blocks, window_size=window_size
        )

        qflat = qflat.npu()
        kflat = kflat.npu()
        vbias = vbias.npu()
        q_seq_lens = q_seq_lens.npu()
        kv_seq_lens = kv_seq_lens.npu()
        num_prompt_tokens = num_prompt_tokens.npu()
        metadata = metadata.npu()

        expected_g_size = q_heads // kv_heads
        expected_score_scale = 1.0 / ((stem_block_size // stem_stride) ** 2)
        expected_score_workspace_bytes = 96 * 256 * 4 * 2
        print(
            "[StemIndexer Expected Init] "
            f"bSize={b} qHeadNum={q_heads} kvHeadNum={kv_heads} gSize={expected_g_size} "
            f"maxQb={max_qb} maxKb={max_kb} flattenDim={flatten_dim} causal={int(causal)} "
            f"stemBlockSize={stem_block_size} stemStride={stem_stride} "
            f"initialBlocks={initial_blocks} windowSize={window_size} "
            f"mBaseSize=96 nBaseSize=256 scoreScale={expected_score_scale:.6f} alpha={alpha:.6f} "
            "kBlockNumRateMedium=0.200000 kBlockNumBiasMedium=30 "
            "kBlockNumRateLarge=0.100000 kBlockNumBiasLarge=30 "
            f"scoreWorkspaceBytes={expected_score_workspace_bytes}",
            flush=True
        )

        npu_indices, npu_seq_len = npu_stem_indexer(
            qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens, metadata,
            causal=causal, stem_block_size=stem_block_size, stem_stride=stem_stride, alpha=alpha,
            initial_blocks=initial_blocks, window_size=window_size,
            k_block_num_rate_medium=0.2, k_block_num_bias_medium=30,
            k_block_num_rate_large=0.1, k_block_num_bias_large=30
        )
        torch_npu.npu.synchronize()

        npu_indices = npu_indices.cpu().to(torch.int32)
        npu_seq_len = npu_seq_len.cpu().to(torch.int32)

        print("compare result")
        torch.set_printoptions(profile="full")
        print("npu_seq_len", npu_seq_len)
        print("cpu_seq_len", cpu_seq_len)
        print("npu_indices", npu_indices[0, 0, 0, :int(cpu_seq_len[0, 0, 0])])
        print("cpu_indices", cpu_indices[0, 0, 0, :int(cpu_seq_len[0, 0, 0])])

        self.assertEqual(tuple(npu_seq_len.shape), tuple(cpu_seq_len.shape))
        self.assertEqual(tuple(npu_indices.shape), tuple(cpu_indices.shape))
        self.assertTrue(torch.equal(npu_seq_len, cpu_seq_len))

        failures = []
        for index in torch.nonzero(cpu_seq_len >= 0, as_tuple=False):
            b_id, q_head_id, q_block_id = [int(item) for item in index]
            valid_len = int(cpu_seq_len[b_id, q_head_id, q_block_id])
            if valid_len == 0:
                continue
            npu_prefix = npu_indices[b_id, q_head_id, q_block_id, :valid_len].tolist()
            cpu_prefix = cpu_indices[b_id, q_head_id, q_block_id, :valid_len].tolist()
            if set(npu_prefix) != set(cpu_prefix):
                failures.append((b_id, q_head_id, q_block_id, npu_prefix, cpu_prefix))
            if len(failures) >= 8:
                break

        self.assertEqual(failures, [])


if __name__ == "__main__":
    run_tests()

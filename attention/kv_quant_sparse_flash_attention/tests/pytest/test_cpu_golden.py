#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
CPU golden 计算链路的单元测试。
不依赖 NPU，通过在已知输入下重现 attention 参考结果来验证 compute_cpu_golden 的正确性。

运行方法（在 pytest 目录下）：
    pytest test_cpu_golden.py -v
"""

import math

import numpy as np
import torch

import kv_quant_sparse_flash_attention_golden as G

SEED = 42


# =============================================================================
# 参考实现（独立于被测代码，用于对比）
# =============================================================================

def _ref_dequantize(fp8_tensor, scale_tensor, tile_size):
    """fp8_tensor (float8_e4m3fn, shape *xD) * antiquant_scale (shape *x num_tiles)，按 tile 广播。"""
    fp32 = fp8_tensor.to(torch.float32)
    D = fp32.shape[-1]
    num_tiles = D // tile_size
    prefix = fp32.shape[:-1]
    scale_broadcast = scale_tensor.reshape(*prefix, num_tiles, 1).expand(
        *prefix, num_tiles, tile_size
    ).reshape(*prefix, D)
    return fp32 * scale_broadcast


def _unpack_key(key_packed, D_base, rope_head_dim, tile_size, q_dtype):
    """从打包的 key tensor 中拆分出 key_base(fp8)、key_rope(float32)、antiquant_scale(float32)。

    key_packed: (..., D_base + 2*rope_head_dim + 4*num_tiles) storage_dtype
    返回: key_base_fp8, key_rope_fp32, antiquant_scale_fp32，前缀维度与输入一致
    """
    rope_slots = rope_head_dim * 2          # q_type 各 2 字节 → 2*rope_head_dim slots
    scale_slots = key_packed.shape[-1] - D_base - rope_slots
    num_tiles = scale_slots // 4            # float32 各 4 字节

    key_base_fp8 = key_packed[..., :D_base]
    rope_raw = key_packed[..., D_base:D_base + rope_slots].contiguous().view(torch.uint8)
    scale_raw = key_packed[..., D_base + rope_slots:].contiguous().view(torch.uint8)

    rope_bytes = rope_raw.reshape(-1).numpy().tobytes()
    if q_dtype == torch.float16:
        key_rope = torch.from_numpy(
            np.frombuffer(rope_bytes, dtype=np.float16).copy()
        ).float().reshape(*key_packed.shape[:-1], rope_head_dim)
    else:
        key_rope = torch.frombuffer(
            bytearray(rope_bytes), dtype=torch.bfloat16
        ).float().reshape(*key_packed.shape[:-1], rope_head_dim)

    scale_bytes = scale_raw.reshape(-1).numpy().tobytes()
    antiquant_scale = torch.from_numpy(
        np.frombuffer(scale_bytes, dtype=np.float32).copy()
    ).reshape(*key_packed.shape[:-1], num_tiles)

    return key_base_fp8, key_rope, antiquant_scale


def _ref_attention_bsnd(q_bsnd, k_full_bsnd, v_bsnd, sp_bsnd,
                        act_seq_q, act_seq_kv, scale_value, sparse_block_size, sparse_mode):
    """参考 attention，工作在 BSND 空间，输出 float32 BSND tensor。

    q_bsnd:       (B, S1, N1, D_q)    float32
    k_full_bsnd:  (B, S2, N2, D_k)    float32，D_k = D_base + rope_head_dim
    v_bsnd:       (B, S2, N2, D_base) float32
    sp_bsnd:      (B, S1, N2, K)      int32
    返回: (B, S1, N1, D_base) float32
    """
    B, S1, N1, _ = q_bsnd.shape
    N2 = k_full_bsnd.shape[2]
    D_v = v_bsnd.shape[-1]
    g = N1 // N2
    y = torch.zeros(B, S1, N1, D_v, dtype=torch.float32)

    for b in range(B):
        act_q = act_seq_q[b]
        act_kv = act_seq_kv[b]
        for n2_idx in range(N2):
            for s1_idx in range(act_q):
                if sparse_mode == 3 and s1_idx < act_q - act_kv:
                    continue
                if sparse_mode == 0:
                    threshold = act_kv
                else:
                    threshold = act_kv - act_q + s1_idx + 1

                sp = sp_bsnd[b, s1_idx, n2_idx, :]
                valid_count = min(sp.shape[0], math.ceil(max(0, threshold) / sparse_block_size))
                s2_indices = []
                for i in range(valid_count):
                    blk_id = int(sp[i].item())
                    if blk_id == -1:
                        break
                    begin = blk_id * sparse_block_size
                    end = min(begin + sparse_block_size, act_kv)
                    if begin >= threshold:
                        continue
                    end = min(end, int(threshold))
                    s2_indices.extend(range(begin, end))

                if not s2_indices:
                    continue

                q_curr = q_bsnd[b, s1_idx, n2_idx * g:(n2_idx + 1) * g, :].float()
                k_sp = k_full_bsnd[b, s2_indices, n2_idx, :].float()
                v_sp = v_bsnd[b, s2_indices, n2_idx, :].float()

                bmm1 = torch.matmul(q_curr, k_sp.T) * scale_value
                sm = torch.softmax(bmm1, dim=-1)
                y[b, s1_idx, n2_idx * g:(n2_idx + 1) * g, :] = torch.matmul(sm, v_sp)

    return y


def _make_params(layout_query, layout_kv, sparse_mode, q_type=torch.float16,
                 B=2, S1=3, S2=8, N1=2, N2=1, D=128, K=3,
                 scale_value=0.125, sparse_block_size=1,
                 tile_size=128, rope_head_dim=64,
                 block_size=16, block_num=None, actual_seq_q=None, actual_seq_kv=None):
    return (
        "cpu_golden_test",
        layout_query, layout_kv,
        q_type,
        None,           # kv_dtype=None → float8_e4m3fn
        B, S1, S2, N1, N2, D, K,
        scale_value,
        2,              # key_quant_mode
        2,              # value_quant_mode
        sparse_block_size,
        tile_size, rope_head_dim,
        sparse_mode,
        2,              # attention_mode
        1,              # quant_scale_repo_mode
        block_size, block_num,
        actual_seq_q, actual_seq_kv,
    )


# =============================================================================
# 测试用例
# =============================================================================

class TestCpuGoldenBSND:
    """BSND layout，sparse_mode=0，基础路径验证。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.B, self.S1, self.S2 = 2, 3, 8
        self.N1, self.N2, self.D, self.K = 2, 1, 128, 3
        self.act_seq_q = [2, 3]
        self.act_seq_kv = [6, 8]
        params = _make_params(
            "BSND", "BSND", sparse_mode=0,
            B=self.B, S1=self.S1, S2=self.S2,
            N1=self.N1, N2=self.N2, D=self.D, K=self.K,
            actual_seq_q=self.act_seq_q, actual_seq_kv=self.act_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        assert out.shape == (self.B, self.S1, self.N1, self.D), \
            f"期望 {(self.B, self.S1, self.N1, self.D)}，实际 {out.shape}"

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_output_vs_reference(self):
        inp = self.inp
        q_dtype = inp["query"].dtype
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D

        key_base_fp8, key_rope, antiquant_scale = _unpack_key(
            inp["key"], D_base, rope_head_dim, tile_size, q_dtype)
        k_full = torch.cat([_ref_dequantize(key_base_fp8, antiquant_scale, tile_size), key_rope], dim=-1)
        v_fp32 = _ref_dequantize(inp["value"], antiquant_scale, tile_size)

        ref = _ref_attention_bsnd(
            inp["query"].float(), k_full, v_fp32, inp["sparse_indices"],
            self.act_seq_q, self.act_seq_kv,
            float(inp["scale_value"]), inp["sparse_block_size"], sparse_mode=0,
        )
        torch.testing.assert_close(self.data["cpu_output"].float(), ref, atol=1e-4, rtol=1e-3,
                                   msg="BSND sparse_mode=0 与参考实现不一致")


class TestCpuGoldenSparseMode3:
    """BSND layout，sparse_mode=3，因果掩码验证。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.B, self.S1, self.S2 = 1, 4, 8
        self.N1, self.N2, self.D, self.K = 2, 1, 128, 4
        self.act_seq_q = [4]
        self.act_seq_kv = [8]
        params = _make_params(
            "BSND", "BSND", sparse_mode=3,
            B=self.B, S1=self.S1, S2=self.S2,
            N1=self.N1, N2=self.N2, D=self.D, K=self.K,
            actual_seq_q=self.act_seq_q, actual_seq_kv=self.act_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        assert out.shape == (self.B, self.S1, self.N1, self.D)

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_causal_prefix_is_zero(self):
        out = self.data["cpu_output"].float()
        act_q = self.act_seq_q[0]
        act_kv = self.act_seq_kv[0]
        zero_len = act_q - act_kv
        if zero_len > 0:
            assert out[0, :zero_len, :, :].abs().max().item() == 0.0, \
                "sparse_mode=3 因果前缀位置输出应为 0"

    def test_output_vs_reference(self):
        inp = self.inp
        q_dtype = inp["query"].dtype
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D

        key_base_fp8, key_rope, antiquant_scale = _unpack_key(
            inp["key"], D_base, rope_head_dim, tile_size, q_dtype)
        k_full = torch.cat([_ref_dequantize(key_base_fp8, antiquant_scale, tile_size), key_rope], dim=-1)
        v_fp32 = _ref_dequantize(inp["value"], antiquant_scale, tile_size)

        ref = _ref_attention_bsnd(
            inp["query"].float(), k_full, v_fp32, inp["sparse_indices"],
            self.act_seq_q, self.act_seq_kv,
            float(inp["scale_value"]), inp["sparse_block_size"], sparse_mode=3,
        )
        torch.testing.assert_close(self.data["cpu_output"].float(), ref, atol=1e-4, rtol=1e-3,
                                   msg="BSND sparse_mode=3 与参考实现不一致")


class TestCpuGoldenTND:
    """TND layout，sparse_mode=0，layout 转换验证。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.B, self.S1, self.S2 = 2, 4, 6
        self.N1, self.N2, self.D, self.K = 2, 1, 128, 3
        self.act_seq_q = [2, 4]
        self.act_seq_kv = [4, 6]
        params = _make_params(
            "TND", "TND", sparse_mode=0,
            B=self.B, S1=self.S1, S2=self.S2,
            N1=self.N1, N2=self.N2, D=self.D, K=self.K,
            actual_seq_q=self.act_seq_q, actual_seq_kv=self.act_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        T = sum(self.act_seq_q)
        assert out.shape == (T, self.N1, self.D), \
            f"期望 TND shape {(T, self.N1, self.D)}，实际 {out.shape}"

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_output_vs_reference(self):
        """将 TND key/v/q/sparse 展开为 BSND，与参考实现对比。"""
        inp = self.inp
        q_dtype = inp["query"].dtype
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D
        act_q = self.act_seq_q
        act_kv = self.act_seq_kv
        S2_max = max(act_kv)
        N2 = self.N2

        # ---- TND key -> 拆分 + dequant ----
        kv_tnd = inp["key"]   # (T_kv, N2, D_packed)
        key_base_fp8, key_rope_tnd, antiquant_scale_tnd = _unpack_key(
            kv_tnd, D_base, rope_head_dim, tile_size, q_dtype)
        k_fp32_tnd = _ref_dequantize(key_base_fp8, antiquant_scale_tnd, tile_size)
        v_fp32_tnd = _ref_dequantize(inp["value"], antiquant_scale_tnd, tile_size)

        # ---- TND -> BSND ----
        k_full_bsnd = torch.zeros(self.B, S2_max, N2, D_base + rope_head_dim)
        v_bsnd = torch.zeros(self.B, S2_max, N2, D_base)
        t = 0
        for b, s in enumerate(act_kv):
            k_full_bsnd[b, :s] = torch.cat([k_fp32_tnd[t:t + s], key_rope_tnd[t:t + s]], dim=-1)
            v_bsnd[b, :s] = v_fp32_tnd[t:t + s]
            t += s

        # ---- TND query -> BSND ----
        q_tnd = inp["query"].float()   # (T_q, N1, D_full)
        N1 = q_tnd.shape[1]
        q_bsnd = torch.zeros(self.B, self.S1, N1, q_tnd.shape[-1])
        t = 0
        for b, s in enumerate(act_q):
            q_bsnd[b, :s] = q_tnd[t:t + s]
            t += s

        # ---- TND sparse_indices -> BSND ----
        sp_tnd = inp["sparse_indices"]   # (T_q, N2, K)
        sp_bsnd = torch.full((self.B, self.S1, N2, self.K), -1, dtype=torch.int32)
        t = 0
        for b, s in enumerate(act_q):
            sp_bsnd[b, :s] = sp_tnd[t:t + s]
            t += s

        ref_bsnd = _ref_attention_bsnd(
            q_bsnd, k_full_bsnd, v_bsnd, sp_bsnd,
            act_q, act_kv,
            float(inp["scale_value"]), inp["sparse_block_size"], sparse_mode=0,
        )

        # ---- cpu_output (TND) -> BSND 对比 ----
        got_tnd = self.data["cpu_output"].float()   # (T_q, N1, D_base)
        got_bsnd = torch.zeros(self.B, self.S1, N1, D_base)
        t = 0
        for b, s in enumerate(act_q):
            got_bsnd[b, :s] = got_tnd[t:t + s]
            t += s

        torch.testing.assert_close(got_bsnd, ref_bsnd, atol=1e-4, rtol=1e-3,
                                   msg="TND sparse_mode=0 与参考实现不一致")


class TestCpuGoldenPABSND:
    """PA_BSND layout，sparse_mode=0，block cache 重建验证。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.B, self.S1, self.S2 = 2, 2, 32
        self.N1, self.N2, self.D, self.K = 2, 1, 128, 4
        self.block_size = 16
        self.act_seq_q = [1, 2]
        self.act_seq_kv = [16, 32]
        block_num = sum(math.ceil(s / self.block_size) for s in self.act_seq_kv) + 2
        params = _make_params(
            "BSND", "PA_BSND", sparse_mode=0,
            B=self.B, S1=self.S1, S2=self.S2,
            N1=self.N1, N2=self.N2, D=self.D, K=self.K,
            block_size=self.block_size, block_num=block_num,
            actual_seq_q=self.act_seq_q, actual_seq_kv=self.act_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        assert out.shape == (self.B, self.S1, self.N1, self.D), \
            f"期望 {(self.B, self.S1, self.N1, self.D)}，实际 {out.shape}"

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_output_vs_reference(self):
        """手动从 block cache 重建 BSND K/V，与参考实现对比。"""
        inp = self.inp
        q_dtype = inp["query"].dtype
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D
        block_table = inp["block_table"]   # (B, max_blk) int32

        # key_packed: (block_num, block_size, N2, D_packed)
        key_base_fp8, key_rope, antiquant_scale = _unpack_key(
            inp["key"], D_base, rope_head_dim, tile_size, q_dtype)
        k_fp32_blocks = _ref_dequantize(key_base_fp8, antiquant_scale, tile_size)
        v_fp32_blocks = _ref_dequantize(inp["value"], antiquant_scale, tile_size)
        k_full_blocks = torch.cat([k_fp32_blocks, key_rope], dim=-1)  # (block_num, block_size, N2, D_full)

        # 按 block_table 重建 BSND
        S2_max = max(self.act_seq_kv)
        N2 = self.N2
        D_full_k = k_full_blocks.shape[-1]
        k_bsnd = torch.zeros(self.B, S2_max, N2, D_full_k)
        v_bsnd = torch.zeros(self.B, S2_max, N2, D_base)
        for b in range(self.B):
            for blk_i in range(block_table.shape[1]):
                blk_id = int(block_table[b, blk_i].item())
                if blk_id < 0:
                    continue
                s_begin = blk_i * self.block_size
                if s_begin >= S2_max:
                    break
                s_end = min(s_begin + self.block_size, S2_max)
                copy_len = s_end - s_begin
                k_bsnd[b, s_begin:s_end] = k_full_blocks[blk_id, :copy_len]
                v_bsnd[b, s_begin:s_end] = v_fp32_blocks[blk_id, :copy_len]

        ref = _ref_attention_bsnd(
            inp["query"].float(), k_bsnd, v_bsnd, inp["sparse_indices"],
            self.act_seq_q, self.act_seq_kv,
            float(inp["scale_value"]), inp["sparse_block_size"], sparse_mode=0,
        )
        torch.testing.assert_close(self.data["cpu_output"].float(), ref, atol=1e-4, rtol=1e-3,
                                   msg="PA_BSND sparse_mode=0 与参考实现不一致")

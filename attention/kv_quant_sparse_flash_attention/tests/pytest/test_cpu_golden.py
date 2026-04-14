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
import pytest
import torch

import kv_quant_sparse_flash_attention_golden as G

# 固定随机种子保证可复现
SEED = 42


# =============================================================================
# 参考实现（独立于被测代码，用于对比）
# =============================================================================

def _ref_dequantize(fp8_tensor, scale_tensor, tile_size):
    """参考反量化：fp8_tensor (float8_e4m3fn) * scale，按 tile 广播。

    fp8_tensor: (*, D) torch float8_e4m3fn
    scale_tensor: (*, num_tiles) torch float32
    返回: float32 tensor，shape 同 fp8_tensor
    """
    fp32 = fp8_tensor.to(torch.float32)
    D = fp32.shape[-1]
    num_tiles = D // tile_size
    prefix = fp32.shape[:-1]
    scale_broadcast = scale_tensor.reshape(*prefix, num_tiles, 1).expand(
        *prefix, num_tiles, tile_size
    ).reshape(*prefix, D)
    return fp32 * scale_broadcast


def _ref_attention_bsnd(q_bsnd, k_bsnd, v_bsnd, sparse_indices_bsnd,
                        act_seq_q_list, act_seq_kv_list,
                        scale_value, sparse_block_size, sparse_mode):
    """参考 attention 实现，工作在 BSND 空间，输出 float32 BSND tensor。

    与 compute_cpu_golden 完全独立：逐 (b, n2, s1) 枚举，按 sparse_indices
    聚合 KV 后做 scaled dot-product attention。
    """
    B, S1, N1, D_q = q_bsnd.shape
    N2 = k_bsnd.shape[2]
    D_v = v_bsnd.shape[-1]
    g = N1 // N2
    y = torch.zeros(B, S1, N1, D_v, dtype=torch.float32)

    for b in range(B):
        act_q = act_seq_q_list[b]
        act_kv = act_seq_kv_list[b]
        for n2_idx in range(N2):
            for s1_idx in range(act_q):
                if sparse_mode == 3 and s1_idx < act_q - act_kv:
                    continue

                if sparse_mode == 0:
                    threshold = act_kv
                else:
                    threshold = act_kv - act_q + s1_idx + 1

                sp = sparse_indices_bsnd[b, s1_idx, n2_idx, :]
                K_count = sp.shape[0]
                valid_count = min(K_count, math.ceil(max(0, threshold) / sparse_block_size))

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

                # q_curr: (g, D_q)，k_sp/v_sp: (sparse_len, D)
                q_curr = q_bsnd[b, s1_idx, n2_idx * g:(n2_idx + 1) * g, :].float()
                k_sp = k_bsnd[b, s2_indices, n2_idx, :].float()
                v_sp = v_bsnd[b, s2_indices, n2_idx, :].float()

                # bmm1: (g, sparse_len)
                bmm1 = torch.matmul(q_curr, k_sp.T) * scale_value
                sm = torch.softmax(bmm1, dim=-1)
                bmm2 = torch.matmul(sm, v_sp)
                y[b, s1_idx, n2_idx * g:(n2_idx + 1) * g, :] = bmm2

    return y


def _make_params(layout_query, layout_kv, sparse_mode, q_type=torch.float16,
                 B=2, S1=3, S2=8, N1=2, N2=1, D=128, K=3,
                 scale_value=0.125, sparse_block_size=1,
                 tile_size=128, rope_head_dim=64,
                 block_size=16, block_num=None, actual_seq_q=None, actual_seq_kv=None):
    """构造符合 generate_and_save_testdata 签名的 params 元组。"""
    return (
        "cpu_golden_test",       # testcase_name
        layout_query, layout_kv,
        q_type,
        None,                    # kv_dtype=None 即 float8_e4m3fn
        B, S1, S2, N1, N2, D, K,
        scale_value,
        2,                       # key_quant_mode
        2,                       # value_quant_mode
        sparse_block_size,
        tile_size, rope_head_dim,
        sparse_mode,
        2,                       # attention_mode
        1,                       # quant_scale_repo_mode
        block_size, block_num,
        actual_seq_q, actual_seq_kv,
    )


def _unpack_key(inp, D_base, rope_head_dim, tile_size):
    """从打包的 key tensor 中提取 key_base(fp8)、key_rope(float32)、antiquant_scale。"""
    key_packed = inp["key"]   # storage_dtype，最后一维 = D_base + 2*rope_head_dim + 4*num_tiles
    rope_slots = rope_head_dim * 2
    scale_slots = key_packed.shape[-1] - D_base - rope_slots
    num_tiles = scale_slots // 4

    key_base_fp8 = key_packed[..., :D_base]
    key_rope_raw = key_packed[..., D_base:D_base + rope_slots].contiguous().view(torch.uint8)
    scale_raw = key_packed[..., D_base + rope_slots:].contiguous().view(torch.uint8)

    # key_rope: bytes -> float16/bf16 -> float32
    q_dtype = inp["query"].dtype
    rope_bytes = key_rope_raw.reshape(-1).numpy().tobytes()
    if q_dtype == torch.float16:
        key_rope = torch.from_numpy(
            np.frombuffer(rope_bytes, dtype=np.float16).copy()
        ).float().reshape(*key_packed.shape[:-1], rope_head_dim)
    else:
        key_rope = torch.frombuffer(
            bytearray(rope_bytes), dtype=torch.bfloat16
        ).float().reshape(*key_packed.shape[:-1], rope_head_dim)

    # antiquant_scale: bytes -> float32
    scale_bytes = scale_raw.reshape(-1).numpy().tobytes()
    antiquant_scale = torch.from_numpy(
        np.frombuffer(scale_bytes, dtype=np.float32).copy()
    ).reshape(*key_packed.shape[:-1], num_tiles)

    return key_base_fp8, key_rope, antiquant_scale


# =============================================================================
# 测试用例
# =============================================================================

class TestCpuGoldenBSND:
    """BSND layout，sparse_mode=0，基础路径验证。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        B, S1, S2 = 2, 3, 8
        N1, N2, D, K = 2, 1, 128, 3
        actual_seq_q = [2, 3]
        actual_seq_kv = [6, 8]
        params = _make_params(
            "BSND", "BSND", sparse_mode=0,
            B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, K=K,
            actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]
        self.B, self.S1, self.S2 = B, S1, S2
        self.N1, self.N2, self.D, self.K = N1, N2, D, K
        self.act_seq_q = actual_seq_q
        self.act_seq_kv = actual_seq_kv

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None, "cpu_output 不应为 None"

    def test_output_shape(self):
        out = self.data["cpu_output"]
        # BSND 输出：(B, S1, N1, D_base)
        D_base = self.D
        assert out.shape == (self.B, self.S1, self.N1, D_base), \
            f"期望 shape {(self.B, self.S1, self.N1, D_base)}，实际 {out.shape}"

    def test_output_finite(self):
        out = self.data["cpu_output"].float()
        assert torch.isfinite(out).all(), "cpu_output 包含 inf 或 nan"

    def test_output_vs_reference(self):
        """与独立参考实现对比，要求 atol=1e-4。"""
        inp = self.inp
        tile_size = inp["tile_size"]
        rope_head_dim = inp["rope_head_dim"]
        D_base = self.D
        scale_value = float(inp["scale_value"])
        sparse_block_size = inp["sparse_block_size"]

        key_base_fp8, key_rope, antiquant_scale = _unpack_key(inp, D_base, rope_head_dim, tile_size)
        k_base_fp32 = _ref_dequantize(key_base_fp8, antiquant_scale, tile_size)
        # k_full: (B, S2, N2, D_base + rope_head_dim)
        k_full = torch.cat([k_base_fp32, key_rope], dim=-1)

        # value 反量化
        v_base_fp8 = inp["value"]
        v_fp32 = _ref_dequantize(v_base_fp8, antiquant_scale, tile_size)

        # q: (B, S1, N1, D_full)
        q = inp["query"].float()
        # sparse_indices: BSND (B, S1, N2, K)
        sp = inp["sparse_indices"]

        ref = _ref_attention_bsnd(
            q, k_full, v_fp32, sp,
            self.act_seq_q, self.act_seq_kv,
            scale_value, sparse_block_size, sparse_mode=0,
        )
        got = self.data["cpu_output"].float()
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3,
                                   msg="BSND sparse_mode=0 与参考实现不一致")


class TestCpuGoldenSparseMode3:
    """BSND layout，sparse_mode=3（因果掩码），验证因果窗口边界。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        B, S1, S2 = 1, 4, 8
        N1, N2, D, K = 2, 1, 128, 4
        actual_seq_q = [4]
        actual_seq_kv = [8]
        params = _make_params(
            "BSND", "BSND", sparse_mode=3,
            B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, K=K,
            actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]
        self.B, self.S1, self.S2 = B, S1, S2
        self.N1, self.N2, self.D, self.K = N1, N2, D, K
        self.act_seq_q = actual_seq_q
        self.act_seq_kv = actual_seq_kv

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        assert out.shape == (self.B, self.S1, self.N1, self.D)

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_causal_prefix_is_zero(self):
        """sparse_mode=3 时，act_kv < act_q 的前缀 token 输出应为 0。"""
        out = self.data["cpu_output"].float()
        act_q = self.act_seq_q[0]
        act_kv = self.act_seq_kv[0]
        # s1_idx < act_q - act_kv 的位置输出应为 0
        zero_len = act_q - act_kv
        if zero_len > 0:
            assert out[0, :zero_len, :, :].abs().max().item() == 0.0, \
                "sparse_mode=3 因果前缀位置输出应为 0"

    def test_output_vs_reference(self):
        inp = self.inp
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D
        scale_value = float(inp["scale_value"])
        sparse_block_size = inp["sparse_block_size"]

        key_base_fp8, key_rope, antiquant_scale = _unpack_key(inp, D_base, rope_head_dim, tile_size)
        k_full = torch.cat([_ref_dequantize(key_base_fp8, antiquant_scale, tile_size), key_rope], dim=-1)
        v_fp32 = _ref_dequantize(inp["value"], antiquant_scale, tile_size)

        ref = _ref_attention_bsnd(
            inp["query"].float(), k_full, v_fp32, inp["sparse_indices"],
            self.act_seq_q, self.act_seq_kv,
            scale_value, sparse_block_size, sparse_mode=3,
        )
        got = self.data["cpu_output"].float()
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3,
                                   msg="BSND sparse_mode=3 与参考实现不一致")


class TestCpuGoldenTND:
    """TND layout，sparse_mode=0，验证 TND <-> BNSD 转换。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        B, S1, S2 = 2, 4, 6
        N1, N2, D, K = 2, 1, 128, 3
        actual_seq_q = [2, 4]
        actual_seq_kv = [4, 6]
        params = _make_params(
            "TND", "TND", sparse_mode=0,
            B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, K=K,
            actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]
        self.B, self.S1, self.S2 = B, S1, S2
        self.N1, self.N2, self.D, self.K = N1, N2, D, K
        self.act_seq_q = actual_seq_q
        self.act_seq_kv = actual_seq_kv

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        T = sum(self.act_seq_q)
        # TND 输出：(T, N1, D_base)
        assert out.shape == (T, self.N1, self.D), \
            f"期望 TND shape {(T, self.N1, self.D)}，实际 {out.shape}"

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_output_vs_bsnd_equivalent(self):
        """将 TND 输出展开后与 BSND 等价计算对比。"""
        inp = self.inp
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D
        scale_value = float(inp["scale_value"])
        sparse_block_size = inp["sparse_block_size"]

        # 从 TND key 恢复 BSND
        # TND key shape: (T_kv, N2, D_packed)
        kv_tnd = inp["key"]
        kv_packed_D = kv_tnd.shape[-1]
        rope_slots = rope_head_dim * 2
        scale_slots = kv_packed_D - D_base - rope_slots
        num_tiles = scale_slots // 4

        key_base_fp8_tnd = kv_tnd[..., :D_base]
        scale_raw = kv_tnd[..., D_base + rope_slots:].contiguous().view(torch.uint8)
        scale_bytes = scale_raw.reshape(-1).numpy().tobytes()
        antiquant_scale_tnd = torch.from_numpy(
            np.frombuffer(scale_bytes, dtype=np.float32).copy()
        ).reshape(*kv_tnd.shape[:-1], num_tiles)

        rope_raw = kv_tnd[..., D_base:D_base + rope_slots].contiguous().view(torch.uint8)
        rope_bytes = rope_raw.reshape(-1).numpy().tobytes()
        q_dtype = inp["query"].dtype
        if q_dtype == torch.float16:
            key_rope_tnd = torch.from_numpy(
                np.frombuffer(rope_bytes, dtype=np.float16).copy()
            ).float().reshape(*kv_tnd.shape[:-1], rope_head_dim)
        else:
            key_rope_tnd = torch.frombuffer(
                bytearray(rope_bytes), dtype=torch.bfloat16
            ).float().reshape(*kv_tnd.shape[:-1], rope_head_dim)

        # 还原 TND -> BSND
        act_kv = self.act_seq_kv
        S2_max = max(act_kv)
        T_kv = kv_tnd.shape[0]
        N2 = kv_tnd.shape[1]

        k_base_bsnd = torch.zeros(self.B, S2_max, N2, D_base)
        scale_bsnd = torch.zeros(self.B, S2_max, N2, num_tiles)
        rope_bsnd = torch.zeros(self.B, S2_max, N2, rope_head_dim)
        v_tnd = inp["value"]
        v_bsnd = torch.zeros(self.B, S2_max, N2, D_base)

        t = 0
        for b, s in enumerate(act_kv):
            k_base_bsnd[b, :s] = key_base_fp8_tnd[t:t + s].to(torch.float32).unsqueeze(-1).squeeze(-1)
            scale_bsnd[b, :s] = antiquant_scale_tnd[t:t + s]
            rope_bsnd[b, :s] = key_rope_tnd[t:t + s]
            v_bsnd[b, :s] = v_tnd[t:t + s].to(torch.float32)
            t += s

        k_fp32_bsnd = _ref_dequantize(key_base_fp8_tnd, antiquant_scale_tnd, tile_size)
        # 重新按 batch 组装 BSND
        k_fp32_bsnd_full = torch.zeros(self.B, S2_max, N2, D_base)
        t = 0
        for b, s in enumerate(act_kv):
            k_fp32_bsnd_full[b, :s] = k_fp32_bsnd[t:t + s]
            t += s
        k_full_bsnd = torch.cat([k_fp32_bsnd_full, rope_bsnd], dim=-1)

        # TND sparse_indices -> BSND
        # TND sparse shape: (T_q, N2, K)
        sp_tnd = inp["sparse_indices"]
        sp_bsnd = torch.zeros(self.B, self.S1, N2, self.K, dtype=torch.int32).fill_(-1)
        t = 0
        for b, s in enumerate(self.act_seq_q):
            sp_bsnd[b, :s] = sp_tnd[t:t + s]
            t += s

        # TND query -> BSND
        q_tnd = inp["query"].float()
        N1 = q_tnd.shape[1]
        q_bsnd = torch.zeros(self.B, self.S1, N1, q_tnd.shape[-1])
        t = 0
        for b, s in enumerate(self.act_seq_q):
            q_bsnd[b, :s] = q_tnd[t:t + s].permute(1, 0, 2).unsqueeze(0).squeeze(0).permute(1, 0, 2)
            t += s

        ref_bsnd = _ref_attention_bsnd(
            q_bsnd, k_full_bsnd, v_bsnd, sp_bsnd,
            self.act_seq_q, self.act_seq_kv,
            scale_value, sparse_block_size, sparse_mode=0,
        )

        # cpu_output 是 TND，展开与 ref_bsnd 对比
        got_tnd = self.data["cpu_output"].float()  # (T_q, N1, D)
        got_bsnd = torch.zeros(self.B, self.S1, N1, D_base)
        t = 0
        for b, s in enumerate(self.act_seq_q):
            got_bsnd[b, :s] = got_tnd[t:t + s].permute(1, 0, 2).unsqueeze(0).squeeze(0).permute(1, 0, 2)
            t += s

        torch.testing.assert_close(got_bsnd, ref_bsnd, atol=1e-4, rtol=1e-3,
                                   msg="TND sparse_mode=0 与 BSND 参考实现不一致")


class TestCpuGoldenPABSND:
    """PA_BSND layout，sparse_mode=0，验证 block cache 重建。"""

    def setup_method(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        B, S1, S2 = 2, 2, 32
        N1, N2, D, K = 2, 1, 128, 4
        block_size = 16
        actual_seq_q = [1, 2]
        actual_seq_kv = [16, 32]
        block_num = sum(math.ceil(s / block_size) for s in actual_seq_kv) + 2
        params = _make_params(
            "BSND", "PA_BSND", sparse_mode=0,
            B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, K=K,
            block_size=block_size, block_num=block_num,
            actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        )
        self.data = G.generate_and_save_testdata(params, compute_golden=True)
        self.inp = self.data["input"]
        self.B, self.S1, self.S2 = B, S1, S2
        self.N1, self.N2, self.D, self.K = N1, N2, D, K
        self.act_seq_q = actual_seq_q
        self.act_seq_kv = actual_seq_kv
        self.block_size = block_size

    def test_output_not_none(self):
        assert self.data["cpu_output"] is not None

    def test_output_shape(self):
        out = self.data["cpu_output"]
        assert out.shape == (self.B, self.S1, self.N1, self.D), \
            f"期望 PA_BSND shape {(self.B, self.S1, self.N1, self.D)}，实际 {out.shape}"

    def test_output_finite(self):
        assert torch.isfinite(self.data["cpu_output"].float()).all()

    def test_output_vs_reference(self):
        """从 block cache 手动重建 BSND K/V，与参考实现对比。"""
        inp = self.inp
        tile_size, rope_head_dim = inp["tile_size"], inp["rope_head_dim"]
        D_base = self.D
        scale_value = float(inp["scale_value"])
        sparse_block_size = inp["sparse_block_size"]
        block_table = inp["block_table"]  # (B, max_blk) int32

        # key_packed shape: (block_num, block_size, N2, D_packed)
        key_packed = inp["key"]
        kv_packed_D = key_packed.shape[-1]
        rope_slots = rope_head_dim * 2
        scale_slots = kv_packed_D - D_base - rope_slots

        key_base_fp8 = key_packed[..., :D_base]
        scale_raw = key_packed[..., D_base + rope_slots:].contiguous().view(torch.uint8)
        scale_bytes = scale_raw.reshape(-1).numpy().tobytes()
        antiquant_scale = torch.from_numpy(
            np.frombuffer(scale_bytes, dtype=np.float32).copy()
        ).reshape(*key_packed.shape[:-1], scale_slots // 4)

        rope_raw = key_packed[..., D_base:D_base + rope_slots].contiguous().view(torch.uint8)
        rope_bytes = rope_raw.reshape(-1).numpy().tobytes()
        q_dtype = inp["query"].dtype
        if q_dtype == torch.float16:
            key_rope = torch.from_numpy(
                np.frombuffer(rope_bytes, dtype=np.float16).copy()
            ).float().reshape(*key_packed.shape[:-1], rope_head_dim)
        else:
            key_rope = torch.frombuffer(
                bytearray(rope_bytes), dtype=torch.bfloat16
            ).float().reshape(*key_packed.shape[:-1], rope_head_dim)

        k_fp32 = _ref_dequantize(key_base_fp8, antiquant_scale, tile_size)
        k_full_blocks = torch.cat([k_fp32, key_rope], dim=-1)  # (block_num, block_size, N2, D_full)
        v_blocks = inp["value"].to(torch.float32)               # (block_num, block_size, N2, D_base)

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
                s_end = min(s_begin + self.block_size, S2_max)
                if s_begin >= S2_max:
                    break
                copy_len = s_end - s_begin
                k_bsnd[b, s_begin:s_end] = k_full_blocks[blk_id, :copy_len]
                v_bsnd[b, s_begin:s_end] = v_blocks[blk_id, :copy_len]

        ref = _ref_attention_bsnd(
            inp["query"].float(), k_bsnd, v_bsnd, inp["sparse_indices"],
            self.act_seq_q, self.act_seq_kv,
            scale_value, sparse_block_size, sparse_mode=0,
        )
        got = self.data["cpu_output"].float()
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3,
                                   msg="PA_BSND sparse_mode=0 与参考实现不一致")

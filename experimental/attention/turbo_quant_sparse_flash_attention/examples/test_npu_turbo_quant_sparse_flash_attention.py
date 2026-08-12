# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

"""TurboQuantSparseFlashAttention 精度用例。

判据取自 experimental 精度标准
(opbase/docs/zh/ops_precision_standard/experimental_standard.md) 的双条件：
    1) 混合容差 |actual - golden| <= atol + rtol * |golden| 的元素占比 >= 0.99
    2) 最大绝对误差 <= 该数据类型的硬上限

标杆的选取：本算子为 4bit 量化算子，其输出与全精度 Attention 之间存在量化本身引入
的固有偏差。若直接对标全精度实现，量化误差会被计入算子误差，无法反映实现是否正确。
因此 golden 采用**与算子相同量化方案的 float64 CPU 实现**——量化/反量化、码本查找、
fp16 scale 舍入均按算子语义复刻，只是全程用 float64 计算，残差中只剩实现误差。
"""

import unittest

import numpy as np
import torch
import torch_npu
import cann_ops_transformer  # noqa: F401  注册 turbo_quant_sparse_flash_attention

# 优先用仓内惯例的 torch_npu TestCase；该模块依赖 expecttest，缺失时退回标准 unittest，
# 使用例在未装测试依赖的环境下同样可跑。
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    TestCase = unittest.TestCase
    run_tests = unittest.main

DEVICE_ID = 0
torch_npu.npu.set_device(DEVICE_ID)

DN = 512  # latent(nope) 维度
DR = 64  # rope 维度
D = DN + DR  # query 最后一维（含 rope）
TILE = 128
NHEAD = 8
BLOCK = 128  # PageAttention block size
INT64_MAX = 9223372036854775807

# 4bit 码本，升序。相邻中心的中点构成 15 条单调边界，最近邻搜索即"数边界"。
CENT = np.array(
    [
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
    ],
    dtype=np.float32,
)

# 规范表格：FLOAT16 -> rtol=atol=2^-9，匹配率 0.99，绝对误差上限 1e-1 或 32*ULP。
# 输出为 BFLOAT16 时尾数位更少（8 bit vs 11 bit），沿用 FLOAT16 阈值属于偏严的取法。
TOL = {
    torch.bfloat16: dict(rtol=2**-9, atol=2**-9, ratio=0.99, max_abs=1e-1),
}


def pack_slot(nope_f32, rope_f32):
    """按算子的存储格式打出 KV slot: [256B nibble | 128B rope(bf16) | 2B fp16 scale]。

    同时返回 golden 需要的 yhat 与 fp16 scale，避免两侧各算一遍量化引入不一致。
    """
    ntok = nope_f32.shape[0]
    vecnorm = np.sqrt((nope_f32.astype(np.float64) ** 2).sum(1) + 1e-16)
    u = nope_f32 / vecnorm[:, None].astype(np.float32)

    bnd = ((CENT[:-1] + CENT[1:]) / 2.0).astype(np.float32)
    nib = (
        np.searchsorted(bnd, u.ravel(), side="right").reshape(ntok, DN).astype(np.uint8)
    )
    yhat = CENT[nib]
    scale = (vecnorm / np.sqrt((yhat.astype(np.float64) ** 2).sum(1) + 1e-16)).astype(
        np.float16
    )

    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)  # 低位 = 偶数维
    rope_b16 = (
        torch.from_numpy(rope_f32)
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .numpy()
    )

    slot = np.zeros((ntok, DN // 2 + DR * 2 + 2), dtype=np.uint8)
    slot[:, : DN // 2] = packed
    slot[:, DN // 2 : DN // 2 + DR * 2] = rope_b16.reshape(ntok, DR * 2)
    slot[:, -2:] = scale.view(np.uint8).reshape(ntok, 2)
    return slot, yhat.astype(np.float64), scale


def build_case(bsz, s2, topk, seed, q_dtype, dist):
    """构造一条用例。

    s2 为每个 batch 的**有效** KV 长度，允许非 block 对齐——此时按 ceil 分配 block，
    尾块只有前若干 token 有效，正好覆盖尾部处理路径。
    dist: 'normal' 或 'uniform'，对应规范中 50/50 的数据分布要求。
    """
    rng = np.random.default_rng(seed)
    blk_per_b = (s2 + BLOCK - 1) // BLOCK
    nblk = bsz * blk_per_b
    ntok = nblk * BLOCK

    if dist == "uniform":
        nope = (rng.uniform(-5.0, 5.0, (ntok, DN)) / np.sqrt(DN)).astype(np.float32)
        rope = (rng.uniform(-5.0, 5.0, (ntok, DR)) * 0.02).astype(np.float32)
        qraw = (rng.uniform(-5.0, 5.0, (bsz, NHEAD, D)) * 0.02).astype(np.float32)
    else:
        sigma = float(rng.uniform(0.1, 2.0))
        nope = (rng.normal(0.0, sigma, (ntok, DN)) / np.sqrt(DN)).astype(np.float32)
        rope = (rng.normal(0.0, sigma, (ntok, DR)) * 0.1).astype(np.float32)
        qraw = (rng.normal(0.0, sigma, (bsz, NHEAD, D)) * 0.1).astype(np.float32)

    slot, yhat, scale = pack_slot(nope, rope)
    kv = torch.from_numpy(slot.view(np.int8)).reshape(nblk, BLOCK, 1, -1).npu()

    # sparse_indices: 每个 query 从本 batch 的有效 KV 里选。topk 超过实际 KV 长度时，
    # 尾部以 -1 填充（生产语义：topk 缓冲宽度固定而 KV 长度可变）。
    # 索引是 **每 batch 局部** 的 KV 位置，全局落点由 block_table 解析；
    # golden 侧需自行加上 batch 的块偏移。
    nvalid = min(topk, s2)
    idx = np.full((bsz, topk), -1, dtype=np.int32)
    for b in range(bsz):
        idx[b, :nvalid] = rng.choice(s2, size=nvalid, replace=False).astype(np.int32)
    si = torch.from_numpy(idx).reshape(bsz, 1, -1).npu()

    q = torch.from_numpy(qraw).to(q_dtype).npu()
    bt = torch.arange(nblk, dtype=torch.int32).reshape(bsz, -1).npu()
    # TND 布局下 actual_seq_lengths 是【累积和】，不是每 batch 的长度
    asq = torch.tensor(np.cumsum([1] * bsz), dtype=torch.int32).npu()
    ask = torch.tensor(np.cumsum([s2] * bsz), dtype=torch.int32).npu()
    ks = torch.ones(nblk, BLOCK, 1, DN // TILE, dtype=torch.float32).npu()

    return dict(
        q=q,
        kv=kv,
        si=si,
        bt=bt,
        asq=asq,
        ask=ask,
        ks=ks,
        idx=idx,
        nvalid=nvalid,
        blk_per_b=blk_per_b,
        yhat=yhat,
        scale=scale,
        rope=rope,
        qraw=qraw,
        q_dtype=q_dtype,
        scale_value=1.0 / (DN**0.5),
    )


def golden(c):
    """float64 CPU 标杆，复刻算子语义。

    q 必须按算子实际收到的 dtype 做舍入：fp16 有 11 位尾数、bf16 只有 8 位，
    若一律用 bf16 建模，fp16 用例会拿一份更粗的 q 去比对，差异被误算成算子误差。
    rope 则固定按 bf16 —— 那是 KV slot 的存储格式，与 query dtype 无关。
    """
    Kn = c["yhat"] * c["scale"].astype(np.float64)[:, None]
    Kr = (
        torch.from_numpy(c["rope"])
        .to(torch.bfloat16)
        .float()
        .numpy()
        .astype(np.float64)
    )
    q = torch.from_numpy(c["qraw"]).to(c["q_dtype"]).float().numpy().astype(np.float64)
    qn, qr = q[..., :DN], q[..., DN:]

    bsz = q.shape[0]
    out = np.empty((bsz, NHEAD, DN), dtype=np.float64)
    for b in range(bsz):
        # 局部索引 -> 全局 token 下标；-1 填充位不参与计算
        sel = c["idx"][b][: c["nvalid"]].astype(np.int64) + b * c["blk_per_b"] * BLOCK
        kn, kr = Kn[sel], Kr[sel]
        sc = (qn[b] @ kn.T + qr[b] @ kr.T) * c["scale_value"]
        sc = sc - sc.max(axis=1, keepdims=True)
        p = np.exp(sc)
        p = p / p.sum(axis=1, keepdims=True)
        out[b] = p @ kn
    return out


def check(actual, expect, dtype):
    """规范的双条件判定。返回 (是否通过, 描述)。"""
    t = TOL[dtype]
    err = np.abs(actual.astype(np.float64) - expect)
    tol = t["atol"] + t["rtol"] * np.abs(expect)
    matched = float((err <= tol).mean())
    max_abs = float(err.max())
    ok = matched >= t["ratio"] and max_abs <= t["max_abs"]
    desc = "匹配率 %.4f (>=%.2f)  最大绝对误差 %.3e (<=%.1e)" % (
        matched,
        t["ratio"],
        max_abs,
        t["max_abs"],
    )
    return ok, desc


class TestTurboQuantSparseFlashAttention(TestCase):
    """覆盖：2^n 与 2^n-1 两类尺寸、均匀与正态两种分布。

    算子仅声明支持 BFLOAT16（FLOAT16 已从支持列表移除），故不再覆盖 fp16。
    """

    CASES = [
        # (bsz, s2, topk, dist)；s2 取 2^n 与 2^n-1 两类，后者非 block 对齐
        (1, 1024, 256, "normal"),
        (1, 1023, 256, "uniform"),
        (4, 2048, 512, "normal"),
        (4, 2047, 512, "uniform"),
        (8, 512, 128, "normal"),
        (8, 511, 128, "uniform"),
        # T=25：非 2 的幂且为奇数，触及分核尾部；与算子实测采集的 query shape 对齐
        (25, 1024, 256, "normal"),
        (25, 1023, 256, "uniform"),
    ]

    @staticmethod
    def _call(c, with_scale=True):
        """with_scale=False 时不传两个 dequant scale 可选输入。"""
        scale = (
            {"key_dequant_scale": c["ks"], "value_dequant_scale": c["ks"]}
            if with_scale
            else {}
        )
        out, _, _ = torch.ops.cann_ops_transformer.turbo_quant_sparse_flash_attention(
            c["q"],
            c["kv"],
            c["kv"],
            c["si"],
            **scale,
            block_table=c["bt"],
            actual_seq_lengths_query=c["asq"],
            actual_seq_lengths_kv=c["ask"],
            scale_value=c["scale_value"],
            key_quant_mode=3,
            value_quant_mode=3,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=INT64_MAX,
            next_tokens=INT64_MAX,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=TILE,
            rope_head_dim=DR,
            return_softmax_lse=False,
        )
        torch.npu.synchronize()
        return out

    def _run_one(self, bsz, s2, topk, dist, dtype, seed):
        c = build_case(bsz, s2, topk, seed, dtype, dist)
        out = self._call(c)
        actual = out.float().cpu().numpy().reshape(bsz, NHEAD, DN)
        ok, desc = check(actual, golden(c), dtype)
        tag = "bsz=%d s2=%-5d topk=%-4d %-7s %s" % (
            bsz,
            s2,
            topk,
            dist,
            str(dtype).split(".")[-1],
        )
        print("  [%s] %s  %s" % ("PASS" if ok else "FAIL", tag, desc), flush=True)
        self.assertTrue(ok, "%s: %s" % (tag, desc))

    def test_precision_bf16(self):
        for i, (bsz, s2, topk, dist) in enumerate(self.CASES):
            with self.subTest(case=i):
                self._run_one(bsz, s2, topk, dist, torch.bfloat16, seed=1000 + i)

    def test_topk_exceeds_actual_kv(self):
        """topk 缓冲宽度 >= 实际 KV 长度的场景，sparse_indices 尾部为 -1 填充。

        该区间历史上触发过尾部处理缺陷（非 32B 对齐的向量写），生产中短 prompt 与
        图捕获 warmup 必经，需单独覆盖。
        """
        for i, (s2, topk) in enumerate([(128, 256), (256, 256), (100, 512)]):
            with self.subTest(case=i):
                self._run_one(1, s2, topk, "normal", torch.bfloat16, seed=3000 + i)

    def test_dequant_scale_not_consumed(self):
        """quant_scale_repo_mode=1（COMBINE）下反量化系数存放于 KV slot 内。

        此时 key_dequant_scale 与 value_dequant_scale 两个可选输入不被 kernel 消费，
        不传应当成功，且结果与传入时逐位相同。
        """
        c = build_case(4, 512, 256, 4000, torch.bfloat16, "normal")
        with_scale = self._call(c, with_scale=True).cpu()
        without_scale = self._call(c, with_scale=False).cpu()
        same = torch.equal(with_scale, without_scale)
        print(
            "  [%s] 不传 dequant scale 结果逐位相同" % ("PASS" if same else "FAIL"),
            flush=True,
        )
        self.assertTrue(same, "不传 dequant scale 时输出发生变化")


if __name__ == "__main__":
    run_tests()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mla_prolog_v5 pybind 接口测试：NPU 实际运行 + CPU golden 全量比对。

关键输入构造契约（与 kernel 实现一致，详见 golden.py 模块 docstring）：
- weight_uq_qr 按头交错：[Qc_h(D) | Qr_h(Dr)] × N
- weight_dkv_kr 分离：[W_dkv(Hckv) | W_kr(Dr)]
- rope_sin/rope_cos 形状 (T, Dr)（3 维输入为 (B,S,Dr)），前后半重复布局

运行前置：
  bash build.sh --soc=ascend910b --install      # 编译+安装算子包
  source ~/Ascend/opp/vendors/mla_prolog_v5_custom/bin/set_env.bash
  bash torch_extension/build.sh                  # 编译 pybind 扩展

运行：python3 test.py
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = (
    True  # 允许创建 FRACTAL_NZ 等 internal format tensor
)

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "torch_extension"
    ),
)
from mla_prolog_v5 import mla_prolog  # noqa: E402
from golden import mla_prolog_golden  # noqa: E402

torch.manual_seed(2026)

# 典型 DeepSeek MLA 形参
HE = 7168
HCQ = 1536
N = 16
D = 128
HCKV = 512
DR = 64
NKV = 1
BLOCK_SIZE = 128
# bf16 可精确表示的 cache 初始标记值（避免 bf16 舍入导致校验误报）
CACHE_INIT_MARK = -12352.0

# bf16 输出比对容差（kernel bf16 计算 vs golden fp32）
RTOL = 2e-2
ATOL = 5e-2


def compare(name: str, out: torch.Tensor, ref: torch.Tensor) -> bool:
    out_f = out.float().cpu()
    ref_f = ref.float().cpu()
    diff = (out_f - ref_f).abs()
    tol = ATOL + RTOL * ref_f.abs()
    bad = diff > tol
    num_bad = int(bad.sum())
    total = ref_f.numel()
    max_abs = float(diff.max()) if total else 0.0
    ok = num_bad == 0
    status = "PASS" if ok else "FAIL"
    print(
        f"    [{status}] {name}: shape={tuple(out.shape)}, max_abs_err={max_abs:.6f}, bad={num_bad}/{total}"
    )
    if not ok:
        idx = bad.nonzero()[:3]
        for i in idx:
            t_i = tuple(i.tolist())
            print(
                f"        first mismatch at {t_i}: out={out_f[t_i]:.6f}, ref={ref_f[t_i]:.6f}"
            )
    return ok


def gen_sin_cos(t: int, device):
    """生成前后半重复布局的 rope sin/cos（(T, Dr)，bf16 设备张量 + fp32 CPU 参照）。"""
    theta = torch.arange(DR // 2, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (2.0 * theta / DR))
    pos = torch.arange(t, dtype=torch.float32).unsqueeze(1)
    s_half = torch.sin(pos * inv)  # (T, Dr/2)
    c_half = torch.cos(pos * inv)
    sin_cpu = torch.cat([s_half, s_half], dim=-1)
    cos_cpu = torch.cat([c_half, c_half], dim=-1)
    sin = sin_cpu.to(device).to(torch.bfloat16)
    cos = cos_cpu.to(device).to(torch.bfloat16)
    return sin, cos, sin_cpu, cos_cpu


def make_nz(weight_2d: torch.Tensor, device):
    """CPU 2D 权重 -> NPU FRACTAL_NZ tensor。"""
    t = weight_2d.to(device).to(torch.bfloat16).contiguous()
    return torch_npu.npu_format_cast(t, 29)  # 29 = ACL_FORMAT_FRACTAL_NZ


def gen_weights(n: int, d: int):
    """生成 CPU fp32 权重。w_uq_qr 按头交错 [Qc_h|Qr_h]；w_dkv_kr 分离 [W_dkv|W_kr]。"""
    w_dq = torch.randn(HE, HCQ) * (1.0 / HE**0.5)
    wuq = torch.randn(HCQ, n, d) * (1.0 / HCQ**0.5)
    wqr = torch.randn(HCQ, n, DR) * (1.0 / HCQ**0.5)
    w_uq_qr = torch.cat([wuq, wqr], dim=2).view(HCQ, n * (d + DR)).contiguous()
    w_uk = torch.randn(n, d, HCKV) * (1.0 / d**0.5)
    w_dkv_kr = torch.randn(HE, HCKV + DR) * (1.0 / HE**0.5)
    return w_dq, w_uq_qr, w_uk, w_dkv_kr


def run_case(
    name,
    *,
    token_shape,
    cache_mode,
    do_rope=True,
    query_norm_flag=True,
    alpha_q=1.0,
    alpha_kv=1.0,
    block_size=BLOCK_SIZE,
):
    print(
        f"\n== case: {name} (cache_mode={cache_mode}, token={token_shape}, do_rope={do_rope}) =="
    )
    device = "npu"
    fused = len(token_shape) == 2  # BS 合轴 (T, He)
    t_len = token_shape[0] if fused else token_shape[0] * token_shape[1]
    b, s = (1, token_shape[0]) if fused else (token_shape[0], token_shape[1])

    # ---------- CPU golden 输入（fp32）----------
    x_cpu = torch.randn(t_len, HE, dtype=torch.float32) * 0.5
    w_dq_cpu, w_uq_qr_cpu, w_uk_cpu, w_dkv_kr_cpu = gen_weights(N, D)
    gamma_cq_cpu = torch.ones(HCQ, dtype=torch.float32)
    gamma_ckv_cpu = torch.ones(HCKV, dtype=torch.float32)

    sin_cpu = cos_cpu = None
    if do_rope:
        _, _, sin_cpu, cos_cpu = gen_sin_cos(t_len, "cpu")

    golden = mla_prolog_golden(
        x_cpu,
        w_dq_cpu,
        w_uq_qr_cpu,
        w_uk_cpu,
        w_dkv_kr_cpu,
        gamma_cq_cpu,
        gamma_ckv_cpu,
        sin_cpu,
        cos_cpu,
        n=N,
        d=D,
        hckv=HCKV,
        dr=DR,
        alpha_q=alpha_q,
        alpha_kv=alpha_kv,
        do_rope=do_rope,
    )

    # ---------- NPU 输入 ----------
    token_x = x_cpu.to(device).to(torch.bfloat16).reshape(token_shape).contiguous()
    weight_dq = make_nz(w_dq_cpu, device)
    weight_uq_qr = make_nz(w_uq_qr_cpu, device)
    weight_dkv_kr = make_nz(w_dkv_kr_cpu, device)
    weight_uk = w_uk_cpu.to(device).to(torch.bfloat16).contiguous()
    gamma_cq = gamma_cq_cpu.to(device).to(torch.bfloat16)
    gamma_ckv = gamma_ckv_cpu.to(device).to(torch.bfloat16)
    rope_sin = rope_cos = None
    if do_rope:
        sin_dev, cos_dev, _, _ = gen_sin_cos(t_len, device)
        if fused:
            rope_sin, rope_cos = sin_dev, cos_dev
        else:
            rope_sin = sin_dev.view(b, s, DR)
            rope_cos = cos_dev.view(b, s, DR)

    # cache
    cache_index = None
    if cache_mode == "TND":
        kv_cache = (
            torch.full((t_len, NKV, HCKV), CACHE_INIT_MARK, dtype=torch.float32)
            .to(torch.bfloat16)
            .to(device)
        )
        kr_cache = (
            torch.full((t_len, NKV, DR), CACHE_INIT_MARK, dtype=torch.float32)
            .to(torch.bfloat16)
            .to(device)
        )
    else:  # PA_BSND
        kv_cache = (
            torch.full((b, block_size, NKV, HCKV), CACHE_INIT_MARK, dtype=torch.float32)
            .to(torch.bfloat16)
            .to(device)
        )
        kr_cache = (
            torch.full((b, block_size, NKV, DR), CACHE_INIT_MARK, dtype=torch.float32)
            .to(torch.bfloat16)
            .to(device)
        )
        if fused:
            cache_index = torch.arange(t_len, dtype=torch.int64, device=device)
        else:
            cache_index = torch.arange(b * s, dtype=torch.int64, device=device).view(
                b, s
            )

    # ---------- 调用 pybind 接口 ----------
    query, query_rope, _, query_norm, _ = mla_prolog(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        gamma_cq,
        gamma_ckv,
        kv_cache,
        kr_cache,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        cache_index=cache_index,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
        qc_qr_scale=alpha_q,
        kc_scale=alpha_kv,
    )
    torch.npu.synchronize()

    # ---------- 全量比对 ----------
    ok = True
    ok &= compare("query", query, golden["query"].view(query.shape))
    ok &= compare("query_rope", query_rope, golden["query_rope"].view(query_rope.shape))
    if query_norm_flag:
        ok &= compare(
            "query_norm", query_norm, golden["query_norm"].view(query_norm.shape)
        )

    kv_npu = kv_cache.float().cpu()
    kr_npu = kr_cache.float().cpu()
    if cache_mode == "TND":
        ok &= compare("kv_cache", kv_npu.view(t_len, HCKV), golden["c_kv"])
        ok &= compare("kr_cache", kr_npu.view(t_len, DR), golden["k_r"])
    else:
        kv_flat = kv_npu.view(-1, HCKV)
        kr_flat = kr_npu.view(-1, DR)
        idx = cache_index.long().cpu().view(-1)
        ok &= compare("kv_cache[slots]", kv_flat[idx], golden["c_kv"])
        ok &= compare("kr_cache[slots]", kr_flat[idx], golden["k_r"])
        # 未写入槽位应保持初始标记（bf16 精确表示值）
        mask = torch.ones(kv_flat.shape[0], dtype=torch.bool)
        mask[idx] = False
        untouched_ok = bool((kv_flat[mask] == CACHE_INIT_MARK).all()) and bool(
            (kr_flat[mask] == CACHE_INIT_MARK).all()
        )
        print(
            f"    [{'PASS' if untouched_ok else 'FAIL'}] untouched slots keep init value "
            f"({int(mask.sum())} slots)"
        )
        ok &= untouched_ok
    return ok


def main():
    results = {}
    results["TND_bf16_fused"] = run_case(
        "TND_bf16_fused", token_shape=(128, HE), cache_mode="TND"
    )
    results["PA_BSND_bf16_fused"] = run_case(
        "PA_BSND_bf16_fused", token_shape=(64, HE), cache_mode="PA_BSND"
    )
    results["PA_BSND_bf16_3d"] = run_case(
        "PA_BSND_bf16_3d", token_shape=(2, 32, HE), cache_mode="PA_BSND"
    )
    results["PA_BSND_bf16_no_rope"] = run_case(
        "PA_BSND_bf16_no_rope",
        token_shape=(64, HE),
        cache_mode="PA_BSND",
        do_rope=False,
    )
    results["PA_BSND_bf16_scale"] = run_case(
        "PA_BSND_bf16_scale",
        token_shape=(64, HE),
        cache_mode="PA_BSND",
        alpha_q=1.3,
        alpha_kv=0.7,
    )

    print("\n================ summary ================")
    all_ok = True
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
        all_ok &= v
    print(
        f"TOTAL: {'PASS' if all_ok else 'FAIL'} ({sum(results.values())}/{len(results)})"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

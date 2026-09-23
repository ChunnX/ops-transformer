# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MhcPreSinkhornPremix torch 接口功能测试（experimental）。

覆盖场景：
1. 单算子模式（eager 调用 aclnn）：premix 传/不传 x outFlag True/False x TND/BSND。
2. aclgraph 模式（torch_npu NPUGraph 捕获-回放）。

用法：
    python3 test_mhc_pre_sinkhorn_premix_torch.py            # 全部用例
    pytest test_mhc_pre_sinkhorn_premix_torch.py             # pytest 方式
"""

import numpy as np
import torch
import torch_npu  # noqa: F401

try:
    import cann_ops_transformer_customize  # noqa: F401 单算子包：注册 torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_premix
except ImportError:
    import cann_ops_transformer  # noqa: F401 全量包

DEVICE = "npu:0"
HC_MULT = 4
NUM_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6

# 对比容差（rtol, atol）。随机大 K 维输入下 fp32 累加顺序差异会放大到 1e-3 量级，
# 严格容差（ST 的 1e-4）需配合 ST 的固定小数值输入；此处为功能验证，误差远小于功能性错误量级
TOL_BF16 = (2e-2, 2e-2)
TOL_FP32 = (5e-3, 1e-3)


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -80.0, 80.0)))


def golden(
    x_np,
    phi,
    alpha,
    bias,
    premix=None,
    hc_mult=HC_MULT,
    num_iters=NUM_ITERS,
    hc_eps=HC_EPS,
    norm_eps=NORM_EPS,
    need_backward=True,
):
    """numpy 参考实现。premix 传入时 hin 使用 premix 作为加权系数，其余输出不变。"""
    # kernel 读入的是 bf16 舍入后的 x，golden 同步舍入
    x_f32 = torch.from_numpy(x_np.astype(np.float32)).to(torch.bfloat16).float().numpy()
    squeeze = False
    if x_f32.ndim == 3:
        x_f32 = x_f32[None, ...]
        squeeze = True
    b, s, n, d = x_f32.shape
    flat_x = x_f32.reshape(b, s, n * d)

    inv_rms = 1.0 / np.sqrt(np.mean(flat_x * flat_x, axis=-1, keepdims=True) + norm_eps)
    hc_before_norm = np.matmul(flat_x, phi.T)
    normalized = hc_before_norm * inv_rms

    h_pre = sigmoid(normalized[..., :hc_mult] * alpha[0] + bias[:hc_mult]) + hc_eps
    h_post = 2.0 * sigmoid(
        normalized[..., hc_mult : 2 * hc_mult] * alpha[1] + bias[hc_mult : 2 * hc_mult]
    )

    residual_logits = (
        normalized[..., 2 * hc_mult :] * alpha[2] + bias[2 * hc_mult :]
    ).reshape(b, s, hc_mult, hc_mult)
    sum_out = np.empty((2 * num_iters, b, s, hc_mult), dtype=np.float32)
    norm_out = np.empty((2 * num_iters, b, s, hc_mult, hc_mult), dtype=np.float32)
    row_max = np.max(residual_logits, axis=-1, keepdims=True)
    row_exp = np.exp(residual_logits - row_max)
    row_sum = np.sum(row_exp, axis=-1)
    sum_out[0] = row_sum + hc_eps
    current = row_exp / row_sum[..., None] + hc_eps
    norm_out[0] = current
    column_sum = np.sum(current, axis=-2)
    sum_out[1] = column_sum + hc_eps
    current = current / (column_sum[..., None, :] + hc_eps)
    norm_out[1] = current
    for it in range(1, num_iters):
        row_sum = np.sum(current, axis=-1)
        sum_out[2 * it] = row_sum + hc_eps
        current = current / (row_sum[..., :, None] + hc_eps)
        norm_out[2 * it] = current
        column_sum = np.sum(current, axis=-2)
        sum_out[2 * it + 1] = column_sum + hc_eps
        current = current / (column_sum[..., None, :] + hc_eps)
        norm_out[2 * it + 1] = current
    h_res = current.reshape(b, s, hc_mult * hc_mult)

    # premix 传入时 hin 直接使用 premix 作为权重（不加 hc_eps），否则用内部 h_pre
    mix_for_y = premix.astype(np.float32) if premix is not None else h_pre
    if squeeze and premix is not None and mix_for_y.ndim == 2:
        mix_for_y = mix_for_y[None, ...]
    hin = np.sum(x_f32 * mix_for_y[..., :, None], axis=-2)
    # hin 在 kernel 内以 fp32 累加后舍入为 bf16，golden 同步做 bf16 舍入
    hin = torch.from_numpy(hin).to(torch.bfloat16).float().numpy()

    if squeeze:
        hin, h_post, h_res = hin[0], h_post[0], h_res[0]
        h_pre, hc_before_norm, inv_rms = h_pre[0], hc_before_norm[0], inv_rms[0]
        sum_out = sum_out[:, 0]
        norm_out = norm_out[:, 0]

    if not need_backward:
        empty = np.empty((0,), dtype=np.float32)
        h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            empty,
            empty,
            empty,
            empty,
            empty,
        )
    return (
        hin.astype(np.float32),
        h_post.astype(np.float32),
        h_res.astype(np.float32),
        h_pre.astype(np.float32),
        hc_before_norm.astype(np.float32),
        inv_rms.astype(np.float32),
        sum_out,
        norm_out,
    )


def make_inputs(t_or_bs, d=4096, is_tnd=True, seed=0, with_premix=False):
    rng = np.random.default_rng(seed)
    n = HC_MULT
    hc_mix = n * n + 2 * n
    if is_tnd:
        x_shape = (t_or_bs, n, d)
        premix_shape = (t_or_bs, n)
    else:
        x_shape = (t_or_bs, 2, n, d)
        premix_shape = (t_or_bs, 2, n)
    x = (rng.standard_normal(x_shape) * 0.5).astype(np.float32)
    phi = (rng.standard_normal((hc_mix, n * d)) / np.sqrt(n * d)).astype(np.float32)
    alpha = (rng.standard_normal(3) * 0.5 + 1.0).astype(np.float32)
    bias = (rng.standard_normal(hc_mix) * 0.5).astype(np.float32)
    # premix 故意取与内部 h_pre 分布不同的随机值，以验证 y 确实使用 premix
    premix = rng.random(premix_shape).astype(np.float32) * 2.0 if with_premix else None
    return x, phi, alpha, bias, premix


def to_npu(x, phi, alpha, bias, premix):
    x_t = torch.from_numpy(x).to(torch.bfloat16).npu()
    phi_t = torch.from_numpy(phi).npu()
    alpha_t = torch.from_numpy(alpha).npu()
    bias_t = torch.from_numpy(bias).npu()
    premix_t = torch.from_numpy(premix).npu() if premix is not None else None
    return x_t, phi_t, alpha_t, bias_t, premix_t


def run_op(x_t, phi_t, alpha_t, bias_t, premix_t, out_flag):
    return torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_premix(
        x_t,
        phi_t,
        alpha_t,
        bias_t,
        premix_t,
        HC_MULT,
        NUM_ITERS,
        HC_EPS,
        NORM_EPS,
        out_flag,
    )


def assert_close(name, actual, expected, rtol, atol):
    actual_np = actual.cpu().float().numpy()
    assert actual_np.shape == expected.shape, (
        f"{name} shape {actual_np.shape} != {expected.shape}"
    )
    diff = np.abs(actual_np - expected)
    thresh = atol + rtol * np.abs(expected)
    assert (diff <= thresh).all(), (
        f"{name} mismatch: max_abs={diff.max()}, max_rel={(diff / np.maximum(np.abs(expected), 1e-6)).max()}"
    )


def check_case(t_or_bs, is_tnd, with_premix, out_flag, seed=0, d=4096):
    x, phi, alpha, bias, premix = make_inputs(t_or_bs, d, is_tnd, seed, with_premix)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix)
    outs = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, out_flag)
    expected = golden(x, phi, alpha, bias, premix, need_backward=out_flag)
    names = [
        "hin",
        "h_post",
        "h_res",
        "h_pre",
        "hc_before_norm",
        "inv_rms",
        "sum_out",
        "norm_out",
    ]
    check_names = names if out_flag else names[:3]
    for name, act, exp in zip(names, outs, expected):
        if name not in check_names:
            assert act.numel() == 0, f"{name} should be empty when out_flag=False"
            continue
        rtol, atol = TOL_BF16 if name == "hin" else TOL_FP32
        assert_close(name, act, exp, rtol, atol)
    tag = f"t={t_or_bs}" if is_tnd else f"bs={t_or_bs}"
    print(
        f"[PASS] single-op {tag} is_tnd={is_tnd} premix={with_premix} out_flag={out_flag}"
    )


def test_single_op():
    # K 分核路径（小 bs）与 M 分核路径（bs >= 256 * aicCoreNum，如 t=8192）均覆盖
    for t, is_tnd in [(16, True), (8, False), (256, True), (64, False), (8192, True)]:
        for with_premix in (False, True):
            for out_flag in (False, True):
                check_case(
                    t,
                    is_tnd,
                    with_premix,
                    out_flag,
                    seed=hash((t, is_tnd, with_premix, out_flag)) % 1000,
                )


def test_premix_changes_hin():
    # 同一份输入，传/不传 premix 时 hin 必须不同（验证 premix 真正生效）
    x, phi, alpha, bias, premix = make_inputs(32, is_tnd=True, seed=7, with_premix=True)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix)
    hin_no, _, _, h_pre_no, *_ = run_op(x_t, phi_t, alpha_t, bias_t, None, True)
    hin_pm, _, _, h_pre_pm, *_ = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    diff = (hin_no.float() - hin_pm.float()).abs().max().item()
    assert diff > 1e-3, f"premix not effective, hin diff={diff}"
    # 传 premix 时内部 h_pre 输出应保持不变
    assert_close("h_pre", h_pre_pm, h_pre_no.cpu().float().numpy(), 0.0, 0.0)
    print("[PASS] premix affects hin and keeps internal h_pre output")


def test_aclgraph():
    t = 32
    x, phi, alpha, bias, premix = make_inputs(t, is_tnd=True, seed=11, with_premix=True)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix)

    # eager 参考结果
    ref = [
        o.cpu().float().numpy()
        for o in run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    ]

    # aclgraph 捕获-回放
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):  # warmup
            run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    torch.npu.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outs = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    graph.replay()
    torch.npu.synchronize()

    names = [
        "hin",
        "h_post",
        "h_res",
        "h_pre",
        "hc_before_norm",
        "inv_rms",
        "sum_out",
        "norm_out",
    ]
    for name, act, exp in zip(names, graph_outs, ref):
        rtol, atol = TOL_BF16 if name == "hin" else TOL_FP32
        assert_close(f"aclgraph:{name}", act, exp, rtol, atol)

    # 修改输入后回放，验证结果随输入更新
    x2, phi2, alpha2, bias2, premix2 = make_inputs(
        t, is_tnd=True, seed=22, with_premix=True
    )
    x_t.copy_(torch.from_numpy(x2).to(torch.bfloat16))
    phi_t.copy_(torch.from_numpy(phi2))
    alpha_t.copy_(torch.from_numpy(alpha2))
    bias_t.copy_(torch.from_numpy(bias2))
    premix_t.copy_(torch.from_numpy(premix2))
    graph.replay()
    torch.npu.synchronize()
    expected2 = golden(x2, phi2, alpha2, bias2, premix2, need_backward=True)
    for name, act, exp in zip(names, graph_outs, expected2):
        rtol, atol = TOL_BF16 if name == "hin" else TOL_FP32
        assert_close(f"aclgraph-replay:{name}", act, exp, rtol, atol)
    print("[PASS] aclgraph capture/replay with premix + backward outputs")


def test_aclgraph_no_premix():
    t = 16
    x, phi, alpha, bias, _ = make_inputs(t, is_tnd=True, seed=33, with_premix=False)
    x_t, phi_t, alpha_t, bias_t, _ = to_npu(x, phi, alpha, bias, None)
    ref = [
        o.cpu().float().numpy()
        for o in run_op(x_t, phi_t, alpha_t, bias_t, None, False)
    ]

    torch.npu.synchronize()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            run_op(x_t, phi_t, alpha_t, bias_t, None, False)
    torch.npu.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outs = run_op(x_t, phi_t, alpha_t, bias_t, None, False)
    graph.replay()
    torch.npu.synchronize()
    for name, act, exp in zip(["hin", "h_post", "h_res"], graph_outs[:3], ref[:3]):
        rtol, atol = TOL_BF16 if name == "hin" else TOL_FP32
        assert_close(f"aclgraph-nopremix:{name}", act, exp, rtol, atol)
    print("[PASS] aclgraph capture/replay without premix")


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    test_single_op()
    test_premix_changes_hin()
    test_aclgraph()
    test_aclgraph_no_premix()
    print("ALL TESTS PASSED")

"""Device-run functional gate (TC-01..TC-10) for the installed SageAttention2 aclnn op.

驱动链路：`import sageattention2` 公开 API（sageattn / sageattn_qk_int8_pv_fp8_asc /
monkey_patch）→ `torch.ops.sageattention2.forward` PrivateUse1 后端 → C++ 桥接
（`./python/sageattention2/csrc/attention.cpp`）→ 两段式 `aclnnSageAttention2` 真机算子。

用例矩阵对照任务书《功能验收用例》TC-01..TC-10，逐条对应 test_tc01..test_tc10。
精度判据为任务书"精度要求"的 L-B 风格混合容差（对 FP32 SDPA 参考逐元素比较）：
FP16 rtol=atol=2^-9、BF16 rtol=atol=2^-6，matched_ratio >= 0.99，
max_abs 上限 FP16 1e-1 / BF16 1e-0；LSE 另按 FP32 容差校验（含 smooth_k 校正语义）。

真机运行（算子目录执行）：

    SAGEATTN_TEST_NPU=1 python3 -m pytest tests/test_npu_standalone.py -v

未设 `SAGEATTN_TEST_NPU=1`、或无 torch/torch_npu/NPU 设备时整个模块 pytest.skip，
宿主机 CI 冒烟直接干净通过，不触发任何设备调用与扩展编译。
参考实现自包含（FP32 分块 softmax 注意力与 logsumexp），不导入 tests/_golden_utils.py。
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
import signal
import sys
import time
import unittest
import warnings
from pathlib import Path

import pytest

@contextmanager
def npu_timeout_guard(seconds: int = 20, name: str = "case", extra_info: str = ""):
    """Watchdog context manager protecting against NPU hardware hangs/deadlocks.

    Uses POSIX signal.alarm to raise a descriptive TimeoutError with test case
    identification and input parameters if execution exceeds `seconds`.
    Supports nesting by tracking and restoring remaining alarm timers.
    """
    if not hasattr(signal, "alarm") or not hasattr(signal, "SIGALRM"):
        yield
        return

    sec_int = max(1, int(seconds))
    start_time = time.monotonic()

    def _timeout_handler(signum, frame):
        detail = f" [参数: {extra_info}]" if extra_info else ""
        msg = (
            f"[NPU-TIMEOUT] {name} 执行超过 {sec_int} 秒！"
            f"可能存在硬件死锁，已被看门狗中断！{detail}"
        )
        raise TimeoutError(msg)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    prev_remaining = signal.alarm(sec_int)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if prev_remaining > 0:
            elapsed = int(time.monotonic() - start_time)
            new_remaining = max(1, prev_remaining - elapsed)
            signal.alarm(new_remaining)


if os.environ.get("SAGEATTN_TEST_NPU", "").strip().lower() not in ("1", "true", "yes"):
    raise unittest.SkipTest("SAGEATTN_TEST_NPU=1 required (on-device TC suite); skipping on host")

try:
    import torch
    import torch.nn.functional as F
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False (no NPU device attached)")

    OPS_PYTHON = Path(__file__).resolve().parents[1] / "python"
    if str(OPS_PYTHON) not in sys.path:
        sys.path.insert(0, str(OPS_PYTHON))
    import sageattention2 as sage2
except (ImportError, RuntimeError) as err:
    raise unittest.SkipTest(f"torch / torch_npu / sageattention2 unavailable: {err}")

DEVICE = "npu"


@pytest.fixture(autouse=True)
def case_timeout_watchdog(request):
    """Fallback watchdog fixture ensuring each test finishes within timeout."""
    test_name = request.node.name
    timeout_sec = int(os.environ.get("SAGE_CASE_TIMEOUT", "20"))
    with npu_timeout_guard(seconds=timeout_sec, name=test_name, extra_info="test-level fallback"):
        yield



# 任务书"精度要求"混合容差表（输出 FP16/BF16 时）
MIXED_TOL = {torch.float16: 2**-9, torch.bfloat16: 2**-6}
MIXED_MAX_ABS = {torch.float16: 1e-1, torch.bfloat16: 1e-0}
MATCHED_RATIO_FLOOR = 0.99


def make_qkv(
    b: int,
    hq: int,
    hkv: int,
    sq: int,
    sk: int,
    d: int,
    dtype: torch.dtype,
    layout: str = "HND",
    seed: int = 0,
):
    """生成 HND/NHD 布局的 q/k/v（CPU 定序生成后搬入 NPU，保证逐次可复现）。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, hq, sq, d, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
    k = torch.randn(b, hkv, sk, d, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
    v = torch.randn(b, hkv, sk, d, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
    if layout == "NHD":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return q, k, v


def to_hnd(t: torch.Tensor, layout: str) -> torch.Tensor:
    return t if layout == "HND" else t.transpose(1, 2)


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   is_causal: bool = False, sm_scale: float | None = None,
                   chunk: int = 1024) -> torch.Tensor:
    """FP32 SDPA 等价参考（HND 域）：按 Query 行分块以约束峰值显存，
    GQA 经 repeat_interleave 展开；数值即 fp32 上采样的 softmax(QK^T * s)V。"""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    rep = q.shape[1] // k.shape[1]
    kf = k.float().repeat_interleave(rep, dim=1) if rep > 1 else k.float()
    vf = v.float().repeat_interleave(rep, dim=1) if rep > 1 else v.float()
    outs = []
    for start in range(0, q.shape[2], chunk):
        qf = q[:, :, start:start + chunk, :].float()
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * sm_scale
        if is_causal:
            qi = torch.arange(start, start + qf.shape[2], device=q.device).view(1, 1, -1, 1)
            ki = torch.arange(k.shape[2], device=q.device).view(1, 1, 1, -1)
            scores = scores.masked_fill(qi < ki, float("-inf"))
        outs.append(torch.matmul(torch.softmax(scores, dim=-1), vf))
    return torch.cat(outs, dim=2)


def lse_reference(q: torch.Tensor, k: torch.Tensor, is_causal: bool = False,
                  sm_scale: float | None = None, chunk: int = 1024) -> torch.Tensor:
    """自然域 LSE 参考（HND 域）：logsumexp(QK^T * s)，对原始（未 smooth）K 计算。

    算子返回值语义（对齐开源 core.py 返回路径）：smooth_k 的 km 平移在 kernel 内
    进入 softmax 分母，出口以 lse_correction * sm_scale 精确补偿，故对 smooth_k
    两种取值，返回 LSE 都应等于对原始 K 的逐行 logsumexp。
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    rep = q.shape[1] // k.shape[1]
    kf = k.float().repeat_interleave(rep, dim=1) if rep > 1 else k.float()
    outs = []
    for start in range(0, q.shape[2], chunk):
        qf = q[:, :, start:start + chunk, :].float()
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * sm_scale
        if is_causal:
            qi = torch.arange(start, start + qf.shape[2], device=q.device).view(1, 1, -1, 1)
            ki = torch.arange(k.shape[2], device=q.device).view(1, 1, 1, -1)
            scores = scores.masked_fill(qi < ki, float("-inf"))
        outs.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outs, dim=2)


def assert_mixed(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, ctx: str) -> None:
    """L-B 风格混合容差判定：rtol/atol 逐元素 + matched_ratio + max_abs 上限。"""
    assert out.shape == ref.shape, f"{ctx}: out shape {tuple(out.shape)} != ref {tuple(ref.shape)}"
    assert out.dtype == dtype, f"{ctx}: out dtype {out.dtype} != {dtype}"
    tol = MIXED_TOL[dtype]
    diff = (out.float() - ref.float()).abs()
    matched = (diff <= tol + tol * ref.float().abs()).float().mean().item()
    max_abs = diff.max().item()
    assert matched >= MATCHED_RATIO_FLOOR, f"{ctx}: matched_ratio {matched:.5f} < {MATCHED_RATIO_FLOOR} (max_abs={max_abs:.4g})"
    assert max_abs <= MIXED_MAX_ABS[dtype], f"{ctx}: max_abs {max_abs:.4g} > {MIXED_MAX_ABS[dtype]}"


def assert_lse(lse: torch.Tensor, ref: torch.Tensor, ctx: str) -> None:
    """LSE 判定：shape [B,Hq,Sq]、FP32、有限值，且与 FP32 参考的误差在量化噪声容差内。"""
    assert lse.shape == ref.shape, f"{ctx}: lse shape {tuple(lse.shape)} != {tuple(ref.shape)}"
    assert lse.dtype == torch.float32, f"{ctx}: lse dtype {lse.dtype} != torch.float32"
    assert bool(torch.isfinite(lse).all()), f"{ctx}: lse has non-finite values"
    err = (lse - ref).abs()
    assert err.mean().item() <= 0.02, f"{ctx}: mean |lse err| {err.mean().item():.4g} > 0.02"
    assert err.max().item() <= 0.15, f"{ctx}: max |lse err| {err.max().item():.4g} > 0.15"


def test_tc01_sageattn_dispatch_fp16_bf16_hnd_nhd():
    """TC-01：sageattn 自动分发到 FP8 PV，FP16/BF16 × HND/NHD 全部可跑且对齐参考。"""
    seed = 100
    for dtype in (torch.float16, torch.bfloat16):
        for layout in ("HND", "NHD"):
            seed += 1
            info = f"shape=[2, 4, 256, 128], dtype={dtype}, layout={layout}, is_causal=False"
            with npu_timeout_guard(seconds=20, name="test_tc01", extra_info=info):
                q, k, v = make_qkv(2, 4, 4, 256, 256, 128, dtype, layout, seed)
                out = sage2.sageattn(q, k, v, tensor_layout=layout)
                ref = sdpa_reference(to_hnd(q, layout), to_hnd(k, layout), to_hnd(v, layout))
                if layout == "NHD":
                    ref = ref.transpose(1, 2)
                assert_mixed(out, ref, dtype, f"TC-01 {dtype}/{layout}")


def test_tc02_explicit_api_gran_x_accum():
    """TC-02：sageattn_qk_int8_pv_fp8_asc 的 per_thread/per_warp × fp32/fp32+fp32/fp32+fp16。"""
    seed = 200
    for gran in ("per_thread", "per_warp"):
        for accum in ("fp32", "fp32+fp32", "fp32+fp16"):
            seed += 1
            info = f"shape=[1, 4, 256, 128], dtype=float16, layout=HND, gran={gran}, accum={accum}"
            with npu_timeout_guard(seconds=20, name="test_tc02", extra_info=info):
                q, k, v = make_qkv(1, 4, 4, 256, 256, 128, torch.float16, "HND", seed)
                out = sage2.sageattn_qk_int8_pv_fp8_asc(
                    q, k, v, qk_quant_gran=gran, pv_accum_dtype=accum
                )
                assert_mixed(out, sdpa_reference(q, k, v), torch.float16, f"TC-02 {gran}/{accum}")


def test_tc03_smooth_k_smooth_v_warning_behavior():
    """TC-03：smooth_v 在两级累加档 warning 后忽略；fp32 档真实生效不告警；smooth_k 开关合法。"""
    info = "shape=[1, 4, 256, 128], dtype=float16, layout=HND, smooth_v/smooth_k behavior"
    with npu_timeout_guard(seconds=20, name="test_tc03", extra_info=info):
        q, k, v = make_qkv(1, 4, 4, 256, 256, 128, torch.float16, "HND", 300)
        with pytest.warns(UserWarning, match="smooth_v will be ignored"):
            sage2.sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp16", smooth_v=True)
        with pytest.warns(UserWarning, match="smooth_v will be ignored"):
            sage2.sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp32", smooth_v=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out_v = sage2.sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32", smooth_v=True)
        assert torch.isfinite(out_v).all()

        out = sage2.sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=False)
        assert_mixed(out, sdpa_reference(q, k, v), torch.float16, "TC-03 smooth_k=False")


def test_tc04_return_lse_shape_and_value():
    """TC-04：return_lse 形状 [B,Hq,Sq]/FP32，数值与自然域 logsumexp 参考一致（含 GQA 校正）。"""
    info = "shape=[1, 4, 512, 128], Hkv=2, dtype=float16, layout=HND, return_lse=True"
    with npu_timeout_guard(seconds=20, name="test_tc04", extra_info=info):
        q, k, v = make_qkv(1, 4, 2, 512, 512, 128, torch.float16, "HND", 400)
        out, lse = sage2.sageattn(q, k, v, return_lse=True)
        assert_mixed(out, sdpa_reference(q, k, v), torch.float16, "TC-04 out")
        assert_lse(lse, lse_reference(q, k), "TC-04 smooth_k=True")

        _, lse_nc = sage2.sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=False, return_lse=True)
        assert_lse(lse_nc, lse_reference(q, k), "TC-04 smooth_k=False")

        single = sage2.sageattn(q, k, v, return_lse=False)
        assert isinstance(single, torch.Tensor), "return_lse=False must return a single tensor"


def test_tc05_head_dim_pad_and_crop():
    """TC-05：D=48 pad 到 64、D=96 pad 到 128，输出裁回 head_dim_og。"""
    seed = 500
    for d in (48, 96):
        seed += 1
        info = f"shape=[1, 4, 128, {d}], pad_d={64 if d == 48 else 128}, dtype=float16, layout=HND"
        with npu_timeout_guard(seconds=20, name="test_tc05", extra_info=info):
            q, k, v = make_qkv(1, 4, 4, 128, 128, d, torch.float16, "HND", seed)
            out = sage2.sageattn(q, k, v)
            assert_mixed(out, sdpa_reference(q, k, v), torch.float16, f"TC-05 D={d}")


def test_tc06_gqa_and_sq_ne_sk():
    """TC-06：GQA（Hq/Hkv=4）与 Sq != Sk 非因果；因果 + Sq != Sk 显式拒绝。"""
    seed = 600
    for layout in ("HND", "NHD"):
        seed += 1
        info = f"GQA Hq=8/Hkv=2, Sq=128, Sk=256, D=128, dtype=float16, layout={layout}, is_causal=False"
        with npu_timeout_guard(seconds=20, name="test_tc06", extra_info=info):
            q, k, v = make_qkv(1, 8, 2, 128, 256, 128, torch.float16, layout, seed)
            out = sage2.sageattn(q, k, v, tensor_layout=layout)
            ref = sdpa_reference(to_hnd(q, layout), to_hnd(k, layout), to_hnd(v, layout))
            if layout == "NHD":
                ref = ref.transpose(1, 2)
            assert_mixed(out, ref, torch.float16, f"TC-06 {layout}")

    info_rej = "causal rejection check: Sq=128 != Sk=256 with is_causal=True"
    with npu_timeout_guard(seconds=20, name="test_tc06", extra_info=info_rej):
        with pytest.raises(ValueError, match="qo_len == kv_len"):
            sage2.sageattn(q, k, v, is_causal=True)


def test_tc07_causal():
    """TC-07：因果 mask 输出与 LSE 均对齐参考，且确实区别于非因果结果。"""
    info = "shape=[1, 4, 256, 128], dtype=float16, layout=HND, is_causal=True"
    with npu_timeout_guard(seconds=20, name="test_tc07", extra_info=info):
        q, k, v = make_qkv(1, 4, 4, 256, 256, 128, torch.float16, "HND", 700)
        out = sage2.sageattn(q, k, v, is_causal=True)
        assert_mixed(out, sdpa_reference(q, k, v, is_causal=True), torch.float16, "TC-07 causal")

        out_plain = sage2.sageattn(q, k, v, is_causal=False)
        assert not torch.allclose(out, out_plain, rtol=0.05, atol=0.05), \
            "TC-07: causal output equals non-causal output, mask not applied"

        _, lse = sage2.sageattn(q, k, v, is_causal=True, return_lse=True)
        assert_lse(lse, lse_reference(q, k, is_causal=True), "TC-07 lse")


def test_tc08_error_cases():
    """TC-08：D>128 / 非连续末维 / 非法 gran / 非法 accum / 非空 attn_mask 显式报错。"""
    info = "parameter validation & illegal argument rejection"
    with npu_timeout_guard(seconds=20, name="test_tc08", extra_info=info):
        q, k, v = make_qkv(1, 4, 4, 128, 128, 128, torch.float16, "HND", 800)

        q_big, k_big, v_big = make_qkv(1, 2, 2, 64, 64, 256, torch.float16, "HND", 801)
        with pytest.raises(ValueError, match="head_dim"):
            sage2.sageattn(q_big, k_big, v_big)

        base = torch.randn(1, 4, 128, 256, device=DEVICE, dtype=torch.float16)
        with pytest.raises(AssertionError, match="contiguous"):
            sage2.sageattn(base[..., ::2], k, v)

        with pytest.raises(ValueError, match="per_warp"):
            sage2.sageattn(q, k, v, qk_quant_gran="per_token")

        with pytest.raises(ValueError, match="pv_accum_dtype"):
            sage2.sageattn(q, k, v, pv_accum_dtype="int8")

        with pytest.raises(ValueError, match="attn_mask is not supported"):
            sage2.sageattn(q, k, v, attn_mask=torch.zeros(1, device=DEVICE))


def test_tc09_long_sequence_8k_16k():
    """TC-09：长序列 S=8192 与 S=16384（任务书 S>=8K/16K 要求；头数取 8 以约束 FP32 参考显存）。"""
    seed = 900
    for s in (8192, 16384):
        seed += 1
        info = f"shape=[1, 8, {s}, 128], dtype=float16, layout=HND, is_causal=False"
        with npu_timeout_guard(seconds=20, name="test_tc09", extra_info=info):
            q, k, v = make_qkv(1, 8, 8, s, s, 128, torch.float16, "HND", seed)
            if s == 8192:
                out, lse = sage2.sageattn(q, k, v, return_lse=True)
                assert lse.shape == (1, 8, s) and lse.dtype == torch.float32
                assert bool(torch.isfinite(lse).all())
            else:
                out = sage2.sageattn(q, k, v)
            assert_mixed(out, sdpa_reference(q, k, v, chunk=1024), torch.float16, f"TC-09 S={s}")
            assert torch.isfinite(out).all()


def test_tc10_sdpa_monkey_patch_smoke():
    """TC-10：F.scaled_dot_product_attention 热替换冒烟——patch 内路由到 sageattn，作用域还原。"""
    info = "shape=[1, 4, 256, 128], dtype=float16, layout=HND, F.scaled_dot_product_attention monkey patch"
    with npu_timeout_guard(seconds=20, name="test_tc10", extra_info=info):
        q, k, v = make_qkv(1, 4, 4, 256, 256, 128, torch.float16, "HND", 1000)
        direct = sage2.sageattn(q, k, v)

        prev = sage2.replace_sdpa()
        try:
            assert sage2.is_sdpa_patched()
            patched = F.scaled_dot_product_attention(q, k, v)
        finally:
            sage2.restore_sdpa(prev)
        assert not sage2.is_sdpa_patched()
        assert torch.equal(patched, direct), "patched SDPA output diverges from direct sageattn call"

        with sage2.sdpa_patch():
            assert sage2.is_sdpa_patched()
            in_ctx = F.scaled_dot_product_attention(q, k, v)
        assert not sage2.is_sdpa_patched()
        assert torch.equal(in_ctx, direct)


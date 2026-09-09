#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro roi_extractor CPU golden (pure torch, FP32 compute)."""

import os

os.environ.setdefault(
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "0"
)  # plain torch, no torch_npu autoload

import math

import torch


def _get_device():
    return torch.device("cpu")


# ─────────────────────────────────────────────
# Golden 参考实现（NPU torch）
# ─────────────────────────────────────────────

# Canonical mathematical semantics copied verbatim from the SPEC machine contract.
_SPEC_FORMULA = "y = roi_extract(features_0..features_{L-1}, rois, index) 逐 ROI n ∈ [0,R):\n(1) 层级选择: s = fp16(sqrt_fp32((rois[n,3]-rois[n,1])*(rois[n,4]-rois[n,2]))); 阈值 B[i]=2^i*finest_scale (i=0..L, fp16 精确表示); l=0 若 s<=B[1]; l=L-1 若 L>1 且 s>B[L-1]; 否则 l 为唯一满足 B[l]<s<=B[l+1] 的层 (1<=l<=L-2); L>=2 时任意 s 恰映射一层; L==1 且 s>2*finest_scale 时该 ROI 输出行不写入(内容未定义)\n(2) 坐标换算(fp32; fp16 输入先转 fp32): x0,y0,x1,y1 = rois[n,1..4]*spatial_scale[l]; aligned=true 时四坐标各减 0.5; roi_w=max(x1-x0,1.0); roi_h=max(y1-y0,1.0); bin_w=roi_w/pooled_width; bin_h=roi_h/pooled_height\n(3) 采样数: sample_num>0 时 sw=sh=sample_num; 否则 sw=ceil(bin_w), sh=ceil(bin_h)\n(4) batch 守卫: b=floor(rois[n,0]); 仅 0<=b<N 时计算, 否则该 ROI 输出行不写入(内容未定义)\n(5) 采样: 对 ph∈[0,pooled_height), pw∈[0,pooled_width), i∈[0,sh), j∈[0,sw): x=x0+(pw+(j+0.5)/sw)*bin_w; yc=y0+(ph+(i+0.5)/sh)*bin_h; acc[c]+=bilinear(features_l[b,c,:,:],x,yc)\n(6) 输出: y_row = (index 提供时) index[n] 否则 n; y[y_row,c,ph,pw] = acc[c]/(sh*sw) 对全部 c∈[0,C); 越界样本贡献 0 且仍计入分母 sh*sw\nbilinear(fm,x,y) (fm 空间尺寸 HxW, x 沿 W, y 沿 H, 输入为原始坐标): 若 x<-1 或 x>W 或 y<-1 或 y>H 返回 0; x=clamp(x,0,W-1); y=clamp(y,0,H-1); x_lo=min(floor(x),W-1); x_hi=min(x_lo+1,W-1); y_lo=min(floor(y),H-1); y_hi=min(y_lo+1,H-1); lx=x-x_lo; hx=1-lx; ly=y-y_lo; hy=1-ly; 返回 fm[y_lo,x_lo]*hy*hx+fm[y_lo,x_hi]*hy*lx+fm[y_hi,x_lo]*ly*hx+fm[y_hi,x_hi]*ly*lx"


def _fp16(v: float) -> float:
    return float(torch.tensor(v).to(torch.float16))


def _bilinear_vec(fm32: torch.Tensor, x: float, yc: float) -> torch.Tensor:
    """Bilinear sample of fm32 [C,H,W] at (x along W, yc along H), fp32, per SPEC §3 step (5)."""
    _, H, W = fm32.shape
    if x < -1.0 or x > W or yc < -1.0 or yc > H:
        return torch.zeros(fm32.shape[0], dtype=torch.float32, device=fm32.device)
    x = min(max(x, 0.0), float(W - 1))
    yc = min(max(yc, 0.0), float(H - 1))
    x_lo = min(int(x // 1), W - 1)
    x_hi = min(x_lo + 1, W - 1)
    y_lo = min(int(yc // 1), H - 1)
    y_hi = min(y_lo + 1, H - 1)
    lx = x - x_lo
    hx = 1.0 - lx
    ly = yc - y_lo
    hy = 1.0 - ly
    return (
        fm32[:, y_lo, x_lo] * (hy * hx)
        + fm32[:, y_lo, x_hi] * (hy * lx)
        + fm32[:, y_hi, x_lo] * (ly * hx)
        + fm32[:, y_hi, x_hi] * (ly * lx)
    )


def roi_extractor_golden_cpu(
    features_0: torch.Tensor,
    features_1: torch.Tensor,
    features_2: torch.Tensor,
    features_3: torch.Tensor,
    features_4: torch.Tensor,
    features_5: torch.Tensor,
    features_6: torch.Tensor,
    features_7: torch.Tensor,
    rois: torch.Tensor,
    index: torch.Tensor,
    finest_scale: int = 56,
    roi_scale_factor: float = 0.0,
    pooled_height: int = 7,
    pooled_width: int = 7,
    sample_num: int = 0,
    pool_mode: object = "avg",
    aligned: bool = True,
):
    """PyTorch 参考实现。

    根据算子规格中的数学公式实现。
    仅使用 torch 标准操作，不依赖 pypto / pypto_pro。

    Args:
        x: 输入 tensor。
           根据实际算子需求调整参数列表（可多输入、可带 gamma/beta/eps 等参数）。

    Returns:
        计算结果 tensor。
        根据实际算子需求调整返回值（可多输出、可返回 tuple）。
    """
    feats = [
        t
        for t in (
            features_0,
            features_1,
            features_2,
            features_3,
            features_4,
            features_5,
            features_6,
            features_7,
        )
        if t is not None
    ]
    if not feats or rois is None or rois.dim() != 2 or rois.shape[1] != 5:
        raise ValueError("need >=1 feature level and rois [R,5]")
    if pool_mode != "avg":
        raise ValueError("only pool_mode='avg' is supported")
    L = len(feats)
    N, C = feats[0].shape[0], feats[0].shape[1]
    R = rois.shape[0]
    for t in feats:
        if t.shape[0] != N or t.shape[1] != C:
            raise ValueError("all levels must share N and C")
    device = _get_device()
    feats = [t.to(device) for t in feats]
    feats32 = [t.float() for t in feats]
    rois = rois.to(device)
    y = torch.full(
        (R, C, pooled_height, pooled_width),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    for n in range(R):
        roi = rois[n].float()
        b = int(torch.floor(roi[0]).item())
        if b < 0 or b >= N:
            continue  # don't-care row keeps NaN
        s = float(torch.sqrt((roi[3] - roi[1]) * (roi[4] - roi[2])).to(torch.float16))
        b1 = _fp16(2.0 * finest_scale)
        b_last = _fp16(float(finest_scale) * (2 ** (L - 1)))
        if s <= b1:
            lvl = 0
        elif L == 1:
            continue  # L==1 large-roi drop: don't-care row keeps NaN
        elif s > b_last:
            lvl = L - 1
        else:
            lvl = 1
            while lvl < L - 1 and not (
                s > _fp16(finest_scale * (2**lvl))
                and s <= _fp16(finest_scale * (2 ** (lvl + 1)))
            ):
                lvl += 1
        sc = 0.25 / (2**lvl)
        x0, y0, x1, y1 = roi[1] * sc, roi[2] * sc, roi[3] * sc, roi[4] * sc
        if aligned:
            x0, y0, x1, y1 = x0 - 0.5, y0 - 0.5, x1 - 0.5, y1 - 0.5
        roi_w = max(float(x1 - x0), 1.0)
        roi_h = max(float(y1 - y0), 1.0)
        bin_w = roi_w / pooled_width
        bin_h = roi_h / pooled_height
        if sample_num > 0:
            sw = sh = sample_num
        else:
            sw = max(1, math.ceil(bin_w))
            sh = max(1, math.ceil(bin_h))
        fm32 = feats32[lvl][b]
        y_row = int(index[n].item()) if index is not None else n
        for ph in range(pooled_height):
            for pw in range(pooled_width):
                acc = torch.zeros(C, dtype=torch.float32, device=device)
                for i in range(sh):
                    for j in range(sw):
                        x = float(x0 + (pw + (j + 0.5) / sw) * bin_w)
                        yc = float(y0 + (ph + (i + 0.5) / sh) * bin_h)
                        acc = acc + _bilinear_vec(fm32, x, yc)
                y[y_row, :, ph, pw] = acc / (sh * sw)
    return y


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================


def _make_inputs(device):
    """Construct every P0 case from the validated machine contract (SPEC §10 recipe).

    Tensors are built on CPU with seeded generators (determinism) then moved to
    `device`; unused feature levels are None.
    """
    cases = []

    def feats(shapes, dtype, seed):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return [
            (torch.randn(s, generator=g) * 0.5).to(dtype).to(device) for s in shapes
        ]

    def uniform(g, lo, hi):
        return lo + (hi - lo) * torch.rand(1, generator=g).item()

    # ---- P0 1: ut_multilevel_fp16 ----
    fs1 = feats(
        [(1, 16, 16, 16), (1, 16, 9, 9), (1, 16, 8, 8), (1, 16, 4, 5)],
        torch.float16,
        1001,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(1011)
    rows = []
    for lo, hi in [(2.0, 112.0), (112.0, 224.0), (224.0, 448.0), (448.0, 800.0)]:
        for _ in range(240):  # rows 0..959: s covers all 4 level bands
            s = uniform(g, lo, hi)
            u = uniform(g, 0.64, 1.5625)  # sqrt(u) in [0.8, 1.25]
            w = s * (u**0.5)
            h = s / (u**0.5)
            x0 = uniform(g, -0.5 * w, 48.0)
            y0 = uniform(g, -0.5 * h, 48.0)
            rows.append([0.0, x0, y0, x0 + w, y0 + h])
    for k in range(10):  # 960..969 edge-touching
        w = uniform(g, 4.0, 60.0)
        h = uniform(g, 4.0, 60.0)
        if k % 2 == 0:
            rows.append([0.0, 0.0, 0.0, w, h])
        else:
            rows.append([0.0, 63.0 - w, 63.0 - h, 63.0, 63.0])
    for k in range(10):  # 970..979 zero-area
        w = uniform(g, 4.0, 40.0)
        h = uniform(g, 4.0, 40.0)
        if k < 5:
            rows.append([0.0, 10.0, 10.0, 10.0, 10.0 + h])
        else:
            rows.append([0.0, 10.0, 10.0, 10.0 + w, 10.0])
    for k in range(10):  # 980..989 partially out of bounds
        if k % 2 == 0:
            rows.append(
                [
                    0.0,
                    -uniform(g, 4.0, 20.0),
                    -uniform(g, 4.0, 20.0),
                    uniform(g, 8.0, 40.0),
                    uniform(g, 8.0, 40.0),
                ]
            )
        else:
            w = uniform(g, 8.0, 40.0)
            h = uniform(g, 8.0, 40.0)
            rows.append(
                [
                    0.0,
                    63.0 - uniform(g, 0.0, 10.0),
                    63.0 - uniform(g, 0.0, 10.0),
                    63.0 + w,
                    63.0 + h,
                ]
            )
    # 990..994 integer-area s-boundary rows (exact in fp16)
    rows.append([0.0, 5.0, 5.0, 117.0, 117.0])  # w=h=112 -> s=112 -> level 0
    rows.append([0.0, 5.0, 5.0, 229.0, 229.0])  # s=224 -> level 1
    rows.append([0.0, 5.0, 5.0, 453.0, 453.0])  # s=448 -> level 2
    rows.append([0.0, -20.0, 5.0, 204.0, 229.0])
    rows.append([0.0, 5.0, -20.0, 453.0, 428.0])
    for k in range(5):  # 995..999 batch index out of range (N=1) -> don't-care
        rows.append([-1.0 if k % 2 == 0 else 1.0, 4.0, 4.0, 20.0, 20.0])
    assert len(rows) == 1000
    rois1 = torch.tensor(rows, dtype=torch.float32).to(torch.float16).to(device)
    args1 = fs1 + [None] * (8 - len(fs1)) + [rois1, None]
    kwargs1 = {
        "finest_scale": 56,
        "roi_scale_factor": 0.0,
        "pooled_height": 7,
        "pooled_width": 7,
        "sample_num": 0,
        "pool_mode": "avg",
        "aligned": True,
    }
    cases.append(("ut_multilevel_fp16", args1, kwargs1))

    # ---- P0 2: st_singlelevel_fp32 (image 304px) ----
    fs2 = feats([(5, 64, 76, 76)], torch.float32, 1002)
    g.manual_seed(1012)
    rows = []
    for _ in range(140):  # normal rois within the image
        x0 = uniform(g, 0.0, 250.0)
        y0 = uniform(g, 0.0, 250.0)
        rows.append(
            [
                float(int(uniform(g, 0, 5))),
                x0,
                y0,
                x0 + uniform(g, 4.0, 50.0),
                y0 + uniform(g, 4.0, 50.0),
            ]
        )
    for k in range(10):  # edge / out-of-bounds
        if k % 2 == 0:
            rows.append(
                [
                    float(k % 5),
                    -uniform(g, 4.0, 20.0),
                    0.0,
                    uniform(g, 8.0, 40.0),
                    uniform(g, 4.0, 40.0),
                ]
            )
        else:
            rows.append(
                [
                    float(k % 5),
                    300.0,
                    300.0,
                    300.0 + uniform(g, 4.0, 30.0),
                    300.0 + uniform(g, 4.0, 30.0),
                ]
            )
    assert len(rows) == 150
    rois2 = torch.tensor(rows, dtype=torch.float32).to(device)
    args2 = fs2 + [None] * 7 + [rois2, None]
    kwargs2 = dict(kwargs1)
    cases.append(("st_singlelevel_fp32", args2, kwargs2))

    # ---- P0 3: dynamic_singlelevel_fp16_index (image 800x1344) ----
    fs3 = feats([(4, 256, 200, 336)], torch.float16, 1003)
    g.manual_seed(1013)
    rows = []
    for i in range(1200):
        bidx = float(i % 4)
        if i % 5 < 4:  # ~80% small rois (s <= 112)
            w = uniform(g, 2.0, 110.0)
            h = uniform(g, 2.0, 110.0)
        else:  # ~20% large rois (s > 112) -> L=1 drop, don't-care
            w = uniform(g, 120.0, 400.0)
            h = uniform(g, 120.0, 400.0)
        x0 = uniform(g, 0.0, 700.0)
        y0 = uniform(g, 0.0, 800.0)
        rows.append([bidx, x0, y0, min(x0 + w, 1344.0), min(y0 + h, 800.0)])
    rois3 = torch.tensor(rows, dtype=torch.float32).to(torch.float16).to(device)
    index3 = torch.arange(1199, -1, -1, dtype=torch.int32).to(device)
    args3 = fs3 + [None] * 7 + [rois3, index3]
    kwargs3 = dict(kwargs1)
    cases.append(("dynamic_singlelevel_fp16_index", args3, kwargs3))

    # ---- P0 4: multilevel_n2_sample2 (image 192px, 2 images) ----
    fs4 = feats(
        [(2, 32, 48, 48), (2, 32, 24, 24), (2, 32, 12, 12), (2, 32, 6, 6)],
        torch.float16,
        1004,
    )
    g.manual_seed(1014)
    rows = []
    for i in range(64):
        s = uniform(g, 4.0, 600.0)
        u = uniform(g, 0.64, 1.5625)
        w = s * (u**0.5)
        h = s / (u**0.5)
        x0 = uniform(g, 0.0, 160.0)
        y0 = uniform(g, 0.0, 160.0)
        rows.append([float(i % 2), x0, y0, x0 + w, y0 + h])
    rois4 = torch.tensor(rows, dtype=torch.float32).to(torch.float16).to(device)
    args4 = fs4 + [None] * (8 - len(fs4)) + [rois4, None]
    kwargs4 = dict(kwargs1)
    kwargs4["sample_num"] = 2
    cases.append(("multilevel_n2_sample2", args4, kwargs4))

    return cases


# ==========================================
# 验证
# ==========================================


def _validate():
    device = _get_device()
    print("=" * 60)
    print("roi_extractor_golden validation report")
    print("=" * 60)
    print(f"Device: {device}")

    cases = _make_inputs(device)
    names = [c[0] for c in cases]
    assert names == [
        "ut_multilevel_fp16",
        "st_singlelevel_fp32",
        "dynamic_singlelevel_fp16_index",
        "multilevel_n2_sample2",
    ], names

    def s_of(roi_row):
        r = roi_row.float()
        return float(torch.sqrt((r[3] - r[1]) * (r[4] - r[2])).to(torch.float16))

    for case_name, args, kwargs in cases:
        rois = args[-2]
        feats = [t for t in args[:8] if t is not None]
        R, C = rois.shape[0], feats[0].shape[1]
        y = roi_extractor_golden_cpu(*args, **kwargs)
        ph, pw = kwargs["pooled_height"], kwargs["pooled_width"]
        assert tuple(y.shape) == (R, C, ph, pw), (case_name, tuple(y.shape))
        assert y.dtype == torch.float32, (case_name, y.dtype)
        nan_rows = torch.isnan(y.float()).any(dim=(1, 2, 3))
        n_nan = int(nan_rows.sum().item())
        if case_name == "ut_multilevel_fp16":
            expect_nan = torch.zeros(R, dtype=torch.bool, device=y.device)
            expect_nan[995:1000] = True  # batch out of range
            assert torch.equal(nan_rows, expect_nan), (case_name, n_nan)
        elif case_name == "st_singlelevel_fp32":
            assert n_nan == 0, (case_name, n_nan)
        elif case_name == "dynamic_singlelevel_fp16_index":
            idx = args[-1].long()
            big = torch.tensor(
                [s_of(rois[n]) > 112.0 for n in range(R)], device=y.device
            )
            big_out = torch.empty_like(big)
            big_out[idx] = big
            assert torch.equal(nan_rows, big_out), (case_name, n_nan)
        else:
            assert n_nan == 0, (case_name, n_nan)
        print(
            f"[case {case_name}] shape={tuple(y.shape)} dtype={y.dtype} "
            f"finite_rows={R - n_nan}/{R} NaN_rows={n_nan} PASS"
        )

    # ---- scenario s1: empty RoI ----
    fs = [t for t in cases[0][1][:8] if t is not None]
    y = roi_extractor_golden_cpu(
        *(
            fs
            + [None] * (8 - len(fs))
            + [torch.zeros(0, 5, dtype=torch.float16, device=device), None]
        ),
        **cases[0][2],
    )
    assert tuple(y.shape) == (0, 16, 7, 7)
    print("[scenario s1 empty-roi] shape=", tuple(y.shape), "PASS")

    # ---- scenario s2: non-square pooled 4x8 ----
    g = torch.Generator(device="cpu")
    g.manual_seed(1021)
    rows = [
        [
            0.0,
            0.0 + 40.0 * torch.rand(1, generator=g).item(),
            0.0 + 40.0 * torch.rand(1, generator=g).item(),
            0.0,
            0.0,
        ]
        for _ in range(256)
    ]
    for i, r in enumerate(rows):
        r[3] = r[1] + 4.0 + 40.0 * torch.rand(1, generator=g).item()
        r[4] = r[2] + 4.0 + 40.0 * torch.rand(1, generator=g).item()
    rois = torch.tensor(rows, dtype=torch.float32).to(torch.float16).to(device)
    kw = dict(cases[0][2])
    kw["pooled_height"] = 4
    kw["pooled_width"] = 8
    y = roi_extractor_golden_cpu(*(fs + [None] * (8 - len(fs)) + [rois, None]), **kw)
    assert tuple(y.shape) == (256, 16, 4, 8) and torch.isfinite(y.float()).all().item()
    print("[scenario s2 pooled-4x8] shape=", tuple(y.shape), "PASS")

    # ---- scenario s3: L=1 large-roi drop ----
    fs1lvl = [
        (
            torch.randn(
                1, 16, 32, 32, generator=torch.Generator(device="cpu").manual_seed(1031)
            )
            * 0.5
        )
        .to(torch.float16)
        .to(device)
    ]
    rows = []
    for i in range(16):
        if i < 8:
            w = h = 8.0 + i  # small: s <= 112
        else:
            w = h = 150.0 + i  # large: s > 112 -> dropped
        rows.append([0.0, 2.0, 2.0, 2.0 + w, 2.0 + h])
    rois = torch.tensor(rows, dtype=torch.float32).to(torch.float16).to(device)
    y = roi_extractor_golden_cpu(*(fs1lvl + [None] * 7 + [rois, None]), **cases[0][2])
    nan_rows = torch.isnan(y.float()).any(dim=(1, 2, 3))
    assert int(nan_rows.sum().item()) == 8 and not bool(nan_rows[:8].any().item())
    print("[scenario s3 L1-drop] NaN rows = 8/16 PASS")

    # ---- scenario s6: 8-level pyramid ----
    shapes = [(1, 16, 512 // (2**i), 512 // (2**i)) for i in range(8)]
    g.manual_seed(1036)
    fs8 = [
        (torch.randn(s, generator=g) * 0.5).to(torch.float16).to(device) for s in shapes
    ]
    rows = []
    for i in range(512):
        s = 4.0 + (3000.0 - 4.0) * torch.rand(1, generator=g).item()
        u = 0.64 + (1.5625 - 0.64) * torch.rand(1, generator=g).item()
        w = s * (u**0.5)
        h = s / (u**0.5)
        x0 = 0.0 + 1500.0 * torch.rand(1, generator=g).item()
        y0 = 0.0 + 1500.0 * torch.rand(1, generator=g).item()
        rows.append([0.0, x0, y0, x0 + w, y0 + h])
    rois = torch.tensor(rows, dtype=torch.float32).to(torch.float16).to(device)
    y = roi_extractor_golden_cpu(*(fs8 + [rois, None]), **cases[0][2])
    assert tuple(y.shape) == (512, 16, 7, 7) and torch.isfinite(y.float()).all().item()
    print("[scenario s6 8-level pyramid] shape=", tuple(y.shape), "PASS")

    print("=" * 60)
    print("validation complete")
    print("=" * 60)


if __name__ == "__main__":
    _validate()

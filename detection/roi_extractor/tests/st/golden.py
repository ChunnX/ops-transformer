#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""RoiExtractor ST golden (NumPy, fp64 reference) + case generation / result comparison.

Usage:
  # generate on-board case files into a directory
  python3 golden.py --gen <case_dir>

  # compare an NPU result binary against the case golden
  python3 golden.py --compare <case_dir>/<case_name>/result.bin <case_dir>/<case_name>

Semantics follow docs/spec.yaml (11-stage PASS): FPN level mapping +
RoIAlign bilinear interpolation with torchvision/mmcv-compatible boundary
rules (out-of-range -> 0 contribution counted in the avg denominator,
lower clamp y'=max(y,0), upper pin y1=min(y0+1,H-1), degenerate RoI
max(w,1)/max(h,1)).
"""

import argparse
import json
import math
import os
import struct

import numpy as np


def roi_extractor_golden(feats, rois, finest_scale=56, roi_scale_factor=0.0,
                         spatial_scale=None, pooled_h=7, pooled_w=7,
                         sample_num=0, pool_mode="avg", aligned=True):
    """Multi-level RoI feature extraction reference (fp64).

    Args:
        feats: list of [N, C, H_l, W_l] arrays (float, any dtype; upcast to fp64)
        rois:  [K, 5] = (batch_idx, x1, y1, x2, y2)
    Returns:
        [K, C, pooled_h, pooled_w] float64.
    """
    levels = len(feats)
    if spatial_scale is None:
        spatial_scale = [0.25, 0.125, 0.0625, 0.03125, 0.015625][:levels]
    assert len(spatial_scale) == levels
    assert pool_mode in ("avg", "max")
    feats = [np.asarray(f).astype(np.float64) for f in feats]
    rois = np.asarray(rois).astype(np.float64)
    k_num, c_dim = rois.shape[0], feats[0].shape[1]
    out = np.zeros((k_num, c_dim, pooled_h, pooled_w), dtype=np.float64)

    coords = rois[:, 1:5].copy()
    if roi_scale_factor is not None and roi_scale_factor > 0:
        coords = coords / float(roi_scale_factor)
    area = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])

    off = 0.5 if aligned else 0.0
    for k in range(k_num):
        # level mapping (comparison form, exact 2^j boundary belongs to the coarser level)
        lvl = 0
        for j in range(1, levels):
            if area[k] >= (finest_scale * (2.0 ** j)) ** 2 * (
                    1.0 if roi_scale_factor is None or roi_scale_factor <= 0 else roi_scale_factor ** 2):
                lvl = j
        s = float(spatial_scale[lvl])
        f = feats[lvl]
        n_dim, _, h_l, w_l = f.shape
        b = int(math.floor(rois[k, 0]))
        b = min(max(b, 0), n_dim - 1)
        rx1 = coords[k, 0] * s - off
        ry1 = coords[k, 1] * s - off
        rx2 = coords[k, 2] * s - off
        ry2 = coords[k, 3] * s - off
        rw = max(rx2 - rx1, 1.0)
        rh = max(ry2 - ry1, 1.0)
        bw = rw / pooled_w
        bh = rh / pooled_h
        nh = max(int(math.ceil(bh)), 1) if sample_num <= 0 else sample_num
        nw = max(int(math.ceil(bw)), 1) if sample_num <= 0 else sample_num
        cnt = float(nh * nw)

        for ph in range(pooled_h):
            for pw in range(pooled_w):
                acc = None
                for ih in range(nh):
                    for iw in range(nw):
                        yy = ry1 + ph * bh + (ih + 0.5) * bh / nh
                        xx = rx1 + pw * bw + (iw + 0.5) * bw / nw
                        if yy < -1.0 or yy > h_l or xx < -1.0 or xx > w_l:
                            val = np.zeros(c_dim, dtype=np.float64)
                        else:
                            yyc = max(yy, 0.0)
                            xxc = max(xx, 0.0)
                            y0 = int(math.floor(yyc))
                            x0 = int(math.floor(xxc))
                            y1 = min(y0 + 1, h_l - 1)
                            x1 = min(x0 + 1, w_l - 1)
                            lam_y = yyc - y0
                            lam_x = xxc - x0
                            g00 = f[b, :, y0, x0]
                            g01 = f[b, :, y0, x1]
                            g10 = f[b, :, y1, x0]
                            g11 = f[b, :, y1, x1]
                            val = ((1 - lam_y) * (1 - lam_x)) * g00 + ((1 - lam_y) * lam_x) * g01 \
                                + (lam_y * (1 - lam_x)) * g10 + (lam_y * lam_x) * g11
                        acc = val if acc is None else (acc + val if pool_mode == "avg" else np.maximum(acc, val))
                out[k, :, ph, pw] = acc / cnt if pool_mode == "avg" else acc
    return out


def _gen_case(rng, name, levels, n, c, base_hw, k, dtype, pooled_hw,
              sample_num, pool_mode, aligned, roi_scale_factor=0.0,
              roi_span=None, finest_scale=56):
    feats = []
    for l in range(levels):
        h = max(base_hw[0] >> l, 2)
        w = max(base_hw[1] >> l, 2)
        feats.append(rng.uniform(-1.0, 1.0, size=(n, c, h, w)).astype(dtype))
    spatial_scale = [0.25 / (2 ** l) for l in range(levels)]
    # RoIs spanning all levels: sizes from finest_scale*0.5 to finest_scale*2^(levels+1)
    rois = np.zeros((k, 5), dtype=dtype)
    for i in range(k):
        lvl = i % levels
        size = finest_scale * (2.0 ** lvl) * (0.8 + 0.4 * rng.uniform(0, 1))
        if roi_span is not None:
            size = roi_span
        half = size / 2.0 / spatial_scale[lvl]
        cx = rng.uniform(0, feats[lvl].shape[3])
        cy = rng.uniform(0, feats[lvl].shape[2])
        x1 = max(cx - half, 0.0)
        y1 = max(cy - half, 0.0)
        x2 = min(cx + half, feats[lvl].shape[3] - 0.5)
        y2 = min(cy + half, feats[lvl].shape[2] - 0.5)
        rois[i] = (float(i % n), x1, y1, x2, y2)
    golden = roi_extractor_golden(
        feats, rois, finest_scale=finest_scale, roi_scale_factor=roi_scale_factor,
        spatial_scale=spatial_scale, pooled_h=pooled_hw[0], pooled_w=pooled_hw[1],
        sample_num=sample_num, pool_mode=pool_mode, aligned=aligned)
    meta = {
        "name": name, "levels": levels, "batch": n, "channels": c,
        "level_hw": [[f.shape[2], f.shape[3]] for f in feats],
        "rois_num": k, "dtype": "float16" if dtype == np.float16 else "float32",
        "pooled_h": pooled_hw[0], "pooled_w": pooled_hw[1],
        "sample_num": sample_num, "finest_scale": finest_scale,
        "roi_scale_factor": roi_scale_factor, "spatial_scale": spatial_scale,
        "pool_mode": pool_mode, "aligned": aligned,
    }
    return meta, feats, rois, golden.astype(np.float32)


def generate(case_dir):
    os.makedirs(case_dir, exist_ok=True)
    cases = [
        # name, levels, N, C, base_hw, K, dtype, pooled, sample_num, mode, aligned, extra
        ("fp32_4lvl_avg_aligned",      4, 2, 64, (64, 64), 16, np.float32, (7, 7),  0, "avg", True, {}),
        ("fp32_4lvl_max_aligned",      4, 2, 64, (64, 64), 16, np.float32, (7, 7),  0, "max", True, {}),
        ("fp32_4lvl_avg_unaligned",    4, 1, 32, (56, 56), 12, np.float32, (7, 7),  0, "avg", False, {}),
        ("fp32_1lvl_degenerate",       1, 1, 32, (32, 32), 8,  np.float32, (7, 7),  0, "avg", True, {}),
        ("fp32_5lvl_sr4_max",          5, 1, 32, (64, 64), 20, np.float32, (7, 7),  4, "max", True, {}),
        ("fp32_nonsquare_pool",        2, 1, 32, (48, 64), 10, np.float32, (7, 14), 0, "avg", True, {}),
        ("fp32_roiscalefactor",        2, 1, 32, (64, 64), 8,  np.float32, (7, 7),  0, "avg", True,
         {"roi_scale_factor": 2.0, "roi_span": 224.0}),
        ("fp32_degenerate_rois",       1, 1, 32, (32, 32), 3,  np.float32, (7, 7),  0, "avg", True,
         {"roi_span": 1.0}),
        ("fp16_4lvl_avg_aligned",      4, 2, 64, (64, 64), 16, np.float16, (7, 7),  0, "avg", True, {}),
        ("fp16_4lvl_max_sr2",          4, 2, 64, (64, 64), 16, np.float16, (7, 7),  2, "max", True, {}),
    ]
    index = []
    for (name, levels, n, c, hw, k, dtype, pooled, sn, mode, aligned, extra) in cases:
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        meta, feats, rois, golden = _gen_case(
            rng, name, levels, n, c, hw, k, dtype, pooled, sn, mode, aligned,
            roi_scale_factor=extra.get("roi_scale_factor", 0.0),
            roi_span=extra.get("roi_span"))
        d = os.path.join(case_dir, name)
        os.makedirs(d, exist_ok=True)
        for l, f in enumerate(feats):
            f.tofile(os.path.join(d, "feats_l%d.bin" % l))
        rois.tofile(os.path.join(d, "rois.bin"))
        golden.tofile(os.path.join(d, "golden.bin"))
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=1)
        index.append(name)
        print("generated:", name, "| feats", [list(f.shape) for f in feats], "| rois", k,
              "| dtype", meta["dtype"])
    with open(os.path.join(case_dir, "index.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(index) + "\n")
    print("total cases:", len(index))


def _load_case(case_path):
    with open(os.path.join(case_path, "meta.json"), "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    dtype = np.float16 if meta["dtype"] == "float16" else np.float32
    feats = []
    for l in range(meta["levels"]):
        h, w = meta["level_hw"][l]
        f = np.fromfile(os.path.join(case_path, "feats_l%d.bin" % l), dtype=dtype)
        feats.append(f.reshape(meta["batch"], meta["channels"], h, w))
    rois = np.fromfile(os.path.join(case_path, "rois.bin"), dtype=dtype).reshape(meta["rois_num"], 5)
    golden = np.fromfile(os.path.join(case_path, "golden.bin"), dtype=np.float32).reshape(
        meta["rois_num"], meta["channels"], meta["pooled_h"], meta["pooled_w"])
    return meta, feats, rois, golden


def compare(result_bin, case_path):
    meta, feats, rois, golden = _load_case(case_path)
    result = np.fromfile(result_bin, dtype=np.float32).reshape(golden.shape)
    if meta["dtype"] == "float16":
        rtol = atol = 1.0e-3
    else:
        rtol = atol = 1.0e-5
    diff = np.abs(result.astype(np.float64) - golden.astype(np.float64))
    tol = atol + rtol * np.abs(golden.astype(np.float64))
    bad = diff > tol
    max_diff = float(diff.max()) if diff.size else 0.0
    ok = not bool(bad.any())
    print("case=%s dtype=%s shape=%s bad=%d/%d max_diff=%.3e (rtol=atol=%g) -> %s"
          % (meta["name"], meta["dtype"], list(golden.shape), int(bad.sum()), bad.size,
             max_diff, rtol, "PASS" if ok else "FAIL"))
    if not ok:
        idx = np.argwhere(bad)
        for i in idx[:5]:
            k, c, ph, pw = [int(v) for v in i]
            print("  mismatch at [%d,%d,%d,%d]: result=%.7f golden=%.7f diff=%.3e"
                  % (k, c, ph, pw, result[k, c, ph, pw], golden[k, c, ph, pw], diff[k, c, ph, pw]))
    return ok


def selftest():
    """Cross-check against torchvision.ops.roi_align (L=1) when available."""
    try:
        import torch
        from torchvision.ops import roi_align
    except ImportError:
        print("torch/torchvision unavailable, skip selftest")
        return True
    rng = np.random.default_rng(7)
    feats = [rng.uniform(-1, 1, size=(1, 8, 32, 32)).astype(np.float32)]
    rois = np.array([[0, 3.2, 4.1, 20.5, 18.7]], dtype=np.float32)
    for aligned in (True, False):
        for sr in (0, 2):
            got = roi_extractor_golden(feats, rois, spatial_scale=[1.0], pooled_h=7, pooled_w=7,
                                       sample_num=sr, aligned=aligned)
            tv = roi_align(torch.from_numpy(feats[0]), torch.from_numpy(rois), (7, 7), 1.0,
                           sr if sr > 0 else -1, aligned).numpy()[0]
            err = np.abs(got[0] - tv).max()
            assert err < 1e-5, "selftest failed aligned=%s sr=%d err=%g" % (aligned, sr, err)
            print("selftest aligned=%s sr=%d max_err=%.2e OK" % (aligned, sr, err))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", metavar="DIR", help="generate case files into DIR")
    ap.add_argument("--compare", nargs=2, metavar=("RESULT_BIN", "CASE_DIR"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.gen:
        generate(args.gen)
        return
    if args.compare:
        raise SystemExit(0 if compare(args.compare[0], args.compare[1]) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()

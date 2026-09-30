#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys

import numpy as np
import torch

# 与量化版一致的"双千分之一"精度标准（适用于所有模式）：
#   单元素相对误差超 ERROR_TOL(1e-3) 的元素占比 ≤ FAIL_RATE_LIMIT(1e-3) 算过。
ERROR_TOL = 1e-3
FAIL_RATE_LIMIT = 1e-3
DATA_TYPE = np.float16
FULL_TENSOR_PRINT_MAX_ELEMENTS = 1024
CORNER_ROWS = 4
CORNER_COLS = 4


def read_matrix(path, dtype):
    """读取 fp16/bf16 原始 2 字节矩阵。bf16 无 numpy 类型，经 uint16 视图还原。"""
    if dtype == "bf16":
        raw = np.fromfile(path, dtype=np.uint16)
        return torch.from_numpy(raw).view(torch.bfloat16)
    return torch.from_numpy(np.fromfile(path, dtype=DATA_TYPE))

# ----------------------------------------------------------------------------------------------------------
# 可视化标注层（纯上报，不改变 pass/fail 判定）
# ----------------------------------------------------------------------------------------------------------


def _density_symbol(cnt):
    """tile 内错误元素数 -> 密度符号。"""
    if cnt <= 0:
        return "."
    if cnt <= 9:
        return ":"
    if cnt <= 49:
        return "o"
    if cnt <= 99:
        return "O"
    return "#"


def _fmt_num(x):
    """统一数值打印格式。"""
    return f"{x:.6g}"


def _banner(title):
    """统一分节横幅：每个 [visual] 小节都以 ===== 标题 ===== 开头。"""
    print(f"\n[visual] ===== {title} =====")


def build_mismatch_mask(golden_np, npu_np, atol, rtol):
    """与 torch.allclose(golden, npu, rtol, atol) 一致的错误 mask。"""
    diff = np.abs(npu_np.astype(np.float32) - golden_np.astype(np.float32))
    thr = atol + rtol * np.abs(npu_np.astype(np.float32))
    return diff > thr, diff


def _tile_counts(mask_np, m, n, tile_m, tile_n):
    """把 (m,n) bool mask 按 tile_m(行) x tile_n(列) 分块求和，返回 (tr, tc) 计数矩阵。"""
    pm = (tile_m - m % tile_m) % tile_m
    pn = (tile_n - n % tile_n) % tile_n
    padded = np.zeros((m + pm, n + pn), dtype=np.int64)
    padded[:m, :n] = mask_np.astype(np.int64)
    tr, tc = (m + pm) // tile_m, (n + pn) // tile_n
    return padded.reshape(tr, tile_m, tc, tile_n).sum(axis=(1, 3))


def _bool_runs(arr):
    """返回 arr 中连续 True 段的 (start, end) 列表（含端点）。"""
    runs = []
    start = None
    for i, v in enumerate(arr):
        if v:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(arr) - 1))
    return runs


def _truncate_runs(runs, cap=20):
    """只保留最长的 cap 段，其余折叠。"""
    if len(runs) <= cap:
        return runs, 0
    by_len = sorted(runs, key=lambda r: r[1] - r[0] + 1, reverse=True)
    return sorted(by_len[:cap]), len(runs) - cap


def render_density_map(mask_np, m, n, tile_m, tile_n, width_limit=120, title="error density map",
                       legend=".:0  ::1-9  o:10-49  O:50-99  #:>=100 errors/cell"):
    """缩略密度图：按 baseM x baseN tile 分块，坐标 = tile 行列号。

    每个格子对应一个计算 tile：(r,c) 覆盖元素 [r*tile_m..(r+1)*tile_m) x [c*tile_n..(c+1)*tile_n)。
    网格过大（>40 行或 >80 列）时自动按 2 倍粗化。
    """
    tm, tn = tile_m, tile_n
    rows_limit, cols_limit = 40, 80
    while (m + tm - 1) // tm > rows_limit or (n + tn - 1) // tn > cols_limit:
        tm *= 2
        tn *= 2
    counts = _tile_counts(mask_np, m, n, tm, tn)
    tr, tc = counts.shape
    coarsen = f"  (auto-coarsened to {tm}x{tn})" if (tm, tn) != (tile_m, tile_n) else ""
    _banner(f"{title}  (tile = {tile_m}x{tile_n}, grid {tr} x {tc}){coarsen}")
    print(f"[visual]   {legend}")
    print(f"[visual]   cell (r,c) = tile [r*{tm}..(r+1)*{tm}) x [c*{tn}..(c+1)*{tn})")
    cell_w = 3
    page = max(1, (width_limit - 14) // cell_w)
    for c0 in range(0, tc, page):
        c1 = min(tc, c0 + page)
        hdr = "   tile-c> " + "".join(f"{c:>3}" for c in range(c0, c1))
        print(f"[visual]  {hdr}")
        for r in range(tr):
            line = f"[visual]  tile-r{r:>3} " + "".join(
                f"{_density_symbol(int(counts[r, c])):^3}" for c in range(c0, c1))
            print(line)


def render_profiles(mask_np, m, n, tile_m, tile_n, label="error"):
    """行/列 profile：元素级连续段 + 每 tile 的分布。label 用于区分 error/zero-output。"""
    row_any = mask_np.any(axis=1)
    col_any = mask_np.any(axis=0)
    bad_rows = int(row_any.sum())
    bad_cols = int(col_any.sum())
    _banner(f"{label} profile")
    print(f"[visual]   bad rows {bad_rows}/{m} ({100.0*bad_rows/m:.1f}%), "
          f"bad cols {bad_cols}/{n} ({100.0*bad_cols/n:.1f}%)")

    # 接近全坏：折叠明细，避免刷屏
    if bad_rows >= 0.98 * m and bad_cols >= 0.98 * n:
        print(f"[visual]   ({label} covers ~entire matrix; runs and tile lists omitted)")
        return

    rr, dropped_r = _truncate_runs(_bool_runs(row_any))
    if rr:
        seg = " ".join(f"[{a},{b}]({b-a+1})" for a, b in rr)
        if dropped_r:
            seg += f" ... +{dropped_r} more"
        print(f"[visual]   {label} bad-row runs: {seg}")
    cr, dropped_c = _truncate_runs(_bool_runs(col_any))
    if cr:
        seg = " ".join(f"[{a},{b}]({b-a+1})" for a, b in cr)
        if dropped_c:
            seg += f" ... +{dropped_c} more"
        print(f"[visual]   {label} bad-col runs: {seg}")

    # 每 tile 的分布（仅打印有异常的，cap 8）
    counts = _tile_counts(mask_np, m, n, tile_m, tile_n)
    rnz = [(r, int((counts[r] > 0).sum())) for r in range(counts.shape[0]) if (counts[r] > 0).any()]
    cnz = [(c, int((counts[:, c] > 0).sum())) for c in range(counts.shape[1]) if (counts[:, c] > 0).any()]
    if rnz:
        rnz = sorted(rnz, key=lambda x: x[1], reverse=True)
        seg = " ".join(f"row-tile{r}({cnt})" for r, cnt in rnz[:8])
        if len(rnz) > 8:
            seg += f" ... +{len(rnz)-8} more"
        print(f"[visual]   {label} row-tiles (tile={tile_m}x{tile_n}): {seg}")
    if cnz:
        cnz = sorted(cnz, key=lambda x: x[1], reverse=True)
        seg = " ".join(f"col-tile{c}({cnt})" for c, cnt in cnz[:8])
        if len(cnz) > 8:
            seg += f" ... +{len(cnz)-8} more"
        print(f"[visual]   {label} col-tiles (tile={tile_m}x{tile_n}): {seg}")


def render_zero_output(npu_np, m, n, tile_m, tile_n):
    """输出零值分析：全坏/未覆盖时最有判别力——区分'被 fixpipe 写了'和'没写（0）'。"""
    zero = npu_np == 0.0
    total = int(zero.sum())
    frac = float(total) / zero.size
    if frac < 0.01:
        return
    _banner(f"zero-output  ({total}/{zero.size} = {100.0*frac:.1f}% exactly 0.0)")
    if frac >= 0.99:
        print("[visual]   ~entire output is 0.0 -> GM mostly unwritten; check fixpipe coverage / "
              "write target, not per-element error")
        return
    render_density_map(zero, m, n, tile_m, tile_n, title="zero-output map",
                       legend=".:0 zeros  ::1-9  o:10-49  O:50-99  #:>=100 zeros/cell")
    render_profiles(zero, m, n, tile_m, tile_n, label="zero-output")


def _tile_components(counts):
    """在 tile 粒度上做连通域（4-邻接 BFS），返回按面积降序的 (bbox_tiles, n_tiles)。"""
    mask_t = counts > 0
    tr, tc = mask_t.shape
    visited = np.zeros((tr, tc), dtype=bool)
    comps = []
    for r in range(tr):
        for c in range(tc):
            if not mask_t[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            r0, c0, r1, c1 = r, c, r, c
            area = 0
            while stack:
                cr, cc = stack.pop()
                area += 1
                r0, c0 = min(r0, cr), min(c0, cc)
                r1, c1 = max(r1, cr), max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < tr and 0 <= nc < tc and mask_t[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(((r0, c0, r1, c1), area))
    comps.sort(key=lambda x: x[1], reverse=True)
    return comps


def render_focus(mask_np, counts, m, n, tile_m, tile_n):
    """对最大错误连通域画框：bbox 元素级点阵（<=64x64）或 tile 级子图。"""
    comps = _tile_components(counts)
    if not comps:
        return
    (r0, c0, r1, c1), area = comps[0]
    total_tiles = counts.shape[0] * counts.shape[1]
    if area >= 0.95 * total_tiles:
        _banner("focus: largest error cluster")
        print("[visual]   cluster covers ~entire matrix; focus grid omitted (see density map)")
        return
    er0, ec0 = r0 * tile_m, c0 * tile_n
    er1, ec1 = min(m - 1, (r1 + 1) * tile_m - 1), min(n - 1, (c1 + 1) * tile_n - 1)
    _banner("focus: largest error cluster")
    print(f"[visual]   tile bbox rows [{r0},{r1}] x cols [{c0},{c1}]  (elements "
          f"[{er0},{er1}] x [{ec0},{ec1}], {area} error tiles)")

    h = er1 - er0 + 1
    w = ec1 - ec0 + 1
    if h <= 64 and w <= 64:
        print("[visual]   element-level dot grid ('.'=ok, 'X'=error):")
        print(f"[visual]   +{'-' * (2 * w)}+")
        for i in range(er0, er1 + 1):
            row_str = "".join(" X" if mask_np[i, j] else " ." for j in range(ec0, ec1 + 1))
            print(f"[visual]   |{row_str}|")
        print(f"[visual]   +{'-' * (2 * w)}+")
        print(f"[visual]   (element rows [{er0},{er1}], cols [{ec0},{ec1}])")
    else:
        sub = counts[r0:r1 + 1, c0:c1 + 1]
        sub_tr, sub_tc = sub.shape
        if sub_tr <= 60 and sub_tc <= 100:
            print(f"[visual]   tile-level grid (tile={tile_m}x{tile_n}, each cell = tile):")
            print(f"[visual]   +{'-' * (2 * sub_tc)}+")
            for rr in range(sub_tr):
                line = "".join(f" {_density_symbol(int(sub[rr, cc]))}" for cc in range(sub_tc))
                print(f"[visual]   |{line}|")
            print(f"[visual]   +{'-' * (2 * sub_tc)}+")
            print(f"[visual]   (tile rows [{r0},{r1}], tile cols [{c0},{c1}])")
        else:
            print("[visual]   (region too large for grid; see density map above)")


def render_top_n(golden_np, npu_np, mask, diff, n_top):
    """按 |diff| 降序列出 Top-N 错误坐标。"""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return
    diffs = diff[mask]
    if idx.shape[0] > n_top:
        take = np.argpartition(diffs, -n_top)[-n_top:]
        order = take[np.argsort(diffs[take])[::-1]]
    else:
        order = np.argsort(diffs)[::-1]
    shown = min(n_top, idx.shape[0])
    _banner(f"top-{shown} mismatches by |diff|  (total {idx.shape[0]}/{mask.size} = "
            f"{100.0*idx.shape[0]/mask.size:.2f}%)")
    print(f"[visual]   {'i':>8} {'j':>8} {'golden':>12} {'npu':>12} {'|diff|':>12}")
    for oi in order:
        i, j = idx[oi]
        print(f"[visual]   {i:>8} {j:>8} {_fmt_num(float(golden_np[i, j])):>12} "
              f"{_fmt_num(float(npu_np[i, j])):>12} {_fmt_num(float(diffs[oi])):>12}")


def pattern_hint(mask_np, counts, tile_m, tile_n):
    """根据错误分布形态给出定位提示。"""
    total = int(mask_np.sum())
    if total == 0:
        return ""
    m, n = mask_np.shape
    row_frac = float(mask_np.any(axis=1).sum()) / m
    col_frac = float(mask_np.any(axis=0).sum()) / n
    comps = _tile_components(counts)
    if row_frac >= 0.95 and col_frac >= 0.95:
        return ("whole matrix corrupted -> suspect structure/layout/transpose/fixpipe, "
                "not a local addressing slip")
    if row_frac >= 0.95:
        return ("errors span MOST rows but few cols (column-band) -> suspect B column / bias / nPos "
                "addressing")
    if col_frac >= 0.95:
        return ("errors span MOST cols but few rows (row-band) -> suspect A row / comm-window / mPos "
                "addressing")
    if not comps:
        return "errors present but tile-clustering empty (unexpected)"
    # 局部：看最大簇元素面积占比与簇数
    (r0, c0, r1, c1), area = comps[0]
    bbox_elems = (r1 - r0 + 1) * (c1 - c0 + 1) * tile_m * tile_n
    biggest_frac = min(1.0, float(bbox_elems) / max(1, total))
    n_comps = len(comps)
    if n_comps >= max(16, total // max(1, m // 100)) and area == 1:
        return (f"errors scattered as {n_comps} isolated tile points -> likely local/numeric, "
                f"inspect a top mismatch in detail")
    return (f"errors localized ({n_comps} clusters, biggest covers {100.0*biggest_frac:.1f}% of mismatches) "
            f"-> suspect tile/slice coordinates (mPos/nPos/rank-block slicing); "
            f"see top clusters & density map")


def render_visual(golden_tensor, npu_tensor, m, n, opts):
    """可视化标注层入口。opts: --visual on/off, --top, --tile。"""
    golden_np = golden_tensor.float().numpy()
    npu_np = npu_tensor.float().numpy()
    mask, diff = build_mismatch_mask(golden_np, npu_np, ERROR_TOL, ERROR_TOL)
    total = int(mask.sum())
    if total == 0:
        _banner("no mismatches  (all elements within tolerance) - all '.'")
        return

    tile_m, tile_n = opts.tile_m, opts.tile_n
    counts = _tile_counts(mask, m, n, tile_m, tile_n)

    _banner(f"mismatch visualization  (total {total}/{m*n} = "
            f"{100.0*total/(m*n):.2f}% elements over tol)")
    render_zero_output(npu_np, m, n, tile_m, tile_n)
    render_profiles(mask, m, n, tile_m, tile_n)
    render_top_n(golden_np, npu_np, mask, diff, max(1, opts.top))

    hint = pattern_hint(mask, counts, tile_m, tile_n)
    if hint:
        _banner("pattern hint")
        print(f"[visual]   {hint}")

    render_density_map(mask, m, n, tile_m, tile_n)
    render_focus(mask, counts, m, n, tile_m, tile_n)


def _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n, rank_id=-1):
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = m * n
    over_tol = (abs_err > ERROR_TOL).sum().item()

    rank_prefix = f"[Rank {rank_id}] " if rank_id >= 0 else ""
    print(
        f"\n{rank_prefix}[verify] shape=({m}, {n}), elements={numel} - summary (large matrix, full tensors omitted)"
    )
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
        f"rmse={(diff.pow(2).mean().sqrt()).item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {numel}")

    cr = min(CORNER_ROWS, m)
    cc = min(CORNER_COLS, n)
    if cr > 0 and cc > 0:
        print(f"  cpu golden (top-left {cr}x{cc}):\n{golden_tensor[:cr, :cc]}")
        print(f"  npu output (top-left {cr}x{cc}):\n{npu_output_tensor[:cr, :cc]}")


def verify_single_rank(m, n, rank_id, base_dir="./output", opts=None, dtype="fp16"):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = read_matrix(output_path, dtype)
    golden = read_matrix(golden_path, dtype)

    expected_size = m * n
    if output.numel() != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] npu output size {output.numel()} != expected size {expected_size}"
        )
    if golden.numel() != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] cpu output size {golden.numel()} != expected size {expected_size}"
        )

    npu_output_tensor = output.reshape(m, n)
    golden_tensor = golden.reshape(m, n)

    numel = m * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print(f"\n[Rank {rank_id}] cpu golden:\n", golden_tensor)
        print(f"[Rank {rank_id}] npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n, rank_id)

    if opts is not None and opts.visual == "on":
        render_visual(golden_tensor, npu_output_tensor, m, n, opts)

    # 双千分之一（与量化版一致，适用所有模式）：
    # 相对误差 > ERROR_TOL 的元素占比 ≤ FAIL_RATE_LIMIT 算过。
    diff = (npu_output_tensor.float() - golden_tensor.float()).abs()
    thr = ERROR_TOL + ERROR_TOL * golden_tensor.float().abs()
    fail_rate = (diff > thr).float().mean().item()
    print(
        f"[Rank {rank_id}] 双千分之一: fail_rate={fail_rate*100:.5f}% "
        f"(limit {FAIL_RATE_LIMIT*100}%)"
    )
    return fail_rate <= FAIL_RATE_LIMIT


def verify_result(m, n, rank_num, base_dir="./output", opts=None, dtype="fp16"):
    all_pass = True
    results = []

    print(f"\n{'=' * 60}")
    print(f"Verifying outputs for {rank_num} ranks")
    print(f"Matrix shape: M={m}, N={n}, dtype={dtype}")
    print("All ranks should produce the same result after All2All")
    print(f"{'=' * 60}")

    for rank_id in range(rank_num):
        try:
            res = verify_single_rank(m, n, rank_id, base_dir, opts, dtype)
            results.append(
                (rank_id, bool(res), None if res else "fail_rate exceeded limit")
            )
            if res:
                print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
            else:
                print(
                    f"[FAIL] Rank {rank_id}: 双千分之一 fail_rate exceeded "
                    f"(ERROR_TOL={ERROR_TOL}, limit={FAIL_RATE_LIMIT})"
                )
                all_pass = False
        except Exception as e:
            results.append((rank_id, False, str(e)))
            print(f"[FAIL] Rank {rank_id}: {e}")
            all_pass = False

    print(f"\n{'=' * 60}")
    print("Verification Summary:")
    print(f"{'=' * 60}")
    for rank_id, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Rank {rank_id}: {status}")
        if error:
            print(f"    Error: {error}")

    return all_pass


def _parse_tile(s):
    """解析 --tile：'256' -> (256,256)；'256x128' -> (256,128)。"""
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    v = int(s)
    return v, v


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Verify NPU AllToAllMatmul output against CPU golden, with mismatch visualization.",
    )
    parser.add_argument("m", type=int, help="matrix M dimension")
    parser.add_argument("n", type=int, help="matrix N dimension")
    parser.add_argument("rank_num", type=int, help="number of ranks")
    parser.add_argument("base_dir", nargs="?", default="./output",
                        help="base directory of output files (default: ./output)")
    parser.add_argument("--visual", choices=["on", "off"], default="on",
                        help="mismatch visualization on/off (default: on)")
    parser.add_argument("--top", type=int, default=20,
                        help="number of top mismatches to list (default: 20)")
    parser.add_argument("--tile", type=str, default="16x16",
                        help="analysis tile (baseM x baseN) granularity, e.g. 256x256 (default: 16x16)")
    parser.add_argument("--dtype", type=str, default="fp16",
                        help="output dtype: fp16 (default) | bf16")
    args = parser.parse_args(argv)
    args.tile_m, args.tile_n = _parse_tile(args.tile)
    if args.tile_m < 1 or args.tile_n < 1:
        parser.error("--tile must be >= 1 (e.g. 256 or 256x128)")
    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.dtype not in ("fp16", "bf16"):
        parser.error("--dtype must be 'fp16' or 'bf16'")
    return args


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    try:
        all_pass = verify_result(args.m, args.n, args.rank_num, args.base_dir, args, args.dtype)
        if not all_pass:
            raise ValueError("[ERROR] Some NPU results differ from CPU.\n")
        print(f"\n[ALL PASS] All {args.rank_num} ranks passed verification!\n")

    except Exception as e:
        print(e)
        sys.exit(1)

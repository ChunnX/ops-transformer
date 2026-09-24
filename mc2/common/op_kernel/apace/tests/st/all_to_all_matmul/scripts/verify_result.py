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

# fp16 输入 + fp16 输出，fp32 中间累加，理论相对误差 ~1e-3。
# atol/rtol 取 1e-2 对齐低精度算子通用验收门槛。
ERROR_TOL = 1e-2
DATA_TYPE = np.float16
FULL_TENSOR_PRINT_MAX_ELEMENTS = 1024
CORNER_ROWS = 4
CORNER_COLS = 4

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


def build_mismatch_mask(golden_np, npu_np, atol, rtol):
    """与 torch.allclose(golden, npu, rtol, atol) 一致的错误 mask。"""
    diff = np.abs(npu_np.astype(np.float32) - golden_np.astype(np.float32))
    thr = atol + rtol * np.abs(npu_np.astype(np.float32))
    return diff > thr, diff


def _tile_counts(mask_np, m, n, tile):
    """把 (m,n) bool mask 按 tile x tile 分块求和，返回 (tr, tc) 计数矩阵（补齐零行/零列）。"""
    pm = (tile - m % tile) % tile
    pn = (tile - n % tile) % tile
    padded = np.zeros((m + pm, n + pn), dtype=np.int64)
    padded[:m, :n] = mask_np.astype(np.int64)
    tr, tc = (m + pm) // tile, (n + pn) // tile
    return padded.reshape(tr, tile, tc, tile).sum(axis=(1, 3))


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


def render_density_map(mask_np, m, n, tile, width_limit=120):
    """缩略密度图：自动粗化到 <=40 行 x <=80 列，避免超大矩阵刷屏。"""
    rows_limit, cols_limit = 40, 80
    map_tile = tile
    while (m + map_tile - 1) // map_tile > rows_limit or (n + map_tile - 1) // map_tile > cols_limit:
        map_tile *= 2
    counts = _tile_counts(mask_np, m, n, map_tile)
    tr, tc = counts.shape
    coarsen = f" (auto-coarsened from {tile}x{tile} to {map_tile}x{map_tile})" if map_tile != tile else ""
    print(f"\n[visual] error density map  (each cell = {map_tile}x{map_tile} elements, "
          f"tile grid {tr} rows x {tc} cols){coarsen}")
    print("[visual]   symbol: .:0  ::1-9  o:10-49  O:50-99  #:>=100 errors/cell")
    cell_w = 3  # 每列固定 3 字符（f"{c:>3}" / f"{sym:^3}"）
    page = max(1, (width_limit - 8) // cell_w)
    for c0 in range(0, tc, page):
        c1 = min(tc, c0 + page)
        hdr = "       " + "".join(f"{c:>3}" for c in range(c0, c1))
        print(f"[visual]    col-tile>{hdr}")
        for r in range(tr):
            line = f"[visual]    {r*map_tile:>7}" + "".join(
                f"{_density_symbol(int(counts[r, c])):^3}" for c in range(c0, c1))
            print(line)


def render_profiles(mask_np, m, n, tile):
    """行/列错误 profile：元素级连续段 + 每 tile 的错误 tile 分布。"""
    row_any = mask_np.any(axis=1)
    col_any = mask_np.any(axis=0)
    bad_rows = int(row_any.sum())
    bad_cols = int(col_any.sum())
    print(f"\n[visual] row/col profile: bad rows {bad_rows}/{m} ({100.0*bad_rows/m:.1f}%), "
          f"bad cols {bad_cols}/{n} ({100.0*bad_cols/n:.1f}%)")

    # 接近全坏：折叠明细，避免刷屏
    if bad_rows >= 0.98 * m and bad_cols >= 0.98 * n:
        print("[visual]   (corruption covers ~entire matrix; row/col runs and tile lists omitted)")
        return

    rr, dropped_r = _truncate_runs(_bool_runs(row_any))
    if rr:
        seg = " ".join(f"[{a},{b}]({b-a+1})" for a, b in rr)
        if dropped_r:
            seg += f" ... +{dropped_r} more"
        print(f"[visual]   bad-row runs: {seg}")
    cr, dropped_c = _truncate_runs(_bool_runs(col_any))
    if cr:
        seg = " ".join(f"[{a},{b}]({b-a+1})" for a, b in cr)
        if dropped_c:
            seg += f" ... +{dropped_c} more"
        print(f"[visual]   bad-col runs: {seg}")

    # 每 tile 的错误 tile 分布（仅打印有错误的，cap 8）
    counts = _tile_counts(mask_np, m, n, tile)
    rnz = [(r, int((counts[r] > 0).sum())) for r in range(counts.shape[0]) if (counts[r] > 0).any()]
    cnz = [(c, int((counts[:, c] > 0).sum())) for c in range(counts.shape[1]) if (counts[:, c] > 0).any()]
    if rnz:
        rnz = sorted(rnz, key=lambda x: x[1], reverse=True)
        seg = " ".join(f"row-tile{r}({cnt})" for r, cnt in rnz[:8])
        if len(rnz) > 8:
            seg += f" ... +{len(rnz)-8} more"
        print(f"[visual]   row-tiles containing errors (tile={tile}): {seg}")
    if cnz:
        cnz = sorted(cnz, key=lambda x: x[1], reverse=True)
        seg = " ".join(f"col-tile{c}({cnt})" for c, cnt in cnz[:8])
        if len(cnz) > 8:
            seg += f" ... +{len(cnz)-8} more"
        print(f"[visual]   col-tiles containing errors (tile={tile}): {seg}")


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


def render_focus(mask_np, counts, m, n, tile):
    """对最大错误连通域画框：bbox 元素级点阵（<=64x64）或 tile 级子图。"""
    comps = _tile_components(counts)
    if not comps:
        return
    (r0, c0, r1, c1), area = comps[0]
    total_tiles = counts.shape[0] * counts.shape[1]
    if area >= 0.95 * total_tiles:
        print("\n[visual] largest error cluster covers ~entire matrix; focus grid omitted "
              "(see density map)")
        return
    er0, ec0 = r0 * tile, c0 * tile
    er1, ec1 = min(m - 1, (r1 + 1) * tile - 1), min(n - 1, (c1 + 1) * tile - 1)
    print(f"\n[visual] largest error cluster: element bbox rows [{er0},{er1}] x cols [{ec0},{ec1}] "
          f"({area} error tiles, element area {(er1-er0+1)*(ec1-ec0+1)})")

    h = er1 - er0 + 1
    w = ec1 - ec0 + 1
    if h <= 64 and w <= 64:
        print("[visual] element-level dot grid ('.'=ok, 'X'=error):")
        print(f"[visual]   +{'-' * (2 * w)}+")
        for i in range(er0, er1 + 1):
            row_str = "".join(" X" if mask_np[i, j] else " ." for j in range(ec0, ec1 + 1))
            print(f"[visual]   |{row_str}|")
        print(f"[visual]   +{'-' * (2 * w)}+")
        print(f"[visual]   (element bbox rows [{er0},{er1}], cols [{ec0},{ec1}])")
    else:
        sub = counts[r0:r1 + 1, c0:c1 + 1]
        sub_tr, sub_tc = sub.shape
        if sub_tr <= 60 and sub_tc <= 100:
            print(f"[visual] tile-level grid (tile={tile}, each cell = tile):")
            print(f"[visual]   +{'-' * (2 * sub_tc)}+")
            for rr in range(sub_tr):
                line = "".join(f" {_density_symbol(int(sub[rr, cc]))}" for cc in range(sub_tc))
                print(f"[visual]   |{line}|")
            print(f"[visual]   +{'-' * (2 * sub_tc + 1)}+")
            print(f"[visual]   (tile rows [{r0},{r1}], tile cols [{c0},{c1}])")
        else:
            print(f"[visual]   (region too large for grid; see density map above)")


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
    print(f"\n[visual] top {min(n_top, idx.shape[0])} mismatches by |diff| "
          f"(total mismatches: {idx.shape[0]} / {mask.size} = {100.0*idx.shape[0]/mask.size:.2f}%):")
    print(f"[visual]   {'i':>8} {'j':>8} {'golden':>12} {'npu':>12} {'|diff|':>12}")
    for oi in order:
        i, j = idx[oi]
        print(f"[visual]   {i:>8} {j:>8} {_fmt_num(float(golden_np[i, j])):>12} "
              f"{_fmt_num(float(npu_np[i, j])):>12} {_fmt_num(float(diffs[oi])):>12}")


def pattern_hint(mask_np, counts, tile):
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
    bbox_elems = (r1 - r0 + 1) * (c1 - c0 + 1) * tile * tile
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
    golden_np = golden_tensor.numpy().astype(np.float32)
    npu_np = npu_tensor.numpy().astype(np.float32)
    mask, diff = build_mismatch_mask(golden_np, npu_np, ERROR_TOL, ERROR_TOL)
    total = int(mask.sum())
    if total == 0:
        print("\n[visual] no mismatches (all elements within tolerance) - all '.'")
        return

    tile = max(1, opts.tile)
    counts = _tile_counts(mask, m, n, tile)

    print(f"\n[visual] ===== mismatch visualization (total {total} elements over tol) =====")
    render_profiles(mask, m, n, tile)
    render_top_n(golden_np, npu_np, mask, diff, max(1, opts.top))

    hint = pattern_hint(mask, counts, tile)
    if hint:
        print(f"\n[visual] pattern hint: {hint}")

    render_density_map(mask, m, n, tile)
    render_focus(mask, counts, m, n, tile)


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


def verify_single_rank(m, n, rank_id, base_dir="./output", opts=None):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = np.fromfile(output_path, dtype=DATA_TYPE)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE)

    expected_size = m * n
    if output.size != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] npu output size {output.size} != expected size {expected_size}"
        )
    if golden.size != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] cpu output size {golden.size} != expected size {expected_size}"
        )

    npu_output_tensor = torch.from_numpy(output).reshape(m, n)
    golden_tensor = torch.from_numpy(golden).reshape(m, n)

    numel = m * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print(f"\n[Rank {rank_id}] cpu golden:\n", golden_tensor)
        print(f"[Rank {rank_id}] npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n, rank_id)

    if opts is not None and opts.visual == "on":
        render_visual(golden_tensor, npu_output_tensor, m, n, opts)

    return torch.allclose(
        golden_tensor, npu_output_tensor, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True
    )


def verify_result(m, n, rank_num, base_dir="./output", opts=None):
    all_pass = True
    results = []

    print(f"\n{'=' * 60}")
    print(f"Verifying outputs for {rank_num} ranks")
    print(f"Matrix shape: M={m}, N={n}")
    print("All ranks should produce the same result after All2All")
    print(f"{'=' * 60}")

    for rank_id in range(rank_num):
        try:
            res = verify_single_rank(m, n, rank_id, base_dir, opts)
            results.append(
                (rank_id, bool(res), None if res else "allclose returned False")
            )
            if res:
                print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
            else:
                print(
                    f"[FAIL] Rank {rank_id}: allclose returned False (rtol=atol={ERROR_TOL})"
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
    parser.add_argument("--tile", type=int, default=16,
                        help="analysis tile granularity, in elements (default: 16)")
    args = parser.parse_args(argv)
    if args.tile < 1:
        parser.error("--tile must be >= 1")
    if args.top < 1:
        parser.error("--top must be >= 1")
    return args


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    try:
        all_pass = verify_result(args.m, args.n, args.rank_num, args.base_dir, args)
        if not all_pass:
            raise ValueError("[ERROR] Some NPU results differ from CPU.\n")
        print(f"\n[ALL PASS] All {args.rank_num} ranks passed verification!\n")

    except Exception as e:
        print(e)
        sys.exit(1)

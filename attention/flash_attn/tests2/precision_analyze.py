import os

import torch

from compare import DEFAULT_TOLERANCE

SEP = "━" * 80

DEFAULT_SCATTER_LIMIT = 10
DEFAULT_SEGMENT_LIMIT = 10

_LAYOUT_COLUMNS = {"BNSD": ["B", "N", "S", "D"], "BSND": ["B", "S", "N", "D"], "TND": ["T", "N", "D"]}


def _col_names(shape, layout):
    return _LAYOUT_COLUMNS.get(layout, [str(i) for i in range(len(shape))])


# ═══════════════════════════════════════════════════════════════
# 表格渲染
# ═══════════════════════════════════════════════════════════════

def _render_table(col_names, col_types, rows):
    """先扫描算列宽, 统一 pad 输出。col_types: 'c'=坐标左对齐, 'n'=数字右对齐。"""
    widths = [len(h) for h in col_names]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    header = "    " + "  ".join(h.rjust(w) for h, w in zip(col_names, widths))
    sep_line = "    " + "  ".join("─" * w for w in widths)

    lines = [header, sep_line]
    for row in rows:
        line = "    " + "  ".join(
            str(c).rjust(w) if col_types[i] == "n" else str(c).ljust(w)
            for i, (c, w) in enumerate(zip(row, widths))
        )
        lines.append(line)
    return lines


# ═══════════════════════════════════════════════════════════════
# 连续段查找
# ═══════════════════════════════════════════════════════════════

def _find_ranges(fail_coords, shape):
    """fail_coords: 按前缀分组的失败坐标列表。在最后维度上找连续区间。"""
    ndim = len(shape)
    dim_size = shape[-1]
    segments = []   # [(prefix, d_range)]
    scatter = []    # [(full_coord)]

    for prefix, group in _group_by_prefix(fail_coords, ndim - 1):
        dim_vals = sorted(set(c[-1] for c in group))
        seg_start = seg_end = dim_vals[0]
        for d in dim_vals[1:]:
            if d == seg_end + 1:
                seg_end = d
            else:
                _emit(segments, scatter, prefix, seg_start, seg_end, dim_size)
                seg_start = seg_end = d
        _emit(segments, scatter, prefix, seg_start, seg_end, dim_size)

    return segments, scatter


def _group_by_prefix(coords, prefix_len):
    """按前 prefix_len 维分组, 保持输入顺序。"""
    groups = {}
    order = []
    for c in coords:
        key = tuple(c[:prefix_len])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    return [(k, groups[k]) for k in order]


def _emit(segments, scatter, prefix, start, end, dim_size):
    if start == end:
        scatter.append(tuple(prefix) + (start,))
    elif start == 0 and end == dim_size - 1:
        segments.append((tuple(prefix), (-1, -1)))  # -1 标记全覆盖
    else:
        segments.append((tuple(prefix), (start, end)))


# ═══════════════════════════════════════════════════════════════
# 紧凑报告
# ═══════════════════════════════════════════════════════════════

def _compact_rows(fail_indices, shape, layout, verbose):
    """构建紧凑表行: 散点+片段。返回(rows, truncated)。"""
    ndim = len(shape)
    col_names = _col_names(shape, layout)

    # 扁平索引 → 坐标列表
    coords = _flat_to_coords(fail_indices, shape)

    segments, scatter = _find_ranges(coords, shape)

    rows = []

    # 片段行
    seg_limit = DEFAULT_SEGMENT_LIMIT if verbose is None else len(segments)
    for prefix, sd in segments[:seg_limit]:
        row = _fmt_prefix_range(prefix, sd, shape, col_names)
        count = _range_count(prefix, sd, shape)
        rows.append(row + (f"{count:,}",))

    # 散点行
    sct_limit = DEFAULT_SCATTER_LIMIT if verbose is None else len(scatter)
    for coord in scatter[:sct_limit]:
        rows.append(tuple(str(v) for v in coord) + ("1",))

    # 全量模式: 不合并, 逐点输出
    if verbose == "all":
        rows = [tuple(str(v) for v in c) + ("1",) for c in coords]

    # 截断提示
    truncated = False
    if verbose is None:
        hidden_s = len(scatter) - DEFAULT_SCATTER_LIMIT
        hidden_g = len(segments) - DEFAULT_SEGMENT_LIMIT
        if hidden_s > 0 or hidden_g > 0:
            parts = []
            if hidden_s > 0:
                parts.append(f"{hidden_s} 散点")
            if hidden_g > 0:
                parts.append(f"{hidden_g} 片段")
            rows.append(("... " + " + ".join(parts)
                        + " (--verbose_diff 查看)",))
            truncated = True

    return rows, len(col_names), truncated


def _flat_to_coords(fail_indices, shape):
    """扁平索引 → 多维坐标列表, 按坐标排序。"""
    ndim = len(shape)
    strides = [1]
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)

    coords = []
    for flat in fail_indices.tolist():
        c = []
        r = flat
        for i in range(ndim):
            c.append(r // strides[i])
            r %= strides[i]
        coords.append(tuple(c))
    return sorted(coords)


def _fmt_prefix_range(prefix, d_range, shape, col_names):
    """前缀 + D区间 → 行单元格元组。(-1,-1) 表示 D 全覆盖。"""
    parts = list(prefix)
    if d_range == (-1, -1):
        parts = list(prefix) + [f"[0..{shape[-1] - 1}]"]
    elif d_range[0] == d_range[1]:
        parts = list(prefix) + [str(d_range[0])]
    else:
        parts = list(prefix) + [f"[{d_range[0]}..{d_range[1]}]"]
    return tuple(parts)


def _range_count(prefix, d_range, shape):
    """片段覆盖的元素数: D区间长度 × 前缀覆盖数(单前缀下=1)。"""
    if d_range == (-1, -1):
        return shape[-1]
    return d_range[1] - d_range[0] + 1


# ═══════════════════════════════════════════════════════════════
# 概要 / 预览
# ═══════════════════════════════════════════════════════════════

def _summary(key, shape, dtype, fail_cnt, total, max_abs, mean_abs, atol, rtol, layout):
    pct = fail_cnt / total * 100 if total else 0.0
    return [
        f"  key='{key}'  shape={tuple(shape)}  dtype={dtype}  layout={layout}",
        f"  总数: {total:,}   失败: {fail_cnt:,} ({pct:.4f}%)   "
        f"MaxΔ: {max_abs:.6f}  MeanΔ: {mean_abs:.6f}  "
        f"阈值: atol={atol} rtol={rtol}",
    ]


def _preview(cpu_flat, npu_flat, total):
    lines = [f"\n  [前10后10预览]", f"  {'idx':>10}  {'cpu':>14}  {'npu':>14}",
             f"  {'─' * 10}  {'─' * 14}  {'─' * 14}"]
    n = min(10, total)
    for i in range(n):
        lines.append(f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}")
    if total > 20:
        lines.append(f"  {'...':>10}")
        for i in range(total - 10, total):
            lines.append(f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}")
    elif total > 10:
        for i in range(10, total):
            lines.append(f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}")
    return lines


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def analyze_failure(
    cpu_out: dict,
    npu_out: dict,
    compare_result: dict,
    case_dir: str = None,
    verbose: str | None = None,
    layout: str = "",
) -> str:
    """精度失败分析: 概要 → 坐标化紧凑表 → Max-RE → 预览, 写 fail_analysis.txt"""
    lines = [SEP, "  精度失败分析报告", SEP]

    for key, result in compare_result.items():
        if not isinstance(result, dict):
            continue
        if result.get("passed", True):
            continue
        if key not in cpu_out or key not in npu_out:
            lines.append(f"\n  [WARN] key '{key}' 不在 cpu_out/npu_out 中, 跳过")
            continue

        cpu_t = cpu_out[key].float()
        npu_t = npu_out[key].float()

        if cpu_t.shape != npu_t.shape:
            lines.append(
                f"\n  [ERROR] key='{key}' shape不匹配: "
                f"cpu={tuple(cpu_t.shape)} npu={tuple(npu_t.shape)}"
            )
            continue

        diff = torch.abs(cpu_t - npu_t)
        cpu_dtype = cpu_out[key].dtype
        default_tol = DEFAULT_TOLERANCE.get(cpu_dtype, {"atol": 1e-3, "rtol": 1e-3})
        rtol = result.get("rtol", default_tol["rtol"])
        atol = result.get("atol", default_tol["atol"])
        fail_mask = ~torch.isclose(cpu_t, npu_t, rtol=rtol, atol=atol)

        fail_cnt = int(fail_mask.sum().item())
        total = cpu_t.numel()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        # ── 概要 ──
        lines += _summary(key, cpu_t.shape, cpu_dtype, fail_cnt, total,
                          max_abs, mean_abs, atol, rtol, layout or "")

        if fail_cnt > 0:
            lines.append("  " + "═" * 64)

            # ── 坐标化紧凑表 ──
            fail_indices = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            col_names = _col_names(cpu_t.shape, layout or "")

            rows, axis_width, truncated = _compact_rows(
                fail_indices, tuple(cpu_t.shape), layout or "", verbose)

            if rows:
                table_header = col_names + ["元素数"]
                col_types = (["c"] * axis_width) + ["n"]
                lines += _render_table(table_header, col_types, rows)
                lines.append("")

            if truncated:
                lines.append(
                    "    (使用 --verbose_diff fold 查看全部, "
                    "--verbose_diff all 查看全部点位)"
                )
                lines.append("")

            # ── Max-RE 前3 ──
            cpu_flat = cpu_t.reshape(-1)
            npu_flat = npu_t.reshape(-1)
            diff_flat = diff.reshape(-1)
            rel = torch.abs(cpu_flat - npu_flat) / (torch.abs(cpu_flat) + 1e-6)

            lines.append(f"  [Max-RE 前3] (绝对误差最大)")
            lines.append(f"  {'idx':>10}  {'cpu':>14}  {'npu':>14}  {'absErr':>12}  {'relErr':>10}")
            lines.append(f"  {'─' * 10}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 10}")
            for i in diff_flat.topk(min(3, total)).indices.tolist():
                lines.append(
                    f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}"
                    f"  {diff_flat[i].item():>12.8f}  {rel[i].item():>10.6f}"
                )

            # ── 预览 (仅默认模式) ──
            if verbose is None:
                lines += _preview(cpu_flat, npu_flat, total)

    lines.append(f"\n{SEP}")
    report = "\n".join(lines)

    if case_dir is not None:
        os.makedirs(case_dir, exist_ok=True)
        report_file = os.path.join(case_dir, "fail_analysis.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def analyze_dimensions(cpu_out: dict, npu_out: dict, compare_result: dict) -> str:
    raise NotImplementedError("维度分析待实现")


def analyze_continuity(cpu_out: dict, npu_out: dict, compare_result: dict) -> str:
    raise NotImplementedError("连续性分析待实现")


def visualize_failure(cpu_out: dict, npu_out: dict, compare_result: dict, save_dir: str = None) -> str:
    raise NotImplementedError("可视化待实现")

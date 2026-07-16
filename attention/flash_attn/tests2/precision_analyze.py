import os

import torch

from compare import DEFAULT_TOLERANCE

SEP = "━" * 80

DEFAULT_SCATTER_LIMIT = 10
DEFAULT_SEGMENT_LIMIT = 10
LARGE_FAIL_THRESHOLD = 1_000_000  # 超过此数量跳过坐标分析, 仅显示概要

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
# 连续段查找 (矢量化)
# ═══════════════════════════════════════════════════════════════

def _flat_to_coords(fail_indices, shape):
    """矢量化: 扁平索引 → 坐标张量 (N, ndim), 按字典序排序。"""
    ndim = len(shape)
    strides = [1]
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)

    coords = torch.zeros(len(fail_indices), ndim, dtype=torch.int64)
    r = fail_indices.long()
    for i in range(ndim):
        coords[:, i] = r // strides[i]
        r = r % strides[i]

    # 稳定排序 (从最后一维到第一维)
    order = torch.arange(len(coords), dtype=torch.int64)
    for i in range(ndim - 1, -1, -1):
        order = order[coords[order, i].argsort(stable=True)]
    return coords[order]


def _find_ranges(coords_t, shape):
    """矢量化: 在已排序坐标张量上找前缀分组和D连续段。接受 list-of-tuples 或 tensor。"""
    if isinstance(coords_t, list):
        coords_t = torch.tensor(coords_t, dtype=torch.int64)
    ndim = len(shape)
    dim_size = shape[-1]
    n = len(coords_t)
    segments = []
    scatter = []

    if n == 0:
        return segments, scatter

    if ndim == 1:
        # 1D: 所有点同一前缀(空), 直接在D维找连续段
        _emit_ranges(segments, scatter, (), coords_t[:, 0], dim_size)
        return segments, scatter

    # 找前缀变化边界: 前 ndim-1 维任意变化处
    prefix_cols = coords_t[:, :ndim - 1]
    prefix_change = torch.zeros(n, dtype=torch.bool)
    prefix_change[1:] = (prefix_cols[1:] != prefix_cols[:-1]).any(dim=1)
    boundaries = prefix_change.nonzero(as_tuple=True)[0].tolist() + [n]

    prev = 0
    for b in boundaries:
        prefix = tuple(coords_t[prev, :ndim - 1].tolist())
        d_col = coords_t[prev:b, -1]
        _emit_ranges(segments, scatter, prefix, d_col, dim_size)
        prev = b

    return segments, scatter


def _emit_ranges(segments, scatter, prefix, d_vals, dim_size):
    """从D值列中找出连续区间并发射片段/散点。"""
    if len(d_vals) == 0:
        return
    unique_d = torch.unique(d_vals)
    if len(unique_d) == 0:
        return
    # 找间隙
    gaps = torch.cat([torch.tensor([True]),
                      (unique_d[1:] - unique_d[:-1]) > 1,
                      torch.tensor([True])])
    gap_pos = gaps.nonzero(as_tuple=True)[0]
    for j in range(len(gap_pos) - 1):
        start = int(unique_d[gap_pos[j]])
        end = int(unique_d[gap_pos[j + 1] - 1])
        if start == end:
            scatter.append(prefix + (start,))
        elif start == 0 and end == dim_size - 1:
            segments.append((prefix, (-1, -1)))
        else:
            segments.append((prefix, (start, end)))


# ═══════════════════════════════════════════════════════════════
# 紧凑报告
# ═══════════════════════════════════════════════════════════════

def _merge_full_coverage(segments, shape):
    """逐级合并全覆盖: 从 S 向 B (从最内层 prefix 维向外)。合并后 prefix 长度不变。"""
    if not segments or len(segments) <= 1:
        return segments

    ndim = len(shape)
    # prefix_len: ndim-2(S) → ndim-3(N)... → 0
    for prefix_len in range(ndim - 2, -1, -1):
        groups = {}
        for prefix, d_range in segments:
            key = prefix[:prefix_len] if prefix_len > 0 else ()
            groups.setdefault(key, []).append((prefix, d_range))

        merged = []
        for key, items in groups.items():
            all_full = all(dr == (-1, -1) for _, dr in items)
            positions = sorted({p[prefix_len] for p, _ in items})
            if all_full and len(positions) == shape[prefix_len]:
                # 合并: prefix 保持不变但该位改为全覆盖标记
                template = list(items[0][0])
                template[prefix_len] = (-1, -1)
                merged.append((tuple(template), (-1, -1)))
            else:
                merged.extend(items)
        segments = merged

    return segments


def _compact_rows(fail_indices, shape, layout, verbose):
    """构建紧凑表行: 散点+片段。返回(rows, truncated)。"""
    ndim = len(shape)
    col_names = _col_names(shape, layout)

    # 扁平索引 → 坐标列表
    coords = _flat_to_coords(fail_indices, shape)

    segments, scatter = _find_ranges(coords, shape)
    segments = _merge_full_coverage(segments, shape)

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




def _fmt_coord_val(v, size):
    """坐标值或区间 → 字符串。"""
    if isinstance(v, tuple):
        if v == (-1, -1):
            return f"[0..{size - 1}]"
        return f"[{v[0]}..{v[1]}]"
    return str(v)


def _fmt_prefix_range(prefix, d_range, shape, col_names):
    """前缀 + D区间 → 行单元格元组。"""
    parts = [_fmt_coord_val(v, shape[i]) for i, v in enumerate(prefix)]
    parts.append(_fmt_coord_val(d_range, shape[-1]))
    return tuple(parts)


def _range_count(prefix, d_range, shape):
    """片段覆盖的元素数。"""
    count = 1
    for i, v in enumerate(prefix):
        if v == (-1, -1):
            count *= shape[i]
    if d_range == (-1, -1):
        count *= shape[-1]
    else:
        count *= d_range[1] - d_range[0] + 1
    return count


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

            fail_indices = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            col_names = _col_names(cpu_t.shape, layout or "")

            cpu_flat = cpu_t.reshape(-1)
            npu_flat = npu_t.reshape(-1)
            diff_flat_t = diff.reshape(-1)

            if fail_cnt > LARGE_FAIL_THRESHOLD:
                lines.append(f"    失败点数过多 ({fail_cnt:,} > {LARGE_FAIL_THRESHOLD:,}), "
                             f"跳过坐标分析以节省内存")
                lines.append("    (尝试缩小 --case_id 范围或使用 --verbose_diff all 查看全部)")
                lines.append("")
            elif verbose == "all":
                # 逐点坐标 + cpu/npu/diff 值
                coords = _flat_to_coords(fail_indices, tuple(cpu_t.shape))
                all_rows = []
                for flat, coord in zip(fail_indices.tolist(), coords):
                    all_rows.append(
                        tuple(str(v) for v in coord)
                        + (f"{cpu_flat[flat].item():>+.6f}",
                           f"{npu_flat[flat].item():>+.6f}",
                           f"{diff_flat_t[flat].item():.6f}")
                    )
                if all_rows:
                    table_header = col_names + ["cpu", "npu", "diff"]
                    col_types = (["c"] * len(col_names)) + ["n", "n", "n"]
                    lines += _render_table(table_header, col_types, all_rows)
            else:
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
            rel = torch.abs(cpu_flat - npu_flat) / (torch.abs(cpu_flat) + 1e-6)

            lines.append(f"  [Max-RE 前3] (绝对误差最大)")
            lines.append(f"  {'idx':>10}  {'cpu':>14}  {'npu':>14}  {'absErr':>12}  {'relErr':>10}")
            lines.append(f"  {'─' * 10}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 10}")
            for i in diff_flat_t.topk(min(3, total)).indices.tolist():
                lines.append(
                    f"  {i:>10}  {cpu_flat[i].item():>+14.8f}  {npu_flat[i].item():>+14.8f}"
                    f"  {diff_flat_t[i].item():>12.8f}  {rel[i].item():>10.6f}"
                )

            # ── 预览 (仅默认模式) ──
            if verbose != "all":
                lines += _preview(cpu_flat, npu_flat, total)

    lines += analyze_dimensions(cpu_out, npu_out, compare_result)
    lines += analyze_continuity(cpu_out, npu_out, compare_result)
    lines += visualize_failure(cpu_out, npu_out, compare_result, case_dir)
    lines.append(f"\n{SEP}")
    report = "\n".join(lines)

    if case_dir is not None:
        os.makedirs(case_dir, exist_ok=True)
        report_file = os.path.join(case_dir, "fail_analysis.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def analyze_dimensions(cpu_out: dict, npu_out: dict, compare_result: dict) -> list[str]:
    """维度分析扩展点。算子可 override, 默认不输出。"""
    return []


def analyze_continuity(cpu_out: dict, npu_out: dict, compare_result: dict) -> list[str]:
    """连续性分析扩展点。算子可 override, 默认不输出。"""
    return []


def visualize_failure(cpu_out: dict, npu_out: dict, compare_result: dict, save_dir: str = None) -> list[str]:
    """可视化扩展点。算子可 override, 默认不输出。"""
    return []

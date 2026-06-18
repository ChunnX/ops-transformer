#!/usr/bin/env python3
"""
Visual simulator for arch35 FlashAttentionScore split-core scheduling.

It mirrors the kernel-side index mapping in:
  attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h

Covered paths:
  1. non-tnd splitCoreMode == 1:
     - partial S1 region: forward half-N + mirrored reverse half-N
     - full S1 region: forward strided assignment
  2. tnd splitCoreMode == 1:
     - flattened per-batch varlen S1 blocks are assigned in forward/reverse cycles
  3. contiguous fallback:
     - splitCoreMode == 0 style ranges, as if sparseStartIdx has already been built

The script is intentionally dependency-free. It prints readable ASCII grids and can
also emit a small SVG for notes or design docs.
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REGION_STYLE = {
    "partial_fwd": ("PF", "#8ecae6"),
    "partial_rev": ("PR", "#ffb703"),
    "full_fwd": ("F", "#90be6d"),
    "tnd_fwd": ("TF", "#80b1d3"),
    "tnd_rev": ("TR", "#fdb462"),
    "contiguous": ("R", "#bdbdbd"),
}

CORE_COLORS = [
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#f4a261",
    "#7b2cbf",
    "#6a994e",
    "#d62828",
    "#118ab2",
    "#bc6c25",
    "#5e548e",
    "#2f3e46",
    "#9d4edd",
]


@dataclass(frozen=True)
class Task:
    core: int
    region: str
    flat_idx: int
    bn2g: int
    s1o: int
    batch: Optional[int] = None
    n2: Optional[int] = None
    g: Optional[int] = None
    s2_start: Optional[int] = None
    s2_end: Optional[int] = None

    @property
    def tag(self) -> str:
        short, _ = REGION_STYLE[self.region]
        return f"C{self.core}{short}"


def ceil_div(x: int, y: int) -> int:
    if y <= 0:
        raise ValueError("divisor must be positive")
    return (x + y - 1) // y


def calc_real_times(relative_pos: int, length: int, core_num: int) -> int:
    return length // core_num + int(relative_pos < length % core_num)


def calc_real_core_idx(
    relative_pos: int,
    times: int,
    offset: int,
    is_partial: bool,
    core_num: int,
    s1_outer: int,
    first_full_idx: int,
) -> int:
    partial_len = first_full_idx + 1
    core_idx = times * core_num + relative_pos
    if is_partial and first_full_idx != -1:
        region_len = partial_len
    else:
        region_len = s1_outer - partial_len
    if region_len <= 0:
        raise ValueError("empty region was selected by the caller")
    bn2g = core_idx // region_len
    s1o = core_idx % region_len
    return bn2g * s1_outer + s1o + offset


def s2_range(
    sparse: str,
    s1o: int,
    s1_base: int,
    actual_s1: int,
    actual_s2: int,
    pre_tokens: int,
    next_tokens: int,
) -> Tuple[int, int]:
    if sparse in {"dense", "all"}:
        return 0, actual_s2
    if sparse in {"causal", "left_up"}:
        return 0, min((s1o + 1) * s1_base, actual_s2)
    if sparse == "right_down":
        return 0, min((s1o + 1) * s1_base + actual_s2 - actual_s1, actual_s2)
    if sparse == "band":
        start = max(s1o * s1_base - pre_tokens, 0)
        end = min((s1o + 1) * s1_base + next_tokens, actual_s2)
        if end <= start:
            return 0, actual_s2
        return start, end
    raise ValueError(f"unsupported sparse mode: {sparse}")


def simulate_non_tnd_split_core(
    core_num: int,
    total_bn2g: int,
    s1_outer: int,
    first_full_idx: int,
    s1_base: int = 128,
    s1_size: Optional[int] = None,
    s2_size: Optional[int] = None,
    sparse: str = "dense",
    pre_tokens: int = 0,
    next_tokens: int = 0,
) -> Dict[int, List[Task]]:
    if first_full_idx < -1 or first_full_idx >= s1_outer:
        raise ValueError("first_full_idx must be -1 or within [0, s1_outer)")
    if total_bn2g <= 0 or core_num <= 0 or s1_outer <= 0:
        raise ValueError("core_num, total_bn2g and s1_outer must be positive")

    actual_s1 = s1_size if s1_size is not None else s1_outer * s1_base
    actual_s2 = s2_size if s2_size is not None else actual_s1
    partial_len = first_full_idx + 1
    half_n = ceil_div(total_bn2g, 2)

    partial_fwd_len = half_n * partial_len
    partial_rev_len = (total_bn2g - half_n) * partial_len
    if first_full_idx == -1:
        full_fwd_len = total_bn2g * s1_outer
    else:
        full_fwd_len = total_bn2g * (s1_outer - partial_len)

    half_n_core_idx = (partial_fwd_len - 1) % core_num if partial_fwd_len else 0
    per_core: Dict[int, List[Task]] = defaultdict(list)

    def append_task(core: int, region: str, flat_idx: int) -> None:
        bn2g = flat_idx // s1_outer
        s1o = flat_idx % s1_outer
        s2_start, s2_end = s2_range(sparse, s1o, s1_base, actual_s1, actual_s2, pre_tokens, next_tokens)
        per_core[core].append(
            Task(
                core=core,
                region=region,
                flat_idx=flat_idx,
                bn2g=bn2g,
                s1o=s1o,
                s2_start=s2_start,
                s2_end=s2_end,
            )
        )

    for core in range(core_num):
        reverse_pos = (half_n_core_idx - core + core_num) % core_num

        for t in range(calc_real_times(core, partial_fwd_len, core_num)):
            flat_idx = calc_real_core_idx(core, t, 0, True, core_num, s1_outer, first_full_idx)
            append_task(core, "partial_fwd", flat_idx)

        for t in range(calc_real_times(reverse_pos, partial_rev_len, core_num)):
            flat_idx = calc_real_core_idx(
                reverse_pos,
                t,
                half_n * s1_outer,
                True,
                core_num,
                s1_outer,
                first_full_idx,
            )
            append_task(core, "partial_rev", flat_idx)

        for t in range(calc_real_times(core, full_fwd_len, core_num)):
            flat_idx = calc_real_core_idx(
                core,
                t,
                partial_len,
                False,
                core_num,
                s1_outer,
                first_full_idx,
            )
            append_task(core, "full_fwd", flat_idx)

    verify_cover(per_core, total_bn2g * s1_outer)
    return dict(per_core)


def tnd_batch_spans(seq_q: Sequence[int], n2: int, g: int, s1_base: int) -> List[Tuple[int, int, int]]:
    spans = []
    start = 0
    n2g = n2 * g
    for q_len in seq_q:
        s1_outer = ceil_div(q_len, s1_base)
        end = start + s1_outer * n2g
        spans.append((start, end, s1_outer))
        start = end
    return spans


def decode_tnd_flat(
    flat_idx: int,
    seq_q: Sequence[int],
    n2: int,
    g: int,
    s1_base: int,
) -> Tuple[int, int, int, int, int]:
    for batch, (start, end, s1_outer) in enumerate(tnd_batch_spans(seq_q, n2, g, s1_base)):
        if flat_idx < end:
            inner = flat_idx - start
            n2g_idx = inner // s1_outer
            s1o = inner % s1_outer
            return batch, n2g_idx // g, n2g_idx % g, n2g_idx, s1o
    raise ValueError(f"flat_idx {flat_idx} is outside TND total size")


def simulate_tnd_split_core(
    core_num: int,
    seq_q: Sequence[int],
    seq_kv: Sequence[int],
    n2: int,
    g: int,
    s1_base: int = 128,
    sparse: str = "dense",
    pre_tokens: int = 0,
    next_tokens: int = 0,
) -> Dict[int, List[Task]]:
    if len(seq_q) != len(seq_kv):
        raise ValueError("seq_q and seq_kv must have the same batch length")
    if core_num <= 0 or n2 <= 0 or g <= 0:
        raise ValueError("core_num, n2 and g must be positive")

    total_size = sum(ceil_div(q_len, s1_base) * n2 * g for q_len in seq_q)
    cycle_core_nums = core_num * 2
    full_cycles = total_size // cycle_core_nums
    remain = total_size % cycle_core_nums
    per_core: Dict[int, List[Task]] = defaultdict(list)

    for core in range(core_num):
        calc_times = full_cycles * 2
        if remain >= core + 1:
            calc_times += 1
            if remain > core_num and (core + 1) > cycle_core_nums - remain:
                calc_times += 1

        for local_idx in range(calc_times):
            calc_loop = local_idx >> 1
            reverse_half = local_idx & 1
            if reverse_half == 0:
                flat_idx = calc_loop * cycle_core_nums + core
                region = "tnd_fwd"
            else:
                flat_idx = calc_loop * cycle_core_nums + (cycle_core_nums - core - 1)
                region = "tnd_rev"
            if flat_idx >= total_size:
                continue

            batch, n2_idx, g_idx, n2g_idx, s1o = decode_tnd_flat(flat_idx, seq_q, n2, g, s1_base)
            s2_start, s2_end = s2_range(
                sparse,
                s1o,
                s1_base,
                seq_q[batch],
                seq_kv[batch],
                pre_tokens,
                next_tokens,
            )
            per_core[core].append(
                Task(
                    core=core,
                    region=region,
                    flat_idx=flat_idx,
                    bn2g=n2g_idx,
                    s1o=s1o,
                    batch=batch,
                    n2=n2_idx,
                    g=g_idx,
                    s2_start=s2_start,
                    s2_end=s2_end,
                )
            )

    verify_cover(per_core, total_size)
    return dict(per_core)


def simulate_contiguous_ranges(total_size: int, core_num: int) -> Dict[int, List[Task]]:
    if total_size <= 0 or core_num <= 0:
        raise ValueError("total_size and core_num must be positive")
    per_core: Dict[int, List[Task]] = defaultdict(list)
    for core in range(core_num):
        start = min(core * ceil_div(total_size, core_num), total_size)
        end = min((core + 1) * ceil_div(total_size, core_num), total_size)
        for flat_idx in range(start, end):
            per_core[core].append(
                Task(core=core, region="contiguous", flat_idx=flat_idx, bn2g=0, s1o=flat_idx)
            )
    verify_cover(per_core, total_size)
    return dict(per_core)


def verify_cover(per_core: Dict[int, List[Task]], total_size: int) -> None:
    produced = [task.flat_idx for tasks in per_core.values() for task in tasks]
    if len(produced) != len(set(produced)):
        raise AssertionError("duplicate flat indices produced")
    expected = set(range(total_size))
    if set(produced) != expected:
        missing = sorted(expected - set(produced))[:10]
        extra = sorted(set(produced) - expected)[:10]
        raise AssertionError(f"coverage mismatch, missing={missing}, extra={extra}")


def grid_for_non_tnd(per_core: Dict[int, List[Task]], total_bn2g: int, s1_outer: int) -> List[List[Optional[Task]]]:
    grid: List[List[Optional[Task]]] = [[None for _ in range(s1_outer)] for _ in range(total_bn2g)]
    for task in iter_tasks(per_core):
        grid[task.bn2g][task.s1o] = task
    return grid


def grids_for_tnd(
    per_core: Dict[int, List[Task]],
    seq_q: Sequence[int],
    n2: int,
    g: int,
    s1_base: int,
) -> List[List[List[Optional[Task]]]]:
    grids: List[List[List[Optional[Task]]]] = []
    for batch, q_len in enumerate(seq_q):
        s1_outer = ceil_div(q_len, s1_base)
        grid = [[None for _ in range(s1_outer)] for _ in range(n2 * g)]
        grids.append(grid)
    for task in iter_tasks(per_core):
        if task.batch is None:
            continue
        grids[task.batch][task.bn2g][task.s1o] = task
    return grids


def iter_tasks(per_core: Dict[int, List[Task]]) -> Iterable[Task]:
    for core in sorted(per_core):
        for task in per_core[core]:
            yield task


def print_summary(per_core: Dict[int, List[Task]]) -> None:
    print("Per-core task counts:")
    for core in sorted(per_core):
        counts: Dict[str, int] = defaultdict(int)
        for task in per_core[core]:
            counts[task.region] += 1
        parts = ", ".join(f"{region}={counts[region]}" for region in sorted(counts))
        print(f"  core {core:02d}: total={len(per_core[core]):3d}  {parts}")
    print()


def print_non_tnd_grid(per_core: Dict[int, List[Task]], total_bn2g: int, s1_outer: int) -> None:
    grid = grid_for_non_tnd(per_core, total_bn2g, s1_outer)
    width = max(5, max(len(task.tag) for task in iter_tasks(per_core)))
    header = "BN2G\\S1".ljust(9) + " ".join(f"{i:>{width}}" for i in range(s1_outer))
    print(header)
    for bn2g, row in enumerate(grid):
        cells = [cell.tag if cell is not None else "." for cell in row]
        print(f"{bn2g:>7}  " + " ".join(f"{cell:>{width}}" for cell in cells))
    print()


def print_tnd_grids(
    per_core: Dict[int, List[Task]],
    seq_q: Sequence[int],
    seq_kv: Sequence[int],
    n2: int,
    g: int,
    s1_base: int,
) -> None:
    grids = grids_for_tnd(per_core, seq_q, n2, g, s1_base)
    width = max(5, max(len(task.tag) for task in iter_tasks(per_core)))
    for batch, grid in enumerate(grids):
        s1_outer = len(grid[0]) if grid else 0
        print(f"Batch {batch}: q={seq_q[batch]}, kv={seq_kv[batch]}, s1_outer={s1_outer}")
        header = "N2G\\S1".ljust(9) + " ".join(f"{i:>{width}}" for i in range(s1_outer))
        print(header)
        for n2g_idx, row in enumerate(grid):
            cells = [cell.tag if cell is not None else "." for cell in row]
            print(f"{n2g_idx:>7}  " + " ".join(f"{cell:>{width}}" for cell in cells))
        print()


def print_timeline(per_core: Dict[int, List[Task]], limit: int = 32) -> None:
    print("Per-core timeline:")
    for core in sorted(per_core):
        tokens = []
        for task in per_core[core][:limit]:
            if task.batch is None:
                tokens.append(f"{task.tag}@{task.bn2g}:{task.s1o}")
            else:
                tokens.append(f"{task.tag}@b{task.batch}/n{task.bn2g}/s{task.s1o}")
        suffix = " ..." if len(per_core[core]) > limit else ""
        print(f"  core {core:02d}: " + " -> ".join(tokens) + suffix)
    print()


def print_task_ranges(per_core: Dict[int, List[Task]], limit: int = 8) -> None:
    print("First task S2 ranges:")
    for core in sorted(per_core):
        shown = per_core[core][:limit]
        parts = []
        for task in shown:
            if task.s2_start is None:
                continue
            loc = f"b{task.batch}/n{task.bn2g}/s{task.s1o}" if task.batch is not None else f"n{task.bn2g}/s{task.s1o}"
            parts.append(f"{loc}=[{task.s2_start},{task.s2_end})")
        if parts:
            suffix = " ..." if len(per_core[core]) > limit else ""
            print(f"  core {core:02d}: " + "; ".join(parts) + suffix)
    print()


def core_color(core: int) -> str:
    return CORE_COLORS[core % len(CORE_COLORS)]


def blend_with_white(hex_color: str, ratio: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * ratio + 255 * (1 - ratio))
    g = int(g * ratio + 255 * (1 - ratio))
    b = int(b * ratio + 255 * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"


def svg_rect(x: int, y: int, w: int, h: int, fill: str, stroke: str = "#ffffff") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def svg_text(x: int, y: int, text: str, size: int = 11, anchor: str = "middle", fill: str = "#1f2933") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Menlo,Consolas,monospace" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{fill}">{html.escape(text)}</text>'
    )


def write_non_tnd_svg(path: Path, per_core: Dict[int, List[Task]], total_bn2g: int, s1_outer: int) -> None:
    grid = grid_for_non_tnd(per_core, total_bn2g, s1_outer)
    cell_w = 48
    cell_h = 34
    left = 78
    top = 72
    width = max(left + s1_outer * cell_w + 30, 460)
    legend_y = top + total_bn2g * cell_h + 34
    height = legend_y + legend_height(per_core, width, 24) + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        svg_text(width // 2, 28, "FA arch35 non-TND splitCoreMode=1", 16),
        svg_text(width // 2, 48, "rows=BN2G, cols=S1 outer block, cell=core+region", 11),
    ]
    for s1 in range(s1_outer):
        parts.append(svg_text(left + s1 * cell_w + cell_w // 2, top - 12, str(s1), 11))
    for bn2g, row in enumerate(grid):
        parts.append(svg_text(left - 12, top + bn2g * cell_h + 21, str(bn2g), 11, anchor="end"))
        for s1, task in enumerate(row):
            x = left + s1 * cell_w
            y = top + bn2g * cell_h
            if task is None:
                parts.append(svg_rect(x, y, cell_w, cell_h, "#eeeeee"))
            else:
                _, region_color = REGION_STYLE[task.region]
                fill = blend_with_white(core_color(task.core), 0.70)
                parts.append(svg_rect(x, y, cell_w, cell_h, fill))
                parts.append(svg_text(x + cell_w // 2, y + 14, f"C{task.core}", 11))
                parts.append(svg_text(x + cell_w // 2, y + 28, REGION_STYLE[task.region][0], 10, fill=region_color))
    append_legend(parts, 24, legend_y, per_core, width)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_tnd_svg(
    path: Path,
    per_core: Dict[int, List[Task]],
    seq_q: Sequence[int],
    seq_kv: Sequence[int],
    n2: int,
    g: int,
    s1_base: int,
) -> None:
    grids = grids_for_tnd(per_core, seq_q, n2, g, s1_base)
    cell_w = 48
    cell_h = 34
    left = 84
    top = 112
    batch_gap = 62
    max_s1_outer = max((len(grid[0]) if grid else 0) for grid in grids)
    total_rows = sum(len(grid) for grid in grids)
    width = max(left + max_s1_outer * cell_w + 50, 460)
    content_end_y = top + total_rows * cell_h + max(len(grids) - 1, 0) * batch_gap
    legend_y = content_end_y + 48
    height = legend_y + legend_height(per_core, width, 24) + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        svg_text(width // 2, 28, "FA arch35 TND splitCoreMode=1", 16),
        svg_text(width // 2, 48, "each batch has its own S1 outer width; rows=N2G", 11),
    ]
    y = top
    for batch, grid in enumerate(grids):
        s1_outer = len(grid[0]) if grid else 0
        parts.append(svg_text(left, y - 34, f"batch {batch}: q={seq_q[batch]}, kv={seq_kv[batch]}", 12, anchor="start"))
        for s1 in range(s1_outer):
            parts.append(svg_text(left + s1 * cell_w + cell_w // 2, y - 12, str(s1), 11))
        for n2g_idx, row in enumerate(grid):
            parts.append(svg_text(left - 12, y + n2g_idx * cell_h + 21, str(n2g_idx), 11, anchor="end"))
            for s1, task in enumerate(row):
                x = left + s1 * cell_w
                yy = y + n2g_idx * cell_h
                if task is None:
                    parts.append(svg_rect(x, yy, cell_w, cell_h, "#eeeeee"))
                else:
                    _, region_color = REGION_STYLE[task.region]
                    fill = blend_with_white(core_color(task.core), 0.70)
                    parts.append(svg_rect(x, yy, cell_w, cell_h, fill))
                    parts.append(svg_text(x + cell_w // 2, yy + 14, f"C{task.core}", 11))
                    parts.append(svg_text(x + cell_w // 2, yy + 28, REGION_STYLE[task.region][0], 10, fill=region_color))
        y += len(grid) * cell_h
        if batch != len(grids) - 1:
            y += batch_gap
    append_legend(parts, 24, legend_y, per_core, width)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def legend_height(per_core: Dict[int, List[Task]], width: int, x: int) -> int:
    region_cols = max(1, min(3, (width - x - 20) // 150))
    region_rows = ceil_div(len(REGION_STYLE), region_cols)
    core_cols = max(1, (width - x - 20) // 58)
    core_rows = ceil_div(len(per_core), core_cols)
    return 20 + region_rows * 22 + 30 + core_rows * 22


def append_legend(parts: List[str], x: int, y: int, per_core: Dict[int, List[Task]], width: int) -> None:
    parts.append(svg_text(x, y, "Legend", 12, anchor="start"))
    yy = y + 20
    region_cols = max(1, min(3, (width - x - 20) // 150))
    for idx, region in enumerate(REGION_STYLE):
        short, color = REGION_STYLE[region]
        xx = x + (idx % region_cols) * 150
        yyy = yy + (idx // region_cols) * 22
        parts.append(svg_rect(xx, yyy - 12, 16, 16, color, stroke="#cccccc"))
        parts.append(svg_text(xx + 22, yyy, f"{short}: {region}", 10, anchor="start"))
    region_rows = ceil_div(len(REGION_STYLE), region_cols)
    core_y = yy + region_rows * 22 + 28
    core_cols = max(1, (width - x - 20) // 58)
    for idx, core in enumerate(sorted(per_core)):
        xx = x + (idx % core_cols) * 58
        yyy = core_y + (idx // core_cols) * 22
        parts.append(svg_rect(xx, yyy - 12, 16, 16, core_color(core), stroke="#cccccc"))
        parts.append(svg_text(xx + 20, yyy, f"C{core}", 10, anchor="start"))


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def explain() -> None:
    print(
        """
arch35 FA split-core model
==========================

1. splitCoreMode == 0
   Host builds sparseStartIdx[] and each AIC scans one contiguous flattened range.
   The exact range depends on host sparse load estimation, not on kernel formulas.

2. non-TND splitCoreMode == 1
   Flattened index = ((B*N2*G) group, S1 outer block), with S1 as the fastest axis.

   partial region:
     firstFullLoadS1OuterIdx >= 0 means S1 blocks [0, firstFull] have partial S2
     coverage. The kernel splits total BN2G into two halves. The first half uses
     forward striding by core id; the second half uses a mirrored core id so causal
     load imbalance is paired across the upper/lower half.

   full region:
     S1 blocks after the partial region are full-S2 blocks and use forward striding.
     If firstFullLoadS1OuterIdx == -1, there is no partial region; all blocks are
     full and use forward striding.

3. TND splitCoreMode == 1
   Flattened index is varlen-aware:
     for each batch b:
       for n2g in N2*G:
         for s1o in ceil(actual_q_len[b] / s1_base):
           emit one flattened S1 task

   Each cycle has 2*core_num flattened tasks. The first core_num tasks are assigned
   forward: core 0,1,2,... The second core_num tasks are assigned backward:
   core N-1,...,2,1,0. This is the kernel's positive/reverse cyclic split.

Use examples:
  python fa_tiling.py non-tnd --core-num 4 --total-bn2g 5 --s1-outer 6 --first-full-idx 2 --svg non_tnd.svg
  python fa_tiling.py tnd --core-num 4 --seq-q 384,640,128 --seq-kv 384,768,256 --n2 2 --g 1 --svg tnd.svg
"""
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize arch35 FA split-core tiling.")
    subparsers = parser.add_subparsers(dest="cmd")

    p = subparsers.add_parser("non-tnd", help="simulate non-TND splitCoreMode=1")
    p.add_argument("--core-num", type=int, default=4)
    p.add_argument("--total-bn2g", type=int, default=5, help="B*N2*G flattened groups")
    p.add_argument("--s1-outer", type=int, default=6)
    p.add_argument("--first-full-idx", type=int, default=2, help="-1 means all S1 blocks are full-S2")
    p.add_argument("--s1-base", type=int, default=128)
    p.add_argument("--s1-size", type=int)
    p.add_argument("--s2-size", type=int)
    p.add_argument("--sparse", choices=["dense", "all", "causal", "left_up", "right_down", "band"], default="dense")
    p.add_argument("--pre-tokens", type=int, default=0)
    p.add_argument("--next-tokens", type=int, default=0)
    p.add_argument("--svg", type=Path)
    p.add_argument("--timeline-limit", type=int, default=32)

    p = subparsers.add_parser("tnd", help="simulate TND splitCoreMode=1")
    p.add_argument("--core-num", type=int, default=4)
    p.add_argument("--seq-q", type=parse_int_list, default=parse_int_list("384,640,128"))
    p.add_argument("--seq-kv", type=parse_int_list, default=parse_int_list("384,768,256"))
    p.add_argument("--n2", type=int, default=2)
    p.add_argument("--g", type=int, default=1)
    p.add_argument("--s1-base", type=int, default=128)
    p.add_argument("--sparse", choices=["dense", "all", "causal", "left_up", "right_down", "band"], default="dense")
    p.add_argument("--pre-tokens", type=int, default=0)
    p.add_argument("--next-tokens", type=int, default=0)
    p.add_argument("--svg", type=Path)
    p.add_argument("--timeline-limit", type=int, default=32)

    p = subparsers.add_parser("contiguous", help="show splitCoreMode=0 contiguous fallback shape")
    p.add_argument("--core-num", type=int, default=4)
    p.add_argument("--total-size", type=int, default=24)
    p.add_argument("--timeline-limit", type=int, default=64)

    subparsers.add_parser("explain", help="print the split-core principle")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd is None:
        args = parser.parse_args(["non-tnd"])

    if args.cmd == "explain":
        explain()
        return

    if args.cmd == "non-tnd":
        per_core = simulate_non_tnd_split_core(
            core_num=args.core_num,
            total_bn2g=args.total_bn2g,
            s1_outer=args.s1_outer,
            first_full_idx=args.first_full_idx,
            s1_base=args.s1_base,
            s1_size=args.s1_size,
            s2_size=args.s2_size,
            sparse=args.sparse,
            pre_tokens=args.pre_tokens,
            next_tokens=args.next_tokens,
        )
        print("Mode: non-TND splitCoreMode=1")
        print(f"core_num={args.core_num}, total_bn2g={args.total_bn2g}, s1_outer={args.s1_outer}, "
              f"first_full_idx={args.first_full_idx}")
        print_summary(per_core)
        print_non_tnd_grid(per_core, args.total_bn2g, args.s1_outer)
        print_timeline(per_core, args.timeline_limit)
        print_task_ranges(per_core)
        if args.svg:
            write_non_tnd_svg(args.svg, per_core, args.total_bn2g, args.s1_outer)
            print(f"SVG written to {args.svg}")
        return

    if args.cmd == "tnd":
        per_core = simulate_tnd_split_core(
            core_num=args.core_num,
            seq_q=args.seq_q,
            seq_kv=args.seq_kv,
            n2=args.n2,
            g=args.g,
            s1_base=args.s1_base,
            sparse=args.sparse,
            pre_tokens=args.pre_tokens,
            next_tokens=args.next_tokens,
        )
        total_size = sum(ceil_div(q_len, args.s1_base) * args.n2 * args.g for q_len in args.seq_q)
        print("Mode: TND splitCoreMode=1")
        print(f"core_num={args.core_num}, seq_q={args.seq_q}, seq_kv={args.seq_kv}, "
              f"n2={args.n2}, g={args.g}, total_flat_s1_tasks={total_size}")
        print_summary(per_core)
        print_tnd_grids(per_core, args.seq_q, args.seq_kv, args.n2, args.g, args.s1_base)
        print_timeline(per_core, args.timeline_limit)
        print_task_ranges(per_core)
        if args.svg:
            write_tnd_svg(args.svg, per_core, args.seq_q, args.seq_kv, args.n2, args.g, args.s1_base)
            print(f"SVG written to {args.svg}")
        return

    if args.cmd == "contiguous":
        per_core = simulate_contiguous_ranges(args.total_size, args.core_num)
        print("Mode: splitCoreMode=0 contiguous fallback")
        print("This is only an even-range placeholder. Real sparseStartIdx is built by host load balancing.")
        print_summary(per_core)
        print_timeline(per_core, args.timeline_limit)
        return

    raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

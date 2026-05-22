#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_sim_log.py  —  Flash Attention 仿真日志 Cycle 耗时分析与可视化
=======================================================================
用法:
    python parse_sim_log.py --aic_log <aic_instr.log> --aiv_log <aiv_instr.log>
    python parse_sim_log.py --aic_log "core*_aic.log" --aiv_log "core*_aiv.log"
    python parse_sim_log.py --aic_log aic.log --aiv_log aiv.log --output timeline.png

日志格式 (instr_log):
    [info] [CYCLE] (PC: 0x...) TYPE : ... MOV_SPR_XN  SPR:COND, XN:Xn=0xMARKVALUE, ...
"""

import re
import glob
import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("[ERROR] 需要安装依赖: pip install numpy matplotlib")
    sys.exit(1)

# ============================================================
# Mark 码表  — 完全对应 kernel 中插入的 set_mark 调用
# 格式: mark_value → (core_type, module, label, color)
# low byte=0x00=start, 0xFF=end
# ============================================================
AIC = "AIC"
AIV = "AIV"
BOTH = "BOTH"

# 颜色方案
_COLORS = {
    "Init":             "#4C8BF5",  # 蓝
    "InitBuffers":      "#34A853",  # 绿
    "BMM1":             "#FF6D00",  # 橙
    "BMM1_CopyQ":       "#FFB300",  # 黄橙
    "BMM1_CopyK":       "#FF8F00",  # 深黄
    "BMM1_Matmul":      "#E65100",  # 深橙
    "BMM1_Fixpipe":     "#BF360C",  # 红橙
    "Vec1":             "#00ACC1",  # 青
    "Vec1_AttenMask":   "#00838F",  # 深青
    "Vec1_SoftmaxVF":   "#006064",  # 深青2
    "Vec1_DataCopy":    "#80DEEA",  # 浅青
    "BMM2":             "#AB47BC",  # 紫
    "BMM2_CopyV":       "#CE93D8",  # 浅紫
    "BMM2_Matmul":      "#8E24AA",  # 深紫
    "BMM2_Fixpipe":     "#6A1B9A",  # 更深紫
    "Vec2":             "#EF5350",  # 红
    "Vec2_FlashUpdate": "#B71C1C",  # 深红
    "FlashDecode":      "#66BB6A",  # 草绿
    "FD_CopyLseIn":     "#A5D6A7",  # 浅草绿
    "FD_ReduceFinalRes":"#388E3C",  # 深草绿
    "FD_CopyFinalResOut":"#1B5E20", # 最深草绿
    "Uninit":           "#90A4AE",  # 灰
}

# mark_value → (core_type, group_key, display_label, color_key)
MARK_TABLE = {
    # ---- Init (AIC & AIV 都执行) ----
    0x10000000: (BOTH, "Init",             "Init",               "Init"),
    0x100000FF: (BOTH, "Init",             "Init",               "Init"),
    # ---- AIC InitBuffers ----
    0x20000000: (AIC,  "AIC_InitBuffers",  "AIC InitBuffers",    "InitBuffers"),
    0x200000FF: (AIC,  "AIC_InitBuffers",  "AIC InitBuffers",    "InitBuffers"),
    # ---- AIV InitBuffers ----
    0x30000000: (AIV,  "AIV_InitBuffers",  "AIV InitBuffers",    "InitBuffers"),
    0x300000FF: (AIV,  "AIV_InitBuffers",  "AIV InitBuffers",    "InitBuffers"),
    # ---- BMM1 (AIC) ----
    0x40000000: (AIC,  "BMM1",             "BMM1",               "BMM1"),
    0x400000FF: (AIC,  "BMM1",             "BMM1",               "BMM1"),
    0x40010000: (AIC,  "BMM1_CopyQ",       "BMM1 CopyQ",         "BMM1_CopyQ"),
    0x400100FF: (AIC,  "BMM1_CopyQ",       "BMM1 CopyQ",         "BMM1_CopyQ"),
    0x40020000: (AIC,  "BMM1_CopyK",       "BMM1 CopyK",         "BMM1_CopyK"),
    0x400200FF: (AIC,  "BMM1_CopyK",       "BMM1 CopyK",         "BMM1_CopyK"),
    0x40030000: (AIC,  "BMM1_Matmul",      "BMM1 Matmul",        "BMM1_Matmul"),
    0x400300FF: (AIC,  "BMM1_Matmul",      "BMM1 Matmul",        "BMM1_Matmul"),
    0x40040000: (AIC,  "BMM1_Fixpipe",     "BMM1 Fixpipe",       "BMM1_Fixpipe"),
    0x400400FF: (AIC,  "BMM1_Fixpipe",     "BMM1 Fixpipe",       "BMM1_Fixpipe"),
    # ---- Vec1 (AIV) ----
    0x50000000: (AIV,  "Vec1",             "Vec1",               "Vec1"),
    0x500000FF: (AIV,  "Vec1",             "Vec1",               "Vec1"),
    0x50010000: (AIV,  "Vec1_AttenMask",   "Vec1 AttenMask",     "Vec1_AttenMask"),
    0x500100FF: (AIV,  "Vec1_AttenMask",   "Vec1 AttenMask",     "Vec1_AttenMask"),
    0x50020000: (AIV,  "Vec1_SoftmaxVF",   "Vec1 SoftmaxVF",     "Vec1_SoftmaxVF"),
    0x500200FF: (AIV,  "Vec1_SoftmaxVF",   "Vec1 SoftmaxVF",     "Vec1_SoftmaxVF"),
    0x50030000: (AIV,  "Vec1_DataCopy",    "Vec1 DataCopy→L1",   "Vec1_DataCopy"),
    0x500300FF: (AIV,  "Vec1_DataCopy",    "Vec1 DataCopy→L1",   "Vec1_DataCopy"),
    # ---- BMM2 (AIC) ----
    0x60000000: (AIC,  "BMM2",             "BMM2",               "BMM2"),
    0x600000FF: (AIC,  "BMM2",             "BMM2",               "BMM2"),
    0x60010000: (AIC,  "BMM2_CopyV",       "BMM2 CopyV",         "BMM2_CopyV"),
    0x600100FF: (AIC,  "BMM2_CopyV",       "BMM2 CopyV",         "BMM2_CopyV"),
    0x60020000: (AIC,  "BMM2_Matmul",      "BMM2 Matmul",        "BMM2_Matmul"),
    0x600200FF: (AIC,  "BMM2_Matmul",      "BMM2 Matmul",        "BMM2_Matmul"),
    0x60030000: (AIC,  "BMM2_Fixpipe",     "BMM2 Fixpipe",       "BMM2_Fixpipe"),
    0x600300FF: (AIC,  "BMM2_Fixpipe",     "BMM2 Fixpipe",       "BMM2_Fixpipe"),
    # ---- Vec2 (AIV) ----
    0x70000000: (AIV,  "Vec2",             "Vec2",               "Vec2"),
    0x700000FF: (AIV,  "Vec2",             "Vec2",               "Vec2"),
    0x70010000: (AIV,  "Vec2_FlashUpdate", "Vec2 FlashUpdate",   "Vec2_FlashUpdate"),
    0x700100FF: (AIV,  "Vec2_FlashUpdate", "Vec2 FlashUpdate",   "Vec2_FlashUpdate"),
    # ---- FlashDecode (AIV) ----
    0x80000000: (AIV,  "FlashDecode",      "FlashDecode",        "FlashDecode"),
    0x800000FF: (AIV,  "FlashDecode",      "FlashDecode",        "FlashDecode"),
    0x80010000: (AIV,  "FD_CopyLseIn",     "FD CopyLseIn",       "FD_CopyLseIn"),
    0x800100FF: (AIV,  "FD_CopyLseIn",     "FD CopyLseIn",       "FD_CopyLseIn"),
    0x80020000: (AIV,  "FD_ReduceFinalRes","FD ReduceFinalRes",  "FD_ReduceFinalRes"),
    0x800200FF: (AIV,  "FD_ReduceFinalRes","FD ReduceFinalRes",  "FD_ReduceFinalRes"),
    0x80030000: (AIV,  "FD_CopyFinalResOut","FD CopyFinalResOut","FD_CopyFinalResOut"),
    0x800300FF: (AIV,  "FD_CopyFinalResOut","FD CopyFinalResOut","FD_CopyFinalResOut"),
    # ---- Uninit (AIC & AIV) ----
    0x90000000: (BOTH, "Uninit",           "Uninit",             "Uninit"),
    0x900000FF: (BOTH, "Uninit",           "Uninit",             "Uninit"),
}

def is_start(mark_val):
    return (mark_val & 0xFF) == 0x00

def is_end(mark_val):
    return (mark_val & 0xFF) == 0xFF

def group_key(mark_val):
    """去掉 low byte，得到同一 module 的 start/end 共用 key"""
    return mark_val & 0xFFFFFF00

# ============================================================
# 解析函数
# ============================================================
_RE_COND = re.compile(
    r'\[info\]\s+\[(\d+)\].*?MOV_SPR_XN\s+SPR:COND,\s+XN:\w+=0x([0-9a-fA-F]+)'
)

def parse_log_file(filepath):
    """解析单个 instr_log 文件，返回 [(cycle_int, mark_val_int), ...]"""
    events = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _RE_COND.search(line)
            if m:
                cycle = int(m.group(1))
                mark  = int(m.group(2), 16)
                events.append((cycle, mark))
    return events

def parse_log_files(patterns):
    """支持文件列表或 glob 模式，返回 {filename: [(cycle, mark), ...]}"""
    result = {}
    file_list = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if not matched:
            if Path(pat).exists():
                matched = [pat]
            else:
                print(f"[WARN] 未找到文件: {pat}")
        file_list.extend(matched)
    if not file_list:
        return result
    for fp in file_list:
        events = parse_log_file(fp)
        if events:
            result[fp] = events
            print(f"  解析 {fp}: {len(events)} 条 COND 写入")
        else:
            print(f"  [WARN] {fp}: 未找到 COND 写入记录")
    return result

# ============================================================
# 统计区间
# ============================================================
def compute_intervals(events):
    """
    从 (cycle, mark) 序列中提取所有 start→end 区间。
    返回: {group_key: [(start_cycle, end_cycle, duration), ...]}
    """
    pending = {}   # group_key → last_start_cycle
    intervals = defaultdict(list)

    for cycle, mark in events:
        gk = group_key(mark)
        if gk not in MARK_TABLE:
            continue
        if is_start(mark):
            pending[gk] = cycle
        elif is_end(mark) and gk in pending:
            start = pending.pop(gk)
            dur = cycle - start
            if dur >= 0:
                intervals[gk].append((start, cycle, dur))

    return intervals

def aggregate_intervals(all_intervals_by_file):
    """
    多核聚合: 对同一 group_key 的所有区间，
    返回 {group_key: {'total': X, 'avg': X, 'min': X, 'max': X,
                       'count': N, 'all_durs': [...]}}
    """
    combined = defaultdict(list)
    for intervals in all_intervals_by_file.values():
        for gk, ilist in intervals.items():
            combined[gk].extend([d for (_, _, d) in ilist])

    result = {}
    for gk, durs in combined.items():
        result[gk] = {
            'total': sum(durs),
            'avg':   sum(durs) / len(durs),
            'min':   min(durs),
            'max':   max(durs),
            'count': len(durs),
            'all_durs': durs,
        }
    return result

def pick_timeline_core(all_intervals_by_file):
    """选一个拥有最多完整区间的核作为 Gantt 展示核"""
    best_file = None
    best_count = 0
    for fp, intervals in all_intervals_by_file.items():
        c = sum(len(v) for v in intervals.values())
        if c > best_count:
            best_count = c
            best_file = fp
    return best_file

# ============================================================
# 绘图辅助数据结构
# ============================================================

# group_key → name  (反查表)
_GK_TO_NAME = {}
for mv, (ct, gname, label, ck) in MARK_TABLE.items():
    gk = group_key(mv)
    _GK_TO_NAME[gk] = gname

# group_key → "0xSTART ~ 0xEND" (打点标记码显示)
_GK_TO_MARK_STR = {}
for _mv in MARK_TABLE:
    _gk = group_key(_mv)
    if _gk not in _GK_TO_MARK_STR:
        _GK_TO_MARK_STR[_gk] = f"0x{_gk:08X} ~ 0x{(_gk | 0xFF):08X}"

# 每行一个 module 的泳道规格: (group_name, indent_level)
# indent=0: 主阶段(粗条); indent=1: 子步骤(细条)
_AIC_ROW_SPECS = [
    ("Init",            0),
    ("AIC_InitBuffers", 0),
    ("BMM1",            0),
    ("BMM1_CopyQ",      1),
    ("BMM1_CopyK",      1),
    ("BMM1_Matmul",     1),
    ("BMM1_Fixpipe",    1),
    ("BMM2",            0),
    ("BMM2_CopyV",      1),
    ("BMM2_Matmul",     1),
    ("BMM2_Fixpipe",    1),
    ("Uninit",          0),
]

_AIV_ROW_SPECS = [
    ("AIV_InitBuffers",    0),
    ("Vec1",               0),
    ("Vec1_AttenMask",     1),
    ("Vec1_SoftmaxVF",     1),
    ("Vec1_DataCopy",      1),
    ("Vec2",               0),
    ("Vec2_FlashUpdate",   1),
    ("FlashDecode",        0),
    ("FD_CopyLseIn",       1),
    ("FD_ReduceFinalRes",  1),
    ("FD_CopyFinalResOut", 1),
]


def _gk_of_name(name):
    """根据 group_name 找 group_key（用于区间查找）"""
    for gk, n in _GK_TO_NAME.items():
        if n == name:
            return gk
    return None


def _get_label(gk):
    """获取 display label"""
    if gk in MARK_TABLE:
        return MARK_TABLE[gk][2]
    return hex(gk)


def _get_color(gk):
    _, _, _, ck = MARK_TABLE.get(gk, (None, None, None, "Init"))
    return _COLORS.get(ck, "#AAAAAA")


def _draw_multirow(ax, row_specs, row_ys, intervals_dict, global_t0, x_max,
                   row_h_main=0.55, row_h_sub=0.38):
    """
    每个 module 独占一行绘制，返回 (ytick_positions, ytick_labels)。
    row_specs: list of (group_name, indent_level)
    row_ys:    list of y-center for each row
    """
    ytick_pos = []
    ytick_lbl = []
    min_inner_w = x_max * 0.035  # 宽于此才在条内写文字

    for (gname, indent), y_ctr in zip(row_specs, row_ys):
        gk = _gk_of_name(gname)
        mark_str = _GK_TO_MARK_STR.get(gk, "") if gk is not None else ""
        disp_label = _get_label(gk) if gk is not None else gname
        prefix = "  " if indent > 0 else ""
        ytick_pos.append(y_ctr)
        ytick_lbl.append(f"{prefix}{disp_label}\n{prefix}[{mark_str}]")

        if gk is None:
            continue

        color  = _get_color(gk)
        rh     = row_h_sub if indent > 0 else row_h_main
        alpha  = 0.68 if indent > 0 else 0.90
        ivals  = intervals_dict.get(gk, [])

        for (s, e, d) in ivals:
            s2, e2 = s - global_t0, e - global_t0
            w = e2 - s2
            if w <= 0:
                continue
            ax.barh(y_ctr, w, left=s2, height=rh,
                    color=color, alpha=alpha, edgecolor='#FFFFFF',
                    linewidth=0.5, zorder=3)
            short = disp_label.split()[-1]
            if w >= min_inner_w:
                # 宽条: 文字写在条内
                ax.text(s2 + w / 2, y_ctr,
                        f"{short}\n{d:,}cy",
                        ha='center', va='center',
                        fontsize=6.2 if indent == 0 else 5.5,
                        color='white', fontweight='bold',
                        clip_on=True, zorder=4)
            elif w >= min_inner_w * 0.25:
                # 窄条: cycle 数标注在条上方
                ax.text(s2 + w / 2, y_ctr + rh / 2 + 0.05,
                        f"{d:,}cy",
                        ha='center', va='bottom', fontsize=5.2,
                        color=color, clip_on=True, zorder=4)

    return ytick_pos, ytick_lbl


# ---- 保留旧名以防万一 ----
def _draw_lane(ax, intervals_dict, order, sub_map, main_y, sub_y,
               row_h, sub_h, t_offset, alpha_main=0.88, alpha_sub=0.72):
    """在 ax 的指定 y 位置绘制一条泳道及其子步骤，返回最大 cycle（相对）。"""
    max_c = 0
    for name in order:
        gk = _gk_of_name(name)
        if gk is None:
            continue
        color = _get_color(gk)
        label = _get_label(gk)
        for (s, e, d) in intervals_dict.get(gk, []):
            s2, e2 = s - t_offset, e - t_offset
            w = e2 - s2
            ax.barh(main_y, w, left=s2, height=row_h,
                    color=color, alpha=alpha_main, edgecolor='#FFFFFF', linewidth=0.5,
                    zorder=3)
            if w > 0:
                ax.text(s2 + w / 2, main_y,
                        f"{label}\n{d:,}cy",
                        ha='center', va='center', fontsize=6.5, color='white',
                        fontweight='bold', clip_on=True, zorder=4)
            max_c = max(max_c, e2)

        for sub_name in sub_map.get(name, []):
            sub_gk = _gk_of_name(sub_name)
            if sub_gk is None:
                continue
            sub_color = _get_color(sub_gk)
            sub_lbl = _get_label(sub_gk)
            for (s, e, d) in intervals_dict.get(sub_gk, []):
                s2, e2 = s - t_offset, e - t_offset
                w = e2 - s2
                ax.barh(sub_y, w, left=s2, height=sub_h,
                        color=sub_color, alpha=alpha_sub, edgecolor='#CCCCCC',
                        linewidth=0.3, zorder=3)
                if w > 0:
                    short = sub_lbl.split()[-1]
                    ax.text(s2 + w / 2, sub_y,
                            f"{short}\n{d:,}cy",
                            ha='center', va='center', fontsize=5.5, color='white',
                            clip_on=True, zorder=4)
                max_c = max(max_c, e2)
    return max_c


def make_figure(aic_intervals_files, aiv_intervals_files, output_path, title=""):
    """
    生成 2 panel 图:
      Panel 1 (上): 统一坐标系 C/V 核时序图 — 每个 module 独占一行
                    上半 = AIC (Cube Core), 分隔线, 下半 = AIV (Vector Core)
      Panel 2 (下): 各阶段平均 Cycle 耗时柱状图
    """
    # ---- 聚合统计 ----
    aic_agg = aggregate_intervals(aic_intervals_files)
    aiv_agg = aggregate_intervals(aiv_intervals_files)

    # ---- 选代表核（区间最多的那个核）----
    aic_repr_file = pick_timeline_core(aic_intervals_files)
    aiv_repr_file = pick_timeline_core(aiv_intervals_files)
    aic_repr = aic_intervals_files.get(aic_repr_file, {}) if aic_repr_file else {}
    aiv_repr = aiv_intervals_files.get(aiv_repr_file, {}) if aiv_repr_file else {}

    # ---- 统一时间零点：取 AIC 和 AIV 日志中所有 mark 最小 cycle ----
    def _repr_min(idict):
        m = float('inf')
        for ilist in idict.values():
            for (s, _, _) in ilist:
                m = min(m, s)
        return m if m != float('inf') else 0

    global_t0 = min(_repr_min(aic_repr), _repr_min(aiv_repr))

    # ---- 行布局 ----
    ROW_STEP  = 0.82   # 行间距（y 轴单位）
    DIVIDER_Y = 0.0    # AIC/AIV 分隔线位置

    n_aic = len(_AIC_ROW_SPECS)   # 12
    n_aiv = len(_AIV_ROW_SPECS)   # 11

    # AIC: Init 在最顶部(最大 y)，Uninit 紧靠分隔线上方
    aic_row_ys = [(n_aic - i) * ROW_STEP for i in range(n_aic)]
    # AIV: InitBuffers 紧靠分隔线下方，FD_CopyFinalResOut 在最底部
    aiv_row_ys = [-(i + 1) * ROW_STEP for i in range(n_aiv)]

    y_top = aic_row_ys[0]  + ROW_STEP * 0.7
    y_bot = aiv_row_ys[-1] - ROW_STEP * 0.7

    # ---- 预先计算 x 范围（使用 global_t0）----
    def _repr_max(idict):
        m = 0
        for ilist in idict.values():
            for (_, e, _) in ilist:
                m = max(m, e - global_t0)
        return m

    x_max = max(_repr_max(aic_repr), _repr_max(aiv_repr), 100) * 1.05

    # ---- 图形尺寸（根据行数动态计算高度）----
    gantt_h_ratio = n_aic + n_aiv          # Gantt 高度比例
    bar_h_ratio   = 8                       # 柱状图高度比例
    fig_height    = max(28, (n_aic + n_aiv) * 1.05 + 12)
    fig = plt.figure(figsize=(34, fig_height))
    fig.patch.set_facecolor('#0F0F1A')

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[gantt_h_ratio, bar_h_ratio],
        hspace=0.22, left=0.17, right=0.97, top=0.95, bottom=0.04
    )
    ax_gantt = fig.add_subplot(gs[0])
    ax_bar   = fig.add_subplot(gs[1])

    for ax in [ax_gantt, ax_bar]:
        ax.set_facecolor('#111122')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#445566')
        ax.tick_params(colors='#BBCCDD', labelsize=8)
        ax.xaxis.label.set_color('#AABBCC')
        ax.yaxis.label.set_color('#AABBCC')

    # ==========================================================
    # Panel 1 — 统一坐标系 C/V 核逐行时序图
    # ==========================================================

    # 背景色带
    ax_gantt.axhspan(DIVIDER_Y, y_top + 0.1,   facecolor='#141830', alpha=0.8, zorder=0)
    ax_gantt.axhspan(y_bot - 0.1, DIVIDER_Y,   facecolor='#181408', alpha=0.8, zorder=0)

    # 行间水平细网格线（辅助阅读）
    for y in aic_row_ys + aiv_row_ys:
        ax_gantt.axhline(y, color='#2A2A3A', linewidth=0.4, zorder=1)

    # 绘制 AIC 逐行泳道
    aic_yt_pos, aic_yt_lbl = _draw_multirow(
        ax_gantt, _AIC_ROW_SPECS, aic_row_ys, aic_repr, global_t0, x_max)

    # 绘制 AIV 逐行泳道
    aiv_yt_pos, aiv_yt_lbl = _draw_multirow(
        ax_gantt, _AIV_ROW_SPECS, aiv_row_ys, aiv_repr, global_t0, x_max)

    # AIC / AIV 分隔线
    ax_gantt.axhline(DIVIDER_Y, color='#8899CC', linewidth=2.8, zorder=6)
    ax_gantt.text(x_max * 0.003, DIVIDER_Y + ROW_STEP * 0.18,
                  "--- AIC (Cube Core) above  |  AIV (Vector Core) below ---",
                  fontsize=8.5, color='#9999CC', fontstyle='italic', zorder=7)

    # 区域大标签（左侧）
    ax_gantt.text(x_max * -0.005, (y_top + DIVIDER_Y) / 2,
                  "[AIC]\nCube Core",
                  ha='right', va='center', fontsize=11, color='#5577FF',
                  fontweight='bold', transform=ax_gantt.transData)
    ax_gantt.text(x_max * -0.005, (DIVIDER_Y + y_bot) / 2,
                  "[AIV]\nVec Core",
                  ha='right', va='center', fontsize=11, color='#FF9933',
                  fontweight='bold', transform=ax_gantt.transData)

    # Y 轴刻度：显示模块名 + 打点标记码
    all_yt_pos = aic_yt_pos + aiv_yt_pos
    all_yt_lbl = aic_yt_lbl + aiv_yt_lbl
    ax_gantt.set_yticks(all_yt_pos)
    ax_gantt.set_yticklabels(all_yt_lbl, fontsize=7.2, color='#CCDDEE',
                              fontfamily='monospace')

    # 总耗时尾部标注
    max_aic = _repr_max(aic_repr)
    max_aiv = _repr_max(aiv_repr)
    for max_c, y_ref, clr, lbl in [
        (max_aic, aic_row_ys[0],  '#6688FF', 'AIC total'),
        (max_aiv, aiv_row_ys[-1], '#FF9933', 'AIV total'),
    ]:
        if max_c > 0:
            ax_gantt.annotate(
                f"{lbl} = {max_c:,} cycles",
                xy=(max_c, y_ref),
                xytext=(max_c * 0.85, y_ref + ROW_STEP * (0.9 if lbl.startswith('AIC') else -0.9)),
                fontsize=9, color=clr, fontweight='bold', zorder=8,
                arrowprops=dict(arrowstyle='->', color=clr, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#050510',
                          edgecolor=clr, alpha=0.9))

    # 流水并行重叠区检测 (BMM2 vs Vec1)
    gk_bmm2 = _gk_of_name("BMM2")
    gk_vec1 = _gk_of_name("Vec1")
    if gk_bmm2 and gk_vec1:
        bmm2_row_y = aic_row_ys[_AIC_ROW_SPECS.index(("BMM2", 0))]
        vec1_row_y = aiv_row_ys[_AIV_ROW_SPECS.index(("Vec1", 0))]
        for (as_, ae, _) in aic_repr.get(gk_bmm2, []):
            for (vs, ve, _) in aiv_repr.get(gk_vec1, []):
                ov_s = max(as_ - global_t0, vs - global_t0)
                ov_e = min(ae - global_t0, ve - global_t0)
                if ov_e > ov_s:
                    ax_gantt.axvspan(ov_s, ov_e, ymin=0.02, ymax=0.98,
                                     color='#FFFF55', alpha=0.05, zorder=2)
                    mid = (ov_s + ov_e) / 2
                    ax_gantt.annotate(
                        '', xy=(ov_e, DIVIDER_Y), xytext=(ov_s, DIVIDER_Y),
                        arrowprops=dict(arrowstyle='<->', color='#FFEE44', lw=1.8))
                    ax_gantt.text(mid, DIVIDER_Y - ROW_STEP * 0.25,
                                  f"pipeline overlap {ov_e-ov_s:,}cy",
                                  ha='center', va='top', fontsize=7,
                                  color='#FFEE77', fontweight='bold', zorder=7)

    # 垂直网格
    ax_gantt.grid(axis='x', color='#222233', linestyle='--',
                  alpha=0.8, linewidth=0.6, zorder=1)
    ax_gantt.set_xlim(-x_max * 0.003, x_max)
    ax_gantt.set_ylim(y_bot - 0.3, y_top + 0.3)
    ax_gantt.set_xlabel(
        "Cycles  [统一零点 = min(AIC, AIV) 最早 mark cycle，AIC 与 AIV 共享同一时间轴]",
        fontsize=9, color='#99AACC')
    ax_gantt.set_title(
        "C/V Core Pipeline Timeline  —  Unified Coordinate System"
        "  (upper=AIC Cube Core | lower=AIV Vector Core)",
        color='#DDEEFF', fontsize=12, fontweight='bold', loc='left', pad=9)

    # ==========================================================
    # Panel 2 — 各阶段平均 Cycle 耗时柱状图
    # ==========================================================
    ax_bar.set_title("[Stats]  Average Cycle Cost per Stage  (all cores mean)",
                      color='#DDEEFF', fontsize=11, fontweight='bold', loc='left', pad=6)

    bar_order_aic = [(gn, 0, aic_agg) for gn, _ in _AIC_ROW_SPECS]
    bar_order_aiv = [(gn, 1, aiv_agg) for gn, _ in _AIV_ROW_SPECS]
    all_bar_order = bar_order_aic + bar_order_aiv

    bar_labels, bar_avgs, bar_mins, bar_maxs, bar_colors, bar_counts = \
        [], [], [], [], [], []
    bar_mark_strs = []

    for gname, core_side, agg in all_bar_order:
        gk = _gk_of_name(gname)
        if gk is None or gk not in agg:
            continue
        stats = agg[gk]
        short_lbl = _get_label(gk)
        mark_str  = _GK_TO_MARK_STR.get(gk, "")
        bar_labels.append(f"{short_lbl}\n[{mark_str}]")
        bar_avgs.append(stats['avg'])
        bar_mins.append(stats['min'])
        bar_maxs.append(stats['max'])
        bar_colors.append(_get_color(gk))
        bar_counts.append(stats['count'])
        bar_mark_strs.append(mark_str)

    if bar_labels:
        x    = np.arange(len(bar_labels))
        bars = ax_bar.bar(x, bar_avgs, color=bar_colors, alpha=0.85,
                          edgecolor='white', linewidth=0.6, zorder=3)

        # min/max 误差线
        err_lo = [a - mn for a, mn in zip(bar_avgs, bar_mins)]
        err_hi = [mx - a  for a, mx in zip(bar_avgs, bar_maxs)]
        ax_bar.errorbar(x, bar_avgs, yerr=[err_lo, err_hi],
                        fmt='none', color='#FFD700', capsize=4,
                        linewidth=1.2, zorder=4)

        # 数值标注（柱顶）
        ymax_val = max(bar_maxs) if bar_maxs else 1000
        for xi, (bar, avg, cnt) in enumerate(zip(bars, bar_avgs, bar_counts)):
            ax_bar.text(xi, bar.get_height() + ymax_val * 0.008,
                        f"{avg:,.0f}cy\nn={cnt}",
                        ha='center', va='bottom', fontsize=6.5,
                        color='#DDEEFF', fontweight='bold', zorder=5)

        # AIC / AIV 背景区
        n_aic_bars = sum(1 for gn, _, agg in bar_order_aic
                         if _gk_of_name(gn) is not None and _gk_of_name(gn) in agg)
        n_aiv_bars = sum(1 for gn, _, agg in bar_order_aiv
                         if _gk_of_name(gn) is not None and _gk_of_name(gn) in agg)
        ymax = ymax_val * 1.30
        if n_aic_bars > 0:
            ax_bar.axvspan(-0.5, n_aic_bars - 0.5,
                           alpha=0.07, color='#4477FF', zorder=0)
            ax_bar.text((n_aic_bars - 1) / 2, ymax * 0.97,
                        "AIC (Cube Core)",
                        ha='center', va='top', fontsize=9.5,
                        color='#5588FF', fontweight='bold')
        if n_aiv_bars > 0:
            ax_bar.axvspan(n_aic_bars - 0.5, n_aic_bars + n_aiv_bars - 0.5,
                           alpha=0.07, color='#FF8800', zorder=0)
            ax_bar.text(n_aic_bars + (n_aiv_bars - 1) / 2, ymax * 0.97,
                        "AIV (Vector Core)",
                        ha='center', va='top', fontsize=9.5,
                        color='#FF9933', fontweight='bold')
        if n_aic_bars > 0 and n_aiv_bars > 0:
            ax_bar.axvline(n_aic_bars - 0.5,
                           color='#888899', linestyle=':', linewidth=1.5)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bar_labels, fontsize=6.8, color='#CCDDEE',
                                fontfamily='monospace', rotation=0)
        ax_bar.set_ylabel("Cycles (average)", color='#AABBCC', fontsize=9)
        ax_bar.set_ylim(0, ymax)
        ax_bar.grid(axis='y', color='#223344', linestyle='--',
                    alpha=0.6, linewidth=0.5, zorder=0)
        ax_bar.legend(
            handles=[mpatches.Patch(facecolor='none', edgecolor='#FFD700',
                                    linewidth=1.5, label='min/max range')],
            loc='upper right', fontsize=8.5, labelcolor='#CCDDEE',
            facecolor='#111122', edgecolor='#556677')

    # ---- 全局标题 ----
    suptitle = title or "Flash Attention Kernel  --  Simulation Cycle Profiling"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', color='#EEEEFF', y=0.97)

    # ---- 控制台文字报告 ----
    all_report = bar_order_aic + bar_order_aiv
    print("\n" + "=" * 78)
    print("  Flash Attention Cycle 耗时统计报告")
    print("=" * 78)
    print(f"  {'模块':<24} {'核':<5} {'打点标记码':<26}"
          f" {'平均cy':>10} {'最小cy':>10} {'最大cy':>10} {'计数':>6}")
    print("-" * 78)
    for gname, _, agg in all_report:
        gk = _gk_of_name(gname)
        if gk is None or gk not in agg:
            continue
        stats = agg[gk]
        ct    = MARK_TABLE.get(gk, (BOTH,))[0]
        lbl   = _get_label(gk)
        mstr  = _GK_TO_MARK_STR.get(gk, "")
        print(f"  {lbl:<24} {ct:<5} {mstr:<26}"
              f" {stats['avg']:>10,.1f} {stats['min']:>10,}"
              f" {stats['max']:>10,} {stats['count']:>6}")
    print("=" * 78)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n[OK] Saved: {output_path}")
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="解析 Flash Attention 仿真日志，提取 set_mark COND 打点，绘制 Cycle 耗时图")
    parser.add_argument("--aic_log", nargs='+', required=True,
                        help="AIC (Cube核) instr_log 文件路径，支持 glob，如 core*_aic.log")
    parser.add_argument("--aiv_log", nargs='+', required=True,
                        help="AIV (Vector核) instr_log 文件路径，支持 glob，如 core*_aiv.log")
    parser.add_argument("--output", type=str, default="fa_cycle_profile.png",
                        help="输出图像路径 (default: fa_cycle_profile.png)")
    parser.add_argument("--title", type=str, default="",
                        help="图标题（留空则自动生成）")
    args = parser.parse_args()

    print(f"\n[解析 AIC 日志]")
    aic_raw = parse_log_files(args.aic_log)
    print(f"[解析 AIV 日志]")
    aiv_raw = parse_log_files(args.aiv_log)

    if not aic_raw and not aiv_raw:
        print("[ERROR] 两个日志均未找到有效 COND 写入记录")
        sys.exit(1)

    print("\n[提取区间]")
    aic_intervals = {fp: compute_intervals(evts) for fp, evts in aic_raw.items()}
    aiv_intervals = {fp: compute_intervals(evts) for fp, evts in aiv_raw.items()}

    total_aic = sum(len(v) for idict in aic_intervals.values() for v in idict.values())
    total_aiv = sum(len(v) for idict in aiv_intervals.values() for v in idict.values())
    print(f"  AIC: {total_aic} 条完整区间 ({len(aic_intervals)} 核)")
    print(f"  AIV: {total_aiv} 条完整区间 ({len(aiv_intervals)} 核)")

    title = args.title or (
        f"Flash Attention Kernel — Simulation Cycle Profiling  "
        f"[AIC×{len(aic_intervals)} AIV×{len(aiv_intervals)}]"
    )

    print(f"\n[绘图] → {args.output}")
    make_figure(aic_intervals, aiv_intervals, args.output, title=title)


if __name__ == "__main__":
    main()

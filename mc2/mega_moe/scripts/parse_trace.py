#!/usr/bin/env python3
"""
MegaMoe trace CSV 解析器（独立运行 / 也可被 mega_moe.py 调用）
==============================================================
从 megamoe_trace_rank<N>_<ts>.csv 读取一 rank 的时间戳矩阵（108 cores × 260 points），
打印 stage / dispatch / wave sub-phase / pair[0] timeline 统计。

CSV layout（与 mega_moe.py 写出对齐）：
    rank, side, core_idx, p0_KERNEL_ENTRY, ..., p4_PROCESS_END,
    p5_disp_e0_begin, p6_disp_e0_end, ..., p20_disp_e7_end,
    p21_w0_WAVE_ENTRY, p22_w0_GMM1_BEGIN, ..., p248_w18_WAVE_END,
    [可能有少量 p249..p259_raw 占位]

每个 csv 包含一个 rank 的全部 108 个 core（AIC 36 + AIV 72，AIV0=偶数，AIV1=奇数）。

约束：cube 与 vec 不在同一 cycle 时钟域 —— **只在同核内（或同 pair 内 = AIC[p] / AIV[2p] / AIV[2p+1]）做减法**。

用法：
    python parse_trace.py megamoe_trace_rank0_*.csv
    python parse_trace.py rank0.csv rank1.csv
    python parse_trace.py *.csv

作为 module:
    from parse_trace import analyze_csv
    analyze_csv('rank0.csv')
"""

import csv
import os
import sys
import glob
import numpy as np

# ============================================================================
# 与 op_kernel/mega_moe_base.h 的 MegaMoeTrace 必须保持一致
# ============================================================================
POINTS_PER_CORE = 260
MAX_AIC = 36
MAX_AIV = 72
TOTAL_CORES = MAX_AIC + MAX_AIV  # 108

# Stage
P_KERNEL_ENTRY = 0
P_ROUTING_DONE = 1
P_CROSSRANK_DONE = 2
P_EXPERT_LOOP_END = 3
P_PROCESS_END = 4
P_STAGE_NUM = 5
STAGE_NAMES = ['KERNEL_ENTRY', 'ROUTING_DONE', 'CROSSRANK_DONE', 'EXPERT_LOOP_END', 'PROCESS_END']

# Dispatch (AIV1 only)
P_DISPATCH_BASE = 5
MAX_EXPERTS_TRACED = 8
DISPATCH_PTS_PER_EXPERT = 2

# Wave
P_WAVE_BASE = P_DISPATCH_BASE + MAX_EXPERTS_TRACED * DISPATCH_PTS_PER_EXPERT  # 21
WAVE_POINTS_PER_ITER = 12
MAX_WAVES_TRACED = (POINTS_PER_CORE - P_WAVE_BASE) // WAVE_POINTS_PER_ITER  # 19

SP_WAVE_ENTRY = 0
SP_GMM1_BEGIN = 1
SP_GMM1_FIRST_WAIT_DONE = 2
SP_GMM1_FIRST_TILE_DONE = 3
SP_GMM1_LAST_TILE_DONE = 4
SP_GMM1_END = 5
SP_GMM2_BEGIN = 6
SP_GMM2_FIRST_WAIT_DONE = 7
SP_GMM2_FIRST_TILE_DONE = 8
SP_GMM2_LAST_TILE_DONE = 9
SP_GMM2_END = 10
SP_WAVE_END = 11
SP_NAMES = [
    'WAVE_ENTRY', 'GMM1_BEGIN', 'GMM1_FIRST_WAIT_DONE', 'GMM1_FIRST_TILE_DONE',
    'GMM1_LAST_TILE_DONE', 'GMM1_END', 'GMM2_BEGIN', 'GMM2_FIRST_WAIT_DONE',
    'GMM2_FIRST_TILE_DONE', 'GMM2_LAST_TILE_DONE', 'GMM2_END', 'WAVE_END',
]

# wave-by-wave 表的 sub-phase 定义：(from_sp, to_sp, short_header, long_legend)
#   仅同核内做减法；不同核类型语义不同所以分三套
AIC_PHASES = [
    (SP_WAVE_ENTRY,           SP_GMM1_BEGIN,            'entr->g1B',  'enter -> GMM1 call'),
    (SP_GMM1_BEGIN,           SP_GMM1_FIRST_WAIT_DONE,  'wDsp',       'GMM1 wait dispatch flag'),
    (SP_GMM1_FIRST_WAIT_DONE, SP_GMM1_FIRST_TILE_DONE,  'g1Frst',     'GMM1 first tile mmad'),
    (SP_GMM1_FIRST_TILE_DONE, SP_GMM1_LAST_TILE_DONE,   'g1Mid',      'GMM1 mid+last tiles'),
    (SP_GMM1_LAST_TILE_DONE,  SP_GMM1_END,              'g1Drn',      'GMM1 end-drain (EndSync)'),
    (SP_GMM1_END,             SP_GMM2_BEGIN,            'g12gap',     'between GMM1 & GMM2 (mega_moe.h gap)'),
    (SP_GMM2_BEGIN,           SP_GMM2_FIRST_WAIT_DONE,  'wSwi',       'GMM2 wait swiglu flag'),
    (SP_GMM2_FIRST_WAIT_DONE, SP_GMM2_FIRST_TILE_DONE,  'g2Frst',     'GMM2 first tile mmad'),
    (SP_GMM2_FIRST_TILE_DONE, SP_GMM2_LAST_TILE_DONE,   'g2Mid',      'GMM2 mid+last tiles'),
    (SP_GMM2_LAST_TILE_DONE,  SP_GMM2_END,              'g2Drn',      'GMM2 end-drain'),
    (SP_GMM2_END,             SP_WAVE_END,              'g2->ex',     'GMM2 end -> wave end'),
    (SP_WAVE_ENTRY,           SP_WAVE_END,              'TOTAL',      'TOTAL wave'),
]
AIV0_PHASES = [
    (SP_WAVE_ENTRY,           SP_GMM1_BEGIN,            'entr->g1B',  'enter -> GMM1 call'),
    (SP_GMM1_BEGIN,           SP_GMM1_FIRST_WAIT_DONE,  'wCub',       'epi wait cube'),
    (SP_GMM1_FIRST_WAIT_DONE, SP_GMM1_FIRST_TILE_DONE,  'epiFr',      'epi first tile'),
    (SP_GMM1_FIRST_TILE_DONE, SP_GMM1_LAST_TILE_DONE,   'epiMd',      'epi mid+last tiles'),
    (SP_GMM1_LAST_TILE_DONE,  SP_GMM1_END,              'epiDr',      'epi end-drain'),
    (SP_GMM1_END,             SP_WAVE_END,              'g1->ex',     'GMM1 end -> wave end (sub0 idle)'),
    (SP_WAVE_ENTRY,           SP_WAVE_END,              'TOTAL',      'TOTAL wave'),
]
AIV1_PHASES = [
    (SP_WAVE_ENTRY,           SP_GMM2_BEGIN,            'entr->g2B',  'enter -> GMM2 call (sub1 idle)'),
    (SP_GMM2_BEGIN,           SP_GMM2_FIRST_WAIT_DONE,  'wRowR',      'combine wait rowReady'),
    (SP_GMM2_FIRST_WAIT_DONE, SP_GMM2_FIRST_TILE_DONE,  'cbFr',       'combine first tile'),
    (SP_GMM2_FIRST_TILE_DONE, SP_GMM2_LAST_TILE_DONE,   'cbMd',       'combine mid+last tiles'),
    (SP_GMM2_LAST_TILE_DONE,  SP_GMM2_END,              'cbEnd',      'combine end -> GMM2 end'),
    (SP_GMM2_END,             SP_WAVE_END,              'g2->ex',     'GMM2 end -> wave end'),
    (SP_WAVE_ENTRY,           SP_WAVE_END,              'TOTAL',      'TOTAL wave'),
]


# ============================================================================
# CSV header（也用于 mega_moe.py 写出时复用，保持一致性）
# ============================================================================
def trace_csv_header():
    cols = ['rank', 'side', 'core_idx']
    cols += [f'p{P_KERNEL_ENTRY}_KERNEL_ENTRY',
             f'p{P_ROUTING_DONE}_ROUTING_DONE',
             f'p{P_CROSSRANK_DONE}_CROSSRANK_DONE',
             f'p{P_EXPERT_LOOP_END}_EXPERT_LOOP_END',
             f'p{P_PROCESS_END}_PROCESS_END']
    for e in range(MAX_EXPERTS_TRACED):
        cols.append(f'p{P_DISPATCH_BASE + e*2}_disp_e{e}_begin')
        cols.append(f'p{P_DISPATCH_BASE + e*2 + 1}_disp_e{e}_end')
    for w in range(MAX_WAVES_TRACED):
        for sp in range(WAVE_POINTS_PER_ITER):
            cols.append(f'p{P_WAVE_BASE + w*WAVE_POINTS_PER_ITER + sp}_w{w}_{SP_NAMES[sp]}')
    expected = 3 + POINTS_PER_CORE
    while len(cols) < expected:
        cols.append(f'p{len(cols) - 3}_raw')
    return cols


# ============================================================================
# CSV → matrix
# ============================================================================
_US_SANE_UPPER = 1 << 60  # us 合理上界（~3.6e17，36500 年）；超出视为脏值


def load_csv(csv_path):
    """读 CSV，返回 (rank, ts_matrix[108, 260])。ts 单位 us，dtype int64。
       脏值过滤：单元格为 uint64 残留值（接近 2^64）会 overflow int64，统一置 0。
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(rows) < TOTAL_CORES:
        print(f"[WARN] {csv_path}: 期望 {TOTAL_CORES} 行 cores，实际 {len(rows)} 行（继续解析已有的）")
    rank = int(rows[0][0]) if rows else -1
    n_rows = min(len(rows), TOTAL_CORES)
    ts = np.zeros((TOTAL_CORES, POINTS_PER_CORE), dtype=np.int64)
    n_dirty = 0
    for i in range(n_rows):
        row = rows[i]
        # row = [rank, side, core_idx, p0, p1, ..., p259]
        if len(row) < 3 + POINTS_PER_CORE:
            continue
        for j in range(POINTS_PER_CORE):
            try:
                v = int(row[3 + j])
            except (ValueError, IndexError):
                ts[i, j] = 0
                continue
            # 合理 us：[0, 2^60)；超出（含负数或天文级 uint64 残留）一律视为脏值 → 0
            if 0 <= v < _US_SANE_UPPER:
                ts[i, j] = v
            else:
                ts[i, j] = 0
                n_dirty += 1
    if n_dirty:
        print(f"[INFO] {csv_path}: filtered {n_dirty} dirty cells (out-of-range u64) → 0")
    return rank, ts


# ============================================================================
# 报告
# ============================================================================
def _summarize(diffs, hotspot_thresh_us=2):
    if diffs.size == 0:
        return "<no data>"
    mn = int(diffs.min()); mx = int(diffs.max()); avg = int(diffs.mean())
    hot = '  <-- HOT' if avg > hotspot_thresh_us else ''
    return f"n={diffs.size:>4}  min={mn:>8}  avg={avg:>8}  max={mx:>8}  (us){hot}"


def _valid_mask(block, sp_idx):
    """脏值过滤：sp 值必须在 [KERNEL_ENTRY, PROCESS_END + 余量] 内才视为有效。
    KERNEL_ENTRY/PROCESS_END 自身 <=0 则核不活跃。"""
    entry = block[:, P_KERNEL_ENTRY]
    end   = block[:, P_PROCESS_END]
    col   = block[:, sp_idx]
    upper = np.where(end > 0, end + (end - entry), np.int64(0))
    return (entry > 0) & (col > 0) & (col >= entry) & (col <= upper)


def infer_wave_to_expert(aiv1):
    """通过 AIV1 dispatch begin（同核内时间）+ wave entry 推断 wave -> expert。
       返回 (n_cores, MAX_WAVES_TRACED) int32，-1 表示未知。
       前提：mega_moe.h 中 wave loop body 在所有 core（含 AIV1）同步 increment waveTraceIdx_，
       故同 pair 的 AIC/AIV0/AIV1 上同一 wave_idx 对应同一个 expert，借 AIV1 一处推断即可。
    """
    n = aiv1.shape[0]
    out = np.full((n, MAX_WAVES_TRACED), -1, dtype=np.int32)
    for ci in range(n):
        events = []  # (begin_us, expert_id)，仅取合法值
        for e in range(MAX_EXPERTS_TRACED):
            b = aiv1[ci, P_DISPATCH_BASE + 2 * e]
            if 0 < b < _US_SANE_UPPER:
                events.append((int(b), e))
        if not events:
            continue
        events.sort()
        for w in range(MAX_WAVES_TRACED):
            we = aiv1[ci, P_WAVE_BASE + w * WAVE_POINTS_PER_ITER + SP_WAVE_ENTRY]
            if not (0 < we < _US_SANE_UPPER):
                continue
            best_e = -1
            for (b, e) in events:
                if b <= we:
                    best_e = e
                else:
                    break
            out[ci, w] = best_e
    return out


def _wave_table(name, block, phase_list, expert_map=None):
    """打印一张 wave-by-wave 表。
       rows = wave_idx (跳过本侧全核都没打过的 wave)；
       cols = phase_list 短标签；cell = avg(us) across active cores。
       末尾两行 = all-wave 聚合 avg 和 max。
       expert_map=(n_cores, MAX_WAVES_TRACED) 时每行额外列出 mode-expert。
    """
    print(f"\n  --- [{name}] WAVE-by-WAVE  (us, avg across active cores) ---")
    col_w = max(7, max(len(t[2]) for t in phase_list))
    headers = [f"{t[2]:>{col_w}}" for t in phase_list]
    has_exp = expert_map is not None
    if has_exp:
        head = f"    {'exp':>3}  {'wave':>4} | " + " | ".join(headers)
    else:
        head = f"    {'wave':>4} | " + " | ".join(headers)
    sep = "    " + "-" * (len(head) - 4)
    print(head)
    print(sep)

    bucket = [[] for _ in phase_list]  # 跨 wave 收集每个 phase 的所有 diffs
    waves_printed = 0
    for w in range(MAX_WAVES_TRACED):
        ofs = P_WAVE_BASE + w * WAVE_POINTS_PER_ITER
        if not _valid_mask(block, ofs + SP_WAVE_ENTRY).any():
            continue
        waves_printed += 1
        cells = []
        for pi, (fs, ts_, _short, _long) in enumerate(phase_list):
            valid = _valid_mask(block, ofs + fs) & _valid_mask(block, ofs + ts_)
            if not valid.any():
                cells.append(f"{'---':>{col_w}}")
                continue
            d = block[valid, ofs + ts_] - block[valid, ofs + fs]
            bucket[pi].append(d)
            cells.append(f"{int(d.mean()):>{col_w}}")
        if has_exp:
            exps = expert_map[:, w]
            exps = exps[exps >= 0]
            if exps.size > 0:
                vals, counts = np.unique(exps, return_counts=True)
                exp_str = f"{vals[np.argmax(counts)]}"
            else:
                exp_str = '?'
            print(f"    {exp_str:>3}  {w:>4} | " + " | ".join(cells))
        else:
            print(f"    {w:>4} | " + " | ".join(cells))

    if waves_printed == 0:
        print("    <no wave data>")
        return

    print(sep)
    avg_cells, max_cells = [], []
    for ds in bucket:
        if ds:
            cat = np.concatenate(ds)
            avg_cells.append(f"{int(cat.mean()):>{col_w}}")
            max_cells.append(f"{int(cat.max()):>{col_w}}")
        else:
            avg_cells.append(f"{'---':>{col_w}}")
            max_cells.append(f"{'---':>{col_w}}")
    if has_exp:
        print(f"    {'':>3}  {'avg':>4} | " + " | ".join(avg_cells))
        print(f"    {'':>3}  {'max':>4} | " + " | ".join(max_cells))
    else:
        print(f"    {'avg':>4} | " + " | ".join(avg_cells))
        print(f"    {'max':>4} | " + " | ".join(max_cells))


def _print_legend(phase_list):
    col_w = max(7, max(len(t[2]) for t in phase_list))
    print("    legend:")
    for (_fs, _ts, short, long_lbl) in phase_list:
        print(f"      {short:>{col_w}}  =  {long_lbl}")


def analyze(ts, rank_id):
    """ts: (108, 260) int64，单位 us。
       输出（从粗到细）：
         1) Stage 跨整个 kernel
         2) Per-expert dispatch (AIV1)
         3) Wave-by-wave 表  AIC / AIV0 / AIV1 各一张，rows=wave，cols=sub-phase，
            每 cell = 该 wave 跨 active core 的 avg；底部 all-wave avg / max
            行首带 'exp' 标注（用 AIV1 dispatch 时间推断 wave→expert）
    """
    aic  = ts[:MAX_AIC]
    aiv  = ts[MAX_AIC:MAX_AIC + MAX_AIV]
    aiv0 = aiv[0::2]   # subBlock 0
    aiv1 = aiv[1::2]   # subBlock 1

    print(f"\n========== [rank {rank_id}] MegaMoe TIMESTAMPS (unit = us) ==========")
    print(f"  active AIC : {(aic[:, P_KERNEL_ENTRY]>0).sum()}/{MAX_AIC}, "
          f"active AIV0: {(aiv0[:, P_KERNEL_ENTRY]>0).sum()}/{MAX_AIC}, "
          f"active AIV1: {(aiv1[:, P_KERNEL_ENTRY]>0).sum()}/{MAX_AIC}")

    # 1) Stage 级
    def stage_report(name, block):
        print(f"\n  --- [{name}] STAGE durations (per-core delta) ---")
        for i in range(1, P_STAGE_NUM):
            valid = _valid_mask(block, i-1) & _valid_mask(block, i)
            if not valid.any():
                continue
            diffs = block[valid, i] - block[valid, i-1]
            print(f"    {STAGE_NAMES[i-1]} -> {STAGE_NAMES[i]:<22}{_summarize(diffs)}")
    stage_report('AIC',  aic)
    stage_report('AIV0', aiv0)
    stage_report('AIV1', aiv1)

    # 2) Per-expert dispatch
    print(f"\n  --- [AIV1] per-expert DISPATCH duration ---")
    for e in range(MAX_EXPERTS_TRACED):
        bi = P_DISPATCH_BASE + e * DISPATCH_PTS_PER_EXPERT
        ei = bi + 1
        valid = _valid_mask(aiv1, bi) & _valid_mask(aiv1, ei)
        if not valid.any():
            continue
        diffs = aiv1[valid, ei] - aiv1[valid, bi]
        print(f"    expert[{e}] dispatch  {_summarize(diffs)}")

    # 3) wave→expert 推断（基于 AIV1 dispatch 时间窗）；同 pair 的 AIC/AIV0 共用
    expert_map = infer_wave_to_expert(aiv1)

    # 4) Wave-by-wave 表（rows=wave, cols=sub-phase, cell = avg across cores）
    _wave_table('AIC',  aic,  AIC_PHASES,  expert_map=expert_map)
    _wave_table('AIV0', aiv0, AIV0_PHASES, expert_map=expert_map)
    _wave_table('AIV1', aiv1, AIV1_PHASES, expert_map=expert_map)

    # Legend（三套 phase 的长名一并打印）
    print(f"\n  --- LEGEND  AIC phases ---")
    _print_legend(AIC_PHASES)
    print(f"\n  --- LEGEND  AIV0 phases ---")
    _print_legend(AIV0_PHASES)
    print(f"\n  --- LEGEND  AIV1 phases ---")
    _print_legend(AIV1_PHASES)

    print(f"========== [rank {rank_id}] END TIMESTAMPS ==========\n")


def analyze_csv(csv_path):
    """直接吃 csv 路径。"""
    rank, ts = load_csv(csv_path)
    print(f"==> Parsing {csv_path}")
    analyze(ts, rank)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    # 支持 glob
    paths = []
    for arg in sys.argv[1:]:
        if any(c in arg for c in '*?['):
            paths.extend(sorted(glob.glob(arg)))
        else:
            paths.append(arg)
    if not paths:
        print("no csv matched")
        sys.exit(1)
    for path in paths:
        if not os.path.isfile(path):
            print(f"[ERROR] {path} not found")
            continue
        try:
            analyze_csv(path)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()

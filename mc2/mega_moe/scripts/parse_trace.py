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

# 三套 phase 定义（仅同核内做减法）
AIC_PHASES = [
    (SP_WAVE_ENTRY,           SP_GMM1_BEGIN,            'enter -> GMM1 call'),
    (SP_GMM1_BEGIN,           SP_GMM1_FIRST_WAIT_DONE,  'GMM1 wait dispatch flag'),
    (SP_GMM1_FIRST_WAIT_DONE, SP_GMM1_FIRST_TILE_DONE,  'GMM1 first tile mmad'),
    (SP_GMM1_FIRST_TILE_DONE, SP_GMM1_LAST_TILE_DONE,   'GMM1 mid+last tiles'),
    (SP_GMM1_LAST_TILE_DONE,  SP_GMM1_END,              'GMM1 end-drain (EndSync)'),
    (SP_GMM1_END,             SP_GMM2_BEGIN,            'between GMM1 & GMM2'),
    (SP_GMM2_BEGIN,           SP_GMM2_FIRST_WAIT_DONE,  'GMM2 wait swiglu flag'),
    (SP_GMM2_FIRST_WAIT_DONE, SP_GMM2_FIRST_TILE_DONE,  'GMM2 first tile mmad'),
    (SP_GMM2_FIRST_TILE_DONE, SP_GMM2_LAST_TILE_DONE,   'GMM2 mid+last tiles'),
    (SP_GMM2_LAST_TILE_DONE,  SP_GMM2_END,              'GMM2 end-drain'),
    (SP_GMM2_END,             SP_WAVE_END,              'GMM2 end -> wave end'),
    (SP_WAVE_ENTRY,           SP_WAVE_END,              'TOTAL wave (AIC)'),
]
AIV0_PHASES = [
    (SP_WAVE_ENTRY,           SP_GMM1_BEGIN,            'enter -> GMM1 call'),
    (SP_GMM1_BEGIN,           SP_GMM1_FIRST_WAIT_DONE,  'epi wait cube'),
    (SP_GMM1_FIRST_WAIT_DONE, SP_GMM1_FIRST_TILE_DONE,  'epi first tile'),
    (SP_GMM1_FIRST_TILE_DONE, SP_GMM1_LAST_TILE_DONE,   'epi mid+last tiles'),
    (SP_GMM1_LAST_TILE_DONE,  SP_GMM1_END,              'epi end-drain'),
    (SP_GMM1_END,             SP_WAVE_END,              'GMM1 end -> wave end (sub0 idle)'),
    (SP_WAVE_ENTRY,           SP_WAVE_END,              'TOTAL wave (AIV0)'),
]
AIV1_PHASES = [
    (SP_WAVE_ENTRY,           SP_GMM2_BEGIN,            'enter -> GMM2 call (idle on sub1)'),
    (SP_GMM2_BEGIN,           SP_GMM2_FIRST_WAIT_DONE,  'combine wait rowReady'),
    (SP_GMM2_FIRST_WAIT_DONE, SP_GMM2_FIRST_TILE_DONE,  'combine first tile'),
    (SP_GMM2_FIRST_TILE_DONE, SP_GMM2_LAST_TILE_DONE,   'combine mid+last tiles'),
    (SP_GMM2_LAST_TILE_DONE,  SP_GMM2_END,              'combine end -> GMM2 end'),
    (SP_GMM2_END,             SP_WAVE_END,              'GMM2 end -> wave end'),
    (SP_WAVE_ENTRY,           SP_WAVE_END,              'TOTAL wave (AIV1)'),
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
def load_csv(csv_path):
    """读 CSV，返回 (rank, ts_matrix[108, 260])。ts 单位 us，dtype int64。"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(rows) < TOTAL_CORES:
        print(f"[WARN] {csv_path}: 期望 {TOTAL_CORES} 行 cores，实际 {len(rows)} 行（继续解析已有的）")
    rank = int(rows[0][0]) if rows else -1
    n = min(len(rows), TOTAL_CORES)
    ts = np.zeros((TOTAL_CORES, POINTS_PER_CORE), dtype=np.int64)
    for i in range(n):
        row = rows[i]
        # row = [rank, side, core_idx, p0, p1, ..., p259]
        if len(row) < 3 + POINTS_PER_CORE:
            continue
        for j in range(POINTS_PER_CORE):
            try:
                ts[i, j] = np.int64(int(row[3 + j]))
            except (ValueError, IndexError):
                ts[i, j] = 0
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


def analyze(ts, rank_id):
    """ts: (108, 260) int64，单位 us。打印 stage / dispatch / wave / timeline 报告。"""
    aic  = ts[:MAX_AIC]
    aiv  = ts[MAX_AIC:MAX_AIC + MAX_AIV]
    aiv0 = aiv[0::2]   # subBlock 0
    aiv1 = aiv[1::2]   # subBlock 1

    print(f"\n========== [rank {rank_id}] MegaMoe TIMESTAMPS (unit = us) ==========")
    print(f"  active AIC : {(aic[:, P_KERNEL_ENTRY]>0).sum()}/{MAX_AIC}, "
          f"active AIV0: {(aiv0[:, P_KERNEL_ENTRY]>0).sum()}/{MAX_AIC}, "
          f"active AIV1: {(aiv1[:, P_KERNEL_ENTRY]>0).sum()}/{MAX_AIC}")

    # ---------- A) Stage 级 ----------
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

    # ---------- B) per-expert dispatch ----------
    print(f"\n  --- [AIV1] per-expert DISPATCH duration ---")
    for e in range(MAX_EXPERTS_TRACED):
        bi = P_DISPATCH_BASE + e * DISPATCH_PTS_PER_EXPERT
        ei = bi + 1
        valid = _valid_mask(aiv1, bi) & _valid_mask(aiv1, ei)
        if not valid.any():
            continue
        diffs = aiv1[valid, ei] - aiv1[valid, bi]
        print(f"    expert[{e}] dispatch  {_summarize(diffs)}")

    # ---------- C) wave sub-phase aggregated ----------
    def wave_phase_report(name, block, phase_list):
        print(f"\n  --- [{name}] WAVE sub-phase (aggregated over cores × waves) ---")
        for (from_sp, to_sp, lbl) in phase_list:
            diffs_all = []
            for w in range(MAX_WAVES_TRACED):
                ofs = P_WAVE_BASE + w * WAVE_POINTS_PER_ITER
                valid = _valid_mask(block, ofs + from_sp) & _valid_mask(block, ofs + to_sp)
                if not valid.any():
                    continue
                diffs_all.append(block[valid, ofs + to_sp] - block[valid, ofs + from_sp])
            if not diffs_all:
                print(f"    sp{from_sp:>2}->sp{to_sp:<2} {lbl:<40}<no data>")
                continue
            diffs = np.concatenate(diffs_all)
            print(f"    sp{from_sp:>2}->sp{to_sp:<2} {lbl:<40}{_summarize(diffs)}")

    wave_phase_report('AIC',  aic,  AIC_PHASES)
    wave_phase_report('AIV0', aiv0, AIV0_PHASES)
    wave_phase_report('AIV1', aiv1, AIV1_PHASES)

    # ---------- D) pair[0] 前几个 wave 的 timeline 示例 ----------
    print(f"\n  --- pair[0] timeline (us relative to each core's KERNEL_ENTRY) ---")
    for w in range(min(MAX_WAVES_TRACED, 5)):
        ofs = P_WAVE_BASE + w * WAVE_POINTS_PER_ITER
        if aic[0, ofs + SP_WAVE_ENTRY] <= aic[0, P_KERNEL_ENTRY]:
            continue
        def rel(block, sp):
            v = block[0, ofs + sp]
            e = block[0, P_KERNEL_ENTRY]
            if v <= e:
                return None
            return v - e
        parts = [f"    wave {w:>2}:"]
        for tag, sp in [('ENTRY', SP_WAVE_ENTRY),
                        ('G1_BEG', SP_GMM1_BEGIN),
                        ('G1_WAIT', SP_GMM1_FIRST_WAIT_DONE),
                        ('G1_END', SP_GMM1_END),
                        ('G2_BEG', SP_GMM2_BEGIN),
                        ('G2_WAIT', SP_GMM2_FIRST_WAIT_DONE),
                        ('G2_END', SP_GMM2_END),
                        ('EXIT', SP_WAVE_END)]:
            ar  = rel(aic,  sp)
            v0r = rel(aiv0, sp)
            v1r = rel(aiv1, sp)
            def fmt(x): return f"{x:>6}" if x is not None else '  ---'
            parts.append(f" {tag}[A{fmt(ar)} V0{fmt(v0r)} V1{fmt(v1r)}]")
        print('  '.join(parts))

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

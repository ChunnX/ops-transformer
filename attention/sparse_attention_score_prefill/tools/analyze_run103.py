#!/usr/bin/env python3
"""分析 run103 的 NPU dump 和 golden 数据，定位最早出现问题的地方"""
import os, sys, glob
import numpy as np
try:
    import torch
except ImportError:
    print("Error: torch not found"); sys.exit(1)

TORCH_DEVICE_BACKEND_AUTOLOAD = "0"

GOLDEN_DIR = "/tmp/sasa_golden_run103"
NPU_DUMP = "/tmp/a2_phase2_max_run103.pt"
NPU_LOG = "/tmp/a2_phase2_max_run103_npu.log"

def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def stats(name, t):
    if isinstance(t, torch.Tensor):
        arr = t.float().numpy()
    else:
        arr = np.array(t, dtype=np.float32)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    finite = arr[np.isfinite(arr)]
    if len(finite) > 0:
        return f"{name:40s} shape={str(list(arr.shape)):20s} min={finite.min():.6f} max={finite.max():.6f} mean={finite.mean():.6f} nan={has_nan} inf={has_inf}"
    else:
        return f"{name:40s} shape={str(list(arr.shape)):20s} ALL NAN/INF nan={has_nan} inf={has_inf}"

print("=" * 100)
print("1. INPUT PARAMETERS")
print("=" * 100)
for f in sorted(glob.glob(os.path.join(GOLDEN_DIR, "input_*.pt"))):
    t = load_pt(f)
    name = os.path.basename(f).replace("input_", "").replace(".pt", "")
    if not isinstance(t, torch.Tensor):
        print(f"  {name:30s} = {t}")
    elif t.numel() <= 20:
        print(f"  {name:30s} = {t.tolist()}")
    else:
        print(f"  {name:30s} shape={list(t.shape)} dtype={t.dtype}")

print()
print("=" * 100)
print("2. NPU OUTPUT vs GOLDEN OUTPUT")
print("=" * 100)
npu_out = load_pt(NPU_DUMP)
golden_out = load_pt(os.path.join(GOLDEN_DIR, "golden_output.pt"))
print(stats("NPU output", npu_out))
print(stats("Golden output", golden_out))
npu_arr = npu_out.float().numpy()
golden_arr = golden_out.float().numpy()
if npu_arr.shape == golden_arr.shape:
    diff = np.abs(npu_arr - golden_arr)
    finite_mask = np.isfinite(diff)
    if finite_mask.any():
        print(f"  Max diff (finite): {diff[finite_mask].max():.6f}")
    print(f"  NPU all -inf: {np.all(npu_arr == float('-inf'))}")
    print(f"  NPU all nan: {np.all(np.isnan(npu_arr))}")
    print(f"  NPU first 16: {npu_arr.flatten()[:16]}")
    print(f"  Golden first 16: {golden_arr.flatten()[:16]}")
else:
    print(f"  SHAPE MISMATCH: npu={npu_arr.shape} golden={golden_arr.shape}")

print()
print("=" * 100)
print("3. GOLDEN PHASE1 WORKSPACE (O_partial, max, sum)")
print("=" * 100)
ws_o = load_pt(os.path.join(GOLDEN_DIR, "phase1_ws_o.pt"))
ws_max = load_pt(os.path.join(GOLDEN_DIR, "phase1_ws_max.pt"))
ws_sum = load_pt(os.path.join(GOLDEN_DIR, "phase1_ws_sum.pt"))
print(stats("phase1_ws_o", ws_o))
print(stats("phase1_ws_max", ws_max))
print(stats("phase1_ws_sum", ws_sum))

# workspace shape: [total_q, kv_heads, topK, groupSize, D]
print(f"\n  Workspace layout: [total_q={ws_o.shape[0]}, kv_heads={ws_o.shape[1]}, topK={ws_o.shape[2]}, groupSize={ws_o.shape[3]}, D={ws_o.shape[4]}]")

print()
print("=" * 100)
print("4. GOLDEN PER-BLOCK RESULTS (first few)")
print("=" * 100)
# Check each (q, h, t) block
for q in range(4):
    for h in range(2):
        for t in range(3):
            tag = f"q{q}_h{h}_t{t}"
            s_path = os.path.join(GOLDEN_DIR, f"scores_S_{tag}.pt")
            p_path = os.path.join(GOLDEN_DIR, f"softmax_P_{tag}.pt")
            o_path = os.path.join(GOLDEN_DIR, f"pv_Opartial_{tag}.pt")
            mx_path = os.path.join(GOLDEN_DIR, f"rowMax_{tag}.pt")
            sm_path = os.path.join(GOLDEN_DIR, f"rowSum_{tag}.pt")
            if os.path.exists(s_path):
                s = load_pt(s_path)
                p = load_pt(p_path)
                o = load_pt(o_path)
                mx = load_pt(mx_path)
                sm = load_pt(sm_path)
                s_has_nan = torch.isnan(s).any().item()
                p_has_nan = torch.isnan(p).any().item()
                o_has_nan = torch.isnan(o).any().item()
                mx_has_inf = torch.isinf(mx).any().item()
                sm_finite = sm[sm > 0]
                sm_all_zero = (sm <= 0).all().item()
                status = "OK" if not (s_has_nan or p_has_nan or o_has_nan) else "NaN!"
                if mx_has_inf and not sm_all_zero:
                    status += " (mx has -inf but sum>0)"
                print(f"  {tag}: S{list(s.shape)} P{list(p.shape)} O{list(o.shape)} "
                      f"mx={mx.float().mean():.4f} sm={sm.float().mean():.4f} "
                      f"nan=[S={s_has_nan} P={p_has_nan} O={o_has_nan}] {status}")

print()
print("=" * 100)
print("5. NPU LOG ANALYSIS — key events timeline")
print("=" * 100)
with open(NPU_LOG, 'r') as f:
    lines = f.readlines()
print(f"  Total log lines: {len(lines)}")
# Find key events
key_patterns = [
    ("Tiling done", "TilingSparseAttentionScorePrefill DONE"),
    ("Kernel init", "[SASAPrefillA2] init"),
    ("QK start", "[CUBE] before QK"),
    ("QK sync", "[QK] before WaitCrossCoreSync"),
    ("QK done", "[CUBE] after QK"),
    ("PV start", "[CUBE] before PV"),
    ("Phase1 done", "phase1 passed"),
    ("SyncAll", "SyncAll passed"),
    ("Scatter", "[A2 scatter]"),
    ("Phase2 lse", "[A2 lse]"),
    ("Phase2 Opartial", "[A2 OpartialBits]"),
    ("saved output", "saved "),
    ("Error", "[ERROR]"),
    ("Assert", "[ASSERT]"),
    ("AIC error", "AIC error"),
    ("Timeout", "[Timeout]"),
]
for label, pattern in key_patterns:
    matches = [(i+1, line.strip()[:120]) for i, line in enumerate(lines) if pattern in line]
    if matches:
        print(f"\n  {label} ({len(matches)} occurrences):")
        for lineno, text in matches[:5]:
            print(f"    L{lineno}: {text}")
        if len(matches) > 5:
            print(f"    ... and {len(matches)-5} more")
    else:
        print(f"  {label}: NOT FOUND")

print()
print("=" * 100)
print("6. SCATTER ANALYSIS — which (q, h, t) slots were scattered?")
print("=" * 100)
scatter_lines = [line.strip() for line in lines if "[A2 scatter]" in line]
scattered_slots = set()
for line in scatter_lines:
    # Format: [AIV Block X/40] [A2 scatter] q=X kv=X slot=X lse=X local=X rowsum=X
    import re
    m = re.search(r'q=(\d+)\s+kv=(\d+)\s+slot=(\d+)\s+lse=(\d+)\s+local=(\d+)\s+rowsum=([\d.]+)', line)
    if m:
        q, kv, slot, lse, local, rowsum = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6))
        scattered_slots.add((q, kv, slot))
        print(f"  q={q} kv={kv} slot={slot} lse={lse} local={local} rowsum={rowsum:.4f}")
    else:
        print(f"  PARSE FAIL: {line[20:]}")

print(f"\n  Scattered slots: {sorted(scattered_slots)}")
print(f"  Expected slots: {[(q,h,t) for q in range(4) for h in range(2) for t in range(3)]}")
missing = set([(q,h,t) for q in range(4) for h in range(2) for t in range(3)]) - scattered_slots
print(f"  Missing slots: {sorted(missing)}")

print()
print("=" * 100)
print("7. PHASE2 ANALYSIS — did Phase 2 run?")
print("=" * 100)
phase2_lse = [line for line in lines if "[A2 lse]" in line]
phase2_opartial = [line for line in lines if "[A2 OpartialBits]" in line]
print(f"  Phase 2 lse prints: {len(phase2_lse)}")
print(f"  Phase 2 Opartial prints: {len(phase2_opartial)}")
if len(phase2_lse) == 0 and len(phase2_opartial) == 0:
    print("  *** PHASE 2 DID NOT RUN OR CRASHED BEFORE PRINTING ***")
    print("  *** Output remains initial -inf values ***")

# Check last 20 lines of log
print(f"\n  Last 10 lines of log:")
for line in lines[-10:]:
    print(f"    {line.rstrip()[:120]}")

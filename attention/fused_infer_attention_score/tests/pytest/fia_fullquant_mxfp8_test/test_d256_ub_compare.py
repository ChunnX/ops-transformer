#!/usr/bin/python3
"""Compare nLoops=1 vs nLoops=2 UB output for D[0:127]"""
import sys, os, torch, torch_npu, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import fia_fullquant_mxfp8_golden as g

g.D = 256; g.N_q = 8; g.N_kv = 8; g.B = 1
g.ACTUAL_SEQ_Q = [128]; g.ACTUAL_SEQ_KV = [128]
g.ENABLE_PA = False; g.ENABLE_ROPE = False
g.INPUT_LAYOUT = "TND"; g.SPARSE_MODE = 3; g.GRAPH_PATH = 0

torch.npu.set_device(0)
data = g.generate_data()
q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr, kr, bt = data
cpu_out, _ = g.cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale,
                                 g.ACTUAL_SEQ_Q, g.ACTUAL_SEQ_KV, qr, kr)
npu_out, _ = g.npu_mxfp8_fa(q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale,
                              g.ACTUAL_SEQ_Q, g.ACTUAL_SEQ_KV, bt, qr, kr)

cpu_out = cpu_out.cpu().float()
npu_out = npu_out.cpu().float()
cpu_tnd = cpu_out.squeeze(0).permute(1, 0, 2)  # [128, 8, 256]

SAVE_FILE = "/home/jyq/npu_output_d256.pt"
COMPARE_FILE = "/home/jyq/npu_output_d256_baseline.pt"

mode = sys.argv[1] if len(sys.argv) > 1 else "save"

if mode == "save":
    torch.save(npu_out, SAVE_FILE)
    # Also print accuracy for reference
    d0 = cpu_tnd[:,:,0:128].flatten(); n0 = npu_out[:,:,0:128].flatten()
    ic = torch.isclose(d0, n0, rtol=0.005, atol=0.000025)
    print(f"[SAVE] D[0:127]: {ic.sum().item()}/{ic.numel()} = {ic.sum().item()/ic.numel()*100:.2f}%")
    d1 = cpu_tnd[:,:,128:256].flatten(); n1 = npu_out[:,:,128:256].flatten()
    ic2 = torch.isclose(d1, n1, rtol=0.005, atol=0.000025)
    print(f"[SAVE] D[128:255]: {ic2.sum().item()}/{ic2.numel()} = {ic2.sum().item()/ic2.numel()*100:.2f}%")

elif mode == "compare":
    baseline = torch.load(COMPARE_FILE, map_location="cpu").float()
    current = npu_out
    # Compare D[0:127] between baseline (nLoops=1) and current (nLoops=2)
    b_d0 = baseline[:,:,0:128]
    c_d0 = current[:,:,0:128]
    diff = (b_d0 - c_d0).abs()
    match = torch.isclose(b_d0, c_d0, rtol=1e-5, atol=1e-6)
    print(f"=== D[0:127] UB OVERWRITE CHECK ===")
    print(f"Baseline (nLoops=1) vs Current (nLoops=2)")
    print(f"Match: {match.sum().item()}/{match.numel()} = {match.sum().item()/match.numel()*100:.2f}%")
    print(f"Max diff: {diff.max().item():.8f}")
    print(f"Mean diff: {diff.mean().item():.8f}")
    if not match.all():
        mismatch = ~match
        # Show first few mismatches
        idxs = mismatch.flatten().nonzero(as_tuple=True)[0][:10]
        for idx in idxs:
            t = idx.item() // (8*128)
            h = (idx.item() % (8*128)) // 128
            d = idx.item() % 128
            print(f"  MISMATCH at tok={t},head={h},d={d}: baseline={b_d0.flatten()[idx]:.6f} current={c_d0.flatten()[idx]:.6f}")
    # Also compare with CPU golden
    for label, ds, de in [("D[0:127]", 0, 128), ("D[128:255]", 128, 256)]:
        c = cpu_tnd[:,:,ds:de].flatten(); n = current[:,:,ds:de].flatten()
        ic = torch.isclose(c, n, rtol=0.005, atol=0.000025)
        print(f"Current vs CPU golden {label}: {ic.sum().item()}/{ic.numel()} = {ic.sum().item()/ic.numel()*100:.2f}%")

#!/usr/bin/env python3
"""测试 TND 格式用例 (0622.py — TND 路径即 unsqueeze→BSND)"""
import torch, torch_npu, math, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
mod = import_module("0622")
sparse_lightning_indexer_grad_kl_loss = mod.sparse_lightning_indexer_grad_kl_loss

WARMUP, BENCH = 3, 5

CASES = [
    {"S1": 1,    "S2": 2048, "N1": 128, "N2": 1, "Nidx1": 64, "Nidx2": 1,
     "DQuery": 512, "DQueryIndex": 128, "d_rope": 64, "K": 2048, "desc": "TND S1=1 (equiv BSND baseline)"},
    {"S1": 64,   "S2": 2048, "N1": 128, "N2": 1, "Nidx1": 64, "Nidx2": 1,
     "DQuery": 512, "DQueryIndex": 128, "d_rope": 64, "K": 2048, "desc": "TND S1=64"},
    {"S1": 128,  "S2": 2048, "N1": 128, "N2": 1, "Nidx1": 64, "Nidx2": 1,
     "DQuery": 512, "DQueryIndex": 128, "d_rope": 64, "K": 2048, "desc": "TND S1=128"},
    {"S1": 512,  "S2": 2048, "N1": 128, "N2": 1, "Nidx1": 64, "Nidx2": 1,
     "DQuery": 512, "DQueryIndex": 128, "d_rope": 64, "K": 2048, "desc": "TND S1=512"},
    {"S1": 2048, "S2": 8192, "N1": 128, "N2": 1, "Nidx1": 64, "Nidx2": 1,
     "DQuery": 512, "DQueryIndex": 128, "d_rope": 64, "K": 2048, "desc": "TND S1=2048 S2=8192"},
]


def gen_inputs(cfg, seed=42):
    S1, S2 = cfg["S1"], cfg["S2"]
    N1, N2 = cfg["N1"], cfg["N2"]
    Nidx1, Nidx2 = cfg["Nidx1"], cfg["Nidx2"]
    D, DI, dr = cfg["DQuery"], cfg["DQueryIndex"], cfg["d_rope"]
    K, G = cfg["K"], N1 // N2
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(seed)
    dt = torch.float16

    # TND tensors (3D: no batch dim)
    q  = torch.randn(S1, N1, D, dtype=dt).npu()
    k  = torch.randn(S2, N2, D, dtype=dt).npu()
    qi = torch.randn(S1, Nidx1, DI, dtype=dt).npu()
    ki = torch.randn(S2, Nidx2, DI, dtype=dt).npu()
    qr = torch.randn(S1, N1, dr, dtype=dt).npu()
    kr = torch.randn(S2, N2, dr, dtype=dt).npu()
    w  = torch.randn(S1, Nidx1, dtype=dt).npu()
    si = torch.zeros(S1, Nidx2, K, dtype=torch.int32).npu()

    for s in range(S1):
        s2_limit = S2 - S1 + s + 1
        if s2_limit <= 0:
            s2_limit = S2
        s2_rl = min(s2_limit, K)
        idx = torch.randperm(s2_limit)[:s2_rl]
        si[s, 0, :s2_rl] = idx.to(torch.int32).npu()
        if s2_rl < K:
            si[s, 0, s2_rl:] = -1
    torch.npu.synchronize()

    with torch.no_grad():
        km = k.squeeze(1); krm = kr.squeeze(1)
        qf = torch.cat([q, qr], dim=-1)
        kf = torch.cat([km, krm], dim=-1)
        kft = kf.transpose(-1, -2)
        mx_list, sm_list = [], []
        for s in range(S1):
            row = torch.matmul(qf[s:s+1], kft)
            rs = row.float() * scale
            rm = rs.max(dim=-1).values
            rsum = torch.exp(rs - rm.unsqueeze(-1)).sum(dim=-1)
            mx_list.append(rm); sm_list.append(rsum)
        mx = torch.cat(mx_list, dim=0); sm = torch.cat(sm_list, dim=0)
        sm_max = mx.reshape(1, S1, N2, G).permute(0, 2, 1, 3).contiguous().float().squeeze(0)
        sm_sum = sm.reshape(1, S1, N2, G).permute(0, 2, 1, 3).contiguous().float().squeeze(0)
    return q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr


def call_tl(q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr, cfg):
    return sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, scale,
        query_rope=qr, key_rope=kr, layout_query="TND",
        actual_seq_lengths_query=[cfg["S1"]],
        actual_seq_lengths_key=[cfg["S2"]], sparse_mode=3)


def call_off(q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr, cfg):
    return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, scale,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=[cfg["S1"]], actual_seq_klen=[cfg["S2"]],
        layout="TND", sparse_mode=3, pre_tokens=65536, next_tokens=65536)


def bench(fn, warmup=WARMUP, iters=BENCH):
    for _ in range(warmup):
        fn(); torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(); torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return sum(ts)/len(ts), ts[len(ts)//2], ts[0]


def main():
    print("=" * 60)
    print("TND 格式测试 (0622.py)")
    print("=" * 60)
    for i, cfg in enumerate(CASES):
        print(f"\n--- [{i}] {cfg['desc']} ---")
        q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr = gen_inputs(cfg)

        try:
            dqi, dki, dw, loss = call_tl(q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr, cfg)
            torch.npu.synchronize()
        except Exception as e:
            print(f"  TileLang FAIL: {e}")
            continue

        try:
            dqi_o, dki_o, dw_o, loss_o = call_off(q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr, cfg)
            torch.npu.synchronize()
        except Exception as e:
            print(f"  Official FAIL: {e}")
            continue

        for name, tl, off in [("dQI", dqi, dqi_o), ("dKI", dki, dki_o), ("dW", dw, dw_o)]:
            if tl.shape != off.shape:
                tl = tl[:off.shape[0]] if tl.dim() == off.dim() else tl
            d = (tl.float() - off.float()).abs()
            re = (d / (off.float().abs() + 1e-6)).mean().item()
            print(f"  {name}: max={d.max().item():.3e} mean={d.mean().item():.3e} rel={re:.3e}")

        la = abs(loss.float().item() - loss_o.float().item())
        lr = la / (abs(loss_o.float().item()) + 1e-6)
        print(f"  Loss: abs={la:.3e} rel={lr:.3e}")

        avg, med, mn = bench(lambda: call_tl(q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr, cfg))
        _, omed, _ = bench(lambda: call_off(q, k, qi, ki, w, si, sm_max, sm_sum, scale, qr, kr, cfg))
        print(f"  Perf: tl_med={med*1e3:.3f}ms off_med={omed*1e3:.3f}ms speedup={omed/med:.2f}x")


if __name__ == "__main__":
    main()

"""场景覆盖测试: pool_size=16, topk=16, mask_mode=3(causal) 的边界计数推导验证.

场景1: S1=48, S2=50池(L_orig=800+t >> S1)  → 每行恒 16+t 个有效索引
场景2: S1=48, S2=3池 (L_orig=48+t ≈ S1)   → 分三段: [0,14-t]:0 | [15-t,14]:16+(i+t-15) | [15,47]:16+t
tailK 扫描 t=0..15, BSND/TND 双布局, 与 golden(indices 集合+values)全对比.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atk"))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch_npu

torch_npu.npu.set_device(0)
import pool_key_indexer_golden as G

G.COMPARE_INDICES = True
from pool_key_indexer_golden import PKIGoldenRef, _get_compute_dtype
from result_compare import compare_indices, compare_values
from cann_ops_transformer import pool_key_indexer

HEAD_DIM, N2, N1 = 128, 1, 8
PS, TOPK = 16, 16
OUTLEN = TOPK + PS - 1  # 31


def expected_counts(s2_pools, s1, t):
    """按推导公式给出每行有效索引数."""
    l_orig = s2_pools * PS + t
    counts = []
    for i in range(s1):
        pools = min(l_orig // PS if l_orig >= 0 else 0, s2_pools)
        cutoff = (l_orig - s1 + i + 1) // PS
        vis_pools = max(0, min(cutoff, pools, TOPK // PS))
        vis_tail = max(0, min(t, (l_orig - s1 + i) - TOPK + 1)) if t > 0 else 0
        counts.append(vis_pools * PS + vis_tail)
    return counts


def run_case(layout, s2_pools, s1, t, seed=123):
    torch.manual_seed(seed)
    B = 1
    if layout == "BSND":
        query = torch.randn(B, s1, N1, HEAD_DIM, dtype=torch.float16)
        pool_key = torch.randn(B, s2_pools, N2, HEAD_DIM, dtype=torch.float16)
        weights = torch.randn(B, s1, N1, dtype=torch.float16)
        kwargs = dict(layout_q="BSND", layout_k="BSND")
        asq = ask = None
        golden_q, golden_k = query, pool_key
    else:  # TND
        query = torch.randn(s1, N1, HEAD_DIM, dtype=torch.float16)
        pool_key = torch.randn(s2_pools, N2, HEAD_DIM, dtype=torch.float16)
        weights = torch.randn(s1, N1, dtype=torch.float16)
        asq = torch.tensor([s1], dtype=torch.int64)
        ask = torch.tensor([s2_pools], dtype=torch.int64)  # 累计池数
        kwargs = dict(layout_q="TND", layout_k="TND")
        golden_q, golden_k = query, pool_key
    ptk = torch.tensor([t], dtype=torch.int64)

    npu_idx, npu_val = pool_key_indexer(
        query.npu(),
        pool_key.npu(),
        weights.npu(),
        ptk.npu(),
        actual_seq_q=(asq.npu() if asq is not None else None),
        actual_seq_k=(ask.npu() if ask is not None else None),
        topk=TOPK,
        pool_size=PS,
        mask_mode=3,
        return_value=True,
        **kwargs,
    )
    torch_npu.npu.synchronize()
    npu_idx = npu_idx.cpu()
    npu_val = npu_val.cpu()

    # 1) 每行有效计数 vs 推导
    exp = expected_counts(s2_pools, s1, t)
    rows = npu_idx[0] if layout == "BSND" else npu_idx  # (S1, OUTLEN)
    got = [(r >= 0).sum().item() for r in rows]
    ok_count = got == exp
    # 2) golden 全对比
    ref = PKIGoldenRef(
        B if layout == "BSND" else 1,
        s1,
        s2_pools * PS,
        N1,
        N2,
        HEAD_DIM,
        layout,
        layout,
        TOPK,
        PS,
        3,
        _get_compute_dtype(torch.float16),
    )
    g_idx, g_val = ref.forward(golden_q, golden_k, weights, asq, ask, pool_tail_k=ptk)
    ok_idx = compare_indices(g_idx, npu_idx)
    ok_val = compare_values(g_val, npu_val, torch.float16)
    return ok_count, got, exp, ok_idx, ok_val


def main():
    failures = 0
    for s2_pools, tag in [(50, "场景1 S2=50池"), (3, "场景2 S2=3池")]:
        for layout in ("BSND", "TND"):
            for t in range(PS):  # t=0..15 全扫
                okc, got, exp, oki, okv = run_case(layout, s2_pools, 48, t)
                status = "PASS" if (okc and oki and okv) else "FAIL"
                if status == "FAIL":
                    failures += 1
                    bad = [
                        (i, e, g) for i, (e, g) in enumerate(zip(exp, got)) if e != g
                    ][:4]
                    print(
                        f"[{status}] {tag} {layout} t={t}: 计数不符示例(i,期望,实际)={bad} "
                        f"idx_cmp={oki} val_cmp={okv}"
                    )
                else:
                    seg = f"rows0/15/47={got[0]}/{got[15]}/{got[47]}"
                    print(f"[{status}] {tag} {layout} t={t}: {seg}")
    print(
        f"\n==== 总计: {'ALL PASS' if failures == 0 else f'{failures} FAILURES'} ===="
    )


if __name__ == "__main__":
    main()

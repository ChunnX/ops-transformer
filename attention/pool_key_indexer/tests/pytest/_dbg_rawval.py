"""Analyze raw-values mismatch: golden vs NPU on one failing case."""

import sys
import torch
import torch_npu

sys.path.insert(0, "/opt/gjw/ops-transformer/attention/pool_key_indexer/tests/pytest")
from test_pool_key_indexer import (
    _build_bsnd_inputs,
    _run_npu,
    _run_golden,
    N2,
    HEAD_DIM,
)

torch_npu.npu.set_device(0)
torch.manual_seed(42)

# bsnd_bf16_ps8_nocausal_val: batch=2 q=32 k=256 n1=16 topk=128 ps=8 mask=0 val=True
batch, q_seq, k_seq, n1, pool_size, topk = 2, 32, 256, 16, 8, 128
dtype = torch.bfloat16
query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
    batch, q_seq, k_seq, n1, pool_size, dtype
)

g_idx, g_val = _run_golden(
    query,
    pool_key,
    weights,
    pool_tail_k,
    layout_q="BSND",
    layout_k="BSND",
    batch=batch,
    q_seq=q_seq,
    k_seq=k_seq,
    n1=n1,
    topk=topk,
    pool_size=pool_size,
    mask_mode=0,
    return_value=True,
    dtype=dtype,
)
n_idx, n_val = _run_npu(
    query,
    pool_key,
    weights,
    pool_tail_k,
    layout_q="BSND",
    layout_k="BSND",
    topk=topk,
    pool_size=pool_size,
    mask_mode=0,
    return_value=True,
)

print("g_val shape:", g_val.shape, "n_val shape:", n_val.shape)
print("g_idx shape:", g_idx.shape, "n_idx shape:", n_idx.shape)

# batch=1 row=0: FAIL with max_err=23.88
b, s = 1, 0
gv = g_val[b, s].float()
nv = n_val[b, s].float()
print("\n--- batch1 row0 raw values ---")
print("golden finite:", torch.isfinite(gv).sum().item(), "/", gv.numel())
print("npu    finite:", torch.isfinite(nv).sum().item(), "/", nv.numel())
print("golden first 8:", [f"{x:.3f}" for x in gv[:8].tolist()])
print("npu    first 8:", [f"{x:.3f}" for x in nv[:8].tolist()])
gs = gv[torch.isfinite(gv)].sort().values
ns = nv[torch.isfinite(nv)].sort().values
print("golden sorted first 5:", [f"{x:.3f}" for x in gs[:5].tolist()])
print("npu    sorted first 5:", [f"{x:.3f}" for x in ns[:5].tolist()])
print("golden sorted last 5:", [f"{x:.3f}" for x in gs[-5:].tolist()])
print("npu    sorted last 5:", [f"{x:.3f}" for x in ns[-5:].tolist()])

# raw values 与 pool indices 的对应关系: golden values[i] = score of pool_idx[i]
gi = g_idx[b, s]
ni = n_idx[b, s]
print("\ngolden idx first 16 (pools):", (gi[:16] // pool_size).tolist())
print("npu    idx first 16 (pools):", (ni[:16] // pool_size).tolist())
print(
    "golden values == scores[pool]? scores unavailable here, check norm-channel consistency"
)
EOF_MARKER = None

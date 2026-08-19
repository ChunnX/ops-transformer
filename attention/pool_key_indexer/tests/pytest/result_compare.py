# -*- coding: utf-8 -*-
"""
Comparison utilities for PoolKeyIndexer output validation.

Two-step comparison strategy (adapted from LIV2 result_compare_method.py):
  1. Row-level set comparison: sort valid indices/values (excluding -1/-inf) and
     check that the two sets overlap.
  2. Value-level rtol/atol comparison for the gathered scores.
"""
import numpy as np
import torch

RTOL = 0.005
ATOL_FP16 = 2.5e-5
ATOL_BF16 = 1e-4


def _atol_for_dtype(dt: torch.dtype) -> float:
    if dt == torch.bfloat16:
        return ATOL_BF16
    return ATOL_FP16


def compare_indices(golden: torch.Tensor, npu: torch.Tensor, pct_thd: float = 0.995) -> bool:
    """Compare index tensors row by row using sorted-set equality.

    Both tensors have shape (B, S1, L) or (T1, L). Invalid positions are -1.
    """
    g = golden.cpu()
    n = npu.cpu()
    if g.shape != n.shape:
        print(f"[FAIL] shape mismatch: golden {g.shape} vs npu {n.shape}")
        return False

    if g.dim() == 2:
        g = g.unsqueeze(0)
        n = n.unsqueeze(0)

    total_rows = 0
    pass_rows = 0
    for b in range(g.shape[0]):
        for s in range(g.shape[1]):
            gv = g[b, s]
            nv = n[b, s]
            g_valid = gv[gv >= 0].sort().values
            n_valid = nv[nv >= 0].sort().values
            total_rows += 1
            if g_valid.numel() == n_valid.numel() and torch.equal(g_valid, n_valid):
                pass_rows += 1
            else:
                if g_valid.numel() != n_valid.numel():
                    print(f"[FAIL] row ({b},{s}): valid count golden={g_valid.numel()} npu={n_valid.numel()}")
                else:
                    diffs = (g_valid != n_valid).nonzero(as_tuple=True)[0]
                    print(f"[FAIL] row ({b},{s}): {len(diffs)} index differences")
                    if len(diffs) > 0:
                        idx = diffs[0].item()
                        print(f"    first diff at pos {idx}: golden={g_valid[idx]} npu={n_valid[idx]}")

    ratio = pass_rows / total_rows if total_rows > 0 else 0
    if ratio >= pct_thd:
        print(f"[PASS] indices: {pass_rows}/{total_rows} rows matched ({ratio:.2%})")
        return True
    print(f"[FAIL] indices: only {pass_rows}/{total_rows} rows matched ({ratio:.2%})")
    return False


def compare_values(golden: torch.Tensor, npu: torch.Tensor, dt: torch.dtype, pct_thd: float = 0.995) -> bool:
    """Compare value tensors row by row using sorted-set closeness.

    Both tensors have shape (B, S1, L) or (T1, L). Invalid positions are -inf.
    """
    g = golden.cpu().to(torch.float32)
    n = npu.cpu().to(torch.float32)
    if g.shape != n.shape:
        print(f"[FAIL] shape mismatch: golden {g.shape} vs npu {n.shape}")
        return False

    if g.dim() == 2:
        g = g.unsqueeze(0)
        n = n.unsqueeze(0)

    atol = _atol_for_dtype(dt)
    total_rows = 0
    pass_rows = 0
    for b in range(g.shape[0]):
        for s in range(g.shape[1]):
            gv = g[b, s]
            nv = n[b, s]
            g_valid = gv[torch.isfinite(gv)].sort().values
            n_valid = nv[torch.isfinite(nv)].sort().values
            total_rows += 1
            if g_valid.numel() == n_valid.numel():
                if torch.allclose(g_valid, n_valid, rtol=RTOL, atol=atol):
                    pass_rows += 1
                else:
                    max_err = (g_valid - n_valid).abs().max().item()
                    print(f"[FAIL] row ({b},{s}): value max_err={max_err:.6f} (rtol={RTOL}, atol={atol})")
            else:
                print(f"[FAIL] row ({b},{s}): valid count golden={g_valid.numel()} npu={n_valid.numel()}")

    ratio = pass_rows / total_rows if total_rows > 0 else 0
    if ratio >= pct_thd:
        print(f"[PASS] values: {pass_rows}/{total_rows} rows matched ({ratio:.2%})")
        return True
    print(f"[FAIL] values: only {pass_rows}/{total_rows} rows matched ({ratio:.2%})")
    return False

#!/usr/bin/env python3
"""Diagnose a final Phase2 result using the tensors saved by the A2 workflow."""

import argparse
from pathlib import Path

import torch


def _metrics(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return float(diff.max()), float(diff.mean())


def _flatten_heads(value):
    # [q, kv, group, d] -> [q, q_heads, d]
    q_count, kv_heads, group_size, head_dim = value.shape
    return value.reshape(q_count, kv_heads * group_size, head_dim)


def _combine(ws_o, ws_max, ws_sum):
    valid = (ws_sum > 0.0) & (ws_max > -1.0e30)
    masked_max = torch.where(valid, ws_max, torch.full_like(ws_max, -torch.inf))
    global_max = masked_max.max(dim=2, keepdim=True).values
    exp_delta = torch.where(
        valid,
        torch.exp(masked_max - global_max),
        torch.zeros_like(masked_max),
    )
    denom = (ws_sum * exp_delta).sum(dim=2).clamp_min(torch.finfo(torch.float32).tiny)
    result = (ws_o * exp_delta.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)
    return _flatten_heads(result)


def _combine_missing_inv_mask(ws_o, ws_max, ws_sum):
    """Model the A2 bug where only lanes 0..31 multiply by 1/rowSum."""
    valid = (ws_sum > 0.0) & (ws_max > -1.0e30)
    masked_max = torch.where(valid, ws_max, torch.full_like(ws_max, -torch.inf))
    global_max = masked_max.max(dim=2, keepdim=True).values
    exp_delta = torch.where(
        valid,
        torch.exp(masked_max - global_max),
        torch.zeros_like(masked_max),
    )
    denom = (ws_sum * exp_delta).sum(dim=2).clamp_min(torch.finfo(torch.float32).tiny)
    scale = ws_sum * exp_delta / denom.unsqueeze(2)
    inv_sum = torch.where(valid, ws_sum.reciprocal(), torch.zeros_like(ws_sum))
    correct = (ws_o * (scale * inv_sum).unsqueeze(-1)).sum(dim=2)
    missing_inv = (ws_o * scale.unsqueeze(-1)).sum(dim=2)
    lane = torch.arange(ws_o.shape[-1]) % 64
    modeled = torch.where((lane < 32).view(1, 1, 1, -1), correct, missing_inv)
    return _flatten_heads(modeled)


def _simple_candidates(ws_o, ws_sum):
    valid = ws_sum > 0.0
    normalized = torch.where(
        valid.unsqueeze(-1),
        ws_o / ws_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1),
        torch.zeros_like(ws_o),
    )
    count = valid.sum(dim=2).clamp_min(1).unsqueeze(-1)
    candidates = {
        "mean_normalized": _flatten_heads(normalized.sum(dim=2) / count),
        "sum_normalized": _flatten_heads(normalized.sum(dim=2)),
    }
    for slot in range(ws_o.shape[2]):
        candidates[f"slot{slot}_normalized"] = _flatten_heads(normalized[:, :, slot])
    return candidates


def _fit_combine_weights(actual, ws_o, ws_max, ws_sum):
    valid = (ws_sum > 0.0) & (ws_max > -1.0e30)
    masked_max = torch.where(valid, ws_max, torch.full_like(ws_max, -torch.inf))
    global_max = masked_max.max(dim=2, keepdim=True).values
    exp_delta = torch.where(
        valid,
        torch.exp(masked_max - global_max),
        torch.zeros_like(masked_max),
    )
    golden_weights = ws_sum * exp_delta
    golden_weights /= golden_weights.sum(dim=2, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny)
    normalized = torch.where(
        valid.unsqueeze(-1),
        ws_o / ws_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1),
        torch.zeros_like(ws_o),
    )
    q_count, kv_heads, _, group_size, _ = ws_o.shape
    for q_idx in range(q_count):
        for kv_idx in range(kv_heads):
            for group_idx in range(group_size):
                head_idx = kv_idx * group_size + group_idx
                matrix = normalized[q_idx, kv_idx, :, group_idx].transpose(0, 1)
                target = actual[q_idx, head_idx]
                fitted = torch.linalg.lstsq(matrix, target).solution
                reconstructed = matrix @ fitted
                residual = (reconstructed - target).abs()
                fit_text = ",".join(f"{float(v):.7g}" for v in fitted)
                golden_text = ",".join(
                    f"{float(v):.7g}" for v in golden_weights[q_idx, kv_idx, :, group_idx])
                print(
                    f"[phase2-final] weight_fit q={q_idx} h={head_idx} "
                    f"fitted=[{fit_text}] golden=[{golden_text}] "
                    f"residual_max={float(residual.max()):.9g} "
                    f"residual_mean={float(residual.mean()):.9g}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    actual = torch.load(run_dir / "actual_final.pt", map_location="cpu").float()
    golden = torch.load(run_dir / "golden_final.pt", map_location="cpu").float()
    ws_o = torch.load(run_dir / "phase1_ws_o.pt", map_location="cpu").float()
    ws_max = torch.load(run_dir / "phase1_ws_max.pt", map_location="cpu").float()
    ws_sum = torch.load(run_dir / "phase1_ws_sum.pt", map_location="cpu").float()

    print(f"[phase2-final] run_dir={run_dir}")
    print(f"[phase2-final] shape={tuple(actual.shape)} atol={args.atol}")
    candidates = dict((
        ("saved_golden", golden),
        ("combine_fp32_ws", _combine(ws_o, ws_max, ws_sum)),
        ("combine_bf16_ws", _combine(ws_o.to(torch.bfloat16).float(), ws_max, ws_sum)),
        ("missing_inv_mask", _combine_missing_inv_mask(ws_o, ws_max, ws_sum)),
        ("missing_inv_mask_bf16_ws", _combine_missing_inv_mask(
            ws_o.to(torch.bfloat16).float(), ws_max, ws_sum)),
    ))
    candidates.update(_simple_candidates(ws_o, ws_sum))
    for name, expected in candidates.items():
        max_abs, mean_abs = _metrics(actual, expected)
        mismatches = int(((actual - expected).abs() > args.atol).sum())
        print(
            f"[phase2-final] candidate={name} max_abs={max_abs:.9g} "
            f"mean_abs={mean_abs:.9g} mismatches={mismatches}"
        )

    diff = (actual - golden).abs()
    lane = torch.arange(actual.shape[-1]) % 64
    for lane_name, lane_mask in (("lane_0_31", lane < 32), ("lane_32_63", lane >= 32)):
        lane_diff = diff[..., lane_mask]
        print(
            f"[phase2-final] {lane_name} max_abs={float(lane_diff.max()):.9g} "
            f"mean_abs={float(lane_diff.mean()):.9g} "
            f"mismatches={int((lane_diff > args.atol).sum())}"
        )
    actual_rows = actual.reshape(-1, actual.shape[-1])
    golden_rows = golden.reshape(-1, golden.shape[-1])
    pair_mean = (actual_rows[:, None, :] - golden_rows[None, :, :]).abs().mean(dim=-1)
    for actual_row in range(actual_rows.shape[0]):
        best_row = int(pair_mean[actual_row].argmin())
        expected_row = actual_row
        print(
            f"[phase2-final] row_map actual={actual_row} expected={expected_row} "
            f"best={best_row} expected_mean={float(pair_mean[actual_row, expected_row]):.9g} "
            f"best_mean={float(pair_mean[actual_row, best_row]):.9g}"
        )
    _fit_combine_weights(actual, ws_o, ws_max, ws_sum)
    half = actual.shape[-1] // 2
    for q_idx in range(actual.shape[0]):
        for head_idx in range(actual.shape[1]):
            row = diff[q_idx, head_idx]
            first = row[:half]
            second = row[half:]
            print(
                f"[phase2-final] row q={q_idx} h={head_idx} "
                f"first_max={float(first.max()):.9g} first_mean={float(first.mean()):.9g} "
                f"second_max={float(second.max()):.9g} second_mean={float(second.mean()):.9g}"
            )


if __name__ == "__main__":
    main()

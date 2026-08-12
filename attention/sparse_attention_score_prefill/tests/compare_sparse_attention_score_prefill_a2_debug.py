"""Compare an A2 DEBUG_MODE return tensor with CPU Phase1 golden workspace."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument(
        "--stage", required=True,
        choices=("final", "rowmax", "rowsum", "opartial"))
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument(
        "--result-file",
        default=None,
        help="JSON result path (default: <dump-dir>/compare_<stage>.json)",
    )
    return parser.parse_args()


def extract_stats(actual, golden):
    q_count, kv_heads, top_k, group_size = golden.shape
    needed = top_k * group_size
    q_heads = kv_heads * group_size
    if tuple(actual.shape[:2]) != (q_count, q_heads) or actual.shape[2] < needed:
        raise ValueError(
            f"actual shape {tuple(actual.shape)} cannot hold "
            f"[q={q_count}, q_heads={q_heads}, values={needed}]")
    result = torch.empty_like(golden)
    for q_idx in range(q_count):
        for kv_idx in range(kv_heads):
            result[q_idx, kv_idx] = actual[q_idx, kv_idx * group_size, :needed].reshape(top_k, group_size)
    return result


def extract_opartial(actual, golden):
    q_count, kv_heads, top_k, group_size, head_dim = golden.shape
    del top_k
    q_heads = kv_heads * group_size
    if tuple(actual.shape) != (q_count, q_heads, head_dim):
        raise ValueError(
            f"actual shape {tuple(actual.shape)} cannot hold "
            f"[q={q_count}, q_heads={q_heads}, d={head_dim}]")
    result = torch.empty_like(golden[:, :, 0])
    for q_idx in range(q_count):
        for kv_idx in range(kv_heads):
            head_start = kv_idx * group_size
            result[q_idx, kv_idx] = actual[q_idx, head_start:head_start + group_size]
    return result


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir).resolve()
    actual = torch.load(dump_dir / "npu_debug_output.pt", map_location="cpu").float()

    if args.stage == "final":
        golden = torch.load(dump_dir / "golden_final.pt", map_location="cpu")
        actual_view = actual
        default_atol = 0.02
    elif args.stage == "rowmax":
        golden = torch.load(dump_dir / "phase1_ws_max.pt", map_location="cpu")
        actual_view = extract_stats(actual, golden)
        default_atol = 0.5
    elif args.stage == "rowsum":
        golden = torch.load(dump_dir / "phase1_ws_sum.pt", map_location="cpu")
        actual_view = extract_stats(actual, golden)
        default_atol = 0.5
    else:
        golden = torch.load(dump_dir / "phase1_ws_o.pt", map_location="cpu")[:, :, 0]
        actual_view = extract_opartial(actual, golden.unsqueeze(2))
        default_atol = 0.02

    # DEBUG_MODE transports floats through the bf16 attention output contract.
    golden_view = golden.to(torch.bfloat16).float()
    atol = default_atol if args.atol is None else args.atol
    actual_finite = torch.isfinite(actual_view)
    golden_finite = torch.isfinite(golden_view)
    finite_pair = actual_finite & golden_finite
    diff = (actual_view - golden_view).abs()
    same_nonfinite = (~actual_finite) & (~golden_finite) & (actual_view == golden_view)
    bad = (
        actual_finite ^ golden_finite
        | (finite_pair & (diff > atol))
        | ((~actual_finite & ~golden_finite) & ~same_nonfinite)
    )
    result_file = Path(args.result_file) if args.result_file else dump_dir / f"compare_{args.stage}.json"
    actual_file = dump_dir / f"actual_{args.stage}.pt"
    golden_file = dump_dir / f"golden_{args.stage}.pt"
    torch.save(actual_view, actual_file)
    torch.save(golden_view, golden_file)

    if finite_pair.any():
        finite_diff = diff[finite_pair]
        max_abs = float(finite_diff.max().item())
        mean_abs = float(finite_diff.mean().item())
        actual_min = float(actual_view[actual_finite].min().item()) if actual_finite.any() else None
        actual_max = float(actual_view[actual_finite].max().item()) if actual_finite.any() else None
        golden_min = float(golden_view[golden_finite].min().item()) if golden_finite.any() else None
        golden_max = float(golden_view[golden_finite].max().item()) if golden_finite.any() else None
    else:
        max_abs = None
        mean_abs = None
        actual_min = None
        actual_max = None
        golden_min = None
        golden_max = None
    mismatch_count = int(bad.sum().item())
    result = {
        "stage": args.stage,
        "dump_dir": str(dump_dir),
        "actual_file": str(actual_file),
        "golden_file": str(golden_file),
        "shape": list(golden_view.shape),
        "values": int(golden_view.numel()),
        "atol": float(atol),
        "actual_range": [actual_min, actual_max],
        "golden_range": [golden_min, golden_max],
        "nonfinite_actual": int((~actual_finite).sum().item()),
        "nonfinite_golden": int((~golden_finite).sum().item()),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mismatches": mismatch_count,
        "verdict": "PASS" if mismatch_count == 0 else "FAIL",
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[a2-compare] stage={args.stage}")
    print(f"[a2-compare] dump_dir={dump_dir}")
    print(f"[a2-compare] actual_file={actual_file}")
    print(f"[a2-compare] golden_file={golden_file}")
    print(f"[a2-compare] result_file={result_file}")
    print(f"[a2-compare] shape={tuple(golden_view.shape)} values={golden_view.numel()}")
    print(
        f"[a2-compare] actual_range=({actual_min}, {actual_max})"
        f" golden_range=({golden_min}, {golden_max})")
    print(f"[a2-compare] nonfinite_actual={(~actual_finite).sum().item()}")
    print(f"[a2-compare] nonfinite_golden={(~golden_finite).sum().item()}")
    if (~actual_finite).any():
        nonfinite_idx = (~actual_finite).nonzero(as_tuple=False)[0]
        item = tuple(int(v) for v in nonfinite_idx.tolist())
        print(
            f"[a2-compare] first_nonfinite={item} actual={actual_view[item].item():.9g} "
            f"golden={golden_view[item].item():.9g}")
    print(f"[a2-compare] max_abs={max_abs}")
    print(f"[a2-compare] mean_abs={mean_abs}")
    print(f"[a2-compare] atol={atol:.9g} mismatches={mismatch_count}")
    if bad.any():
        idx = tuple(int(v) for v in bad.nonzero(as_tuple=False)[0].tolist())
        print(
            f"[a2-compare] first_mismatch={idx} actual={actual_view[idx].item():.9g} "
            f"golden={golden_view[idx].item():.9g} diff={diff[idx].item():.9g}")
        print("[a2-compare] result=FAIL")
        raise SystemExit(1)
    print("[a2-compare] result=PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Element-wise comparison between A2 ``desc=940`` and golden P tensors.

The A2 kernel emits one 4 x 128 BF16 P tile per eligible vector sub-block.
Golden tensors are saved separately for each (query token, KV head, top-k
slot), with causal rows shorter than 128.  This tool pads those golden rows
with zeros and identifies the golden tile corresponding to each device dump
tile before writing an auditable element-wise report.
"""

import argparse
import json
from pathlib import Path

import torch


TILE_ROWS = 4
TILE_COLS = 128
TILE_VALUES = TILE_ROWS * TILE_COLS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--dump-file", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--report-file", default=None)
    parser.add_argument("--result-file", default=None)
    return parser.parse_args()


def load_dump_values(path):
    raw = path.read_text(encoding="utf-8").replace(",", " ").split()
    try:
        return torch.tensor([float(value) for value in raw], dtype=torch.float32)
    except ValueError as exc:
        raise ValueError(f"invalid numeric payload in {path}") from exc


def load_golden_tiles(dump_dir):
    candidates = []
    for path in sorted(dump_dir.glob("softmax_P_q*_h*_t*.pt")):
        golden = torch.load(path, map_location="cpu", weights_only=True).float()
        if golden.ndim != 2 or golden.shape[0] != TILE_ROWS or golden.shape[1] > TILE_COLS:
            continue
        padded = torch.zeros((TILE_ROWS, TILE_COLS), dtype=torch.float32)
        padded[:, : golden.shape[1]] = golden
        candidates.append({
            "name": path.stem.removeprefix("softmax_P_"),
            "path": str(path),
            "valid_cols": int(golden.shape[1]),
            "values": padded.flatten(),
        })
    if not candidates:
        raise FileNotFoundError(f"no softmax_P_q*_h*_t*.pt under {dump_dir}")
    return candidates


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir).resolve()
    dump_file = Path(args.dump_file).resolve()
    report_file = Path(args.report_file) if args.report_file else dump_dir / "p_elementwise_compare.txt"
    result_file = Path(args.result_file) if args.result_file else dump_dir / "compare_p_dump.json"
    actual = load_dump_values(dump_file)
    if actual.numel() == 0 or actual.numel() % TILE_VALUES:
        raise ValueError(
            f"desc=940 has {actual.numel()} values; expected a positive multiple of {TILE_VALUES}")

    candidates = load_golden_tiles(dump_dir)
    available = list(candidates)
    report = []
    matches = []
    all_diff = []
    for tile_index, offset in enumerate(range(0, actual.numel(), TILE_VALUES)):
        device_tile = actual[offset:offset + TILE_VALUES]
        scored = []
        for candidate in available:
            diff = (device_tile - candidate["values"]).abs()
            scored.append((float(diff.mean().item()), float(diff.max().item()), candidate, diff))
        _, _, chosen, diff = min(scored, key=lambda item: (item[0], item[1], item[2]["name"]))
        available.remove(chosen)
        all_diff.append(diff)
        mismatch = diff > args.atol
        tile_result = {
            "tile": tile_index,
            "dump_offset": offset,
            "golden": chosen["name"],
            "golden_file": chosen["path"],
            "golden_shape": [TILE_ROWS, chosen["valid_cols"]],
            "values": TILE_VALUES,
            "mismatches": int(mismatch.sum().item()),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }
        matches.append(tile_result)
        report.append(
            f"[{chosen['name']}] tile={tile_index} dump_offset={offset} "
            f"golden_shape=({TILE_ROWS}, {chosen['valid_cols']})")
        for row in range(TILE_ROWS):
            row_bad = 0
            for col in range(TILE_COLS):
                index = offset + row * TILE_COLS + col
                device_value = float(device_tile[row * TILE_COLS + col].item())
                golden_value = float(chosen["values"][row * TILE_COLS + col].item())
                delta = device_value - golden_value
                bad = abs(delta) > args.atol
                row_bad += int(bad)
                report.append(
                    f"{chosen['name']} idx={index:4d} row={row} col={col:3d} "
                    f"dump={device_value:.9f} golden={golden_value:.9f} "
                    f"diff={delta:+.9f} {'MISMATCH' if bad else 'OK'}")
            report.append(f"{chosen['name']} row={row} mismatches={row_bad}")

    diff = torch.cat(all_diff)
    mismatch_count = int((diff > args.atol).sum().item())
    result = {
        "dump_dir": str(dump_dir),
        "dump_file": str(dump_file),
        "report_file": str(report_file),
        "atol": args.atol,
        "values": int(diff.numel()),
        "mismatches": mismatch_count,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "tiles": matches,
        "verdict": "PASS" if mismatch_count == 0 else "FAIL",
    }
    report.insert(
        0,
        f"total={result['values']} mismatches={mismatch_count} "
        f"exact_tol={args.atol:g} max_abs={result['max_abs']:.9f} "
        f"mean_abs={result['mean_abs']:.9f}")
    report_file.write_text("\n".join(report) + "\n", encoding="utf-8")
    result_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[a2-p-compare] dump_dir={dump_dir}")
    print(f"[a2-p-compare] dump_file={dump_file}")
    print(f"[a2-p-compare] report_file={report_file}")
    print(f"[a2-p-compare] result_file={result_file}")
    for tile in matches:
        print(
            f"[a2-p-compare] tile={tile['tile']} offset={tile['dump_offset']} "
            f"golden={tile['golden']} mismatches={tile['mismatches']} "
            f"max_abs={tile['max_abs']:.9f}")
    print(
        f"[a2-p-compare] values={result['values']} atol={args.atol:g} "
        f"mismatches={mismatch_count} max_abs={result['max_abs']:.9f} "
        f"mean_abs={result['mean_abs']:.9f} result={result['verdict']}")
    raise SystemExit(0 if mismatch_count == 0 else 1)


if __name__ == "__main__":
    main()

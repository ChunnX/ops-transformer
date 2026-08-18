"""Compare PV ``desc=981`` O_partial dumps with the Phase 1 golden workspace.

``desc=981`` is emitted immediately after the PV scatter.  In normal
``DEBUG_MODE=0`` runs the NPU output is the final Phase 2 attention result,
not raw O_partial, and must not be compared directly with this dump.
"""

import argparse
import re
from pathlib import Path

import torch


def _records(run_dir):
    log_lines = (run_dir / "kernel.log").read_text(encoding="utf-8", errors="replace").splitlines()
    offsets_by_block = {}
    records = []
    addresses = []
    for line in log_lines:
        block_match = re.search(r"\[AIC Block (\d+)(?:/\d+)?\]", line)
        block = int(block_match.group(1)) if block_match else None
        offset_match = re.search(r"wsOOff\s*=\s*(\d+)", line)
        if block is not None and offset_match:
            offsets_by_block[block] = int(offset_match.group(1))
        match = re.search(r"DumpTensor: desc=981, addr=0x([0-9a-fA-F]+)", line)
        if match:
            addresses.append(int(match.group(1), 16))
            records.append((block, offsets_by_block.get(block)))
    dump_files = sorted(run_dir.glob("dump_desc981_*.txt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if len(addresses) != len(dump_files):
        raise RuntimeError(f"kernel.log has {len(addresses)} desc=981 headers for {len(dump_files)} dumps")
    if not addresses:
        return []
    # Core output can interleave, but each line includes its AIC block id.
    # Preserve a last offset per block rather than globally pairing a dump
    # with another core's wsOOff line.
    if all(offset is not None for _, offset in records):
        return [(offset, dump_file) for (_, offset), dump_file in zip(records, dump_files)]
    # Older logs lack a matching per-core wsOOff marker. Their lowest dump is
    # the base O_partial slot in the debug cases, so retain this fallback.
    workspace_base = min(addresses)
    return [((address - workspace_base) // 4, dump_file)
            for address, dump_file in zip(addresses, dump_files)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    group_size = args.q_heads // args.kv_heads
    slot_elems = group_size * args.head_dim
    per_q_stride = args.kv_heads * args.top_k * slot_elems
    records = _records(run_dir)
    golden_checks = []
    golden = torch.load(run_dir / "phase1_ws_o.pt", map_location="cpu").float()
    for offset, dump_file in records:
        values = [float(value) for value in dump_file.read_text(encoding="utf-8").split(",") if value.strip()]
        dump = torch.tensor(values, dtype=torch.float32).reshape(group_size, args.head_dim)
        q = offset // per_q_stride
        remainder = offset % per_q_stride
        kv = remainder // (args.top_k * slot_elems)
        slot = (remainder % (args.top_k * slot_elems)) // slot_elems
        if q >= golden.shape[0] or kv >= args.kv_heads or slot >= args.top_k:
            continue
        golden_slot = golden[q, kv, slot]
        delta_golden = (dump - golden_slot).abs()
        golden_checks.append((delta_golden, q, kv, slot, dump_file))

    def summary(items):
        if not items:
            return {"records": 0}
        all_delta = torch.cat([delta.reshape(-1) for delta, *_ in items])
        finite = torch.isfinite(all_delta)
        return {
            "records": len(items),
            "values": int(all_delta.numel()),
            "nonfinite": int((~finite).sum()),
            "max_abs": float(all_delta[finite].max()) if finite.any() else None,
            "mean_abs": float(all_delta[finite].mean()) if finite.any() else None,
        }

    print(f"[phase2-read] run_dir={run_dir}")
    print(f"[phase2-read] kernel_log={run_dir / 'kernel.log'}")
    print(f"[phase2-read] desc981_records={len(records)}")
    print(f"[phase2-read] desc981_vs_phase1_golden={summary(golden_checks)}")
    for delta, q, kv, slot, dump_file in golden_checks:
        if not torch.isfinite(delta).all() or float(delta.max()) > 0.02:
            print(f"[phase2-read] pv_mismatch q={q} kv={kv} slot={slot} file={dump_file} max_abs={float(delta.nan_to_num().max())}")
            break


if __name__ == "__main__":
    main()

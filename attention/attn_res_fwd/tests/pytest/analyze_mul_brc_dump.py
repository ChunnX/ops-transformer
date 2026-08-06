#!/usr/bin/env python3
"""Parse MulRowByBrcBlock SetMaskCount DumpTensor logs and compare with golden broadcast mul."""
from __future__ import annotations

import argparse
import io
import re
import sys
from typing import Dict, List, Tuple

import torch

SCRIPT_DIR = __file__.rsplit("/", 1)[0]
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import attn_res_fwd_golden as golden  # noqa: E402

DUMP_RE = re.compile(
    r"DumpTensor: desc=(\d+),.*?dump_size=(\d+)\.\s*\n.*?DumpTensor: \[(.*?)\]",
    re.DOTALL,
)
PRINTF_RE = re.compile(
    r"\[MulBrc\] meta=(\d+) chunk=(\d+) start=(\d+) blocks=(\d+) maskCount=(\d+) phase=(\w+)"
)

# desc = 2700 + chunk*10 + slot
DUMP_MUL_BRC_BASE = 2700
SLOT_SRC_HEAD = 0
SLOT_BRC = 1
SLOT_DST_HEAD = 2
SLOT_SRC_MID = 3
SLOT_DST_MID = 4


def parse_dumps(text: str) -> Dict[int, Dict[int, List[float]]]:
    out: Dict[int, Dict[int, List[float]]] = {}
    for m in DUMP_RE.finditer(text):
        desc = int(m.group(1))
        if desc < DUMP_MUL_BRC_BASE or desc >= DUMP_MUL_BRC_BASE + 100:
            continue
        rel = desc - DUMP_MUL_BRC_BASE
        chunk = rel // 10
        slot = rel % 10
        vals = [float(x.strip()) for x in m.group(3).split(",") if x.strip()]
        out.setdefault(chunk, {})[slot] = vals
    return out


def parse_printfs(text: str) -> List[dict]:
    rows = []
    for m in PRINTF_RE.finditer(text):
        rows.append(
            {
                "meta": int(m.group(1)),
                "chunk": int(m.group(2)),
                "start": int(m.group(3)),
                "blocks": int(m.group(4)),
                "maskCount": int(m.group(5)),
                "phase": m.group(6),
            }
        )
    return rows


def golden_brc_mul(src_head: List[float], brc: List[float]) -> List[float]:
    scale = brc[0] if brc else 0.0
    return [x * scale for x in src_head]


def analyze_log(log_path: str, meta_filter: int | None = 0) -> int:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = parse_dumps(text)
    infos = [r for r in parse_printfs(text) if meta_filter is None or r["meta"] == meta_filter]
    post = [r for r in infos if r["phase"] == "post"]

    print(f"log={log_path} mul_brc_chunks={len(chunks)} printf_post={len(post)}")
    if not chunks:
        print("WARNING: no MulBrc DumpTensor found (desc 2700-2799)")
        return 1

    worst_mere = 0.0
    for info in sorted(post, key=lambda r: r["chunk"]):
        cid = info["chunk"]
        if cid not in chunks:
            continue
        c = chunks[cid]
        src = c.get(SLOT_SRC_HEAD, [])
        brc = c.get(SLOT_BRC, [])
        dst = c.get(SLOT_DST_HEAD, [])
        if not src or not brc or not dst:
            print(f"chunk {cid}: missing slots src={len(src)} brc={len(brc)} dst={len(dst)}")
            continue
        expect = golden_brc_mul(src, brc)
        n = min(len(expect), len(dst))
        rel = []
        for i in range(n):
            denom = max(abs(expect[i]), 1e-10)
            rel.append(abs(dst[i] - expect[i]) / denom)
        chunk_mere = max(rel) if rel else 0.0
        worst_mere = max(worst_mere, chunk_mere)
        scale = brc[0]
        print(
            f"\n--- chunk={cid} meta={info['meta']} start={info['start']} "
            f"blocks={info['blocks']} maskCount={info['maskCount']} scale={scale:.6f} ---"
        )
        print(f"  brc[0:8]     = {brc[:8]}")
        print(f"  src[0:8]     = {src[:8]}")
        print(f"  dst[0:8]     = {dst[:8]}")
        print(f"  expect[0:8]  = {expect[:8]}")
        print(f"  chunk MERE   = {chunk_mere:.6f}")
        if chunk_mere > 1e-3:
            bad = sorted(range(n), key=lambda i: rel[i], reverse=True)[:5]
            for i in bad:
                print(
                    f"    idx={i} src={src[i]:.6f} dst={dst[i]:.6f} "
                    f"expect={expect[i]:.6f} rel={rel[i]:.6f}"
                )
        if SLOT_SRC_MID in c and SLOT_DST_MID in c:
            src_m = c[SLOT_SRC_MID]
            dst_m = c[SLOT_DST_MID]
            exp_m = golden_brc_mul(src_m, brc)
            nm = min(len(exp_m), len(dst_m))
            rel_m = [abs(dst_m[i] - exp_m[i]) / max(abs(exp_m[i]), 1e-10) for i in range(nm)]
            print(f"  mid MERE     = {max(rel_m) if rel_m else 0:.6f} (offset +1024 within chunk)")

    print(f"\n=== worst chunk MERE = {worst_mere:.6f} ===")
    if worst_mere > 1e-2:
        print("DIAGNOSIS: SetMaskCount Mul dst != src * brc[0]; likely repeat/mask stride mismatch.")
        print("  for-loop uses mask=64 repeat=1 dstRepStride=8 (64 elems/step)")
        print("  count-mask uses repeat=chunk dstRepStride=1 (8 elems/step) with total maskCount elems")
    return 0 if worst_mere <= 1e-2 else 1


def run_and_capture(device: int = 0) -> str:
    import contextlib
    import os

    import torch_npu  # noqa: F401
    import attn_res_fwd_torch_runner as torch_runner
    from npu_device import resolve_device_id

    dev = resolve_device_id() if device < 0 else device
    torch.npu.set_device(dev)
    attn_res_fwd = torch_runner.get_attn_res_fwd_fn()
    T, N, H, seed = 1, 1, 7168, 0
    ps, br, pw, nw, eps = golden.prepare_tensors(T, N, H, seed=seed, device="cpu")
    d = f"npu:{dev}"
    ps, br, pw, nw = ps.to(d), br.to(d), pw.to(d), nw.to(d)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        attn_res_fwd(ps, br, pw, nw, eps)
        torch.npu.synchronize()
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="parse existing log file")
    ap.add_argument("--run", action="store_true", help="run t1_n1 and capture log")
    ap.add_argument("--meta", type=int, default=0)
    ap.add_argument("--device", type=int, default=-1)
    args = ap.parse_args()

    if args.run:
        log_text = run_and_capture(args.device)
        out = "/tmp/mul_brc_dump_t1_n1.log"
        with open(out, "w", encoding="utf-8") as f:
            f.write(log_text)
        print(f"captured -> {out}")
        return analyze_log(out, args.meta)
    if not args.log:
        ap.error("provide --log or --run")
    return analyze_log(args.log, args.meta)


if __name__ == "__main__":
    sys.exit(main())

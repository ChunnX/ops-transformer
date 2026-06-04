"""Performance runner: build sweep script, run msprof, return op_summary.csv path."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def build_perf_script(cases: List[Dict], runs: int, cold_thr: int,
                      device_id: int = 0) -> str:
    """Build a single sweep script with hot/cold separation.
    - hot (S1 > cold_thr): generate data once, reuse for `runs` iterations
    - cold (S1 <= cold_thr): regenerate data each iteration + flush HBM
    Each case is wrapped in try/except for crash isolation.
    """
    hot = [c for c in cases if c.get("S1", 9999) > cold_thr]
    cold = [c for c in cases if c.get("S1", 9999) <= cold_thr]

    lines = [
        "import sys, traceback",
        "import torch",
        "import torch_npu",
        f"torch.npu.set_device({device_id})",
        "from npu_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata",
        "device = torch.device('npu:0')",
        f"print(f'PERF total={len(cases)} hot={len(hot)} cold={len(cold)} runs={runs}')",
        "",
        "def _to_int32(v):",
        "    if v is None: return None",
        "    return torch.tensor(v, dtype=torch.int32, device=device)",
        "",
    ]

    for i, c in enumerate(hot):
        lines.extend(_gen_hot(c, i, runs, len(hot)))
    for i, c in enumerate(cold):
        lines.extend(_gen_cold(c, i, runs, len(cold)))

    lines.append("print('PERF_DONE')")
    return "\n".join(lines)


def _pt_dtype(ds: str) -> str:
    return "torch.float16" if ds == "fp16" else "torch.bfloat16"


def _extract(c: Dict):
    b = c.get("B", 1)
    n1 = c["N1"]
    n2 = c.get("N2", n1)
    s1 = c.get("S1", 1)
    s2 = c.get("S2", s1)
    d = c["D"]
    lq = c.get("layout_q", c.get("input_layout", "BNSD"))
    lkv = c.get("layout_kv", lq)
    lo = c.get("layout_out", lq)
    mm = c.get("mask_mode", 0)
    if mm in (0, 3):
        wl, wr = -1, -1
    else:
        wl = c.get("win_left", 2147483647)
        wr = c.get("win_right", 2147483647)
    dt = _pt_dtype(c.get("Dtype", "fp16"))
    block_size = c.get("block_size", 128)
    seqused_kv = c.get("seqused_kv", [s2])
    block_table_shape = c.get("block_table_shape", [])
    return b, n1, n2, s1, s2, d, lq, lkv, lo, mm, wl, wr, dt, block_size, seqused_kv, block_table_shape


def _q_shape(b, n1, s1, d, lq):
    if lq == "BNSD":
        return f"({b}, {n1}, {s1}, {d}"
    elif lq == "BSND":
        return f"({b}, {s1}, {n1}, {d}"
    elif lq == "TND":
        return f"({s1}, {n1}, {d}"
    else:  # BSH
        return f"({b}, {s1}, {n1} * {d}"


def _kv_shape(b, n2, s2, d, lkv):
    if lkv in ("BNSD", "PA_BBND", "PA_BNBD", "PA_NZ"):
        return f"({b}, {n2}, {s2}, {d}"
    elif lkv == "BSND":
        return f"({b}, {s2}, {n2}, {d}"
    elif lkv == "TND":
        return f"({s2}, {n2}, {d}"
    else:  # BSH
        return f"({b}, {s2}, {n2} * {d}"


def _meta_call_lines(n1, n2, d, b, s1, s2, mm, wl, wr, lq, lkv, lo,
                     is_tnd, have_pa, seqused_kv, block_size) -> List[str]:
    """Generate metadata call lines."""
    lines = []
    if is_tnd:
        # For TND, need cu_seqlens
        lines.append(f"cu_q = torch.tensor([0, {s1}], dtype=torch.int32, device=device)")
        lines.append(f"seq_q = torch.tensor([{s1}], dtype=torch.int32, device=device)")
        lines.append("cu_q_meta = cu_q")
        lines.append("seq_q_meta = seq_q")
        if have_pa:
            lines.append("cu_kv_meta = None")
            lines.append(f"seq_kv_meta = _to_int32({seqused_kv})")
        else:
            lines.append(f"cu_kv = torch.tensor([0, {s2}], dtype=torch.int32, device=device)")
            lines.append("cu_kv_meta = cu_kv")
            lines.append(f"seq_kv = torch.tensor([{s2}], dtype=torch.int32, device=device)")
            lines.append("seq_kv_meta = seq_kv")
        lines.append(f"batch_size = 1")
    else:
        lines.append("cu_q_meta = None")
        lines.append("seq_q_meta = None")
        if have_pa:
            lines.append("cu_kv_meta = None")
            lines.append(f"seq_kv_meta = _to_int32({seqused_kv})")
        else:
            lines.append("cu_kv_meta = None")
            lines.append("seq_kv_meta = None")
        lines.append(f"batch_size = {b}")

    lines.extend([
        "metadata = npu_flash_attn_metadata(",
        f"    cu_seqlens_q=cu_q_meta, cu_seqlens_kv=cu_kv_meta,",
        f"    seqused_q=seq_q_meta, seqused_kv=seq_kv_meta,",
        f"    num_heads_q={n1}, num_heads_kv={n2}, head_dim={d},",
        f"    batch_size=batch_size, max_seqlen_q={s1}, max_seqlen_kv={s2},",
        f"    mask_mode={mm}, win_left={wl}, win_right={wr},",
        f"    layout_q='{lq}', layout_kv='{lkv}', layout_out='{lo}')",
    ])
    return lines


def _gen_block_table_lines(b, seqused_kv, block_size, block_table_shape) -> List[str]:
    """Generate block_table creation lines for PA layout.
    Uses literal integer values, not variable references.
    """
    if block_table_shape:
        bt_b = block_table_shape[0]
        max_blk = block_table_shape[1]
    else:
        bt_b = b
        kv_max = max(seqused_kv) if isinstance(seqused_kv, list) else seqused_kv
        max_blk = (kv_max + block_size - 1) // block_size
    lines = [
        f"_bt = torch.full(({bt_b}, {max_blk}), -1, dtype=torch.int32, device=device)",
        "_bidx = 0",
    ]
    if isinstance(seqused_kv, list):
        kv_repr = repr(seqused_kv)
        lines.append(f"_skv = {kv_repr}")
        lines.append(f"for _bi in range({bt_b}):")
        lines.append("    _seq = _skv[_bi] if _bi < len(_skv) else _skv[0]")
    else:
        lines.append(f"for _bi in range({bt_b}):")
        lines.append(f"    _seq = {seqused_kv}")
    lines.extend([
        f"    _bn = (_seq + {block_size} - 1) // {block_size}",
        "    for _bj in range(_bn):",
        "        _bt[_bi][_bj] = _bidx",
        "        _bidx += 1",
    ])
    return lines


def _kernel_call(d, mm, lq, lkv, lo) -> List[str]:
    return [
        f"    npu_flash_attn(q, k, v, block_table=block_table,",
        f"        attn_mask=_m, metadata=metadata,",
        f"        cu_seqlens_q=cu_q_meta, cu_seqlens_kv=cu_kv_meta,",
        f"        seqused_q=seq_q_meta, seqused_kv=seq_kv_meta,",
        f"        softmax_scale=1/({d}**0.5), mask_mode={mm},",
        f"        win_left=-1, win_right=-1,",
        f"        max_seqlen_q=-1, max_seqlen_kv=-1,",
        f"        layout_q='{lq}', layout_kv='{lkv}', layout_out='{lo}',",
        "        return_softmax_lse=0)",
    ]


def _gen_hot(c, idx, runs, total) -> List[str]:
    b, n1, n2, s1, s2, d, lq, lkv, lo, mm, wl, wr, dt, block_size, seqused_kv, bt_shape = _extract(c)
    is_tnd = (lq == "TND")
    have_pa = lkv in ("PA_BBND", "PA_BNBD", "PA_NZ")
    case_tag = f"hot{idx}"

    lines = [
        f"# hot {idx}: B={b} N1={n1} N2={n2} S1={s1} S2={s2} D={d} {lq}/{lkv}",
        f"if {mm} != 0:",
        f"    _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)",
        f"else:",
        f"    _m = None",
        f"q = torch.randn{_q_shape(b, n1, s1, d, lq)}, dtype={dt}, device=device)",
        f"k = torch.randn{_kv_shape(b, n2, s2, d, lkv)}, dtype={dt}, device=device)",
        f"v = torch.randn{_kv_shape(b, n2, s2, d, lkv)}, dtype={dt}, device=device)",
    ]
    if have_pa:
        lines.extend(_gen_block_table_lines(b, seqused_kv, block_size, bt_shape))
        lines.append("block_table = _bt")
    else:
        lines.append("block_table = None")

    lines.extend(_meta_call_lines(n1, n2, d, b, s1, s2, mm, wl, wr, lq, lkv, lo,
                                   is_tnd, have_pa, seqused_kv, block_size))
    lines.append("torch.npu.synchronize()")

    # Per-case crash isolation
    lines.append(f"print('[CASE_START] {case_tag}')")
    lines.append("try:")
    lines.append(f"    for _ in range({runs}):")
    lines.append("        torch.npu.synchronize()")
    # Need to re-expand _kernel_call inline with the right indentation
    lines.append(f"        npu_flash_attn(q, k, v, block_table=block_table,")
    lines.append(f"            attn_mask=_m, metadata=metadata,")
    lines.append(f"            cu_seqlens_q=cu_q_meta, cu_seqlens_kv=cu_kv_meta,")
    lines.append(f"            seqused_q=seq_q_meta, seqused_kv=seq_kv_meta,")
    lines.append(f"            softmax_scale=1/({d}**0.5), mask_mode={mm},")
    lines.append(f"            win_left={wl}, win_right={wr},")
    lines.append(f"            max_seqlen_q={s1}, max_seqlen_kv={s2},")
    lines.append(f"            layout_q='{lq}', layout_kv='{lkv}', layout_out='{lo}',")
    lines.append(f"            return_softmax_lse=0)")
    lines.append("    torch.npu.synchronize()")
    lines.append(f"    print('[CASE_END] {case_tag}  OK')")
    lines.append("except Exception as e:")
    lines.append(f"    print(f'[CASE_END] {case_tag}  CRASH  B={{b}} N1={{n1}} N2={{n2}} S1={{s1}} S2={{s2}} D={{d}}  {{e}}')")
    lines.append("    torch.npu.synchronize()")
    lines.append("    torch.npu.empty_cache()")
    lines.append(f"print('  [hot] {idx+1}/{total}')")
    lines.append("")
    return lines


def _gen_cold(c, idx, runs, total) -> List[str]:
    b, n1, n2, s1, s2, d, lq, lkv, lo, mm, wl, wr, dt, block_size, seqused_kv, bt_shape = _extract(c)
    is_tnd = (lq == "TND")
    have_pa = lkv in ("PA_BBND", "PA_BNBD", "PA_NZ")
    case_tag = f"cold{idx}"

    lines = [
        f"# cold {idx}: B={b} N1={n1} N2={n2} S1={s1} S2={s2} D={d} {lq}/{lkv}",
        f"if {mm} != 0:",
        f"    _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)",
        f"else:",
        f"    _m = None",
    ]
    lines.extend(_meta_call_lines(n1, n2, d, b, s1, s2, mm, wl, wr, lq, lkv, lo,
                                   is_tnd, have_pa, seqused_kv, block_size))
    lines.append("torch.npu.synchronize()")

    # Per-case crash isolation (wrap entire loop, not individual iterations)
    lines.append(f"print('[CASE_START] {case_tag}')")
    lines.append("try:")
    lines.append(f"    for _ in range({runs}):")
    lines.append(f"        q = torch.randn{_q_shape(b, n1, s1, d, lq)}, dtype={dt}, device=device)")
    lines.append(f"        k = torch.randn{_kv_shape(b, n2, s2, d, lkv)}, dtype={dt}, device=device)")
    lines.append(f"        v = torch.randn{_kv_shape(b, n2, s2, d, lkv)}, dtype={dt}, device=device)")
    if have_pa:
        lines.append("        # rebuild block_table per cold iteration")
        lines.extend(["        " + ln for ln in _gen_block_table_lines(b, seqused_kv, block_size, bt_shape)])
        lines.append("        bt = _bt")
    else:
        lines.append("        bt = None")
    # Flush HBM
    lines.append(f"        _f1 = torch.randn(32, 32768, 1, 128, dtype={dt}, device=device)")
    lines.append(f"        _f2 = torch.randn(32, 32768, 1, 128, dtype={dt}, device=device)")
    lines.append("        torch.npu.synchronize()")
    lines.append(f"        npu_flash_attn(q, k, v, block_table=bt,")
    lines.append(f"            attn_mask=_m, metadata=metadata,")
    lines.append(f"            cu_seqlens_q=cu_q_meta, cu_seqlens_kv=cu_kv_meta,")
    lines.append(f"            seqused_q=seq_q_meta, seqused_kv=seq_kv_meta,")
    lines.append(f"            softmax_scale=1/({d}**0.5), mask_mode={mm},")
    lines.append(f"            win_left={wl}, win_right={wr},")
    lines.append(f"            max_seqlen_q={s1}, max_seqlen_kv={s2},")
    lines.append(f"            layout_q='{lq}', layout_kv='{lkv}', layout_out='{lo}',")
    lines.append(f"            return_softmax_lse=0)")
    lines.append("    torch.npu.synchronize()")
    lines.append("    torch.npu.empty_cache()")
    lines.append(f"    print('[CASE_END] {case_tag}  OK')")
    lines.append("except Exception as e:")
    lines.append(f"    print(f'[CASE_END] {case_tag}  CRASH  B={{b}} N1={{n1}} N2={{n2}} S1={{s1}} S2={{s2}} D={{d}}  {{e}}')")
    lines.append("    torch.npu.synchronize()")
    lines.append("    torch.npu.empty_cache()")
    lines.append(f"print('  [cold] {idx+1}/{total}')")
    lines.append("")
    return lines


def run_msprof(script: str, output_dir: Path) -> Optional[Path]:
    """Run msprof, return path to op_summary.csv or None on failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "_perf_sweep.py"
    script_path.write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = ["msprof", f"--output={msprof_dir}", "--aic-mode=task-based",
           sys.executable, str(script_path)]
    print(f"[msprof] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"[msprof stderr]\n{result.stderr.strip()}")
    if result.returncode != 0:
        print(f"[msprof] non-zero exit code: {result.returncode}")

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        print("[msprof] no PROF_* directory found (entire sweep may have crashed)")
        return None

    csv_files = sorted(prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        print("[msprof] no op_summary CSV found")
        return None

    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] op_summary -> {dst}")
    return dst

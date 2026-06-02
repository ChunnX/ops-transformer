"""
Parse every OPPROF_* directory under a device-test run and emit:

  <out>/<tag>.log   — full, human-readable dump of visualize_data.bin + CSVs.
  <out>/STATS.txt   — compact aggregate: header + geomean-per-shape + one
                      row per test with cos / PASS / Task Duration / per-pipe
                      ratio and wait ratio / L2 hit / iCache miss.

visualize_data.bin is the primary source (5 concatenated JSON blobs separated
by a binary prefix ending in `ZZ`). CSVs supply time/ratio and wait ratios.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# =============================================================================
# visualize_data.bin — return list of parsed JSON blobs
# =============================================================================
def parse_visualize_data(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    blobs = []
    dec = json.JSONDecoder()
    i = 0
    while i < len(data):
        idx = data.find(b"ZZ{", i)
        if idx == -1:
            break
        start = idx + 2
        try:
            text = data[start:].decode("utf-8", errors="replace")
            obj, end = dec.raw_decode(text)
            blobs.append(obj)
            i = start + len(text[:end].encode("utf-8"))
        except json.JSONDecodeError:
            i = idx + 3
    return blobs


# =============================================================================
# CSV parsing
# =============================================================================
def _row_to_dict(hdr, row):
    out = {}
    for h, v in zip(hdr, row):
        h = h.strip()
        v = v.strip()
        if not h:
            continue
        try:
            out[h] = float(v)
        except ValueError:
            out[h] = v
    return out


def parse_single_row_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr, None)
        row = next(rdr, None)
        return _row_to_dict(hdr, row) if hdr and row else {}


def parse_multi_row_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr, None)
        if not hdr:
            return rows
        for row in rdr:
            if not row or all(not c.strip() for c in row):
                continue
            rows.append(_row_to_dict(hdr, row))
    return rows


def pick_matmul_row(rows):
    if not rows:
        return {}
    for r in rows:
        if str(r.get("Op Name", "")).startswith("QuantGroupedMatmulDequant"):
            return r
    return rows[0]


def parse_opprof(opprof_dir):
    return {
        "opprof_dir": opprof_dir,
        "blobs":      parse_visualize_data(os.path.join(opprof_dir, "visualize_data.bin")),
        "op_rows":    parse_multi_row_csv(os.path.join(opprof_dir, "OpBasicInfo.csv")),
        "pipe":       parse_single_row_csv(os.path.join(opprof_dir, "PipeUtilization.csv")),
        "mem":        parse_single_row_csv(os.path.join(opprof_dir, "Memory.csv")),
        "memL0":      parse_single_row_csv(os.path.join(opprof_dir, "MemoryL0.csv")),
        "memUB":      parse_single_row_csv(os.path.join(opprof_dir, "MemoryUB.csv")),
        "l2":         parse_single_row_csv(os.path.join(opprof_dir, "L2Cache.csv")),
        "conflict":   parse_single_row_csv(os.path.join(opprof_dir, "ResourceConflictRatio.csv")),
        "arith":      parse_single_row_csv(os.path.join(opprof_dir, "ArithmeticUtilization.csv")),
    }


# =============================================================================
# Small helpers
# =============================================================================
def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _asint(v):
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        try:
            return int(float(str(v).strip()))
        except (ValueError, TypeError):
            return None


def _fmt_int(v):
    i = _asint(v)
    return f"{i:,}" if isinstance(i, int) else "-"


def _pad_table(rows, headers=None, align="right"):
    """Render rows (list of lists) as a padded column table. All cells stringified."""
    table = []
    if headers:
        table.append(list(headers))
    for r in rows:
        table.append(list(r))
    if not table:
        return []
    ncols = max(len(r) for r in table)
    table = [r + [""] * (ncols - len(r)) for r in table]
    widths = [0] * ncols
    for r in table:
        for i, c in enumerate(r):
            s = str(c)
            if len(s) > widths[i]:
                widths[i] = len(s)
    out = []
    for r in table:
        cells = []
        for i, c in enumerate(r):
            s = str(c)
            cells.append(s.rjust(widths[i]) if align == "right" else s.ljust(widths[i]))
        out.append("  " + "  ".join(cells))
    return out


def _pipe_rows_from_visualize(blobs):
    """blob[4].table_per_block[0].table_detail[name=='Pipe'].row → {name: [vals]}."""
    for b in blobs:
        for tb in (b.get("table_per_block", []) or []):
            for td in tb.get("table_detail", []):
                if td.get("table_name") == "Pipe":
                    rows = {}
                    for r in td.get("row", []):
                        rows[r.get("name", "?")] = r.get("value", [])
                    return rows
    return {}


# =============================================================================
# Per-test full dump — human-readable text
# =============================================================================
def format_test_log(parsed, summary_line="", config_dict=None):
    blobs    = parsed["blobs"]
    pipe_csv = parsed["pipe"]
    conflict = parsed["conflict"]
    op_rows  = parsed["op_rows"]
    out      = []
    w        = out.append

    cfg = config_dict or {}
    tag = (f"{cfg.get('name','?')}_b{cfg.get('B','?')}_q{cfg.get('Q','?')}_"
           f"{cfg.get('variant','?')}") if cfg else "opprof"

    w("=" * 78)
    head = f"[{tag}]"
    if summary_line:
        head += "  " + summary_line
    w(head)
    w("=" * 78)
    if cfg:
        w(f"Config:      shape={cfg.get('name','?')}  "
          f"B={cfg.get('B','?')} M={cfg.get('M','?')} "
          f"K={cfg.get('K','?')} N={cfg.get('N','?')} "
          f"Q={cfg.get('Q','?')}  variant={cfg.get('variant','?')}")
    w(f"OPPROF dir:  {parsed['opprof_dir']}")
    if len(op_rows) > 1:
        names = [str(r.get("Op Name", "?")) for r in op_rows]
        w(f"Ops in dir:  {len(names)} → " + ", ".join(names))
    w("")

    # ----- blob[0] basic info
    basic = next((b for b in blobs if "block_dim" in b and "name" in b), None)
    if basic:
        w("-- visualize_data.bin blob[0] — basic info --")
        w(f"  name:           {basic.get('name','?')}")
        w(f"  op_type:        {basic.get('op_type','?')}")
        w(f"  soc:            {basic.get('soc','?')}")
        w(f"  block_dim:      {basic.get('block_dim','?')}")
        w(f"  mix_block_dim:  {basic.get('mix_block_dim','')}")
        w(f"  cur_freq:       {basic.get('cur_freq','?')}  MHz")
        w(f"  rated_freq:     {basic.get('rated_freq','?')}  MHz")
        w(f"  device_id:      {basic.get('device_id','?')}")
        w(f"  pid:            {basic.get('pid','?')}")
        dur = basic.get("duration")
        try:
            w(f"  duration:       {float(dur):.3f} µs")
        except (TypeError, ValueError):
            w(f"  duration:       {dur}")
        bd = basic.get("block_detail") or {}
        if bd.get("head_name") and bd.get("row"):
            w("  block_detail:")
            for line in _pad_table([r.get("value", []) for r in bd["row"]],
                                   headers=bd["head_name"]):
                w("  " + line)
        if basic.get("advice"):
            w("  advice:")
            for a in basic["advice"]:
                w(f"    - {a}")
        w("")

    # ----- blob[1] subblock_detail (% with origin_value)
    pct_blob = None
    for b in blobs:
        rows = b.get("subblock_detail")
        if rows and any("origin_value" in r for r in rows):
            pct_blob = rows
            break
    if pct_blob:
        w("-- visualize_data.bin blob[1] — subblock_detail (% of total cycles) --")
        rows = []
        for r in pct_blob:
            origin = _asint(r.get("origin_value"))
            rows.append([
                r.get("name", "?"),
                f"{_num(r.get('value')):.4f}",
                f"{origin:,}" if origin is not None else str(r.get("origin_value")),
                str(r.get("unit", "")),
                str(r.get("block_id", "")),
                str(r.get("block_type", "")),
            ])
        for line in _pad_table(rows, headers=["name", "value%", "origin_cycles",
                                              "unit", "block_id", "block_type"]):
            w(line)
        w("")

    # ----- blob[2] subblock_detail (instruction / block counts)
    cnt_blob = None
    for b in blobs:
        rows = b.get("subblock_detail")
        if rows and all("origin_value" not in r for r in rows):
            cnt_blob = rows
            break
    if cnt_blob:
        w("-- visualize_data.bin blob[2] — subblock_detail (counts) --")
        rows = []
        for r in cnt_blob:
            val = r.get("value", "")
            val_i = _asint(val)
            val_s = f"{val_i:,}" if val_i is not None else str(val)
            rows.append([
                r.get("name", "?"),
                val_s,
                str(r.get("unit", "")),
                str(r.get("block_id", "")),
                str(r.get("block_type", "")),
            ])
        for line in _pad_table(rows, headers=["name", "value", "unit",
                                              "block_id", "block_type"]):
            w(line)
        w("")

    # ----- blob[3] core_memory_map
    cmm_list = next((b.get("core_memory_map") for b in blobs
                     if "core_memory_map" in b), None)
    if cmm_list and isinstance(cmm_list, list) and cmm_list:
        entry = cmm_list[0]
        w("-- visualize_data.bin blob[3] — core_memory_map --")
        w(f"  core_no:     {entry.get('core_no', '?')}")
        w(f"  op_type:     {entry.get('op_type', '?')}")
        w(f"  soc:         {entry.get('soc', '?')}")
        for pk in ("Cube", "Vector"):
            e = entry.get(pk)
            if e:
                w(f"  {pk:<10}  cycle={_asint(e.get('cycle')):,}   "
                  f"ratio={_num(e.get('ratio')):.4f}   "
                  f"total_cycles={_asint(e.get('total_cycles')):,}")
        l2 = entry.get("L2cache") or {}
        if l2:
            def _fi(x): return f"{_asint(x):,}" if _asint(x) is not None else "?"
            w(f"  L2cache     hit={_fi(l2.get('hit'))}   "
              f"miss={_fi(l2.get('miss'))}   "
              f"total={_fi(l2.get('total_request'))}   "
              f"hit_ratio={l2.get('hit_ratio','?')}%")
        if entry.get("advice"):
            w("  advice:")
            for a in entry["advice"]:
                w(f"    - {a}")
        mu = entry.get("memory_unit") or []
        if mu:
            w("  memory_unit:")
            rows = []
            for m in mu:
                bw = m.get("bandwidth")
                try:
                    bw_s = f"{float(bw):.6f}"
                except (ValueError, TypeError):
                    bw_s = str(bw)
                rows.append([
                    str(m.get("memory_path", "?")),
                    _fmt_int(m.get("request")),
                    bw_s,
                    str(m.get("bandwidth_suffix", "")),
                    _fmt_int(m.get("requests_suffix")) if str(m.get("requests_suffix", "")).isdigit() else str(m.get("requests_suffix", "")),
                    str(m.get("peak_ratio", "NA")),
                    str(m.get("display", "")),
                ])
            for line in _pad_table(rows,
                                   headers=["path", "request", "bandwidth(GB/s)",
                                            "bw_suffix", "req_suffix",
                                            "peak_ratio", "display"]):
                w("  " + line)
        w("")

    # ----- blob[4] table_per_block → every table, every row
    tpb_list = next((b.get("table_per_block") for b in blobs
                     if "table_per_block" in b), None)
    if tpb_list and isinstance(tpb_list, list) and tpb_list:
        entry = tpb_list[0]
        w("-- visualize_data.bin blob[4] — table_per_block --")
        w(f"  block_id:        {entry.get('block_id', '?')}")
        w(f"  table_op_type:   {entry.get('table_op_type', '?')}")
        if entry.get("advice"):
            w("  advice:")
            for a in entry["advice"]:
                w(f"    - {a}")
        for td in entry.get("table_detail", []):
            name = td.get("table_name", "?")
            w(f"\n  [{name}]")
            hdr = list(td.get("header_name", []))
            if hdr and hdr[0] == "":
                hdr[0] = "row"
            body = []
            for r in td.get("row", []):
                body.append([str(r.get("name", "?"))] + [str(v) for v in r.get("value", [])])
            for line in _pad_table(body, headers=hdr):
                w("  " + line)
        w("")

    # ----- Derived pipe summary: CSV time/ratio + visualize_data instructions/waits
    aic_time = _num(pipe_csv.get("aic_time(us)"))
    aic_cycles = _asint(pipe_csv.get("aic_total_cycles")) or 0
    vd_pipe = _pipe_rows_from_visualize(blobs)
    w("-- Pipe summary (merged CSV + visualize_data) --")
    w(f"  aic_time={aic_time:.3f} µs   aic_total_cycles={aic_cycles:,}   "
      f"iCache miss={_num(pipe_csv.get('aic_icache_miss_rate'))*100:.3f}%")
    key_map = [
        ("Cube",   "aic_cube_time(us)",   "aic_cube_ratio",   "aic_cube_wait_ratio",   None),
        ("Vector", "aic_vec_time(us)",    "aic_vec_ratio",    "aic_vec_wait_ratio",    None),
        ("MTE1",   "aic_mte1_time(us)",   "aic_mte1_ratio",   "aic_mte1_wait_ratio",   "MTE1"),
        ("MTE2",   "aic_mte2_time(us)",   "aic_mte2_ratio",   "aic_mte2_wait_ratio",   "MTE2"),
        ("MTE3",   "aic_mte3_time(us)",   "aic_mte3_ratio",   "aic_mte3_wait_ratio",   "MTE3"),
        ("Scalar", "aic_scalar_time(us)", "aic_scalar_ratio", None,                    "Scalar"),
    ]
    best_name = None
    best_ratio = -1.0
    for name, _, rk, _, _ in key_map:
        r = _num(pipe_csv.get(rk))
        if r > best_ratio:
            best_ratio = r
            best_name = name
    pipe_rows = []
    for name, tk, rk, wk, vk in key_map:
        t = _num(pipe_csv.get(tk))
        r = _num(pipe_csv.get(rk)) * 100
        wr = _num(conflict.get(wk)) * 100 if wk else None
        vd_row = vd_pipe.get(vk) if vk else None
        instr = _asint(vd_row[0]) if vd_row and len(vd_row) > 0 else None
        cyc = _asint(vd_row[1]) if vd_row and len(vd_row) > 1 else None
        wait_cyc = _asint(vd_row[2]) if vd_row and len(vd_row) > 2 else None
        act_rate = None
        if vd_row and len(vd_row) > 3:
            try:
                act_rate = float(vd_row[3])
            except (ValueError, TypeError):
                act_rate = None
        # Derive Cube/Vector wait cycles from wait_ratio × total cycles.
        if wait_cyc is None and wr is not None and aic_cycles:
            wait_cyc = int(aic_cycles * wr / 100.0)
        # Derive Cube/Vector active cycles from ratio × total cycles.
        if cyc is None and aic_cycles:
            cyc = int(aic_cycles * (r / 100.0))
        pipe_rows.append([
            name + ("  <BN>" if name == best_name else ""),
            f"{t:.2f}",
            f"{r:.2f}",
            f"{wr:.1f}" if wr is not None else "-",
            _fmt_int(instr),
            _fmt_int(cyc),
            _fmt_int(wait_cyc),
            f"{act_rate:.2f}" if act_rate is not None else "-",
        ])
    for line in _pad_table(pipe_rows,
                           headers=["Pipe", "Time(µs)", "Ratio(%)", "Wait(%)",
                                    "Instructions", "Cycles", "WaitCyc",
                                    "ActRate(%)"]):
        w(line)
    w("")

    # ----- Each OPPROF CSV dumped fully
    csv_specs = [
        ("OpBasicInfo.csv",           op_rows),
        ("PipeUtilization.csv",       [pipe_csv] if pipe_csv else []),
        ("Memory.csv",                [parsed["mem"]] if parsed["mem"] else []),
        ("MemoryL0.csv",              [parsed["memL0"]] if parsed["memL0"] else []),
        ("MemoryUB.csv",              [parsed["memUB"]] if parsed["memUB"] else []),
        ("L2Cache.csv",               [parsed["l2"]] if parsed["l2"] else []),
        ("ResourceConflictRatio.csv", [parsed["conflict"]] if parsed["conflict"] else []),
        ("ArithmeticUtilization.csv", [parsed["arith"]] if parsed["arith"] else []),
    ]
    for fname, rows_ in csv_specs:
        if not rows_:
            continue
        w(f"-- {fname} --")
        keys = list(rows_[0].keys())
        if len(rows_) == 1:
            body = [[k, str(rows_[0].get(k, ""))] for k in keys]
            for line in _pad_table(body, headers=["column", "value"], align="left"):
                w(line)
        else:
            body = [[str(r.get(k, "")) for k in keys] for r in rows_]
            for line in _pad_table(body, headers=keys):
                w(line)
        w("")

    return "\n".join(out)


# =============================================================================
# Aggregate STATS.txt
# =============================================================================
def _vd_pipe_instructions(blobs, pipe_name):
    """Return instruction count for a pipe from visualize_data.bin's Pipe table."""
    rows = _pipe_rows_from_visualize(blobs)
    vals = rows.get(pipe_name)
    if not vals:
        return None
    return _asint(vals[0]) if len(vals) > 0 else None


def _mem_unit(blobs, path_substr):
    """Return the memory_unit entry whose 'memory_path' contains path_substr."""
    cmm_list = next((b.get("core_memory_map") for b in blobs
                     if "core_memory_map" in b), None)
    if not cmm_list or not isinstance(cmm_list, list) or not cmm_list:
        return None
    for m in (cmm_list[0].get("memory_unit") or []):
        if path_substr in str(m.get("memory_path", "")):
            return m
    return None


def _mem_bw(blobs, path_substr):
    """Effective bandwidth GB/s on memory_unit path matching substring."""
    m = _mem_unit(blobs, path_substr)
    if not m:
        return None
    try:
        return float(m.get("bandwidth"))
    except (TypeError, ValueError):
        return None


def _mem_req(blobs, path_substr):
    """Request count on memory_unit path matching substring."""
    m = _mem_unit(blobs, path_substr)
    if not m:
        return None
    return _asint(m.get("request"))


def summary_row(parsed, cfg, summary_line=""):
    op = pick_matmul_row(parsed["op_rows"])
    pipe_csv = parsed["pipe"]
    conflict = parsed["conflict"]
    l2 = parsed["l2"]
    blobs = parsed["blobs"]

    cos_m = re.search(r"cos=([0-9.]+)", summary_line or "")
    pass_m = re.search(r"PASS=(\w+)", summary_line or "")

    return {
        "tag": (f"{cfg.get('name','?')}_{cfg.get('variant','?')}") if cfg
                else os.path.basename(os.path.dirname(parsed["opprof_dir"])),
        "shape":   cfg.get("name", "?"),
        "E":       cfg.get("E"),
        "M":       cfg.get("M"),
        "K":       cfg.get("K"),
        "N":       cfg.get("N"),
        "variant": cfg.get("variant", "?"),
        "cos":     cos_m.group(1) if cos_m else "",
        "pass":    pass_m.group(1) if pass_m else "",
        "task_us": _num(op.get("Task Duration(us)")),
        "block_dim": int(_num(op.get("Block Dim"))),
        # Pipe ratios + wait % (fraction of cycles pipe was waiting).
        "cube_r":    _num(pipe_csv.get("aic_cube_ratio")) * 100,
        "cube_w":    _num(conflict.get("aic_cube_wait_ratio")) * 100,
        "vec_r":     _num(pipe_csv.get("aic_vec_ratio")) * 100,
        "vec_w":     _num(conflict.get("aic_vec_wait_ratio")) * 100,
        "mte1_r":    _num(pipe_csv.get("aic_mte1_ratio")) * 100,
        "mte1_w":    _num(conflict.get("aic_mte1_wait_ratio")) * 100,
        "mte2_r":    _num(pipe_csv.get("aic_mte2_ratio")) * 100,
        "mte2_w":    _num(conflict.get("aic_mte2_wait_ratio")) * 100,
        "mte3_r":    _num(pipe_csv.get("aic_mte3_ratio")) * 100,
        "mte3_w":    _num(conflict.get("aic_mte3_wait_ratio")) * 100,
        "scalar_r":  _num(pipe_csv.get("aic_scalar_ratio")) * 100,
        # Instruction counts (visualize_data.bin Pipe table).
        "mte1_instr": _vd_pipe_instructions(blobs, "MTE1"),
        "mte2_instr": _vd_pipe_instructions(blobs, "MTE2"),
        "mte3_instr": _vd_pipe_instructions(blobs, "MTE3"),
        # Memory path bandwidths (from core_memory_map.memory_unit).
        "gm_read_bw":  _mem_bw(blobs, "GM->L1") or _mem_bw(blobs, "DDR"),
        "gm_write_bw": _mem_bw(blobs, "UB->GM") or _mem_bw(blobs, "GM"),
        "l2_hit":    _num(l2.get("aic_l2_cache_hit_rate(%)")),
        "icache_miss": _num(pipe_csv.get("aic_icache_miss_rate")) * 100,
        # Per-task derived (expected MTE2 bytes for one forward pass).
        #   weight_int8 bytes per core  ≈ E·K·N (if grouped-full) / coreNum
        #   x_fp16 bytes                ≈ M·K·2
        # Computed lazily in format_stats_table.
    }


def format_stats_table(rows, out_dir):
    lines = []
    w = lines.append

    pass_count = sum(1 for r in rows if r.get("pass") == "True")
    fail_count = sum(1 for r in rows if r.get("pass") == "False")

    w(f"Device test STATS   out={out_dir}")
    w(f"  Tests:      {len(rows)}  (pass={pass_count}, fail={fail_count})")
    w("")

    # Geomean Task Duration per shape family
    by_shape = defaultdict(list)
    for r in rows:
        t = r.get("task_us") or 0.0
        if t > 0.0:
            by_shape[r["shape"]].append(t)
    if by_shape:
        w("Geomean Task Duration per shape family:")
        hdr = ["shape", "n", "geomean µs", "min µs", "max µs"]
        body = []
        for shape in sorted(by_shape):
            vs = by_shape[shape]
            gm = math.exp(sum(math.log(v) for v in vs) / len(vs))
            body.append([shape, str(len(vs)),
                         f"{gm:.3f}", f"{min(vs):.3f}", f"{max(vs):.3f}"])
        for line in _pad_table(body, headers=hdr):
            w(line)
        w("")

    # --- Primary table: per-test cos + pipe ratios + L2 ---
    hdr = ["tag", "E", "M", "K", "N", "cos%", "PASS",
           "Task µs", "BlkDim",
           "Cube%|W%", "Vec%|W%", "MTE1%|W%", "MTE2%|W%", "MTE3%|W%",
           "Scalar%", "L2 hit%", "iCache%"]
    body = []
    for r in sorted(rows, key=lambda r: (r["shape"], r["variant"])):
        body.append([
            r["tag"],
            str(r.get("E") or "-"),
            str(r.get("M") or "-"),
            str(r.get("K") or "-"),
            str(r.get("N") or "-"),
            r["cos"] or "-",
            r["pass"] or "-",
            f"{r['task_us']:.3f}" if r["task_us"] else "-",
            str(r["block_dim"]),
            f"{r['cube_r']:.2f}|{r['cube_w']:.1f}",
            f"{r['vec_r']:.2f}|{r['vec_w']:.1f}",
            f"{r['mte1_r']:.2f}|{r['mte1_w']:.1f}",
            f"{r['mte2_r']:.2f}|{r['mte2_w']:.1f}",
            f"{r['mte3_r']:.2f}|{r['mte3_w']:.1f}",
            f"{r['scalar_r']:.2f}",
            f"{r['l2_hit']:.2f}",
            f"{r['icache_miss']:.3f}",
        ])
    w("Per-test pipe breakdown (Ratio% | Wait%):")
    for line in _pad_table(body, headers=hdr):
        w(line)
    w("")

    # --- Secondary table: MTE bandwidth and instruction counts ---
    hdr2 = ["tag", "MTE2 instr", "MTE3 instr", "MTE1 instr",
            "GM read GB/s", "GM write GB/s",
            "est W·MTE2 B", "MTE2 B/cyc"]
    body2 = []
    PEAK_MTE2_BC = 17.0   # 310P3 DDR read = 17 B/cyc
    for r in sorted(rows, key=lambda r: (r["shape"], r["variant"])):
        E = r.get("E") or 1
        K = r.get("K") or 0
        N = r.get("N") or 0
        # Estimated weight bytes per forward pass: E·K·N int8 worth of DRAM
        # reads if each expert's weight is fully walked once and no L2
        # reuse kicks in. Divide by block_dim for per-core.
        bd = r.get("block_dim") or 1
        w_bytes = int(E * K * N)
        per_core = w_bytes // max(bd, 1)
        # MTE2 B/cyc effective = MTE2 instr count × blk_size × pipe_ratio…
        # Simpler: use aic_main_mem_read bandwidth and convert.
        task_us = r.get("task_us") or 0.0
        if task_us > 0:
            # ddr_bytes_read ≈ per_core × blk_dim = w_bytes
            # effective B/cyc = w_bytes / (task_us × 1e-6 × 1e9_cyc) — but
            # we don't know clock. Use gm_read_bw GB/s directly — compare
            # to peak 17 B/cyc × freq ≈ 17 × 1.08 GHz ≈ 18.3 GB/s per core.
            pass
        body2.append([
            r["tag"],
            _fmt_int(r.get("mte2_instr")),
            _fmt_int(r.get("mte3_instr")),
            _fmt_int(r.get("mte1_instr")),
            f"{r.get('gm_read_bw'):.2f}" if r.get("gm_read_bw") is not None else "-",
            f"{r.get('gm_write_bw'):.2f}" if r.get("gm_write_bw") is not None else "-",
            _fmt_int(per_core),
            "-",   # placeholder for future B/cyc derivation
        ])
    w(f"Bandwidth & instruction counts (peak MTE2 = {PEAK_MTE2_BC} B/cyc):")
    for line in _pad_table(body2, headers=hdr2):
        w(line)
    w("")

    return "\n".join(lines)


# =============================================================================
# Discovery
# =============================================================================
_OPPROF_RE = re.compile(r"^OPPROF_\d{14}_[A-Z0-9]+$")
_SUMMARY_LINE_RE = re.compile(r"^\s*(\S+)\s+\.\.\.\s+(.+)$")


def find_opprof_dirs(root):
    root = Path(root)
    if not root.is_dir():
        return
    for tag_dir in sorted(root.iterdir()):
        if not tag_dir.is_dir():
            continue
        for child in sorted(tag_dir.iterdir()):
            if child.is_dir() and _OPPROF_RE.match(child.name):
                yield tag_dir.name, child


def load_summary(summary_path):
    out = {}
    if not summary_path or not os.path.exists(summary_path):
        return out
    with open(summary_path) as f:
        for line in f:
            m = _SUMMARY_LINE_RE.match(line.rstrip())
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


# =============================================================================
# CLI — iterate OPPROF_* dirs, emit per-test logs + STATS.txt
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Produce per-test .log files and an aggregate STATS.txt "
                    "from OPPROF_* directories under a device-test run."
    )
    p.add_argument("--root", required=True,
                   help="Run output directory containing <tag>/OPPROF_* subdirs")
    p.add_argument("--out", default=None,
                   help="STATS.txt path (default: <root>/STATS.txt)")
    p.add_argument("--summary", default=None,
                   help="SUMMARY.txt for cos/PASS attribution "
                        "(default: <root>/SUMMARY.txt)")
    p.add_argument("--no-per-test-logs", action="store_true",
                   help="Skip writing <tag>.log files")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    summary_path = args.summary or os.path.join(root, "SUMMARY.txt")
    stats_out = args.out or os.path.join(root, "STATS.txt")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from configs import build_all_configs, config_tag as _cfg_tag
        config_by_tag = {_cfg_tag(c): c for c in build_all_configs()}
    except Exception:
        config_by_tag = {}

    summary_lines = load_summary(summary_path)

    rows = []
    found = 0
    for tag, opprof in find_opprof_dirs(root):
        parsed = parse_opprof(opprof)
        cfg = config_by_tag.get(tag, {})
        sline = summary_lines.get(tag, "")
        rows.append(summary_row(parsed, cfg, sline))
        if not args.no_per_test_logs:
            log_text = format_test_log(parsed, summary_line=sline, config_dict=cfg)
            with open(os.path.join(root, f"{tag}.log"), "w") as f:
                f.write(log_text + "\n")
        found += 1

    if not found:
        print(f"No OPPROF_* dirs found under {root}", file=sys.stderr)
        return 1

    with open(stats_out, "w") as f:
        f.write(format_stats_table(rows, root) + "\n")
    print(f"Wrote {stats_out} and {found} per-test .log files under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

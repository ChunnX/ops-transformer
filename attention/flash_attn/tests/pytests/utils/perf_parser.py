"""Parse msprof op_summary.csv, group by case, cold/hot stats with crash detection."""

import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_op_summary(csv_path: str,
                     operator_name: str = "FlashAttn") -> List[Tuple[float, float]]:
    """Return [(start_time_us, duration_us), ...] sorted by start time."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    entries = [
        (float(r["Task Start Time(us)"]), float(r["Task Duration(us)"]))
        for r in rows
        if r["OP Type"] == operator_name
    ]
    entries.sort(key=lambda x: x[0])
    return entries


def _read_perf_log(log_dir: str) -> dict:
    """Read _perf_log.jsonl, return {tag: {"ok": bool, "error": str}}."""
    log_path = Path(log_dir) / "_perf_log.jsonl"
    if not log_path.exists():
        return {}
    result = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                result[d["tag"]] = d
            except json.JSONDecodeError:
                pass
    return result


def compute_stats(entries: List[Tuple[float, float]],
                  cases: List[Dict],
                  runs: int,
                  cold_thr: int,
                  log_dir: str = None) -> Tuple[Dict[str, Dict], List[str]]:
    """Group entries by case and compute statistics.

    Uses _perf_log.jsonl for crash detection and entry alignment.
    """
    hot_cases = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) > cold_thr]
    cold_cases = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) <= cold_thr]

    # Build tag→case key mapping from log
    perf_log = _read_perf_log(log_dir) if log_dir else {}
    tag_to_key = {}
    for idx, (orig_i, c) in enumerate(hot_cases):
        tag_to_key[f"hot{idx}"] = _case_key(c)
    for idx, (orig_i, c) in enumerate(cold_cases):
        tag_to_key[f"cold{idx}"] = _case_key(c)

    # Determine crashed cases from log
    crashed_tags = {tag for tag, info in perf_log.items()
                    if not info.get("ok", False)}
    crash_keys = {tag_to_key.get(t, t) for t in crashed_tags}

    # Build expected entries, skipping crashed cases
    expected = []
    for idx, (orig_i, c) in enumerate(hot_cases):
        tag = f"hot{idx}"
        if tag in crashed_tags:
            continue
        for _ in range(runs):
            expected.append((_case_key(c), "hot"))
    for idx, (orig_i, c) in enumerate(cold_cases):
        tag = f"cold{idx}"
        if tag in crashed_tags:
            continue
        for _ in range(runs):
            expected.append((_case_key(c), "cold"))

    if len(entries) > len(expected) and expected:
        ratio = len(entries) // len(expected)
        if ratio > 1:
            print(f"[perf] entries={len(entries)} expected={len(expected)} ratio={ratio}")
            entries = [entries[i * ratio] for i in range(len(expected))]

    n = min(len(entries), len(expected))
    raw = defaultdict(lambda: {"hot": [], "cold": []})

    for i in range(n):
        c_key, mode = expected[i]
        raw[c_key][mode].append(entries[i][1])

    result = {}
    crashed = []

    # Add crashed cases first
    for key in crash_keys:
        error_msg = ""
        for tag, info in perf_log.items():
            if tag_to_key.get(tag) == key:
                error_msg = info.get("error", "").replace('"', "'")
                break
        result[key] = {
            "avg_us": 0.0, "min_us": 0.0, "max_us": 0.0,
            "mode": "CRASH",
            "error": error_msg,
        }
        crashed.append(key)

    # Add OK cases
    for key, d in raw.items():
        if key in crash_keys:
            continue
        if d["hot"]:
            keep = d["hot"][1:]
            avg = sum(keep) / len(keep) if keep else 0.0
            mode = "hot"
            all_durs = d["hot"]
        elif d["cold"]:
            vals = sorted(d["cold"])
            keep = vals[1:-1] if len(vals) >= 3 else vals
            avg = sum(keep) / len(keep) if keep else 0.0
            mode = "cold"
            all_durs = d["cold"]
        else:
            continue
        result[key] = {
            "avg_us": round(avg, 2),
            "min_us": round(min(all_durs), 2) if all_durs else 0.0,
            "max_us": round(max(all_durs), 2) if all_durs else 0.0,
            "mode": mode,
        }

    return result, crashed


def _case_key(c: Dict) -> str:
    parts = []
    for k in ("B", "N1", "N2", "S1", "S2", "D", "Dtype",
              "layout_q", "layout_kv", "mask_mode"):
        if k in c:
            parts.append(f"{k}={c[k]}")
    return "_".join(parts)

"""Parse msprof op_summary.csv, group by case, cold/hot stats with crash detection."""

import csv
from collections import defaultdict
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


def compute_stats(entries: List[Tuple[float, float]],
                  cases: List[Dict],
                  runs: int,
                  cold_thr: int) -> Tuple[Dict[str, Dict], List[str]]:
    """Group entries by case and compute statistics.

    Hot (S1 > cold_thr): discard 1st run, avg remaining runs-1
    Cold (S1 <= cold_thr): remove max/min, avg middle values

    Returns:
        (results, crashed_cases) where
        - results: {case_key: {"avg_us", "min_us", "max_us", "mode"}}
        - crashed_cases: list of case_keys that had no entries
    """
    hot_cases = [c for c in cases if c.get("S1", 9999) > cold_thr]
    cold_cases = [c for c in cases if c.get("S1", 9999) <= cold_thr]

    expected = []
    for c in hot_cases:
        for _ in range(runs):
            expected.append((c, "hot"))
    for c in cold_cases:
        for _ in range(runs):
            expected.append((c, "cold"))

    if len(entries) > len(expected) and expected:
        ratio = len(entries) // len(expected)
        if ratio > 1:
            print(f"[perf] entries={len(entries)} expected={len(expected)} ratio={ratio}")
            entries = [entries[i * ratio] for i in range(len(expected))]

    n = min(len(entries), len(expected))
    raw = defaultdict(lambda: {"hot": [], "cold": []})

    for i in range(n):
        c, mode = expected[i]
        raw[_case_key(c)][mode].append(entries[i][1])

    result = {}
    crashed = []

    for key, d in raw.items():
        if not d["hot"] and not d["cold"]:
            result[key] = {"avg_us": 0.0, "min_us": 0.0, "max_us": 0.0, "mode": "CRASH"}
            crashed.append(key)
            continue

        if d["hot"]:
            keep = d["hot"][1:]  # discard 1st
            avg = sum(keep) / len(keep) if keep else 0.0
            mode = "hot"
            all_durs = d["hot"]
        elif d["cold"]:
            vals = sorted(d["cold"])
            keep = vals[1:-1] if len(vals) >= 3 else vals  # remove min and max
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

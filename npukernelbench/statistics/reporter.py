"""Statistics formatting and display — per-case detail + per-operator summary."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def generate_statistics(
    builds,
    config,
    tests: Optional[list] = None,
) -> None:
    """Generate and print success statistics.

    Prints two sections:
    1. Per-case detail table — one row per (operator, case)
    2. Per-operator summary — aggregated pass rates + total score
    """
    from npukernelbench.statistics.collector import collect_results

    rows = collect_results(builds, tests)
    if not rows:
        print("No results to display.")
        return

    df = pd.DataFrame(rows)
    has_cases = "case_id" in df.columns and df["case_id"].notna().any()

    # ── Section 1: Per-case detail ──────────────────────────────────
    if has_cases:
        _print_case_detail(df)

    # ── Section 2: Per-operator summary ─────────────────────────────
    _print_operator_summary(df)


# ────────────────────────────────────────────────────────────────────
# Per-case detail
# ────────────────────────────────────────────────────────────────────

_DETAIL_COLS = [
    "op_name", "case_id", "compile_success",
    "precision_success", "avg_abs_error", "max_abs_error",
    "avg_rel_error", "max_rel_error", "avg_acc",
]

_SUMMARY_DETAIL_COLS = [
    "op_name", "case_id", "compile",
    "precision", "avg_abs", "max_abs", "avg_rel", "max_rel", "acc",
]


def _print_case_detail(df: pd.DataFrame) -> None:
    """Print one row per case with precision metrics."""
    # Build display dataframe with clean column names
    detail = pd.DataFrame()
    detail["op_name"] = df["op_name"].values
    detail["case_id"] = df["case_id"].values
    detail["compile"] = df["compile_success"].apply(
        lambda x: "PASS" if x is True else ("FAIL" if x is False else "-"))
    detail["precision"] = df["precision_success"].apply(
        lambda x: "PASS" if x is True else ("FAIL" if x is False else "-"))

    for src, dst in [("avg_abs_error", "avg_abs"), ("max_abs_error", "max_abs"),
                     ("avg_rel_error", "avg_rel"), ("max_rel_error", "max_rel")]:
        detail[dst] = df[src].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) and x is not None else "-")

    detail["acc"] = df["avg_acc"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "-"
    )

    print("\n" + "=" * 100)
    print("Per-Case Precision Detail")
    print("=" * 100)

    # Group by operator for readability
    for op_name, group in detail.groupby("op_name", sort=False):
        print(f"\n  [{op_name}]  (cases: {len(group)})\n")
        try:
            from tabulate import tabulate
            print(tabulate(
                group.reset_index(drop=True),
                headers="keys", tablefmt="simple", showindex=False,
            ))
        except ImportError:
            print(group.to_string(index=False))
        print()


# ────────────────────────────────────────────────────────────────────
# Per-operator summary
# ────────────────────────────────────────────────────────────────────

def _print_operator_summary(df: pd.DataFrame) -> None:
    """Print aggregated pass-rate summary and total score."""
    # Determine grouping key
    if "case_id" in df.columns and df["case_id"].notna().any():
        group_keys = ["level_id", "category", "op_name"]
    else:
        group_keys = ["level_id", "category", "op_name"]

    grouped = df.groupby(group_keys)
    stats = []

    for keys, group in grouped:
        if len(group_keys) == 3:
            level, cat, op = keys
        else:
            level, cat, op = keys[0], keys[1], keys[2]

        total = len(group)
        compile_pass = _count_pass(group, "compile_success")
        precision_pass = _count_pass(group, "precision_success")
        perf_pass = _count_pass(group, "perf_success")

        stats.append({
            "op_level": int(level) if level else 0,
            "op_definition": f"{cat}/{op}",
            "total_cases": total,
            "compile_pass": compile_pass,
            "precision_pass": precision_pass,
            "perf_pass": perf_pass,
        })

    stats_df = pd.DataFrame(stats).sort_values(["op_level", "op_definition"])

    print("=" * 100)
    print("Summary by Operator")
    print("=" * 100)

    try:
        from tabulate import tabulate
        for level in stats_df["op_level"].unique():
            level_data = stats_df[stats_df["op_level"] == level]
            print(f"\n[Level {level}]\n")
            print(tabulate(
                level_data.reset_index(drop=True),
                headers="keys", tablefmt="grid", showindex=False,
            ))
            print()
    except ImportError:
        print(stats_df.to_string(index=False))

    _print_score(stats_df)


def _count_pass(group: pd.DataFrame, col: str, default: int = 0) -> int:
    if col not in group:
        return default
    non_null = group[col].dropna()
    if non_null.empty:
        return default
    return int(non_null.sum())


def _print_score(stats_df: pd.DataFrame) -> None:
    weight = {1: 0.2, 2: 0.3, 3: 0.5}
    summary = stats_df.groupby("op_level").sum(numeric_only=True)

    score_col = "precision_pass" if stats_df["precision_pass"].sum() > 0 else "compile_pass"

    score = 0.0
    for level, row in summary.iterrows():
        if level in weight and row["total_cases"] > 0:
            score += weight[level] * row[score_col] / row["total_cases"]

    print(f"NPUKernelBench total score (based on {score_col}): {score:.2f}")
    print()

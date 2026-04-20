"""Result collection and aggregation — one row per (operator, sample, case)."""

from __future__ import annotations

from typing import List, Optional

from npukernelbench.models.task import BuildHandle, TestHandle


def collect_results(
    builds: List[BuildHandle],
    tests: Optional[List[TestHandle]] = None,
) -> List[dict]:
    """Produce one row per TestHandle (i.e. per case).

    Falls back to one row per BuildHandle when no tests are provided.
    Each row contains build-level results (code_gen, compile) plus
    case-level results (precision, perf) with detailed metrics.
    """
    if not tests:
        # No test handles — one row per build (no case granularity)
        return [_build_row(b) for b in builds]

    rows = []
    for t in tests:
        build = t.build
        row = {
            # Identity
            "op_name": build.metadata.op_name,
            "category": build.metadata.category,
            "level_id": build.metadata.level_id,
            "sample_id": build.sample_id,
            "case_id": t.case_id,
            # Build-level results
            "code_gen_success": build.code_gen_result.success if build.code_gen_result else None,
            "compile_success": build.compile_result.success if build.compile_result else None,
        }

        # Case-level precision
        pr = t.precision_result
        if pr is not None:
            row["precision_success"] = pr.success
            row["n_success"] = pr.n_success
            row["n_failed"] = pr.n_failed
            # Extract metrics from results_dict if available
            if pr.results_dict:
                row.update(_extract_metrics(pr.results_dict))
            else:
                row.update(_empty_metrics())
        else:
            row["precision_success"] = None
            row["n_success"] = 0
            row["n_failed"] = 0
            row.update(_empty_metrics())

        # Case-level perf
        pf = t.perf_result
        if pf is not None:
            row["perf_success"] = pf.success
            if pf.results_dict:
                row.update(_extract_perf_metrics(pf.results_dict))
        else:
            row["perf_success"] = None

        rows.append(row)

    return rows


def _build_row(build: BuildHandle) -> dict:
    """Fallback: one row per build without case detail."""
    return {
        "op_name": build.metadata.op_name,
        "category": build.metadata.category,
        "level_id": build.metadata.level_id,
        "sample_id": build.sample_id,
        "case_id": None,
        "code_gen_success": build.code_gen_result.success if build.code_gen_result else None,
        "compile_success": build.compile_result.success if build.compile_result else None,
        "precision_success": None,
        "perf_success": None,
    }


def _extract_metrics(results_dict: list[dict]) -> dict:
    """Extract best/average precision metrics from results_dict entries."""
    if not results_dict:
        return _empty_metrics()

    # Take the last entry (most representative)
    entry = results_dict[-1]
    return {
        "avg_abs_error": entry.get("avg_abs_error"),
        "avg_rel_error": entry.get("avg_rel_error"),
        "max_abs_error": entry.get("max_abs_error"),
        "max_rel_error": entry.get("max_rel_error"),
        "avg_acc": entry.get("avg_acc"),
    }


def _extract_perf_metrics(results_dict: dict) -> dict:
    """Extract perf metrics."""
    if not results_dict:
        return {}
    # Average across all entries
    times_model = [v.get("avg_model_time", 0) for v in results_dict.values() if isinstance(v, dict)]
    times_new = [v.get("avg_model_new_time", 0) for v in results_dict.values() if isinstance(v, dict)]
    return {
        "avg_model_time": sum(times_model) / len(times_model) if times_model else None,
        "avg_model_new_time": sum(times_new) / len(times_new) if times_new else None,
    }


def _empty_metrics() -> dict:
    return {
        "avg_abs_error": None,
        "avg_rel_error": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "avg_acc": None,
    }

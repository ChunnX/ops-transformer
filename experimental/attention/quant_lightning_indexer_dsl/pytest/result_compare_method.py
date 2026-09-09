"""ASC-style ResultCompare entry points backed by the independent DSL policy."""

from qli_test_utils.result_compare import compare_qli_outputs


def _percent(report):
    rows = set(report["sparse"].get("mismatched_rows", []))
    rows.update(report["candidate"].get("mismatched_rows", []))
    return 100.0 if not rows else 0.0


def check_result(
    cpu_result, npu_result, topk_value=None, output_idx_offset=None,
    params=None, cpu_topk_value=None, npu_topk_value=None,
):
    del topk_value, output_idx_offset, params, cpu_topk_value, npu_topk_value
    report = compare_qli_outputs(cpu_result, npu_result)
    return ("Pass" if report["pass"] else "Failed", _percent(report))


def check_result_return_value(
    cpu_topk_value, npu_topk_value, params, cpu_result, npu_result,
    topk_value=None, output_idx_offset=None,
):
    del cpu_topk_value, npu_topk_value, params, topk_value, output_idx_offset
    report = compare_qli_outputs(cpu_result, npu_result)
    return ("Pass" if report["pass"] else "Failed", _percent(report))


from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from golden import (
    CombineGolden,
    Feature,
    GoldenHandle,
    expected_dispatch_rows_from_metadata,
    feature_main,
    feature_scales,
    is_tuple_feature,
)

try:
    import toolss
except ModuleNotFoundError as err:  # pragma: no cover - 本地 toolss 可能缺少非 compare 依赖
    if err.name == "psutil":
        sys.modules.setdefault("psutil", types.ModuleType("psutil"))
        try:
            import toolss
        except Exception:
            toolss = None
    else:
        toolss = None
except Exception:  # pragma: no cover - toolss 仅用于失败诊断
    toolss = None


COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_GRAY = "\033[90m"
COLOR_RESET = "\033[0m"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
# toolss 内部会计算 1 / diff_thd，不能传 0；max_diff_hd=0 仍保证非零误差失败。
TOOLSS_DIFF_THD = 1e-12
TOOLSS_PCT_THD = 0.0
TOOLSS_MAX_DIFF_HD = 0.0


@dataclass
class CompareConfig:
    server_index: int
    rank_per_dev: int
    server_ranks: slice
    dispatch_with_handle: bool
    dispatch_with_topk_weights: bool
    check_invalid_padding: bool
    strict_dispatch_metadata_order: bool
    print_detail_max_elements: int


@dataclass
class CompareRecord:
    key: str
    name: str
    rank: int
    item: str
    order: int
    ok: bool
    msg: str
    actual: Any = None
    expected: Any = None
    csv_rows: List[Tuple[str, Any]] = field(default_factory=list)
    toolss_details: List[Tuple[str, str]] = field(default_factory=list)


class CompareResult(dict):
    def __init__(self) -> None:
        super().__init__()
        self.csv_rows: List[Tuple[str, Any]] = []

    def update(self, other=(), /, **kwargs) -> None:
        super().update(other, **kwargs)
        if hasattr(other, "csv_rows"):
            self.csv_rows.extend(other.csv_rows)


def _valid_rows(handle: GoldenHandle) -> int:
    return int(handle.num_recv_tokens_per_expert.sum().item())


def _metadata_compare_view(metadata: torch.Tensor, valid_rows: int, check_slot_idx: bool) -> torch.Tensor:
    view = metadata[:valid_rows]
    if check_slot_idx:
        return view
    return view[:, :3]


def _sort_metadata(metadata: torch.Tensor, valid_rows: int, check_slot_idx: bool) -> torch.Tensor:
    view = _metadata_compare_view(metadata, valid_rows, check_slot_idx)
    rows = sorted(tuple(int(v) for v in row.tolist()) for row in view)
    if not rows:
        return torch.empty((0, view.shape[1]), dtype=torch.int32)
    return torch.tensor(rows, dtype=torch.int32)


def _strip_compare_name(name: str, msg: str) -> str:
    prefix = f"{name} "
    return msg[len(prefix):] if msg.startswith(prefix) else msg


def _tensor_detail(tensor: Optional[torch.Tensor], config: CompareConfig) -> str:
    if tensor is None:
        return "None"
    tensor_cpu = tensor.detach().cpu().contiguous()
    data_for_print = tensor_cpu.view(torch.uint8) if "float8" in str(tensor_cpu.dtype) else tensor_cpu
    flat = data_for_print.reshape(-1)
    max_elements = config.print_detail_max_elements
    if max_elements <= 0 or flat.numel() <= max_elements:
        values = flat.tolist()
    else:
        values = flat[:max_elements].tolist()
        values.append("...")
    return f"shape={tuple(tensor_cpu.shape)}, dtype={tensor_cpu.dtype}, values={values}"


def _feature_detail(feature: Optional[Feature], config: CompareConfig) -> str:
    if feature is None:
        return "None"
    if is_tuple_feature(feature):
        return f"(x={_tensor_detail(feature_main(feature), config)}, scales={_tensor_detail(feature_scales(feature), config)})"
    return _tensor_detail(feature_main(feature), config)


def _compare_object_detail(value: Any, config: CompareConfig) -> str:
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        return _tensor_detail(value, config)
    if is_tuple_feature(value):
        return _feature_detail(value, config)
    return repr(value)


def _brief_compare_object_detail(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}"
    if is_tuple_feature(value):
        main = feature_main(value)
        scales = feature_scales(value)
        return f"(x=shape={tuple(main.shape)}, dtype={main.dtype}, scales=shape={tuple(scales.shape)}, dtype={scales.dtype})"
    return repr(value)


def _sorted_records(records: List[CompareRecord]) -> List[CompareRecord]:
    return sorted(records, key=lambda record: (record.rank, record.order, record.item, record.key))


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if is_tuple_feature(value):
        main = feature_main(value)
        scales = feature_scales(value)
        return main.numel() == 0 and (scales is None or scales.numel() == 0)
    return False


def _is_empty_record(record: CompareRecord) -> bool:
    return record.ok and _is_empty_value(record.actual) and _is_empty_value(record.expected)


def _colored_status(record: CompareRecord) -> str:
    if _is_empty_record(record):
        return f"{COLOR_GRAY}-{COLOR_RESET}"
    ok = record.ok
    if ok:
        return f"{COLOR_GREEN}PASS{COLOR_RESET}"
    return f"{COLOR_RED}FAIL{COLOR_RESET}"


def _visible_len(text: str) -> int:
    return len(ANSI_PATTERN.sub("", text))


def _pad_table_cell(text: str, width: int) -> str:
    return text + " " * (width - _visible_len(text))


def _center_table_cell(text: str, width: int) -> str:
    pad = width - _visible_len(text)
    left = pad // 2
    right = pad - left
    return " " * left + text + " " * right


def _result_key(tag: str, item: str, rank: int) -> str:
    return f"{tag.replace('.', '_')}_{item.replace('.', '_')}_rank{rank}"


def tensor_for_exact_compare(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    if "float8" in str(tensor.dtype):
        return tensor.view(torch.uint8)
    return tensor


def _tensor_to_toolss_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor_for_exact_compare(tensor.detach().cpu())
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy().astype(np.float32, copy=False)


def _fallback_data_compare_np(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    diff_thd: float,
    pct_thd: float,
    max_diff_hd: float,
) -> Tuple[Any, ...]:
    actual_flat = actual.flatten().astype(np.float32, copy=False)
    expected_flat = expected.flatten().astype(np.float32, copy=False)
    if actual_flat.size != expected_flat.size:
        result = "FAIL"
        pct_rlt = 0.0
        max_error = 0.0
    elif actual_flat.size == 0:
        result = "PASS"
        pct_rlt = 100.0
        max_error = 0.0
    else:
        diff_abs = np.abs(actual_flat - expected_flat)
        diff_mask = diff_abs > diff_thd
        pct_rlt = float(actual_flat.size - np.count_nonzero(diff_mask)) / float(actual_flat.size) * 100.0
        max_error = float(diff_abs.max()) if diff_abs.size else 0.0
        pct_limit = (1 - pct_thd) * 100.0
        result = "PASS" if pct_rlt >= pct_limit and max_error <= max_diff_hd else "FAIL"
    print("---------------------------------------------------------------------------------------")
    print("DiffThd  \t PctThd   \t PctRlt   \t Result")
    print("---------------------------------------------------------------------------------------")
    print("%.4f     \t %.2f%%   \t %.6f%%   \t %s" % (diff_thd, (1 - pct_thd) * 100.0, pct_rlt, result))
    if result == "FAIL":
        print("Max-AbsoluteError is: %s. Threshold is: %s." % (max_error, max_diff_hd))
    return result, pct_rlt


def _run_toolss_compare_np(
    name: str,
    actual_np: np.ndarray,
    expected_np: np.ndarray,
) -> Tuple[Tuple[Any, ...], str]:
    output = StringIO()
    try:
        with redirect_stdout(output):
            if toolss is not None:
                res_output = toolss.data_compare_np(
                    actual_np,
                    expected_np,
                    diff_thd=TOOLSS_DIFF_THD,
                    pct_thd=TOOLSS_PCT_THD,
                    max_diff_hd=TOOLSS_MAX_DIFF_HD,
                )
            else:
                res_output = _fallback_data_compare_np(
                    actual_np,
                    expected_np,
                    diff_thd=TOOLSS_DIFF_THD,
                    pct_thd=TOOLSS_PCT_THD,
                    max_diff_hd=TOOLSS_MAX_DIFF_HD,
                )
    except Exception as err:
        output.write(f"[WARNING] toolss 诊断失败: {name}, {err}\n")
        with redirect_stdout(output):
            res_output = _fallback_data_compare_np(
                actual_np,
                expected_np,
                diff_thd=TOOLSS_DIFF_THD,
                pct_thd=TOOLSS_PCT_THD,
                max_diff_hd=TOOLSS_MAX_DIFF_HD,
            )
    if not isinstance(res_output, tuple):
        res_output = (res_output,)
    return res_output, output.getvalue()


def _toolss_tensor_result(name: str, actual: torch.Tensor, expected: torch.Tensor) -> Tuple[Tuple[Any, ...], str]:
    actual_np = _tensor_to_toolss_numpy(actual)
    expected_np = _tensor_to_toolss_numpy(expected)
    return _run_toolss_compare_np(name, actual_np, expected_np)


def _toolss_value_result(name: str, actual: Any, expected: Any) -> Tuple[Tuple[Any, ...], str]:
    actual_np = np.array([actual], dtype=np.float32)
    expected_np = np.array([expected], dtype=np.float32)
    return _run_toolss_compare_np(name, actual_np, expected_np)


def diagnose_tensor_diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    _, detail = _toolss_tensor_result(name, actual, expected)
    print(detail, end="" if detail.endswith("\n") else "\n")
    return True


def diagnose_value_diff(name: str, actual: Any, expected: Any) -> bool:
    if not isinstance(actual, (int, float, bool)) or not isinstance(expected, (int, float, bool)):
        return False
    _, detail = _toolss_value_result(name, actual, expected)
    print(detail, end="" if detail.endswith("\n") else "\n")
    return True


def _toolss_result_ok(res_output: Tuple[Any, ...]) -> bool:
    return bool(res_output) and res_output[0] == "PASS"


def _compare_tensor_by_toolss(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> Tuple[bool, str, List[Tuple[str, Any]], List[Tuple[str, str]]]:
    if actual.shape != expected.shape:
        msg = f"{name} shape 不一致: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
        return False, msg, [(name, ("FAIL", 0.0, msg))], []
    if actual.dtype != expected.dtype:
        msg = f"{name} dtype 不一致: actual={actual.dtype}, expected={expected.dtype}"
        return False, msg, [(name, ("FAIL", 0.0, msg))], []
    res_output, detail = _toolss_tensor_result(name, actual, expected)
    ok = _toolss_result_ok(res_output)
    msg = f"{name} PASS" if ok else f"{name} toolss 结果不通过: {res_output}"
    return ok, msg, [(name, res_output)], [(name, detail)]


def _compare_value_by_toolss(
    name: str,
    actual: Any,
    expected: Any,
) -> Tuple[bool, str, List[Tuple[str, Any]], List[Tuple[str, str]]]:
    if not isinstance(actual, (int, float, bool)) or not isinstance(expected, (int, float, bool)):
        ok = actual == expected
        msg = f"{name} PASS" if ok else f"{name} 值不一致: actual={actual}, expected={expected}"
        return ok, msg, [] if ok else [(name, ("FAIL", 0.0, msg))], []
    res_output, detail = _toolss_value_result(name, actual, expected)
    ok = _toolss_result_ok(res_output)
    msg = f"{name} PASS" if ok else f"{name} toolss 结果不通过: {res_output}"
    return ok, msg, [(name, res_output)], [(name, detail)]


def _compare_feature_by_toolss(
    name: str,
    actual: Feature,
    expected: Feature,
) -> Tuple[bool, str, List[Tuple[str, Any]], List[Tuple[str, str]]]:
    ok, msg, rows, details = _compare_tensor_by_toolss(f"{name}.x", feature_main(actual), feature_main(expected))
    if not ok:
        return ok, msg, rows, details
    actual_scales = feature_scales(actual)
    expected_scales = feature_scales(expected)
    if actual_scales is None or expected_scales is None:
        if actual_scales is expected_scales:
            return True, f"{name} PASS", rows, details
        msg = f"{name} tuple/tensor 格式不一致"
        rows.append((name, ("FAIL", 0.0, msg)))
        return False, msg, rows, details
    scales_ok, scales_msg, scales_rows, scales_details = _compare_tensor_by_toolss(f"{name}.scales", actual_scales, expected_scales)
    rows.extend(scales_rows)
    details.extend(scales_details)
    if not scales_ok:
        return False, scales_msg, rows, details
    return True, f"{name} PASS", rows, details


def _compare_by_toolss_if_possible(
    name: str,
    actual: Any,
    expected: Any,
) -> Optional[Tuple[bool, str, List[Tuple[str, Any]], List[Tuple[str, str]]]]:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _compare_tensor_by_toolss(name, actual, expected)
    if is_tuple_feature(actual) and is_tuple_feature(expected):
        return _compare_feature_by_toolss(name, actual, expected)
    if isinstance(actual, (int, float, bool)) and isinstance(expected, (int, float, bool)):
        return _compare_value_by_toolss(name, actual, expected)
    return None


def exact_tensor_equal(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diagnose: bool = True,
) -> Tuple[bool, str]:
    if actual.shape != expected.shape:
        return False, f"{name} shape 不一致: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
    if actual.dtype != expected.dtype:
        return False, f"{name} dtype 不一致: actual={actual.dtype}, expected={expected.dtype}"
    actual_for_compare = tensor_for_exact_compare(actual)
    expected_for_compare = tensor_for_exact_compare(expected)
    if torch.equal(actual_for_compare, expected_for_compare):
        return True, f"{name} PASS"
    if diagnose:
        diagnose_tensor_diff(name, actual, expected)
    mismatch = (actual_for_compare != expected_for_compare).nonzero()
    first = mismatch[0].tolist() if mismatch.numel() else []
    return False, f"{name} 内容不一致，首个不一致位置: {first}"


def exact_value_equal(name: str, actual: Any, expected: Any) -> Tuple[bool, str]:
    if actual == expected:
        return True, f"{name} PASS"
    return False, f"{name} 值不一致: actual={actual}, expected={expected}"


def exact_feature_equal(name: str, actual: Feature, expected: Feature, *, diagnose: bool = True) -> Tuple[bool, str]:
    ok, msg = exact_tensor_equal(f"{name}.x", feature_main(actual), feature_main(expected), diagnose=diagnose)
    if not ok:
        return ok, msg
    actual_scales = feature_scales(actual)
    expected_scales = feature_scales(expected)
    if actual_scales is None or expected_scales is None:
        if actual_scales is expected_scales:
            return True, f"{name} PASS"
        return False, f"{name} tuple/tensor 格式不一致"
    return exact_tensor_equal(f"{name}.scales", actual_scales, expected_scales, diagnose=diagnose)


def _diagnose_tensor_with_header(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    print(f"[INFO] {name} toolss detail:")
    return diagnose_tensor_diff(name, actual, expected)


def _record_compare(
    records: List[CompareRecord],
    result: CompareResult,
    key: str,
    name: str,
    ok: bool,
    msg: str,
    actual: Any = None,
    expected: Any = None,
    *,
    rank: int,
    item: str,
    order: int,
) -> None:
    compare_result = _compare_by_toolss_if_possible(name, actual, expected)
    csv_rows: List[Tuple[str, Any]] = []
    toolss_details: List[Tuple[str, str]] = []
    if compare_result is not None:
        ok, msg, csv_rows, toolss_details = compare_result
        result.csv_rows.extend(csv_rows)
    elif not ok:
        csv_rows = [(name, ("FAIL", 0.0, msg))]
        result.csv_rows.extend(csv_rows)
    records.append(CompareRecord(key, name, rank, item, order, ok, msg, actual, expected, csv_rows, toolss_details))
    result[key] = ok


def _format_compare_record(record: CompareRecord, config: CompareConfig) -> str:
    if record.ok:
        return f"[INFO] {record.name} PASSED! toolss detail follows."
    reason = _strip_compare_name(record.name, record.msg)
    return (
        f"[ERROR] {record.name} FAILED! reason={reason}; "
        f"actual={_compare_object_detail(record.actual, config)}; "
        f"expected={_compare_object_detail(record.expected, config)}"
    )


def _print_compare_record_detail(record: CompareRecord, config: CompareConfig) -> None:
    has_toolss_output = False
    if record.csv_rows:
        has_toolss_output = True
        for csv_name, res_output in record.csv_rows:
            print(f"[INFO] {csv_name} toolss result: {res_output}")
    if record.toolss_details:
        has_toolss_output = True
        for detail_name, detail in record.toolss_details:
            print(f"[INFO] {detail_name} toolss detail:")
            if detail:
                print(detail, end="" if detail.endswith("\n") else "\n")
            else:
                print("[INFO] toolss detail empty")
    if has_toolss_output:
        return
    print(
        f"[INFO] {record.name} toolss detail skipped: "
        f"actual={_brief_compare_object_detail(record.actual)}, "
        f"expected={_brief_compare_object_detail(record.expected)}"
    )


def _print_compare_records(tag: str, records: List[CompareRecord], config: CompareConfig) -> None:
    print(f"\n################## {tag} compare detail ##################")
    last_rank = None
    sorted_records = _sorted_records(records)
    for record in sorted_records:
        if record.rank != last_rank:
            print(f"\n---------- card rank{record.rank} ----------")
            last_rank = record.rank
        print(_format_compare_record(record, config))
        _print_compare_record_detail(record, config)
    _print_compare_table(tag, sorted_records)


def _print_compare_table(tag: str, records: List[CompareRecord]) -> None:
    print(f"\n################## {tag} compare table ##################")
    rows = [["card", "item", "result"]]
    rank_groups: List[List[CompareRecord]] = []
    for record in records:
        if not rank_groups or rank_groups[-1][0].rank != record.rank:
            rank_groups.append([])
        rank_groups[-1].append(record)
    rank_start_indices = set()
    for group in rank_groups:
        rank_start_indices.add(len(rows))
        middle_index = len(group) // 2
        for index, record in enumerate(group):
            card = f"rank{record.rank}" if index == middle_index else ""
            rows.append([card, record.item, _colored_status(record)])
    widths = [max(_visible_len(row[index]) for row in rows) for index in range(len(rows[0]))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    for index, row in enumerate(rows):
        if index in rank_start_indices and index != 1:
            print(border)
        cells = [_center_table_cell(row[0], widths[0])]
        cells.extend(_pad_table_cell(row[col], widths[col]) for col in range(1, len(row)))
        print("| " + " | ".join(cells) + " |")
        if index == 0:
            print(border)
    print(border)


def compare_dispatch_outputs(
    rank_outputs: List[Dict[str, Any]],
    case_data: Dict[str, Any],
    config: CompareConfig,
    tag: str = "dispatch",
) -> Dict[str, bool]:
    result = CompareResult()
    records: List[CompareRecord] = []
    x_per_rank = case_data["x_per_rank"]
    topk_weights_per_rank = (
        case_data["topk_weights_per_rank"]
        if config.dispatch_with_topk_weights
        else None
    )
    expected_goldens = case_data["dispatch_goldens"][config.server_ranks]

    for local_rank_id, output in enumerate(rank_outputs):
        global_rank = config.server_index * config.rank_per_dev + local_rank_id
        expected = expected_goldens[local_rank_id]
        actual_handle: GoldenHandle = output["handle"]
        valid_rows = _valid_rows(actual_handle)

        expected_x_by_meta, expected_w_by_meta = expected_dispatch_rows_from_metadata(
            x_per_rank,
            topk_weights_per_rank,
            actual_handle.recv_src_metadata[:valid_rows],
        )
        actual_recv_x = output["recv_x"]
        if config.check_invalid_padding:
            actual_x_for_compare = actual_recv_x
            expected_x_for_compare = expected.recv_x
        elif is_tuple_feature(actual_recv_x):
            actual_x_for_compare = (
                feature_main(actual_recv_x)[:valid_rows],
                feature_scales(actual_recv_x)[:valid_rows],
            )
            expected_x_for_compare = expected_x_by_meta
        else:
            actual_x_for_compare = actual_recv_x[:valid_rows]
            expected_x_for_compare = expected_x_by_meta
        ok, msg = exact_feature_equal(
            f"{tag}.recv_x.rank{global_rank}",
            actual_x_for_compare,
            expected_x_for_compare,
            diagnose=False,
        )
        _record_compare(
            records,
            result,
            _result_key(tag, "recv_x", global_rank),
            f"{tag}.recv_x.rank{global_rank}",
            ok,
            msg,
            actual_x_for_compare,
            expected_x_for_compare,
            rank=global_rank,
            item="recv_x",
            order=10,
        )

        recv_topk_idx_ok = output["recv_topk_idx"] is None
        _record_compare(
            records,
            result,
            _result_key(tag, "recv_topk_idx", global_rank),
            f"{tag}.recv_topk_idx.rank{global_rank}",
            recv_topk_idx_ok,
            "训练接口期望 None",
            output["recv_topk_idx"],
            None,
            rank=global_rank,
            item="recv_topk_idx",
            order=20,
        )

        if config.dispatch_with_handle:
            _record_compare(
                records,
                result,
                _result_key(tag, "recv_topk_weights", global_rank),
                f"{tag}.recv_topk_weights.rank{global_rank}",
                True,
                "cache 模式不比较",
                None,
                None,
                rank=global_rank,
                item="recv_topk_weights",
                order=30,
            )
        elif expected_w_by_meta is None:
            recv_topk_weights_ok = output["recv_topk_weights"] is None
            _record_compare(
                records,
                result,
                _result_key(tag, "recv_topk_weights", global_rank),
                f"{tag}.recv_topk_weights.rank{global_rank}",
                recv_topk_weights_ok,
                "当前分支期望 None",
                output["recv_topk_weights"],
                None,
                rank=global_rank,
                item="recv_topk_weights",
                order=30,
            )
        elif output["recv_topk_weights"] is None:
            _record_compare(
                records,
                result,
                _result_key(tag, "recv_topk_weights", global_rank),
                f"{tag}.recv_topk_weights.rank{global_rank}",
                False,
                "实际为 None",
                None,
                expected_w_by_meta,
                rank=global_rank,
                item="recv_topk_weights",
                order=30,
            )
        else:
            ok, msg = exact_tensor_equal(
                f"{tag}.recv_topk_weights.rank{global_rank}",
                output["recv_topk_weights"][:valid_rows],
                expected_w_by_meta,
                diagnose=False,
            )
            _record_compare(
                records,
                result,
                _result_key(tag, "recv_topk_weights", global_rank),
                f"{tag}.recv_topk_weights.rank{global_rank}",
                ok,
                msg,
                output["recv_topk_weights"][:valid_rows],
                expected_w_by_meta,
                rank=global_rank,
                item="recv_topk_weights",
                order=30,
            )

        handle_pairs = []
        if config.dispatch_with_handle:
            handle_pairs.append(
                ("dst_buffer_slot_idx", actual_handle.dst_buffer_slot_idx, expected.handle.dst_buffer_slot_idx, 40)
            )
        handle_pairs.extend([
            ("recv_src_metadata", None, None, 50),
            (
                "num_recv_payloads_per_rank",
                actual_handle.num_recv_payloads_per_rank,
                expected.handle.num_recv_payloads_per_rank,
                60,
            ),
            (
                "num_recv_tokens_per_expert",
                actual_handle.num_recv_tokens_per_expert,
                expected.handle.num_recv_tokens_per_expert,
                70,
            ),
            ("topk_idx", actual_handle.topk_idx, expected.handle.topk_idx, 80),
        ])
        for name, actual, exp, order in handle_pairs:
            if name == "recv_src_metadata":
                check_slot_idx = config.dispatch_with_handle
                if config.strict_dispatch_metadata_order:
                    actual = _metadata_compare_view(actual_handle.recv_src_metadata, valid_rows, check_slot_idx)
                    exp = _metadata_compare_view(expected.handle.recv_src_metadata, valid_rows, check_slot_idx)
                else:
                    actual = _sort_metadata(actual_handle.recv_src_metadata, valid_rows, check_slot_idx)
                    exp = _sort_metadata(expected.handle.recv_src_metadata, _valid_rows(expected.handle), check_slot_idx)
                ok, msg = exact_tensor_equal(f"{tag}.handle.{name}.rank{global_rank}", actual, exp, diagnose=False)
            else:
                ok, msg = exact_tensor_equal(f"{tag}.handle.{name}.rank{global_rank}", actual, exp, diagnose=False)
            _record_compare(
                records,
                result,
                _result_key(tag, f"handle.{name}", global_rank),
                f"{tag}.handle.{name}.rank{global_rank}",
                ok,
                msg,
                actual,
                exp,
                rank=global_rank,
                item=f"handle.{name}",
                order=order,
            )

        handle_value_pairs = [
            ("num_experts", actual_handle.num_experts, expected.handle.num_experts, 90),
            ("expert_alignment", actual_handle.expert_alignment, expected.handle.expert_alignment, 100),
            (
                "num_max_tokens_per_rank",
                actual_handle.num_max_tokens_per_rank,
                expected.handle.num_max_tokens_per_rank,
                110,
            ),
            ("num_expanded_tokens", actual_handle.num_expanded_tokens, expected.handle.num_expanded_tokens, 120),
        ]
        for name, actual, exp, order in handle_value_pairs:
            ok, msg = exact_value_equal(f"{tag}.handle.{name}.rank{global_rank}", actual, exp)
            _record_compare(
                records,
                result,
                _result_key(tag, f"handle.{name}", global_rank),
                f"{tag}.handle.{name}.rank{global_rank}",
                ok,
                msg,
                actual,
                exp,
                rank=global_rank,
                item=f"handle.{name}",
                order=order,
            )

    _print_compare_records(tag, records, config)
    return result


def compare_combine_outputs(
    rank_outputs: List[Dict[str, Any]],
    expected_goldens: List[CombineGolden],
    tag: str,
    config: CompareConfig,
) -> Dict[str, bool]:
    result = CompareResult()
    records: List[CompareRecord] = []
    expected_host = expected_goldens[config.server_ranks]
    for local_rank_id, output in enumerate(rank_outputs):
        global_rank = config.server_index * config.rank_per_dev + local_rank_id
        expected = expected_host[local_rank_id]
        ok, msg = exact_feature_equal(
            f"{tag}.combined_x.rank{global_rank}",
            output["combined_x"],
            expected.combined_x,
            diagnose=False,
        )
        _record_compare(
            records,
            result,
            _result_key(tag, "combined_x", global_rank),
            f"{tag}.combined_x.rank{global_rank}",
            ok,
            msg,
            output["combined_x"],
            expected.combined_x,
            rank=global_rank,
            item="combined_x",
            order=10,
        )
        if expected.combined_topk_weights is not None:
            ok, msg = exact_tensor_equal(
                f"{tag}.combined_topk_weights.rank{global_rank}",
                output["combined_topk_weights"],
                expected.combined_topk_weights,
                diagnose=False,
            )
            _record_compare(
                records,
                result,
                _result_key(tag, "combined_topk_weights", global_rank),
                f"{tag}.combined_topk_weights.rank{global_rank}",
                ok,
                msg,
                output["combined_topk_weights"],
                expected.combined_topk_weights,
                rank=global_rank,
                item="combined_topk_weights",
                order=20,
            )
        else:
            ok = output["combined_topk_weights"] is None
            _record_compare(
                records,
                result,
                _result_key(tag, "combined_topk_weights", global_rank),
                f"{tag}.combined_topk_weights.rank{global_rank}",
                ok,
                "当前分支期望 None",
                output["combined_topk_weights"],
                None,
                rank=global_rank,
                item="combined_topk_weights",
                order=20,
            )
    _print_compare_records(tag, records, config)
    return result


def print_summary(tag: str, result: Dict[str, bool]) -> None:
    print(f"\n################## {tag} result ##################")
    failed_items = [name for name, ok in result.items() if not ok]
    print(f"[INFO] {tag} summary: total={len(result)}, passed={len(result) - len(failed_items)}, failed={len(failed_items)}")
    if not failed_items:
        print(f"[INFO] {tag} PASSED!")
    else:
        print(f"[ERROR] {tag} FAILED! failed_items={failed_items}")

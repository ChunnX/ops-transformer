#!/usr/bin/env python3
"""Export batch profiling results to an Excel workbook.

The batch runner writes profiler traces but intentionally does not calculate
performance at the end of the run. This script performs that calculation
offline and keeps the original profiling files untouched.

Examples:
    python3 batch_run_once/export_batch_performance.py
    python3 batch_run_once/export_batch_performance.py \
        --profiling-root ./Profiling_Result_mc2_cascade_16p \
        --result-dir ./csv_result
"""

import argparse
import csv
import math
import sys
from copy import copy
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan kernel_details CSV files produced by mega_moe_once.py and "
            "export the same tail-50 performance statistics to an xlsx file."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=None,
        help=(
            "Profiling result root. Default: first existing "
            "Profiling_Result_mc2_cascade_16p under cwd, this script's "
            "directory, or the project root."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing mega_moe_result.csv and receiving the xlsx. "
            "Default: csv_result under the profiling root's parent."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output xlsx path. Default: <result-dir>/batch_performance_result.xlsx.",
    )
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=None,
        help=(
            "Optional mega_moe_result.csv used to associate profiling traces "
            "with case names. Default: <result-dir>/mega_moe_result.csv."
        ),
    )
    parser.add_argument(
        "--tail-size",
        type=int,
        default=50,
        help="Number of final timeline samples used per rank. Default: 50.",
    )
    parser.add_argument(
        "--tail-skip",
        type=int,
        default=2,
        help="Number of final profiler-boundary samples skipped per rank before tail-size. Default: 2.",
    )
    parser.add_argument(
        "--case-match-window-seconds",
        type=float,
        default=1800.0,
        help=(
            "Maximum mtime-to-result timestamp distance for case association. "
            "Default: 1800 seconds."
        ),
    )
    parser.add_argument(
        "--rank-per-dev",
        type=int,
        default=0,
        help=(
            "Ranks on this server. Default: infer from discovered Device_id "
            "values. Set explicitly for multi-server profiling."
        ),
    )
    parser.add_argument(
        "--server-index",
        type=int,
        default=0,
        help="Server index used by the batch run. Default: 0.",
    )
    return parser.parse_args()


def find_default_profiling_root():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "Profiling_Result_mc2_cascade_16p",
        script_dir / "Profiling_Result_mc2_cascade_16p",
        script_dir.parent / "Profiling_Result_mc2_cascade_16p",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_result_events(result_csv):
    if not result_csv.is_file():
        return []

    events = []
    with result_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if len(row) < 4:
                continue
            case_name = row[0].strip()
            if not case_name or case_name == "case_name":
                continue
            try:
                record_time = datetime.strptime(row[3].strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            events.append((record_time.timestamp(), case_name))
    return sorted(events)


def find_profile_session(profile_csv_path, profiling_root):
    relative_path = profile_csv_path.relative_to(profiling_root)
    parts = relative_path.parts
    if "ASCEND_PROFILER_OUTPUT" in parts:
        index = parts.index("ASCEND_PROFILER_OUTPUT")
        session_parts = parts[:index]
        return str(Path(*session_parts)) if session_parts else "."
    return str(relative_path.parent)


def match_case_name(file_mtime, result_events, max_delta_seconds):
    if not result_events:
        return "", "unmatched", math.nan

    event_time, case_name = min(
        result_events,
        key=lambda event: abs(event[0] - file_mtime),
    )
    delta_seconds = abs(event_time - file_mtime)
    if delta_seconds > max_delta_seconds:
        return "", "unmatched", delta_seconds
    return case_name, "mega_moe_result.csv", delta_seconds


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def get_global_rank_id(local_rank_id, rank_per_dev, server_index):
    global_rank_id = rank_per_dev * server_index + local_rank_id
    return global_rank_id


def make_unmapped_case_name(profile_session):
    return f"session:{profile_session}"


def set_column_widths(worksheet, dataframe):
    for column_index, column_name in enumerate(dataframe.columns, start=1):
        values = dataframe[column_name].fillna("").astype(str)
        value_width = max((len(value) for value in values.head(500)), default=0)
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=column_index).column_letter
        ].width = min(max(len(str(column_name)) + 2, value_width + 2), 48)


def write_dataframe(writer, sheet_name, dataframe):
    dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.book[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        font = copy(cell.font)
        font.bold = True
        font.color = "FFFFFF"
        cell.font = font
        fill = copy(cell.fill)
        fill.fill_type = "solid"
        fill.fgColor = "1F4E78"
        cell.fill = fill
    set_column_widths(worksheet, dataframe)


TIMELINE_SMALL_KIND = "small_ops"
TIMELINE_MEGA_KIND = "mega_moe"

TIMELINE_OP_METRICS = {
    "small_ops_wall": ("小算子真实耗时(wall)", "wall_span_us", TIMELINE_SMALL_KIND),
    "small_ops_resource": ("小算子耗时(sum)", "resource_sum_us", TIMELINE_SMALL_KIND),
    "communication": ("通信耗时", "communication_us", TIMELINE_SMALL_KIND),
    "moe_expert": ("MOE专家耗时", "moe_expert_us", TIMELINE_SMALL_KIND),
    "shared_expert": ("共享专家耗时", "shared_expert_us", TIMELINE_SMALL_KIND),
    "Add": ("Add耗时", "add_us", TIMELINE_SMALL_KIND),
    "MegaMoe": ("融合MegaMoe", "duration_us", TIMELINE_MEGA_KIND),
}

TIMELINE_SUMMARY_COLUMNS = [
    "case_name", "case_source", "op_type", "display_name", "rank_count",
    "sample_count", "average_us", "min_us", "min_of_medians_us",
    "max_of_medians_us", "median_of_medians_us", "min_of_mins_us",
]

TIMELINE_RAW_COLUMNS = [
    "case_name", "case_source", "case_match_delta_seconds", "profile_session",
    "kernel_details_file", "file_mtime", "local_rank_id", "global_rank_id",
    "timeline_kind", "iteration_id", "wall_span_us", "resource_sum_us",
    "communication_us", "moe_expert_us", "shared_expert_us", "add_us",
    "overlap_saved_us", "duration_us", "op_count",
]

TIMELINE_FILE_COLUMNS = [
    "case_name", "case_source", "case_match_delta_seconds", "profile_session",
    "kernel_details_file", "file_mtime", "local_rank_id", "global_rank_id",
]


def timeline_to_float_series(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def timeline_text_has(row, *keywords):
    text = f"{row.get('Type', '')} {row.get('Name', '')}".lower()
    return any(keyword.lower() in text for keyword in keywords)


def read_kernel_details(kernel_details_path):
    kernel_details = pd.read_csv(kernel_details_path)
    kernel_details.rename(columns=lambda column: str(column).strip(), inplace=True)
    required_columns = {"Type", "Start Time(us)", "Duration(us)"}
    if not required_columns.issubset(kernel_details.columns):
        return pd.DataFrame()
    kernel_details["start_us"] = timeline_to_float_series(kernel_details["Start Time(us)"])
    kernel_details["duration_us"] = timeline_to_float_series(kernel_details["Duration(us)"])
    kernel_details["end_us"] = kernel_details["start_us"] + kernel_details["duration_us"]
    if "Stream ID" in kernel_details.columns:
        kernel_details["stream_id"] = pd.to_numeric(kernel_details["Stream ID"], errors="coerce").fillna(-1).astype(int)
    else:
        kernel_details["stream_id"] = -1
    if "Device_id" in kernel_details.columns:
        kernel_details["local_rank_id"] = pd.to_numeric(kernel_details["Device_id"], errors="coerce").fillna(-1).astype(int)
    else:
        kernel_details["local_rank_id"] = -1
    return kernel_details.dropna(subset=["start_us", "duration_us", "end_us"])


def get_timeline_moe_streams(kernel_details):
    route_mask = kernel_details.apply(lambda row: timeline_text_has(row, "MoeInitRouting", "MoeFinalizeRouting"), axis=1)
    return set(kernel_details.loc[route_mask, "stream_id"].dropna().astype(int).tolist())


def classify_timeline_row(row, moe_streams):
    if timeline_text_has(row, "MegaMoe"):
        return "MegaMoe"
    if timeline_text_has(row, "allGather", "reduceScatter"):
        return "communication"
    if timeline_text_has(row, "Add"):
        return "Add"
    if timeline_text_has(row, "MoeInitRouting", "MoeFinalizeRouting"):
        return "moe_expert"
    if timeline_text_has(row, "GroupedMatmul", "ClippedSwiglu", "DynamicMxQuant", "AntiMxQuant", "DynamicQuant"):
        if not moe_streams:
            return "moe_expert"
        return "moe_expert" if int(row["stream_id"]) in moe_streams else "shared_expert"
    if timeline_text_has(row, "ReduceSum", "Tile", "Fill"):
        return "shared_expert"
    if timeline_text_has(row, "MoeDistributeDispatch", "MoeDistributeCombine"):
        return "communication"
    return None


def is_timeline_allgather(row):
    return row.get("timeline_group") == "communication" and timeline_text_has(row, "allGather")


def is_dispatch_mx_quant_row(row):
    return (
        row.get("timeline_group") == "moe_expert"
        and timeline_text_has(row, "DynamicMxQuant")
    )


def trim_segment_to_real_combine(segment):
    if not (segment["timeline_group"] == "shared_expert").any():
        return segment[segment["timeline_group"] != "Add"]

    combine_base_mask = segment.apply(
        lambda row: timeline_text_has(row, "reduceScatter", "MoeFinalizeRouting", "MoeDistributeCombine"),
        axis=1,
    )
    if combine_base_mask.any():
        combine_base_indices = segment.index[combine_base_mask].tolist()
        search_start = combine_base_indices[-1]
        add_rows = segment[
            (segment.index > search_start)
            & (segment["timeline_group"] == "Add")
        ]
    else:
        add_rows = segment[segment["timeline_group"] == "Add"]

    if add_rows.empty:
        return segment

    combine_add_end_us = float(add_rows.iloc[0]["end_us"])
    return segment[segment["start_us"] < combine_add_end_us]


def is_combine_comm_quant_row(row):
    return (
        row.get("timeline_group") == "moe_expert"
        and timeline_text_has(row, "DynamicMxQuant", "AntiMxQuant")
    )


def drop_combine_comm_quant_rows(segment):
    combine_base_mask = segment.apply(
        lambda row: timeline_text_has(row, "MoeFinalizeRouting", "MoeDistributeCombine"),
        axis=1,
    )
    combine_base_indices = segment.index[combine_base_mask].tolist()
    if not combine_base_indices:
        return segment, 0.0

    drop_indices = set()
    for combine_base_index in combine_base_indices:
        gmm_mask = (
            (segment.index < combine_base_index)
            & segment.apply(lambda row: timeline_text_has(row, "GroupedMatmul"), axis=1)
        )
        gmm_indices = segment.index[gmm_mask].tolist()
        if not gmm_indices:
            continue

        reduce_mask = (
            (segment.index > combine_base_index)
            & segment.apply(lambda row: timeline_text_has(row, "reduceScatter", "MoeDistributeCombine"), axis=1)
        )
        reduce_indices = segment.index[reduce_mask].tolist()
        start_index = gmm_indices[-1]
        end_index = reduce_indices[0] if reduce_indices else segment.index[-1] + 1
        quant_mask = (
            (segment.index > start_index)
            & (segment.index < end_index)
            & segment.apply(is_combine_comm_quant_row, axis=1)
        )
        drop_indices.update(segment.index[quant_mask].tolist())

    if not drop_indices:
        return segment, 0.0

    drop_rows = segment.loc[sorted(drop_indices)]
    excluded_wall_us = float(drop_rows["duration_us"].sum())
    return segment.drop(index=sorted(drop_indices)), excluded_wall_us


def extract_small_iterations(kernel_details):
    moe_streams = get_timeline_moe_streams(kernel_details)
    timeline = kernel_details.copy()
    timeline["timeline_group"] = timeline.apply(lambda row: classify_timeline_row(row, moe_streams), axis=1)
    timeline = timeline[timeline["timeline_group"].notna() & (timeline["timeline_group"] != "MegaMoe")]
    timeline = timeline.sort_values(["start_us", "end_us"], kind="stable").reset_index(drop=True)
    if timeline.empty:
        return []

    anchor_positions = [
        index for index, row in timeline.iterrows()
        if timeline_text_has(row, "MoeInitRouting", "MoeDistributeDispatch")
    ]
    if not anchor_positions:
        anchor_positions = [0]

    start_positions = []
    previous_anchor_position = -1
    for anchor_position in anchor_positions:
        search_start = previous_anchor_position + 1
        allgather_positions = [
            index for index in range(search_start, anchor_position)
            if is_timeline_allgather(timeline.iloc[index])
        ]
        dispatch_quant_positions = [
            index for index in range(search_start, anchor_position)
            if is_dispatch_mx_quant_row(timeline.iloc[index])
        ]
        if allgather_positions:
            first_allgather_position = allgather_positions[0]
            quant_before_allgather = [
                index for index in dispatch_quant_positions
                if index < first_allgather_position
            ]
            start_position = (
                quant_before_allgather[-1]
                if quant_before_allgather
                else first_allgather_position
            )
        elif dispatch_quant_positions:
            start_position = dispatch_quant_positions[-1]
        else:
            continue
        if start_positions and start_position <= start_positions[-1]:
            continue
        start_positions.append(start_position)
        previous_anchor_position = anchor_position

    if not start_positions:
        start_positions = [anchor_positions[0]]

    rows = []
    local_rank_id = int(timeline["local_rank_id"].dropna().iloc[0]) if not timeline["local_rank_id"].dropna().empty else -1
    for iteration_id, start_position in enumerate(start_positions):
        end_position = start_positions[iteration_id + 1] if iteration_id + 1 < len(start_positions) else len(timeline)
        segment = timeline.iloc[start_position:end_position]
        if segment.empty:
            continue
        segment = trim_segment_to_real_combine(segment)
        if segment.empty:
            continue
        segment, excluded_wall_us = drop_combine_comm_quant_rows(segment)
        if segment.empty:
            continue
        duration_by_group = segment.groupby("timeline_group")["duration_us"].sum().to_dict()
        resource_sum_us = float(segment["duration_us"].sum())
        raw_wall_span_us = float(segment["end_us"].max() - segment["start_us"].min())
        wall_span_us = max(0.0, raw_wall_span_us - excluded_wall_us)
        rows.append({
            "local_rank_id": local_rank_id,
            "timeline_kind": TIMELINE_SMALL_KIND,
            "iteration_id": iteration_id,
            "wall_span_us": wall_span_us,
            "resource_sum_us": resource_sum_us,
            "communication_us": float(duration_by_group.get("communication", 0.0)),
            "moe_expert_us": float(duration_by_group.get("moe_expert", 0.0)),
            "shared_expert_us": float(duration_by_group.get("shared_expert", 0.0)),
            "add_us": float(duration_by_group.get("Add", 0.0)),
            "overlap_saved_us": max(0.0, resource_sum_us - wall_span_us),
            "duration_us": math.nan,
            "op_count": int(len(segment)),
        })
    return rows


def extract_mega_moe_iterations(kernel_details):
    rows = []
    mega_rows = kernel_details[kernel_details.apply(lambda row: timeline_text_has(row, "MegaMoe"), axis=1)]
    if mega_rows.empty:
        return rows
    for iteration_id, row in mega_rows.sort_values(["start_us", "end_us"], kind="stable").reset_index(drop=True).iterrows():
        rows.append({
            "local_rank_id": int(row.get("local_rank_id", -1)),
            "timeline_kind": TIMELINE_MEGA_KIND,
            "iteration_id": int(iteration_id),
            "wall_span_us": math.nan,
            "resource_sum_us": math.nan,
            "communication_us": math.nan,
            "moe_expert_us": math.nan,
            "shared_expert_us": math.nan,
            "add_us": math.nan,
            "overlap_saved_us": math.nan,
            "duration_us": float(row["duration_us"]),
            "op_count": 1,
        })
    return rows


def collect_timeline_rows(profiling_root, result_events, max_delta_seconds, rank_per_dev, server_index):
    timeline_rows = []
    file_rows = []
    warnings = []
    kernel_detail_paths = sorted(profiling_root.rglob("kernel_details*.csv"), key=lambda path: path.stat().st_mtime)
    if not kernel_detail_paths:
        return timeline_rows, file_rows, warnings

    parsed_files = []
    device_ids = []
    for kernel_details_path in kernel_detail_paths:
        try:
            kernel_details = read_kernel_details(kernel_details_path)
        except Exception as exc:
            warnings.append({"level": "warning", "source": str(kernel_details_path), "message": f"Failed to read kernel_details: {exc}"})
            continue
        if kernel_details.empty:
            warnings.append({"level": "warning", "source": str(kernel_details_path), "message": "No usable kernel timeline rows."})
            continue
        local_rank_id = to_int(kernel_details["local_rank_id"].iloc[0])
        if local_rank_id >= 0:
            device_ids.append(local_rank_id)
        parsed_files.append((kernel_details_path, kernel_details, local_rank_id))

    if not parsed_files:
        return timeline_rows, file_rows, warnings
    if rank_per_dev <= 0:
        rank_per_dev = max(device_ids) + 1 if device_ids else 1

    for kernel_details_path, kernel_details, local_rank_id in parsed_files:
        file_mtime = kernel_details_path.stat().st_mtime
        profile_session = find_profile_session(kernel_details_path, profiling_root)
        case_name, case_source, match_delta_seconds = match_case_name(file_mtime, result_events, max_delta_seconds)
        if not case_name:
            case_name = make_unmapped_case_name(profile_session)
        global_rank_id = get_global_rank_id(local_rank_id, rank_per_dev, server_index)
        relative_path = str(kernel_details_path.relative_to(profiling_root))
        file_row = {
            "case_name": case_name,
            "case_source": case_source,
            "case_match_delta_seconds": match_delta_seconds,
            "profile_session": profile_session,
            "kernel_details_file": relative_path,
            "file_mtime": datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "local_rank_id": local_rank_id,
            "global_rank_id": global_rank_id,
        }
        file_rows.append(file_row)
        for row in extract_small_iterations(kernel_details) + extract_mega_moe_iterations(kernel_details):
            row.update(file_row)
            timeline_rows.append(row)
    return timeline_rows, file_rows, warnings


def build_timeline_summary(timeline_df, discovered_cases, tail_size, tail_skip):
    rows = []
    for case_name, case_source in discovered_cases:
        case_df = timeline_df[(timeline_df["case_name"] == case_name) & (timeline_df["case_source"] == case_source)]
        for op_type, (display_name, metric_name, timeline_kind) in TIMELINE_OP_METRICS.items():
            metric_df = case_df[case_df["timeline_kind"] == timeline_kind]
            per_rank_medians = []
            per_rank_mins = []
            all_tail_values = []
            for _, rank_rows in metric_df.groupby("global_rank_id", dropna=False):
                values = pd.to_numeric(rank_rows[metric_name], errors="coerce").dropna()
                if tail_skip > 0 and len(values) > tail_skip:
                    values = values.iloc[:-tail_skip]
                values = values.tail(tail_size)
                if values.empty:
                    continue
                per_rank_medians.append(float(values.median()))
                per_rank_mins.append(float(values.min()))
                all_tail_values.extend(values.tolist())
            row = {
                "case_name": case_name,
                "case_source": case_source,
                "op_type": op_type,
                "display_name": display_name,
            }
            if per_rank_medians:
                row.update({
                    "rank_count": len(per_rank_medians),
                    "sample_count": len(all_tail_values),
                    "average_us": sum(all_tail_values) / len(all_tail_values),
                    "min_us": min(all_tail_values),
                    "min_of_medians_us": min(per_rank_medians),
                    "max_of_medians_us": max(per_rank_medians),
                    "median_of_medians_us": pd.Series(per_rank_medians).median(),
                    "min_of_mins_us": min(per_rank_mins),
                })
            rows.append(row)
    return pd.DataFrame(rows, columns=TIMELINE_SUMMARY_COLUMNS)


def build_timeline_missing_warnings(summary_df):
    if summary_df.empty:
        return []
    missing = summary_df[summary_df["rank_count"].isna()]
    return [
        {"level": "warning", "source": row.case_name, "message": f"No timeline samples for {row.op_type}."}
        for row in missing.itertuples(index=False)
    ]


def export_xlsx(output_path, summary_df, raw_df, file_df, warning_df, metadata_df):
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            write_dataframe(writer, "Summary", summary_df)
            write_dataframe(writer, "PerIteration", raw_df)
            write_dataframe(writer, "ProfileFiles", file_df)
            write_dataframe(writer, "Warnings", warning_df)
            write_dataframe(writer, "Metadata", metadata_df)
    except ImportError as exc:
        raise RuntimeError("Writing xlsx requires openpyxl. Install it with: pip install openpyxl") from exc


def main():
    args = parse_args()
    if args.tail_size <= 0:
        raise ValueError("--tail-size must be greater than zero.")
    if args.tail_skip < 0:
        raise ValueError("--tail-skip must be greater than or equal to zero.")
    profiling_root = args.profiling_root.expanduser().resolve() if args.profiling_root else find_default_profiling_root().resolve()
    if not profiling_root.is_dir():
        print(f"[ERROR] Profiling directory not found: {profiling_root}", file=sys.stderr)
        return 2

    result_dir = args.result_dir.expanduser().resolve() if args.result_dir else (profiling_root.parent / "csv_result").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.result_csv.expanduser().resolve() if args.result_csv else result_dir / "mega_moe_result.csv"
    output_path = args.output.expanduser().resolve() if args.output else result_dir / "batch_performance_result.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_events = parse_result_events(result_csv)
    raw_rows, file_rows, warnings = collect_timeline_rows(
        profiling_root=profiling_root,
        result_events=result_events,
        max_delta_seconds=args.case_match_window_seconds,
        rank_per_dev=args.rank_per_dev,
        server_index=args.server_index,
    )
    if not file_rows:
        print(f"[ERROR] No readable kernel_details*.csv found under: {profiling_root}", file=sys.stderr)
        return 2

    raw_df = pd.DataFrame(raw_rows, columns=TIMELINE_RAW_COLUMNS)
    file_df = pd.DataFrame(file_rows, columns=TIMELINE_FILE_COLUMNS)
    discovered_cases = list(file_df[["case_name", "case_source"]].drop_duplicates().itertuples(index=False, name=None))
    summary_df = build_timeline_summary(raw_df, discovered_cases, args.tail_size, args.tail_skip)
    warnings.extend(build_timeline_missing_warnings(summary_df))
    warning_df = pd.DataFrame(warnings, columns=["level", "source", "message"])
    metadata_df = pd.DataFrame([
        {"key": "profiling_root", "value": str(profiling_root)},
        {"key": "result_csv", "value": str(result_csv)},
        {"key": "result_csv_events", "value": len(result_events)},
        {"key": "tail_size", "value": args.tail_size},
        {"key": "tail_skip", "value": args.tail_skip},
        {"key": "case_match_window_seconds", "value": args.case_match_window_seconds},
        {"key": "kernel_details_files", "value": len(file_df)},
        {"key": "timeline_rows", "value": len(raw_df)},
        {"key": "summary_rows", "value": len(summary_df)},
        {"key": "generated_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    ])
    export_xlsx(output_path, summary_df, raw_df, file_df, warning_df, metadata_df)

    matched_files = int((file_df["case_source"] != "unmatched").sum())
    print(f"[INFO] 已扫描 {len(file_df)} 个 kernel_details 文件。")
    print(f"[INFO] 已匹配 case_name 的文件数: {matched_files}/{len(file_df)}。")
    per_case_summary = file_df.groupby(["case_name", "case_source"], dropna=False).size().rename("kernel_details_files").reset_index()
    per_case_rows = (
        raw_df.groupby(["case_name", "case_source"], dropna=False).size().rename("timeline_rows").reset_index()
        if not raw_df.empty
        else pd.DataFrame(columns=["case_name", "case_source", "timeline_rows"])
    )
    per_case_summary = per_case_summary.merge(per_case_rows, on=["case_name", "case_source"], how="left").fillna({"timeline_rows": 0})
    for case_info in per_case_summary.itertuples(index=False):
        print(
            f"[INFO] case_name={case_info.case_name}, "
            f"kernel_details_files={case_info.kernel_details_files}, "
            f"timeline_rows={int(case_info.timeline_rows)}, "
            f"source={case_info.case_source}"
        )
    print(f"[INFO] 性能汇总已写入: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

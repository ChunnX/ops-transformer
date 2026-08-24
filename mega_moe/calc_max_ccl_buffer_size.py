#!/usr/bin/env python3
"""Find the Excel case requiring the largest A5 MegaMoe CCL buffer."""

import argparse
import math
from pathlib import Path

import pandas as pd


ALIGN_32 = 32
ALIGN_256 = 256
ALIGN_512 = 512
MIB = 1024 * 1024
PEERMEM_DATA_OFFSET = 60 * 1024


def ceil_align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def calc_a5_ccl_buffer_size(
    ep_world_size: int,
    moe_expert_num: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
) -> dict[str, int]:
    """Mirror CalcHalfBufferSizeMBA5 in torch extension mega_moe.cpp."""
    if not 2 <= ep_world_size <= 1024:
        raise ValueError(f"ep_world_size must be in [2, 1024], got {ep_world_size}")
    if not 1 <= moe_expert_num <= 2048:
        raise ValueError(f"moe_expert_num must be in [1, 2048], got {moe_expert_num}")
    if moe_expert_num % ep_world_size != 0:
        raise ValueError(
            f"moe_expert_num({moe_expert_num}) must be divisible by "
            f"ep_world_size({ep_world_size})"
        )
    if num_max_tokens_per_rank < 1:
        raise ValueError(
            "num_max_tokens_per_rank must be >= 1, "
            f"got {num_max_tokens_per_rank}"
        )
    if not 1 <= num_topk <= 32:
        raise ValueError(f"num_topk must be in [1, 32], got {num_topk}")
    if not 1024 <= hidden <= 8192:
        raise ValueError(f"hidden must be in [1024, 8192], got {hidden}")

    expert_per_rank = moe_expert_num // ep_world_size

    compare_count = ceil_align(num_max_tokens_per_rank * num_topk * 4, ALIGN_256) // 4
    mask_align_size = ceil_align(compare_count // 8, ALIGN_32)
    mask_slot_size = mask_align_size + ALIGN_32
    mask_recv_size = ceil_align(
        expert_per_rank * ep_world_size * mask_slot_size,
        ALIGN_512,
    )

    mx_scale_num = (hidden + 31) // 32
    data_bytes = ceil_align(hidden, ALIGN_256)
    token_bytes = ceil_align(data_bytes + mx_scale_num, ALIGN_32)
    quant_token_scale_size = ceil_align(
        num_max_tokens_per_rank * token_bytes,
        ALIGN_512,
    )

    combine_send_size = ceil_align(
        num_max_tokens_per_rank * hidden * num_topk * 2,
        ALIGN_512,
    )
    total_size = (
        PEERMEM_DATA_OFFSET
        + mask_recv_size
        + quant_token_scale_size
        + combine_send_size
    )

    return {
        "peermem_data_offset": PEERMEM_DATA_OFFSET,
        "mask_recv_size": mask_recv_size,
        "quant_token_scale_size": quant_token_scale_size,
        "combine_send_size": combine_send_size,
        "total_size": total_size,
    }


def parse_excel_int(value, column: str, sheet: str, excel_row: int) -> int:
    if value is None or pd.isna(value):
        raise ValueError(f"{sheet}!row {excel_row}: {column} is empty")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{sheet}!row {excel_row}: {column} is not an integer: {value!r}"
        ) from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(
            f"{sheet}!row {excel_row}: {column} is not an integer: {value!r}"
        )
    return int(number)


def resolve_moe_expert_num(row, ep_world_size: int, sheet: str, excel_row: int) -> int:
    local_value = row.get("local_moe_expert_num")
    if local_value is not None and not pd.isna(local_value):
        local_moe_expert_num = parse_excel_int(
            local_value,
            "local_moe_expert_num",
            sheet,
            excel_row,
        )
        return ep_world_size * local_moe_expert_num

    if "moe_expert_num" in row:
        return parse_excel_int(
            row["moe_expert_num"],
            "moe_expert_num",
            sheet,
            excel_row,
        )

    raise ValueError(
        f"{sheet}!row {excel_row}: local_moe_expert_num or moe_expert_num is required"
    )


def read_cases(excel_path: Path, requested_sheets: list[str] | None) -> list[dict]:
    workbook = pd.ExcelFile(excel_path)
    sheets = requested_sheets or workbook.sheet_names
    unknown_sheets = sorted(set(sheets) - set(workbook.sheet_names))
    if unknown_sheets:
        raise ValueError(f"sheet not found: {', '.join(unknown_sheets)}")

    records = []
    for sheet in sheets:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        frame.columns = [str(column).strip() for column in frame.columns]
        required_columns = {"bs", "h", "k", "ep_world_size"}
        missing_columns = sorted(required_columns - set(frame.columns))
        has_expert_column = bool(
            {"local_moe_expert_num", "moe_expert_num"} & set(frame.columns)
        )
        if missing_columns or not has_expert_column:
            details = []
            if missing_columns:
                details.append(f"missing columns: {', '.join(missing_columns)}")
            if not has_expert_column:
                details.append("missing local_moe_expert_num/moe_expert_num")
            if requested_sheets:
                raise ValueError(f"sheet {sheet}: {'; '.join(details)}")
            print(f"[INFO] skip sheet {sheet}: {'; '.join(details)}")
            continue

        for frame_index, row in frame.iterrows():
            if row.isna().all():
                continue
            excel_row = int(frame_index) + 2
            ep_world_size = parse_excel_int(
                row["ep_world_size"], "ep_world_size", sheet, excel_row
            )
            moe_expert_num = resolve_moe_expert_num(
                row, ep_world_size, sheet, excel_row
            )
            bs = parse_excel_int(row["bs"], "bs", sheet, excel_row)
            topk = parse_excel_int(row["k"], "k", sheet, excel_row)
            hidden = parse_excel_int(row["h"], "h", sheet, excel_row)
            breakdown = calc_a5_ccl_buffer_size(
                ep_world_size,
                moe_expert_num,
                bs,
                topk,
                hidden,
            )
            records.append(
                {
                    "sheet": sheet,
                    "excel_row": excel_row,
                    "case_name": str(row.get("case_name", "")).strip(),
                    "bs": bs,
                    "h": hidden,
                    "k": topk,
                    "ep_world_size": ep_world_size,
                    "moe_expert_num": moe_expert_num,
                    "local_moe_expert_num": moe_expert_num // ep_world_size,
                    "breakdown": breakdown,
                }
            )

    if not records:
        raise ValueError("no valid case rows found")
    return records


def parse_args() -> argparse.Namespace:
    default_excel = Path(__file__).resolve().parent / "batch_run_once" / "test_mega.xlsx"
    parser = argparse.ArgumentParser(
        description="Find the largest A5 MegaMoe ccl_buffer_size in an Excel workbook."
    )
    parser.add_argument(
        "excel",
        nargs="?",
        type=Path,
        default=default_excel,
        help=f"Excel path. Default: {default_excel}",
    )
    parser.add_argument(
        "-s",
        "--sheet",
        action="append",
        dest="sheets",
        help="Sheet to scan. Repeat to scan multiple sheets; omit to scan all sheets.",
    )
    return parser.parse_args()


def print_result(excel_path: Path, records: list[dict]) -> None:
    max_size = max(record["breakdown"]["total_size"] for record in records)
    max_records = [
        record for record in records if record["breakdown"]["total_size"] == max_size
    ]

    print(f"Excel: {excel_path.resolve()}")
    print(f"Scanned cases: {len(records)}")
    print(f"Max ccl_buffer_size: {max_size:,} bytes ({max_size / MIB:.3f} MiB)")
    if len(max_records) > 1:
        print(f"Tied max cases: {len(max_records)}")

    for index, record in enumerate(max_records, start=1):
        breakdown = record["breakdown"]
        print("-" * 72)
        if len(max_records) > 1:
            print(f"Max case {index}")
        print(f"sheet: {record['sheet']}")
        print(f"excel_row: {record['excel_row']}")
        print(f"case_name: {record['case_name']}")
        print(f"bs: {record['bs']}")
        print(f"h: {record['h']}")
        print(f"k: {record['k']}")
        print(f"ep_world_size: {record['ep_world_size']}")
        print(f"local_moe_expert_num: {record['local_moe_expert_num']}")
        print(f"moe_expert_num: {record['moe_expert_num']}")
        print("buffer breakdown (bytes):")
        print(f"  peermem_data_offset: {breakdown['peermem_data_offset']:,}")
        print(f"  mask_recv_size: {breakdown['mask_recv_size']:,}")
        print(f"  quant_token_scale_size: {breakdown['quant_token_scale_size']:,}")
        print(f"  combine_send_size: {breakdown['combine_send_size']:,}")


def main() -> None:
    args = parse_args()
    if not args.excel.is_file():
        raise FileNotFoundError(f"Excel file not found: {args.excel}")
    records = read_cases(args.excel, args.sheets)
    print_result(args.excel, records)


if __name__ == "__main__":
    main()

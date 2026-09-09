"""Generate independent QSLI PT fixtures from the transfer-test Excel."""

import argparse
import ast
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsli_test_utils import generate_qsli_test_data


def _value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() in ("True", "False", "None"):
        return ast.literal_eval(value.strip())
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", "-S", default="TestCases")
    args = parser.parse_args()
    frame = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    required = [
        "Testcase_Name", "batch_size", "query_length", "sequence_length",
        "cmp_ratio", "storage_mode", "mask_mode", "topk",
        "candidate_block_length", "output_idx_offset", "seed",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    output = Path(args.pt_dir)
    output.mkdir(parents=True, exist_ok=True)
    for _, row in frame.iterrows():
        params = {name: _value(row[name]) for name in required if name != "Testcase_Name"}
        target = output / f"{row['Testcase_Name']}.pt"
        torch.save(generate_qsli_test_data(params), target)
        print(f"saved: {target}")


if __name__ == "__main__":
    main()

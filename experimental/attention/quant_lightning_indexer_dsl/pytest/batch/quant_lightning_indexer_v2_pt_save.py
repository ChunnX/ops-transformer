"""Read transfer cases from Excel and serialize deterministic PT inputs."""

from pathlib import Path
import argparse

import torch

import quant_lightning_indexer_v2_golden
from qliv2_parameter_normalization import normalize_cell
from qliv2_test_utils import PARAM_NAMES, QliV2ResultWriter


def load_excel_test_cases(excel_file_path: str, sheetname: str):
    import pandas as pd

    frame = pd.read_excel(excel_file_path, sheet_name=sheetname)
    required = ["Testcase_Name", *PARAM_NAMES]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Excel is missing QLI columns: {missing}")
    cases = []
    for _, row in frame.iterrows():
        values = tuple(normalize_cell(row[name]) for name in PARAM_NAMES)
        explicit_name = normalize_cell(row.get("Testcase_Name"))
        cases.append((QliV2ResultWriter.case_name(values, explicit_name), values))
    return cases


def save_test_case(test_cases, file_path):
    output = Path(file_path)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for case_name, test_data in test_cases:
        case = quant_lightning_indexer_v2_golden.generate_qliv2_test_data(test_data)
        path = output / f"{case_name}.pt"
        torch.save(case, path)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_file")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", default="Sheet1")
    args = parser.parse_args()
    paths = save_test_case(
        load_excel_test_cases(args.excel_file, args.sheet), args.pt_dir
    )
    print(f"saved {len(paths)} PT cases to {args.pt_dir}")


if __name__ == "__main__":
    main()

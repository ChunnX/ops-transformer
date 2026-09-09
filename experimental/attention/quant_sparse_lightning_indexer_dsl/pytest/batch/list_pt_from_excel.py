"""Map the Excel Testcase_Name column to existing PT files."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", "-S", default="TestCases")
    args = parser.parse_args()
    frame = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    if "Testcase_Name" not in frame.columns:
        raise ValueError("Column 'Testcase_Name' is required")
    root = Path(args.pt_dir)
    files = [root / f"{name}.pt" for name in frame["Testcase_Name"]]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing PT cases: {missing}")
    print(",".join(str(path) for path in files))


if __name__ == "__main__":
    main()

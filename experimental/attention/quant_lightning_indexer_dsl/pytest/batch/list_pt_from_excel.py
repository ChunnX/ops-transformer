"""List PT paths corresponding to named rows in a QLI Excel case table."""

from pathlib import Path
import argparse

from batch.quant_lightning_indexer_v2_pt_save import load_excel_test_cases


def list_pt_from_excel(excel_path, sheet_name, pt_dir):
    return [
        str(Path(pt_dir) / f"{case_name}.pt")
        for case_name, _ in load_excel_test_cases(excel_path, sheet_name)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", default="Sheet1")
    args = parser.parse_args()
    print(",".join(list_pt_from_excel(args.excel_path, args.sheet, args.pt_dir)))


if __name__ == "__main__":
    main()

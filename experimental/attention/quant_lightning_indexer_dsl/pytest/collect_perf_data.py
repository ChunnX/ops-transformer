"""Summarize batch result workbooks using the mainline helper filename."""

import argparse

def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("result_xlsx")
    args = parser.parse_args()
    frame = pd.read_excel(args.result_xlsx)
    print(frame.groupby("result", dropna=False).size().to_string())


if __name__ == "__main__":
    main()

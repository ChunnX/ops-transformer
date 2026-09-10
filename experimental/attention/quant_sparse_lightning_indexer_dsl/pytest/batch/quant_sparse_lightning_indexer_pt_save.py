"""Generate independent QSLI PT fixtures from the transfer-test Excel."""

import argparse
import ast
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsli_test_utils import QsliCaseSelector, generate_qsli_test_data


def _value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in ("True", "False", "None") or text.startswith(("[", "(", "{")):
            return ast.literal_eval(text)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", "-S", default="TestCases")
    parser.add_argument("--indexes", default="")
    args = parser.parse_args()
    frame = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    required = [
        "Testcase_Name", "batch_size", "q_seq", "k_seq", "q_t_size", "k_t_size",
        "q_head_num", "k_head_num", "head_dim", "block_size", "block_num",
        "qk_dtype", "weight_dtype", "dequant_dtype", "actual_seq_dtype",
        "cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k",
        "cmp_residual_k", "max_seqlen_q", "quant_mode", "layout_query",
        "layout_key", "sparse_count", "sparse_mode", "query_datarange",
        "key_datarange", "weights_datarange", "q_scale_datarange",
        "k_scale_datarange", "cmp_ratio", "return_value", "output_idx_offset",
        "candidate_block_indices", "candidate_block_length", "candidate_block_size",
        "block_table", "storage_layout", "seed",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    output = Path(args.pt_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.indexes:
        selected = QsliCaseSelector.parse_indexes(args.indexes, len(frame))
        frame = frame.iloc[[index - 1 for index in selected]]
    for _, row in frame.iterrows():
        params = {name: _value(row[name]) for name in required if name != "Testcase_Name"}
        target = output / f"{row['Testcase_Name']}.pt"
        torch.save(generate_qsli_test_data(params), target)
        print(f"saved: {target}")


if __name__ == "__main__":
    main()

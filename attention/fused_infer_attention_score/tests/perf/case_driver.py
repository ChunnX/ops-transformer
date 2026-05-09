#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Single FIA case driver, designed to be wrapped by `msprof op`.

Usage:
    msprof op --warm-up=10 --launch-count=3 --output=./out \
        python3 case_driver.py --cases cases.json --case-id N

Reads case_id from cases.json, dispatches to scenarios/<scenario>.build(case)
which returns (positional_args, kwargs) for torch_npu.npu_fused_infer_attention_score.

Calls the op exactly once between two synchronize() calls so msprof captures
the FIA op in isolation.
"""

import argparse
import json
import logging
import os
import random
import sys

import torch
import torch_npu

# Ensure scenarios package is importable when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import scenarios  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("case_driver")


def load_case(cases_path, case_id):
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        if c["case_id"] == case_id:
            return c
    raise SystemExit(f"case_id {case_id} not found in {cases_path}")


def main():
    parser = argparse.ArgumentParser(description="Single-case FIA driver for msprof.")
    parser.add_argument("--cases", required=True, help="Path to cases.json")
    parser.add_argument("--case-id", required=True, type=int)
    parser.add_argument("--device", type=int, default=int(os.environ.get("FIA_PROFILE_DEVICE", "0")))
    args = parser.parse_args()

    case = load_case(args.cases, args.case_id)
    scenario = case.get("scenario", "no_quant")

    logger.info("case_id=%d scenario=%s tag=%s shape=B%d_Nq%d_Nkv%d_Sq%d_Skv%d_D%d %s %s",
                case["case_id"], scenario, case.get("tag", ""),
                case.get("batch_size", 0), case.get("q_head_num", 0),
                case.get("kv_head_num", 0), case.get("q_seq", 0),
                case.get("kv_seq", 0), case.get("head_dim", 0),
                case.get("in_layout", ""), case.get("dtype", ""))

    torch.manual_seed(42)
    random.seed(42)
    torch_npu.npu.set_device(args.device)

    pos_args, kwargs = scenarios.build(case)

    torch.npu.synchronize()
    out = torch_npu.npu_fused_infer_attention_score(*pos_args, **kwargs)
    torch.npu.synchronize()

    # Touch the result so DCE can't drop the call.
    if isinstance(out, tuple):
        _ = out[0].shape
    else:
        _ = out.shape


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate single-call tests for packed/unpacked window reuse on EP16."""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "parent_cases", Path(__file__).resolve().parents[1] / "generate_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for group in range(4):
        for half in range(2):
            cases.append(
                module.make_case(
                    f"transition_poison_{group}_{half}",
                    bs=256,
                    h=2048,
                    k=8,
                    moe_expert_num=128,
                    loop=1,
                    dis_pattern="cyclic",
                )
            )
        for half in range(2):
            cases.append(
                module.make_case(
                    f"transition_delayed_{group}_{half}",
                    bs=8 if group % 2 == 0 else 256,
                    h=2048,
                    k=8,
                    moe_expert_num=128,
                    loop=1,
                    comm_quant_mode=0 if group % 2 == 0 else 2,
                    dis_pattern="cyclic",
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cases).to_excel(args.output / "combine_transition.xlsx", index=False)
    (args.output / "combine_transition.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
    stress = [
        module.make_case(
            "stress_alternating_qwen_next",
            bs=256,
            h=2048,
            k=10,
            moe_expert_num=512,
            loop=1,
            dis_pattern="cyclic",
        )
    ]
    quant_stress = [
        module.make_case(
            f"stress_alternating_int8_{dtype}",
            bs=bs,
            h=2048,
            k=10,
            moe_expert_num=512,
            dtype=dtype,
            comm_quant_mode=2,
            loop=1,
            dis_pattern="cyclic",
        )
        for dtype, bs in (("bf16", 256), ("fp16", 17))
    ]
    boundary = []
    models = json.loads(
        (Path(__file__).resolve().parent / "models/model_manifest.json").read_text()
    )
    for model in models:
        if not model["ep16_supported"]:
            continue
        tag = model["model"].split("/")[-1].lower().replace(".", "_").replace("-", "_")
        boundary.append(
            module.make_case(
                "adaptive_fp16_" + tag,
                bs=256,
                h=model["h"],
                k=model["topk"],
                moe_expert_num=model["experts"],
                dtype="fp16",
                dis_pattern="cyclic",
            )
        )
    for label, change in (
        ("empty_mask", {"token_mask_true_num": 0}),
        ("token_mask", {"token_mask_true_num": 127}),
        ("expert_mask", {"is_expert_mask": True}),
        ("random", {"dis_pattern": "random"}),
        ("pressure", {"dis_pattern": "pressure"}),
        ("performance_info", {"need_performance_info": 1}),
        ("h8192_k16", {"h": 8192, "k": 16, "moe_expert_num": 1024}),
    ):
        boundary.append(
            module.make_case(
                "adaptive_boundary_" + label,
                **({"bs": 256, "dis_pattern": "cyclic"} | change),
            )
        )
    for name, rows in (
        ("combine_stream_stress", stress),
        ("combine_quant_stream_stress", quant_stress),
        ("combine_adaptive_boundary", boundary),
    ):
        pd.DataFrame(rows).to_excel(args.output / (name + ".xlsx"), index=False)
        (args.output / (name + ".json")).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

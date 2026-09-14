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
"""Generate deterministic native dispatch/combine cases for ep_test Stage v2."""

import argparse
import json
from pathlib import Path

import pandas as pd


def make_case(name, **changes):
    row = dict(
        case_name=name,
        bs=32,
        h=3584,
        k=8,
        dtype="bf16",
        world_size=16,
        server_num=1,
        moe_expert_num=256,
        is_performance=False,
        loop=10,
        graph_type=0,
        comm_quant_mode=0,
        quant_mode=0,
        comm_alg="",
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
        token_mask_true_num=-1,
        is_expert_mask=False,
        has_scale=False,
        expert_token_nums_type=0,
        need_performance_info=0,
        dis_pattern="force",
        seed=20260914,
        HCCL_BUFFSIZE_fullmesh=2048,
        HCCL_BUFFSIZE_hierarchy=2048,
    )
    row.update(changes)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = [make_case(f"bf16_b{bs}", bs=bs) for bs in (1, 8, 16, 32, 64, 128, 256)]
    common += [
        make_case("fp16", dtype="fp16"),
        make_case("hidden7168", h=7168),
        make_case("topk1", k=1),
        make_case("topk16", k=16),
        make_case("expert16", moe_expert_num=16),
        make_case("random", dis_pattern="random"),
        make_case("pressure", dis_pattern="pressure"),
        make_case("token_mask", token_mask_true_num=17),
        make_case("empty_mask", token_mask_true_num=0),
        make_case("expert_mask", is_expert_mask=True),
        make_case("individual_counts", expert_token_nums_type=1),
        make_case("performance_info", need_performance_info=1),
    ]
    dispatch = common + [
        make_case(f"int8_b{bs}", bs=bs, quant_mode=2) for bs in (1, 16, 32, 128, 256)
    ]
    dispatch += [
        make_case("int8_smooth", quant_mode=2, has_scale=True),
        make_case("int8_pressure", quant_mode=2, dis_pattern="pressure"),
        make_case("int8_hidden7168", quant_mode=2, h=7168),
    ]
    combine = common + [
        make_case(f"int8_b{bs}", bs=bs, comm_quant_mode=2)
        for bs in (1, 16, 32, 128, 256)
    ]
    combine += [
        make_case("int8_pressure", comm_quant_mode=2, dis_pattern="pressure"),
        make_case("int8_hidden7168", comm_quant_mode=2, h=7168),
    ]
    suites = {
        "dispatch_accuracy": dispatch,
        "combine_accuracy": combine,
        "both_accuracy": [
            make_case("roundtrip_b32"),
            make_case("roundtrip_b128", bs=128),
        ],
    }
    for op in ("dispatch", "combine"):
        suites[op + "performance"] = [
            make_case(
                f"{op}_b{bs}_q{q}",
                bs=bs,
                is_performance=True,
                loop=100,
                **({"quant_mode": q} if op == "dispatch" else {"comm_quant_mode": q}),
            )
            for bs in (8, 32, 128, 256)
            for q in (0, 2)
        ]
    suites["combine_nonquant_recheck"] = [
        case
        for case in suites["combineperformance"]
        if case["bs"] in (128, 256) and case["comm_quant_mode"] == 0
    ]
    for name, rows in suites.items():
        pd.DataFrame(rows).to_excel(args.output / (name + ".xlsx"), index=False)
    (args.output / "cases.json").write_text(json.dumps(suites, indent=2) + "\n")
    print(json.dumps({name: len(rows) for name, rows in suites.items()}))


if __name__ == "__main__":
    main()

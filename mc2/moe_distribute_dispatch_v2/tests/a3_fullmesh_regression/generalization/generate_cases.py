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
"""EP16 model-derived cases; B and communication dtype are test choices."""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--sources", type=Path, default=Path(__file__).resolve().parent / "models"
)
args = parser.parse_args()
ROOT = args.output
(ROOT / "cases").mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "prior_cases", Path(__file__).resolve().parents[1] / "generate_cases.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
make_case = module.make_case
models = json.loads((args.sources / "model_manifest.json").read_text())
suites = {
    f"{op}_{kind}": []
    for op in ("dispatch", "combine")
    for kind in ("accuracy", "performance")
}
suites["both_accuracy"] = []
for op in ("dispatch", "combine"):
    suites[f"{op}_large_accuracy"] = []
for m in models:
    if not m["ep16_supported"]:
        continue
    tag = m["model"].split("/")[-1].lower().replace(".", "_").replace("-", "_")
    shape = dict(
        h=m["h"], k=m["topk"], moe_expert_num=m["experts"], dis_pattern="cyclic"
    )
    for op in ("dispatch", "combine"):
        for b in (1, 8, 32, 128, 256):
            for q in (0, 2):
                quant = {"quant_mode" if op == "dispatch" else "comm_quant_mode": q}
                name = f"{op}_{tag}_b{b}_q{q}"
                suites[f"{op}_accuracy"].append(make_case(name, bs=b, **shape, **quant))
                if b in (8, 256):
                    suites[f"{op}_performance"].append(
                        make_case(
                            name, bs=b, is_performance=True, loop=100, **shape, **quant
                        )
                    )
        for b, q in ((257, 0), (512, 0), (512, 2)):
            quant = {"quant_mode" if op == "dispatch" else "comm_quant_mode": q}
            suites[f"{op}_large_accuracy"].append(
                make_case(f"{op}_{tag}_b{b}_q{q}", bs=b, **shape, **quant)
            )
        for label, options in (
            ("fp16_tail", dict(bs=17, dtype="fp16")),
            ("random_tail", dict(bs=127, dis_pattern="random")),
            ("pressure_tail", dict(bs=255, dis_pattern="pressure")),
            ("masked_tail", dict(bs=33, token_mask_true_num=17)),
        ):
            suites[f"{op}_accuracy"].append(
                make_case(f"{op}_{tag}_{label}", **(shape | options))
            )
    suites["both_accuracy"].append(make_case(f"both_{tag}", bs=17, **shape))
for op in ("dispatch", "combine"):
    for label, options in (
        (
            f"h8192_k{12 if op == 'dispatch' else 16}_e1024",
            dict(h=8192, k=12 if op == "dispatch" else 16, moe_expert_num=1024, bs=17),
        ),
        ("h1024_k1_e16", dict(h=1024, k=1, moe_expert_num=16, bs=1)),
        ("empty_mask", dict(bs=33, token_mask_true_num=0)),
        ("expert_mask", dict(bs=33, is_expert_mask=True)),
        ("individual_counts", dict(bs=33, expert_token_nums_type=1)),
        (
            "int8_fp16",
            dict(
                bs=17,
                dtype="fp16",
                **({"quant_mode": 2} if op == "dispatch" else {"comm_quant_mode": 2}),
            ),
        ),
    ):
        suites[f"{op}_accuracy"].append(make_case(f"{op}_boundary_{label}", **options))
suites["combine_diagnostic"] = [make_case("combine_diag_b256", bs=256, loop=3)]
suites["combine_recheck"] = [
    make_case(f"combine_b{b}_q0", bs=b, is_performance=True, loop=100)
    for b in (128, 256)
]
# Explicitly select the migrated dispatch kernel; large-B compatibility
# cases intentionally retain the default algorithm.
for suite in ("dispatch_accuracy", "dispatch_performance", "both_accuracy"):
    for case in suites[suite]:
        case["comm_alg"] = "fullmesh_v2"
        case["case_name"] = (
            case["case_name"].replace("dispatch_", "dispatch_fm2_", 1)
            if suite.startswith("dispatch")
            else case["case_name"] + "_fm2"
        )
# K16 is supported by default dispatch, while fullmesh_v2 is limited to K12.
suites["dispatch_large_accuracy"].append(
    make_case(
        "dispatch_boundary_default_h8192_k16_e1024",
        h=8192,
        k=16,
        moe_expert_num=1024,
        bs=17,
    )
)
for name, rows in suites.items():
    pd.DataFrame(rows).to_excel(ROOT / "cases" / (name + ".xlsx"), index=False)
(ROOT / "cases/cases.json").write_text(json.dumps(suites, indent=2) + "\n")
print({name: len(rows) for name, rows in suites.items()})

"""Create model-shaped ATK cases for the msSanitizer pass.

These are the repository's production and regression dimensions, not a
replacement for the normal full-domain accuracy JSON.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
source = ROOT / "result" / "atk_test_block_attention_residuals" / "json" / "all_atk_test_block_attention_residuals.json"
target = ROOT / "result" / "atk_test_block_attention_residuals" / "json" / "mssanitizer_model_atk_test_block_attention_residuals.json"

model_shapes = [
    (544, 12, 7168),   # production case
    (1024, 1, 7168),   # previous large-shape regression
    (4096, 4, 7168),   # previous large-shape regression
]

cases = []
for case_id, (tokens, blocks, hidden) in enumerate(model_shapes):
    case = copy.deepcopy(json.loads(source.read_text())[0])
    case["id"] = case_id
    case["default_seed"] = case_id
    for item in case["inputs"]:
        shapes = {
            "prefix_sum": [tokens, hidden],
            "block_res": [tokens, blocks, hidden],
            "proj_weight": [1, hidden],
            "norm_weight": [hidden],
        }
        if item["name"] in shapes:
            item["shape"] = shapes[item["name"]]
            item["dtype"] = "bf16"
        elif item["name"] == "valid_block_num":
            item["range_values"] = blocks
        elif item["name"] == "need_backward":
            item["range_values"] = False
    cases.append(case)

target.write_text(json.dumps(cases, indent=2) + "\n")
print(f"wrote {target} ({len(cases)} model cases)")

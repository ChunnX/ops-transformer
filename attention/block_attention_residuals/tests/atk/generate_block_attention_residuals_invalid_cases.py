"""Build reproducible ATK expected-error cases from the formal valid case set."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
source = ROOT / "result" / "atk_test_block_attention_residuals" / "json" / "all_atk_test_block_attention_residuals.json"
target = ROOT / "result" / "atk_test_block_attention_residuals" / "json" / "invalid_atk_test_block_attention_residuals.json"

cases = json.loads(source.read_text())
base = copy.deepcopy(cases[0])

valid_block_num = copy.deepcopy(base)
valid_block_num["id"] = 1000
valid_block_num["expected_error_msg"] = (
    "only default valid_block_num=N is supported"
)
for item in valid_block_num["inputs"]:
    if item["name"] == "valid_block_num":
        item["range_values"] = 0

norm_eps = copy.deepcopy(base)
norm_eps["id"] = 1001
norm_eps["expected_error_msg"] = "normEps must be > 0"
for item in norm_eps["inputs"]:
    if item["name"] == "norm_eps":
        item["range_values"] = 0.0

target.write_text(json.dumps([valid_block_num, norm_eps], indent=2) + "\n")
print(f"wrote {target} ({2} expected-error cases)")

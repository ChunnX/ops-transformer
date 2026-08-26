#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Generate ST test cases for MoeTokenUnpermuteGrad ascend950 compatibility
# ----------------------------------------------------------------------------
import json
import os

test_cases = []
case_id = 0

# Shape scenarios: (num_tokens, topk, hidden)
shapes = [
    (10, 1, 64),
    (10, 3, 64),
    (100, 2, 128),
    (100, 4, 256),
    (128, 1, 512),
    (256, 2, 1024),
    (512, 1, 2048),
    (100, 3, 4096),
    (50, 2, 8192),
    (10, 1, 16),
]

# probs_none path: token dtype variations
for token_dtype in ["bf16", "fp16", "fp32"]:
    for num_tokens, topk, hidden in shapes:
        num_expert_tokens = num_tokens * topk
        test_cases.append({
            "id": case_id,
            "name": "aclnn_moe_token_unpermute_grad_probs_none_950",
            "aclnn_name": "MoeTokenUnpermuteGrad",
            "api_type": "ascend_aclnn_moe_token_unpermute_grad_probs_none_950",
            "aclnn_api_type": "AclnnBaseApi_aclnn_moe_token_unpermute_grad_probs_none_950",
            "standard": {"acc": "md5", "perf": "not_key"},
            "inputs": [
                {"name": "permutedTokens", "type": "tensor", "required": True, "dtype": token_dtype,
                 "shape": [num_expert_tokens, hidden], "range_values": [-2, 2], "backward": True, "align_32B": None},
                {"name": "unpermutedTokensGrad", "type": "tensor", "required": True, "dtype": token_dtype,
                 "shape": [num_tokens, hidden], "range_values": [-2, 2], "backward": True, "align_32B": None},
                {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32",
                 "shape": [num_expert_tokens], "range_values": [0, num_tokens - 1], "backward": True, "align_32B": None},
                {"name": "probs", "type": "tensor", "required": True, "dtype": token_dtype,
                 "shape": [num_tokens, topk], "range_values": [-2, 2], "backward": True, "align_32B": None},
                {"name": "padded_mode", "type": "attr", "required": True, "dtype": "bool",
                 "shape": None, "range_values": False, "backward": False, "align_32B": None},
                [{"name": "restore_shape", "type": "attr_tuple", "required": True, "dtype": "int",
                  "shape": None, "range_values": num_tokens, "backward": False, "align_32B": None},
                 {"name": "restore_shape", "type": "attr_tuple", "required": True, "dtype": "int",
                  "shape": None, "range_values": num_tokens, "backward": False, "align_32B": None}]
            ]
        })
        case_id += 1

# probs_not_none path: token dtype x probs dtype combinations
token_dtypes = ["bf16", "fp16", "fp32"]
prob_dtypes = ["bf16", "fp16", "fp32"]
for token_dtype in token_dtypes:
    for prob_dtype in prob_dtypes:
        for num_tokens, topk, hidden in shapes[:6]:
            num_expert_tokens = num_tokens * topk
            test_cases.append({
                "id": case_id,
                "name": "aclnn_moe_token_unpermute_grad_probs_not_none_950",
                "aclnn_name": "MoeTokenUnpermuteGrad",
                "api_type": "ascend_aclnn_moe_token_unpermute_grad_probs_not_none_950",
                "aclnn_api_type": "AclnnBaseApi_aclnn_moe_token_unpermute_grad_probs_not_none_950",
                "standard": {"acc": "md5", "perf": "not_key"},
                "inputs": [
                    {"name": "permutedTokens", "type": "tensor", "required": True, "dtype": token_dtype,
                     "shape": [num_expert_tokens, hidden], "range_values": [-2, 2], "backward": True, "align_32B": None},
                    {"name": "unpermutedTokensGrad", "type": "tensor", "required": True, "dtype": token_dtype,
                     "shape": [num_tokens, hidden], "range_values": [-2, 2], "backward": True, "align_32B": None},
                    {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32",
                     "shape": [num_expert_tokens], "range_values": [0, num_tokens - 1], "backward": True, "align_32B": None},
                    {"name": "probs", "type": "tensor", "required": True, "dtype": prob_dtype,
                     "shape": [num_tokens, topk], "range_values": [-2, 2], "backward": True, "align_32B": None},
                    {"name": "padded_mode", "type": "attr", "required": True, "dtype": "bool",
                     "shape": None, "range_values": False, "backward": False, "align_32B": None},
                    [{"name": "restore_shape", "type": "attr_tuple", "required": True, "dtype": "int",
                      "shape": None, "range_values": num_tokens, "backward": False, "align_32B": None},
                     {"name": "restore_shape", "type": "attr_tuple", "required": True, "dtype": "int",
                      "shape": None, "range_values": num_tokens, "backward": False, "align_32B": None}]
                ]
            })
            case_id += 1

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atk_aclnnMoeTokenUnpermuteGradAscend950.json")
with open(output_path, "w") as f:
    json.dump(test_cases, f)
print(f"Generated {len(test_cases)} test cases to {output_path}")

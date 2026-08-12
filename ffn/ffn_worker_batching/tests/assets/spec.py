#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TestSpec adapter for aclnnFfnWorkerBatching."""

import importlib.util
import numpy as np
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def _load_impl(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"ffn_wb_impl_{stem}_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


golden_module = _load_impl("golden")
inputs_module = _load_impl("inputs")
golden_module._DATA_CACHE = inputs_module._DATA_CACHE


class AclnnFfnWorkerBatchingTestSpec:
    """TestSpec for aclnnFfnWorkerBatching (ACLNN mode).

    golden / customize_inputs receive torch.Tensor parameters following
    aclnnFfnWorkerBatchingGetWorkspaceSize signature (without workspaceSize
    and executor).  The scheduleContext is the only input tensor; outputs
    are auto-identified via output_tensor_indexes in the CSV.
    """
    golden = golden_module.ffn_worker_batching_golden
    customize_inputs = inputs_module.ffn_worker_batching_inputs

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        """Custom compare.

        The kernel only writes outputs for the first actual_token_num tokens;
        remaining positions retain the TTK buffer init value (1).  Golden
        fills those with 0, so we truncate comparison to valid elements.

        - y (0): compare first actual_token_num rows.
        - group_list (1): compare rows before the first [0, 0] terminator.
        - session_ids (2), micro_batch_ids (3), token_ids (4),
          expert_offsets (5): compare first actual_token_num elements.
        - dynamic_scale (6), actual_token_num (7): full binary_equal.
        """
        import torch

        def to_np(t):
            if isinstance(t, torch.Tensor):
                if t.dtype == torch.bfloat16:
                    return t.to(torch.float32).numpy()
                return t.numpy()
            return np.asarray(t)

        num_outputs = len(outputs) // 2
        npu_outputs = outputs[:num_outputs]
        golden_outputs = outputs[num_outputs:]

        actual_token_num = int(to_np(golden_outputs[7]).flat[0])

        results = []
        for idx in range(num_outputs):
            npu = to_np(npu_outputs[idx])
            gold = to_np(golden_outputs[idx])

            if idx == 1:
                npu_arr = npu.reshape(-1, 2)
                gold_arr = gold.reshape(-1, 2)
                valid = 0
                for row in gold_arr:
                    if row[0] == 0 and row[1] == 0:
                        break
                    valid += 1
                if valid == 0:
                    passed = True
                    precision = "100%"
                else:
                    match = int(np.sum(npu_arr[:valid] == gold_arr[:valid]))
                    total = valid * 2
                    passed = (match == total)
                    precision = f"{match / total * 100}%"
                results.append({
                    "pass": passed,
                    "precision": precision,
                    "error_info": None if passed else "group_list mismatch before [0,0] terminator",
                })
            elif idx in (0, 2, 3, 4, 5):
                npu_flat = npu.flatten()
                gold_flat = gold.flatten()
                if idx == 0:
                    n = actual_token_num * (npu.shape[-1] if npu.ndim > 1 else 1)
                else:
                    n = actual_token_num
                n = min(n, npu_flat.size)
                diff = int(np.sum(npu_flat[:n] != gold_flat[:n]))
                passed = (diff == 0)
                precision = f"{(n - diff) / n * 100}%" if n > 0 else "100%"
                results.append({
                    "pass": passed,
                    "precision": precision,
                    "error_info": None if passed else f"{diff} mismatches in first {n} elements",
                })
            else:
                npu_flat = npu.flatten()
                gold_flat = gold.flatten()
                diff = int(np.sum(npu_flat != gold_flat))
                total = npu_flat.size
                passed = (diff == 0)
                precision = f"{(total - diff) / total * 100}%" if total > 0 else "100%"
                results.append({
                    "pass": passed,
                    "precision": precision,
                    "error_info": None if passed else f"{diff} mismatches",
                })
        return results


__spec__ = {"aclnnFfnWorkerBatching": "AclnnFfnWorkerBatchingTestSpec"}

#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Precision comparison aligned with the SparseFlashMla pytest policy."""

from numbers import Integral

import numpy as np


class SparseFlashMlaComparator:
    """Apply the pytest relative-error and global failure-ratio policy."""

    DEFAULT_RTOL = 0.005
    BFLOAT16_RTOL = 0.0078125
    DEFAULT_ATOL = 0.000025
    BFLOAT16_ATOL = 0.0001
    FAIL_RATIO = 0.005
    MAX_RELATIVE_ERROR = 10.0
    RELATIVE_FLOOR = (1.0 / (1 << 14)) / 0.005
    RELATIVE_EPSILON = 2e-9

    @staticmethod
    def is_bfloat16(value):
        return "bfloat16" in str(getattr(value, "dtype", ""))

    @classmethod
    def as_float32(cls, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if cls.is_bfloat16(value):
                value = value.float()
            value = value.numpy()
        return np.asarray(value).astype(np.float32)

    @classmethod
    def compare_output(cls, npu_out, golden_out):
        if golden_out is None:
            return {"pass": True, "precision": "SUPPRESSED"}
        if npu_out is None:
            return {
                "pass": False,
                "precision": "NO_OUTPUT",
                "error_info": "NPU output is None",
            }

        is_bfloat16 = cls.is_bfloat16(npu_out)
        rtol = cls.BFLOAT16_RTOL if is_bfloat16 else cls.DEFAULT_RTOL
        atol = cls.BFLOAT16_ATOL if is_bfloat16 else cls.DEFAULT_ATOL
        npu = cls.as_float32(npu_out)
        golden = cls.as_float32(golden_out)
        if npu.shape != golden.shape:
            return {
                "pass": False,
                "precision": "shape_mismatch",
                "error_info": f"output shape mismatch: npu={npu.shape}, golden={golden.shape}",
            }
        if golden.size == 0:
            return {"pass": True, "precision": 100.0}

        npu_flat = npu.reshape(-1)
        golden_flat = golden.reshape(-1)
        mismatch = ~np.isclose(
            npu_flat, golden_flat, rtol=rtol, atol=atol, equal_nan=True
        )
        diff_idx = np.where(mismatch)[0]
        fail_ratio = diff_idx.size / golden_flat.size
        max_relative_error = 0.0
        if diff_idx.size:
            diff_abs = np.abs(golden_flat - npu_flat)
            denominator = np.maximum(
                np.maximum(np.abs(npu_flat), np.abs(golden_flat)), cls.RELATIVE_FLOOR
            )
            relative_error = diff_abs / (denominator + cls.RELATIVE_EPSILON)
            max_relative_error = float(np.max(relative_error[diff_idx]))

        passed = (
            fail_ratio <= cls.FAIL_RATIO
            and max_relative_error < cls.MAX_RELATIVE_ERROR
        )
        precision = (golden_flat.size - diff_idx.size) / golden_flat.size * 100
        error_info = None
        if not passed:
            error_info = (
                f"SMLA precision failed: mismatches={diff_idx.size}, "
                f"fail_ratio={fail_ratio:.6g}, "
                f"max_relative_error={max_relative_error:.6g}"
            )
        return {
            "pass": passed,
            "precision": precision,
            "diff_indices": diff_idx[:1000].tolist(),
            "error_info": error_info,
            "metrics": {
                "rtol": rtol,
                "atol": atol,
                "fail_ratio": fail_ratio,
                "fail_ratio_limit": cls.FAIL_RATIO,
                "max_relative_error": max_relative_error,
                "max_relative_error_limit": cls.MAX_RELATIVE_ERROR,
            },
        }


class SparseFlashMlaBatchConsistencyChecker:
    """Validate SMLA relation metadata and retain prior outputs for cross-case checks."""

    def __init__(self, comparator):
        self.comparator = comparator
        self.relation_store = {}

    @staticmethod
    def config_failure(message):
        return {
            "pass": False,
            "precision": "batch_config=FAIL",
            "error_info": message,
        }

    def parse_relations(self, batch_consistency_id, batch_axis,
                        batch_slice_info, batch_seed):
        fields = (batch_consistency_id, batch_axis, batch_slice_info, batch_seed)
        if all(field is None for field in fields):
            return None, None
        if any(field is None for field in fields):
            return None, self.config_failure("incomplete batch consistency metadata")

        try:
            if not (len(batch_axis) == len(batch_slice_info) == len(batch_seed)):
                raise ValueError("batch metadata top-level counts differ")
            if not batch_axis or tuple(batch_axis[0]) != (0,):
                raise ValueError("SMLA batch compare requires q axis 0")
            if batch_slice_info[0] is None or batch_seed[0] is None:
                raise ValueError("SMLA batch compare requires q slices and seeds")
            if any(value is not None for value in batch_slice_info[1:]):
                raise ValueError("SMLA batch compare supports q relations only")
            if any(value is not None for value in batch_seed[1:]):
                raise ValueError("SMLA batch compare supports q seeds only")
            if len(batch_slice_info[0]) != 1 or len(batch_seed[0]) != 1:
                raise ValueError("SMLA q metadata must contain exactly one axis group")

            slices = batch_slice_info[0][0]
            seeds = batch_seed[0][0]
            if not slices or len(slices) != len(seeds):
                raise ValueError("SMLA q slice and seed counts differ or are empty")

            relations = []
            expected_ids = []
            for slice_value, seed in zip(slices, seeds):
                if not isinstance(slice_value, (tuple, list)) or len(slice_value) != 3:
                    raise ValueError(f"invalid q slice {slice_value!r}")
                if not all(isinstance(value, Integral) for value in slice_value):
                    raise ValueError(f"q slice must contain integers: {slice_value!r}")
                if not isinstance(seed, Integral):
                    raise ValueError(f"q seed must be an integer: {seed!r}")
                start, stop, step = (int(value) for value in slice_value)
                seed = int(seed)
                if step != 1 or start < 0 or start >= stop:
                    raise ValueError(
                        f"q slice must be non-empty, non-negative and contiguous: {slice_value!r}"
                    )
                relation = (0, 0, seed, len(range(start, stop, step)))
                relations.append(((start, stop, step), relation))
                expected_ids.append(f"{seed}_0_{start}_{stop}_{step}")
        except (IndexError, TypeError, ValueError) as error:
            return None, self.config_failure(str(error))

        expected_batch_id = ((tuple(expected_ids),),)
        if batch_consistency_id != expected_batch_id:
            return None, self.config_failure(
                "SMLA batch_consistency_id does not match batch slice metadata: "
                f"expected={expected_batch_id!r}, actual={batch_consistency_id!r}"
            )
        return relations, None

    def compare_output(self, npu_output, batch_consistency_id,
                       batch_axis, batch_slice_info, batch_seed):
        relations, error = self.parse_relations(
            batch_consistency_id, batch_axis, batch_slice_info, batch_seed
        )
        if relations is None and error is None:
            return None
        if error is not None:
            return error
        if npu_output is None:
            return self.config_failure("SMLA batch output is None")

        npu = self.comparator.as_float32(npu_output)
        if npu.ndim == 0:
            return self.config_failure("SMLA batch output must have a batch axis")
        current_groups = {}
        for slice_value, relation in relations:
            if slice_value[1] > npu.shape[0]:
                return self.config_failure(
                    f"SMLA q slice {slice_value!r} exceeds output batch size {npu.shape[0]}"
                )
            selector = (slice(*slice_value),) + (slice(None),) * (npu.ndim - 1)
            current_groups.setdefault(relation, []).append(np.ascontiguousarray(npu[selector]))

        for relation, values in current_groups.items():
            if any(not np.array_equal(values[0], value) for value in values[1:]):
                return {
                    "pass": False,
                    "precision": "batch_intra=FAIL",
                    "error_info": f"SMLA intra-case relation {relation} differs",
                }

        cross_seen = {}
        for relation, values in current_groups.items():
            previous = self.relation_store.get(relation)
            cross_seen[relation] = previous is not None
            if previous is not None:
                previous_shape, previous_dtype, previous_value = previous
                if previous_shape != values[0].shape or previous_dtype != values[0].dtype.str:
                    return {
                        "pass": False,
                        "precision": "batch_cross=CONFIG_FAIL",
                        "error_info": f"SMLA cross-case shape/dtype for {relation} differs",
                    }
                if not np.array_equal(previous_value, values[0]):
                    return {
                        "pass": False,
                        "precision": "batch_cross=FAIL",
                        "error_info": f"SMLA cross-case relation {relation} differs",
                    }

        for relation, values in current_groups.items():
            self.relation_store[relation] = (
                values[0].shape, values[0].dtype.str, values[0].copy()
            )

        cross_status = "PASS" if all(cross_seen.values()) else "PENDING"
        return {
            "pass": True,
            "precision": f"batch_intra=PASS;batch_cross={cross_status}",
        }


COMPARATOR = SparseFlashMlaComparator()
BATCH_CONSISTENCY_CHECKER = SparseFlashMlaBatchConsistencyChecker(COMPARATOR)


def compare(*outputs, batch_consistency_id=None, batch_axis=None,
            batch_slice_info=None, batch_seed=None):
    """Run the pytest precision policy before exact batch-relation checks."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    results = [
        COMPARATOR.compare_output(npu_out, golden_out)
        for npu_out, golden_out in zip(outputs[:half], outputs[half:])
    ]
    if not all(result["pass"] for result in results):
        return results

    batch_result = BATCH_CONSISTENCY_CHECKER.compare_output(
        outputs[0], batch_consistency_id, batch_axis, batch_slice_info, batch_seed
    )
    if batch_result is not None:
        results.append(batch_result)
    return results

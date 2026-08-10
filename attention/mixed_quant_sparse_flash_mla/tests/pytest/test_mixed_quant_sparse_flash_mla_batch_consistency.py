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

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch


testcase_path = Path(os.environ.get("MQSMLA_PT_DIR", "mqsmla_testcase"))
device_id = int(os.environ.get("MQSMLA_DEVICE_ID", 0))
testcase_files = sorted(testcase_path.glob("*.pt")) if testcase_path.is_dir() else []
testcase_ids = [path.stem for path in testcase_files]

Q_BATCH_TENSORS = (
    "q",
    "ori_sparse_indices",
    "cmp_sparse_indices",
    "ori_topk_length",
    "cmp_topk_length",
)
BATCH_VECTOR_TENSORS = (
    "seqused_q",
    "seqused_ori_kv",
    "seqused_cmp_kv",
    "cmp_residual_kv",
)


def _batch_index(batch_ids):
    return torch.tensor(batch_ids, dtype=torch.long)


def make_batch_permutation(batch_size, seed):
    if batch_size < 2:
        raise ValueError("batch reorder requires B >= 2")
    generator = torch.Generator().manual_seed(seed)
    batch_ids = torch.randperm(batch_size, generator=generator).tolist()
    if batch_ids == list(range(batch_size)):
        batch_ids = batch_ids[1:] + batch_ids[:1]
    return batch_ids


def parse_batch_groups(raw_groups):
    try:
        groups = json.loads(raw_groups)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"batch_groups must be valid JSON: {error}") from error

    if not isinstance(groups, list):
        raise ValueError("batch_groups must be a JSON list of lists")
    for group in groups:
        if not isinstance(group, list):
            raise ValueError("each batch group must be a JSON list")
        if any(not isinstance(batch_id, int) or isinstance(batch_id, bool) for batch_id in group):
            raise ValueError("batch group IDs must be integers")
    return groups


def validate_batch_groups(groups, batch_size):
    if len(groups) < 2:
        raise ValueError("batch split requires at least two groups")
    if any(not group for group in groups):
        raise ValueError("batch split groups must not be empty")

    batch_ids = [batch_id for group in groups for batch_id in group]
    expected_ids = list(range(batch_size))
    if sorted(batch_ids) != expected_ids:
        raise ValueError(
            "batch groups must contain every batch ID exactly once: "
            f"expected={expected_ids}, actual={batch_ids}"
        )

    # Split mode changes only batch composition, not order inside a group.
    return [sorted(group) for group in groups]


def make_random_split_groups(batch_size, split_count, seed):
    if split_count < 2:
        raise ValueError("split_count must be >= 2")
    if split_count > batch_size:
        raise ValueError(
            f"split_count ({split_count}) must be <= batch size ({batch_size})"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(batch_size, generator=generator)
    groups = [sorted(chunk.tolist()) for chunk in torch.tensor_split(shuffled, split_count)]
    return validate_batch_groups(groups, batch_size)


def _select_axis_zero(tensor, batch_ids):
    if tensor is None:
        return None
    return tensor.index_select(0, _batch_index(batch_ids))


def _select_cu_seqlens(cu_seqlens, batch_ids):
    if cu_seqlens is None:
        return None
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    selected_lengths = lengths.index_select(0, _batch_index(batch_ids))
    return torch.cat(
        (
            cu_seqlens.new_zeros(1),
            selected_lengths.cumsum(0, dtype=cu_seqlens.dtype),
        )
    )


def _select_tnd_tensor(tensor, cu_seqlens, batch_ids):
    if tensor is None:
        return None
    if cu_seqlens is None:
        raise ValueError("TND tensor requires cu_seqlens")
    segments = []
    for batch_id in batch_ids:
        start = int(cu_seqlens[batch_id].item())
        end = int(cu_seqlens[batch_id + 1].item())
        segments.append(tensor[start:end])
    return torch.cat(segments, dim=0)


def _max_tensor_value(tensor, default=0):
    if tensor is None or tensor.numel() == 0:
        return default
    return int(tensor.max().item())


def _refresh_metadata_shape_fields(test_data):
    metadata = test_data["metadata_input"]
    op_input = test_data["op_input"]
    metadata["batch_size"] = int(metadata["seqused_q"].numel())
    if op_input["layout_q"] == "BSND":
        metadata["max_seqlen_q"] = int(op_input["q"].shape[1])
    else:
        cu_seqlens_q = metadata["cu_seqlens_q"]
        metadata["max_seqlen_q"] = _max_tensor_value(
            cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        )
    metadata["max_seqlen_ori_kv"] = _max_tensor_value(
        metadata.get("seqused_ori_kv")
    )
    metadata["max_seqlen_cmp_kv"] = _max_tensor_value(
        metadata.get("seqused_cmp_kv")
    )


def select_batches(test_data, batch_ids):
    """Create a batch view without regenerating any Q/KV sample data."""
    if not batch_ids:
        raise ValueError("batch_ids must not be empty")

    original_batch_size = int(test_data["metadata_input"]["batch_size"])
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError(f"batch_ids must be unique, but got {batch_ids}")
    if min(batch_ids) < 0 or max(batch_ids) >= original_batch_size:
        raise IndexError(
            f"batch_ids {batch_ids} exceed original batch size {original_batch_size}"
        )

    selected = {
        **test_data,
        "params": dict(test_data["params"]),
        "metadata_input": dict(test_data["metadata_input"]),
        "op_input": dict(test_data["op_input"]),
        "cpu_output": None,
        "cpu_lse": None,
        "original_batch_ids": list(batch_ids),
    }
    original_op = test_data["op_input"]
    original_metadata = test_data["metadata_input"]
    selected_op = selected["op_input"]
    selected_metadata = selected["metadata_input"]
    layout_q = original_op["layout_q"]
    layout_kv = original_op["layout_kv"]

    original_q_cu = original_op.get("cu_seqlens_q")
    for key in Q_BATCH_TENSORS:
        tensor = original_op.get(key)
        if layout_q == "BSND":
            selected_op[key] = _select_axis_zero(tensor, batch_ids)
        elif layout_q == "TND":
            selected_op[key] = _select_tnd_tensor(
                tensor, original_q_cu, batch_ids
            )
        else:
            raise ValueError(f"unsupported layout_q: {layout_q}")

    for container_name in ("op_input", "metadata_input"):
        original_container = test_data[container_name]
        selected_container = selected[container_name]
        for key in BATCH_VECTOR_TENSORS:
            if key in original_container:
                selected_container[key] = _select_axis_zero(
                    original_container.get(key), batch_ids
                )

    selected_op["cu_seqlens_q"] = _select_cu_seqlens(
        original_op.get("cu_seqlens_q"), batch_ids
    )
    selected_metadata["cu_seqlens_q"] = _select_cu_seqlens(
        original_metadata.get("cu_seqlens_q"), batch_ids
    )

    for prefix in ("ori", "cmp"):
        kv_key = f"{prefix}_kv"
        cu_key = f"cu_seqlens_{prefix}_kv"
        block_table_key = f"{prefix}_block_table"
        original_kv = original_op.get(kv_key)
        original_kv_cu = original_op.get(cu_key)
        if original_kv_cu is None:
            original_kv_cu = original_metadata.get(cu_key)

        if layout_kv == "PA_BBND":
            # Physical KV blocks remain unchanged. The block table keeps the
            # logical batch-to-physical-block mapping intact.
            selected_op[kv_key] = original_kv
        elif layout_kv == "BSND":
            selected_op[kv_key] = _select_axis_zero(original_kv, batch_ids)
        elif layout_kv == "TND":
            selected_op[kv_key] = _select_tnd_tensor(
                original_kv, original_kv_cu, batch_ids
            )
        else:
            raise ValueError(f"unsupported layout_kv: {layout_kv}")

        selected_op[block_table_key] = _select_axis_zero(
            original_op.get(block_table_key), batch_ids
        )
        selected_op[cu_key] = _select_cu_seqlens(
            original_op.get(cu_key), batch_ids
        )
        selected_metadata[cu_key] = _select_cu_seqlens(
            original_metadata.get(cu_key), batch_ids
        )

    _refresh_metadata_shape_fields(selected)

    case_name = test_data["Testcase_Name"]
    selected_name = f"{case_name}_batch_{'_'.join(map(str, batch_ids))}"
    selected["Testcase_Name"] = selected_name
    selected["params"]["Testcase_Name"] = selected_name
    selected["params"]["B"] = len(batch_ids)
    return selected


def _sample_from_layout(tensor, layout, cu_seqlens, batch_id):
    if tensor is None:
        return None
    if layout == "BSND":
        return tensor[batch_id]
    if layout == "TND":
        start = int(cu_seqlens[batch_id].item())
        end = int(cu_seqlens[batch_id + 1].item())
        return tensor[start:end]
    raise ValueError(f"unsupported layout: {layout}")


def _assert_tensor_equal(expected, actual, label):
    assert expected is not None and actual is not None, f"{label}: unexpected None"
    assert expected.dtype == actual.dtype, (
        f"{label}: dtype changed from {expected.dtype} to {actual.dtype}"
    )
    assert expected.shape == actual.shape, (
        f"{label}: shape changed from {tuple(expected.shape)} to {tuple(actual.shape)}"
    )
    expected_bytes = expected.contiguous().reshape(-1).view(torch.uint8)
    actual_bytes = actual.contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(expected_bytes, actual_bytes), (
        f"{label}: tensor byte content changed"
    )


def assert_sample_inputs_unchanged(original, transformed, batch_ids):
    """Verify that every transformed position still references the same sample."""
    original_op = original["op_input"]
    transformed_op = transformed["op_input"]
    layout_q = original_op["layout_q"]
    layout_kv = original_op["layout_kv"]

    for local_id, original_id in enumerate(batch_ids):
        for key in Q_BATCH_TENSORS:
            original_tensor = original_op.get(key)
            transformed_tensor = transformed_op.get(key)
            if original_tensor is None:
                assert transformed_tensor is None, f"{key}: None state changed"
                continue
            original_sample = _sample_from_layout(
                original_tensor,
                layout_q,
                original_op.get("cu_seqlens_q"),
                original_id,
            )
            transformed_sample = _sample_from_layout(
                transformed_tensor,
                layout_q,
                transformed_op.get("cu_seqlens_q"),
                local_id,
            )
            _assert_tensor_equal(
                original_sample,
                transformed_sample,
                f"sample={original_id}, input={key}",
            )

        for container_name in ("op_input", "metadata_input"):
            original_container = original[container_name]
            transformed_container = transformed[container_name]
            for key in BATCH_VECTOR_TENSORS:
                original_tensor = original_container.get(key)
                transformed_tensor = transformed_container.get(key)
                if original_tensor is None:
                    assert transformed_tensor is None, (
                        f"{container_name}.{key}: None state changed"
                    )
                    continue
                _assert_tensor_equal(
                    original_tensor[original_id],
                    transformed_tensor[local_id],
                    f"sample={original_id}, input={container_name}.{key}",
                )

        for prefix in ("ori", "cmp"):
            kv_key = f"{prefix}_kv"
            cu_key = f"cu_seqlens_{prefix}_kv"
            block_table_key = f"{prefix}_block_table"
            original_kv = original_op.get(kv_key)
            transformed_kv = transformed_op.get(kv_key)
            if original_kv is not None:
                if layout_kv == "PA_BBND":
                    assert transformed_kv is original_kv, (
                        f"{kv_key}: physical cache must remain unchanged"
                    )
                else:
                    original_cu = original_op.get(cu_key)
                    if original_cu is None:
                        original_cu = original["metadata_input"].get(cu_key)
                    transformed_cu = transformed_op.get(cu_key)
                    if transformed_cu is None:
                        transformed_cu = transformed["metadata_input"].get(cu_key)
                    _assert_tensor_equal(
                        _sample_from_layout(
                            original_kv, layout_kv, original_cu, original_id
                        ),
                        _sample_from_layout(
                            transformed_kv, layout_kv, transformed_cu, local_id
                        ),
                        f"sample={original_id}, input={kv_key}",
                    )

            original_table = original_op.get(block_table_key)
            transformed_table = transformed_op.get(block_table_key)
            if original_table is None:
                assert transformed_table is None, (
                    f"{block_table_key}: None state changed"
                )
            else:
                _assert_tensor_equal(
                    original_table[original_id],
                    transformed_table[local_id],
                    f"sample={original_id}, input={block_table_key}",
                )

    original_sinks = original_op.get("sinks")
    transformed_sinks = transformed_op.get("sinks")
    if original_sinks is None:
        assert transformed_sinks is None, "sinks: None state changed"
    else:
        _assert_tensor_equal(original_sinks, transformed_sinks, "input=sinks")


def _logical_output_sample(output, test_data, batch_id, is_lse=False):
    metadata_input = test_data["metadata_input"]
    op_input = test_data["op_input"]
    layout_q = op_input["layout_q"]
    valid_q = int(metadata_input["seqused_q"][batch_id].item())

    if layout_q == "BSND":
        if is_lse:
            return output[batch_id, :, :valid_q, :]
        return output[batch_id, :valid_q]

    cu_seqlens_q = op_input["cu_seqlens_q"]
    start = int(cu_seqlens_q[batch_id].item())
    end = start + valid_q
    if is_lse:
        return output[:, start:end, :]
    return output[start:end]


def tensor_md5(tensor):
    raw_bytes = (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.md5(raw_bytes).hexdigest()


def _flat_index_to_coordinate(flat_index, shape):
    coordinate = []
    for size in reversed(shape):
        coordinate.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(coordinate))


def _assert_same_md5(expected, actual, label):
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        pytest.fail(
            f"{label}: output metadata mismatch, "
            f"expected_dtype={expected.dtype}, actual_dtype={actual.dtype}, "
            f"expected_shape={tuple(expected.shape)}, "
            f"actual_shape={tuple(actual.shape)}"
        )

    expected_md5 = tensor_md5(expected)
    actual_md5 = tensor_md5(actual)
    if expected_md5 == actual_md5:
        return

    details = [
        f"{label}: MD5 mismatch",
        f"expected_md5={expected_md5}",
        f"actual_md5={actual_md5}",
        f"expected_shape={tuple(expected.shape)}",
        f"actual_shape={tuple(actual.shape)}",
    ]
    if expected.shape == actual.shape and expected.numel() > 0:
        mismatch = expected != actual
        mismatch_count = int(mismatch.sum().item())
        details.append(f"mismatch_count={mismatch_count}/{expected.numel()}")
        if mismatch_count:
            flat_index = int(mismatch.reshape(-1).nonzero()[0].item())
            details.append(
                f"first_mismatch={_flat_index_to_coordinate(flat_index, expected.shape)}"
            )
        if torch.is_floating_point(expected) and torch.is_floating_point(actual):
            max_abs = torch.nan_to_num(
                (expected.float() - actual.float()).abs(), nan=float("inf")
            ).max()
            details.append(f"max_abs_diff={float(max_abs.item())}")
    pytest.fail(", ".join(details))


def _assert_sample_outputs_equal(
    baseline_output,
    transformed_output,
    baseline_lse,
    transformed_lse,
    original_data,
    transformed_data,
    original_id,
    transformed_id,
    mode,
):
    expected = _logical_output_sample(
        baseline_output, original_data, original_id
    )
    actual = _logical_output_sample(
        transformed_output, transformed_data, transformed_id
    )
    _assert_same_md5(
        expected,
        actual,
        f"mode={mode}, sample={original_id}, output=attention",
    )

    if original_data["op_input"].get("return_softmax_lse", False):
        expected_lse = _logical_output_sample(
            baseline_lse, original_data, original_id, is_lse=True
        )
        actual_lse = _logical_output_sample(
            transformed_lse, transformed_data, transformed_id, is_lse=True
        )
        _assert_same_md5(
            expected_lse,
            actual_lse,
            f"mode={mode}, sample={original_id}, output=lse",
        )


def _run_npu(test_data):
    from batch import mixed_quant_sparse_flash_mla_process

    npu_output, _, _, npu_lse = (
        mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_ci(
            test_data, device_id=device_id
        )
    )
    cpu_output = npu_output.detach().cpu()
    cpu_lse = None
    if test_data["op_input"].get("return_softmax_lse", False):
        cpu_lse = npu_lse.detach().cpu()
    return cpu_output, cpu_lse


def _run_reorder_consistency(test_data, baseline_output, baseline_lse, seed):
    batch_size = int(test_data["metadata_input"]["batch_size"])
    batch_ids = make_batch_permutation(batch_size, seed)
    reordered_data = select_batches(test_data, batch_ids)
    assert_sample_inputs_unchanged(test_data, reordered_data, batch_ids)
    reordered_output, reordered_lse = _run_npu(reordered_data)

    for transformed_id, original_id in enumerate(batch_ids):
        _assert_sample_outputs_equal(
            baseline_output,
            reordered_output,
            baseline_lse,
            reordered_lse,
            test_data,
            reordered_data,
            original_id,
            transformed_id,
            mode="mode1-reorder",
        )


def _run_split_consistency(
    test_data,
    baseline_output,
    baseline_lse,
    split_count,
    seed,
    raw_groups=None,
):
    batch_size = int(test_data["metadata_input"]["batch_size"])
    if raw_groups is not None:
        split_groups = validate_batch_groups(
            parse_batch_groups(raw_groups), batch_size
        )
    else:
        split_groups = make_random_split_groups(batch_size, split_count, seed)

    for split_id, batch_ids in enumerate(split_groups):
        split_data = select_batches(test_data, batch_ids)
        assert_sample_inputs_unchanged(test_data, split_data, batch_ids)
        split_output, split_lse = _run_npu(split_data)

        for transformed_id, original_id in enumerate(batch_ids):
            _assert_sample_outputs_equal(
                baseline_output,
                split_output,
                baseline_lse,
                split_lse,
                test_data,
                split_data,
                original_id,
                transformed_id,
                mode=f"mode2-split-{split_id}",
            )


def run_batch_consistency_case(
    test_data, mode="all", seed=20260806, split_count=2, raw_groups=None
):
    _refresh_metadata_shape_fields(test_data)
    batch_size = int(test_data["metadata_input"]["batch_size"])
    if batch_size < 2:
        pytest.skip("batch consistency requires B >= 2")

    baseline_output, baseline_lse = _run_npu(test_data)
    if mode in ("1", "all"):
        _run_reorder_consistency(
            test_data, baseline_output, baseline_lse, seed
        )
    if mode in ("2", "all"):
        try:
            _run_split_consistency(
                test_data,
                baseline_output,
                baseline_lse,
                split_count,
                seed,
                raw_groups,
            )
        except ValueError as error:
            pytest.fail(f"mode2-split configuration is invalid: {error}")


@contextmanager
def batch_consistency_level(target_device_id=device_id):
    import torch_npu

    torch_npu.npu.set_device(target_device_id)
    old_level = torch_npu.npu._get_deterministic_level()
    torch_npu.npu.set_deterministic_level(1)
    assert torch_npu.npu._get_deterministic_level() == 1
    try:
        yield
    finally:
        torch_npu.npu.set_deterministic_level(old_level)


@pytest.fixture(scope="module")
def enable_batch_consistency():
    with batch_consistency_level():
        yield


@pytest.fixture(scope="module", params=testcase_files, ids=testcase_ids)
def baseline_case(request, enable_batch_consistency):
    test_data = torch.load(request.param, map_location="cpu")
    # Normalize derived metadata so .pt files generated before the TND
    # max_seqlen_q fix remain valid consistency baselines.
    _refresh_metadata_shape_fields(test_data)
    batch_size = int(test_data["metadata_input"]["batch_size"])
    if batch_size < 2:
        pytest.skip("batch consistency requires B >= 2")
    baseline_output, baseline_lse = _run_npu(test_data)
    return test_data, baseline_output, baseline_lse


@pytest.mark.batch_consistency
def test_batch_reorder_consistency(baseline_case, pytestconfig):
    if pytestconfig.getoption("consistency_mode") not in ("1", "all"):
        pytest.skip("batch reorder is disabled by --consistency-mode")

    test_data, baseline_output, baseline_lse = baseline_case
    seed = pytestconfig.getoption("consistency_seed")
    _run_reorder_consistency(
        test_data,
        baseline_output,
        baseline_lse,
        seed,
    )


@pytest.mark.batch_consistency
def test_batch_split_consistency(baseline_case, pytestconfig):
    if pytestconfig.getoption("consistency_mode") not in ("2", "all"):
        pytest.skip("batch split is disabled by --consistency-mode")

    test_data, baseline_output, baseline_lse = baseline_case
    raw_groups = pytestconfig.getoption("batch_groups")
    try:
        _run_split_consistency(
            test_data,
            baseline_output,
            baseline_lse,
            pytestconfig.getoption("split_count"),
            pytestconfig.getoption("consistency_seed"),
            raw_groups,
        )
    except ValueError as error:
        pytest.fail(f"mode2-split configuration is invalid: {error}")


def _synthetic_test_data(layout_q):
    seqused_q = torch.tensor([2, 1, 3, 2], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 2, 3, 6, 8], dtype=torch.int32)
    batch_size = len(seqused_q)
    if layout_q == "TND":
        q = torch.arange(16, dtype=torch.int32).reshape(8, 1, 2)
        sparse_indices = torch.arange(8, dtype=torch.int32).reshape(8, 1)
        topk_length = torch.arange(8, dtype=torch.int32)
        op_cu_seqlens_q = cu_seqlens_q
    else:
        q = torch.arange(32, dtype=torch.int32).reshape(4, 4, 1, 2)
        sparse_indices = torch.arange(16, dtype=torch.int32).reshape(4, 4, 1)
        topk_length = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        op_cu_seqlens_q = None

    seqused_ori_kv = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
    seqused_cmp_kv = torch.tensor([2, 3, 3, 4], dtype=torch.int32)
    cmp_residual_kv = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    ori_kv = torch.arange(48, dtype=torch.int32).reshape(6, 2, 1, 4)
    cmp_kv = torch.arange(32, dtype=torch.int32).reshape(4, 2, 1, 4)

    op_input = {
        "q": q,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "ori_sparse_indices": sparse_indices,
        "cmp_sparse_indices": sparse_indices + 100,
        "ori_topk_length": topk_length,
        "cmp_topk_length": topk_length + 100,
        "ori_block_table": torch.arange(8, dtype=torch.int32).reshape(4, 2),
        "cmp_block_table": torch.arange(8, 16, dtype=torch.int32).reshape(4, 2),
        "cu_seqlens_q": op_cu_seqlens_q,
        "seqused_q": seqused_q,
        "cu_seqlens_ori_kv": None,
        "cu_seqlens_cmp_kv": None,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "sinks": torch.arange(2, dtype=torch.int32),
        "layout_q": layout_q,
        "layout_kv": "PA_BBND",
        "return_softmax_lse": False,
    }
    metadata_input = {
        "batch_size": batch_size,
        "max_seqlen_q": 4,
        "max_seqlen_ori_kv": 7,
        "max_seqlen_cmp_kv": 4,
        "cu_seqlens_q": op_cu_seqlens_q,
        "seqused_q": seqused_q,
        "cu_seqlens_ori_kv": torch.tensor([0, 4, 9, 15, 22], dtype=torch.int32),
        "cu_seqlens_cmp_kv": torch.tensor([0, 2, 5, 8, 12], dtype=torch.int32),
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
    }
    return {
        "Testcase_Name": f"synthetic_{layout_q}",
        "params": {"Testcase_Name": f"synthetic_{layout_q}", "B": batch_size},
        "metadata_input": metadata_input,
        "op_input": op_input,
        "cpu_output": None,
        "cpu_lse": None,
    }


@pytest.mark.batch_consistency
@pytest.mark.parametrize("layout_q", ("BSND", "TND"))
def test_select_batches_preserves_sample_inputs(layout_q):
    test_data = _synthetic_test_data(layout_q)
    batch_ids = [2, 0]
    selected = select_batches(test_data, batch_ids)

    assert selected["original_batch_ids"] == batch_ids
    assert selected["metadata_input"]["batch_size"] == len(batch_ids)
    assert selected["op_input"]["ori_kv"] is test_data["op_input"]["ori_kv"]
    assert selected["op_input"]["cmp_kv"] is test_data["op_input"]["cmp_kv"]
    if layout_q == "TND":
        assert selected["op_input"]["cu_seqlens_q"].dtype == torch.int32
        assert selected["metadata_input"]["cu_seqlens_q"].dtype == torch.int32
    assert_sample_inputs_unchanged(test_data, selected, batch_ids)


@pytest.mark.batch_consistency
def test_batch_group_controls_are_reproducible():
    assert make_batch_permutation(4, 7) == make_batch_permutation(4, 7)
    groups = make_random_split_groups(8, 3, 7)
    assert groups == make_random_split_groups(8, 3, 7)
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    assert validate_batch_groups(parse_batch_groups("[[2,0],[3],[1]]"), 4) == [
        [0, 2],
        [3],
        [1],
    ]


@pytest.mark.batch_consistency
def test_shared_consistency_runner_reuses_generated_inputs(monkeypatch):
    calls = []

    def fake_run_npu(test_data):
        calls.append(test_data.get("original_batch_ids"))
        return test_data["op_input"]["q"].clone(), None

    monkeypatch.setitem(globals(), "_run_npu", fake_run_npu)
    run_batch_consistency_case(
        _synthetic_test_data("BSND"), mode="all", seed=7, split_count=2
    )

    assert calls[0] is None
    assert len(calls) == 4  # baseline + reorder + two split groups


@pytest.mark.batch_consistency
@pytest.mark.parametrize(
    "groups,batch_size",
    (
        ([[0, 1]], 2),
        ([[0], []], 1),
        ([[0], [0]], 2),
        ([[0], [2]], 2),
    ),
)
def test_validate_batch_groups_rejects_invalid_partition(groups, batch_size):
    with pytest.raises(ValueError):
        validate_batch_groups(groups, batch_size)

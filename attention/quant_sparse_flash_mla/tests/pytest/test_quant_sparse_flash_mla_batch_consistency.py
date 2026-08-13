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


testcase_path = Path(os.environ.get("QSMLA_PT_DIR", "qsmla_testcase"))
device_id = int(os.environ.get("QSMLA_DEVICE_ID", 0))
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
    if not isinstance(groups, list) or any(not isinstance(group, list) for group in groups):
        raise ValueError("batch_groups must be a JSON list of lists")
    if any(
        not isinstance(batch_id, int) or isinstance(batch_id, bool)
        for group in groups
        for batch_id in group
    ):
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
        (cu_seqlens.new_zeros(1), selected_lengths.cumsum(0, dtype=cu_seqlens.dtype))
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
        return int(default.item()) if isinstance(default, torch.Tensor) else int(default)
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
        metadata.get("seqused_ori_kv"), metadata.get("max_seqlen_ori_kv", 0)
    )
    metadata["max_seqlen_cmp_kv"] = _max_tensor_value(
        metadata.get("seqused_cmp_kv"), metadata.get("max_seqlen_cmp_kv", 0)
    )


def select_batches(test_data, batch_ids):
    """Select logical samples without regenerating quantized Q/KV data."""
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
            selected_op[key] = _select_tnd_tensor(tensor, original_q_cu, batch_ids)
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
        table_key = f"{prefix}_block_table"
        original_kv = original_op.get(kv_key)
        original_kv_cu = original_op.get(cu_key)
        if original_kv_cu is None:
            original_kv_cu = original_metadata.get(cu_key)
        if layout_kv == "PA_BBND":
            selected_op[kv_key] = original_kv
        elif layout_kv == "BSND":
            selected_op[kv_key] = _select_axis_zero(original_kv, batch_ids)
        elif layout_kv == "TND":
            selected_op[kv_key] = _select_tnd_tensor(original_kv, original_kv_cu, batch_ids)
        else:
            raise ValueError(f"unsupported layout_kv: {layout_kv}")
        selected_op[table_key] = _select_axis_zero(original_op.get(table_key), batch_ids)
        selected_op[cu_key] = _select_cu_seqlens(original_op.get(cu_key), batch_ids)
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
    start = int(cu_seqlens[batch_id].item())
    end = int(cu_seqlens[batch_id + 1].item())
    return tensor[start:end]


def _assert_tensor_equal(expected, actual, label):
    assert expected is not None and actual is not None, f"{label}: unexpected None"
    assert expected.dtype == actual.dtype, f"{label}: dtype changed"
    assert expected.shape == actual.shape, f"{label}: shape changed"
    expected_bytes = expected.contiguous().reshape(-1).view(torch.uint8)
    actual_bytes = actual.contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(expected_bytes, actual_bytes), f"{label}: tensor bytes changed"


def assert_sample_inputs_unchanged(original, transformed, batch_ids):
    original_op = original["op_input"]
    transformed_op = transformed["op_input"]
    layout_q = original_op["layout_q"]
    layout_kv = original_op["layout_kv"]
    for local_id, original_id in enumerate(batch_ids):
        for key in Q_BATCH_TENSORS:
            tensor = original_op.get(key)
            transformed_tensor = transformed_op.get(key)
            if tensor is None:
                assert transformed_tensor is None
                continue
            _assert_tensor_equal(
                _sample_from_layout(
                    tensor, layout_q, original_op.get("cu_seqlens_q"), original_id
                ),
                _sample_from_layout(
                    transformed_tensor, layout_q,
                    transformed_op.get("cu_seqlens_q"), local_id,
                ),
                f"sample={original_id}, input={key}",
            )
        for container_name in ("op_input", "metadata_input"):
            for key in BATCH_VECTOR_TENSORS:
                tensor = original[container_name].get(key)
                transformed_tensor = transformed[container_name].get(key)
                if tensor is None:
                    assert transformed_tensor is None
                else:
                    _assert_tensor_equal(
                        tensor[original_id], transformed_tensor[local_id],
                        f"sample={original_id}, input={container_name}.{key}",
                    )
        for prefix in ("ori", "cmp"):
            kv_key = f"{prefix}_kv"
            cu_key = f"cu_seqlens_{prefix}_kv"
            table_key = f"{prefix}_block_table"
            original_kv = original_op.get(kv_key)
            transformed_kv = transformed_op.get(kv_key)
            if layout_kv == "PA_BBND":
                assert transformed_kv is original_kv
            elif original_kv is not None:
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
            table = original_op.get(table_key)
            transformed_table = transformed_op.get(table_key)
            if table is None:
                assert transformed_table is None
            else:
                _assert_tensor_equal(
                    table[original_id], transformed_table[local_id],
                    f"sample={original_id}, input={table_key}",
                )
    for key in ("sinks", "q_descale", "ori_kv_descale", "cmp_kv_descale"):
        tensor = original_op.get(key)
        transformed_tensor = transformed_op.get(key)
        if tensor is None:
            assert transformed_tensor is None
        else:
            _assert_tensor_equal(tensor, transformed_tensor, f"input={key}")


def _logical_output_sample(output, test_data, batch_id, is_lse=False):
    metadata = test_data["metadata_input"]
    op_input = test_data["op_input"]
    valid_q = int(metadata["seqused_q"][batch_id].item())
    if op_input["layout_q"] == "BSND":
        return output[batch_id, :, :valid_q, :] if is_lse else output[batch_id, :valid_q]
    cu_seqlens_q = op_input["cu_seqlens_q"]
    start = int(cu_seqlens_q[batch_id].item())
    end = start + valid_q
    return output[:, start:end, :] if is_lse else output[start:end]


def tensor_md5(tensor):
    raw_bytes = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.md5(raw_bytes).hexdigest()


def _mismatch_details(expected, actual, label):
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return (
            f"{label}: output metadata mismatch, expected_dtype={expected.dtype}, "
            f"actual_dtype={actual.dtype}, expected_shape={tuple(expected.shape)}, "
            f"actual_shape={tuple(actual.shape)}"
        )
    expected_md5 = tensor_md5(expected)
    actual_md5 = tensor_md5(actual)
    if expected_md5 == actual_md5:
        return None
    mismatch = expected != actual
    mismatch_count = int(mismatch.sum().item())
    details = (
        f"{label}: MD5 mismatch, expected_md5={expected_md5}, actual_md5={actual_md5}, "
        f"shape={tuple(expected.shape)}, mismatch_count={mismatch_count}/{expected.numel()}"
    )
    if mismatch_count:
        coordinate = tuple(int(value) for value in mismatch.nonzero()[0].tolist())
        details += f", first_mismatch={coordinate}"
    if torch.is_floating_point(expected):
        max_abs = torch.nan_to_num((expected.float() - actual.float()).abs(), nan=float("inf")).max()
        details += f", max_abs_diff={float(max_abs.item())}"
    return details


def _assert_sample_outputs_equal(
    baseline_output, transformed_output, baseline_lse, transformed_lse,
    original_data, transformed_data, original_id, transformed_id, mode,
):
    failures = []
    failure = _mismatch_details(
        _logical_output_sample(baseline_output, original_data, original_id),
        _logical_output_sample(transformed_output, transformed_data, transformed_id),
        f"mode={mode}, sample={original_id}, output=attention",
    )
    if failure:
        failures.append(failure)
    if original_data["op_input"].get("return_softmax_lse", False):
        failure = _mismatch_details(
            _logical_output_sample(baseline_lse, original_data, original_id, True),
            _logical_output_sample(transformed_lse, transformed_data, transformed_id, True),
            f"mode={mode}, sample={original_id}, output=lse",
        )
        if failure:
            failures.append(failure)
    if failures:
        pytest.fail(" | ".join(failures))


def _run_npu(test_data):
    from batch import quant_sparse_flash_mla_process
    npu_output, _, _, npu_lse = quant_sparse_flash_mla_process.test_qsmla_quant_process_ci(
        test_data, device_id=device_id
    )
    output = npu_output.detach().cpu()
    lse = npu_lse.detach().cpu() if test_data["op_input"].get("return_softmax_lse") else None
    return output, lse


def _run_reorder_consistency(test_data, baseline_output, baseline_lse, seed):
    batch_ids = make_batch_permutation(int(test_data["metadata_input"]["batch_size"]), seed)
    transformed = select_batches(test_data, batch_ids)
    assert_sample_inputs_unchanged(test_data, transformed, batch_ids)
    output, lse = _run_npu(transformed)
    for transformed_id, original_id in enumerate(batch_ids):
        _assert_sample_outputs_equal(
            baseline_output, output, baseline_lse, lse, test_data, transformed,
            original_id, transformed_id, "mode1-reorder",
        )


def _run_split_consistency(
    test_data, baseline_output, baseline_lse, split_count, seed, raw_groups=None,
):
    batch_size = int(test_data["metadata_input"]["batch_size"])
    groups = (
        validate_batch_groups(parse_batch_groups(raw_groups), batch_size)
        if raw_groups is not None
        else make_random_split_groups(batch_size, split_count, seed)
    )
    for split_id, batch_ids in enumerate(groups):
        transformed = select_batches(test_data, batch_ids)
        assert_sample_inputs_unchanged(test_data, transformed, batch_ids)
        output, lse = _run_npu(transformed)
        for transformed_id, original_id in enumerate(batch_ids):
            _assert_sample_outputs_equal(
                baseline_output, output, baseline_lse, lse, test_data, transformed,
                original_id, transformed_id, f"mode2-split-{split_id}",
            )


def run_batch_consistency_case(
    test_data, mode="all", seed=20260806, split_count=2, raw_groups=None,
):
    _refresh_metadata_shape_fields(test_data)
    if int(test_data["metadata_input"]["batch_size"]) < 2:
        pytest.skip("batch consistency requires B >= 2")
    baseline_output, baseline_lse = _run_npu(test_data)
    if mode in ("1", "all"):
        _run_reorder_consistency(test_data, baseline_output, baseline_lse, seed)
    if mode in ("2", "all"):
        try:
            _run_split_consistency(
                test_data, baseline_output, baseline_lse, split_count, seed, raw_groups
            )
        except ValueError as error:
            pytest.fail(f"mode2-split configuration is invalid: {error}")


@contextmanager
def batch_consistency_level(target_device_id=device_id):
    import torch_npu
    torch_npu.npu.set_device(target_device_id)
    old_level = torch_npu.npu._get_deterministic_level()
    torch_npu.npu.set_deterministic_level(3)
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
    test_data = torch.load(request.param, map_location="cpu", weights_only=False)
    _refresh_metadata_shape_fields(test_data)
    if int(test_data["metadata_input"]["batch_size"]) < 2:
        pytest.skip("batch consistency requires B >= 2")
    baseline_output, baseline_lse = _run_npu(test_data)
    return test_data, baseline_output, baseline_lse


@pytest.mark.batch_consistency
def test_batch_reorder_consistency(baseline_case, pytestconfig):
    if pytestconfig.getoption("consistency_mode") not in ("1", "all"):
        pytest.skip("batch reorder is disabled by --consistency-mode")
    test_data, output, lse = baseline_case
    _run_reorder_consistency(
        test_data, output, lse, pytestconfig.getoption("consistency_seed")
    )


@pytest.mark.batch_consistency
def test_batch_split_consistency(baseline_case, pytestconfig):
    if pytestconfig.getoption("consistency_mode") not in ("2", "all"):
        pytest.skip("batch split is disabled by --consistency-mode")
    test_data, output, lse = baseline_case
    try:
        _run_split_consistency(
            test_data, output, lse, pytestconfig.getoption("split_count"),
            pytestconfig.getoption("consistency_seed"),
            pytestconfig.getoption("batch_groups"),
        )
    except ValueError as error:
        pytest.fail(f"mode2-split configuration is invalid: {error}")


@pytest.mark.batch_consistency
def test_batch_group_controls_are_reproducible():
    assert make_batch_permutation(4, 7) == make_batch_permutation(4, 7)
    assert make_random_split_groups(8, 3, 7) == make_random_split_groups(8, 3, 7)
    assert validate_batch_groups(parse_batch_groups("[[2,0],[3],[1]]"), 4) == [
        [0, 2], [3], [1]
    ]


@pytest.mark.batch_consistency
@pytest.mark.parametrize(
    "groups,batch_size", (([[0, 1]], 2), ([[0], []], 1), ([[0], [0]], 2), ([[0], [2]], 2)),
)
def test_validate_batch_groups_rejects_invalid_partition(groups, batch_size):
    with pytest.raises(ValueError):
        validate_batch_groups(groups, batch_size)

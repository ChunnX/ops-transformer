import csv
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import compare as compare_module
import moe_train_test
from cache import load_case_cache, save_case_cache
from compare import exact_feature_equal, exact_tensor_equal
from golden import (
    GoldenHandle,
    feature_main,
    feature_scales,
    gen_combine_golden_cpu,
    gen_dispatch_golden_cpu,
    expected_dispatch_rows_from_metadata,
)


def _is_tuple_feature(x):
    return isinstance(x, tuple)


def _main(x):
    return x[0] if _is_tuple_feature(x) else x


def _scales(x):
    return x[1] if _is_tuple_feature(x) else None


def _cat_feature(features):
    if _is_tuple_feature(features[0]):
        return torch.cat([_main(x) for x in features], dim=0), torch.cat([_scales(x) for x in features], dim=0)
    return torch.cat(features, dim=0)


def _index_feature(feature, idx):
    if _is_tuple_feature(feature):
        return _main(feature).index_select(0, idx), _scales(feature).index_select(0, idx)
    return feature.index_select(0, idx)


def _slice_feature(feature, rows):
    if _is_tuple_feature(feature):
        return _main(feature)[:rows], _scales(feature)[:rows]
    return feature[:rows]


def _rank_token_from_lengths(lengths, global_rows):
    src_rank = torch.empty_like(global_rows, dtype=torch.int64)
    token_idx = torch.empty_like(global_rows, dtype=torch.int64)
    start = 0
    for rank, length in enumerate(lengths):
        end = start + int(length)
        mask = (global_rows >= start) & (global_rows < end)
        src_rank[mask] = rank
        token_idx[mask] = global_rows[mask] - start
        start = end
    return src_rank, token_idx


def _rank_token_from_x_per_rank(x_per_rank, global_rows):
    return _rank_token_from_lengths([_main(x).shape[0] for x in x_per_rank], global_rows)


def _reference_dispatch_by_argsort(x_per_rank, topk_idx_per_rank, topk_weights_per_rank, num_experts):
    """参考 moe_operator_test_v3v4_noq.py 中 moe_dispatch 的 EP-only 子集。"""
    ep_world_size = len(x_per_rank)
    num_local_experts = num_experts // ep_world_size
    num_topk = topk_idx_per_rank[0].shape[1]
    global_x = _cat_feature(x_per_rank)
    global_topk_idx = torch.cat(topk_idx_per_rank, dim=0)
    global_topk_weights = None if topk_weights_per_rank is None else torch.cat(topk_weights_per_rank, dim=0)

    valid_mask = global_topk_idx.flatten() >= 0
    flat_experts = global_topk_idx.flatten()
    valid_flat_indices = torch.arange(flat_experts.numel(), dtype=torch.int64)[valid_mask]
    sorted_valid_order = torch.argsort(flat_experts[valid_mask], stable=True)
    sorted_col_idx = valid_flat_indices[sorted_valid_order]
    sorted_expert = flat_experts[sorted_col_idx]
    src_token = sorted_col_idx // num_topk
    topk_pos = sorted_col_idx % num_topk
    expanded_x = _index_feature(global_x, src_token)
    expanded_w = None if global_topk_weights is None else global_topk_weights.flatten()[sorted_col_idx]
    counts = torch.bincount(sorted_expert, minlength=num_experts).to(torch.int64)

    result = []
    for rank in range(ep_world_size):
        lo = rank * num_local_experts
        hi = (rank + 1) * num_local_experts
        rank_mask = (sorted_expert >= lo) & (sorted_expert < hi)
        rank_rows = rank_mask.nonzero().flatten()
        local_counts = counts[lo:hi]
        if rank_rows.numel() == 0:
            local_metadata = torch.empty((0, 3), dtype=torch.int32)
        else:
            src_rank, token_idx = _rank_token_from_x_per_rank(x_per_rank, src_token[rank_rows])
            local_metadata = torch.stack(
                (
                    src_rank.to(torch.int32),
                    token_idx.to(torch.int32),
                    topk_pos[rank_rows].to(torch.int32),
                ),
                dim=1,
            )
        local_w = None if expanded_w is None else expanded_w[rank_rows]
        result.append((_index_feature(expanded_x, rank_rows), local_w, local_metadata, local_counts))
    return result


def _reference_decode_vars(x_per_rank, topk_idx_per_rank, topk_weights_per_rank, num_experts):
    """显式构造 decode 脚本 golden 变量，供报告中的变量映射做 UT 约束。"""
    num_topk = topk_idx_per_rank[0].shape[1]
    global_x = _cat_feature(x_per_rank)
    global_topk_idx = torch.cat(topk_idx_per_rank, dim=0)
    global_topk_weights = None if topk_weights_per_rank is None else torch.cat(topk_weights_per_rank, dim=0)

    valid_mask = global_topk_idx.flatten() >= 0
    valid_flat_indices = torch.arange(global_topk_idx.numel(), dtype=torch.int64)[valid_mask]
    sorted_valid_order = torch.argsort(global_topk_idx.flatten()[valid_mask], stable=True)
    sorted_col_idx = valid_flat_indices[sorted_valid_order]
    dst_row_to_src_row = sorted_col_idx // num_topk
    expanded_x = _index_feature(global_x, dst_row_to_src_row)
    expanded_topk_weights = None if global_topk_weights is None else global_topk_weights.flatten()[sorted_col_idx]
    golden_moe_token = torch.bincount(global_topk_idx.flatten()[sorted_col_idx], minlength=num_experts).to(torch.int64)
    src_rank, token_idx = _rank_token_from_x_per_rank(x_per_rank, dst_row_to_src_row)
    metadata = torch.stack(
        (
            src_rank.to(torch.int32),
            token_idx.to(torch.int32),
            (sorted_col_idx % num_topk).to(torch.int32),
        ),
        dim=1,
    )
    return {
        "expanded_x": expanded_x,
        "sorted_col_idx": sorted_col_idx,
        "dst_row_to_src_row": dst_row_to_src_row,
        "expanded_topk_weights": expanded_topk_weights,
        "golden_moe_token": golden_moe_token,
        "metadata": metadata,
    }


def _reference_dst_slots_and_payload_counts(topk_idx_per_rank, num_experts, num_max_tokens_per_rank):
    ep_world_size = len(topk_idx_per_rank)
    num_local_experts = num_experts // ep_world_size
    dst_slots = [torch.full_like(t, -1, dtype=torch.int32) for t in topk_idx_per_rank]
    payload_counts = torch.zeros((ep_world_size, ep_world_size), dtype=torch.int32)
    for src_rank, topk_idx in enumerate(topk_idx_per_rank):
        slot_counter = [0 for _ in range(ep_world_size)]
        for token_idx in range(topk_idx.shape[0]):
            seen = {}
            for topk_pos in range(topk_idx.shape[1]):
                expert = int(topk_idx[token_idx, topk_pos].item())
                if expert < 0:
                    continue
                dst_rank = expert // num_local_experts
                if dst_rank not in seen:
                    slot = slot_counter[dst_rank]
                    if slot >= num_max_tokens_per_rank:
                        raise AssertionError("payload slot overflow in reference")
                    seen[dst_rank] = slot
                    slot_counter[dst_rank] += 1
                    payload_counts[dst_rank, src_rank] += 1
                dst_slots[src_rank][token_idx, topk_pos] = seen[dst_rank]
    return dst_slots, payload_counts


def _reference_combine_by_metadata(expanded_x_per_rank, handles, topk_weights_per_rank=None):
    ep_world_size = len(expanded_x_per_rank)
    combined_x_acc = []
    combined_topk_weights = []
    for rank, handle in enumerate(handles):
        rows = handle.topk_idx.shape[0]
        main = torch.zeros((rows, feature_main(expanded_x_per_rank[rank]).shape[1]), dtype=torch.float32)
        scales = feature_scales(expanded_x_per_rank[rank])
        if scales is not None:
            combined_x_acc.append((main, torch.zeros((rows, scales.shape[1]), dtype=torch.float32)))
        else:
            combined_x_acc.append(main)
        if topk_weights_per_rank is None:
            combined_topk_weights.append(None)
        else:
            combined_topk_weights.append(torch.zeros_like(handle.topk_idx, dtype=topk_weights_per_rank[rank].dtype))

    combine_entries = []
    for sender_rank, feature in enumerate(expanded_x_per_rank):
        valid_rows = handles[sender_rank].num_valid_expanded_tokens
        for row in range(valid_rows):
            src_rank, token_idx, topk_pos = [int(v) for v in handles[sender_rank].recv_src_metadata[row, :3].tolist()]
            if src_rank < 0:
                continue
            combine_entries.append((src_rank, token_idx, topk_pos, sender_rank, row))
    for src_rank, token_idx, topk_pos, sender_rank, row in sorted(combine_entries):
        feature = expanded_x_per_rank[sender_rank]
        dst_main = feature_main(combined_x_acc[src_rank])
        src_main = feature_main(feature)
        dst_main[token_idx] += src_main[row].float()
        dst_scales = feature_scales(combined_x_acc[src_rank])
        src_scales = feature_scales(feature)
        if dst_scales is not None and src_scales is not None:
            dst_scales[token_idx] += src_scales[row].float()
        if topk_weights_per_rank is not None:
            combined_topk_weights[src_rank][token_idx, topk_pos] = topk_weights_per_rank[sender_rank][row]
    combined_x = []
    for rank, acc in enumerate(combined_x_acc):
        main = feature_main(acc).to(feature_main(expanded_x_per_rank[rank]).dtype)
        scales = feature_scales(acc)
        template_scales = feature_scales(expanded_x_per_rank[rank])
        if scales is None or template_scales is None:
            combined_x.append(main)
        else:
            combined_x.append((main, scales.to(template_scales.dtype)))
    return combined_x, combined_topk_weights


def _reference_ep_recv_count(topk_idx_per_rank, num_experts):
    ep_world_size = len(topk_idx_per_rank)
    num_local_experts = num_experts // ep_world_size
    per_rank = []
    for dst_rank in range(ep_world_size):
        counts = torch.zeros((num_local_experts, ep_world_size), dtype=torch.int32)
        for src_rank, topk_idx in enumerate(topk_idx_per_rank):
            for token_idx in range(topk_idx.shape[0]):
                for topk_pos in range(topk_idx.shape[1]):
                    expert = int(topk_idx[token_idx, topk_pos].item())
                    if expert < 0:
                        continue
                    if expert // num_local_experts == dst_rank:
                        counts[expert % num_local_experts, src_rank] += 1
        per_rank.append(counts.cumsum(dim=1, dtype=torch.int32))
    return per_rank


def _train_ep_recv_count_from_handle(handle, world_size):
    num_local_experts = handle.num_recv_tokens_per_expert.shape[0]
    counts = torch.zeros((num_local_experts, world_size), dtype=torch.int32)
    start = 0
    for local_expert, token_num in enumerate(handle.num_recv_tokens_per_expert.tolist()):
        end = start + int(token_num)
        for row in handle.recv_src_metadata[start:end]:
            src_rank = int(row[0].item())
            if src_rank >= 0:
                counts[local_expert, src_rank] += 1
        start = end
    return counts.cumsum(dim=1, dtype=torch.int32)


def _reference_decode_combine_pure_sum(decode, topk_idx_per_rank, world_size):
    token_lengths = [topk_idx.shape[0] for topk_idx in topk_idx_per_rank]
    hidden = feature_main(decode["expanded_x"]).shape[1]
    output_dtype = feature_main(decode["expanded_x"]).dtype
    output = [
        torch.zeros((token_lengths[rank], hidden), dtype=torch.float32)
        for rank in range(world_size)
    ]
    x_k = feature_main(decode["expanded_x"]).index_select(0, torch.argsort(decode["sorted_col_idx"]))
    valid_flat_indices = set(int(i) for i in decode["sorted_col_idx"].tolist())
    expand_idx = 0
    for flat_idx in range(sum(token_lengths) * topk_idx_per_rank[0].shape[1]):
        if flat_idx not in valid_flat_indices:
            continue
        token_global = torch.tensor([flat_idx // topk_idx_per_rank[0].shape[1]], dtype=torch.int64)
        src_rank, token_idx = _rank_token_from_lengths(token_lengths, token_global)
        rank = int(src_rank[0].item())
        row = int(token_idx[0].item())
        output[rank][row] += x_k[expand_idx].float()
        expand_idx += 1
    return [item.to(output_dtype) for item in output]


def _make_scales(rank, num_tokens, dim_scales, scales_dtype="float32"):
    if scales_dtype in ("fp8_e8m0", "float8_e8m0"):
        start = rank * 100
        return torch.arange(start, start + num_tokens * dim_scales, dtype=torch.int32).reshape(num_tokens, dim_scales).to(torch.uint8)
    return torch.arange(rank * 100, rank * 100 + num_tokens * dim_scales, dtype=torch.float32).reshape(num_tokens, dim_scales)


def _make_case(
    world_size,
    num_tokens,
    hidden,
    num_topk,
    num_experts,
    dtype=torch.bfloat16,
    dim_scales=0,
    scales_dtype="float32",
    include_invalid=False,
    num_tokens_per_rank=None,
):
    x_per_rank = []
    topk_idx_per_rank = []
    topk_weights_per_rank = []
    if num_tokens_per_rank is None:
        num_tokens_per_rank = [num_tokens for _ in range(world_size)]
    for rank in range(world_size):
        real_num_tokens = int(num_tokens_per_rank[rank])
        base = torch.arange(rank * 1000, rank * 1000 + num_tokens * hidden, dtype=torch.bfloat16).reshape(num_tokens, hidden)
        if "float8" in str(dtype):
            x = base.to(dtype)
            x_per_rank.append((x[:real_num_tokens], _make_scales(rank, num_tokens, dim_scales, scales_dtype)[:real_num_tokens]))
        else:
            x_per_rank.append(base.to(dtype)[:real_num_tokens])
        rows = []
        for token_idx in range(num_tokens):
            start = (rank * num_tokens + token_idx) % num_experts
            experts = [(start + i * (rank + 1)) % num_experts for i in range(num_topk)]
            if include_invalid and (rank + token_idx) % 3 == 0:
                experts[-1] = -1
            rows.append(torch.tensor(experts, dtype=torch.int32))
        topk_idx_per_rank.append(torch.stack(rows, dim=0)[:real_num_tokens])
        weights = torch.arange(
            rank * 10,
            rank * 10 + num_tokens * num_topk,
            dtype=torch.float32,
        ).reshape(num_tokens, num_topk) / 10.0
        topk_weights_per_rank.append(weights[:real_num_tokens])
    return x_per_rank, topk_idx_per_rank, topk_weights_per_rank


def _assert_dispatch_matches_reference(testcase, golden, reference, topk_idx_per_rank, num_experts, num_max_tokens_per_rank, do_cpu_sync):
    ref_dst_slots, ref_payload_counts = _reference_dst_slots_and_payload_counts(
        topk_idx_per_rank, num_experts, num_max_tokens_per_rank
    )
    for rank, dispatch in enumerate(golden):
        ref_x, ref_w, ref_meta, ref_counts = reference[rank]
        valid_rows = ref_meta.shape[0]
        testcase.assertTrue(exact_feature_equal(f"rank{rank}.recv_x", _slice_feature(dispatch.recv_x, valid_rows), ref_x)[0])
        if ref_w is None:
            testcase.assertIsNone(dispatch.recv_topk_weights)
        else:
            testcase.assertTrue(exact_tensor_equal(f"rank{rank}.recv_topk_weights", dispatch.recv_topk_weights[:valid_rows], ref_w)[0])
        testcase.assertTrue(exact_tensor_equal(f"rank{rank}.recv_src_metadata", dispatch.handle.recv_src_metadata[:valid_rows, :3], ref_meta)[0])
        testcase.assertTrue(exact_tensor_equal(f"rank{rank}.num_recv_tokens_per_expert", dispatch.handle.num_recv_tokens_per_expert, ref_counts)[0])
        testcase.assertTrue(exact_tensor_equal(f"rank{rank}.dst_buffer_slot_idx", dispatch.handle.dst_buffer_slot_idx, ref_dst_slots[rank])[0])
        for metadata_row in dispatch.handle.recv_src_metadata[:valid_rows]:
            src_rank, token_idx, topk_pos, slot_idx = [int(v) for v in metadata_row.tolist()]
            testcase.assertEqual(slot_idx, int(ref_dst_slots[src_rank][token_idx, topk_pos].item()))
        testcase.assertTrue(exact_tensor_equal(f"rank{rank}.num_recv_payloads_per_rank", dispatch.handle.num_recv_payloads_per_rank, ref_payload_counts[rank])[0])
        testcase.assertTrue(exact_tensor_equal(f"rank{rank}.topk_idx", dispatch.handle.topk_idx, topk_idx_per_rank[rank])[0])
        if not do_cpu_sync:
            tail = dispatch.handle.recv_src_metadata[valid_rows:]
            if tail.numel() > 0:
                testcase.assertEqual(tail.unique().tolist(), [-1])


class TestMoeTrainGolden(unittest.TestCase):
    def test_full_golden_matches_reference_for_multiple_shapes(self):
        cases = [
            {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": False},
            {"world_size": 2, "num_tokens": 4, "hidden": 5, "num_topk": 3, "num_experts": 6, "include_invalid": True},
            {"world_size": 3, "num_tokens": 4, "hidden": 3, "num_topk": 2, "num_experts": 6, "include_invalid": True},
        ]
        for case in cases:
            with self.subTest(case=case):
                x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
                num_max_tokens_per_rank = case["num_tokens"]
                dispatch = gen_dispatch_golden_cpu(
                    x_per_rank,
                    topk_idx_per_rank,
                    topk_weights_per_rank,
                    num_experts=case["num_experts"],
                    num_max_tokens_per_rank=num_max_tokens_per_rank,
                    do_cpu_sync=True,
                )
                reference = _reference_dispatch_by_argsort(
                    x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"]
                )
                _assert_dispatch_matches_reference(
                    self,
                    dispatch,
                    reference,
                    topk_idx_per_rank,
                    case["num_experts"],
                    num_max_tokens_per_rank,
                    do_cpu_sync=True,
                )

                combine = gen_combine_golden_cpu(
                    [d.recv_x for d in dispatch],
                    [d.handle for d in dispatch],
                    topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
                )
                ref_combined_x, ref_combined_w = _reference_combine_by_metadata(
                    [d.recv_x for d in dispatch],
                    [d.handle for d in dispatch],
                    topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
                )
                for rank, item in enumerate(combine):
                    self.assertTrue(exact_feature_equal(f"combine.rank{rank}", item.combined_x, ref_combined_x[rank])[0])
                    self.assertTrue(exact_tensor_equal(f"combine.topk.rank{rank}", item.combined_topk_weights, ref_combined_w[rank])[0])

    def test_full_golden_matches_reference_for_variable_bs(self):
        case = {"world_size": 3, "num_tokens": 5, "hidden": 4, "num_topk": 2, "num_experts": 6, "include_invalid": True}
        real_num_tokens_per_rank = [5, 2, 4]
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(
            **case,
            num_tokens_per_rank=real_num_tokens_per_rank,
        )
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=False,
        )
        reference = _reference_dispatch_by_argsort(
            x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"]
        )
        _assert_dispatch_matches_reference(
            self,
            dispatch,
            reference,
            topk_idx_per_rank,
            case["num_experts"],
            case["num_tokens"],
            do_cpu_sync=False,
        )

        combine = gen_combine_golden_cpu(
            [d.recv_x for d in dispatch],
            [d.handle for d in dispatch],
            topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
        )
        ref_combined_x, ref_combined_w = _reference_combine_by_metadata(
            [d.recv_x for d in dispatch],
            [d.handle for d in dispatch],
            topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
        )
        for rank, item in enumerate(combine):
            self.assertEqual(dispatch[rank].handle.topk_idx.shape, (real_num_tokens_per_rank[rank], case["num_topk"]))
            self.assertEqual(dispatch[rank].handle.dst_buffer_slot_idx.shape, (real_num_tokens_per_rank[rank], case["num_topk"]))
            self.assertEqual(feature_main(item.combined_x).shape[0], real_num_tokens_per_rank[rank])
            self.assertTrue(exact_feature_equal(f"variable.combine.rank{rank}", item.combined_x, ref_combined_x[rank])[0])
            self.assertTrue(exact_tensor_equal(f"variable.combine.topk.rank{rank}", item.combined_topk_weights, ref_combined_w[rank])[0])

    def test_train_golden_maps_decode_variables_explicitly(self):
        case = {"world_size": 2, "num_tokens": 4, "hidden": 3, "num_topk": 2, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        decode = _reference_decode_vars(x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"])
        num_local_experts = case["num_experts"] // case["world_size"]

        for rank, item in enumerate(dispatch):
            expert_lo = rank * num_local_experts
            expert_hi = (rank + 1) * num_local_experts
            expert_ids_sorted = torch.cat(topk_idx_per_rank, dim=0).flatten()[decode["sorted_col_idx"]]
            mask = (expert_ids_sorted >= expert_lo) & (expert_ids_sorted < expert_hi)
            ref_x = _index_feature(decode["expanded_x"], mask.nonzero().flatten())
            ref_meta = decode["metadata"][mask]
            ref_weights = decode["expanded_topk_weights"][mask]
            self.assertTrue(exact_feature_equal(f"decode.expanded_x.rank{rank}", item.recv_x, ref_x)[0])
            self.assertTrue(exact_tensor_equal(f"decode.metadata.rank{rank}", item.handle.recv_src_metadata[:, :3], ref_meta)[0])
            self.assertTrue(exact_tensor_equal(f"decode.weights.rank{rank}", item.recv_topk_weights, ref_weights)[0])
            self.assertTrue(
                exact_tensor_equal(
                    f"decode.golden_moe_token.rank{rank}",
                    item.handle.num_recv_tokens_per_expert,
                    decode["golden_moe_token"][expert_lo:expert_hi],
                )[0]
            )
            self.assertTrue(
                exact_tensor_equal(
                    f"decode.golden_ep_recv_count.rank{rank}",
                    _train_ep_recv_count_from_handle(item.handle, case["world_size"]),
                    _reference_ep_recv_count(topk_idx_per_rank, case["num_experts"])[rank],
                )[0]
            )

    def test_train_combine_matches_decode_middle_idx_with_unit_scales(self):
        case = {"world_size": 2, "num_tokens": 4, "hidden": 3, "num_topk": 2, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        decode = _reference_decode_vars(x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"])
        ref = _reference_decode_combine_pure_sum(decode, topk_idx_per_rank, case["world_size"])
        for rank, item in enumerate(combine):
            self.assertTrue(exact_feature_equal(f"decode.middle_idx.combine.rank{rank}", item.combined_x, ref[rank])[0])

    def test_combine_accumulates_in_float32_before_cast(self):
        def make_handle(metadata):
            topk_idx = torch.zeros((1, 3), dtype=torch.int32)
            return GoldenHandle(
                dst_buffer_slot_idx=topk_idx.clone(),
                recv_src_metadata=torch.tensor(metadata, dtype=torch.int32),
                num_recv_payloads_per_rank=torch.zeros((3,), dtype=torch.int32),
                num_recv_tokens_per_expert=torch.tensor([1], dtype=torch.int64),
                topk_idx=topk_idx,
                num_experts=3,
                expert_alignment=1,
                num_max_tokens_per_rank=1,
                num_expanded_tokens=1,
            )

        expanded_x_per_rank = [
            torch.tensor([[10000.0]], dtype=torch.float16),
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[-10000.0]], dtype=torch.float16),
        ]
        handles = [
            make_handle([[0, 0, 0]]),
            make_handle([[0, 0, 1]]),
            make_handle([[0, 0, 2]]),
        ]

        combine = gen_combine_golden_cpu(expanded_x_per_rank, handles)

        self.assertEqual(float(feature_main(combine[0].combined_x)[0, 0].item()), 1.0)

    def test_combine_accumulates_in_decode_middle_idx_order(self):
        def make_handle(metadata):
            topk_idx = torch.zeros((1, 3), dtype=torch.int32)
            return GoldenHandle(
                dst_buffer_slot_idx=topk_idx.clone(),
                recv_src_metadata=torch.tensor(metadata, dtype=torch.int32),
                num_recv_payloads_per_rank=torch.zeros((3,), dtype=torch.int32),
                num_recv_tokens_per_expert=torch.tensor([1], dtype=torch.int64),
                topk_idx=topk_idx,
                num_experts=3,
                expert_alignment=1,
                num_max_tokens_per_rank=1,
                num_expanded_tokens=1,
            )

        expanded_x_per_rank = [
            torch.tensor([[1e20]], dtype=torch.bfloat16),
            torch.tensor([[1.0]], dtype=torch.bfloat16),
            torch.tensor([[-1e20]], dtype=torch.bfloat16),
        ]
        handles = [
            make_handle([[0, 0, 0]]),
            make_handle([[0, 0, 2]]),
            make_handle([[0, 0, 1]]),
        ]

        combine = gen_combine_golden_cpu(expanded_x_per_rank, handles)

        self.assertEqual(float(feature_main(combine[0].combined_x)[0, 0].item()), 1.0)

    def test_full_golden_without_dispatch_topk_weights_cached_branch(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            None,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        reference = _reference_dispatch_by_argsort(x_per_rank, topk_idx_per_rank, None, case["num_experts"])
        _assert_dispatch_matches_reference(
            self,
            dispatch,
            reference,
            topk_idx_per_rank,
            case["num_experts"],
            case["num_tokens"],
            do_cpu_sync=True,
        )
        self.assertTrue(all(d.recv_topk_weights is None for d in dispatch))
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        ref_combined_x, ref_combined_w = _reference_combine_by_metadata([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        for rank, item in enumerate(combine):
            self.assertTrue(exact_feature_equal(f"combine.cached.rank{rank}", item.combined_x, ref_combined_x[rank])[0])
            self.assertIsNone(item.combined_topk_weights)
            self.assertIsNone(ref_combined_w[rank])

    def test_full_golden_upper_bound_matches_reference_valid_rows(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 4, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=False,
        )
        reference = _reference_dispatch_by_argsort(x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"])
        _assert_dispatch_matches_reference(
            self,
            dispatch,
            reference,
            topk_idx_per_rank,
            case["num_experts"],
            case["num_tokens"],
            do_cpu_sync=False,
        )
        local_moe = case["num_experts"] // case["world_size"]
        expected_alloc = case["world_size"] * case["num_tokens"] * min(case["num_topk"], local_moe)
        self.assertTrue(all(d.handle.num_expanded_tokens == expected_alloc for d in dispatch))

    def test_full_golden_fp8_tuple_matches_reference_for_e4m3_and_e5m2(self):
        fp8_names = ["float8_e4m3fn", "float8_e5m2"]
        for dtype_name in fp8_names:
            dtype = getattr(torch, dtype_name, None)
            if dtype is None:
                continue
            for scales_dtype in ("float32", "fp8_e8m0"):
                with self.subTest(dtype=dtype_name, scales_dtype=scales_dtype):
                    case = {
                        "world_size": 2,
                        "num_tokens": 3,
                        "hidden": 4,
                        "num_topk": 2,
                        "num_experts": 4,
                        "dtype": dtype,
                        "dim_scales": 3,
                        "scales_dtype": scales_dtype,
                        "include_invalid": True,
                    }
                    x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
                    dispatch = gen_dispatch_golden_cpu(
                        x_per_rank,
                        topk_idx_per_rank,
                        topk_weights_per_rank,
                        num_experts=case["num_experts"],
                        num_max_tokens_per_rank=case["num_tokens"],
                        do_cpu_sync=True,
                    )
                    reference = _reference_dispatch_by_argsort(
                        x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"]
                    )
                    _assert_dispatch_matches_reference(
                        self,
                        dispatch,
                        reference,
                        topk_idx_per_rank,
                        case["num_experts"],
                        case["num_tokens"],
                        do_cpu_sync=True,
                    )
                    combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
                    ref_combined_x, _ = _reference_combine_by_metadata([d.recv_x for d in dispatch], [d.handle for d in dispatch])
                    for rank, item in enumerate(combine):
                        self.assertTrue(exact_feature_equal(f"fp8.combine.rank{rank}", item.combined_x, ref_combined_x[rank])[0])
                    expected_scales_dtype = torch.uint8 if scales_dtype == "fp8_e8m0" else torch.float32
                    self.assertTrue(all(feature_scales(d.recv_x).dtype == expected_scales_dtype for d in dispatch))

    def test_dispatch_input_validation_rejects_bad_fp8_scales_dtype(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("当前 torch 不支持 float8_e4m3fn")
        x_per_rank = [
            (
                torch.ones((2, 3), dtype=torch.bfloat16).to(fp8_dtype),
                torch.ones((2, 2), dtype=torch.int32),
            ),
            (
                torch.ones((2, 3), dtype=torch.bfloat16).to(fp8_dtype),
                torch.ones((2, 2), dtype=torch.int32),
            ),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        ]
        with self.assertRaisesRegex(ValueError, "scales dtype"):
            gen_dispatch_golden_cpu(
                x_per_rank,
                topk_idx_per_rank,
                None,
                num_experts=4,
                num_max_tokens_per_rank=2,
            )

    def test_dispatch_routes_fp8_tuple_e8m0_scales(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("当前 torch 不支持 float8_e4m3fn")
        case = {
            "world_size": 2,
            "num_tokens": 3,
            "hidden": 4,
            "num_topk": 2,
            "num_experts": 4,
            "dtype": fp8_dtype,
            "dim_scales": 3,
            "scales_dtype": "fp8_e8m0",
            "include_invalid": True,
        }
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        reference = _reference_dispatch_by_argsort(
            x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"]
        )
        _assert_dispatch_matches_reference(
            self,
            dispatch,
            reference,
            topk_idx_per_rank,
            case["num_experts"],
            case["num_tokens"],
            do_cpu_sync=True,
        )
        self.assertTrue(all(feature_scales(d.recv_x).dtype == torch.uint8 for d in dispatch))

    def test_golden_cache_roundtrip_fp8_tuple_e8m0_scales(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("当前 torch 不支持 float8_e4m3fn")
        case = {
            "world_size": 2,
            "num_tokens": 2,
            "hidden": 3,
            "num_topk": 2,
            "num_experts": 4,
            "dtype": fp8_dtype,
            "dim_scales": 5,
            "scales_dtype": "fp8_e8m0",
            "include_invalid": True,
        }
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
            "combine_goldens": combine,
        }
        params = {"case": "fp8_e8m0_cache", **case, "dtype": "float8_e4m3fn"}
        with TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            scripts = [Path(__file__).resolve()]
            save_case_cache(cache_root, "case0", case_data, params, scripts)
            loaded = load_case_cache(cache_root, "case0", params, scripts)
            self.assertIsNotNone(loaded)
            for rank in range(case["world_size"]):
                self.assertEqual(feature_scales(loaded["x_per_rank"][rank]).dtype, torch.uint8)
                self.assertTrue(exact_feature_equal(f"cache.fp8.e8m0.x{rank}", loaded["x_per_rank"][rank], x_per_rank[rank])[0])

    def test_golden_cache_roundtrip(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu(
            [d.recv_x for d in dispatch],
            [d.handle for d in dispatch],
            topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
        )
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
            "combine_goldens": combine,
        }
        params = {"case": "roundtrip", **case}
        with TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            scripts = [Path(__file__).resolve()]
            save_case_cache(cache_root, "case0", case_data, params, scripts)
            loaded = load_case_cache(cache_root, "case0", params, scripts)
            self.assertIsNotNone(loaded)
            for rank in range(case["world_size"]):
                self.assertTrue(exact_feature_equal(f"cache.x{rank}", loaded["x_per_rank"][rank], x_per_rank[rank])[0])
                self.assertTrue(torch.equal(loaded["topk_idx_per_rank"][rank], topk_idx_per_rank[rank]))
                self.assertTrue(torch.equal(loaded["topk_weights_per_rank"][rank], topk_weights_per_rank[rank]))
                self.assertTrue(exact_feature_equal(f"cache.dispatch_x{rank}", loaded["dispatch_goldens"][rank].recv_x, dispatch[rank].recv_x)[0])
                self.assertTrue(torch.equal(loaded["dispatch_goldens"][rank].handle.recv_src_metadata, dispatch[rank].handle.recv_src_metadata))
                self.assertTrue(exact_feature_equal(f"cache.combine_x{rank}", loaded["combine_goldens"][rank].combined_x, combine[rank].combined_x)[0])

    def test_golden_cache_roundtrip_fp8_tuple(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("当前 torch 不支持 float8_e4m3fn")
        case = {
            "world_size": 2,
            "num_tokens": 2,
            "hidden": 3,
            "num_topk": 2,
            "num_experts": 4,
            "dtype": fp8_dtype,
            "dim_scales": 5,
            "include_invalid": True,
        }
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
            "combine_goldens": combine,
        }
        params = {"case": "fp8_cache", **case, "dtype": "float8_e4m3fn"}
        with TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            scripts = [Path(__file__).resolve()]
            save_case_cache(cache_root, "case0", case_data, params, scripts)
            loaded = load_case_cache(cache_root, "case0", params, scripts)
            self.assertIsNotNone(loaded)
            for rank in range(case["world_size"]):
                self.assertTrue(exact_feature_equal(f"cache.fp8.x{rank}", loaded["x_per_rank"][rank], x_per_rank[rank])[0])
                self.assertTrue(exact_feature_equal(f"cache.fp8.dispatch{rank}", loaded["dispatch_goldens"][rank].recv_x, dispatch[rank].recv_x)[0])

    def test_golden_cache_invalidates_on_params_md5_and_flags(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 3, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
            "combine_goldens": combine,
        }
        params = {"case": "invalidate", **case}
        with TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            script_path = cache_root / "script_ref.py"
            script_path.write_text("version = 1\n", encoding="utf-8")
            scripts = [script_path]
            save_case_cache(cache_root, "case0", case_data, params, scripts)

            self.assertIsNotNone(load_case_cache(cache_root, "case0", params, scripts))
            self.assertIsNone(load_case_cache(cache_root, "case0", {**params, "hidden": 99}, scripts))
            self.assertIsNone(load_case_cache(cache_root, "case0", params, scripts, enabled=False))
            self.assertIsNone(load_case_cache(cache_root, "case0", params, scripts, force_generate=True))

            script_path.write_text("version = 2\n", encoding="utf-8")
            self.assertIsNone(load_case_cache(cache_root, "case0", params, scripts))

    def test_runner_generate_case_cpu_respects_save_golden_switch(self):
        old_values = (
            moe_train_test.load_case_cache_file,
            moe_train_test.save_case_cache_file,
            moe_train_test.is_save_golden,
            moe_train_test.world_size,
            moe_train_test.rank_per_dev,
            moe_train_test.server_index,
            moe_train_test.num_tokens,
            moe_train_test.hidden,
            moe_train_test.num_topk,
            moe_train_test.num_experts,
            moe_train_test.num_max_tokens_per_rank,
            moe_train_test.x_dtype,
            moe_train_test.scales_dtype,
            moe_train_test.dim_scales,
            moe_train_test.dispatch_with_handle,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.combine_with_topk_weights,
            moe_train_test.is_variable_bs,
            moe_train_test.is_x_random,
        )
        save_calls = []
        try:
            moe_train_test.load_case_cache_file = lambda *args, **kwargs: None
            moe_train_test.save_case_cache_file = lambda *args, **kwargs: save_calls.append((args, kwargs))
            moe_train_test.world_size = 2
            moe_train_test.rank_per_dev = 2
            moe_train_test.server_index = 0
            moe_train_test.num_tokens = 2
            moe_train_test.hidden = 3
            moe_train_test.num_topk = 1
            moe_train_test.num_experts = 2
            moe_train_test.num_max_tokens_per_rank = 2
            moe_train_test.x_dtype = "bfloat16"
            moe_train_test.scales_dtype = "float32"
            moe_train_test.dim_scales = 1
            moe_train_test.dispatch_with_handle = False
            moe_train_test.dispatch_with_topk_weights = False
            moe_train_test.combine_with_topk_weights = False
            moe_train_test.is_variable_bs = False
            moe_train_test.is_x_random = False

            moe_train_test.is_save_golden = False
            moe_train_test.generate_case_cpu()
            self.assertEqual(save_calls, [])

            moe_train_test.is_save_golden = True
            moe_train_test.generate_case_cpu()
            self.assertEqual(len(save_calls), 1)
        finally:
            (
                moe_train_test.load_case_cache_file,
                moe_train_test.save_case_cache_file,
                moe_train_test.is_save_golden,
                moe_train_test.world_size,
                moe_train_test.rank_per_dev,
                moe_train_test.server_index,
                moe_train_test.num_tokens,
                moe_train_test.hidden,
                moe_train_test.num_topk,
                moe_train_test.num_experts,
                moe_train_test.num_max_tokens_per_rank,
                moe_train_test.x_dtype,
                moe_train_test.scales_dtype,
                moe_train_test.dim_scales,
                moe_train_test.dispatch_with_handle,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.combine_with_topk_weights,
                moe_train_test.is_variable_bs,
                moe_train_test.is_x_random,
            ) = old_values

    def test_runner_default_does_not_save_golden(self):
        self.assertIs(moe_train_test.is_save_golden, False)

    def test_runner_dispatch_do_cpu_sync_value_normalizes_integer_config(self):
        old_value = moe_train_test.dispatch_do_cpu_sync
        try:
            for raw_value, expected in ((0, False), (1, True)):
                moe_train_test.dispatch_do_cpu_sync = raw_value
                actual = moe_train_test.dispatch_do_cpu_sync_value()
                self.assertIs(actual, expected)
                self.assertIs(moe_train_test.case_params()["dispatch_do_cpu_sync"], expected)
        finally:
            moe_train_test.dispatch_do_cpu_sync = old_value

    def test_dispatch_input_validation_rejects_unsupported_shapes(self):
        x_per_rank = [
            torch.ones((2, 3), dtype=torch.bfloat16),
            torch.ones((2, 4), dtype=torch.bfloat16),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        ]
        with self.assertRaisesRegex(ValueError, "hidden"):
            gen_dispatch_golden_cpu(
                x_per_rank,
                topk_idx_per_rank,
                None,
                num_experts=4,
                num_max_tokens_per_rank=2,
            )

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is not None:
            bad_tuple = [
                (torch.ones((2, 3), dtype=torch.bfloat16).to(fp8_dtype), torch.ones((1, 2), dtype=torch.float32)),
                (torch.ones((2, 3), dtype=torch.bfloat16).to(fp8_dtype), torch.ones((2, 2), dtype=torch.float32)),
            ]
            with self.assertRaisesRegex(ValueError, "scales"):
                gen_dispatch_golden_cpu(
                    bad_tuple,
                    topk_idx_per_rank,
                    None,
                    num_experts=4,
                    num_max_tokens_per_rank=2,
                )

    def test_dispatch_and_combine_support_fp16_x_dtype(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case, dtype=torch.float16)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        reference = _reference_dispatch_by_argsort(
            x_per_rank, topk_idx_per_rank, topk_weights_per_rank, case["num_experts"]
        )
        _assert_dispatch_matches_reference(
            self,
            dispatch,
            reference,
            topk_idx_per_rank,
            case["num_experts"],
            case["num_tokens"],
            do_cpu_sync=True,
        )

        combine = gen_combine_golden_cpu(
            [d.recv_x for d in dispatch],
            [d.handle for d in dispatch],
            topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
        )
        ref_combined_x, ref_combined_w = _reference_combine_by_metadata(
            [d.recv_x for d in dispatch],
            [d.handle for d in dispatch],
            topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
        )
        for rank, item in enumerate(combine):
            self.assertEqual(feature_main(dispatch[rank].recv_x).dtype, torch.float16)
            self.assertIsNone(feature_scales(dispatch[rank].recv_x))
            self.assertEqual(feature_main(item.combined_x).dtype, torch.float16)
            self.assertIsNone(feature_scales(item.combined_x))
            self.assertTrue(exact_feature_equal(f"combine.fp16.rank{rank}", item.combined_x, ref_combined_x[rank])[0])
            self.assertTrue(exact_tensor_equal(f"combine.fp16.topk.rank{rank}", item.combined_topk_weights, ref_combined_w[rank])[0])

    def test_dispatch_matches_noq_reference_subset(self):
        x_per_rank = [
            torch.tensor([[1, 10], [2, 20]], dtype=torch.bfloat16),
            torch.tensor([[3, 30], [4, 40]], dtype=torch.bfloat16),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, 2], [2, 3]], dtype=torch.int32),
            torch.tensor([[1, 3], [0, 1]], dtype=torch.int32),
        ]
        topk_weights_per_rank = [
            torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5], [0.8, 0.2]], dtype=torch.float32),
        ]

        golden = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=4,
            num_max_tokens_per_rank=2,
            do_cpu_sync=True,
        )
        reference = _reference_dispatch_by_argsort(
            x_per_rank, topk_idx_per_rank, topk_weights_per_rank, num_experts=4
        )

        for rank, dispatch in enumerate(golden):
            ref_x, ref_w, ref_meta, ref_counts = reference[rank]
            self.assertTrue(exact_tensor_equal(f"rank{rank}.recv_x", feature_main(dispatch.recv_x), ref_x)[0])
            self.assertTrue(exact_tensor_equal(f"rank{rank}.recv_topk_weights", dispatch.recv_topk_weights, ref_w)[0])
            self.assertTrue(exact_tensor_equal(f"rank{rank}.recv_src_metadata", dispatch.handle.recv_src_metadata[:, :3], ref_meta)[0])
            self.assertTrue(exact_tensor_equal(f"rank{rank}.num_recv_tokens_per_expert", dispatch.handle.num_recv_tokens_per_expert, ref_counts)[0])

    def test_rank_level_dedup_payload_count_differs_from_expand_count(self):
        x_per_rank = [
            torch.tensor([[1, 10], [2, 20]], dtype=torch.bfloat16),
            torch.tensor([[3, 30], [4, 40]], dtype=torch.bfloat16),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
        ]
        topk_weights_per_rank = [
            torch.ones((2, 2), dtype=torch.float32),
            torch.ones((2, 2), dtype=torch.float32),
        ]

        golden = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=4,
            num_max_tokens_per_rank=2,
            do_cpu_sync=True,
        )

        self.assertEqual(golden[0].handle.dst_buffer_slot_idx[0].tolist(), [0, 0])
        self.assertEqual(golden[0].handle.num_recv_payloads_per_rank.tolist(), [1, 2])
        self.assertEqual(golden[0].handle.num_valid_expanded_tokens, 4)
        self.assertEqual(golden[0].handle.num_recv_payloads, 3)

    def test_dispatch_routes_fp8_tuple_scales(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("当前 torch 不支持 float8_e4m3fn")

        x_per_rank = [
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16).to(fp8_dtype),
                torch.tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=torch.float32),
            ),
            (
                torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16).to(fp8_dtype),
                torch.tensor([[16.0, 17.0, 18.0], [19.0, 20.0, 21.0]], dtype=torch.float32),
            ),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, 2], [2, 3]], dtype=torch.int32),
            torch.tensor([[1, 3], [0, 1]], dtype=torch.int32),
        ]
        topk_weights_per_rank = [torch.ones((2, 2), dtype=torch.float32) for _ in range(2)]

        golden = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=4,
            num_max_tokens_per_rank=2,
            do_cpu_sync=True,
        )

        self.assertIsNotNone(feature_scales(golden[0].recv_x))
        self.assertEqual(feature_scales(golden[0].recv_x).shape, (4, 3))
        expected_rank0_scales = torch.tensor(
            [[10.0, 11.0, 12.0], [19.0, 20.0, 21.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(feature_scales(golden[0].recv_x), expected_rank0_scales))

    def test_combine_is_pure_sum_and_topk_weight_scatter(self):
        x_per_rank = [
            torch.tensor([[1, 10], [2, 20]], dtype=torch.bfloat16),
            torch.tensor([[3, 30], [4, 40]], dtype=torch.bfloat16),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, 2], [2, 3]], dtype=torch.int32),
            torch.tensor([[1, 3], [0, 1]], dtype=torch.int32),
        ]
        topk_weights_per_rank = [
            torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5], [0.8, 0.2]], dtype=torch.float32),
        ]
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=4,
            num_max_tokens_per_rank=2,
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu(
            [d.recv_x for d in dispatch],
            [d.handle for d in dispatch],
            topk_weights_per_rank=[d.recv_topk_weights for d in dispatch],
        )

        expected_rank0 = torch.tensor([[2, 20], [4, 40]], dtype=torch.bfloat16)
        expected_rank1 = torch.tensor([[6, 60], [8, 80]], dtype=torch.bfloat16)
        self.assertTrue(exact_feature_equal("combined_rank0", combine[0].combined_x, expected_rank0)[0])
        self.assertTrue(exact_feature_equal("combined_rank1", combine[1].combined_x, expected_rank1)[0])
        self.assertTrue(torch.equal(combine[0].combined_topk_weights, topk_weights_per_rank[0]))
        self.assertTrue(torch.equal(combine[1].combined_topk_weights, topk_weights_per_rank[1]))

    def test_upper_bound_dispatch_keeps_invalid_tail(self):
        x_per_rank = [
            torch.tensor([[1, 10]], dtype=torch.bfloat16),
            torch.tensor([[2, 20]], dtype=torch.bfloat16),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0, -1]], dtype=torch.int32),
            torch.tensor([[-1, 1]], dtype=torch.int32),
        ]
        topk_weights_per_rank = [torch.ones((1, 2), dtype=torch.float32) for _ in range(2)]
        golden = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=4,
            num_max_tokens_per_rank=1,
            do_cpu_sync=False,
        )
        self.assertEqual(golden[0].handle.num_expanded_tokens, 4)
        self.assertEqual(golden[0].handle.num_valid_expanded_tokens, 2)
        self.assertEqual(golden[0].handle.recv_src_metadata[2:].tolist(), [[-1, -1, -1, -1], [-1, -1, -1, -1]])

    def test_expected_dispatch_rows_from_metadata(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 2, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        metadata = torch.tensor([[0, 1, 0], [1, 2, 1], [-1, -1, -1]], dtype=torch.int32)
        expected_x, expected_w = expected_dispatch_rows_from_metadata(x_per_rank, topk_weights_per_rank, metadata)
        self.assertTrue(exact_tensor_equal("metadata.lookup.x0", feature_main(expected_x)[0], feature_main(x_per_rank[0])[1])[0])
        self.assertTrue(exact_tensor_equal("metadata.lookup.x1", feature_main(expected_x)[1], feature_main(x_per_rank[1])[2])[0])
        self.assertEqual(float(expected_w[0]), float(topk_weights_per_rank[0][1, 0]))
        self.assertEqual(float(expected_w[1]), float(topk_weights_per_rank[1][2, 1]))

    def test_runner_compare_dispatch_and_combine_cpu_outputs(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
            "combine_goldens": combine,
        }
        dispatch_outputs = [
            {
                "recv_x": d.recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": d.recv_topk_weights,
                "handle": d.handle,
            }
            for d in dispatch
        ]
        combine_outputs = [
            {"combined_x": c.combined_x, "combined_topk_weights": c.combined_topk_weights}
            for c in combine
        ]
        old_value = moe_train_test.compare_print_detail_max_elements
        stdout = StringIO()
        try:
            moe_train_test.compare_print_detail_max_elements = 16
            with redirect_stdout(stdout):
                dispatch_result = moe_train_test.compare_dispatch_outputs(dispatch_outputs, case_data)
                combine_result = moe_train_test.compare_combine_outputs(combine_outputs, combine, "combine")
                continue_dispatch_result = moe_train_test.compare_dispatch_outputs(
                    dispatch_outputs,
                    case_data,
                    "continue.dispatch",
                )
                continue_combine_result = moe_train_test.compare_combine_outputs(
                    combine_outputs,
                    combine,
                    "continue.combine",
                )
        finally:
            moe_train_test.compare_print_detail_max_elements = old_value
        self.assertTrue(all(dispatch_result.values()))
        self.assertTrue(all(combine_result.values()))
        self.assertTrue(all(continue_dispatch_result.values()))
        self.assertTrue(all(continue_combine_result.values()))
        output = stdout.getvalue()
        self.assertIn("################## dispatch compare detail ##################", output)
        self.assertIn("################## combine compare detail ##################", output)
        self.assertIn("################## continue.dispatch compare table ##################", output)
        self.assertIn("################## continue.combine compare table ##################", output)
        self.assertIn("################## dispatch compare table ##################", output)
        self.assertIn("################## combine compare table ##################", output)
        self.assertIn("[INFO] dispatch.recv_x.rank0 PASSED! toolss detail follows.", output)
        self.assertIn("[INFO] dispatch.recv_x.rank0 toolss detail:", output)
        self.assertIn("[INFO] combine.combined_x.rank0 PASSED! toolss detail follows.", output)
        self.assertIn("[INFO] combine.combined_x.rank0 toolss detail:", output)
        self.assertIn("DiffThd", output)
        rank0_idx = output.index("---------- card rank0 ----------")
        rank1_idx = output.index("---------- card rank1 ----------")
        self.assertLess(rank0_idx, rank1_idx)
        dispatch_order = [
            "[INFO] dispatch.recv_x.rank0",
            "[INFO] dispatch.recv_topk_idx.rank0",
            "[INFO] dispatch.recv_topk_weights.rank0",
            "[INFO] dispatch.handle.recv_src_metadata.rank0",
            "[INFO] dispatch.handle.num_recv_payloads_per_rank.rank0",
            "[INFO] dispatch.handle.num_recv_tokens_per_expert.rank0",
            "[INFO] dispatch.handle.topk_idx.rank0",
            "[INFO] dispatch.handle.num_experts.rank0",
            "[INFO] dispatch.handle.expert_alignment.rank0",
            "[INFO] dispatch.handle.num_max_tokens_per_rank.rank0",
            "[INFO] dispatch.handle.num_expanded_tokens.rank0",
        ]
        last_idx = rank0_idx
        for marker in dispatch_order:
            current_idx = output.index(marker)
            self.assertLess(last_idx, current_idx)
            last_idx = current_idx
        self.assertNotIn("[INFO] dispatch.handle.dst_buffer_slot_idx.rank0", output)
        self.assertRegex(output, r"\|\s+\| recv_x\s+\| \x1b\[32mPASS\x1b\[0m\s+\|")
        self.assertRegex(output, r"\|\s+\| recv_x\s+\|[^\n]*\n\|\s+\| recv_topk_idx\s+\| \x1b\[90m-\x1b\[0m\s+\|")
        self.assertRegex(output, r"\|\s*(?:rank\d+)?\s*\| combined_topk_weights\s+\| \x1b\[90m-\x1b\[0m\s+\|")
        self.assertRegex(output, r"\| rank0\s+\| handle.num_recv_tokens_per_expert\s+\|")
        self.assertRegex(output, r"\|\s+\| handle.num_expanded_tokens\s+\|[^\n]*\n\+[-+]+\+\n\|\s+\| recv_x\s+\|")
        self.assertRegex(output, r"\| rank1\s+\| handle.num_recv_tokens_per_expert\s+\|")

    def test_runner_compare_dispatch_slot_idx_only_checked_in_cached_mode(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        bad_slot_idx = dispatch[0].handle.dst_buffer_slot_idx.clone()
        bad_slot_idx[0, 0] += 1
        bad_handle = GoldenHandle(
            dst_buffer_slot_idx=bad_slot_idx,
            recv_src_metadata=dispatch[0].handle.recv_src_metadata,
            num_recv_payloads_per_rank=dispatch[0].handle.num_recv_payloads_per_rank,
            num_recv_tokens_per_expert=dispatch[0].handle.num_recv_tokens_per_expert,
            topk_idx=dispatch[0].handle.topk_idx,
            num_experts=dispatch[0].handle.num_experts,
            expert_alignment=dispatch[0].handle.expert_alignment,
            num_max_tokens_per_rank=dispatch[0].handle.num_max_tokens_per_rank,
            num_expanded_tokens=dispatch[0].handle.num_expanded_tokens,
        )
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
        }
        dispatch_outputs = [
            {
                "recv_x": dispatch[0].recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": dispatch[0].recv_topk_weights,
                "handle": bad_handle,
            },
            {
                "recv_x": dispatch[1].recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": dispatch[1].recv_topk_weights,
                "handle": dispatch[1].handle,
            },
        ]
        old_values = (moe_train_test.dispatch_with_handle, moe_train_test.dispatch_with_topk_weights)
        try:
            moe_train_test.dispatch_with_handle = False
            moe_train_test.dispatch_with_topk_weights = True
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = moe_train_test.compare_dispatch_outputs(dispatch_outputs, case_data)
            self.assertNotIn("dispatch_handle_dst_buffer_slot_idx_rank0", result)
            self.assertNotIn("[INFO] dispatch.handle.dst_buffer_slot_idx.rank0", stdout.getvalue())

            cached_outputs = [
                {**dispatch_outputs[0], "recv_topk_weights": None},
                {**dispatch_outputs[1], "recv_topk_weights": None},
            ]
            moe_train_test.dispatch_with_handle = True
            moe_train_test.dispatch_with_topk_weights = False
            with redirect_stdout(StringIO()):
                result = moe_train_test.compare_dispatch_outputs(cached_outputs, case_data)
        finally:
            moe_train_test.dispatch_with_handle, moe_train_test.dispatch_with_topk_weights = old_values
        self.assertFalse(result["dispatch_handle_dst_buffer_slot_idx_rank0"])
        self.assertTrue(result["dispatch_handle_dst_buffer_slot_idx_rank1"])

    def test_runner_compare_recv_src_metadata_slot_idx_only_checked_in_cached_mode(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        bad_metadata = dispatch[0].handle.recv_src_metadata.clone()
        bad_metadata[0, 3] += 1
        bad_handle = GoldenHandle(
            dst_buffer_slot_idx=dispatch[0].handle.dst_buffer_slot_idx,
            recv_src_metadata=bad_metadata,
            num_recv_payloads_per_rank=dispatch[0].handle.num_recv_payloads_per_rank,
            num_recv_tokens_per_expert=dispatch[0].handle.num_recv_tokens_per_expert,
            topk_idx=dispatch[0].handle.topk_idx,
            num_experts=dispatch[0].handle.num_experts,
            expert_alignment=dispatch[0].handle.expert_alignment,
            num_max_tokens_per_rank=dispatch[0].handle.num_max_tokens_per_rank,
            num_expanded_tokens=dispatch[0].handle.num_expanded_tokens,
        )
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
        }
        dispatch_outputs = [
            {
                "recv_x": dispatch[0].recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": dispatch[0].recv_topk_weights,
                "handle": bad_handle,
            },
            {
                "recv_x": dispatch[1].recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": dispatch[1].recv_topk_weights,
                "handle": dispatch[1].handle,
            },
        ]
        old_values = (moe_train_test.dispatch_with_handle, moe_train_test.dispatch_with_topk_weights)
        try:
            moe_train_test.dispatch_with_handle = False
            moe_train_test.dispatch_with_topk_weights = True
            with redirect_stdout(StringIO()):
                result = moe_train_test.compare_dispatch_outputs(dispatch_outputs, case_data)
            self.assertTrue(result["dispatch_handle_recv_src_metadata_rank0"])

            moe_train_test.dispatch_with_handle = True
            with redirect_stdout(StringIO()):
                result = moe_train_test.compare_dispatch_outputs(dispatch_outputs, case_data)
            self.assertFalse(result["dispatch_handle_recv_src_metadata_rank0"])
            self.assertTrue(result["dispatch_handle_recv_src_metadata_rank1"])
        finally:
            moe_train_test.dispatch_with_handle, moe_train_test.dispatch_with_topk_weights = old_values

    def test_runner_compare_cached_dispatch_skips_topk_weights(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
        }
        dispatch_outputs = [
            {
                "recv_x": item.recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": item.recv_topk_weights,
                "handle": item.handle,
            }
            for item in dispatch
        ]
        old_values = (moe_train_test.dispatch_with_handle, moe_train_test.dispatch_with_topk_weights)
        try:
            moe_train_test.dispatch_with_handle = True
            moe_train_test.dispatch_with_topk_weights = True
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = moe_train_test.compare_dispatch_outputs(dispatch_outputs, case_data)
        finally:
            moe_train_test.dispatch_with_handle, moe_train_test.dispatch_with_topk_weights = old_values
        self.assertTrue(result["dispatch_recv_topk_weights_rank0"])
        self.assertTrue(result["dispatch_recv_topk_weights_rank1"])
        self.assertIn("[INFO] dispatch.recv_topk_weights.rank0 toolss detail skipped:", stdout.getvalue())

    def test_runner_cached_dispatch_init_keeps_topk_weights_when_formal_omits(self):
        calls = []

        class FakeHandle:
            def __init__(self):
                self.dst_buffer_slot_idx = torch.zeros((2,), dtype=torch.int32)
                self.recv_src_metadata = torch.zeros((2, 4), dtype=torch.int32)
                self.num_recv_payloads_per_rank = torch.ones((2,), dtype=torch.int32)
                self.num_recv_tokens_per_expert = torch.ones((2,), dtype=torch.int32)
                self.topk_idx = torch.zeros((2, 1), dtype=torch.int32)
                self.num_experts = 2
                self.expert_alignment = 1
                self.num_max_tokens_per_rank = 2
                self.num_expanded_tokens = 2

        class FakeBuffer:
            def __init__(self, *args, **kwargs):
                pass

            def dispatch(
                self,
                x,
                topk_idx=None,
                topk_weights=None,
                handle=None,
                num_experts=None,
                expert_alignment=None,
                do_cpu_sync=None,
            ):
                effective_handle = FakeHandle() if handle is None else handle
                calls.append({
                    "x": x,
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    "handle": handle,
                    "effective_handle": effective_handle,
                    "do_cpu_sync": do_cpu_sync,
                })
                return x, None, topk_weights, effective_handle

        class FakeNpu:
            def synchronize(self):
                pass

        class FakeDist:
            def destroy_process_group(self):
                pass

        class FakeQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        fake_torch_npu = types.ModuleType("torch_npu")
        fake_torch_npu.npu = FakeNpu()
        queue = FakeQueue()
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        topk_idx = torch.zeros((2, 1), dtype=torch.int32)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)

        had_torch_npu = "torch_npu" in sys.modules
        old_torch_npu = sys.modules.get("torch_npu")
        had_tensor_npu = hasattr(torch.Tensor, "npu")
        old_tensor_npu = getattr(torch.Tensor, "npu", None)
        old_values = (
            moe_train_test.set_device,
            moe_train_test.init_hccl_comm,
            moe_train_test.ElasticBuffer,
            moe_train_test.dist,
            moe_train_test.is_full_core,
            moe_train_test.dispatch_with_topk_weights,
        )
        try:
            sys.modules["torch_npu"] = fake_torch_npu
            torch.Tensor.npu = lambda tensor: tensor
            moe_train_test.set_device = lambda rank: None
            moe_train_test.init_hccl_comm = lambda rank: object()
            moe_train_test.ElasticBuffer = FakeBuffer
            moe_train_test.dist = FakeDist()
            moe_train_test.is_full_core = True
            moe_train_test.dispatch_with_topk_weights = True

            with redirect_stdout(StringIO()):
                moe_train_test.run_dispatch_npu(
                    queue,
                    0,
                    x,
                    topk_idx,
                    None,
                    cached_handle=None,
                    init_topk_weights=topk_weights,
                    init_cached_handle=True,
                )
        finally:
            (
                moe_train_test.set_device,
                moe_train_test.init_hccl_comm,
                moe_train_test.ElasticBuffer,
                moe_train_test.dist,
                moe_train_test.is_full_core,
                moe_train_test.dispatch_with_topk_weights,
            ) = old_values
            if had_tensor_npu:
                torch.Tensor.npu = old_tensor_npu
            else:
                delattr(torch.Tensor, "npu")
            if had_torch_npu:
                sys.modules["torch_npu"] = old_torch_npu
            else:
                sys.modules.pop("torch_npu", None)

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["x"], x)
        self.assertIs(calls[0]["topk_idx"], topk_idx)
        self.assertIs(calls[0]["topk_weights"], topk_weights)
        self.assertIsNone(calls[0]["handle"])
        self.assertFalse(calls[0]["do_cpu_sync"])
        self.assertIs(calls[1]["x"], x)
        self.assertIsNone(calls[1]["topk_idx"])
        self.assertIsNone(calls[1]["topk_weights"])
        self.assertIs(calls[1]["handle"], calls[0]["effective_handle"])
        self.assertFalse(calls[1]["do_cpu_sync"])
        self.assertIsNotNone(queue.items[0][1]["cached_input_handle"])
        self.assertIsNone(queue.items[0][1]["recv_topk_weights"])

    def test_runner_cached_continue_init_keeps_topk_weights_when_formal_omits(self):
        dispatch_calls = []
        combine_calls = []

        class FakeHandle:
            def __init__(self):
                self.dst_buffer_slot_idx = torch.zeros((2,), dtype=torch.int32)
                self.recv_src_metadata = torch.tensor(
                    [[0, 0, 0, 0], [0, 1, 0, 1]],
                    dtype=torch.int32,
                )
                self.num_recv_payloads_per_rank = torch.ones((2,), dtype=torch.int32)
                self.num_recv_tokens_per_expert = torch.tensor([2, 0], dtype=torch.int32)
                self.topk_idx = torch.zeros((2, 1), dtype=torch.int32)
                self.num_experts = 2
                self.expert_alignment = 1
                self.num_max_tokens_per_rank = 2
                self.num_expanded_tokens = 2

        class FakeBuffer:
            def __init__(self, *args, **kwargs):
                pass

            def dispatch(
                self,
                x,
                topk_idx=None,
                topk_weights=None,
                handle=None,
                num_experts=None,
                expert_alignment=None,
                do_cpu_sync=None,
            ):
                effective_handle = FakeHandle() if handle is None else handle
                dispatch_calls.append({
                    "x": x,
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    "handle": handle,
                    "effective_handle": effective_handle,
                    "do_cpu_sync": do_cpu_sync,
                })
                return x, None, topk_weights, effective_handle

            def combine(self, combine_x, handle=None, topk_weights=None):
                combine_calls.append({
                    "combine_x": combine_x,
                    "handle": handle,
                    "topk_weights": topk_weights,
                })
                return combine_x, topk_weights

        class FakeNpu:
            def synchronize(self):
                pass

        class FakeDist:
            def destroy_process_group(self):
                pass

        class FakeQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        fake_torch_npu = types.ModuleType("torch_npu")
        fake_torch_npu.npu = FakeNpu()
        queue = FakeQueue()
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        topk_idx = torch.zeros((2, 1), dtype=torch.int32)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)

        had_torch_npu = "torch_npu" in sys.modules
        old_torch_npu = sys.modules.get("torch_npu")
        had_tensor_npu = hasattr(torch.Tensor, "npu")
        old_tensor_npu = getattr(torch.Tensor, "npu", None)
        old_values = (
            moe_train_test.set_device,
            moe_train_test.init_hccl_comm,
            moe_train_test.ElasticBuffer,
            moe_train_test.dist,
            moe_train_test.is_full_core,
            moe_train_test.dispatch_do_cpu_sync,
            moe_train_test.combine_with_topk_weights,
        )
        try:
            sys.modules["torch_npu"] = fake_torch_npu
            torch.Tensor.npu = lambda tensor: tensor
            moe_train_test.set_device = lambda rank: None
            moe_train_test.init_hccl_comm = lambda rank: object()
            moe_train_test.ElasticBuffer = FakeBuffer
            moe_train_test.dist = FakeDist()
            moe_train_test.is_full_core = True
            moe_train_test.dispatch_do_cpu_sync = True
            moe_train_test.combine_with_topk_weights = False

            with redirect_stdout(StringIO()):
                moe_train_test.run_continue_npu(
                    queue,
                    0,
                    x,
                    topk_idx,
                    None,
                    init_topk_weights=topk_weights,
                    init_cached_handle=True,
                )
        finally:
            (
                moe_train_test.set_device,
                moe_train_test.init_hccl_comm,
                moe_train_test.ElasticBuffer,
                moe_train_test.dist,
                moe_train_test.is_full_core,
                moe_train_test.dispatch_do_cpu_sync,
                moe_train_test.combine_with_topk_weights,
            ) = old_values
            if had_tensor_npu:
                torch.Tensor.npu = old_tensor_npu
            else:
                delattr(torch.Tensor, "npu")
            if had_torch_npu:
                sys.modules["torch_npu"] = old_torch_npu
            else:
                sys.modules.pop("torch_npu", None)

        self.assertEqual(len(dispatch_calls), 2)
        self.assertIs(dispatch_calls[0]["topk_idx"], topk_idx)
        self.assertIs(dispatch_calls[0]["topk_weights"], topk_weights)
        self.assertIsNone(dispatch_calls[0]["handle"])
        self.assertFalse(dispatch_calls[0]["do_cpu_sync"])
        self.assertIsNone(dispatch_calls[1]["topk_idx"])
        self.assertIsNone(dispatch_calls[1]["topk_weights"])
        self.assertIs(dispatch_calls[1]["handle"], dispatch_calls[0]["effective_handle"])
        self.assertFalse(dispatch_calls[1]["do_cpu_sync"])
        self.assertEqual(len(combine_calls), 1)
        self.assertIs(combine_calls[0]["handle"], dispatch_calls[1]["effective_handle"])
        self.assertIsNotNone(queue.items[0][1]["cached_input_handle"])
        self.assertIsNone(queue.items[0][1]["recv_topk_weights"])

    def test_runner_cached_dispatch_primes_handle_from_npu_dispatch(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 1, "num_experts": 2, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
        }

        def clone_handle(handle, offset):
            return GoldenHandle(
                dst_buffer_slot_idx=handle.dst_buffer_slot_idx + offset,
                recv_src_metadata=handle.recv_src_metadata,
                num_recv_payloads_per_rank=handle.num_recv_payloads_per_rank,
                num_recv_tokens_per_expert=handle.num_recv_tokens_per_expert,
                topk_idx=handle.topk_idx,
                num_experts=handle.num_experts,
                expert_alignment=handle.expert_alignment,
                num_max_tokens_per_rank=handle.num_max_tokens_per_rank,
                num_expanded_tokens=handle.num_expanded_tokens,
            )

        primed_handles = [clone_handle(item.handle, rank + 10) for rank, item in enumerate(dispatch)]
        cached_outputs = [
            {
                "recv_x": item.recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": None,
                "handle": primed_handles[rank],
                "cached_input_handle": primed_handles[rank],
            }
            for rank, item in enumerate(dispatch)
        ]
        calls = []
        compare_seen = {}

        def fake_gen_npu(target_func, **kwargs):
            calls.append((target_func, kwargs))
            return cached_outputs

        def fake_compare(rank_outputs, compare_case_data, tag):
            compare_seen["rank_outputs"] = rank_outputs
            compare_seen["case_data"] = compare_case_data
            compare_seen["tag"] = tag
            return {"dispatch_recv_x_rank0": True}

        old_values = (
            moe_train_test.dispatch_with_handle,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.is_performance,
            moe_train_test.gen_npu,
            moe_train_test.compare_dispatch_outputs,
            moe_train_test.clean_previous_profiling_result,
        )
        try:
            moe_train_test.dispatch_with_handle = True
            moe_train_test.dispatch_with_topk_weights = True
            moe_train_test.is_performance = False
            moe_train_test.gen_npu = fake_gen_npu
            moe_train_test.compare_dispatch_outputs = fake_compare
            moe_train_test.clean_previous_profiling_result = lambda paths: None

            with redirect_stdout(StringIO()):
                result = moe_train_test._run_dispatch_test(case_data)
        finally:
            (
                moe_train_test.dispatch_with_handle,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.is_performance,
                moe_train_test.gen_npu,
                moe_train_test.compare_dispatch_outputs,
                moe_train_test.clean_previous_profiling_result,
            ) = old_values

        self.assertEqual(result, {"dispatch_recv_x_rank0": True})
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], moe_train_test.run_dispatch_npu)
        self.assertEqual(calls[0][1]["cached_handle"], [None, None])
        self.assertEqual(calls[0][1]["topk_weights"], [None, None])
        self.assertIs(calls[0][1]["init_topk_weights"][0], topk_weights_per_rank[0])
        self.assertIs(calls[0][1]["init_topk_weights"][1], topk_weights_per_rank[1])
        self.assertEqual(calls[0][1]["init_cached_handle"], [True, True])
        self.assertIs(compare_seen["rank_outputs"], cached_outputs)
        self.assertIs(compare_seen["case_data"]["dispatch_goldens"][0].handle, primed_handles[0])
        self.assertIs(compare_seen["case_data"]["dispatch_goldens"][1].handle, primed_handles[1])
        self.assertIs(case_data["dispatch_goldens"][0].handle, dispatch[0].handle)

    def test_runner_compare_dispatch_without_topk_weights_switch(self):
        case = {"world_size": 2, "num_tokens": 3, "hidden": 4, "num_topk": 2, "num_experts": 4, "include_invalid": True}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            None,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
            "combine_goldens": combine,
        }
        dispatch_outputs = [
            {
                "recv_x": d.recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": d.recv_topk_weights,
                "handle": d.handle,
            }
            for d in dispatch
        ]
        old_values = (
            moe_train_test.dispatch_with_handle,
            moe_train_test.dispatch_with_topk_weights,
        )
        try:
            moe_train_test.dispatch_with_handle = False
            moe_train_test.dispatch_with_topk_weights = False
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = moe_train_test.compare_dispatch_outputs(dispatch_outputs, case_data)
            self.assertTrue(all(result.values()))
            self.assertTrue(all(d.recv_topk_weights is None for d in dispatch))
            self.assertEqual(moe_train_test.dispatch_topk_weights_for_npu(case_data), [None, None])
            output = stdout.getvalue()
            self.assertIn("[INFO] dispatch.recv_topk_weights.rank0 PASSED! toolss detail follows.", output)
            self.assertIn("[INFO] dispatch.recv_topk_weights.rank0 toolss detail skipped:", output)
        finally:
            (
                moe_train_test.dispatch_with_handle,
                moe_train_test.dispatch_with_topk_weights,
            ) = old_values

    def test_runner_continue_uses_combine_x_for_combine_golden(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 1, "num_experts": 2, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=False,
        )
        combine_x_per_rank = [
            feature_main(item.recv_x).to(torch.float16) + (rank + 1) * 10
            for rank, item in enumerate(dispatch)
        ]
        combine = gen_combine_golden_cpu(combine_x_per_rank, [item.handle for item in dispatch])
        rank_outputs = [
            {
                "recv_x": dispatch[rank].recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": dispatch[rank].recv_topk_weights,
                "handle": dispatch[rank].handle,
                "combine_x": combine_x_per_rank[rank],
                "combined_x": combine[rank].combined_x,
                "combined_topk_weights": combine[rank].combined_topk_weights,
            }
            for rank in range(case["world_size"])
        ]
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
        }

        old_values = (
            moe_train_test.gen_npu,
            moe_train_test.clean_previous_profiling_result,
            moe_train_test.is_performance,
            moe_train_test.world_size,
            moe_train_test.rank_per_dev,
            moe_train_test.server_index,
            moe_train_test.dispatch_with_topk_weights,
        )
        try:
            moe_train_test.gen_npu = lambda target_func, **kwargs: rank_outputs
            moe_train_test.clean_previous_profiling_result = lambda paths: None
            moe_train_test.is_performance = False
            moe_train_test.world_size = case["world_size"]
            moe_train_test.rank_per_dev = case["world_size"]
            moe_train_test.server_index = 0
            moe_train_test.dispatch_with_topk_weights = True
            with redirect_stdout(StringIO()):
                result = moe_train_test._run_continue_test(case_data)
        finally:
            (
                moe_train_test.gen_npu,
                moe_train_test.clean_previous_profiling_result,
                moe_train_test.is_performance,
                moe_train_test.world_size,
                moe_train_test.rank_per_dev,
                moe_train_test.server_index,
                moe_train_test.dispatch_with_topk_weights,
            ) = old_values
        self.assertTrue(all(result.values()))

    def test_runner_cached_continue_primes_handle_from_npu_dispatch(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 1, "num_experts": 2, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        case_data = {
            "x_per_rank": x_per_rank,
            "topk_idx_per_rank": topk_idx_per_rank,
            "topk_weights_per_rank": topk_weights_per_rank,
            "dispatch_goldens": dispatch,
        }

        def clone_handle(handle, offset):
            return GoldenHandle(
                dst_buffer_slot_idx=handle.dst_buffer_slot_idx + offset,
                recv_src_metadata=handle.recv_src_metadata,
                num_recv_payloads_per_rank=handle.num_recv_payloads_per_rank,
                num_recv_tokens_per_expert=handle.num_recv_tokens_per_expert,
                topk_idx=handle.topk_idx,
                num_experts=handle.num_experts,
                expert_alignment=handle.expert_alignment,
                num_max_tokens_per_rank=handle.num_max_tokens_per_rank,
                num_expanded_tokens=handle.num_expanded_tokens,
            )

        primed_handles = [clone_handle(item.handle, rank + 10) for rank, item in enumerate(dispatch)]
        combine_x_per_rank = [feature_main(item.recv_x) for item in dispatch]
        rank_outputs = [
            {
                "recv_x": dispatch[rank].recv_x,
                "recv_topk_idx": None,
                "recv_topk_weights": None,
                "handle": primed_handles[rank],
                "cached_input_handle": primed_handles[rank],
                "combine_x": combine_x_per_rank[rank],
                "combine_topk_weights": None,
                "combined_x": combine_x_per_rank[rank],
                "combined_topk_weights": None,
            }
            for rank in range(case["world_size"])
        ]
        calls = []
        compare_seen = {}

        def fake_gen_npu(target_func, **kwargs):
            calls.append((target_func, kwargs))
            return rank_outputs

        def fake_compare_dispatch(rank_outputs_arg, compare_case_data, tag):
            compare_seen["rank_outputs"] = rank_outputs_arg
            compare_seen["case_data"] = compare_case_data
            compare_seen["tag"] = tag
            return {"continue_dispatch_recv_x_rank0": True}

        def fake_compare_combine(rank_outputs_arg, expected_goldens, tag):
            compare_seen["combine_tag"] = tag
            return {"continue_combine_combined_x_rank0": True}

        old_values = (
            moe_train_test.dispatch_with_handle,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.combine_with_topk_weights,
            moe_train_test.is_performance,
            moe_train_test.gen_npu,
            moe_train_test.compare_dispatch_outputs,
            moe_train_test.compare_combine_outputs,
            moe_train_test.clean_previous_profiling_result,
            moe_train_test.world_size,
            moe_train_test.rank_per_dev,
            moe_train_test.server_index,
        )
        try:
            moe_train_test.dispatch_with_handle = True
            moe_train_test.dispatch_with_topk_weights = True
            moe_train_test.combine_with_topk_weights = False
            moe_train_test.is_performance = False
            moe_train_test.gen_npu = fake_gen_npu
            moe_train_test.compare_dispatch_outputs = fake_compare_dispatch
            moe_train_test.compare_combine_outputs = fake_compare_combine
            moe_train_test.clean_previous_profiling_result = lambda paths: None
            moe_train_test.world_size = case["world_size"]
            moe_train_test.rank_per_dev = case["world_size"]
            moe_train_test.server_index = 0

            with redirect_stdout(StringIO()):
                result = moe_train_test._run_continue_test(case_data)
        finally:
            (
                moe_train_test.dispatch_with_handle,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.combine_with_topk_weights,
                moe_train_test.is_performance,
                moe_train_test.gen_npu,
                moe_train_test.compare_dispatch_outputs,
                moe_train_test.compare_combine_outputs,
                moe_train_test.clean_previous_profiling_result,
                moe_train_test.world_size,
                moe_train_test.rank_per_dev,
                moe_train_test.server_index,
            ) = old_values

        self.assertEqual(result, {
            "continue_dispatch_recv_x_rank0": True,
            "continue_combine_combined_x_rank0": True,
        })
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], moe_train_test.run_continue_npu)
        self.assertEqual(calls[0][1]["topk_weights"], [None, None])
        self.assertIs(calls[0][1]["init_topk_weights"][0], topk_weights_per_rank[0])
        self.assertIs(calls[0][1]["init_topk_weights"][1], topk_weights_per_rank[1])
        self.assertEqual(calls[0][1]["init_cached_handle"], [True, True])
        self.assertIs(compare_seen["rank_outputs"], rank_outputs)
        self.assertEqual(compare_seen["tag"], "continue.dispatch")
        self.assertEqual(compare_seen["combine_tag"], "continue.combine")
        self.assertIs(compare_seen["case_data"]["dispatch_goldens"][0].handle, primed_handles[0])
        self.assertIs(compare_seen["case_data"]["dispatch_goldens"][1].handle, primed_handles[1])
        self.assertIs(case_data["dispatch_goldens"][0].handle, dispatch[0].handle)

    def test_runner_splits_recv_scales_and_casts_fp8_combine_x(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("torch 不支持 float8_e4m3fn")
        recv_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16).to(fp8_dtype)
        recv_scales = torch.ones((2, 1), dtype=torch.float32)
        recv_x_main, split_scales = moe_train_test._split_recv_x_and_scales((recv_x, recv_scales))
        self.assertIs(recv_x_main, recv_x)
        self.assertIs(split_scales, recv_scales)
        combine_x = moe_train_test._make_combine_x((recv_x, recv_scales))
        self.assertIsInstance(combine_x, torch.Tensor)
        self.assertIn(combine_x.dtype, (torch.float16, torch.bfloat16))
        self.assertEqual(tuple(combine_x.shape), tuple(recv_x.shape))
        plain_x = torch.ones((2, 2), dtype=torch.bfloat16)
        self.assertIs(moe_train_test._make_combine_x(plain_x), plain_x)

    def test_runner_combine_uses_cast_input_for_fp8_recv_x(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("torch 不支持 float8_e4m3fn")
        x_per_rank = [
            (
                torch.tensor([[1, 10], [2, 20]], dtype=torch.float16).to(fp8_dtype),
                torch.ones((2, 1), dtype=torch.float32),
            ),
            (
                torch.tensor([[3, 30], [4, 40]], dtype=torch.float16).to(fp8_dtype),
                torch.ones((2, 1), dtype=torch.float32),
            ),
        ]
        topk_idx_per_rank = [
            torch.tensor([[0], [1]], dtype=torch.int32),
            torch.tensor([[0], [1]], dtype=torch.int32),
        ]
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            None,
            num_experts=2,
            num_max_tokens_per_rank=2,
            do_cpu_sync=False,
        )
        combine_x_per_rank = moe_train_test.combine_x_for_dispatch_goldens(dispatch)
        combine = gen_combine_golden_cpu(combine_x_per_rank, [d.handle for d in dispatch])
        captured = {}

        def fake_gen_npu(target_func, **kwargs):
            captured.update(kwargs)
            return [
                {"combined_x": c.combined_x, "combined_topk_weights": c.combined_topk_weights}
                for c in combine
            ]

        old_values = (
            moe_train_test.gen_npu,
            moe_train_test.clean_previous_profiling_result,
            moe_train_test.is_performance,
            moe_train_test.combine_with_topk_weights,
            moe_train_test.dispatch_with_handle,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.world_size,
            moe_train_test.rank_per_dev,
            moe_train_test.server_index,
        )
        try:
            moe_train_test.gen_npu = fake_gen_npu
            moe_train_test.clean_previous_profiling_result = lambda paths: None
            moe_train_test.is_performance = False
            moe_train_test.combine_with_topk_weights = False
            moe_train_test.dispatch_with_handle = False
            moe_train_test.dispatch_with_topk_weights = True
            moe_train_test.world_size = 2
            moe_train_test.rank_per_dev = 2
            moe_train_test.server_index = 0
            with redirect_stdout(StringIO()):
                result = moe_train_test._run_combine_test({
                    "dispatch_goldens": dispatch,
                    "combine_goldens": combine,
                })
        finally:
            (
                moe_train_test.gen_npu,
                moe_train_test.clean_previous_profiling_result,
                moe_train_test.is_performance,
                moe_train_test.combine_with_topk_weights,
                moe_train_test.dispatch_with_handle,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.world_size,
                moe_train_test.rank_per_dev,
                moe_train_test.server_index,
            ) = old_values

        self.assertTrue(all(result.values()))
        self.assertEqual(len(captured["combine_x"]), 2)
        for combine_x in captured["combine_x"]:
            self.assertIsInstance(combine_x, torch.Tensor)
            self.assertIn(combine_x.dtype, (torch.float16, torch.bfloat16))

    def test_runner_combine_topk_weights_independent_from_dispatch_weights(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 1, "num_experts": 2, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, _ = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            None,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=False,
        )
        self.assertTrue(all(item.recv_topk_weights is None for item in dispatch))

        old_values = (
            moe_train_test.combine_with_topk_weights,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.dispatch_with_handle,
            moe_train_test.world_size,
            moe_train_test.rank_per_dev,
            moe_train_test.server_index,
        )
        try:
            moe_train_test.combine_with_topk_weights = True
            moe_train_test.dispatch_with_topk_weights = False
            moe_train_test.dispatch_with_handle = True
            moe_train_test.world_size = case["world_size"]
            moe_train_test.rank_per_dev = case["world_size"]
            moe_train_test.server_index = 0

            combine_topk_weights = moe_train_test.combine_topk_weights_for_golden(dispatch)
            combine_x_per_rank = moe_train_test.combine_x_for_dispatch_goldens(dispatch)
            combine = gen_combine_golden_cpu(
                combine_x_per_rank,
                [item.handle for item in dispatch],
                topk_weights_per_rank=combine_topk_weights,
            )
            captured = {}

            def fake_gen_npu(target_func, **kwargs):
                captured.update(kwargs)
                return [
                    {"combined_x": item.combined_x, "combined_topk_weights": item.combined_topk_weights}
                    for item in combine
                ]

            old_runner_values = (
                moe_train_test.gen_npu,
                moe_train_test.clean_previous_profiling_result,
                moe_train_test.is_performance,
            )
            try:
                moe_train_test.gen_npu = fake_gen_npu
                moe_train_test.clean_previous_profiling_result = lambda paths: None
                moe_train_test.is_performance = False
                with redirect_stdout(StringIO()):
                    result = moe_train_test._run_combine_test({
                        "dispatch_goldens": dispatch,
                        "combine_goldens": combine,
                    })
            finally:
                (
                    moe_train_test.gen_npu,
                    moe_train_test.clean_previous_profiling_result,
                    moe_train_test.is_performance,
                ) = old_runner_values
        finally:
            (
                moe_train_test.combine_with_topk_weights,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.dispatch_with_handle,
                moe_train_test.world_size,
                moe_train_test.rank_per_dev,
                moe_train_test.server_index,
            ) = old_values

        self.assertTrue(all(result.values()))
        self.assertEqual(len(captured["combine_topk_weights"]), case["world_size"])
        for rank, weight in enumerate(captured["combine_topk_weights"]):
            self.assertIsNotNone(weight)
            self.assertEqual(weight.shape[0], feature_main(dispatch[rank].recv_x).shape[0])

    def test_runner_compare_always_prints_pass_detail(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        combine_outputs = [
            {"combined_x": c.combined_x, "combined_topk_weights": c.combined_topk_weights}
            for c in combine
        ]
        old_value = moe_train_test.compare_print_detail_max_elements
        try:
            moe_train_test.compare_print_detail_max_elements = 4
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = moe_train_test.compare_combine_outputs(combine_outputs, combine, "combine")
            self.assertTrue(all(result.values()))
            output = stdout.getvalue()
            self.assertIn("[INFO] combine.combined_x.rank0 PASSED! toolss detail follows.", output)
            self.assertIn("[INFO] combine.combined_x.rank0 toolss detail:", output)
            self.assertIn("DiffThd", output)
        finally:
            moe_train_test.compare_print_detail_max_elements = old_value

    def test_runner_compare_pass_detail_uses_real_toolss_output(self):
        case = {"world_size": 2, "num_tokens": 1, "hidden": 2, "num_topk": 1, "num_experts": 2, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        combine_outputs = [
            {"combined_x": c.combined_x, "combined_topk_weights": c.combined_topk_weights}
            for c in combine
        ]
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = moe_train_test.compare_combine_outputs(combine_outputs, combine, "combine")
        self.assertTrue(all(result.values()))
        output = stdout.getvalue()
        self.assertIn("ExpectOut", output)
        self.assertIn("RealOut", output)
        self.assertIn("FpDiff", output)
        self.assertNotIn("toolss 诊断失败", output)

    def test_runner_compare_failure_prints_variable_before_detail(self):
        case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 2, "num_experts": 4, "include_invalid": False}
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
        dispatch = gen_dispatch_golden_cpu(
            x_per_rank,
            topk_idx_per_rank,
            topk_weights_per_rank,
            num_experts=case["num_experts"],
            num_max_tokens_per_rank=case["num_tokens"],
            do_cpu_sync=True,
        )
        combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
        bad_combined_x = combine[0].combined_x.clone()
        feature_main(bad_combined_x)[0, 0] += 1
        combine_outputs = [
            {"combined_x": bad_combined_x, "combined_topk_weights": combine[0].combined_topk_weights},
            {"combined_x": combine[1].combined_x, "combined_topk_weights": combine[1].combined_topk_weights},
        ]
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = moe_train_test.compare_combine_outputs(combine_outputs, combine, "combine")
        self.assertFalse(result["combine_combined_x_rank0"])
        output = stdout.getvalue()
        error_idx = output.index("[ERROR] combine.combined_x.rank0 FAILED!")
        detail_idx = output.index("[INFO] combine.combined_x.rank0 toolss detail:")
        toolss_idx = output.index("DiffThd", detail_idx)
        self.assertLess(error_idx, detail_idx)
        self.assertLess(detail_idx, toolss_idx)
        self.assertIn("################## combine compare table ##################", output)
        self.assertRegex(output, r"\|\s+\| combined_x\s+\| \x1b\[31mFAIL\x1b\[0m\s+\|")

    def test_runner_print_summary_is_structured(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            moe_train_test.print_summary("dispatch", {"a": True, "b": False})
        output = stdout.getvalue()
        self.assertIn("[INFO] dispatch summary: total=2, passed=1, failed=1", output)
        self.assertIn("[ERROR] dispatch FAILED! failed_items=['b']", output)

    def test_runner_save_accuracy_result_to_csv(self):
        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_accuracy_csv"
                moe_train_test.is_save_result_to_csv = True

                case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 2, "num_experts": 4}
                x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
                dispatch = gen_dispatch_golden_cpu(
                    x_per_rank,
                    topk_idx_per_rank,
                    topk_weights_per_rank,
                    num_experts=case["num_experts"],
                    num_max_tokens_per_rank=case["num_tokens"],
                )
                combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
                rank_outputs = [
                    {"combined_x": item.combined_x, "combined_topk_weights": item.combined_topk_weights}
                    for item in combine
                ]

                with redirect_stdout(StringIO()):
                    result = moe_train_test.compare_combine_outputs(rank_outputs, combine, "combine")
                moe_train_test.save_accuracy_result("combine", result)

                accuracy_path = Path(tmp) / "csv_result" / "moe_distribute.csv"
                with open(accuracy_path, "r", encoding="UTF8") as f:
                    rows = list(csv.reader(f))
                self.assertEqual(
                    rows[0][:2],
                    ["case_accuracy_csv", "combine.combined_x.rank0"],
                )
                self.assertEqual(
                    rows[1][:2],
                    ["case_accuracy_csv", "combine.combined_x.rank1"],
                )
                self.assertTrue(rows[0][2].startswith("('PASS'"))
                self.assertTrue(rows[1][2].startswith("('PASS'"))
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
            ) = old_values

    def test_runner_compare_and_csv_use_toolss_result(self):
        calls = []

        class FakeToolss:
            @staticmethod
            def data_compare_np(*args, **kwargs):
                calls.append((args, kwargs))
                print("[FAKE_TOOLSS_DETAIL_FAIL]")
                return "FAIL", 12.5, "from_toolss"

        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
            compare_module.toolss,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_toolss_csv"
                moe_train_test.is_save_result_to_csv = True
                compare_module.toolss = FakeToolss()

                case = {"world_size": 2, "num_tokens": 1, "hidden": 2, "num_topk": 1, "num_experts": 2}
                x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
                dispatch = gen_dispatch_golden_cpu(
                    x_per_rank,
                    topk_idx_per_rank,
                    topk_weights_per_rank,
                    num_experts=case["num_experts"],
                    num_max_tokens_per_rank=case["num_tokens"],
                )
                combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
                rank_outputs = [
                    {"combined_x": item.combined_x, "combined_topk_weights": item.combined_topk_weights}
                    for item in combine
                ]

                stdout = StringIO()
                with redirect_stdout(stdout):
                    result = moe_train_test.compare_combine_outputs(rank_outputs, combine, "combine")
                self.assertFalse(result["combine_combined_x_rank0"])
                self.assertEqual(len(calls), 2)
                self.assertIn("[FAKE_TOOLSS_DETAIL_FAIL]", stdout.getvalue())
                self.assertIn("[INFO] combine.combined_x.rank0 toolss result: ('FAIL', 12.5, 'from_toolss')", stdout.getvalue())
                moe_train_test.save_accuracy_result("combine", result)
                self.assertEqual(len(calls), 2)

                accuracy_path = Path(tmp) / "csv_result" / "moe_distribute.csv"
                with open(accuracy_path, "r", encoding="UTF8") as f:
                    rows = list(csv.reader(f))
                self.assertEqual(rows[0][2], "('FAIL', 12.5, 'from_toolss')")
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
                compare_module.toolss,
            ) = old_values

    def test_runner_compare_pass_detail_always_prints_and_calls_toolss_once(self):
        calls = []

        class FakeToolss:
            @staticmethod
            def data_compare_np(*args, **kwargs):
                calls.append((args, kwargs))
                print("[FAKE_TOOLSS_DETAIL_PASS]")
                return "PASS", 100.0, "from_toolss"

        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
            compare_module.toolss,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_toolss_once"
                moe_train_test.is_save_result_to_csv = True
                compare_module.toolss = FakeToolss()

                case = {"world_size": 2, "num_tokens": 1, "hidden": 2, "num_topk": 1, "num_experts": 2}
                x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
                dispatch = gen_dispatch_golden_cpu(
                    x_per_rank,
                    topk_idx_per_rank,
                    topk_weights_per_rank,
                    num_experts=case["num_experts"],
                    num_max_tokens_per_rank=case["num_tokens"],
                )
                combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
                rank_outputs = [
                    {"combined_x": item.combined_x, "combined_topk_weights": item.combined_topk_weights}
                    for item in combine
                ]

                stdout = StringIO()
                with redirect_stdout(stdout):
                    result = moe_train_test.compare_combine_outputs(rank_outputs, combine, "combine")
                self.assertTrue(all(result.values()))
                self.assertEqual(len(calls), 2)
                self.assertIn("[FAKE_TOOLSS_DETAIL_PASS]", stdout.getvalue())
                self.assertIn("[INFO] combine.combined_x.rank0 PASSED! toolss detail follows.", stdout.getvalue())
                self.assertIn("[INFO] combine.combined_x.rank0 toolss result: ('PASS', 100.0, 'from_toolss')", stdout.getvalue())
                moe_train_test.save_accuracy_result("combine", result)
                self.assertEqual(len(calls), 2)

                accuracy_path = Path(tmp) / "csv_result" / "moe_distribute.csv"
                with open(accuracy_path, "r", encoding="UTF8") as f:
                    rows = list(csv.reader(f))
                self.assertEqual(rows[0][2], "('PASS', 100.0, 'from_toolss')")
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
                compare_module.toolss,
            ) = old_values

    def test_runner_save_accuracy_result_can_disable_csv(self):
        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_no_csv"
                moe_train_test.is_save_result_to_csv = False
                moe_train_test.save_accuracy_result("dispatch", {"dispatch_recv_x_rank0": True})
                self.assertFalse((Path(tmp) / "csv_result").exists())
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
            ) = old_values

    def test_runner_save_accuracy_missing_toolss_result_fails(self):
        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_missing_toolss"
                moe_train_test.is_save_result_to_csv = True
                with self.assertRaisesRegex(RuntimeError, "未生成 toolss 对比结果"):
                    moe_train_test.save_accuracy_result("dispatch", {"dispatch_recv_x_rank0": True})

                accuracy_path = Path(tmp) / "csv_result" / "moe_distribute.csv"
                with open(accuracy_path, "r", encoding="UTF8") as f:
                    rows = list(csv.reader(f))
                self.assertEqual(rows[0][0], "case_missing_toolss")
                self.assertEqual(rows[0][1], "dispatch.toolss_result_missing")
                self.assertTrue(rows[0][2].startswith("('FAIL', 0.0"))
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
            ) = old_values

    def test_runner_structural_compare_failure_writes_fail_csv(self):
        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_structural_fail"
                moe_train_test.is_save_result_to_csv = True

                case = {"world_size": 2, "num_tokens": 2, "hidden": 2, "num_topk": 2, "num_experts": 4}
                x_per_rank, topk_idx_per_rank, topk_weights_per_rank = _make_case(**case)
                dispatch = gen_dispatch_golden_cpu(
                    x_per_rank,
                    topk_idx_per_rank,
                    topk_weights_per_rank,
                    num_experts=case["num_experts"],
                    num_max_tokens_per_rank=case["num_tokens"],
                )
                combine = gen_combine_golden_cpu([d.recv_x for d in dispatch], [d.handle for d in dispatch])
                bad_combined_x = combine[0].combined_x[:1]
                rank_outputs = [
                    {"combined_x": bad_combined_x, "combined_topk_weights": combine[0].combined_topk_weights},
                    {"combined_x": combine[1].combined_x, "combined_topk_weights": combine[1].combined_topk_weights},
                ]

                with redirect_stdout(StringIO()):
                    result = moe_train_test.compare_combine_outputs(rank_outputs, combine, "combine")
                self.assertFalse(result["combine_combined_x_rank0"])
                moe_train_test.save_accuracy_result("combine", result)

                accuracy_path = Path(tmp) / "csv_result" / "moe_distribute.csv"
                with open(accuracy_path, "r", encoding="UTF8") as f:
                    rows = list(csv.reader(f))
                self.assertEqual(rows[0][0], "case_structural_fail")
                self.assertEqual(rows[0][1], "combine.combined_x.rank0")
                self.assertTrue(rows[0][2].startswith("('FAIL', 0.0"))
                self.assertIn("shape 不一致", rows[0][2])
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
            ) = old_values

    def test_runner_profiling_parser(self):
        with TemporaryDirectory() as tmp:
            profiling_dir = Path(tmp)
            summary = profiling_dir / "op_summary_0.csv"
            summary.write_text(
                "Device_id,OP Type,Task Duration(us),Task Wait Time(us)\n"
                "0,MoeEpDispatch,10,0\n"
                "0,MoeEpDispatchEpilogue,20,0\n"
                "0,MoeEpCombine,30,0\n",
                encoding="utf-8",
            )
            perf = moe_train_test.get_performance_from_profiling(
                profiling_dir, ["MoeEpDispatch", "MoeEpDispatchEpilogue", "MoeEpCombine"]
            )
        self.assertEqual(perf["MoeEpDispatch"]["mean_us"], 10.0)
        self.assertEqual(perf["MoeEpDispatchEpilogue"]["min_us"], 20.0)
        self.assertEqual(perf["MoeEpCombine"]["max_us"], 30.0)
        self.assertEqual(perf["MoeEpDispatchTotal"]["mean_us"], 30.0)
        self.assertEqual(perf["MoeEpDispatchTotal"]["count"], 1.0)

    def test_runner_profiling_parser_uses_last_50_samples(self):
        old_tail_count = moe_train_test.performance_tail_count
        try:
            moe_train_test.performance_tail_count = 50
            with TemporaryDirectory() as tmp:
                profiling_dir = Path(tmp)
                summary = profiling_dir / "op_summary_0.csv"
                rows = ["Device_id,OP Type,Task Duration(us),Task Wait Time(us)"]
                rows.extend(f"0,MoeEpDispatch,{duration},0" for duration in range(1, 61))
                summary.write_text("\n".join(rows) + "\n", encoding="utf-8")
                perf = moe_train_test.get_performance_from_profiling(profiling_dir, ["MoeEpDispatch"])
            self.assertEqual(perf["MoeEpDispatch"]["mean_us"], 35.5)
            self.assertEqual(perf["MoeEpDispatch"]["min_us"], 11.0)
            self.assertEqual(perf["MoeEpDispatch"]["max_us"], 60.0)
            self.assertEqual(perf["MoeEpDispatch"]["count"], 50.0)
            self.assertEqual(perf["MoeEpDispatch"]["total_count"], 60.0)
        finally:
            moe_train_test.performance_tail_count = old_tail_count

    def test_runner_profiling_parser_uses_last_50_samples_per_rank_file(self):
        old_tail_count = moe_train_test.performance_tail_count
        try:
            moe_train_test.performance_tail_count = 50
            with TemporaryDirectory() as tmp:
                profiling_dir = Path(tmp)
                rows0 = ["Device_id,OP Type,Task Duration(us),Task Wait Time(us)"]
                rows0.extend(f"0,MoeEpDispatch,{duration},0" for duration in range(1, 61))
                (profiling_dir / "op_summary_0.csv").write_text("\n".join(rows0) + "\n", encoding="utf-8")

                rows1 = ["Device_id,OP Type,Task Duration(us),Task Wait Time(us)"]
                rows1.extend(f"1,MoeEpDispatch,{duration},0" for duration in range(101, 161))
                (profiling_dir / "op_summary_1.csv").write_text("\n".join(rows1) + "\n", encoding="utf-8")

                perf = moe_train_test.get_performance_from_profiling(profiling_dir, ["MoeEpDispatch"])
            self.assertEqual(perf["MoeEpDispatch"]["count"], 100.0)
            self.assertEqual(perf["MoeEpDispatch"]["total_count"], 120.0)
            self.assertEqual(perf["MoeEpDispatch"]["mean_us"], 85.5)
            self.assertEqual(perf["MoeEpDispatch"]["min_us"], 11.0)
            self.assertEqual(perf["MoeEpDispatch"]["max_us"], 160.0)
        finally:
            moe_train_test.performance_tail_count = old_tail_count

    def test_runner_save_performance_result_to_csv(self):
        old_values = (
            moe_train_test.REPO_ROOT,
            moe_train_test.current_case_name,
            moe_train_test.is_save_result_to_csv,
        )
        try:
            with TemporaryDirectory() as tmp:
                moe_train_test.REPO_ROOT = Path(tmp)
                moe_train_test.current_case_name = "case_performance_csv"
                moe_train_test.is_save_result_to_csv = True
                performance = {
                    "MoeEpDispatch": {
                        "mean_us": 10.5,
                        "min_us": 10.0,
                        "max_us": 11.0,
                        "count": 50.0,
                        "total_count": 60.0,
                        "samples": [10.0],
                    },
                    "MoeEpDispatchEpilogue": {
                        "mean_us": 20.5,
                        "min_us": 20.0,
                        "max_us": 21.0,
                        "count": 50.0,
                        "total_count": 60.0,
                        "samples": [20.0],
                    },
                    "MoeEpDispatchTotal": {
                        "mean_us": 31.0,
                        "min_us": 30.0,
                        "max_us": 32.0,
                        "count": 50.0,
                        "total_count": 60.0,
                        "samples": [30.0],
                    },
                }
                moe_train_test.save_performance_result("dispatch", performance, moe_train_test.dispatch_op_types)

                performance_path = Path(tmp) / "csv_result" / "dispatch_performance_result.csv"
                with open(performance_path, "r", encoding="UTF8") as f:
                    rows = list(csv.reader(f))
                self.assertEqual(
                    rows[0],
                    [
                        "op_type",
                        "case_name",
                        "mc2_shared_average",
                        "mc2_moe_average",
                        "mc2_shared_min",
                        "mc2_moe_min",
                        "recordtime",
                    ],
                )
                self.assertEqual(
                    [row[0] for row in rows[1:]],
                    ["MoeEpDispatch", "MoeEpDispatchEpilogue", "MoeEpDispatchTotal"],
                )
                self.assertEqual(rows[1][:6], ["MoeEpDispatch", "case_performance_csv", "10.5", "10.5", "10.0", "10.0"])
        finally:
            (
                moe_train_test.REPO_ROOT,
                moe_train_test.current_case_name,
                moe_train_test.is_save_result_to_csv,
            ) = old_values

    def test_runner_performance_op_types(self):
        self.assertEqual(moe_train_test.dispatch_op_types, ["MoeEpDispatch", "MoeEpDispatchEpilogue"])
        self.assertEqual(moe_train_test.combine_op_types, ["MoeEpCombine"])
        self.assertNotIn("aclnnMoeEpDispatch", moe_train_test.continue_op_types)
        self.assertNotIn("aclnnMoeEpCombine", moe_train_test.continue_op_types)

    def test_runner_performance_exp_runs_on_second_iter(self):
        old_exp = moe_train_test.torch.exp
        calls = []

        def fake_exp(tensor):
            calls.append(tensor)
            return tensor

        try:
            moe_train_test.torch.exp = fake_exp
            tensor = object()
            moe_train_test.run_performance_exp(0, tensor)
            self.assertEqual(len(calls), 0)
            moe_train_test.run_performance_exp(1, tensor)
            self.assertEqual(len(calls), 100)
            self.assertTrue(all(item is tensor for item in calls))
        finally:
            moe_train_test.torch.exp = old_exp

    def test_runner_apply_stream_limit(self):
        class FakeNpu:
            def __init__(self):
                self.calls = []
                self.stream = object()

            def current_stream(self):
                return self.stream

            def set_stream_limit(self, stream, aic_core_num, aiv_core_num):
                self.calls.append((stream, aic_core_num, aiv_core_num))

        old_values = (
            moe_train_test.is_full_core,
            moe_train_test.core_num,
            moe_train_test.aic_core_num,
            moe_train_test.aiv_core_num,
        )
        had_npu = hasattr(moe_train_test.torch, "npu")
        old_npu = getattr(moe_train_test.torch, "npu", None)
        fake_npu = FakeNpu()
        try:
            moe_train_test.torch.npu = fake_npu
            moe_train_test.is_full_core = False
            moe_train_test.core_num = 8
            moe_train_test.aic_core_num = moe_train_test.core_num
            moe_train_test.aiv_core_num = 2 * moe_train_test.core_num
            stdout = StringIO()
            with redirect_stdout(stdout):
                moe_train_test.apply_stream_limit()
            self.assertEqual(fake_npu.calls, [(fake_npu.stream, 8, 16)])
            self.assertIn("当前单算子模式仅使用：8 aic核，16 aiv核", stdout.getvalue())

            fake_npu.calls.clear()
            moe_train_test.is_full_core = True
            stdout = StringIO()
            with redirect_stdout(stdout):
                moe_train_test.apply_stream_limit()
            self.assertEqual(fake_npu.calls, [])
            self.assertIn("当前单算子模式仅使用：8 aic核，16 aiv核", stdout.getvalue())
        finally:
            (
                moe_train_test.is_full_core,
                moe_train_test.core_num,
                moe_train_test.aic_core_num,
                moe_train_test.aiv_core_num,
            ) = old_values
            if had_npu:
                moe_train_test.torch.npu = old_npu
            else:
                delattr(moe_train_test.torch, "npu")

    def test_runner_check_setting_performance_invalid(self):
        old_values = (moe_train_test.is_performance, moe_train_test.loop, moe_train_test.repeat, moe_train_test.warmup)
        try:
            moe_train_test.is_performance = 1
            moe_train_test.loop = 2
            moe_train_test.repeat = 2
            moe_train_test.warmup = 1
            with self.assertRaisesRegex(ValueError, "性能模式"):
                moe_train_test.check_setting()
        finally:
            moe_train_test.is_performance, moe_train_test.loop, moe_train_test.repeat, moe_train_test.warmup = old_values

    def test_runner_check_setting_allows_independent_combine_topk_weights(self):
        old_values = (
            moe_train_test.combine_with_topk_weights,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.dispatch_with_handle,
        )
        try:
            moe_train_test.combine_with_topk_weights = True

            moe_train_test.dispatch_with_topk_weights = False
            moe_train_test.dispatch_with_handle = False
            moe_train_test.check_setting()

            moe_train_test.dispatch_with_topk_weights = True
            moe_train_test.dispatch_with_handle = True
            moe_train_test.check_setting()

            moe_train_test.dispatch_with_handle = False
            moe_train_test.check_setting()
        finally:
            (
                moe_train_test.combine_with_topk_weights,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.dispatch_with_handle,
            ) = old_values

    def test_runner_build_current_case_name(self):
        old_values = (
            moe_train_test.num_tokens,
            moe_train_test.hidden,
            moe_train_test.num_topk,
            moe_train_test.num_experts,
            moe_train_test.num_max_tokens_per_rank,
            moe_train_test.is_variable_bs,
            moe_train_test.x_dtype,
            moe_train_test.scales_dtype,
            moe_train_test.is_random_scales_dtype,
            moe_train_test.dispatch_with_topk_weights,
            moe_train_test.dispatch_do_cpu_sync,
            moe_train_test.dispatch_with_handle,
            moe_train_test.world_size,
        )
        try:
            moe_train_test.num_tokens = 8
            moe_train_test.hidden = 64
            moe_train_test.num_topk = 2
            moe_train_test.num_experts = 4
            moe_train_test.num_max_tokens_per_rank = 16
            moe_train_test.is_variable_bs = True
            moe_train_test.x_dtype = "fp8_e4m3fn"
            moe_train_test.scales_dtype = "fp8_e8m0"
            moe_train_test.is_random_scales_dtype = True
            moe_train_test.dispatch_with_topk_weights = False
            moe_train_test.dispatch_do_cpu_sync = False
            moe_train_test.dispatch_with_handle = True
            moe_train_test.world_size = 2

            self.assertEqual(
                moe_train_test.build_current_case_name(1),
                "MOE_EP_TRAIN_nt8_vbs1_h64_topk2_experts4_nmt16_xdtypefp8_e4m3fn_sdtypefp8_e8m0_rsdtype1_dwt0_sync0_cached1_ranks2_ID0001",
            )
        finally:
            (
                moe_train_test.num_tokens,
                moe_train_test.hidden,
                moe_train_test.num_topk,
                moe_train_test.num_experts,
                moe_train_test.num_max_tokens_per_rank,
                moe_train_test.is_variable_bs,
                moe_train_test.x_dtype,
                moe_train_test.scales_dtype,
                moe_train_test.is_random_scales_dtype,
                moe_train_test.dispatch_with_topk_weights,
                moe_train_test.dispatch_do_cpu_sync,
                moe_train_test.dispatch_with_handle,
                moe_train_test.world_size,
            ) = old_values

    def test_runner_default_dim_scales_uses_fp32_scales_dtype_helper(self):
        self.assertEqual(moe_train_test._default_dim_scales(256, "float32"), 2)
        self.assertEqual(moe_train_test._default_dim_scales(256, "float"), 2)
        self.assertEqual(moe_train_test._default_dim_scales(256, "fp8_e8m0"), 8)
        self.assertEqual(moe_train_test._default_dim_scales(257, "float32"), 3)
        self.assertEqual(moe_train_test._default_dim_scales(257, "fp8_e8m0"), 10)
        self.assertEqual(moe_train_test._default_dim_scales(1, "fp8_e8m0"), 2)
        self.assertTrue(moe_train_test._is_supported_scales_dtype_name("float"))

    def test_runner_check_setting_x_dtype_support(self):
        old_values = (moe_train_test.x_dtype, moe_train_test.scales_dtype, moe_train_test.dim_scales)
        try:
            for dtype_name in ("bfloat16", "bf16", "float16", "fp16"):
                with self.subTest(dtype_name=dtype_name):
                    moe_train_test.x_dtype = dtype_name
                    moe_train_test.scales_dtype = "float32"
                    moe_train_test.dim_scales = 2
                    moe_train_test.check_setting()

            for dtype_name in ("fp8_e4m3fn", "float8_e4m3fn", "fp8_e5m2", "float8_e5m2"):
                for scales_dtype in ("float32", "float", "fp8_e8m0", "float8_e8m0"):
                    with self.subTest(dtype_name=dtype_name, scales_dtype=scales_dtype):
                        moe_train_test.x_dtype = dtype_name
                        moe_train_test.scales_dtype = scales_dtype
                        moe_train_test.dim_scales = 2
                        moe_train_test.check_setting()

            moe_train_test.x_dtype = "fp8_e4m3fn"
            moe_train_test.scales_dtype = "bad_scales"
            moe_train_test.dim_scales = 2
            with self.assertRaisesRegex(ValueError, "不支持的 scales_dtype"):
                moe_train_test.check_setting()

            moe_train_test.scales_dtype = "fp8_e8m0"
            moe_train_test.dim_scales = 0
            with self.assertRaisesRegex(ValueError, "dim_scales"):
                moe_train_test.check_setting()
        finally:
            moe_train_test.x_dtype, moe_train_test.scales_dtype, moe_train_test.dim_scales = old_values

    def test_runner_random_scales_dtype_is_seeded(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("torch 不支持 float8_e4m3fn")

        old_values = (
            moe_train_test.x_dtype,
            moe_train_test.scales_dtype,
            moe_train_test.is_random_scales_dtype,
            moe_train_test.dim_scales,
            moe_train_test.world_size,
            moe_train_test.num_tokens,
            moe_train_test.hidden,
            moe_train_test.is_x_random,
            moe_train_test.random_seed,
        )
        try:
            moe_train_test.x_dtype = "fp8_e4m3fn"
            moe_train_test.is_random_scales_dtype = True
            moe_train_test.world_size = 2
            moe_train_test.num_tokens = 3
            moe_train_test.hidden = 257
            moe_train_test.is_x_random = False

            for seed, expected_effective in ((1092, "float32"), (1099, "fp8_e8m0")):
                with self.subTest(seed=seed, expected_effective=expected_effective):
                    moe_train_test.scales_dtype = "float32"
                    moe_train_test.dim_scales = 4
                    moe_train_test.random_seed = seed

                    effective = moe_train_test.resolve_scales_dtype_cpu()
                    self.assertEqual(effective, expected_effective)
                    self.assertEqual(effective, moe_train_test.resolve_scales_dtype_cpu())
                    expected_dim_scales = moe_train_test._default_dim_scales(
                        moe_train_test.hidden,
                        expected_effective,
                    )
                    self.assertEqual(moe_train_test.resolve_dim_scales_cpu(effective), expected_dim_scales)
                    xs = moe_train_test.gen_x_per_rank_cpu()
                    expected_dtype = torch.uint8 if effective == "fp8_e8m0" else torch.float32
                    self.assertTrue(all(feature_scales(x).dtype == expected_dtype for x in xs))
                    self.assertTrue(all(feature_scales(x).shape == (3, expected_dim_scales) for x in xs))

            moe_train_test.is_random_scales_dtype = False
            moe_train_test.scales_dtype = "fp8_e8m0"
            moe_train_test.dim_scales = 4
            self.assertEqual(moe_train_test.resolve_scales_dtype_cpu(), "fp8_e8m0")
            self.assertEqual(moe_train_test.resolve_dim_scales_cpu(), 4)
            xs = moe_train_test.gen_x_per_rank_cpu()
            self.assertTrue(all(feature_scales(x).dtype == torch.uint8 for x in xs))
            self.assertTrue(all(feature_scales(x).shape == (3, 4) for x in xs))
        finally:
            (
                moe_train_test.x_dtype,
                moe_train_test.scales_dtype,
                moe_train_test.is_random_scales_dtype,
                moe_train_test.dim_scales,
                moe_train_test.world_size,
                moe_train_test.num_tokens,
                moe_train_test.hidden,
                moe_train_test.is_x_random,
                moe_train_test.random_seed,
            ) = old_values

    def test_runner_gen_x_per_rank_cpu_uses_tensor_or_fp8_tuple(self):
        old_values = (
            moe_train_test.x_dtype,
            moe_train_test.scales_dtype,
            moe_train_test.dim_scales,
            moe_train_test.world_size,
            moe_train_test.num_tokens,
            moe_train_test.hidden,
            moe_train_test.is_x_random,
        )
        try:
            moe_train_test.world_size = 2
            moe_train_test.num_tokens = 3
            moe_train_test.hidden = 4
            moe_train_test.is_x_random = False

            moe_train_test.x_dtype = "bfloat16"
            moe_train_test.scales_dtype = "float32"
            xs = moe_train_test.gen_x_per_rank_cpu()
            self.assertTrue(all(not isinstance(x, tuple) for x in xs))
            self.assertTrue(all(feature_main(x).dtype == torch.bfloat16 for x in xs))

            for dtype_name in ("float16", "fp16"):
                with self.subTest(dtype_name=dtype_name):
                    moe_train_test.x_dtype = dtype_name
                    xs = moe_train_test.gen_x_per_rank_cpu()
                    self.assertTrue(all(not isinstance(x, tuple) for x in xs))
                    self.assertTrue(all(feature_main(x).dtype == torch.float16 for x in xs))
                    self.assertTrue(all(feature_scales(x) is None for x in xs))

            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is not None:
                moe_train_test.x_dtype = "fp8_e4m3fn"
                moe_train_test.scales_dtype = "float32"
                moe_train_test.dim_scales = 5
                xs = moe_train_test.gen_x_per_rank_cpu()
                self.assertTrue(all(isinstance(x, tuple) for x in xs))
                self.assertTrue(all(feature_main(x).dtype == fp8_dtype for x in xs))
                self.assertTrue(all(feature_scales(x).dtype == torch.float32 for x in xs))
                self.assertTrue(all(feature_scales(x).shape == (3, 5) for x in xs))

                moe_train_test.scales_dtype = "fp8_e8m0"
                xs = moe_train_test.gen_x_per_rank_cpu()
                self.assertTrue(all(isinstance(x, tuple) for x in xs))
                self.assertTrue(all(feature_main(x).dtype == fp8_dtype for x in xs))
                self.assertTrue(all(feature_scales(x).dtype == torch.uint8 for x in xs))
                self.assertTrue(all(feature_scales(x).shape == (3, 5) for x in xs))

                moe_train_test.scales_dtype = "bad_scales"
                with self.assertRaisesRegex(ValueError, "不支持的 scales_dtype"):
                    moe_train_test.gen_x_per_rank_cpu()
        finally:
            (
                moe_train_test.x_dtype,
                moe_train_test.scales_dtype,
                moe_train_test.dim_scales,
                moe_train_test.world_size,
                moe_train_test.num_tokens,
                moe_train_test.hidden,
                moe_train_test.is_x_random,
            ) = old_values

    def test_runner_variable_bs_uses_prefix_sliced_inputs(self):
        old_values = (
            moe_train_test.x_dtype,
            moe_train_test.scales_dtype,
            moe_train_test.dim_scales,
            moe_train_test.world_size,
            moe_train_test.num_tokens,
            moe_train_test.hidden,
            moe_train_test.num_topk,
            moe_train_test.num_experts,
            moe_train_test.is_x_random,
            moe_train_test.is_variable_bs,
            moe_train_test.random_seed,
        )
        try:
            moe_train_test.world_size = 3
            moe_train_test.num_tokens = 5
            moe_train_test.hidden = 4
            moe_train_test.num_topk = 2
            moe_train_test.num_experts = 6
            moe_train_test.is_x_random = False
            moe_train_test.is_variable_bs = True
            moe_train_test.random_seed = 2026

            real_num_tokens_per_rank = moe_train_test.gen_real_num_tokens_per_rank_cpu()
            self.assertEqual(len(real_num_tokens_per_rank), 3)
            self.assertTrue(all(1 <= item <= 5 for item in real_num_tokens_per_rank))
            self.assertIn(5, real_num_tokens_per_rank)

            full_num_tokens_per_rank = [moe_train_test.num_tokens] * moe_train_test.world_size
            moe_train_test.x_dtype = "bfloat16"
            full_x = moe_train_test.gen_x_per_rank_cpu(full_num_tokens_per_rank)
            sliced_x = moe_train_test.gen_x_per_rank_cpu(real_num_tokens_per_rank)
            full_topk = moe_train_test.gen_topk_idx_per_rank_cpu(full_num_tokens_per_rank)
            sliced_topk = moe_train_test.gen_topk_idx_per_rank_cpu(real_num_tokens_per_rank)
            full_weights = moe_train_test.gen_topk_weights_per_rank_cpu(full_num_tokens_per_rank)
            sliced_weights = moe_train_test.gen_topk_weights_per_rank_cpu(real_num_tokens_per_rank)
            for rank, rows in enumerate(real_num_tokens_per_rank):
                self.assertTrue(exact_tensor_equal(f"variable.x.rank{rank}", feature_main(sliced_x[rank]), feature_main(full_x[rank])[:rows])[0])
                self.assertTrue(exact_tensor_equal(f"variable.topk.rank{rank}", sliced_topk[rank], full_topk[rank][:rows])[0])
                self.assertTrue(exact_tensor_equal(f"variable.weights.rank{rank}", sliced_weights[rank], full_weights[rank][:rows])[0])

            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is not None:
                moe_train_test.x_dtype = "fp8_e4m3fn"
                moe_train_test.scales_dtype = "fp8_e8m0"
                moe_train_test.dim_scales = 3
                full_x = moe_train_test.gen_x_per_rank_cpu(full_num_tokens_per_rank)
                sliced_x = moe_train_test.gen_x_per_rank_cpu(real_num_tokens_per_rank)
                for rank, rows in enumerate(real_num_tokens_per_rank):
                    self.assertTrue(exact_tensor_equal(f"variable.fp8.x.rank{rank}", feature_main(sliced_x[rank]), feature_main(full_x[rank])[:rows])[0])
                    self.assertTrue(exact_tensor_equal(f"variable.fp8.scales.rank{rank}", feature_scales(sliced_x[rank]), feature_scales(full_x[rank])[:rows])[0])
        finally:
            (
                moe_train_test.x_dtype,
                moe_train_test.scales_dtype,
                moe_train_test.dim_scales,
                moe_train_test.world_size,
                moe_train_test.num_tokens,
                moe_train_test.hidden,
                moe_train_test.num_topk,
                moe_train_test.num_experts,
                moe_train_test.is_x_random,
                moe_train_test.is_variable_bs,
                moe_train_test.random_seed,
            ) = old_values

    def test_init_hccl_comm_initializes_ep_hcomm_name(self):
        class FakeBackend:
            def __init__(self):
                self.calls = []

            def get_hccl_comm_name(self, rank):
                self.calls.append(rank)
                return f"hcomm-{rank}"

        class FakeGroup:
            def __init__(self, backend):
                self.backend = backend
                self.devices = []

            def _get_backend(self, device):
                self.devices.append(device)
                return self.backend

        class FakeDist:
            def __init__(self, group):
                self.group = group
                self.init_kwargs = None
                self.new_group_kwargs = None

            def init_process_group(self, **kwargs):
                self.init_kwargs = kwargs

            def new_group(self, **kwargs):
                self.new_group_kwargs = kwargs
                return self.group

        backend = FakeBackend()
        group = FakeGroup(backend)
        fake_dist = FakeDist(group)
        old_values = (
            moe_train_test.dist,
            moe_train_test.torch.device,
            moe_train_test.world_size,
            moe_train_test.master_ip,
        )
        try:
            moe_train_test.dist = fake_dist
            moe_train_test.torch.device = lambda name: name
            moe_train_test.world_size = 4
            moe_train_test.master_ip = "127.0.0.1"

            with redirect_stdout(StringIO()):
                self.assertIs(moe_train_test.init_hccl_comm(2), group)
            self.assertEqual(fake_dist.init_kwargs["backend"], "hccl")
            self.assertEqual(fake_dist.init_kwargs["rank"], 2)
            self.assertEqual(fake_dist.init_kwargs["world_size"], 4)
            self.assertEqual(fake_dist.new_group_kwargs, {"backend": "hccl", "ranks": [0, 1, 2, 3]})
            self.assertEqual(group.devices, ["npu"])
            self.assertEqual(backend.calls, [2])
        finally:
            (
                moe_train_test.dist,
                moe_train_test.torch.device,
                moe_train_test.world_size,
                moe_train_test.master_ip,
            ) = old_values


if __name__ == "__main__":
    unittest.main()

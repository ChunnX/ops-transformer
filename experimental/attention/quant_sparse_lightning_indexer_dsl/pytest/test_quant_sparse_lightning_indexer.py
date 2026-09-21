"""NPU precision tests for fused QSLI discrete MXFP4 KV gather."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch_npu


TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

import result_compare_method  # noqa: E402
from quant_sparse_lightning_indexer_golden import (  # noqa: E402
    pack_mxfp4,
    quant_sparse_lightning_indexer_golden,
)
import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl  # noqa: F401,E402


quant_sparse_lightning_indexer = (
    torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer
)


def _case(batch: int, s2: int, seed: int) -> dict[str, torch.Tensor]:
    """One TND query per batch with shuffled PA pages and candidates."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_batch = s2 // 128
    physical_pages = batch * pages_per_batch
    q = pack_mxfp4(
        torch.randn((batch, 32, 128), generator=generator) * 0.25
    )
    k = pack_mxfp4(
        torch.randn(
            (physical_pages, 128, 1, 128), generator=generator
        ) * 0.25
    )
    candidate = torch.full((batch, 1, 2048), -1, dtype=torch.int32)
    block_table = torch.empty((batch, pages_per_batch), dtype=torch.int32)
    logical_candidate_blocks = s2 // 8
    candidate_blocks = min(logical_candidate_blocks, 2048)
    for batch_idx in range(batch):
        page_begin = batch_idx * pages_per_batch
        block_table[batch_idx] = (
            torch.randperm(pages_per_batch, generator=generator).to(torch.int32)
            + page_begin
        )
        candidate[batch_idx, 0, :candidate_blocks] = torch.randperm(
            logical_candidate_blocks, generator=generator
        )[:candidate_blocks].to(torch.int32)
    return {
        "q": q,
        "k": k,
        "w": torch.randn((batch, 32), generator=generator),
        "descale_q": torch.randint(
            125, 130, (batch, 32, 2, 2),
            generator=generator, dtype=torch.uint8
        ),
        "descale_k": torch.randint(
            125, 130, (physical_pages, 128, 1, 2, 2),
            generator=generator, dtype=torch.uint8,
        ),
        "candidate": candidate,
        "candidate_length": torch.full(
            (batch, 1), candidate_blocks, dtype=torch.int32
        ),
        "cu": torch.arange(batch + 1, dtype=torch.int32),
        "used_k": torch.full((batch,), s2, dtype=torch.int32),
        "block_table": block_table,
        "offset": (
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1) * 10000
            + 37
        ),
    }


def _remove_output_offset(
    indices: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    logical = indices.clone()
    expanded_offsets = offsets.reshape(indices.shape[0], 1, 1).expand_as(indices)
    valid = logical >= 0
    logical[valid] -= expanded_offsets[valid]
    return logical


def _asc_compare_params(
    case: dict[str, torch.Tensor], topk: int
) -> tuple:
    batch = int(case["used_k"].numel())
    q_lengths = torch.diff(case["cu"]).tolist()
    return (
        batch,
        max(q_lengths),
        int(case["used_k"].max()),
        int(case["q"].shape[0]),
        int(case["k"].shape[0] * 128),
        32,
        1,
        128,
        128,
        int(case["k"].shape[0]),
        torch.uint8,
        torch.float32,
        torch.uint8,
        torch.int32,
        case["cu"].tolist(),
        case["used_k"].tolist(),
        1,
        1,
        "TND",
        "PA_BBND",
        topk,
        0,
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (125, 129),
        (125, 129),
    )


def _check_sparse_values(
    actual_indices: torch.Tensor,
    actual_values: torch.Tensor,
    golden: dict[str, torch.Tensor],
) -> None:
    logical = _remove_output_offset(
        actual_indices, golden["_output_idx_offset"]
    )
    cu = golden["_cu_seqlens_q"].tolist()
    reference = torch.zeros_like(actual_values, dtype=torch.float32)
    for batch, (begin, end) in enumerate(zip(cu, cu[1:])):
        for row in range(begin, end):
            valid = logical[row, 0] >= 0
            reference[row, 0, valid] = golden["_topk_value"][
                batch, 0, row - begin, logical[row, 0, valid].long()
            ]
    actual_np = actual_values.to(torch.float32).numpy().reshape(-1)
    reference_np = reference.numpy().reshape(-1)
    assert result_compare_method.judge_value_by_isclose(
        actual_np, reference_np
    ), "sparse_values failed the local ASC result comparison"


def _run_and_compare(case: dict[str, torch.Tensor], topk: int) -> None:
    golden = quant_sparse_lightning_indexer_golden(
        case["q"], case["k"], case["w"],
        case["descale_q"], case["descale_k"],
        case["candidate"], case["candidate_length"],
        cu_seqlens_q=case["cu"],
        seqused_k=case["used_k"],
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=topk,
        candidate_block_size=8,
        return_value=True,
    )
    actual_outputs = quant_sparse_lightning_indexer(
        case["q"].npu(), case["k"].npu(), case["w"].npu(),
        case["descale_q"].npu(), case["descale_k"].npu(),
        case["candidate"].npu(), case["candidate_length"].npu(),
        topk, 1, 8,
        cu_seqlens_q=case["cu"].npu(),
        seqused_k=case["used_k"].npu(),
        block_table=case["block_table"].npu(),
        output_idx_offset=case["offset"].npu(),
        return_value=True,
    )
    torch.npu.synchronize()
    actual = {
        name: value.cpu()
        for name, value in zip(
            ("sparse_indices", "sparse_values"), actual_outputs
        )
    }

    expected_indices = _remove_output_offset(
        golden["sparse_indices"], golden["_output_idx_offset"]
    )
    actual_indices = _remove_output_offset(
        actual["sparse_indices"], golden["_output_idx_offset"]
    )
    result, fulfill_percent = result_compare_method.check_result(
        expected_indices,
        actual_indices,
        golden["_topk_value"],
        _asc_compare_params(case, topk),
    )
    assert result == "Pass", f"TopK compare failed: {fulfill_percent:.6f}%"
    _check_sparse_values(
        actual["sparse_indices"], actual["sparse_values"], golden
    )


@pytest.mark.npu
def test_qsli_shuffled_pa_candidate_scale_and_offset_matches_golden():
    _run_and_compare(_case(batch=2, s2=256, seed=221), topk=64)


@pytest.mark.npu
@pytest.mark.slow
def test_qsli_batch8_sequence8k_matches_golden():
    """Target acceptance shape: B=8, S2=8192, C=2048, valid C=1024."""
    _run_and_compare(_case(batch=8, s2=8192, seed=8192), topk=512)


@pytest.mark.npu
@pytest.mark.slow
def test_qsli_batch1_sequence128k_matches_golden():
    """Long-PA acceptance shape: B=1, S2=131072, C=2048, topk=512."""
    _run_and_compare(_case(batch=1, s2=128 * 1024, seed=131072), topk=512)


@pytest.mark.npu
def test_candidate_block_length_is_exact_prefix_count():
    case = _case(batch=1, s2=256, seed=9)
    case["candidate_length"][0, 0] = 17
    case["candidate"][0, 0, 17:] = 2**30
    _run_and_compare(case, topk=64)

"""Independent QSLI case generation, execution, selection, and result helpers."""

from __future__ import annotations

import re
from pathlib import Path

import torch

import result_compare_method
from qsli_parameter_normalization import normalize_qsli_params
from quant_sparse_lightning_indexer_golden import quant_sparse_lightning_indexer_golden
import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl  # noqa: F401,E402


quant_sparse_lightning_indexer = (
    torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer
)

PAGE_SIZE = 128
BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
N1 = 32
N2 = 1
PACKED_D = 64


def _make_dim0_view(tensor, padding):
    rows = tensor.shape[0]
    flat = tensor.reshape(rows, -1)
    storage = torch.zeros((rows, flat.shape[1] + padding), dtype=tensor.dtype)
    storage[:, :flat.shape[1]] = flat
    return storage[:, :flat.shape[1]].view(tensor.shape), storage


def generate_qsli_test_data(params):
    p = normalize_qsli_params(params)
    batch, q_len, s2 = p["batch_size"], p["query_length"], p["sequence_length"]
    if s2 < 1 or q_len < 1:
        raise ValueError("query_length and sequence_length must be positive")
    generator = torch.Generator(device="cpu").manual_seed(p["seed"])
    rows = batch * q_len
    pages = (s2 + PAGE_SIZE - 1) // PAGE_SIZE
    logical_blocks = (s2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    valid_blocks = min(CANDIDATE_CAPACITY, logical_blocks)
    requested = p["candidate_block_length"]
    valid_blocks = valid_blocks if requested is None else min(int(requested), valid_blocks)

    q = torch.randint(0, 256, (rows, N1, PACKED_D), generator=generator, dtype=torch.uint8)
    k = torch.randint(0, 256, (pages, PAGE_SIZE, N2, PACKED_D), generator=generator, dtype=torch.uint8)
    descale_k = torch.randint(125, 130, (pages, PAGE_SIZE, N2, 2, 2), generator=generator, dtype=torch.uint8)
    k_storage = scale_storage = fused_storage = None
    if p["storage_mode"] == "dim0_noncontiguous":
        key_size = PAGE_SIZE * N2 * PACKED_D
        scale_size = PAGE_SIZE * N2 * 4
        fused_storage = torch.empty((pages, key_size + scale_size), dtype=torch.uint8)
        fused_storage[:, :key_size] = k.reshape(pages, -1)
        fused_storage[:, key_size:] = descale_k.reshape(pages, -1)
        k = fused_storage[:, :key_size].view_as(k)
        descale_k = fused_storage[:, key_size:].view_as(descale_k)
    elif p["storage_mode"] == "key_dim0_noncontiguous":
        k, k_storage = _make_dim0_view(k, 64)
    elif p["storage_mode"] == "scale_dim0_noncontiguous":
        descale_k, scale_storage = _make_dim0_view(descale_k, 32)
    elif p["storage_mode"] != "contiguous":
        raise ValueError(f"unsupported storage_mode: {p['storage_mode']}")

    candidate = torch.full((rows, N2, CANDIDATE_CAPACITY), -1, dtype=torch.int32)
    for b in range(batch):
        order = torch.randperm(logical_blocks, generator=generator)[:valid_blocks]
        candidate[b*q_len:(b+1)*q_len, 0, :valid_blocks] = order.to(torch.int32)
    candidate_length = torch.full((rows, N2), valid_blocks, dtype=torch.int32)
    block_table = torch.stack([
        torch.randperm(pages, generator=generator).to(torch.int32)
        for _ in range(batch)
    ])
    cu = torch.arange(batch + 1, dtype=torch.int32) * q_len
    offset = None
    if p["output_idx_offset"]:
        offset = torch.arange(rows, dtype=torch.int32).reshape(rows, 1) * 3 + 37
    tensors = {
        "q": q,
        "k": k,
        "weights": torch.rand((rows, N1), generator=generator),
        "descale_q": torch.randint(125, 130, (rows, N1, 2, 2), generator=generator, dtype=torch.uint8),
        "descale_k": descale_k,
        "candidate": candidate,
        "candidate_length": candidate_length,
        "cu": cu,
        "used_q": torch.full((batch,), q_len, dtype=torch.int32),
        "used_k": torch.full((batch,), s2, dtype=torch.int32),
        "residual": None if p["cmp_ratio"] == 1 else torch.arange(batch, dtype=torch.int32) % p["cmp_ratio"],
        "block_table": block_table,
        "offset": offset,
        "_fused_storage": fused_storage,
        "_k_storage": k_storage,
        "_scale_storage": scale_storage,
    }
    golden = quant_sparse_lightning_indexer_golden(
        q, k, tensors["weights"], tensors["descale_q"], descale_k,
        candidate, candidate_length, cu_seqlens_q=cu, seqused_q=tensors["used_q"],
        seqused_k=tensors["used_k"], cmp_residual_k=tensors["residual"],
        block_table=block_table, output_idx_offset=offset, topk=p["topk"],
        candidate_block_size=BLOCK_SIZE, max_seqlen_q=q_len,
        mask_mode=p["mask_mode"], cmp_ratio=p["cmp_ratio"], return_value=True,
    )
    return {"params": p, "tensors": tensors, "cpu_result": golden}


def _to_npu_tensors(tensors):
    npu = {}
    ignored = {"k", "descale_k", "_fused_storage", "_k_storage", "_scale_storage"}
    for name, value in tensors.items():
        if name not in ignored:
            npu[name] = value.npu() if isinstance(value, torch.Tensor) else value
    pages = tensors["k"].shape[0]
    key_size, scale_size = PAGE_SIZE * N2 * PACKED_D, PAGE_SIZE * N2 * 4
    if tensors["_fused_storage"] is not None:
        storage = tensors["_fused_storage"].npu()
        npu["k"] = storage[:, :key_size].view(pages, PAGE_SIZE, N2, PACKED_D)
        npu["descale_k"] = storage[:, key_size:].view(pages, PAGE_SIZE, N2, 2, 2)
    else:
        if tensors["_k_storage"] is not None:
            storage = tensors["_k_storage"].npu()
            npu["k"] = storage[:, :key_size].view(pages, PAGE_SIZE, N2, PACKED_D)
        else:
            npu["k"] = tensors["k"].npu()
        if tensors["_scale_storage"] is not None:
            storage = tensors["_scale_storage"].npu()
            npu["descale_k"] = storage[:, :scale_size].view(pages, PAGE_SIZE, N2, 2, 2)
        else:
            npu["descale_k"] = tensors["descale_k"].npu()
    return npu


def run_qsli_case(case_data, device_id=0):
    import torch_npu
    torch_npu.npu.set_device(device_id)
    p, tensors = case_data["params"], case_data["tensors"]
    n = _to_npu_tensors(tensors)
    actual_outputs = quant_sparse_lightning_indexer(
        n["q"], n["k"], n["weights"], n["descale_q"], n["descale_k"],
        n["candidate"], n["candidate_length"], p["topk"], 1, BLOCK_SIZE,
        cu_seqlens_q=n["cu"],
        seqused_q=n["used_q"], seqused_k=n["used_k"],
        cmp_residual_k=n["residual"], block_table=n["block_table"],
        output_idx_offset=n["offset"], max_seqlen_q=p["query_length"],
        mask_mode=p["mask_mode"], cmp_ratio=p["cmp_ratio"], return_value=True,
    )
    torch.npu.synchronize()
    return {
        name: value.cpu()
        for name, value in zip(
            ("sparse_indices", "sparse_values"), actual_outputs
        )
    }


def _remove_offset(indices, offset):
    result = indices.clone()
    if offset is None:
        return result
    expanded = offset.reshape(indices.shape[0], 1, 1).expand_as(indices)
    valid = result >= 0
    result[valid] -= expanded[valid]
    return result


def compare_qsli_case(case_data, actual):
    p, golden = case_data["params"], case_data["cpu_result"]
    offset = golden["_output_idx_offset"]
    expected_idx = _remove_offset(golden["sparse_indices"], offset)
    actual_idx = _remove_offset(actual["sparse_indices"], offset)
    rows, s2 = actual_idx.shape[0], p["sequence_length"]
    batch = p["batch_size"]
    cu = case_data["tensors"]["cu"].tolist()
    used_k = case_data["tensors"]["used_k"].tolist()
    compare_params = (
        batch, p["query_length"], s2, rows, batch*s2, N1, N2, 128,
        PAGE_SIZE, (s2+PAGE_SIZE-1)//PAGE_SIZE, torch.uint8, torch.float32,
        torch.uint8, torch.int32, cu, used_k,
        1, 1, "TND", "PA_BBND", p["topk"], p["mask_mode"],
        (-1, 1), (-1, 1), (0, 1), (125, 129), (125, 129),
    )
    result, fulfill = result_compare_method.check_result(
        expected_idx, actual_idx, golden["_topk_value"], compare_params
    )
    if result != "Pass":
        raise AssertionError(f"index result={result}, fulfill_percent={fulfill}")
    reference = torch.zeros_like(actual["sparse_values"], dtype=torch.float32)
    golden_cu = golden["_cu_seqlens_q"].tolist()
    for batch, (begin, end) in enumerate(zip(golden_cu, golden_cu[1:])):
        for row in range(begin, end):
            valid = actual_idx[row, 0] >= 0
            reference[row, 0, valid] = golden["_topk_value"][
                batch, 0, row-begin, actual_idx[row, 0, valid].long()
            ]
    value_pass = result_compare_method.judge_value_by_isclose(
        actual["sparse_values"].float().numpy().reshape(-1),
        reference.numpy().reshape(-1),
    )
    if not value_pass:
        raise AssertionError("sparse_values failed result_compare_method")
    return "Pass", fulfill


class QsliCaseSelector:
    @staticmethod
    def natural_key(path):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).name)]

    @classmethod
    def resolve(cls, pt_dir, explicit_files="", case_names="", case_indexes=""):
        files = [Path(x) for x in explicit_files.split(",") if x] if explicit_files else sorted(Path(pt_dir).glob("*.pt"), key=cls.natural_key)
        if case_names:
            wanted = [x.strip() for x in case_names.split(",") if x.strip()]
            by_name = {x.stem: x for x in files}
            files = [by_name[x] for x in wanted]
        if case_indexes:
            indexes = [int(x.strip()) for x in case_indexes.split(",") if x.strip()]
            files = [files[x-1] for x in indexes]
        if not files:
            raise ValueError(f"no PT cases found in: {pt_dir}")
        return files

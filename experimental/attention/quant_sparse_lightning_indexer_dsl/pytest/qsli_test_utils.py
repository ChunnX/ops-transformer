"""Independent QSLI case generation, execution, selection, and result helpers."""

from __future__ import annotations

import math
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
FP4_E2M1_VALUES = torch.tensor(
    (
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ),
    dtype=torch.float32,
)
E8M0_ONE_VALUE = 127


def _range_pair(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid {name}: {value!r}")
    return low, high


def _make_mxfp4_bytes(logical_shape, data_range, generator):
    low, high = _range_pair(data_range, "FP4 E2M1 numeric range")
    if logical_shape[-1] % 2:
        raise ValueError(f"MXFP4 head_dim must be even, got {logical_shape[-1]}")
    valid_codes = torch.nonzero(
        (FP4_E2M1_VALUES >= low) & (FP4_E2M1_VALUES <= high),
        as_tuple=False,
    ).flatten().to(torch.uint8)
    if valid_codes.numel() == 0:
        raise ValueError(
            f"FP4 E2M1 range contains no representable value: {data_range}"
        )
    raw = valid_codes[torch.randint(
        valid_codes.numel(), tuple(logical_shape), generator=generator,
        dtype=torch.long,
    )]
    raw_rows = raw.reshape(-1, logical_shape[-1])
    coverage_count = min(valid_codes.numel(), logical_shape[-1])
    coverage = torch.arange(coverage_count).unsqueeze(0) + torch.arange(
        raw_rows.shape[0]
    ).unsqueeze(1)
    raw_rows[:, :coverage_count] = valid_codes[coverage % valid_codes.numel()]
    return (raw[..., 0::2] | (raw[..., 1::2] << 4)).contiguous()


def _make_e8m0_bytes(shape, scale_range):
    low, high = _range_pair(scale_range, "E8M0 scale range")
    if low <= 0:
        raise ValueError(f"E8M0 scale range must be positive, got {scale_range}")
    min_code = max(0, math.ceil(math.log2(low)) + E8M0_ONE_VALUE)
    max_code = min(254, math.floor(math.log2(high)) + E8M0_ONE_VALUE)
    while min_code <= 254 and math.ldexp(1.0, min_code - E8M0_ONE_VALUE) < low:
        min_code += 1
    while max_code >= 0 and math.ldexp(1.0, max_code - E8M0_ONE_VALUE) > high:
        max_code -= 1
    if min_code > max_code:
        raise ValueError(
            f"E8M0 scale range contains no representable value: {scale_range}"
        )
    count = max_code - min_code + 1
    return (
        torch.arange(math.prod(shape), dtype=torch.int64)
        .remainder(count)
        .add(min_code)
        .reshape(shape)
        .to(torch.uint8)
    )


def _make_dim0_view(tensor, padding):
    rows = tensor.shape[0]
    flat = tensor.reshape(rows, -1)
    storage = torch.zeros((rows, flat.shape[1] + padding), dtype=tensor.dtype)
    storage[:, :flat.shape[1]] = flat
    return storage[:, :flat.shape[1]].view(tensor.shape), storage


def _int_tensor(value, default):
    source = default if value is None else value
    return torch.tensor(source, dtype=torch.int32)


def _make_candidate_inputs(p, rows, logical_blocks, generator, cu):
    explicit_indices = p["candidate_block_indices"]
    explicit_lengths = p["candidate_block_length"]
    candidate = torch.full(
        (rows, N2, CANDIDATE_CAPACITY), -1, dtype=torch.int32
    )
    auto_indices = explicit_indices is None or (
        isinstance(explicit_indices, str) and explicit_indices.upper() == "AUTO"
    )
    if auto_indices:
        valid_blocks = min(CANDIDATE_CAPACITY, logical_blocks)
        valid_blocks = (
            valid_blocks
            if explicit_lengths is None
            else min(int(explicit_lengths), valid_blocks)
        )
        for batch_index in range(p["batch_size"]):
            order = torch.randperm(logical_blocks, generator=generator)[:valid_blocks]
            begin = int(cu[batch_index])
            end = int(cu[batch_index + 1])
            candidate[begin:end, 0, :valid_blocks] = order.to(torch.int32)
        lengths = torch.full((rows, N2), valid_blocks, dtype=torch.int32)
        return candidate, lengths

    source = torch.tensor(explicit_indices, dtype=torch.int32)
    if source.ndim != 3 or tuple(source.shape[:2]) != (rows, N2):
        raise ValueError(
            "candidate_block_indices from Excel must have shape "
            f"(T1,N2,L), got {tuple(source.shape)}"
        )
    if source.shape[2] > CANDIDATE_CAPACITY:
        raise ValueError(
            f"candidate_block_indices L must be <= {CANDIDATE_CAPACITY}"
        )
    if explicit_lengths is None:
        raise ValueError(
            "candidate_block_length is required when candidate_block_indices is explicit"
        )
    lengths = torch.tensor(explicit_lengths, dtype=torch.int32)
    if tuple(lengths.shape) != (rows, N2):
        raise ValueError(
            "candidate_block_length from Excel must have shape "
            f"(T1,N2), got {tuple(lengths.shape)}"
        )
    candidate[:, :, :source.shape[2]] = source
    for row_index in range(rows):
        for head_index in range(N2):
            length = int(lengths[row_index, head_index])
            if length < 0 or length > source.shape[2]:
                raise ValueError(
                    "candidate_block_length must be within the explicit candidate width"
                )
            valid = source[row_index, head_index, :length]
            if torch.any(valid < 0) or torch.any(valid >= logical_blocks):
                raise ValueError(
                    "candidate_block_indices contains an out-of-range logical block"
                )
    return candidate, lengths


def generate_qsli_test_data(params):
    p = normalize_qsli_params(params)
    batch, q_len, s2 = p["batch_size"], p["query_length"], p["sequence_length"]
    if s2 < 1 or q_len < 1:
        raise ValueError("query_length and sequence_length must be positive")
    if p["qk_dtype"] not in ("FLOAT4_E2M1", "FLOAT4_E2M1FN_X2", "torch.float4_e2m1fn_x2"):
        raise ValueError(f"QSLI requires MXFP4 qk_dtype, got {p['qk_dtype']}")
    if p["dequant_dtype"] not in ("FLOAT8_E8M0", "FLOAT8_E8M0FNU", "torch.float8_e8m0fnu"):
        raise ValueError(f"QSLI requires E8M0 dequant_dtype, got {p['dequant_dtype']}")
    if p["weight_dtype"] not in ("FP32", "FLOAT32", "torch.float32"):
        raise ValueError(f"QSLI requires FP32 weight_dtype, got {p['weight_dtype']}")
    if p["actual_seq_dtype"] not in ("INT32", "torch.int32"):
        raise ValueError(f"QSLI requires INT32 actual_seq_dtype, got {p['actual_seq_dtype']}")
    if p["quant_mode"] != 1:
        raise ValueError("QSLI DSL public contract requires quant_mode=1")
    if p["candidate_block_size"] != BLOCK_SIZE:
        raise ValueError(f"QSLI candidate_block_size must be {BLOCK_SIZE}")
    if p["layout_q"] != "TND" or p["layout_kv"] != "PA_BBND":
        raise ValueError("QSLI requires layout_q=TND and layout_kv=PA_BBND")
    if p["q_t_size"] < 1:
        raise ValueError("q_t_size must be positive")
    if p["k_t_size"] != p["sequence_length"]:
        raise ValueError("k_t_size must equal k_seq")
    if (p["q_head_num"], p["k_head_num"], p["head_dim"]) != (N1, N2, 128):
        raise ValueError("QSLI DSL requires q_head_num=32, k_head_num=1, head_dim=128")
    if p["block_size"] != PAGE_SIZE:
        raise ValueError("QSLI DSL requires block_size=128")
    generator = torch.Generator(device="cpu").manual_seed(p["seed"])
    rows = p["q_t_size"]
    if p["cu_seqlens_q"] is None and rows != batch * q_len:
        raise ValueError(
            "variable-length TND requires explicit cu_seqlens_q when "
            "q_t_size != batch_size * q_seq"
        )
    default_cu = [index * q_len for index in range(batch + 1)]
    cu = _int_tensor(p["cu_seqlens_q"], default_cu)
    if tuple(cu.shape) != (batch + 1,) or int(cu[0]) != 0 or int(cu[-1]) != rows:
        raise ValueError("cu_seqlens_q must have shape (B+1), start at 0, and end at q_t_size")
    q_lengths = (cu[1:] - cu[:-1]).tolist()
    if any(length < 0 or length > q_len for length in q_lengths):
        raise ValueError("each TND query length must be within [0, q_seq]")
    used_q = _int_tensor(p["seqused_q"], q_lengths)
    if tuple(used_q.shape) != (batch,) or any(
        int(value) < 0 or int(value) > int(q_lengths[index])
        for index, value in enumerate(used_q.tolist())
    ):
        raise ValueError("seqused_q must have shape (B,) and fit cu_seqlens_q")
    pages = (s2 + PAGE_SIZE - 1) // PAGE_SIZE
    if p["block_num"] != pages:
        raise ValueError(f"block_num must equal ceil(k_seq/128)={pages}")
    logical_blocks = (s2 + BLOCK_SIZE - 1) // BLOCK_SIZE

    q = _make_mxfp4_bytes(
        (rows, N1, PACKED_D * 2), p["query_datarange"], generator
    )
    k = _make_mxfp4_bytes(
        (pages, PAGE_SIZE, N2, PACKED_D * 2), p["key_datarange"], generator
    )
    descale_q = _make_e8m0_bytes(
        (rows, N1, 2, 2), p["q_scale_datarange"]
    )
    descale_k = _make_e8m0_bytes(
        (pages, PAGE_SIZE, N2, 2, 2), p["k_scale_datarange"]
    )
    weight_low, weight_high = _range_pair(
        p["weights_datarange"], "weights_datarange"
    )
    weights = torch.rand((rows, N1), generator=generator, dtype=torch.float32)
    weights = weights * (weight_high - weight_low) + weight_low
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

    candidate, candidate_length = _make_candidate_inputs(
        p, rows, logical_blocks, generator, cu
    )
    auto_block_table = p["block_table"] is None or (
        isinstance(p["block_table"], str) and p["block_table"].upper() == "AUTO"
    )
    block_table = (
        torch.stack([
            torch.randperm(pages, generator=generator).to(torch.int32)
            for _ in range(batch)
        ])
        if auto_block_table
        else _int_tensor(p["block_table"], p["block_table"])
    )
    if tuple(block_table.shape) != (batch, pages):
        raise ValueError(
            f"block_table must have shape ({batch},{pages}), got {tuple(block_table.shape)}"
        )
    cu_k = _int_tensor(p["cu_seqlens_k"], list(range(0, batch * s2 + 1, s2)))
    if tuple(cu_k.shape) != (batch + 1,) or int(cu_k[0]) != 0:
        raise ValueError("cu_seqlens_k must have shape (B+1,) and start at zero")
    used_k = _int_tensor(p["seqused_k"], [s2] * batch)
    residual_value = p["cmp_residual_k"]
    if residual_value is None and p["cmp_ratio"] != 1:
        residual_value = [index % p["cmp_ratio"] for index in range(batch)]
    residual = (
        None if residual_value is None
        else _int_tensor(residual_value, residual_value)
    )
    offset_value = p["output_idx_offset"]
    if isinstance(offset_value, bool):
        offset = (
            torch.arange(rows, dtype=torch.int32).reshape(rows, 1) * 3 + 37
            if offset_value else None
        )
    else:
        offset = _int_tensor(offset_value, offset_value)
        if tuple(offset.shape) != (rows, N2):
            raise ValueError(
                f"output_idx_offset must have shape ({rows},{N2}), got {tuple(offset.shape)}"
            )
    tensors = {
        "q": q,
        "k": k,
        "weights": weights,
        "descale_q": descale_q,
        "descale_k": descale_k,
        "candidate": candidate,
        "candidate_length": candidate_length,
        "cu": cu,
        "used_q": used_q,
        "used_k": used_k,
        "residual": residual,
        "block_table": block_table,
        "offset": offset,
        "_fused_storage": fused_storage,
        "_k_storage": k_storage,
        "_scale_storage": scale_storage,
    }
    golden_kwargs = dict(
        cu_seqlens_q=cu, seqused_q=tensors["used_q"],
        seqused_k=tensors["used_k"], cmp_residual_k=tensors["residual"],
        block_table=block_table, output_idx_offset=offset, topk=p["topk"],
        candidate_block_size=p["candidate_block_size"],
        quant_mode=p["quant_mode"], max_seqlen_q=p["max_seqlen_q"],
        mask_mode=p["mask_mode"], cmp_ratio=p["cmp_ratio"],
        layout_q=p["layout_q"], layout_kv=p["layout_kv"],
        return_value=p["return_value"],
    )
    golden = quant_sparse_lightning_indexer_golden(
        q, k, tensors["weights"], tensors["descale_q"], descale_k,
        candidate, candidate_length, **golden_kwargs,
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
        n["candidate"], n["candidate_length"], p["topk"],
        p["quant_mode"], p["candidate_block_size"],
        cu_seqlens_q=n["cu"],
        seqused_q=n["used_q"], seqused_k=n["used_k"],
        cmp_residual_k=n["residual"], block_table=n["block_table"],
        output_idx_offset=n["offset"], max_seqlen_q=p["max_seqlen_q"],
        mask_mode=p["mask_mode"], cmp_ratio=p["cmp_ratio"],
        layout_q=p["layout_q"], layout_kv=p["layout_kv"],
        return_value=p["return_value"],
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
        batch, p["query_length"],
        s2, rows, batch*s2, N1, N2, 128,
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

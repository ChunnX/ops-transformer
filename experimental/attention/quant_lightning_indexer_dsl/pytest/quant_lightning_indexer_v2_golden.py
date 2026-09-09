"""Independent DSL Golden and NPU execution adapter using the ASC pytest API.

The numerical implementation remains in ``qli_test_utils.golden``.  This file
only presents the same single-case and PT-batch entry points used by mainline
QLI V2 pytest.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch_npu

from qli_test_utils import quant_lightning_indexer_golden
from qliv2_test_utils import PARAM_NAMES
import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl  # noqa: F401,E402


quant_lightning_indexer = (
    torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer
)


PAGE_SIZE = 128
N1 = 32
N2 = 1
PACKED_D = 64
FP4_PACK_NUM = 2
E8M0_ONE_VALUE = 127
FP4_E2M1_VALUES = torch.tensor(
    (
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ),
    dtype=torch.float32,
)


def _range_pair(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid {name}: {value!r}")
    return low, high


def _make_mxfp4_bytes(logical_shape, data_range, data_dtype, generator):
    if data_dtype != torch.float4_e2m1fn_x2:
        raise TypeError(
            "MXFP4 requires qk_dtype=FLOAT4_E2M1/torch.float4_e2m1fn_x2, "
            f"got {data_dtype}"
        )
    low, high = _range_pair(data_range, "FP4 E2M1 numeric range")
    if logical_shape[-1] % FP4_PACK_NUM:
        raise ValueError(f"MXFP4 head_dim must be even, got {logical_shape[-1]}")
    valid_codes = torch.nonzero(
        (FP4_E2M1_VALUES >= low) & (FP4_E2M1_VALUES <= high),
        as_tuple=False,
    ).flatten().to(torch.uint8)
    if valid_codes.numel() == 0:
        raise ValueError(
            f"FP4 E2M1 range contains no representable value: {data_range}"
        )
    raw_indices = torch.randint(
        valid_codes.numel(), tuple(logical_shape), generator=generator,
        dtype=torch.long,
    )
    raw = valid_codes[raw_indices]
    raw_rows = raw.reshape(-1, logical_shape[-1])
    coverage_count = min(valid_codes.numel(), logical_shape[-1])
    coverage = torch.arange(coverage_count).unsqueeze(0) + torch.arange(
        raw_rows.shape[0]
    ).unsqueeze(1)
    raw_rows[:, :coverage_count] = valid_codes[coverage % valid_codes.numel()]
    return (raw[..., 0::2] | (raw[..., 1::2] << 4)).contiguous()


def _make_e8m0_bytes(shape, scale_range, scale_dtype):
    if scale_dtype != torch.float8_e8m0fnu:
        raise TypeError(
            "MXFP4 requires dequant_dtype=FLOAT8_E8M0/torch.float8_e8m0fnu, "
            f"got {scale_dtype}"
        )
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


def _dsl_quant_mode(value):
    mode = int(value)
    if mode == 5:
        return 1
    if mode != 1:
        raise ValueError(f"MXFP4 quant_mode must be DSL 1 or legacy ASC 5, got {mode}")
    return mode


@dataclass
class QliV2InputData:
    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_q: torch.Tensor
    seqused_k: torch.Tensor
    cmp_residual_k: torch.Tensor | None
    block_table: torch.Tensor
    params: tuple


def _params_dict(test_data):
    if len(test_data) != len(PARAM_NAMES):
        raise ValueError(
            f"QLI V2 parameter count mismatch: got {len(test_data)}, "
            f"expected {len(PARAM_NAMES)}"
        )
    return dict(zip(PARAM_NAMES, test_data))


def _int_tensor(value, default):
    source = default if value is None else value
    return torch.tensor(source, dtype=torch.int32)


def _with_axis0_gap(tensor):
    strides = list(tensor.stride())
    strides[0] *= 2
    result = torch.empty_strided(tensor.shape, tuple(strides), dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def generate_qliv2_test_data(test_data):
    """Generate deterministic public-ABI tensors without invoking the NPU."""
    p = _params_dict(test_data)
    batch = int(p["batch_size"])
    s1 = int(p["q_seq"])
    s2 = int(p["k_seq"])
    total_q = int(p["q_t_size"])
    if total_q != batch * s1:
        raise ValueError("q_t_size must equal batch_size * q_seq for this TND paramset")
    if (int(p["q_head_num"]), int(p["k_head_num"]), int(p["head_dim"])) != (32, 1, 128):
        raise ValueError("fixed QLI axes must be N1=32, N2=1, D=128")
    if int(p["block_size"]) != PAGE_SIZE:
        raise ValueError("DSL transfer paramset fixes PA block_size at 128")
    if int(p["k_t_size"]) != s2:
        raise ValueError("k_t_size must equal k_seq for this transfer paramset")

    generator = torch.Generator(device="cpu").manual_seed(int(p["seed"]))
    pages = (s2 + PAGE_SIZE - 1) // PAGE_SIZE
    if int(p["block_num"]) != pages:
        raise ValueError(f"block_num must equal ceil(k_seq/128)={pages}")
    if p["weight_dtype"] != torch.float32:
        raise TypeError(f"weight_dtype must be FP32/torch.float32, got {p['weight_dtype']}")
    if p["actual_seq_dtype"] != torch.int32:
        raise TypeError(f"actual_seq_dtype must be INT32/torch.int32, got {p['actual_seq_dtype']}")
    _dsl_quant_mode(p["quant_mode"])
    q = _make_mxfp4_bytes(
        (total_q, N1, PACKED_D * FP4_PACK_NUM),
        p["query_datarange"], p["qk_dtype"], generator,
    )
    k = _make_mxfp4_bytes(
        (pages, PAGE_SIZE, N2, PACKED_D * FP4_PACK_NUM),
        p["key_datarange"], p["qk_dtype"], generator,
    )
    weight_low, weight_high = _range_pair(
        p["weights_datarange"], "weights_datarange"
    )
    w = torch.rand((total_q, N1), generator=generator, dtype=torch.float32)
    w = w * (weight_high - weight_low) + weight_low
    q_descale = _make_e8m0_bytes(
        (total_q, N1, 2, 2), p["q_scale_datarange"], p["dequant_dtype"]
    )
    k_descale = _make_e8m0_bytes(
        (pages, PAGE_SIZE, N2, 2, 2),
        p["k_scale_datarange"], p["dequant_dtype"],
    )

    storage_layout = str(p["storage_layout"])
    if storage_layout in ("key_axis0_noncontiguous", "axis0_noncontiguous"):
        k = _with_axis0_gap(k)
    if storage_layout in ("scale_axis0_noncontiguous", "axis0_noncontiguous"):
        k_descale = _with_axis0_gap(k_descale)

    default_cu = list(range(0, total_q + 1, s1))
    cu = _int_tensor(p["cu_seqlens_q"], default_cu)
    used_q = _int_tensor(p["seqused_q"], [s1] * batch)
    tails = (0, 1, 7, 9, 31, 63, 95, 127)
    default_used_k = [max(1, s2 - tails[index % len(tails)]) for index in range(batch)]
    used_k = _int_tensor(p["seqused_k"], default_used_k)
    ratio = int(p["cmp_ratio"])
    residual = p["cmp_residual_k"]
    if residual is None and ratio != 1:
        residual = [index % ratio for index in range(batch)]
    residual_tensor = None if residual is None else _int_tensor(residual, residual)

    page_ids = torch.arange(pages, dtype=torch.int32)
    block_table = torch.stack(
        [torch.roll(page_ids, shifts=index) for index in range(batch)]
    )
    return QliV2InputData(
        q, k, w, q_descale, k_descale, cu, used_q, used_k,
        residual_tensor, block_table, tuple(test_data),
    )


def _operator_kwargs(p, data):
    return {
        "cu_seqlens_q": data.cu_seqlens_q,
        "seqused_q": data.seqused_q,
        "seqused_k": data.seqused_k,
        "cmp_residual_k": data.cmp_residual_k,
        "block_table": data.block_table,
        "topk": int(p["sparse_count"]),
        "quant_mode": _dsl_quant_mode(p["quant_mode"]),
        "max_seqlen_q": int(p["max_seqlen_q"]),
        "mask_mode": int(p["sparse_mode"]),
        "cmp_ratio": int(p["cmp_ratio"]),
        "layout_q": str(p["layout_query"]),
        "layout_kv": str(p["layout_key"]),
        "return_value": bool(p["return_value"]),
        "candidate_topk_blocks": int(p["candidate_topk_blocks"]),
        "candidate_block_size": int(p["candidate_block_size"]),
    }


def _to_npu(value):
    return value.npu() if isinstance(value, torch.Tensor) else value


def _to_npu_preserving_strides(value):
    if value.is_contiguous():
        return value.npu()
    result = torch.empty_strided(
        value.shape, value.stride(), dtype=value.dtype, device="npu"
    )
    result.copy_(value)
    return result


def run_qliv2_case_data(data, device_id=0):
    """Run the independent CPU Golden and one fused NPU kernel call."""
    torch_npu.npu.set_device(device_id)
    p = _params_dict(data.params)
    kwargs = _operator_kwargs(p, data)
    cpu_result = quant_lightning_indexer_golden(
        data.q, data.k, data.w, data.q_descale, data.k_descale, **kwargs
    )
    k_npu = _to_npu_preserving_strides(data.k)
    k_scale_npu = _to_npu_preserving_strides(data.k_descale)
    npu_kwargs = {name: _to_npu(value) for name, value in kwargs.items()}
    topk = npu_kwargs.pop("topk")
    quant_mode = npu_kwargs.pop("quant_mode")
    npu_outputs = quant_lightning_indexer(
        data.q.npu(), k_npu, data.w.npu(), data.q_descale.npu(),
        k_scale_npu, topk, quant_mode, **npu_kwargs,
    )
    torch.npu.synchronize()
    output_names = (
        "sparse_indices",
        "sparse_values",
        "candidate_block_indices",
        "candidate_block_length",
    )
    npu_result = {
        name: value.cpu() for name, value in zip(output_names, npu_outputs)
    }
    return cpu_result, npu_result


def qliv2_output_single(test_data):
    data = generate_qliv2_test_data(test_data)
    cpu_result, npu_result = run_qliv2_case_data(data)
    return (
        cpu_result,
        npu_result,
        cpu_result.get("_reference_scores"),
        cpu_result.get("sparse_values"),
        npu_result.get("sparse_values"),
    )


def generate_cpu_golden(input_data):
    p = _params_dict(input_data.params)
    return quant_lightning_indexer_golden(
        input_data.q, input_data.k, input_data.w,
        input_data.q_descale, input_data.k_descale,
        **_operator_kwargs(p, input_data),
    )


__all__ = [
    "QliV2InputData", "generate_cpu_golden", "generate_qliv2_test_data",
    "qliv2_output_single", "run_qliv2_case_data",
]

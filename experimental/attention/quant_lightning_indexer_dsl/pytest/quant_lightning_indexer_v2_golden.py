"""Independent DSL Golden and NPU execution adapter using the ASC pytest API.

The numerical implementation remains in ``qli_test_utils.golden``.  This file
only presents the same single-case and PT-batch entry points used by mainline
QLI V2 pytest.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    generator = torch.Generator(device="cpu").manual_seed(int(p["seed"]))
    pages = (s2 + PAGE_SIZE - 1) // PAGE_SIZE
    q = torch.randint(0, 256, (total_q, N1, PACKED_D), generator=generator, dtype=torch.uint8)
    k = torch.randint(0, 256, (pages, PAGE_SIZE, N2, PACKED_D), generator=generator, dtype=torch.uint8)
    w = torch.rand((total_q, N1), generator=generator, dtype=torch.float32)
    q_descale = torch.randint(125, 130, (total_q, N1, 2, 2), generator=generator, dtype=torch.uint8)
    k_descale = torch.randint(125, 130, (pages, PAGE_SIZE, N2, 2, 2), generator=generator, dtype=torch.uint8)

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
        "quant_mode": int(p["quant_mode"]),
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

#!/usr/bin/env python3
"""B-I MXFP4 KV 精度 case suite。

默认只运行 CPU record/reference 校验；在已部署本分支算子包的 A5 环境运行：

    python3 mxfp4_bi_precision_cases.py --run-npu

P0 覆盖：全 16 行 V0 UB、tail=3/成对 gather、BF16/FP16、BSND/TND/PA、
非零 INT8 storage_offset、乱序 sparse_indices、-1 终止，以及非同值 KV 的
QK -> softmax -> PV 精度比较。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import struct

from mxfp4_bi_input_example import (
    MX_GROUP_SIZE,
    NOPE_DIM,
    RECORD_BYTES,
    ROPE_DIM,
    SCALE_COUNT,
    decode_nope,
    e8m0_value,
    pack_bi_record,
)


@dataclass(frozen=True)
class PrecisionCase:
    name: str
    q_dtype: str
    layout_query: str
    layout_kv: str
    s2: int
    sparse_ids: tuple[int, ...]
    storage_offset_bytes: int


P0_CASES = (
    PrecisionCase(
        "bsnd_bf16_full16_offset32",
        "bf16",
        "BSND",
        "BSND",
        16,
        tuple(range(15, -1, -1)),
        32,
    ),
    PrecisionCase("bsnd_fp16_tail3_offset0", "fp16", "BSND", "BSND", 3, (2, 0, 1, -1), 0),
    PrecisionCase("bsnd_bf16_tail3_offset1", "bf16", "BSND", "BSND", 3, (2, 0, 1, -1), 1),
    PrecisionCase("tnd_bf16_tail3_offset32", "bf16", "TND", "TND", 3, (2, 0, 1, -1), 32),
    PrecisionCase("pa_bsnd_bf16_tail3_offset32", "bf16", "BSND", "PA_BSND", 3, (2, 0, 1, -1), 32),
)
TND_PREFIX_CASE_NAME = "tnd_bf16_batch2_prefix_offset32"


def float_to_bf16_bits(value: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    return (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16


def bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def pack_rope(values: list[float], q_dtype: str) -> bytes:
    if q_dtype == "bf16":
        return struct.pack("<64H", *(float_to_bf16_bits(value) for value in values))
    if q_dtype == "fp16":
        return struct.pack("<64e", *values)
    raise ValueError(f"unsupported q_dtype={q_dtype}")


def decode_rope(raw: bytes, q_dtype: str) -> list[float]:
    if len(raw) != ROPE_DIM * 2:
        raise ValueError(f"expected {ROPE_DIM * 2} RoPE bytes, got {len(raw)}")
    if q_dtype == "bf16":
        return [bf16_bits_to_float(bits) for bits in struct.unpack("<64H", raw)]
    if q_dtype == "fp16":
        return list(struct.unpack("<64e", raw))
    raise ValueError(f"unsupported q_dtype={q_dtype}")


def make_records(case: PrecisionCase, token_offset: int = 0) -> list[bytes]:
    """每 token 使用不同 FP4/scale/RoPE，迫使 softmax/PV 真正区分各条 KV。"""
    fp4_codes = tuple(range(16))
    records = []
    for token in range(case.s2):
        logical_token = token + token_offset
        # Adjacent D values deliberately differ, so each packed byte exercises low/high nibble ordering.
        nope_codes = [fp4_codes[(logical_token + d) % len(fp4_codes)] for d in range(NOPE_DIM)]
        scales = [126 + ((logical_token + group) % 4) for group in range(SCALE_COUNT)]
        rope_values = [0.03125 * (logical_token + 1) * ((d % 9) - 4) for d in range(ROPE_DIM)]
        records.append(pack_bi_record(nope_codes, pack_rope(rope_values, case.q_dtype), scales))
    return records


def decode_record(record: bytes, q_dtype: str) -> list[float]:
    return decode_nope(record) + decode_rope(record[256:384], q_dtype)


def records_to_int8(records: list[bytes]):
    import torch

    packed = b"".join(records)
    return torch.tensor([byte if byte < 128 else byte - 256 for byte in packed], dtype=torch.int8)


def build_cpu_inputs(case: PrecisionCase, records: list[bytes]):
    import torch

    q_dtype = torch.bfloat16 if case.q_dtype == "bf16" else torch.float16
    q_values = [0.015625 * (((d * 3) % 17) - 8) for d in range(576)]
    query = torch.tensor(q_values, dtype=q_dtype)
    query = query.reshape(1, 1, 1, 576) if case.layout_query == "BSND" else query.reshape(1, 1, 576)

    if case.layout_kv == "PA_BSND":
        # PA block_size must be 16-aligned. Logical records deliberately reside in physical block 1,
        # so block_table cannot be silently ignored as an identity mapping.
        padding = [pack_bi_record([0] * NOPE_DIM, bytes(ROPE_DIM * 2), [0] * SCALE_COUNT)] * (16 - case.s2)
        sentinel = [pack_bi_record([0xE] * NOPE_DIM, bytes(ROPE_DIM * 2), [127] * SCALE_COUNT)] * 16
        key = records_to_int8(sentinel + records + padding).reshape(2, 16, 1, RECORD_BYTES)
        block_table = torch.tensor([[1]], dtype=torch.int32)
    elif case.layout_kv == "BSND":
        key = records_to_int8(records).reshape(1, case.s2, 1, RECORD_BYTES)
        block_table = None
    else:  # TND
        key = records_to_int8(records).reshape(case.s2, 1, RECORD_BYTES)
        block_table = None

    sparse_shape = (1, 1, 1, len(case.sparse_ids)) if case.layout_query == "BSND" else (1, 1, len(case.sparse_ids))
    return {
        "query": query,
        "key": key,
        "value": key.clone(),  # MLA V0 reads key; the public host schema still requires value.
        "sparse_indices": torch.tensor(case.sparse_ids, dtype=torch.int32).reshape(sparse_shape),
        "key_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "value_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "actual_seq_lengths_query": torch.tensor([1], dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor([case.s2], dtype=torch.int32),
        "block_table": block_table,
    }


def selected_indices(case: PrecisionCase) -> list[int]:
    result = []
    for sparse_id in case.sparse_ids:
        if sparse_id == -1:
            break
        if not 0 <= sparse_id < case.s2:
            raise ValueError(f"{case.name}: sparse index {sparse_id} is outside [0, {case.s2})")
        result.append(sparse_id)
    return result


def cpu_reference(case: PrecisionCase, cpu_inputs, records: list[bytes]):
    """与当前 B=1/S1=1/N1=N2=1 case 对应的 dequant -> QK -> mask -> softmax -> PV reference。"""
    import torch

    q = cpu_inputs["query"].float().reshape(1, 576)
    decoded = torch.tensor([decode_record(record, case.q_dtype) for record in records], dtype=torch.float32)
    indices = selected_indices(case)
    k = decoded[indices, :]
    v = decoded[indices, :NOPE_DIM]
    scores = q @ k.T * (1.0 / math.sqrt(576.0))
    # sparse_mode=3: for S1=1 and actualS2=s2, every selected index < s2 is visible.
    probs = torch.softmax(scores, dim=-1)
    output = probs @ v
    return output.reshape(1, 1, 1, NOPE_DIM) if case.layout_query == "BSND" else output.reshape(1, 1, NOPE_DIM)


def move_inputs_to_npu(cpu_inputs, storage_offset_bytes: int):
    import torch

    if storage_offset_bytes < 0:
        raise ValueError("storage_offset_bytes must be non-negative")
    npu_inputs = {}
    for name, tensor in cpu_inputs.items():
        if tensor is None:
            npu_inputs[name] = None
        elif name not in {"key", "value"}:
            npu_inputs[name] = tensor.npu()
        else:
            # B-I whole record is INT8: this storage_offset is already a byte offset, never offset * 2.
            backing = torch.zeros(tensor.numel() + storage_offset_bytes, dtype=torch.int8).npu()
            backing[storage_offset_bytes:].copy_(tensor.reshape(-1).npu())
            view = backing[storage_offset_bytes:].reshape(tensor.shape)
            if view.storage_offset() != storage_offset_bytes:
                raise AssertionError(f"{name}: storage_offset={view.storage_offset()}, expected={storage_offset_bytes}")
            npu_inputs[name] = view
    return npu_inputs


def validate_encoding_boundaries() -> None:
    assert e8m0_value(0) == 2.0 ** -127
    assert e8m0_value(1) == 2.0 ** -126
    assert e8m0_value(253) == 2.0 ** 126
    assert e8m0_value(254) == 2.0 ** 127
    try:
        e8m0_value(255)
    except ValueError:
        pass
    else:
        raise AssertionError("E8M0 0xff must be rejected by the CPU record builder")

    boundary_scales = (0, 1, 126, 127, 128, 129, 253, 254, 120, 121, 122, 123, 124, 125, 130, 131)
    boundary_record = pack_bi_record([0x1] * NOPE_DIM, bytes(ROPE_DIM * 2), boundary_scales)
    decoded = decode_nope(boundary_record)
    for group, scale_byte in enumerate(boundary_scales):
        expected = 0.5 * e8m0_value(scale_byte)
        if decoded[group * MX_GROUP_SIZE] != expected or decoded[(group + 1) * MX_GROUP_SIZE - 1] != expected:
            raise AssertionError(f"E8M0 group {group} was not applied to its full D=32 range")


def run_cpu_case(case: PrecisionCase) -> None:
    records = make_records(case)
    cpu_inputs = build_cpu_inputs(case, records)
    expected = cpu_reference(case, cpu_inputs, records)
    if tuple(expected.shape) != ((1, 1, 1, NOPE_DIM) if case.layout_query == "BSND" else (1, 1, NOPE_DIM)):
        raise AssertionError(f"{case.name}: unexpected CPU output shape {tuple(expected.shape)}")
    if not expected.isfinite().all():
        raise AssertionError(f"{case.name}: CPU reference contains NaN/Inf")
    print(f"CPU passed: {case.name}, output_shape={tuple(expected.shape)}, sparse={selected_indices(case)}")


def run_npu_case(case: PrecisionCase) -> None:
    import torch
    import torch_npu

    records = make_records(case)
    cpu_inputs = build_cpu_inputs(case, records)
    expected = cpu_reference(case, cpu_inputs, records)
    inputs = move_inputs_to_npu(cpu_inputs, case.storage_offset_bytes)
    output = torch_npu.npu_kv_quant_sparse_flash_attention(
        inputs["query"], inputs["key"], inputs["value"], inputs["sparse_indices"], 1.0 / math.sqrt(576.0), 2, 2,
        key_dequant_scale=inputs["key_dequant_scale"],
        value_dequant_scale=inputs["value_dequant_scale"],
        block_table=inputs["block_table"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        sparse_block_size=1,
        layout_query=case.layout_query,
        layout_kv=case.layout_kv,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    torch.npu.synchronize()
    actual = output.cpu().float()
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{case.name}: output shape {tuple(actual.shape)}, expected {tuple(expected.shape)}")
    close = torch.isclose(actual, expected, rtol=2e-2, atol=5e-2)
    ratio = float(close.float().mean())
    max_abs_error = float((actual - expected).abs().max())
    if ratio < 0.99:
        raise AssertionError(
            f"{case.name}: precision failed: ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
            f"actual[:8]={actual.flatten()[:8].tolist()}, expected[:8]={expected.flatten()[:8].tolist()}"
        )
    print(
        f"NPU passed: {case.name}, ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
        f"key_offset={inputs['key'].storage_offset()}"
    )


def build_tnd_prefix_inputs():
    """TND B=2：第二个 batch 的 KV 起点必须使用 actual_seq_lengths_kv 前缀和。"""
    import torch

    batch0 = PrecisionCase("tnd_batch0", "bf16", "TND", "TND", 3, (2, 0, 1, -1), 32)
    batch1 = PrecisionCase("tnd_batch1", "bf16", "TND", "TND", 4, (3, 1, 0, -1), 32)
    records0 = make_records(batch0, token_offset=0)
    records1 = make_records(batch1, token_offset=23)
    q0 = torch.tensor([0.015625 * (((d * 3) % 17) - 8) for d in range(576)], dtype=torch.bfloat16)
    q1 = torch.tensor([0.015625 * (((d * 5) % 19) - 9) for d in range(576)], dtype=torch.bfloat16)
    inputs = {
        "query": torch.stack((q0, q1)).reshape(2, 1, 576),
        "key": records_to_int8(records0 + records1).reshape(7, 1, RECORD_BYTES),
        "sparse_indices": torch.tensor(((2, 0, 1, -1), (3, 1, 0, -1)), dtype=torch.int32).reshape(2, 1, 4),
        # TND actual lengths are cumulative: batch0=[0,3), batch1=[3,7).
        "actual_seq_lengths_query": torch.tensor((1, 2), dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor((3, 7), dtype=torch.int32),
        "key_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "value_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "block_table": None,
    }
    inputs["value"] = inputs["key"].clone()
    return inputs, ((records0, (2, 0, 1, -1)), (records1, (3, 1, 0, -1)))


def cpu_reference_tnd_prefix(inputs, batches):
    import torch

    outputs = []
    for batch_idx, (records, sparse_ids) in enumerate(batches):
        q = inputs["query"][batch_idx].float().reshape(1, 576)
        decoded = torch.tensor([decode_record(record, "bf16") for record in records], dtype=torch.float32)
        indices = []
        for sparse_id in sparse_ids:
            if sparse_id == -1:
                break
            indices.append(sparse_id)
        scores = q @ decoded[indices, :].T * (1.0 / math.sqrt(576.0))
        outputs.append(torch.softmax(scores, dim=-1) @ decoded[indices, :NOPE_DIM])
    return torch.cat(outputs, dim=0).reshape(2, 1, NOPE_DIM)


def run_tnd_prefix_case(run_npu: bool) -> None:
    inputs, batches = build_tnd_prefix_inputs()
    expected = cpu_reference_tnd_prefix(inputs, batches)
    if not expected.isfinite().all():
        raise AssertionError("TND prefix CPU reference contains NaN/Inf")
    print(f"CPU passed: {TND_PREFIX_CASE_NAME}, output_shape={tuple(expected.shape)}")
    if not run_npu:
        return

    import torch
    import torch_npu

    npu_inputs = move_inputs_to_npu(inputs, 32)
    output = torch_npu.npu_kv_quant_sparse_flash_attention(
        npu_inputs["query"], npu_inputs["key"], npu_inputs["value"], npu_inputs["sparse_indices"],
        1.0 / math.sqrt(576.0), 2, 2,
        key_dequant_scale=npu_inputs["key_dequant_scale"],
        value_dequant_scale=npu_inputs["value_dequant_scale"],
        actual_seq_lengths_query=npu_inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=npu_inputs["actual_seq_lengths_kv"],
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    torch.npu.synchronize()
    actual = output.cpu().float()
    close = torch.isclose(actual, expected, rtol=2e-2, atol=5e-2)
    ratio = float(close.float().mean())
    max_abs_error = float((actual - expected).abs().max())
    if ratio < 0.99:
        raise AssertionError(f"{TND_PREFIX_CASE_NAME}: ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}")
    print(f"NPU passed: {TND_PREFIX_CASE_NAME}, ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-npu", action="store_true", help="执行真实 NPU 算子并做精度比对")
    parser.add_argument("--case", choices=["all", *(case.name for case in P0_CASES), TND_PREFIX_CASE_NAME], default="all")
    args = parser.parse_args()
    validate_encoding_boundaries()
    selected = P0_CASES if args.case == "all" else tuple(case for case in P0_CASES if case.name == args.case)
    for case in selected:
        run_cpu_case(case)
        if args.run_npu:
            run_npu_case(case)
    if args.case in {"all", TND_PREFIX_CASE_NAME}:
        run_tnd_prefix_case(args.run_npu)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B-I MXFP4 KV 精度 case。

默认仅校验 400B record 的字节布局，不依赖 torch/NPU：

    python3 mxfp4_bi_input_example.py

在已安装本分支算子包的 A5 NPU 环境执行精度对比：

    python3 mxfp4_bi_input_example.py --run-npu

该 case 把 FP4 code 直接写入缓存，避免先引入量化器这个额外变量。16 条
KV record 完全相同，所以任意合法 sparse/causal softmax 权重的加权结果都应
等于同一条反量化后的 NoPE value；CPU reference 仍显式执行 QK、softmax、PV。
"""

from __future__ import annotations

import argparse
import math
import struct
from typing import Iterable


NOPE_DIM = 512
ROPE_DIM = 64
RECORD_BYTES = 400
FP4_BYTES = 256
ROPE_OFFSET = 256
SCALE_OFFSET = 384
SCALE_COUNT = 16
MX_GROUP_SIZE = 32


def fp4_e2m1_value(code: int) -> float:
    """解码一个 E2M1 code；bit3 是符号，bit[2:1] 是指数，bit0 是尾数。"""
    if not 0 <= code < 16:
        raise ValueError(f"FP4 code must be in [0, 15], got {code}")
    sign = -1.0 if (code & 0x8) else 1.0
    exponent = (code >> 1) & 0x3
    mantissa = code & 0x1
    if exponent == 0:
        return sign * (mantissa * 0.5)
    return sign * math.ldexp(1.0 + mantissa * 0.5, exponent - 1)


def e8m0_value(scale_byte: int) -> float:
    if not 0 <= scale_byte < 0xFF:
        raise ValueError(f"E8M0 scale must be in [0, 254], got {scale_byte}")
    return math.ldexp(1.0, scale_byte - 127)


def pack_bi_record(nope_codes: Iterable[int], rope_raw: bytes, scales: Iterable[int]) -> bytes:
    """按 B-I ABI 组装一个逻辑 KV token 的 400B record。"""
    nope_codes = list(nope_codes)
    scales = list(scales)
    if len(nope_codes) != NOPE_DIM:
        raise ValueError(f"need {NOPE_DIM} NoPE FP4 codes, got {len(nope_codes)}")
    if len(rope_raw) != ROPE_DIM * 2:
        raise ValueError(f"need {ROPE_DIM * 2} raw RoPE bytes, got {len(rope_raw)}")
    if len(scales) != SCALE_COUNT:
        raise ValueError(f"need {SCALE_COUNT} E8M0 scales, got {len(scales)}")

    record = bytearray(RECORD_BYTES)
    for d in range(0, NOPE_DIM, 2):
        low, high = nope_codes[d], nope_codes[d + 1]
        if not 0 <= low < 16 or not 0 <= high < 16:
            raise ValueError(f"invalid FP4 pair at D={d}: ({low}, {high})")
        # D[2i] occupies low nibble; D[2i+1] occupies high nibble.
        record[d // 2] = low | (high << 4)
    record[ROPE_OFFSET:SCALE_OFFSET] = rope_raw
    for group, scale in enumerate(scales):
        e8m0_value(scale)  # also rejects the 0xff NaN encoding
        record[SCALE_OFFSET + group] = scale
    return bytes(record)


def decode_nope(record: bytes) -> list[float]:
    """CPU shadow decoder used only to validate this example's input bytes."""
    if len(record) != RECORD_BYTES:
        raise ValueError(f"record length must be {RECORD_BYTES}")
    output = []
    for byte_index, packed in enumerate(record[:FP4_BYTES]):
        group = (2 * byte_index) // MX_GROUP_SIZE
        scale = e8m0_value(record[SCALE_OFFSET + group])
        output.append(fp4_e2m1_value(packed & 0xF) * scale)
        output.append(fp4_e2m1_value(packed >> 4) * scale)
    return output


def make_precision_records(s2: int = 16) -> list[bytes]:
    """构造 16 条相同 record，覆盖一次完整 V0 UB batch。"""
    if s2 != 16:
        raise ValueError("the smoke case intentionally uses exactly 16 KV records")
    # Raw BF16 RoPE is all-zero. The output drops the RoPE tail, but V0 must still copy these 128 bytes unchanged.
    rope_raw = bytes(ROPE_DIM * 2)
    # low/high nibbles differ; scale alternates between 1 and 2 every 32 NoPE elements.
    nope_codes = [0x2 if d % 2 == 0 else 0x6 for d in range(NOPE_DIM)]
    scales = [127 + (group % 2) for group in range(SCALE_COUNT)]
    record = pack_bi_record(nope_codes, rope_raw, scales)
    return [record] * s2


def validate_precision_records(records: list[bytes]) -> None:
    if len(records) != 16:
        raise AssertionError("smoke case must contain 16 records")
    first = records[0]
    if first[0] != 0x62:
        raise AssertionError(f"wrong nibble order: first packed byte is 0x{first[0]:02x}, expected 0x62")
    if first[ROPE_OFFSET:SCALE_OFFSET] != bytes(ROPE_DIM * 2):
        raise AssertionError("raw RoPE suffix changed while packing")
    decoded = decode_nope(first)
    if decoded[:2] != [1.0, 4.0]:
        raise AssertionError(f"unexpected first NoPE pair: {decoded[:2]}")
    if decoded[32:34] != [2.0, 8.0]:
        raise AssertionError(f"per-32 E8M0 grouping is wrong: {decoded[32:34]}")


def make_torch_inputs(records: list[bytes]):
    """返回可直接传给 torch_npu API 的最小 BSND/A5 B-I case。"""
    import torch

    packed = b"".join(records)
    signed_bytes = [byte if byte < 128 else byte - 256 for byte in packed]
    key = torch.tensor(signed_bytes, dtype=torch.int8).reshape(1, len(records), 1, RECORD_BYTES)
    value = torch.zeros((1, len(records), 1, NOPE_DIM), dtype=torch.int8)
    # Zero query makes every QK score equal, while PV still checks the dequantized NoPE values.
    query = torch.zeros((1, 1, 1, 576), dtype=torch.bfloat16)
    sparse_indices = torch.arange(len(records), dtype=torch.int32).reshape(1, 1, 1, len(records))
    actual_seq_q = torch.tensor([1], dtype=torch.int32)
    actual_seq_kv = torch.tensor([len(records)], dtype=torch.int32)
    # quant_scale_repo_mode=1 keeps these two inputs mandatory in the current public host contract.
    key_dequant_scale = torch.ones((1,), dtype=torch.float32)
    value_dequant_scale = torch.ones((1,), dtype=torch.float32)
    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse_indices,
        "actual_seq_lengths_query": actual_seq_q,
        "actual_seq_lengths_kv": actual_seq_kv,
        "key_dequant_scale": key_dequant_scale,
        "value_dequant_scale": value_dequant_scale,
    }


def cpu_reference(inputs, records: list[bytes]):
    """BSND B=1/S1=1/N1=N2=1 的 CPU reference：dequant -> QK -> softmax -> PV。"""
    import torch

    q = inputs["query"].float().reshape(1, 576)
    # This case deliberately keeps all RoPE values at zero. V for MLA output is NoPE D[0:512].
    k_row = torch.tensor(decode_nope(records[0]) + [0.0] * ROPE_DIM, dtype=torch.float32)
    v_row = k_row[:NOPE_DIM]
    k = k_row.repeat(len(records), 1)
    v = v_row.repeat(len(records), 1)
    scores = q @ k.T * (1.0 / math.sqrt(576.0))
    # sparse_mode=3 masks positions >= S2-S1+s1+1. For S2=16/S1=1/s1=0, all 0..15 remain valid.
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).reshape(1, 1, 1, NOPE_DIM)


def move_inputs_to_npu(inputs, storage_offset_bytes: int):
    """在 NPU 上从大 INT8 storage 切出 key/value view，保留非零 byte storage_offset。"""
    import torch

    if storage_offset_bytes < 0:
        raise ValueError("storage_offset_bytes must be non-negative")
    npu_inputs = {}
    for name, tensor in inputs.items():
        if name not in {"key", "value"}:
            npu_inputs[name] = tensor.npu()
            continue
        # B-I record is ACL_INT8: offset unit is one byte. Do not multiply it by two.
        storage = torch.zeros(tensor.numel() + storage_offset_bytes, dtype=torch.int8).npu()
        storage[storage_offset_bytes:].copy_(tensor.reshape(-1).npu())
        view = storage[storage_offset_bytes:].reshape(tensor.shape)
        if view.storage_offset() != storage_offset_bytes:
            raise AssertionError(
                f"{name} view storage_offset={view.storage_offset()}, expected {storage_offset_bytes}"
            )
        npu_inputs[name] = view
    return npu_inputs


def run_npu_precision_case(records: list[bytes], storage_offset_bytes: int) -> None:
    import torch
    import torch_npu

    cpu_inputs = make_torch_inputs(records)
    inputs = move_inputs_to_npu(cpu_inputs, storage_offset_bytes)
    expected = cpu_reference(cpu_inputs, records)
    output = torch_npu.npu_kv_quant_sparse_flash_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["sparse_indices"],
        1.0 / math.sqrt(576.0),
        2,
        2,
        key_dequant_scale=inputs["key_dequant_scale"],
        value_dequant_scale=inputs["value_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    torch.npu.synchronize()
    if output.shape != (1, 1, 1, NOPE_DIM):
        raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output.float()).all():
        raise AssertionError("NPU output contains NaN or Inf")
    actual = output.cpu().float()
    # BF16 output and the device softmax/reduction are allowed a small accumulated-rounding error.
    close = torch.isclose(actual, expected, rtol=2e-2, atol=5e-2)
    ratio = float(close.float().mean())
    max_abs_error = float((actual - expected).abs().max())
    if ratio < 0.99:
        raise AssertionError(
            f"precision failed: close_ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
            f"actual[:8]={actual.flatten()[:8].tolist()}, expected[:8]={expected.flatten()[:8].tolist()}"
        )
    print(
        f"precision passed: close_ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
        f"output shape={tuple(output.shape)}, dtype={output.dtype}, "
        f"key_storage_offset_bytes={inputs['key'].storage_offset()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-npu", action="store_true", help="在已部署本分支算子包的 NPU 环境运行 CPU/NPU 精度对比")
    parser.add_argument(
        "--storage-offset-bytes",
        type=int,
        default=32,
        help="从 NPU INT8 大 cache 切出 key/value view 的 byte offset；默认 32，用于验证非零 offset 路径",
    )
    args = parser.parse_args()
    records = make_precision_records()
    validate_precision_records(records)
    print("case construction passed: 16 x 400B records; first NoPE pair=[1.0, 4.0]; group-1 pair=[2.0, 8.0]")
    if args.run_npu:
        run_npu_precision_case(records, args.storage_offset_bytes)


if __name__ == "__main__":
    main()

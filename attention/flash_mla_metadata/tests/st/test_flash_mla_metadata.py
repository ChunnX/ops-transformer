# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Exercise input_layer/quant_mode on NPU with the paired FlashMla operators."""

import argparse
import importlib
import math

import torch
import torch_npu  # noqa: F401


LAYOUTS = ("BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD")


def query_layout(tensor, layout, lengths):
    layout = layout.split("_")[0]
    if layout == "BNSD":
        return tensor.contiguous()
    if layout == "TND":
        return torch.cat([row[:, :length].transpose(0, 1) for row, length in zip(tensor, lengths)]).contiguous()
    result = tensor.transpose(1, 2)
    if layout == "BSH":
        result = result.flatten(2)
    return result.contiguous()


def cache_layout(tensor, layout):
    if layout == "BNSD":
        return tensor.contiguous()
    if layout == "BSH":
        return tensor[:, 0].contiguous()
    blocks, heads, seq, dim = tensor.shape
    d0 = 32 // tensor.element_size()
    return tensor.reshape(blocks, heads, seq, dim // d0, d0).permute(0, 1, 3, 2, 4).contiguous()


def output_rows(output, layout, lengths, heads):
    if layout.startswith("TND"):
        result = output.transpose(0, 1) if layout.endswith("_NTD") else output
        return [row.transpose(0, 1) for row in result.split(lengths)]
    if layout.endswith("_NBSD"):
        result = output.permute(1, 0, 2, 3)
    elif layout == "BSH":
        result = output.reshape(output.shape[0], output.shape[1], heads, -1).transpose(1, 2)
    elif layout == "BSND":
        result = output.transpose(1, 2)
    else:
        result = output
    return [row[:, :length] for row, length in zip(result, lengths)]


def check_metadata(metadata):
    assert metadata.dtype == torch.int32 and tuple(metadata.shape) == (480,)
    assert metadata.is_contiguous() and metadata.storage_offset() == 0
    assert metadata.data_ptr() % 64 == 0
    # Reading the header here is test verification after producer execution.
    header = metadata[:7].cpu().tolist()
    assert header[:2] == [0x34414C4D, 1], header
    assert header[3] == 128 and header[5] == 1920, header


def run_nonquant(ops, device):
    passed = 0
    heads, q_size = 4, 2
    q_lengths = [1, 2]
    for layout in LAYOUTS:
        tnd = layout.startswith("TND")
        for paged in (False, True):
            kv_size = 128 if paged else 64
            kv_lengths = [96, 32] if paged else [48, 32]
            seq_q = torch.tensor([1, 3] if tnd else q_lengths, dtype=torch.int64, device=device)
            seq_kv = torch.tensor(kv_lengths if paged or not tnd else [48, 80], dtype=torch.int64, device=device)
            table = torch.tensor([[1], [0]], dtype=torch.int32, device=device) if paged else None
            metadata = ops.flash_mla_metadata(
                heads, 2, sum(q_lengths) if tnd else q_size,
                sum(kv_lengths) if tnd and not paged else kv_size,
                actual_seq_lengths_q=seq_q, actual_seq_lengths_kv=seq_kv,
                input_layer=layout, block_size=128 if paged else 0, quant_mode=0,
            )
            check_metadata(metadata)
            # Deliberately reuse the same metadata across dtypes and PA storage layouts.
            for dtype in (torch.float16, torch.bfloat16):
                torch.manual_seed(17)
                q = torch.randn(2, heads, q_size, 512, dtype=dtype) * 0.1
                k = torch.randn(2, 1, kv_size, 512, dtype=dtype) * 0.1
                v = torch.randn(2, 1, kv_size, 512, dtype=dtype)
                qr = torch.randn(2, heads, q_size, 64, dtype=dtype) * 0.1
                kr = torch.randn(2, 1, kv_size, 64, dtype=dtype) * 0.1
                query = query_layout(q, layout, q_lengths).to(device)
                query_rope = query_layout(qr, layout, q_lengths).to(device)
                scale = 1 / math.sqrt(576)
                for kv_layout in (("BSH", "BNSD", "NZ") if paged else (layout,)):
                    if paged:
                        key, value, key_rope = [cache_layout(x.flip(0), kv_layout).to(device) for x in (k, v, kr)]
                    else:
                        key, value, key_rope = [query_layout(x, layout, kv_lengths).to(device) for x in (k, v, kr)]
                    out, lse = ops.flash_mla(
                        query, [key], [value], query_rope, key_rope, metadata, heads,
                        actual_seq_lengths_q=seq_q, actual_seq_lengths_kv=seq_kv,
                        block_table=table, input_layout=layout, block_size=128 if paged else 0,
                        scale_value=scale, softmax_lse_flag=True,
                    )
                    actual_rows = output_rows(out.cpu().double(), layout, q_lengths, heads)
                    lse_rows = output_rows(lse.cpu().double(), "TND" if tnd else "BNSD", q_lengths, heads)
                    for b, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
                        scores = (q[b, :, :q_len].double() @ k[b, :, :kv_len].double().transpose(-1, -2)
                                  + qr[b, :, :q_len].double() @ kr[b, :, :kv_len].double().transpose(-1, -2)) * scale
                        expected = scores.softmax(-1) @ v[b, :, :kv_len].double()
                        torch.testing.assert_close(actual_rows[b], expected, rtol=4e-3, atol=4e-3)
                        torch.testing.assert_close(lse_rows[b], scores.logsumexp(-1, keepdim=True), rtol=4e-3, atol=4e-3)
                    passed += 1
                    print(f"PASS nonquant {layout} {dtype} KV={kv_layout} PA={paged}", flush=True)
    return passed


def run_quant(ops, device):
    passed = 0
    heads, block = 4, 128
    for layout in ("BSH", "BSND", "TND", "BSH_NBSD", "BSND_NBSD", "TND_NTD"):
        tnd = layout.startswith("TND")
        torch.manual_seed(23)
        q_lengths, kv_lengths = [1, 1], [96, 32]
        # Zero Q/RoPE gives uniform attention, with an exact CPU reference for
        # the INT8 path independent of its intermediate probability quantization.
        q = torch.zeros(2, heads, 1, 512, dtype=torch.int8)
        k = torch.randint(-8, 8, (2, 1, block, 512), dtype=torch.int8)
        v = torch.randint(-8, 8, (2, 1, block, 512), dtype=torch.int8)
        qr = torch.zeros(2, heads, 1, 64, dtype=torch.bfloat16)
        kr = torch.randn(2, 1, block, 64, dtype=torch.bfloat16) * 0.1
        query = query_layout(q, layout, q_lengths).to(device)
        query_rope = query_layout(qr, layout, q_lengths).to(device)
        key, value, key_rope = [cache_layout(x.flip(0), "NZ").to(device) for x in (k, v, kr)]
        seq_q_values = [1, 2] if tnd else q_lengths
        seq_q = torch.tensor(seq_q_values, dtype=torch.int64, device=device)
        seq_kv = torch.tensor(kv_lengths, dtype=torch.int64, device=device)
        table = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        q_scale = torch.full((2, heads) if tnd else (2, 1, heads), 0.03125, dtype=torch.float32, device=device)
        kv_scale = torch.full((1,), 0.0625, dtype=torch.float32, device=device)
        metadata = ops.flash_mla_metadata(
            heads, 2, 2 if tnd else 1, block, actual_seq_lengths_q=seq_q, actual_seq_lengths_kv=seq_kv,
            input_layer=layout, block_size=block, quant_mode=1,
        )
        check_metadata(metadata)
        shared = dict(block_table=table, block_size=block, input_layout=layout, num_key_value_heads=1,
                      scale_value=1 / math.sqrt(576), softmax_lse_flag=True,
                      query_rope=query_rope, key_rope=key_rope, dequant_scale_query=q_scale,
                      key_antiquant_scale=kv_scale, value_antiquant_scale=kv_scale, query_quant_mode=3)
        out, lse = ops.flash_mla(
            query, [key], [value], metadata=metadata, num_heads=heads,
            actual_seq_lengths_q=seq_q, actual_seq_lengths_kv=seq_kv, **shared,
        )
        actual_rows = output_rows(out.cpu().double(), layout, q_lengths, heads)
        lse_rows = output_rows(lse.cpu().double(), "TND" if tnd else "BNSD", q_lengths, heads)
        for b, length in enumerate(kv_lengths):
            expected = (v[b, :, :length].double().mean(-2, keepdim=True) * 0.0625).expand(heads, 1, 512)
            torch.testing.assert_close(actual_rows[b], expected, rtol=4e-3, atol=4e-3)
            torch.testing.assert_close(lse_rows[b], torch.full((heads, 1, 1), math.log(length), dtype=torch.float64),
                                       rtol=4e-3, atol=4e-3)
        passed += 1
        print(f"PASS INT8 {layout} against uniform-attention CPU reference", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="cann_ops_transformer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quant-mode", type=int, choices=(0, 1), default=None)
    args = parser.parse_args()
    ops = importlib.import_module(args.package)
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    passed = 0
    if args.quant_mode in (None, 0):
        passed += run_nonquant(ops, device)
    if args.quant_mode in (None, 1):
        passed += run_quant(ops, device)
    print(f"PASS: {passed} device cases", flush=True)


if __name__ == "__main__":
    main()

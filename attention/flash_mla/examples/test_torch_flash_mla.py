# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Run the paired torch APIs and compare FP16 dense attention with a CPU reference."""

import argparse
import importlib
import math

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--package", default="cann_ops_transformer")
    args = parser.parse_args()
    ops = importlib.import_module(args.package)
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    torch.manual_seed(7)
    batch, heads, q_size, kv_size = 2, 4, 1, 128
    scale = 1.0 / math.sqrt(512 + 64)
    q_cpu = torch.randn(batch, heads, q_size, 512, dtype=torch.float16) * 0.1
    k_cpu = torch.randn(batch, 1, kv_size, 512, dtype=torch.float16) * 0.1
    v_cpu = torch.randn(batch, 1, kv_size, 512, dtype=torch.float16)
    qr_cpu = torch.randn(batch, heads, q_size, 64, dtype=torch.float16) * 0.1
    kr_cpu = torch.randn(batch, 1, kv_size, 64, dtype=torch.float16) * 0.1
    query, key, value, query_rope, key_rope = [
        t.to(device) for t in (q_cpu, k_cpu, v_cpu, qr_cpu, kr_cpu)
    ]
    seq_q = torch.full((batch,), q_size, dtype=torch.int64, device=device)
    # A contiguous slice exercises the accepted nonzero seq storage offset.
    seq_kv_storage = torch.full((batch + 1,), kv_size, dtype=torch.int64, device=device)
    seq_kv = seq_kv_storage[1:]
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    for lengths in ((128, 96), (64, 32)):
        with torch.npu.stream(stream):
            for index, length in enumerate(lengths):
                seq_kv[index].fill_(length)
            metadata = ops.flash_mla_metadata(
                heads,
                batch,
                q_size,
                kv_size,
                actual_seq_lengths_q=seq_q,
                actual_seq_lengths_kv=seq_kv,
                input_layer="BNSD",
                quant_mode=0,
            )
            out, lse = ops.flash_mla(
                query,
                [key],
                [value],
                query_rope,
                key_rope,
                metadata,
                heads,
                actual_seq_lengths_q=seq_q,
                actual_seq_lengths_kv=seq_kv,
                input_layout="BNSD",
                scale_value=scale,
                softmax_lse_flag=True,
            )
        stream.synchronize()
        assert (
            metadata.device == device
            and metadata.dtype == torch.int32
            and metadata.shape == (480,)
        )
        assert metadata.storage_offset() == 0 and metadata.data_ptr() % 64 == 0
        actual, actual_lse = out.cpu(), lse.cpu()
        for index, length in enumerate(lengths):
            scores = (
                q_cpu[index].double()
                @ k_cpu[index, :, :length].double().transpose(-1, -2)
                + qr_cpu[index].double()
                @ kr_cpu[index, :, :length].double().transpose(-1, -2)
            ) * scale
            expected = scores.softmax(-1) @ v_cpu[index, :, :length].double()
            expected_lse = scores.logsumexp(-1, keepdim=True)
            torch.testing.assert_close(
                actual[index].double(), expected, rtol=4e-3, atol=4e-3
            )
            torch.testing.assert_close(
                actual_lse[index].double(), expected_lse, rtol=4e-3, atol=4e-3
            )
        assert torch.isfinite(actual).all() and torch.isfinite(actual_lse).all()
        print(f"lengths={lengths}: attention and LSE passed")


if __name__ == "__main__":
    main()

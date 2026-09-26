# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Final-package dispatch, mapping, short-cache rejection and repeat-call tests."""

import argparse
import json
import statistics
from pathlib import Path
import torch
import torch_npu
from aclnn_runner import AclnnRunner
from test_turbo_quant_attention_score import make_inputs, check


def patterned(bits, dtype, batch=1, tokens=9, heads=16):
    x = make_inputs(tokens, 129, bits, batch, heads, 2, dtype)
    n = x[2].shape[0]
    page = torch.arange(n)[:, None, None, None]
    pos = torch.arange(128)[None, :, None, None]
    head = torch.arange(2)[None, None, :, None]
    dim = torch.arange(128)[None, None, None, :]
    codes = (page * 3 + pos * 5 + head * 7 + dim) % (1 << bits)
    packed = torch.zeros(n, 128, 2, bits * 16, dtype=torch.int64)
    for d in range(128):
        byte, shift = divmod(d * bits, 8)
        packed[..., byte] |= codes[..., d] << shift
        if shift + bits > 8:
            packed[..., byte + 1] |= codes[..., d] >> (8 - shift)
    x[2] = (packed & 255).to(torch.uint8)
    x[3] = ((torch.arange(x[3].numel()).reshape(x[3].shape) * 37 + 19) % 256).byte()
    x[4] = (0.5 + page[..., 0] * 0.125 + pos[..., 0] / 256 + head[..., 0] / 16).to(
        dtype
    )
    x[5] = (0.25 + page[..., 0] / 16 + pos[..., 0] / 512 + head[..., 0] / 32).to(dtype)
    x[6] = torch.arange(n - 1, -1, -1, dtype=torch.int32).reshape(batch, 2)
    x[0].zero_()
    x[1].zero_()
    for t in range(tokens):
        for h in range(heads):
            x[0][t, h, (t * 17 + h * 11) % 128] = 1 + (t % 3) / 2
            x[1][t, h, (t * 7 + h * 3) % 128] = 0.5
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--perf", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20260926)
    torch.npu.set_device(0)
    run = AclnnRunner()
    results = []

    def save(r):
        results.append(r)
        print(json.dumps(r), flush=True)
        Path(a.output).write_text(json.dumps(results, indent=2))

    if a.perf:
        for tokens, length in [
            (1, 1),
            (1, 127),
            (1, 129),
            (1, 4096),
            (1, 32768),
            (1, 131072),
            (512, 4096),
            (512, 32768),
        ]:
            cpu = make_inputs(tokens, length)
            inp = [t.npu() for t in cpu]
            torch.npu.synchronize()
            for _ in range(5):
                out, _ = run(inp, 3, length)
                del out
            times = []
            for _ in range(20):
                out, timing = run(inp, 3, length)
                times.append(timing["device_us"])
                del out
            save(
                dict(
                    test="perf",
                    tokens=tokens,
                    length=length,
                    median_us=statistics.median(times),
                    p95_us=sorted(times)[18],
                    samples=times,
                )
            )
        return
    for bits in (2, 3, 4):
        for dtype in (torch.bfloat16, torch.float16):
            for lens_type in (torch.int32, torch.int64):
                for host in (False, True):
                    for batch, tokens, heads in [(1, 9, 16), (3, 3, 8)]:
                        cpu = patterned(bits, dtype, batch, tokens, heads)
                        cpu[7] = cpu[7].to(lens_type)
                        inp = [t.npu() for t in cpu]
                        torch.npu.synchronize()
                        first = None
                        for _ in range(3):
                            out, _ = run(inp, bits, 129, host_lengths=host)
                            acc = check(out, cpu, bits)
                            if first is None:
                                first = out.cpu()
                            else:
                                torch.testing.assert_close(
                                    out.cpu(), first, rtol=0, atol=0
                                )
                        # An independent mapping-sensitive reference must detect a wrong page table.
                        wrong = [*cpu]
                        wrong[6] = cpu[6].flip(1).contiguous()
                        try:
                            check(out, wrong, bits)
                        except AssertionError:
                            pass
                        else:
                            raise AssertionError(
                                "mapping test cannot detect swapped pages"
                            )
                        save(
                            dict(
                                test="mapping_dispatch_repeat",
                                bits=bits,
                                dtype=str(dtype),
                                length_dtype=str(lens_type),
                                host=host,
                                batch=batch,
                                **acc,
                            )
                        )
    for field in (3, 4, 5):
        for dtype in (torch.bfloat16, torch.float16):
            for host in (False, True):
                cpu = patterned(3, dtype)
                cpu[field] = cpu[field][:-1].contiguous()
                inp = [t.npu() for t in cpu]
                torch.npu.synchronize()
                try:
                    run(inp, 3, 129, host_lengths=host)
                except RuntimeError as e:
                    assert "GetWorkspaceSize failed" in str(e)
                    save(
                        dict(
                            test="short_cache",
                            field=field,
                            dtype=str(dtype),
                            host=host,
                            rejected=True,
                            error=str(e),
                        )
                    )
                else:
                    raise AssertionError("short cache accepted")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Run under msprof --aic-metrics=PipeUtilization with the selected operator package.

Keep metadata generation outside the repeated FlashAttn calls. The manifest maps
consecutive FlashAttn rows in op_summary.csv to cases; discard warmup rows before
comparing median Task Duration(us), aic_scalar_time(us), and aiv_scalar_time(us).
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph", action="store_true", help="Measure 64-call NPU Graph replay without PMU")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats must be >= 1")
    torch.npu.set_device(0)
    manifest = []
    for dtype in (torch.float16, torch.bfloat16):
        for d, dv in ((64, 64), (72, 72), (128, 128), (192, 128), (256, 256)):
            for sq, sk, nq in ((1, 257, 3), (17, 129, 3), (128, 513, 8), (2048, 4096, 8)):
                torch.manual_seed(417)
                q = torch.randn(1, sq, nq, d, dtype=dtype, device="npu")
                k = torch.randn(1, sk, 1, d, dtype=dtype, device="npu")
                v = torch.randn(1, sk, 1, dv, dtype=dtype, device="npu")
                options = dict(
                    max_seqlen_q=sq, max_seqlen_kv=sk, mask_mode=0,
                    win_left=-1, win_right=-1,
                    layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                )
                metadata = flash_attn_metadata(
                    batch_size=1, num_heads_q=nq, num_heads_kv=1,
                    head_dim=d, head_dim_v=dv, **options,
                )
                name = f"{dtype}_{d}_{dv}_{sq}_{sk}_{nq}"
                def call():
                    return flash_attn(
                        q, k, v, metadata=metadata, softmax_scale=d ** -0.5,
                        return_softmax_lse=True, **options,
                    )

                record = dict(name=name, dtype=str(dtype), d=d, dv=dv, sq=sq, sk=sk, nq=nq)
                if args.graph:
                    for _ in range(args.warmup):
                        output = call()
                    # Always get an eager result even when warmup=0.
                    if args.warmup == 0:
                        output = call()
                    torch.npu.synchronize()
                    expected = [tensor.cpu() for tensor in output]
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        for _ in range(64):
                            output = call()
                    for _ in range(3):
                        graph.replay()
                    torch.npu.synchronize()
                    samples = []
                    for _ in range(9):
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / 64)
                    assert all(torch.equal(t.cpu(), ref) for t, ref in zip(output, expected)), name
                    record.update(median_us=statistics.median(samples), samples_us=samples)
                    graph.reset()
                else:
                    for _ in range(args.warmup + args.repeats):
                        output = call()
                    torch.npu.synchronize()
                    record.update(warmup=args.warmup, repeats=args.repeats)
                manifest.append(record)
                print(name, flush=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

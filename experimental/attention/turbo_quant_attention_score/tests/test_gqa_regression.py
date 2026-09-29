# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Exercise all GQA heads and reject unsupported query dimensions on an installed package."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from aclnn_runner import AclnnRunner
from test_submission_regression import patterned
from test_turbo_quant_attention_score import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20260929)
    torch.npu.set_device(0)
    run = AclnnRunner()
    results = []

    def save(result):
        results.append(result)
        print(json.dumps(result), flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2))

    for ratio in (1, 2, 3, 4, 5, 8):
        for bits in (2, 3, 4):
            for dtype in (torch.bfloat16, torch.float16):
                for host in (False, True):
                    for batch, tokens in ((1, 7), (1, 8), (1, 9), (3, 3)):
                        cpu = patterned(bits, dtype, batch, tokens, 2 * ratio)
                        # Cover both tensor length dtypes; host API converts to int64.
                        if tokens % 2:
                            cpu[7] = cpu[7].long()
                        inputs = [t.npu() for t in cpu]
                        first = None
                        for repeat in range(3):
                            out, _ = run(
                                inputs,
                                bits,
                                129,
                                host_lengths=host,
                                output_fill=float("nan"),
                            )
                            actual = out.cpu()
                            assert not torch.isnan(actual).any(), "unwritten output"
                            accuracy = check(out, cpu, bits)
                            if first is None:
                                first = actual.clone()
                            else:
                                torch.testing.assert_close(
                                    actual, first, rtol=0, atol=0
                                )
                        save(
                            dict(
                                test="gqa_all_heads",
                                ratio=ratio,
                                bits=bits,
                                dtype=str(dtype),
                                host=host,
                                batch=batch,
                                tokens=tokens,
                                length_dtype=str(cpu[7].dtype),
                                repeats=3,
                                **accuracy,
                            )
                        )

    for dim in (64, 127, 129, 256):
        for field in (0, 1):
            for host in (False, True):
                cpu = patterned(3, torch.bfloat16, heads=6)
                shape = (*cpu[field].shape[:2], dim)
                cpu[field] = torch.ones(shape, dtype=cpu[field].dtype)
                inputs = [t.npu() for t in cpu]
                try:
                    run(inputs, 3, 129, host_lengths=host)
                except RuntimeError as exc:
                    assert "GetWorkspaceSize failed" in str(exc)
                    save(
                        dict(
                            test="invalid_query_dim",
                            dim=dim,
                            field=field,
                            host=host,
                            rejected=True,
                            error=str(exc),
                        )
                    )
                else:
                    raise AssertionError("unsupported query dimension accepted")
    # Equal invalid dimensions must also be rejected, not just shape mismatches.
    for dim in (64, 127, 129, 256):
        for host in (False, True):
            cpu = patterned(3, torch.bfloat16, heads=10)
            for field in (0, 1):
                cpu[field] = torch.ones(
                    (*cpu[field].shape[:2], dim), dtype=cpu[field].dtype
                )
            inputs = [t.npu() for t in cpu]
            try:
                run(inputs, 3, 129, host_lengths=host)
            except RuntimeError as exc:
                assert "GetWorkspaceSize failed" in str(exc)
                save(
                    dict(
                        test="invalid_both_query_dims",
                        dim=dim,
                        host=host,
                        rejected=True,
                        error=str(exc),
                    )
                )
            else:
                raise AssertionError("unsupported equal query dimensions accepted")


if __name__ == "__main__":
    main()

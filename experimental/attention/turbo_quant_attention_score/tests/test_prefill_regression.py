# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Installed-package regression for streamed prefill and data-dependent fallback."""

import argparse
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from aclnn_runner import AclnnRunner
from test_turbo_quant_attention_score import make_inputs, check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    torch.manual_seed(20261002)
    run = AclnnRunner()
    records = []

    def record(name, inputs, bits, length, host=False, validate=True):
        output, timing = run(
            [x.npu() for x in inputs],
            bits,
            length,
            host_lengths=host,
            output_fill=float("nan"),
        )
        rec = {"case": name, **timing}
        if validate:
            rec.update(check(output, inputs, bits))
        records.append(rec)
        Path(args.output).write_text(json.dumps(records, indent=2))
        print(json.dumps(rec), flush=True)
        return output.cpu()

    for dtype in [torch.bfloat16, torch.float16]:
        for bits in [2, 3, 4]:
            for groups in [1, 2, 3, 4, 5, 8]:
                for lens_dtype in [torch.int32, torch.int64]:
                    cpu = make_inputs(
                        128, 512, bits, heads=2 * groups, kv_heads=2, norm_dtype=dtype
                    )
                    cpu[-1] = cpu[-1].to(lens_dtype)
                    record(
                        f"cube_{dtype}_{bits}_g{groups}_{lens_dtype}", cpu, bits, 512
                    )
    cpu = make_inputs(256, 512, 3, heads=12, kv_heads=4)
    reference = record("base_repeat", cpu, 3, 512)
    record("host_lengths", cpu, 3, 512, host=True)
    for repeat in range(3):
        for kind in [
            "short_length",
            "empty",
            "bad_page",
            "short_norm",
            "large_query",
            "nan_query",
            "nan_norm",
        ]:
            changed = [x.clone() for x in cpu]
            if kind == "short_length":
                changed[-1][0] = 379
            elif kind == "empty":
                changed[-1][0] = 0
            elif kind == "bad_page":
                changed[-2][0, 1] = changed[2].shape[0] + 7
            elif kind == "short_norm":
                changed[4] *= 0.0001
            elif kind == "large_query":
                changed[0][0, 0, 0] = 1e10
            elif kind == "nan_query":
                changed[0][0, 0, 0] = float("nan")
            elif kind == "nan_norm":
                changed[4][int(changed[-2][0, 0]), 0, 0] = float("nan")
            special = kind in ["bad_page", "nan_query", "nan_norm"]
            actual = record(f"{kind}_{repeat}", changed, 3, 512, validate=not special)
            if special:
                mask = torch.zeros_like(actual, dtype=torch.bool)
                if kind == "bad_page":
                    mask[..., 128:256] = True
                    assert torch.isneginf(actual[mask]).all()
                elif kind == "nan_query":
                    mask[0, 0, :] = True
                    assert torch.isnan(actual[mask]).all()
                else:
                    mask[:, :3, 0] = True
                    assert torch.isnan(actual[mask]).all()
                torch.testing.assert_close(
                    actual[~mask], reference[~mask], atol=1e-5, rtol=1e-3
                )
            restored = record(f"restored_{kind}_{repeat}", cpu, 3, 512)
            torch.testing.assert_close(restored, reference, atol=0, rtol=0)
    # Shapes outside the bounded Cube dispatch remain on the original path.
    for tokens, length in [(127, 512), (129, 512), (160, 511), (33, 257)]:
        x = make_inputs(tokens, length, 3, heads=6, kv_heads=2)
        record(f"vector_shape_{tokens}_{length}", x, 3, length)
    print("PREFILL_REGRESSION_PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""mhc_post_backward 动态模式验证脚本（对齐 TTK：aclgraph 捕获-回放，目标 Ascend 950）。

TTK 图模式走 _compile_model_aclgraph 捕获-回放（GE 路径未适配），语义上的"动态"=
每个新 shape 重新捕获一张图 + 同图变数据回放。本脚本按此模式组织用例。

用法:
    python test_mhc_post_backward_dynamic.py                  # 950 机器全量用例
    python test_mhc_post_backward_dynamic.py --cpu-check      # 无卡本地 dynamo 模拟
    可选: --n 4|6|8 --d 128 --b 2 --dtype bf16|fp16

用例（950 机器）:
    1. eager vs CPU golden
    2. NPUGraph 捕获-回放 S=8/16/32 逐 shape 重捕获（h_res）
    3. NPUGraph 变数据回放（h_res）
    4. eager (h_res=None)
    5. NPUGraph 捕获-回放 + 变数据（h_res=None）
    6. torchair GE static 对照（h_res / h_res=None）
    7. torchair GE dynamic=True 对照（预期 868 known-limit，不计入失败）
"""

import argparse
import importlib
import sys
import time
import traceback

FAILURES = []
NOTES = []


def log(msg):
    print(msg, flush=True)


def run_case(name, fn, risky=False):
    log(f"---- {name} ----")
    try:
        passed = fn()
    except Exception:
        traceback.print_exc()
        passed = False
    if passed:
        log(f"PASS  {name}")
    else:
        log(f"FAIL{'(risk)' if risky else ''}  {name}")
        (NOTES if risky else FAILURES).append(name)
    return passed


def make_inputs(torch, b, s, n, d, dtype, has_hres, seed=0, device="npu"):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(shape, dt):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(dt)

    x = rnd((b, s, n, d), dtype)
    grad_y = rnd((b, s, n, d), dtype)
    h_out = rnd((b, s, d), dtype)
    h_post = rnd((b, s, n), torch.float32)
    h_res = (
        rnd((b, s, n, n), torch.float32)
        if has_hres
        else torch.empty(0, dtype=torch.float32)
    )
    return tuple(t.to(device) for t in (grad_y, x, h_res, h_out, h_post))


def golden(torch, grad_y, x, h_res, h_out, h_post):
    gy = grad_y.float().cpu()
    xs = x.float().cpu()
    ho = h_out.float().cpu()
    hp = h_post.float().cpu()
    if h_res is None or h_res.numel() == 0:
        gx = gy.clone()
    else:
        gx = torch.matmul(h_res.float().cpu(), gy)
    ghr = torch.matmul(xs, gy.transpose(-1, -2))
    gho = (gy * hp.unsqueeze(-1)).sum(dim=-2)
    ghp = (gy * ho.unsqueeze(-2)).sum(dim=-1)
    return gx, ghr, gho, ghp


def check_close(torch, outs, exp, rtol, atol, tag):
    names = ["grad_x", "grad_h_res", "grad_h_out", "grad_h_post"]
    ok = True
    for name, a, e in zip(names, outs, exp):
        a32 = a.float().cpu()
        if not torch.allclose(a32, e.float().cpu(), rtol=rtol, atol=atol):
            diff = (a32 - e.float().cpu()).abs()
            log(
                f"  [{tag}] {name} MISMATCH max_abs={diff.max().item():.4e} "
                f"mean_abs={diff.mean().item():.4e}"
            )
            ok = False
    return ok


def npu_graph_capture_replay(torch, op, inp, rtol, atol, tag):
    """warmup 旁路流 -> NPUGraph 捕获 -> replay -> 与 eager 参考比对。"""
    ref = op(*inp)
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            op(*inp)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outs = op(*inp)
    graph.replay()
    torch.npu.synchronize()
    return check_close(torch, outs, ref, rtol, atol, tag), graph, outs, ref


def cpu_check(args):
    """无卡本地：meta device + dynamo dynamic 模拟，校验 torch 前端动态能力。"""
    import torch
    import torch_npu  # noqa: F401

    importlib.import_module("cann_ops_transformer.ops.mhc_post_backward")
    op = torch.ops.cann_ops_transformer.mhc_post_backward

    def mk(S, has_hres):
        B, n, D = args.b, args.n, args.d
        dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        e = lambda *s, d=dt: torch.empty(*s, dtype=d, device="meta")
        hr = e(B, S, n, n, d=torch.float32) if has_hres else e(0, d=torch.float32)
        return e(B, S, n, D), e(B, S, n, D), hr, e(B, S, D), e(B, S, n, d=torch.float32)

    cm = torch.compile(op, backend="eager", dynamic=True)
    ok = True
    for has_hres in (True, False):
        for S in (8, 16, 32):
            try:
                outs = cm(*mk(S, has_hres))
                log(
                    f"  [cpu-check] h_res={has_hres} S={S} -> "
                    f"{[tuple(o.shape) for o in outs]}"
                )
            except Exception as e:
                log(
                    f"  [cpu-check] h_res={has_hres} S={S} FAILED: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                ok = False
    log("PASS  cpu-check dynamo dynamic 前端模拟" if ok else "FAIL  cpu-check")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, choices=[4, 6, 8])
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument(
        "--cpu-check",
        action="store_true",
        help="无卡本地 dynamo 模拟，不需要 NPU/torchair",
    )
    args = parser.parse_args()

    if args.cpu_check:
        return cpu_check(args)

    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        log("[ENV] torch_npu 未安装")
        return 2
    if not torch.npu.is_available():
        log("[ENV] 无可用 NPU 设备（可加 --cpu-check 本地模拟）")
        return 2
    log(f"[ENV] torch={torch.__version__} device={torch.npu.get_device_name(0)}")
    TORCHAIR_OK = True
    try:
        importlib.import_module(
            "cann_ops_transformer.ops.graph_convert_mhc_post_backward"
        )
    except ImportError as e:
        TORCHAIR_OK = False
        log(f"[ENV][WARN] torchair/graph_convert 不可用: {e}，跳过 GE 对照用例")
    importlib.import_module("cann_ops_transformer.ops.mhc_post_backward")
    op = torch.ops.cann_ops_transformer.mhc_post_backward

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    n, d, b, rt, at = args.n, args.d, args.b, args.rtol, args.atol

    def eager_case(has_hres):
        inp = make_inputs(torch, b, 8, n, d, dtype, has_hres)
        exp = golden(torch, *inp)
        outs = op(*inp)
        return check_close(torch, outs, exp, rt, at, f"eager h_res={has_hres}")

    def npu_graph_case(has_hres):
        ok = True
        for i, s in enumerate((8, 16, 32)):
            inp = make_inputs(torch, b, s, n, d, dtype, has_hres, seed=i)
            t0 = time.time()
            passed, graph, outs, _ = npu_graph_capture_replay(
                torch, op, inp, rt, at, f"graph S={s} h_res={has_hres}"
            )
            log(
                f"  [NPUGraph] S={s} 捕获+回放 {time.time() - t0:.1f}s "
                f"-> {'OK' if passed else 'MISMATCH'}"
            )
            ok &= passed
        return ok

    def npu_graph_replay_newdata_case(has_hres):
        inp = make_inputs(torch, b, 16, n, d, dtype, has_hres, seed=7)
        _, graph, outs, _ = npu_graph_capture_replay(
            torch, op, inp, rt, at, "replay-init"
        )
        new = make_inputs(torch, b, 16, n, d, dtype, has_hres, seed=8)
        for old, fresh in zip(inp, new):
            old.copy_(fresh)
        graph.replay()
        torch.npu.synchronize()
        ref = op(*inp)
        return check_close(torch, outs, ref, rt, at, f"replay-newdata h_res={has_hres}")

    def ge_static_case(has_hres):
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        cfg = CompilerConfig()
        cfg.mode = "reduce-overhead"
        cm = torch.compile(
            op,
            backend=torchair.get_npu_backend(compiler_config=cfg),
            fullgraph=True,
            dynamic=False,
        )
        inp = make_inputs(torch, b, 8, n, d, dtype, has_hres)
        exp = golden(torch, *inp)
        outs = cm(*inp)
        return check_close(torch, outs, exp, rt, at, f"GE-static h_res={has_hres}")

    def ge_dynamic_control(has_hres):
        import torchair

        try:
            cm = torch.compile(
                op, backend=torchair.get_npu_backend(), fullgraph=True, dynamic=True
            )
            inp = make_inputs(torch, b, 8, n, d, dtype, has_hres)
            outs = cm(*inp)
            inp16 = make_inputs(torch, b, 16, n, d, dtype, has_hres, seed=1)
            outs16 = cm(*inp16)
            ok = check_close(torch, outs16, golden(torch, *inp16), rt, at, "GE-dyn")
            log("  [GE-dynamic] 意外通过（该 torchair 版本 GE 路径支持动态）")
            return ok
        except Exception as e:
            NOTES.append(
                f"GE dynamic 对照失败(known-limit): "
                f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            )
            log(f"  [GE-dynamic] KNOWN-LIMIT 复现: {str(e).splitlines()[0][:200]}")
            return None

    run_case("1. eager vs golden (h_res)", lambda: eager_case(True))
    run_case(
        "2. NPUGraph 逐 shape 重捕获 S=8/16/32 (h_res)", lambda: npu_graph_case(True)
    )
    run_case(
        "3. NPUGraph 变数据回放 (h_res)", lambda: npu_graph_replay_newdata_case(True)
    )
    run_case("4. eager (h_res=None)", lambda: eager_case(False))
    run_case(
        "5. NPUGraph 捕获-回放+变数据 (h_res=None)",
        lambda: npu_graph_case(False) and npu_graph_replay_newdata_case(False),
    )
    if TORCHAIR_OK:
        run_case(
            "6. torchair GE static reduce-overhead (h_res)",
            lambda: ge_static_case(True),
            risky=True,
        )
        run_case(
            "6b. torchair GE static reduce-overhead (h_res=None)",
            lambda: ge_static_case(False),
            risky=True,
        )
        ge_dynamic_control(True)
        ge_dynamic_control(False)

    log("\n==== summary ====")
    log(f"failed: {FAILURES or '无'}")
    for note in NOTES:
        log(f"note: {note}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

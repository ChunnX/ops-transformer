#!/usr/bin/env python3
"""000108 极端值 golden 离线诊断/再生成工具。

用法 (yzp 容器内, 需先 source CANN 环境):
  python scripts/golden_diag_000108.py            # 诊断: 跑 CPU golden 并与 NPU dump 对比
  python scripts/golden_diag_000108.py --regen    # 再生成: 把 golden 结果写回 ttk_manual bins

流程: 手动加载 ttk_manual/<case>/input_*.bin → 按 CSV attributes 补写 slot 8-11 真值 →
调用 impl.golden.cpu_qfa_mxfp8 (纯 CPU) → 与 /tmp/npu_out0_f32.npy / /tmp/npu_out1_lse.npy
(由 ttk --dump-on-fail 抓取的 NPU 输出) 做 class/value 对比。
"""
import ast
import csv
import math
import os
import sys

import numpy as np
import torch

ASSETS = "/home/y00789489/ops-transformer-yzp/attention/quant_flash_attn/tests/assets"
CASE = (
    "Prefill_FP8_E4M3FN_FP8_E4M3FN_BF16_redline_MXFP8_5_64_4_15764_"
    "15764_128_128_TND_PA_LSE_Mask_000108"
)
CASE_DIR = f"/home/y00789489/ttk_manual/{CASE}/"
NPU_OUT0 = "/tmp/npu_out0_f32.npy"
NPU_OUT1 = "/tmp/npu_out1_lse.npy"

sys.path.insert(0, ASSETS)


def load_case():
    with open(os.path.join(ASSETS, "qfa_mxfp8_excel.csv")) as f:
        row = next(r for r in csv.DictReader(f) if CASE in r["testcase_name"])
    attrs = ast.literal_eval(row["attributes"])
    shapes = ast.literal_eval(row["tensor_view_shapes"])
    dtypes = ast.literal_eval(row["tensor_dtypes"])
    return attrs, shapes, dtypes


def load_tensors(attrs, shapes, dtypes):
    np_dtype = {
        "float8_e4m3fn": np.uint8,
        "float8_e8m0": np.uint8,
        "int32": np.int32,
        "float32": np.float32,
        "int8": np.int8,
        "bfloat16": np.uint16,
    }
    view_dtype = {"bfloat16": torch.bfloat16}
    tensors = []
    for i, (sh, dt) in enumerate(zip(shapes, dtypes)):
        a = np.fromfile(f"{CASE_DIR}input_{i}_{dt}.bin", dtype=np_dtype[dt])
        t = torch.from_numpy(a.copy())
        t = t.view(view_dtype.get(dt, t.dtype)).reshape(sh)
        tensors.append(t)
    # slot 8-11: 真值 in-place 写入 (复刻 inputs.py customize 语义)
    tensors[8].copy_(torch.tensor(attrs["cu_seqlens_q"], dtype=torch.int32))
    if tensors[9].numel() > 0 and attrs.get("cu_seqlens_kv"):
        tensors[9].copy_(torch.tensor(attrs["cu_seqlens_kv"], dtype=torch.int32))
    tensors[10].copy_(torch.tensor(attrs["seqused_q"], dtype=torch.int32))
    tensors[11].copy_(torch.tensor(attrs["seqused_kv"], dtype=torch.int32))
    return tensors


def match_pct(golden, npu, rtol, atol):
    g, n = golden.numpy(), npu
    nan_eq = np.isnan(g) & np.isnan(n)
    inf_eq = (g == n) & np.isinf(g)
    finite = np.isfinite(g) & np.isfinite(n)
    close = np.abs(g.astype(np.float64) - n.astype(np.float64)) <= (
        atol + rtol * np.abs(n.astype(np.float64))
    )
    ok = nan_eq | inf_eq | (finite & close)
    return ok.mean() * 100.0


def class_stats(a, name):
    a = np.asarray(a)
    print(
        f"  {name}: finite={np.isfinite(a).mean()*100:.3f}% "
        f"-inf={(a == -np.inf).mean()*100:.3f}% +inf={(a == np.inf).mean()*100:.3f}% "
        f"nan={np.isnan(a).mean()*100:.3f}%"
    )


def main():
    regen = "--regen" in sys.argv
    attrs, shapes, dtypes = load_case()
    tensors = load_tensors(attrs, shapes, dtypes)

    # CANN PYTHONPATH 的内置 tbe/impl 包会遮蔽 assets/impl 命名空间包,
    # 用 importlib 按文件路径加载 (依赖 quant_flash_attn_golden 等可从 ASSETS 解析)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "qfa_impl_golden", os.path.join(ASSETS, "impl", "golden.py")
    )
    impl_golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl_golden)
    cpu_qfa_mxfp8 = impl_golden.cpu_qfa_mxfp8

    kw = dict(
        batch_size=-1,
        N_q=attrs["N_q"],
        N_kv=attrs["N_kv"],
        D=attrs["D"],
        max_seqlen_q=attrs["max_seqlen_q"],
        max_seqlen_kv=attrs["max_seqlen_kv"],
        enable_pa=attrs["enable_pa"],
        kv_cache_layout=attrs["kv_cache_layout"],
        block_size=attrs["block_size"],
        mask_mode=attrs["mask_mode"],
        q_scale_layout=attrs["q_scale_layout"],
        quant_mode=attrs["quant_mode"],
        enable_lse=bool(attrs["enable_lse"]),
        softmax_scale=attrs["softmax_scale"],
        p_scale_value=attrs["p_scale_value"],
        enable_v_tail=bool(attrs.get("enable_v_tail", 0)),
    )
    result = cpu_qfa_mxfp8(*tensors, **kw)
    out_tnd, lse_nt = result[0], result[1]

    np.save("/tmp/golden_out0.npy", out_tnd.to(torch.float32).numpy())
    if lse_nt is not None:
        np.save("/tmp/golden_lse1.npy", lse_nt.to(torch.float32).numpy())

    print("[GOLDEN]")
    class_stats(out_tnd.numpy(), "out")
    if lse_nt is not None:
        class_stats(lse_nt.numpy(), "lse")

    if os.path.exists(NPU_OUT0):
        npu_out = np.load(NPU_OUT0)
        npu_lse = np.load(NPU_OUT1)
        print("[MATCH vs NPU]")
        print("  out PctRlt = %.4f%%" % match_pct(out_tnd, npu_out, 0.0078125, 1e-8))
        if lse_nt is not None:
            g, n = lse_nt.numpy(), npu_lse

            def _cls(a):
                c = np.zeros(a.shape, dtype=np.int8)
                c[a == np.inf] = 1
                c[a == -np.inf] = -1
                c[np.isnan(a)] = 2
                return c

            cg, cn = _cls(g), _cls(n)
            same = cg == cn
            print("  lse PctRlt = %.4f%%" % match_pct(lse_nt, npu_lse, 0.0078125, 1e-8))
            print("  lse class-match = %.4f%%" % (same.mean() * 100))
            lbl = {1: "+inf", -1: "-inf", 0: "finite", 2: "nan"}
            from collections import Counter

            flips = Counter(zip(cg[~same].tolist(), cn[~same].tolist()))
            for (a, b), cnt in flips.most_common():
                print("    flip golden=%-7s npu=%-7s : %d" % (lbl[a], lbl[b], cnt))
            # out class agreement
            go, no = out_tnd.numpy(), npu_out
            out_same = (
                (np.isnan(go) & np.isnan(no))
                | (np.isinf(go) & np.isinf(no) & (np.sign(go) == np.sign(no)))
                | (
                    np.isfinite(go)
                    & np.isfinite(no)
                    & (
                        np.abs(go.astype(np.float64) - no.astype(np.float64))
                        <= 1e-8 + 0.0078125 * np.abs(no.astype(np.float64))
                    )
                )
            )
            print("  out class-match = %.4f%%" % (out_same.mean() * 100))

    if regen:
        out_tnd.to(torch.float32).numpy().tofile(
            CASE_DIR + "golden_0_float32__shape_78820x64x128.bin"
        )
        if lse_nt is not None:
            lse_nt.to(torch.float32).numpy().tofile(
                CASE_DIR + "golden_1_float32__shape_64x78820.bin"
            )
        print("[REGEN] golden bins written to", CASE_DIR)


if __name__ == "__main__":
    main()

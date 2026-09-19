#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Channel-A compare pipeline self-test (no NPU / no example needed).

本环境（无 NPU）下验证「golden 可产出期望输出 + §2.3 断言可执行」：
1. 按 dump 契约合成输入（与参数化 example 约定的 value_spec 语义一致）；
2. 用 tests/assets/impl/golden.py::agm_v2_sim_golden 计算 golden；
3. 以 golden 本身作为 actual 生成 dump → 比对必须 PASS（正路径）；
4. 定向注入损坏（字节翻转 / ±0 符号位翻转 / NaN payload 变化 / 超容差
   扰动 / 确定性破坏）→ 对应断言必须 FAIL（断言有效性证明）。

Usage: python3 selftest_compare.py [--workdir DIR]
Exit code 0 = 全部自检通过。
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
import compare_dump  # noqa: E402

_TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
_BIT_DTYPES = {"fp16": torch.int16, "bf16": torch.int32}


def _load_golden_module():
    golden_path = SIM_DIR.parent / "assets" / "impl" / "golden.py"
    spec = importlib.util.spec_from_file_location(
        "agm_v2_assets_golden_selftest", golden_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 输入合成（value_spec 语义与 example 契约一致，见 tests/sim/README.md）
# ---------------------------------------------------------------------------


def gen_values(shape, dtype_key, spec, rng):
    """Synthesize inputs per value_spec (clean band [-5,5] 分层采样等)."""
    torch_dtype = _TORCH_DTYPES[dtype_key]
    numel = int(np.prod(shape)) if len(shape) else 1
    if numel == 0:
        return torch.empty([int(d) for d in shape], dtype=torch_dtype)
    if spec == "clean":
        # 80% uniform(-5,5) + 20% uniform(-1,1)（小值带分层，含负镜像）
        big = torch.from_numpy(
            rng.uniform(-5.0, 5.0, size=numel).astype(np.float32))
        small = torch.from_numpy(
            rng.uniform(-1.0, 1.0, size=numel).astype(np.float32))
        take_small = torch.from_numpy(
            rng.uniform(0.0, 1.0, size=numel) < 0.2)
        vals = torch.where(take_small, small, big)
    elif spec == "signed_zero":
        vals = torch.from_numpy(
            rng.uniform(-5.0, 5.0, size=numel).astype(np.float32))
        zero_mask = torch.from_numpy(
            rng.uniform(0.0, 1.0, size=numel) < 0.2)
        neg_zero = torch.from_numpy(
            np.where(rng.uniform(0, 1, size=numel) < 0.5,
                     np.float32(-0.0), np.float32(0.0)))
        vals = torch.where(zero_mask, neg_zero, vals)
    elif spec == "inf_nan":
        vals = torch.from_numpy(
            rng.uniform(-5.0, 5.0, size=numel).astype(np.float32))
        u = rng.uniform(0.0, 1.0, size=numel)
        vals = torch.where(
            torch.from_numpy(u < 0.001), torch.tensor(float("inf")), vals)
        vals = torch.where(
            torch.from_numpy((u >= 0.001) & (u < 0.002)),
            torch.tensor(float("-inf")), vals)
        vals = torch.where(
            torch.from_numpy((u >= 0.002) & (u < 0.005)),
            torch.tensor(float("nan")), vals)
    elif spec == "magnitude_bands":
        # 分层量级带：[1,5] / [1e-2,1e-1] / [1e-4,1e-3] / [0,1e-6]，含负镜像
        bands = [(1.0, 5.0), (1e-2, 1e-1), (1e-4, 1e-3), (0.0, 1e-6)]
        pick = rng.integers(0, len(bands), size=numel)
        out = np.empty(numel, dtype=np.float32)
        for i, (lo, hi) in enumerate(bands):
            m = pick == i
            out[m] = rng.uniform(lo, hi, size=int(m.sum()))
        out *= np.where(rng.uniform(0, 1, size=numel) < 0.5, 1.0, -1.0)
        vals = torch.from_numpy(out)
    else:
        raise ValueError(f"unknown value_spec: {spec}")
    return vals.to(torch_dtype).reshape([int(d) for d in shape])


# ---------------------------------------------------------------------------
# dump 写出（与参数化 example 的接口契约一致）
# ---------------------------------------------------------------------------


def _write_tensor(path, tensor):
    np.frombuffer(compare_dump._tensor_bytes(tensor.contiguous()),
                  dtype=np.uint8).tofile(path)


def write_dump(dump_dir, case, rank_tensors, actuals, meta_extra=None):
    """Write one case dump.

    rank_tensors[r]: {"x1", "x2"(API view), "x2_storage"(transB 时的
      W.t() 连续 storage，(n,k) 行主序), "bias"}
      transB 时以 x2_storage 落盘，meta 声明 view shape=(k,n),
      strides=(1,k)（CCU 外层实参形态）；非 transB 时直接落 x2 (k,n) 行主序。
    actuals[r]: {"output", "gather_out"(可 None)}
    """
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    rank_size = case["rank_size"]
    dtype_key = case["dtype"]
    trans_b = bool(case.get("trans_b", False))
    k, n = case["k"], case["n"]

    tensors_meta = {}
    if trans_b:
        # transB 契约（950 CCU/ai_cpu 外层 aclnn 实参形态）：storage 为
        # W.t() 连续（(n,k) 行主序），view 为 (k,n)、strides (1,k)
        # （op_api CheckShape 按 dim0 取 k 轴、IsTransposeLastTwoDims 按
        # stride 判定、TransX2Tensor 重建 inner (n,k) desc）
        tensors_meta["x2"] = {"shape": [k, n], "strides": [1, k]}
    else:
        tensors_meta["x2"] = {"shape": [k, n], "strides": [n, 1]}

    meta = {
        "case_id": case["id"],
        "desc": case.get("desc", ""),
        "dtype": dtype_key,
        "m": case["m"], "k": k, "n": n,
        "rank_size": rank_size,
        "comm_mode": case.get("comm_mode", "ai_cpu"),
        "bias": bool(case.get("bias", False)),
        "trans_b": trans_b,
        "gather_out": bool(case.get("gather_out", True)),
        "value_spec": case.get("value_spec", "clean"),
        "repeat_launch": int(case.get("repeat_launch", 1)),
        "tensors": tensors_meta,
    }
    if meta_extra:
        meta.update(meta_extra)

    for r in range(rank_size):
        rt = rank_tensors[r]
        _write_tensor(dump_dir / f"x1_rank{r}.bin", rt["x1"])
        x2_storage = rt["x2_storage"] if trans_b else rt["x2"]
        _write_tensor(dump_dir / f"x2_rank{r}.bin", x2_storage)
        if case.get("bias", False) and rt.get("bias") is not None:
            _write_tensor(dump_dir / f"bias_rank{r}.bin", rt["bias"])
        act = actuals[r]
        _write_tensor(dump_dir / f"output_rank{r}.bin", act["output"])
        if case.get("gather_out", True) and act.get("gather_out") is not None:
            _write_tensor(dump_dir / f"gather_out_rank{r}.bin",
                          act["gather_out"])
    with open(dump_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def build_case_dump(dump_dir, case, seed, corrupt=None):
    """Synthesize inputs, compute golden, write dump with given corruption.

    corrupt: None -> actual = golden（正路径）
             callable(case, rank, output, gather, rng) -> (output, gather)
    """
    rng = np.random.default_rng(seed)
    m, k, n = case["m"], case["k"], case["n"]
    rank_size = case["rank_size"]
    trans_b = bool(case.get("trans_b", False))

    rank_tensors = []
    rank_inputs = []
    for r in range(rank_size):
        x1 = gen_values([m, k], case["dtype"],
                        case.get("value_spec", "clean"), rng)
        # 逻辑权重 W (k,n) contiguous；transB 时按 CCU 外层实参形态构造：
        # storage = W.t() 连续（(n,k) 行主序），API view = storage.t()
        # → (k,n) view、strides (1,k)
        w = gen_values([k, n], case["dtype"],
                       case.get("value_spec", "clean"), rng)
        if trans_b:
            x2_storage = w.t().contiguous()
            x2 = x2_storage.t()
        else:
            x2_storage = w
            x2 = w
        bias = None
        if case.get("bias", False):
            bias = gen_values([n], case["dtype"], "clean", rng)
        rank_tensors.append({"x1": x1, "x2": x2, "x2_storage": x2_storage,
                             "bias": bias})
        rank_inputs.append({"x1": x1, "x2": x2, "bias": bias})

    goldens = _load_golden_module().agm_v2_sim_golden(case, rank_inputs)

    actuals = []
    for r in range(rank_size):
        output = goldens["output"][r].clone()
        gather = None
        if case.get("gather_out", True) and goldens["gather_out"] is not None:
            gather = goldens["gather_out"][r].clone()
        if corrupt is not None:
            output, gather = corrupt(case, r, output, gather, rng)
        actuals.append({"output": output, "gather_out": gather})

    write_dump(dump_dir, case, rank_tensors, actuals)
    # 确定性用例：追加 run1 dump（默认与 run0 相同 → 应判一致）
    if int(case.get("repeat_launch", 1)) > 1:
        for r in range(rank_size):
            shutil.copyfile(dump_dir / f"output_rank{r}.bin",
                            dump_dir / f"output_rank{r}_run1.bin")
    return goldens


# ---------------------------------------------------------------------------
# 定向损坏注入
# ---------------------------------------------------------------------------


def _corrupt_gatherout_value(case, r, output, gather, rng):
    """gatherOut 中一个有限元素 +1（字节级可见损坏）。"""
    if gather is not None and gather.numel() > 0:
        flat = gather.reshape(-1)
        finite = torch.isfinite(flat).nonzero(as_tuple=False).flatten()
        if finite.numel() > 0:
            idx = int(finite[int(rng.integers(0, finite.numel()))])
            flat[idx] = flat[idx] + 1
    return output, gather


def _corrupt_sign_zero(case, r, output, gather, rng):
    """把 gatherOut 中一个 +0 的符号位翻成 -0（数值相等，位不同）。"""
    if gather is not None and gather.numel() > 0:
        flat = gather.reshape(-1)
        bit_dtype = _BIT_DTYPES[case["dtype"]]
        bits = flat.view(bit_dtype)
        for idx in range(flat.numel()):
            if int(bits[idx]) == 0:  # +0（0x0000）
                # 置符号位：fp16 0x8000（-32768）/ bf16 0x80000000
                bits[idx] = -32768 if bit_dtype == torch.int16 else -2147483648
                break
    return output, gather


def _corrupt_nan_payload(case, r, output, gather, rng):
    """gatherOut 中一个 NaN 改为不同 payload 的 NaN（值级均为 NaN）。"""
    if gather is not None and gather.numel() > 0:
        flat = gather.reshape(-1)
        bit_dtype = _BIT_DTYPES[case["dtype"]]
        bits = flat.view(bit_dtype)
        nans = torch.isnan(flat).nonzero(as_tuple=False).flatten()
        if nans.numel() > 0:
            idx = int(nans[0])
            if bit_dtype == torch.int16:
                bits[idx] = 0x7E01  # canonical nan 是 0x7E00
            else:
                bits[idx] = 0x7FC00001
    return output, gather


def _corrupt_output_big(case, r, output, gather, rng):
    """output 整体 +0.5（超过 fp16 max_abs_error_limit=0.1）。"""
    if output.numel() > 0:
        output = (output.float() + 0.5).to(_TORCH_DTYPES[case["dtype"]])
    return output, gather


def _corrupt_output_within_tol(case, r, output, gather, rng):
    """output 整体 +atol/2（应仍通过混合容差——容差不误杀）。"""
    if output.numel() > 0:
        perturbed = output.float() + (
            compare_dump.PRECISION_CRITERIA[case["dtype"]]["atol"] / 2.0)
        output = perturbed.to(_TORCH_DTYPES[case["dtype"]])
    return output, gather


SELFTESTS = [
    # (name, case dict, corrupt, expect_pass)
    ("fp16_basic_pass",
     {"id": "ST-L0-01", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     None, True),
    ("fp16_transb_bias_pass",
     {"id": "ST-L1-101", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "bias": True, "trans_b": True,
      "value_spec": "clean"},
     None, True),
    ("bf16_basic_pass",
     {"id": "ST-L0-02", "dtype": "bf16", "m": 64, "k": 256, "n": 96,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     None, True),
    ("bf16_mag_bands_pass",
     {"id": "ST-L1-152", "dtype": "bf16", "m": 64, "k": 256, "n": 96,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "magnitude_bands"},
     None, True),
    ("rank4_ccu_pass",
     {"id": "ST-L1-140", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 4, "comm_mode": "ccu", "value_spec": "clean"},
     None, True),
    ("signed_zero_bitwise_pass",
     {"id": "ST-L1-150", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "signed_zero"},
     None, True),
    ("inf_nan_propagation_info",
     {"id": "ST-L1-151", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "inf_nan"},
     None, True),
    ("empty_m0_pass",
     {"id": "ST-L1-160", "dtype": "fp16", "m": 0, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     None, True),
    ("n0_comm_only_pass",
     {"id": "ST-L1-162", "dtype": "fp16", "m": 128, "k": 768, "n": 0,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     None, True),
    ("gather_out_false_pass",
     {"id": "ST-L1-102", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "trans_b": True,
      "gather_out": False, "value_spec": "clean"},
     None, True),
    ("determinism_pass",
     {"id": "ST-L1-170", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean",
      "repeat_launch": 2},
     None, True),
    ("within_tol_still_pass",
     {"id": "ST-POS-TOL", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     _corrupt_output_within_tol, True),
    ("gatherout_value_flip_fail",
     {"id": "ST-NEG-B1", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     _corrupt_gatherout_value, False),
    ("sign_zero_flip_fail",
     {"id": "ST-NEG-B2", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "signed_zero"},
     _corrupt_sign_zero, False),
    ("nan_payload_fail",
     {"id": "ST-NEG-B3", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "inf_nan"},
     _corrupt_nan_payload, False),
    ("output_over_tolerance_fail",
     {"id": "ST-NEG-T1", "dtype": "fp16", "m": 128, "k": 768, "n": 128,
      "rank_size": 2, "comm_mode": "ai_cpu", "value_spec": "clean"},
     _corrupt_output_big, False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None,
                        help="自检工作目录（默认 selftest_runs/，自动清空）")
    args = parser.parse_args()
    workdir = Path(args.workdir) if args.workdir else SIM_DIR / "selftest_runs"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    all_ok = True
    for i, (name, case, corrupt, expect_pass) in enumerate(SELFTESTS):
        dump_dir = workdir / name
        build_case_dump(dump_dir, case, seed=20260000 + i, corrupt=corrupt)
        report = compare_dump.compare_case(dump_dir)
        ok = report["pass"] == expect_pass
        all_ok = all_ok and ok
        print(f"[selftest:{'OK ' if ok else 'BAD'}] {name}: "
              f"expect_pass={expect_pass} got={report['pass']}")
        if not ok:
            for rep in report["tensors"]:
                if rep.get("role") == "crosscheck":
                    continue
                print(f"    {rep}")

    # 确定性破坏负例：翻转 run1 的一个字节 → determinism_bitwise FAIL
    det_dir = workdir / "determinism_break"
    det_case = {"id": "ST-NEG-D1", "dtype": "fp16", "m": 128, "k": 768,
                "n": 128, "rank_size": 2, "comm_mode": "ai_cpu",
                "value_spec": "clean", "repeat_launch": 2}
    build_case_dump(det_dir, det_case, seed=20269999)
    run1 = det_dir / "output_rank0_run1.bin"
    data = bytearray(run1.read_bytes())
    data[0] ^= 0x01
    run1.write_bytes(bytes(data))
    report = compare_dump.compare_case(det_dir)
    det_ok = not report["pass"]
    all_ok = all_ok and det_ok
    print(f"[selftest:{'OK ' if det_ok else 'BAD'}] determinism_break: "
          f"expect_pass=False got={report['pass']}")

    print("=" * 60)
    print(f"selftest {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

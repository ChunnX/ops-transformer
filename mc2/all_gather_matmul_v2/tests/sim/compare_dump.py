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
"""Channel-A compare script: example dump -> CPU golden -> 2.1 §2.3 verdict.

权威精度断言（唯一取值处 = 2.1 §2.3 精度判定口径表）:

  output   FP16: |actual-golden| <= atol + rtol*|golden|, rtol = atol = 2^-9,
                matched_ratio >= 0.99 且 max_abs_error <= 0.1
  output   BF16: 同上, rtol = atol = 2^-6, max_abs_error <= 1.0
  gatherOut     : 逐位字节一致 (tobytes()/view(uint8))，NaN 位形与 ±0 符号位
                均严格检出，阈值 0（bitwise，字节级）
  特例（2.1 §2.3 口径一致性声明）: x1 含 ±inf/nan 的用例，output 的混合容差
  判定无意义（nan≠nan），走传播语义核对（信息性，不作为通过依据），
  gatherOut 仍按 bitwise 严格判定（硬门槛）。

辅助交叉核对（不作通过依据）：对 output 另算 ST acc 口径
（cv_fused_double_benchmark：max_re_ratio<=5, avg_re_ratio<=1.5，vs 双精度
golden）。两口径结论不一致时以 §2.3 为准，并在报告中标注分叉。

每条用例逐输出张量产出「张量 / 判定指标 / 实测值 / 阈值 / 结论」并随结果
落盘（report json + summary jsonl）。

Usage:
  python3 compare_dump.py --dump-dir <dir> [--out-dir <dir>] [--json-out FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_TESTS = Path(__file__).resolve().parents[1]  # .../tests
ASSETS_IMPL = REPO_TESTS / "assets" / "impl" / "golden.py"

# ---------------------------------------------------------------------------
# 2.1 §2.3 精度判定口径表（唯一取值处，不得放宽/换指标）
# ---------------------------------------------------------------------------

PRECISION_CRITERIA = {
    "fp16": {
        "rtol": 2.0**-9,  # ≈ 1.953e-3
        "atol": 2.0**-9,
        "matched_ratio_min": 0.99,
        "max_abs_error_limit": 0.1,  # max(1e-1, 32*ULP)=max(0.1, 32*2^-10)=0.1
        "source": "ops-precision-standard/references/float_compute.md §3 "
        "FLOAT16 列; 2.1 §2.3 口径表",
    },
    "bf16": {
        "rtol": 2.0**-6,  # ≈ 1.5625e-2
        "atol": 2.0**-6,
        "matched_ratio_min": 0.99,
        "max_abs_error_limit": 1.0,  # max(1e-0, 32*ULP)=max(1.0, 32*2^-7)=1.0
        "source": "ops-precision-standard/references/float_compute.md §3 "
        "BF16 列; 2.1 §2.3 口径表",
    },
}

# gatherOut：逐位字节一致（2.1 §2.3 S1 实现注记：不用 np.array_equal——
# 其对 NaN 恒 False 且 +0==-0 判相等，±0 符号位翻转不检出）。
GATHEROUT_METRIC = "bitwise_bytes"
GATHEROUT_THRESHOLD = 0

# 辅助交叉核对阈值（仓内 ST standard.acc = cv_fused_double_benchmark，
# 仅交叉核对，不作通过依据）
CROSSCHECK_RE_RATIOS = {"max_re_ratio": 5.0, "avg_re_ratio": 1.5}

_TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# tensor 基础工具（numpy 无原生 bfloat16，字节级操作经 torch 重解释）
# ---------------------------------------------------------------------------


def _tensor_bytes(t):
    """Bit-exact bytes of a torch/numpy tensor (bf16 经 uint16 重解释)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            return t.view(torch.uint16).numpy().tobytes()
        return t.numpy().tobytes()
    return np.ascontiguousarray(t).tobytes()


def _to_f32(t):
    """Upcast to float32 numpy array (golden/actual 统一比较域)."""
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).numpy()
    return np.asarray(t, dtype=np.float32)


# ---------------------------------------------------------------------------
# dump 装载
# ---------------------------------------------------------------------------


def load_dump(dump_dir):
    """Load meta.json + per-rank raw bins of the API actual args.

    Dump 契约（与参数化 example 的接口约定，见 tests/sim/README.md）:
      meta.json — case 参数与每张 API 实参张量的 shape/strides（元素单位）
      x1_rank{r}.bin / x2_rank{r}.bin / bias_rank{r}.bin(可选)
      output_rank{r}.bin / gather_out_rank{r}.bin(可选)
      确定性用例（repeat_launch>1）追加 output_rank{r}_run{i}.bin /
      gather_out_rank{r}_run{i}.bin，i>=1；run0 无后缀。
    """
    dump_dir = Path(dump_dir)
    with open(dump_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    dtype = _TORCH_DTYPES[meta["dtype"]]
    itemsize = torch.empty((), dtype=dtype).element_size()

    def _load(fname, shape, strides):
        path = dump_dir / fname
        if not path.exists():
            return None
        raw = np.fromfile(path, dtype=np.uint8)
        elems = 1
        for d in shape:
            elems *= int(d)
        if elems == 0:
            return torch.empty([int(d) for d in shape], dtype=dtype)
        if raw.size < elems * itemsize:
            raise ValueError(
                f"{fname}: dump bytes {raw.size} < expected {elems * itemsize}"
            )
        # 完整 storage 交给 as_strided（padded stride 场景不截断）
        storage = torch.from_numpy(raw).view(dtype)
        return torch.as_strided(
            storage, [int(d) for d in shape], [int(s) for s in strides]
        )

    rank_size = meta["rank_size"]
    desc = meta.get("tensors", {})
    # transB 默认契约（950 CCU/ai_cpu 外层 aclnn 实参形态）：x2 view 为
    # (k,n) + strides (1,k)，storage 为 (n,k) 行主序（W.t() 连续）。
    # inner/TTK 的 (n,k) 形态同样接受（golden 按视图 shape 与 k 判别）。
    defaults = {
        "x1": {"shape": [meta["m"], meta["k"]], "strides": [meta["k"], 1]},
        "x2": {
            "shape": desc.get("x2", {}).get("shape", [meta["k"], meta["n"]]),
            "strides": desc.get("x2", {}).get(
                "strides", [1, meta["k"]] if meta.get("trans_b") else [meta["n"], 1]
            ),
        },
        "bias": {"shape": [meta["n"]], "strides": [1]},
        "output": {
            "shape": [meta["m"] * rank_size, meta["n"]],
            "strides": [meta["n"], 1],
        },
        "gather_out": {
            "shape": [meta["m"] * rank_size, meta["k"]],
            "strides": [meta["k"], 1],
        },
    }
    load_flags = {
        "x1": True,
        "x2": True,
        "bias": bool(meta.get("bias", False)),
        "output": True,
        "gather_out": bool(meta.get("gather_out", True)),
    }
    tensors = {}
    for name in ("x1", "x2", "bias", "output", "gather_out"):
        if not load_flags[name]:
            tensors[name] = [None] * rank_size
            continue
        per_rank = []
        for r in range(rank_size):
            spec = dict(defaults[name])
            spec.update(desc.get(name, {}))
            per_rank.append(
                _load(f"{name}_rank{r}.bin", spec["shape"], spec["strides"])
            )
        tensors[name] = per_rank

    # 确定性 repeat runs
    repeat = int(meta.get("repeat_launch", 1))
    repeat_outputs = []
    if repeat > 1:
        for r in range(rank_size):
            runs = [tensors["output"][r]]
            for i in range(1, repeat):
                spec = dict(defaults["output"])
                spec.update(desc.get("output", {}))
                runs.append(
                    _load(f"output_rank{r}_run{i}.bin", spec["shape"], spec["strides"])
                )
            repeat_outputs.append(runs)
    tensors["output_runs"] = repeat_outputs
    return meta, tensors


# ---------------------------------------------------------------------------
# 判定实现
# ---------------------------------------------------------------------------


def check_mixed_tolerance(actual, golden, dtype_key):
    """§2.3 output 混合容差：逐元素 |a-g| <= atol + rtol*|g|。

    返回 (report dict, passed)。golden/actual 均为已按声明 dtype 舍入的
    张量；误差在 float32 域计算。空 tensor（numel==0）按通过处理
    （matched_ratio 定义为 1.0）。
    """
    crit = PRECISION_CRITERIA[dtype_key]
    a = _to_f32(actual)
    g = _to_f32(golden)
    total = int(g.size)
    if total == 0:
        return {
            "metric": "mixed_tolerance",
            "measured": {"matched_ratio": 1.0, "max_abs_error": 0.0, "numel": 0},
            "threshold": {
                k: crit[k]
                for k in ("rtol", "atol", "matched_ratio_min", "max_abs_error_limit")
            },
            "conclusion": "PASS",
        }, True
    err = np.abs(a - g)
    finite_mask = np.isfinite(g) & np.isfinite(a)
    tol = crit["atol"] + crit["rtol"] * np.abs(g)
    elem_pass = (err <= tol) & finite_mask
    # 非有限元素（inf/nan 混入时）不参与容差通过率——该类用例 output 走
    # 传播语义核对（check_propagation），此处仅统计有限域。
    matched = int(elem_pass.sum())
    finite_total = int(finite_mask.sum())
    matched_ratio = (matched / finite_total) if finite_total > 0 else 1.0
    finite_err = np.where(finite_mask, err, 0.0)
    max_abs_error = float(finite_err.max()) if finite_err.size else 0.0
    passed = (
        matched_ratio >= crit["matched_ratio_min"]
        and max_abs_error <= crit["max_abs_error_limit"]
    )
    return {
        "metric": "mixed_tolerance",
        "measured": {
            "matched_ratio": round(float(matched_ratio), 6),
            "max_abs_error": round(max_abs_error, 6),
            "numel": total,
            "finite_numel": finite_total,
        },
        "threshold": {
            k: crit[k]
            for k in ("rtol", "atol", "matched_ratio_min", "max_abs_error_limit")
        },
        "conclusion": "PASS" if passed else "FAIL",
    }, passed


def check_bitwise(actual, golden):
    """§2.3 gatherOut 逐位字节一致：tobytes() 级比较。

    NaN 位形（payload 差异）与 ±0 符号位均严格检出。返回 (report, passed)。
    """
    a_bytes = _tensor_bytes(actual)
    g_bytes = _tensor_bytes(golden)
    mismatched = 0
    if a_bytes != g_bytes:
        ab = np.frombuffer(a_bytes, dtype=np.uint8)
        gb = np.frombuffer(g_bytes, dtype=np.uint8)
        if ab.size != gb.size:
            mismatched = max(ab.size, gb.size)
        else:
            mismatched = int((ab != gb).sum())
    passed = a_bytes == g_bytes
    return {
        "metric": GATHEROUT_METRIC,
        "measured": {"mismatched_bytes": mismatched, "total_bytes": len(g_bytes)},
        "threshold": GATHEROUT_THRESHOLD,
        "conclusion": "PASS" if passed else "FAIL",
    }, passed


def check_propagation(actual, golden):
    """±inf/nan 传播语义核对（信息性，不作通过依据，2.1 §2.3 特例）。

    对 golden 为 inf/-inf/nan 的元素，actual 对应元素应为同类（nan→nan，
    ±inf→同号 ±inf）。仅报告 class-matched 比例。
    """
    a = _to_f32(actual)
    g = _to_f32(golden)
    g_nan = np.isnan(g)
    g_pos_inf = np.isposinf(g)
    g_neg_inf = np.isneginf(g)
    nan_ok = bool((np.isnan(a) & g_nan).sum() == g_nan.sum()) if g_nan.any() else True
    pos_ok = (
        bool((np.isposinf(a) & g_pos_inf).sum() == g_pos_inf.sum())
        if g_pos_inf.any()
        else True
    )
    neg_ok = (
        bool((np.isneginf(a) & g_neg_inf).sum() == g_neg_inf.sum())
        if g_neg_inf.any()
        else True
    )
    special = int(g_nan.sum() + g_pos_inf.sum() + g_neg_inf.sum())
    ok = nan_ok and pos_ok and neg_ok
    return {
        "metric": "special_value_propagation",
        "measured": {
            "special_elements": special,
            "nan_class_match": nan_ok,
            "pos_inf_match": pos_ok,
            "neg_inf_match": neg_ok,
        },
        "threshold": "informational（不作通过依据）",
        "conclusion": "INFO_CONSISTENT" if ok else "INFO_INCONSISTENT",
    }, None  # None => 不 gating


def crosscheck_st_acc(actual, golden, golden_f64):
    """辅助交叉核对：ST acc 口径 cv_fused_double_benchmark（不作通过依据）。

    max_re_ratio / avg_re_ratio 以双精度 golden 为基准的相对误差比。
    """
    a = _to_f32(actual).astype(np.float64)
    g = _to_f32(golden_f64).astype(np.float64)
    if g.size == 0:
        return {
            "metric": "cv_fused_double_benchmark",
            "measured": {"max_re_ratio": 0.0, "avg_re_ratio": 0.0},
            "threshold": dict(CROSSCHECK_RE_RATIOS),
            "conclusion": "PASS",
        }
    with np.errstate(invalid="ignore", divide="ignore"):
        re = np.abs(a - g) / np.maximum(np.abs(g), 1e-30)
    max_re = float(re.max())
    avg_re = float(re.mean())
    ok = (
        max_re <= CROSSCHECK_RE_RATIOS["max_re_ratio"]
        and avg_re <= CROSSCHECK_RE_RATIOS["avg_re_ratio"]
    )
    return {
        "metric": "cv_fused_double_benchmark",
        "measured": {
            "max_re_ratio": round(max_re, 6),
            "avg_re_ratio": round(avg_re, 6),
        },
        "threshold": dict(CROSSCHECK_RE_RATIOS),
        "conclusion": "PASS" if ok else "FAIL",
        "note": "交叉核对项，不作通过依据（2.1 §2.3 辅助交叉核对）",
    }


# ---------------------------------------------------------------------------
# golden 调用（tests/assets/impl/golden.py 的通道 A 实现）
# ---------------------------------------------------------------------------


def _load_golden_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("agm_v2_assets_golden", ASSETS_IMPL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden_f64(case, rank_inputs, golden_mod):
    """双精度 golden（交叉核对基准）：不落 dtype 舍入。"""
    gathered = torch.cat([inp["x1"].to(torch.float64) for inp in rank_inputs], dim=0)
    outs = []
    for inp in rank_inputs:
        x2 = inp["x2"]
        if case.get("trans_b", False):
            x2 = golden_mod.logical_weight_from_view(x2, case["k"])
        g = torch.matmul(gathered, x2.to(torch.float64))
        bias = inp.get("bias")
        if bias is not None and bias.numel() > 0:
            g = g + bias.to(torch.float64)
        outs.append(g)
    return outs


# ---------------------------------------------------------------------------
# 单用例主流程
# ---------------------------------------------------------------------------


def compare_case(dump_dir):
    meta, tensors = load_dump(dump_dir)
    golden_mod = _load_golden_module()

    dtype_key = meta["dtype"]
    if dtype_key not in PRECISION_CRITERIA:
        raise ValueError(f"unsupported dtype in meta.json: {dtype_key}")

    rank_size = meta["rank_size"]
    case = {
        "dtype": dtype_key,
        "m": meta["m"],
        "k": meta["k"],
        "n": meta["n"],
        "rank_size": rank_size,
        "bias": bool(meta.get("bias", False)),
        "trans_b": bool(meta.get("trans_b", False)),
        "gather_out": bool(meta.get("gather_out", True)),
    }
    rank_inputs = []
    for r in range(rank_size):
        rank_inputs.append(
            {
                "x1": tensors["x1"][r],
                "x2": tensors["x2"][r],
                "bias": tensors["bias"][r],
            }
        )

    goldens = golden_mod.agm_v2_sim_golden(case, rank_inputs)
    goldens_f64 = _golden_f64(case, rank_inputs, golden_mod)

    has_special = meta.get("value_spec") == "inf_nan"
    tensor_reports = []
    case_pass = True
    divergences = []

    for r in range(rank_size):
        # ---- output（§2.3 权威断言） ----
        actual = tensors["output"][r]
        golden = goldens["output"][r]
        if has_special:
            rep, _ = check_propagation(actual, golden)
            tensor_reports.append({"tensor": "output", "rank": r, **rep})
        else:
            rep, ok = check_mixed_tolerance(actual, golden, dtype_key)
            tensor_reports.append({"tensor": "output", "rank": r, **rep})
            if not ok:
                case_pass = False
        # 辅助交叉核对（不作通过依据；结论分叉时标注）
        cc = crosscheck_st_acc(actual, golden, goldens_f64[r])
        if (not has_special) and cc["conclusion"] != rep["conclusion"]:
            divergences.append(
                {
                    "tensor": "output",
                    "rank": r,
                    "authoritative": rep["conclusion"],
                    "crosscheck": cc["conclusion"],
                    "note": "两口径结论不一致，以 §2.3 口径表为准（交叉核对项仅参考）",
                }
            )
        tensor_reports.append(
            {"tensor": "output", "rank": r, **cc, "role": "crosscheck"}
        )

        # ---- gatherOut（§2.3 bitwise 硬门槛） ----
        if case["gather_out"] and tensors["gather_out"][r] is not None:
            rep, ok = check_bitwise(tensors["gather_out"][r], goldens["gather_out"][r])
            tensor_reports.append({"tensor": "gather_out", "rank": r, **rep})
            if not ok:
                case_pass = False

    # ---- 确定性（repeat_launch>1：run 间 bitwise 一致，硬门槛） ----
    repeat = int(meta.get("repeat_launch", 1))
    if repeat > 1:
        for r in range(rank_size):
            runs = tensors["output_runs"][r]
            base = _tensor_bytes(runs[0])
            for i in range(1, repeat):
                cur = _tensor_bytes(runs[i])
                ok = base == cur
                tensor_reports.append(
                    {
                        "tensor": "output",
                        "rank": r,
                        "metric": "determinism_bitwise",
                        "measured": {
                            f"run0_vs_run{i}": "identical" if ok else "differ"
                        },
                        "threshold": "bitwise identical（其他要求 #3）",
                        "conclusion": "PASS" if ok else "FAIL",
                    }
                )
                if not ok:
                    case_pass = False

    return {
        "case_id": meta.get("case_id", Path(dump_dir).name),
        "desc": meta.get("desc", ""),
        "dtype": dtype_key,
        "params": {
            "m": case["m"],
            "k": case["k"],
            "n": case["n"],
            "rank_size": case["rank_size"],
            "comm_mode": meta.get("comm_mode", ""),
            "bias": case["bias"],
            "trans_b": case["trans_b"],
            "gather_out": case["gather_out"],
            "value_spec": meta.get("value_spec", "clean"),
            "value_scale": meta.get("value_scale", 1.0),
        },
        "pass": case_pass,
        "tensors": tensor_reports,
        "crosscheck_divergences": divergences,
        "criteria_source": "2.1 §2.3 精度判定口径表"
        "（ops-precision-standard float_compute §3 实例化）",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        required=True,
        help="example dump 目录（含 meta.json 与 per-rank bins）",
    )
    parser.add_argument(
        "--out-dir", default=None, help="报告输出目录（默认 dump 目录）"
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="报告 json 文件名（默认 report_<case_id>.json）",
    )
    parser.add_argument("--summary", default=None, help="追加汇总 jsonl 路径（可选）")
    args = parser.parse_args()

    report = compare_case(args.dump_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out or f"report_{report['case_id']}.json"
    out_path = out_dir / json_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    print(f"[compare] case={report['case_id']} pass={report['pass']} report={out_path}")
    for rep in report["tensors"]:
        if rep.get("role") == "crosscheck":
            continue
        print(
            "  {tensor}[rank{rank}] {metric}: {conclusion} "
            "measured={measured} threshold={threshold}".format(
                tensor=rep["tensor"],
                rank=rep.get("rank", "-"),
                metric=rep["metric"],
                conclusion=rep["conclusion"],
                measured=json.dumps(rep["measured"], ensure_ascii=False),
                threshold=json.dumps(rep.get("threshold"), ensure_ascii=False),
            )
        )
    for d in report["crosscheck_divergences"]:
        print(f"  [crosscheck-divergence] {d}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""AttnResFwd 单 shape 多实现精度比对 + Ascend Profiler 性能采集。

仓内入口（推荐）：
  cd attention/attn_res_fwd/tests/pytest
  source env_kdy.sh && source cann_env.sh
  python attenres_lite.py
  bash test_run.sh lite

详见 ATTENRES_LITE.md。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_npu

SCRIPT_DIR = Path(__file__).resolve().parent
# pytest → tests → attn_res_fwd → attention → ops-transformer
OPS_TRANSFORMER_ROOT = SCRIPT_DIR.parents[4]
TORCH_EXT_ROOT = OPS_TRANSFORMER_ROOT / "torch_extension"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TORCH_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCH_EXT_ROOT))

from precision_compare import cal_relative_diff_np, data_compare  # noqa: E402

BF16_MERE_THRESHOLD = 1.0 / 128.0  # 2^-7，与 ACLNN / pytest 一致


def resolve_npu_device() -> int:
    try:
        import npu_device

        return npu_device.resolve_device_id(force_auto=os.environ.get("FORCE_AUTO_DEVICE") == "1")
    except Exception:
        return int(os.environ.get("DEVICE_ID", "7"))


NPU_DEVICE = resolve_npu_device()
torch.npu.set_device(NPU_DEVICE)
print(f"Using NPU device: {NPU_DEVICE}")
if os.environ.get("CANN_PKG_USED"):
    print(f"CANN package: {os.environ['CANN_PKG_USED']}")
elif os.environ.get("ATTN_RES_FWD_CANN_PKG"):
    print(f"CANN package: {os.environ['ATTN_RES_FWD_CANN_PKG']}")


def apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm_eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def apply_attn_res_cust(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm_eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = (probs.unsqueeze(-1) * v_float.unsqueeze(1)).sum(dim=2).squeeze(1)
    return hidden_states.to(v.dtype)


def Attn_Res_Fwd_nomm(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    v_normed = torch_npu.npu_rms_norm(v_float, norm_weight.float(), norm_eps)[0]
    scores = (v_normed * proj_weight.squeeze(0).float()).sum(-1, keepdim=False)
    probs = scores.softmax(dim=-1).unsqueeze(1)
    hidden_states = (probs.unsqueeze(-1) * v_float.unsqueeze(1)).sum(dim=2).squeeze(1)
    return hidden_states.to(v.dtype)


def Attn_Res_Fwd(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    v_normed = torch_npu.npu_rms_norm(v_float, norm_weight.float(), norm_eps)[0]
    scores = (v_normed * proj_weight.squeeze(0).float()).sum(-1, keepdim=False)
    probs = scores.softmax(dim=-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v.float()).squeeze(1)
    return hidden_states.to(v.dtype)


def Attn_Res_Fwd_MMVersion(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    v_normed = torch_npu.npu_rms_norm(v_float, norm_weight.float(), norm_eps)[0]
    scores = torch.matmul(v_normed, proj_weight.T.float()).squeeze(-1)
    probs = scores.softmax(dim=-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v.float()).squeeze(1)
    return hidden_states.to(v.dtype)


def Attn_Res_Fwd_FAVersion(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    v_normed = torch_npu.npu_rms_norm(v_float, norm_weight.float(), norm_eps)[0]
    scores = torch.matmul(v_normed, proj_weight.T.float()).squeeze(-1)
    probs = scores.softmax(dim=-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v.float()).squeeze(1)
    return hidden_states.to(v.dtype)


def attn_res_fwd_cann(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
    """自定义算子 Torch 适配插件入口（OpBuilder JIT → aclnnAttnResFwd）。"""
    import attn_res_fwd_torch_runner as torch_runner

    fn = torch_runner.get_attn_res_fwd_fn()
    return fn(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps)


IMPLEMENTATIONS: Dict[str, Callable] = {
    "apply_attn_res": apply_attn_res,
    "apply_attn_res_cust": apply_attn_res_cust,
    "Attn_Res_Fwd_nomm": Attn_Res_Fwd_nomm,
    "Attn_Res_Fwd": Attn_Res_Fwd,
    "Attn_Res_Fwd_MMVersion": Attn_Res_Fwd_MMVersion,
    "Attn_Res_Fwd_FAVersion": Attn_Res_Fwd_FAVersion,
    "attn_res_fwd_cann": attn_res_fwd_cann,
}


def prepare_tensors(
    num_tokens=544,
    num_blocks=12,
    hidden_size=7168,
    dtype=torch.bfloat16,
    seed=0,
):
    torch.manual_seed(seed)
    device = f"npu:{NPU_DEVICE}"
    # CPU 生成再 .to(npu)，避免部分 OPP 下 NPU 直接 randn(bf16) 失败
    prefix_sum = torch.randn(num_tokens, hidden_size, device="cpu", dtype=dtype).to(device)
    block_residual = torch.randn(num_tokens, num_blocks, hidden_size, device="cpu", dtype=dtype).to(device)
    norm_weight = torch.randn(hidden_size, device="cpu", dtype=dtype).to(device)
    proj_weight = torch.randn(1, hidden_size, device="cpu", dtype=dtype).to(device)
    norm_eps = 1e-5
    return prefix_sum, block_residual, proj_weight, norm_weight, norm_eps


def clone_inputs(tensors):
    prefix_sum, block_residual, proj_weight, norm_weight, norm_eps = tensors
    return (
        prefix_sum.clone(),
        block_residual.clone(),
        proj_weight.clone(),
        norm_weight.clone(),
        norm_eps,
    )


def _find_profiler_csv(base_dir: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _parse_kernel_duration_us(profiler_dir: str) -> Tuple[Optional[float], Optional[str]]:
    csv_path = _find_profiler_csv(profiler_dir, "kernel_details.csv")
    if csv_path is None:
        return None, None
    df = pd.read_csv(csv_path)
    duration_col = next((c for c in df.columns if c.strip().startswith("Duration")), None)
    if duration_col is None:
        return None, csv_path
    return float(df[duration_col].astype(float).sum()), csv_path


def _parse_op_statistic_time_us(profiler_dir: str) -> Tuple[Optional[float], Optional[str]]:
    csv_path = _find_profiler_csv(profiler_dir, "op_statistic.csv")
    if csv_path is None:
        return None, None
    df = pd.read_csv(csv_path)
    total_time_col = next((c for c in df.columns if "Total Time" in c), None)
    if total_time_col is None:
        return None, csv_path
    return float(df[total_time_col].astype(float).sum()), csv_path


def _print_profiler_summary(profiler_dir: str, tag: str = "") -> dict:
    print(f"\n===== profiler summary ({tag}) =====")
    print(f"profiler_dir: {profiler_dir}")
    kernel_duration, kernel_csv = _parse_kernel_duration_us(profiler_dir)
    if kernel_duration is not None:
        print(f"kernel_details Duration(us) sum: {kernel_duration:.3f}")
        print(f"kernel_details csv: {kernel_csv}")
    else:
        print("kernel_details parse skipped")
    op_total_time, op_csv = _parse_op_statistic_time_us(profiler_dir)
    if op_total_time is not None:
        print(f"op_statistic Total Time(us) sum: {op_total_time:.3f}")
        print(f"op_statistic csv: {op_csv}")
    else:
        print("op_statistic parse skipped")
    return {
        "tag": tag,
        "profiler_dir": profiler_dir,
        "kernel_duration_us": kernel_duration,
        "op_total_time_us": op_total_time,
    }


def _create_profiler(result_path: str):
    os.makedirs(result_path, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
    )


def benchmark_implementation(
    name: str,
    fn: Callable,
    tensors,
    output_root: str,
    warmup: int = 3,
) -> dict:
    profiler_dir = os.path.join(output_root, name)

    def run_once():
        ps, br, pw, nw, eps = clone_inputs(tensors)
        return fn(ps, br, pw, nw, eps)

    prof = _create_profiler(profiler_dir)
    for _ in range(warmup):
        run_once()
        torch.npu.synchronize()
    prof.start()
    run_once()
    torch.npu.synchronize()
    prof.stop()
    return _print_profiler_summary(profiler_dir, tag=name)


DEFAULT_BASELINE = "apply_attn_res_cust"
DEFAULT_COMPARE = ["Attn_Res_Fwd_MMVersion", "attn_res_fwd_cann"]


def _compare_precision(
    out: np.ndarray,
    golden: np.ndarray,
    verbose: bool,
) -> Tuple[bool, float, float]:
    thd = BF16_MERE_THRESHOLD
    if verbose:
        result, pct, max_err = data_compare(out, golden, thd, thd, thd)
        return result == "success", pct, max_err

    real = np.asarray(out).flatten().astype(np.float32)
    expect = np.asarray(golden).flatten().astype(np.float32)
    if real.size != expect.size:
        return False, 0.0, 0.0
    if real.size == 0:
        return True, 100.0, 0.0

    diff_abs = np.abs(real - expect)
    diff_index = np.where(diff_abs > 0)
    if diff_index[0].size == 0:
        return True, 100.0, 0.0

    rdiff = cal_relative_diff_np(real[diff_index], expect[diff_index], thd)
    err_diff = rdiff[rdiff > thd]
    pct = float(real.size - err_diff.size) / float(real.size) * 100.0
    max_err = float(max(err_diff)) if err_diff.size else 0.0
    passed = pct >= (1 - thd) * 100.0 and (err_diff.size == 0 or max_err < thd)
    return passed, pct, max_err


def _resolve_impls(
    baseline: str,
    compare: Optional[List[str]],
) -> Tuple[str, List[str]]:
    if baseline not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown baseline '{baseline}', available={list(IMPLEMENTATIONS)}")

    if compare is None:
        names = list(DEFAULT_COMPARE)
    else:
        names = list(dict.fromkeys(compare))

    unknown = [n for n in names if n not in IMPLEMENTATIONS]
    if unknown:
        raise ValueError(f"Unknown implementations: {unknown}, available={list(IMPLEMENTATIONS)}")

    if baseline not in names:
        names.insert(0, baseline)
    return baseline, names


def run_single_shape(
    *,
    num_tokens: int,
    num_blocks: int,
    hidden_size: int,
    seed: int,
    baseline: str,
    compare: Optional[List[str]],
    output_root: str,
    warmup: int,
    with_perf: bool,
    with_precision: bool,
    verbose: bool,
) -> pd.DataFrame:
    baseline, impl_names = _resolve_impls(baseline, compare)
    os.makedirs(output_root, exist_ok=True)

    tensors = prepare_tensors(num_tokens, num_blocks, hidden_size, seed=seed)
    shape_label = f"T={num_tokens} N={num_blocks} H={hidden_size} B={num_blocks + 1} seed={seed}"

    print(f"\n===== attenres_lite: {shape_label} =====")
    print(f"baseline: {baseline}")
    print(f"implementations: {impl_names}")
    print(f"BF16 MERE threshold: {BF16_MERE_THRESHOLD}")
    print(f"output_root: {output_root}")

    golden = None
    golden_np = None
    if with_precision:
        ps, br, pw, nw, eps = clone_inputs(tensors)
        golden = IMPLEMENTATIONS[baseline](ps, br, pw, nw, eps)
        torch.npu.synchronize()
        golden_np = golden.float().cpu().numpy()

    rows: List[dict] = []
    perf_summaries: dict = {}

    for name in impl_names:
        row = {
            "impl": name,
            "T": num_tokens,
            "N": num_blocks,
            "H": hidden_size,
            "B": num_blocks + 1,
            "seed": seed,
            "baseline": baseline,
            "is_baseline": name == baseline,
        }

        if with_precision and golden is not None and golden_np is not None:
            ps, br, pw, nw, eps = clone_inputs(tensors)
            out = IMPLEMENTATIONS[name](ps, br, pw, nw, eps)
            torch.npu.synchronize()
            if name == baseline:
                passed, pct, max_err = True, 100.0, 0.0
                print(f"\n[{name}] baseline (skip self-compare)")
            else:
                passed, pct, max_err = _compare_precision(
                    out.float().cpu().numpy(),
                    golden_np,
                    verbose=verbose,
                )
                status = "PASS" if passed else "FAIL"
                print(f"\n[{status}] {name} vs {baseline}  MERE={max_err:.6f}  pct={pct:.4f}%")
            row.update({"precision_pass": passed, "mere": max_err, "pct": pct})
        else:
            row.update({"precision_pass": None, "mere": None, "pct": None})

        if with_perf:
            print(f"\n===== benchmark {name} =====")
            summary = benchmark_implementation(
                name=name,
                fn=IMPLEMENTATIONS[name],
                tensors=tensors,
                output_root=output_root,
                warmup=warmup,
            )
            perf_summaries[name] = summary
            row.update(
                {
                    "kernel_us": summary.get("kernel_duration_us"),
                    "op_total_us": summary.get("op_total_time_us"),
                    "profiler_dir": summary.get("profiler_dir"),
                }
            )
        else:
            row.update({"kernel_us": None, "op_total_us": None, "profiler_dir": None})

        rows.append(row)

    if with_perf and perf_summaries:
        baseline_us = perf_summaries.get(baseline, {}).get("kernel_duration_us")
        for row in rows:
            us = row.get("kernel_us")
            row["speedup_vs_baseline"] = (baseline_us / us) if baseline_us and us else None

        print("\n===== performance summary =====")
        hdr = f"{'Implementation':<28} {'Duration(us)':>14} {'Op Total(us)':>14} {'Speedup':>10}"
        print(hdr)
        print("-" * len(hdr))
        for row in rows:
            us = row["kernel_us"]
            op_us = row["op_total_us"]
            spd = row["speedup_vs_baseline"]
            us_str = "N/A" if us is None else f"{us:.3f}"
            op_str = "N/A" if op_us is None else f"{op_us:.3f}"
            spd_str = "1.000x" if row["is_baseline"] else (f"{spd:.3f}x" if spd else "N/A")
            print(f"{row['impl']:<28} {us_str:>14} {op_str:>14} {spd_str:>10}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_root, "single_shape_compare.csv")
    df.to_csv(csv_path, index=False)
    print(f"\ncsv: {csv_path}")

    if with_precision:
        compared = [r for r in rows if not r["is_baseline"]]
        failed = [r["impl"] for r in compared if not r.get("precision_pass")]
        passed = sum(1 for r in compared if r.get("precision_pass"))
        print(f"precision: {passed}/{len(compared)} pass vs {baseline}")
        if failed:
            raise SystemExit(f"precision failed: {failed}")

    return df


def build_argparser() -> argparse.ArgumentParser:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = str(SCRIPT_DIR / "profiler_output" / f"attenres_lite_{ts}")

    parser = argparse.ArgumentParser(
        description="AttnResFwd 单 shape 多实现精度比对 + 性能采集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-tokens", type=int, default=544, help="T")
    parser.add_argument("--num-blocks", type=int, default=12, help="N")
    parser.add_argument("--hidden-size", type=int, default=7168, help="H")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        choices=list(IMPLEMENTATIONS.keys()),
        help="精度 golden 基准实现",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help=f"待测实现列表，默认 {DEFAULT_COMPARE}",
    )
    parser.add_argument("--output-root", default=default_out)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-perf", action="store_true", help="仅精度，不跑 profiler")
    parser.add_argument("--no-check", action="store_true", help="仅性能，跳过精度")
    parser.add_argument("--verbose", action="store_true", help="精度比对打印 data_compare 明细")
    parser.add_argument("--list", action="store_true", help="列出可用实现并退出")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.list:
        print("Available implementations:")
        for name in IMPLEMENTATIONS:
            mark = " (default baseline)" if name == DEFAULT_BASELINE else ""
            print(f"  - {name}{mark}")
        return

    run_single_shape(
        num_tokens=args.num_tokens,
        num_blocks=args.num_blocks,
        hidden_size=args.hidden_size,
        seed=args.seed,
        baseline=args.baseline,
        compare=args.compare,
        output_root=args.output_root,
        warmup=args.warmup,
        with_perf=not args.no_perf,
        with_precision=not args.no_check,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

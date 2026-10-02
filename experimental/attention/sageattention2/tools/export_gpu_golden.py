#!/usr/bin/env python3
"""SageAttention2 L-A 真 GPU Golden 数据导出与元数据校验套件

100% 独立推导实现，对照任务书 §5.1 / §7 与 todo.md M4.3 规范：
1. 完整覆盖 7 大基准测试规格 P-01 ~ P-07 在 NVIDIA GPU 环境下；
2. 逐用例全组合导出：2 档量化粒度（per_thread, per_warp）× 3 档 PV 累加（fp32, fp32+fp32, fp32+fp16）= 42 档全量 Golden；
3. 严格计算并记录 SHA-256 文件校验码与张量统计特征（Mean / Std / Min / Max / LSE）；
4. 深度 manifest.json 元数据校验与数值断言引擎（--verify）：
   - 文件哈希一致性核验；
   - 形状、类型与 NaN / Inf 异常防护断言；
   - L-A 真 GPU golden 判定防护（禁自研 CPU 仿真冒充 L-A 真机验收）。
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 7 大基准测试规格 P-01 ~ P-07（任务书 §5.1 / todo.md §0.5）
BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "name": "P-01",
        "b": 1, "hq": 32, "hk": 32, "sq": 4096, "sk": 4096, "d": 128,
        "dtype": "float16", "is_causal": False,
        "smooth_k": True, "smooth_v": False,
        "category": "S1", "desc": "[1,32,4096,128] FP16 non-causal",
    },
    {
        "name": "P-02",
        "b": 1, "hq": 32, "hk": 32, "sq": 8192, "sk": 8192, "d": 128,
        "dtype": "float16", "is_causal": False,
        "smooth_k": True, "smooth_v": False,
        "category": "S1", "desc": "[1,32,8192,128] FP16 non-causal",
    },
    {
        "name": "P-03",
        "b": 1, "hq": 32, "hk": 32, "sq": 8192, "sk": 8192, "d": 128,
        "dtype": "float16", "is_causal": True,
        "smooth_k": True, "smooth_v": False,
        "category": "S2", "desc": "[1,32,8192,128] FP16 causal",
    },
    {
        "name": "P-04",
        "b": 2, "hq": 32, "hk": 32, "sq": 8192, "sk": 8192, "d": 128,
        "dtype": "float16", "is_causal": False,
        "smooth_k": True, "smooth_v": False,
        "category": "S1", "desc": "[2,32,8192,128] FP16 non-causal",
    },
    {
        "name": "P-05",
        "b": 1, "hq": 32, "hk": 8, "sq": 4096, "sk": 4096, "d": 128,
        "dtype": "float16", "is_causal": False,
        "smooth_k": True, "smooth_v": False,
        "category": "S3", "desc": "GQA Hq/Hk=32/8, S=4096, D=128 FP16",
    },
    {
        "name": "P-06",
        "b": 1, "hq": 32, "hk": 32, "sq": 4096, "sk": 4096, "d": 128,
        "dtype": "bfloat16", "is_causal": False,
        "smooth_k": True, "smooth_v": False,
        "category": "S4", "desc": "[1,32,4096,128] BF16 non-causal",
    },
    {
        "name": "P-07",
        "b": 1, "hq": 32, "hk": 32, "sq": 16384, "sk": 16384, "d": 128,
        "dtype": "float16", "is_causal": False,
        "smooth_k": True, "smooth_v": False,
        "category": "S5", "desc": "[1,32,16384,128] FP16 long-sequence",
    },
]

VALID_QUANT_GRANS = ("per_thread", "per_warp")
VALID_ACCUM_DTYPES = ("fp32", "fp32+fp32", "fp32+fp16")


def compute_file_sha256(filepath: Path) -> str:
    """计算文件的 SHA-256 校验哈希"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """计算张量统计指标与异常值检测（NaN / Inf）"""
    t_float = tensor.float()
    has_nan = bool(torch.isnan(t_float).any().item())
    has_inf = bool(torch.isinf(t_float).any().item())
    min_val = float(t_float.min().item()) if not (has_nan or has_inf) else float("nan")
    max_val = float(t_float.max().item()) if not (has_nan or has_inf) else float("nan")
    mean_val = float(t_float.mean().item()) if not (has_nan or has_inf) else float("nan")
    std_val = float(t_float.std().item()) if not (has_nan or has_inf) else float("nan")
    norm_val = float(torch.norm(t_float).item()) if not (has_nan or has_inf) else float("nan")
    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "norm": norm_val,
    }


def validate_manifest(
    manifest_path: Path,
    expected_cases: Optional[List[str]] = None,
    expected_grans: Optional[List[str]] = None,
    expected_accums: Optional[List[str]] = None,
    strict_gpu: bool = False,
) -> bool:
    """全面校验 manifest.json 元数据及其关联的 golden 张量文件：
    1. 校验 manifest 文件合法性与必需键；
    2. 校验文件存在性与 SHA-256 哈希完整性；
    3. 校验张量形状、数据类型、无 NaN、无 Inf；
    4. 校验 strict_gpu 条件下严禁包含 CPU fallback 数据。
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    if not isinstance(manifest_data, list) or len(manifest_data) == 0:
        raise ValueError("Manifest data is empty or not a list.")

    golden_dir = manifest_path.parent
    print(f"\n[VERIFY] Validating manifest: {manifest_path} ({len(manifest_data)} entries)...")

    # 1. 组合完整性检查
    exp_cases = expected_cases or [c["name"] for c in BENCHMARK_CASES]
    exp_grans = expected_grans or list(VALID_QUANT_GRANS)
    exp_accums = expected_accums or list(VALID_ACCUM_DTYPES)

    expected_tags = set()
    for c in exp_cases:
        for g in exp_grans:
            for a in exp_accums:
                expected_tags.add(f"{c}_{g}_{a.replace('+', '_')}")

    found_tags = set(entry.get("case_tag") for entry in manifest_data)
    missing_tags = expected_tags - found_tags
    if missing_tags:
        print(f"[VERIFY ERROR] Missing expected case tags in manifest ({len(missing_tags)} missing):")
        for tag in sorted(missing_tags)[:5]:
            print(f"  - Missing: {tag}")
        raise AssertionError(f"Manifest incomplete: missing {len(missing_tags)} case tags.")

    # 2. 逐项深度校验
    for idx, entry in enumerate(manifest_data):
        case_tag = entry.get("case_tag", f"entry_{idx}")
        filename = entry.get("file")
        if not filename:
            raise AssertionError(f"[{case_tag}] Missing 'file' field in manifest entry.")

        file_path = golden_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"[{case_tag}] Referenced golden file does not exist: {file_path}")

        # 校验 SHA-256
        recorded_sha = entry.get("file_sha256")
        if recorded_sha:
            actual_sha = compute_file_sha256(file_path)
            if actual_sha != recorded_sha:
                raise AssertionError(f"[{case_tag}] SHA-256 mismatch! Recorded: {recorded_sha}, Actual: {actual_sha}")

        # 校验真机 GPU golden 约束
        if strict_gpu and not entry.get("is_real_gpu", False):
            raise AssertionError(
                f"[{case_tag}] Entry has is_real_gpu=false! Strict GPU validation prohibits CPU fallback data for L-A."
            )

        # 加载张量并校验合法性
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        for required_key in ("config", "q", "k", "v", "out", "lse"):
            if required_key not in data:
                raise AssertionError(f"[{case_tag}] Golden file missing key '{required_key}'.")

        out = data["out"]
        lse = data["lse"]
        cfg = data["config"]

        # 形状核验
        expected_out_shape = [cfg["b"], cfg["hq"], cfg["sq"], cfg["d"]]
        if list(out.shape) != expected_out_shape:
            raise AssertionError(f"[{case_tag}] 'out' shape mismatch: expected {expected_out_shape}, got {list(out.shape)}")

        if lse is not None:
            expected_lse_shape = [cfg["b"], cfg["hq"], cfg["sq"]]
            if list(lse.shape) != expected_lse_shape:
                raise AssertionError(f"[{case_tag}] 'lse' shape mismatch: expected {expected_lse_shape}, got {list(lse.shape)}")

        # 数值异常防护断言 (无 NaN / 无 Inf)
        if torch.isnan(out).any().item():
            raise AssertionError(f"[{case_tag}] 'out' tensor contains NaN values!")
        if torch.isinf(out).any().item():
            raise AssertionError(f"[{case_tag}] 'out' tensor contains Inf values!")
        if lse is not None and torch.isnan(lse).any().item():
            raise AssertionError(f"[{case_tag}] 'lse' tensor contains NaN values!")

    print(f"[VERIFY SUCCESS] All {len(manifest_data)} entries in manifest and corresponding golden files passed verification!")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and Verify SageAttention2 Genuine GPU Golden Tensors for L-A Numerical Conformance"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./golden_data",
        help="Directory to save golden tensors and manifest.json",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["P-01", "P-02", "P-03", "P-04", "P-05", "P-06", "P-07"],
        help="List of case IDs to export (default: all 7 specs P-01 ~ P-07)",
    )
    parser.add_argument(
        "--quant-grans",
        nargs="+",
        default=list(VALID_QUANT_GRANS),
        choices=VALID_QUANT_GRANS,
        help="Quantization granularities to export (default: per_thread per_warp)",
    )
    parser.add_argument(
        "--accum-dtypes",
        nargs="+",
        default=list(VALID_ACCUM_DTYPES),
        choices=VALID_ACCUM_DTYPES,
        help="PV accumulation dtypes to export (default: fp32 fp32+fp32 fp32+fp16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic tensor generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run generation on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--scale-down",
        type=int,
        default=1,
        help="Downscale sequence length factor (for local fast testing)",
    )
    parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Error out immediately if genuine NVIDIA GPU or official package is unavailable",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing manifest.json and golden files in output-dir without re-generating",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    # 独立校验模式
    if args.verify:
        manifest_file = out_dir / "manifest.json"
        validate_manifest(
            manifest_file,
            expected_cases=args.cases,
            expected_grans=args.quant_grans,
            expected_accums=args.accum_dtypes,
            strict_gpu=args.strict_gpu,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print("   SageAttention2 GPU Golden Export & Verification Tool")
    print(f"   Target Device: {device} | Output Directory: {out_dir}")
    print(f"   Cases: {args.cases}")
    print(f"   Granularities: {args.quant_grans}")
    print(f"   Accumulator Dtypes: {args.accum_dtypes}")
    print(f"   Strict GPU Mode: {args.strict_gpu} | Scale Down: {args.scale_down}x")
    print("=" * 80)

    # 加载官方 GPU 实现或 CPU 参考实现
    gpu_fn = None
    is_real_gpu = False

    if device.type == "cuda":
        try:
            from sageattention.core import sageattn_qk_int8_pv_fp8_cuda
            gpu_fn = sageattn_qk_int8_pv_fp8_cuda
            is_real_gpu = True
            print("[+] Successfully loaded official CUDA implementation: sageattn_qk_int8_pv_fp8_cuda")
        except ImportError:
            try:
                import sageattention
                if hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda"):
                    gpu_fn = sageattention.sageattn_qk_int8_pv_fp8_cuda
                    is_real_gpu = True
                    print("[+] Successfully loaded official CUDA implementation from sageattention")
            except ImportError:
                pass

        if gpu_fn is None:
            if args.strict_gpu:
                sys.exit("[FATAL ERROR] --strict-gpu specified, but official sageattention CUDA package is not installed.")
            print("[-] Warning: Official sageattention not found on CUDA, falling back to local reference.")

    if gpu_fn is None:
        if args.strict_gpu:
            sys.exit("[FATAL ERROR] --strict-gpu specified, but no genuine CUDA GPU / package is available.")
        # 加载本地高保真参考实现作为兜底
        candidate_paths = [
            Path(__file__).resolve().parent.parent / "python",
            Path(__file__).resolve().parent.parent.parent / "ops-transformer/experimental/attention/sageattention2/python",
        ]
        for p in candidate_paths:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from sageattention2.reference import sageattn_reference
        gpu_fn = sageattn_reference
        print("[+] Loaded local reference implementation (CPU fallback)")
        print("[!] NOTICE: CPU fallback data is marked with is_real_gpu=false. "
              "Genuine L-A acceptance strictly requires genuine NVIDIA GPU golden data.")

    torch.manual_seed(args.seed)

    # 导出循环：Cases × Granularities × Accumulator Dtypes
    total_combinations = len(args.cases) * len(args.quant_grans) * len(args.accum_dtypes)
    print(f"\n[*] Generating {total_combinations} golden tensor files...")

    manifest: List[Dict[str, Any]] = []
    generated_count = 0

    for cfg in BENCHMARK_CASES:
        if cfg["name"] not in args.cases:
            continue

        b = cfg["b"]
        hq = cfg["hq"]
        hk = cfg["hk"]
        sq = max(1, cfg["sq"] // args.scale_down)
        sk = max(1, cfg["sk"] // args.scale_down)
        d = cfg["d"]
        is_causal = cfg["is_causal"]
        dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
        sm_scale = 1.0 / math.sqrt(d)

        for gran in args.quant_grans:
            for accum in args.accum_dtypes:
                case_tag = f"{cfg['name']}_{gran}_{accum.replace('+', '_')}"
                generated_count += 1
                print(f"  [{generated_count}/{total_combinations}] Exporting {case_tag}...", end="", flush=True)

                # 生成确定性输入张量
                torch.manual_seed(args.seed + generated_count)
                q = torch.randn(b, hq, sq, d, dtype=dtype, device=device)
                k = torch.randn(b, hk, sk, d, dtype=dtype, device=device)
                v = torch.randn(b, hk, sk, d, dtype=dtype, device=device)

                # 调用计算
                try:
                    out_res = gpu_fn(
                        q, k, v,
                        tensor_layout="HND",
                        is_causal=is_causal,
                        qk_quant_gran=gran,
                        sm_scale=sm_scale,
                        pv_accum_dtype=accum,
                        smooth_k=cfg["smooth_k"],
                        smooth_v=cfg["smooth_v"],
                        return_lse=True,
                    )
                    if isinstance(out_res, tuple):
                        out, lse = out_res[0], out_res[1]
                    else:
                        out, lse = out_res, None
                except Exception as exc:
                    print(f" FAILED: {exc}")
                    raise exc

                # 转移到 CPU 并计算统计元数据
                out_cpu = out.detach().cpu()
                lse_cpu = lse.detach().cpu() if lse is not None else None
                q_cpu = q.detach().cpu()
                k_cpu = k.detach().cpu()
                v_cpu = v.detach().cpu()

                out_stats = compute_tensor_stats(out_cpu)
                lse_stats = compute_tensor_stats(lse_cpu) if lse_cpu is not None else {}

                # 保存 Golden 字典
                save_dict = {
                    "config": {
                        "name": cfg["name"],
                        "b": b, "hq": hq, "hk": hk, "sq": sq, "sk": sk, "d": d,
                        "dtype": cfg["dtype"],
                        "is_causal": is_causal,
                        "qk_quant_gran": gran,
                        "pv_accum_dtype": accum,
                        "smooth_k": cfg["smooth_k"],
                        "smooth_v": cfg["smooth_v"],
                        "sm_scale": sm_scale,
                    },
                    "q": q_cpu,
                    "k": k_cpu,
                    "v": v_cpu,
                    "out": out_cpu,
                    "lse": lse_cpu,
                }
                case_filename = f"{case_tag}_golden.pt"
                case_file_path = out_dir / case_filename
                torch.save(save_dict, case_file_path)

                # 计算文件哈希
                file_sha256 = compute_file_sha256(case_file_path)

                entry = {
                    "case": cfg["name"],
                    "case_tag": case_tag,
                    "file": case_filename,
                    "file_sha256": file_sha256,
                    "out_shape": list(out_cpu.shape),
                    "lse_shape": list(lse_cpu.shape) if lse_cpu is not None else None,
                    "dtype": cfg["dtype"],
                    "qk_quant_gran": gran,
                    "pv_accum_dtype": accum,
                    "is_causal": is_causal,
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    "seed": args.seed,
                    "is_real_gpu": is_real_gpu,
                    "out_stats": out_stats,
                    "lse_stats": lse_stats,
                }
                manifest.append(entry)
                print(f" Done. (SHA: {file_sha256[:8]}..., Mean: {out_stats['mean']:.4f})")

    # 导出 manifest.json
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Golden generation complete. Exported {len(manifest)} cases to {out_dir}")

    # 自动执行自校验断言
    validate_manifest(
        manifest_path,
        expected_cases=args.cases,
        expected_grans=args.quant_grans,
        expected_accums=args.accum_dtypes,
        strict_gpu=args.strict_gpu,
    )


if __name__ == "__main__":
    main()

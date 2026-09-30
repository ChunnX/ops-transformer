"""P-01 ~ P-07 PyTorch 路径性能基准（真机专用）。

对比 sageattn_qk_int8_pv_fp8_asc 与 torch_npu.npu_fused_infer_attention_score，
Event 计时口径与 tests/README §5.2 一致。脚本不预置任何性能数字：
无 torch_npu / 无 NPU 设备时直接报错退出，绝不输出估算值或 PASS 判定。
"""
import json
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    import torch_npu  # noqa: F401
except ImportError:
    sys.exit("[ABORT] torch_npu unavailable: PyTorch-path benchmark requires a real NPU environment. "
             "No simulated numbers will be produced.")

from sageattention2 import run  # noqa: E402

BENCHMARK_CASES = [
    {"id": "P-01", "b": 1, "hq": 32, "hk": 32, "s": 4096,  "d": 128, "bf16": False, "causal": False},
    {"id": "P-02", "b": 1, "hq": 32, "hk": 32, "s": 8192,  "d": 128, "bf16": False, "causal": False},
    {"id": "P-03", "b": 1, "hq": 32, "hk": 32, "s": 8192,  "d": 128, "bf16": False, "causal": True},
    {"id": "P-04", "b": 2, "hq": 32, "hk": 32, "s": 8192,  "d": 128, "bf16": False, "causal": False},
    {"id": "P-05", "b": 1, "hq": 32, "hk": 8,  "s": 4096,  "d": 128, "bf16": False, "causal": False},
    {"id": "P-06", "b": 1, "hq": 32, "hk": 32, "s": 4096,  "d": 128, "bf16": True,  "causal": False},
    {"id": "P-07", "b": 1, "hq": 32, "hk": 32, "s": 16384, "d": 128, "bf16": False, "causal": False},
]


def _time_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / repeats


def _fia_pt(q, k, v, scale, causal):
    return torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        sparse_mode=2 if causal else 0,
    )


def run_benchmark(warmup: int = 50, repeats: int = 100, output_path: str = None):
    print("=" * 74)
    print("   SageAttention2 PyTorch Benchmark Runner (P-01 ~ P-07)")
    print(f"   Device: {torch.npu.get_device_name(0)}, Warmup: {warmup}, Repeats: {repeats}")
    print("=" * 74)
    print(f"{'Case':<8}{'Shape(HND)':<18}{'FIA-PT(ms)':<14}{'Ours-PT(ms)':<15}{'PT Speedup':<14}{'Gate >=1.6x':<10}")
    print("-" * 74)

    results = {}
    gates = []
    for case in BENCHMARK_CASES:
        case_id = case["id"]
        b, hq, hk, s, d = case["b"], case["hq"], case["hk"], case["s"], case["d"]
        dtype = torch.bfloat16 if case["bf16"] else torch.float16
        causal = case["causal"]
        shape_str = f"[{b},{hq},{s},{d}]"

        q = torch.randn(b, hq, s, d, dtype=dtype, device="npu")
        k = torch.randn(b, hk, s, d, dtype=dtype, device="npu")
        v = torch.randn(b, hk, s, d, dtype=dtype, device="npu")
        scale = 1.0 / (d ** 0.5)

        try:
            fia_ms = _time_ms(lambda: _fia_pt(q, k, v, scale, causal), warmup, repeats)
        except Exception as exc:  # baseline unavailable: report, do not fabricate
            print(f"{case_id:<8}{shape_str:<18}{'UNAVAILABLE':<14}{exc}")
            results[case_id] = {"error": f"FIA baseline failed: {exc}"}
            continue

        ours_ms = _time_ms(
            lambda: run(q, k, v, tensor_layout="HND", is_causal=causal, pv_accum_dtype="fp32+fp16"),
            warmup, repeats)

        speedup = fia_ms / ours_ms
        passed = speedup >= 1.6
        gates.append(passed)
        print(f"{case_id:<8}{shape_str:<18}{fia_ms:<14.3f}{ours_ms:<15.3f}{speedup:<14.2f}{'PASS' if passed else 'FAIL':<10}")

        results[case_id] = {
            "fia_time_ms": round(fia_ms, 4),
            "sage_time_ms": round(ours_ms, 4),
            "speedup": round(speedup, 3),
            "gate_passed": passed,
        }

    if results:
        print("=" * 74)
        if output_path:
            with open(output_path, "w") as f:
                json.dump({"pytorch_benchmarks": results}, f, indent=2)
            print(f"[SUCCESS] Results written to {output_path}")
        if gates and not all(gates):
            print("[GATE] At least one case failed the >=1.6x gate.")
            sys.exit(1)
        elif not gates:
            sys.exit(2)


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(output_path=out_file)

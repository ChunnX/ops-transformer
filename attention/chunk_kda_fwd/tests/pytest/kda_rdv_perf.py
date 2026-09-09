"""将 CGDR RDV 参数映射到 CANN 内置 KDA，独立采集性能或校验精度。"""
from __future__ import annotations

import argparse
import csv
import gc
import faulthandler
import hashlib
import json
import os
import signal
from pathlib import Path
import time

import torch
import torch_npu

import cann_ops_transformer
from cann_ops_transformer.ops import chunk_kda_fwd

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CASES = REPO / "attention/chunk_gated_delta_rule/docs/rdv_perf/new_streaming/comparison.csv"
SEED = 20260908


def read_cases(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def lengths_of(case):
    value = json.loads(case["seqlen"])
    return value if isinstance(value, list) else [value] * int(case["B"])


def unsupported_reason(case):
    dk, dv = int(case["dk"]), int(case["dv"])
    if min(dk, dv) < 16 or max(dk, dv) > 256 or dk % 16 or dv % 16:
        return "CANN KDA requires Dk/Dv in [16, 256], multiples of 16; no padding applied"
    return ""


def make_inputs(case, pad_generic_tail=False, pad_sequence_tails=False):
    b, nk, nv, dk, dv = (int(case[k]) for k in ["B", "nk", "nv", "dk", "dv"])
    lengths = lengths_of(case)
    t = sum(lengths)
    torch.manual_seed(SEED)
    torch.npu.manual_seed(SEED)
    dtype = torch.bfloat16
    source_state_dtype = getattr(torch, case["state_dtype"])
    q = torch.rand((t, nk, dk), dtype=dtype, device="npu")
    k = torch.rand((t, nk, dk), dtype=dtype, device="npu")
    v = torch.rand((t, nv, dv), dtype=dtype, device="npu")
    # 保持 CGDR 生成 beta/state 前的随机数消费顺序；KDA 门控使用另一固定 seed。
    if case["has_g"] == "True":
        scalar_gate = -torch.rand((t, nv), dtype=torch.float32, device="npu")
        del scalar_gate
    beta = torch.rand((t, nv), dtype=dtype, device="npu")
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    state = torch.rand((b, nv, dv, dk), dtype=source_state_dtype, device="npu").float()
    if case["contiguous"] == "False":
        padded = torch.zeros((b, nv, dv + 1, dk), dtype=torch.float32, device="npu")
        padded[:, :, :dv, :] = state
        state = padded[:, :, :dv, :]
    torch.npu.manual_seed(SEED + 1)
    g = (-torch.rand((t, nv, dk), dtype=torch.float32, device="npu")
         if case["has_g"] == "True" else torch.zeros((t, nv, dk), dtype=torch.float32, device="npu"))
    a_log = None
    if case["has_g"] == "True" and (dk, dv) == (128, 128):
        # 将目标 log-gate 逆变换成 safe-gate 的 raw 输入；在计时外完成。
        # 零点用最小正概率替代，避免传入 -Inf；实际 gate 由同一公式进入参考计算。
        probability = (-g / 5.0).clamp_min(2 ** -24)
        g = probability.log() - torch.log1p(-probability)
        a_log = torch.zeros(nv, dtype=torch.float32, device="npu")
    if (pad_sequence_tails or (pad_generic_tail and (dk, dv) != (128, 128))) and any(x % 64 for x in lengths):
        assert len(set(lengths)) == 1
        length = lengths[0]
        padded_length = (length + 63) // 64 * 64
        values = []
        for index, value in enumerate((q, k, v, g, beta)):
            # 中性补齐：q/k/v/beta 为 0；raw g=-100 或已激活 g=0，使衰减在 FP32 中为 1。
            fill = -100.0 if index == 3 and a_log is not None else 0.0
            padded_value = torch.full((b, padded_length, *value.shape[1:]), fill,
                                      dtype=value.dtype, device=value.device)
            padded_value[:, :length] = value.view(b, length, *value.shape[1:])
            values.append(padded_value.flatten(0, 1))
        q, k, v, g, beta = values
        lengths = [padded_length] * b
    torch.npu.synchronize()
    assert state.is_contiguous() == (case["contiguous"] == "True")
    return q, k, v, g, beta, state, lengths, dk ** -0.5, a_log


def reference(inputs):
    """CPU FP64 逐 token 递推，不调用被测 KDA 或 CGDR 实现。"""
    q, k, v, g, beta, initial, lengths, scale, a_log = inputs
    q, k, v, g, beta, initial = [x.cpu().double() for x in (q, k, v, g, beta, initial)]
    if a_log is not None:
        g = -5.0 * torch.sigmoid(g)
    head_map = torch.arange(v.shape[1]) // (v.shape[1] // q.shape[1])
    outputs = torch.empty_like(v)
    finals = torch.empty_like(initial)
    offset = 0
    for bid, length in enumerate(lengths):
        state = initial[bid].clone()
        for pos in range(offset, offset + length):
            key = k[pos, head_map]
            query = q[pos, head_map] * scale
            state = state * g[pos].exp().unsqueeze(1)
            prediction = torch.einsum("hvk,hk->hv", state, key)
            delta = beta[pos, :, None] * (v[pos] - prediction)
            state = state + delta.unsqueeze(-1) * key.unsqueeze(1)
            outputs[pos] = torch.einsum("hvk,hk->hv", state, query)
        finals[bid] = state
        offset += length
    return outputs, finals


def precision_metrics(actual, expected):
    """沿用 CGDR 泛化验证的 BF16 门限，分块归约以控制长序列 CPU 内存。"""
    actual = actual.cpu().float()
    assert actual.shape == expected.shape
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    block = 1 << 20
    sum_error2 = sum_reference2 = max_abs = legacy_ratio = 0.0
    finite = True
    for offset in range(0, actual.numel(), block):
        a, e = actual[offset:offset + block].double(), expected[offset:offset + block]
        error = (a - e).abs()
        finite = finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(e).all())
        sum_error2 += error.square().sum().item()
        sum_reference2 += e.square().sum().item()
        max_abs = max(max_abs, error.max().item())
        legacy_ratio = max(legacy_ratio, (error / (0.002 + 0.02 * e.abs())).max().item())
    rms = (sum_reference2 / actual.numel()) ** 0.5
    rmse = (sum_error2 / actual.numel()) ** 0.5
    relative_rms, normalized_max = rmse / max(rms, 2e-5), 0.0
    for offset in range(0, actual.numel(), block):
        a, e = actual[offset:offset + block].double(), expected[offset:offset + block]
        error = (a - e).abs()
        normalized_max = max(normalized_max, (error / e.abs().clamp_min(max(rms, 2e-5))).max().item())
    return {"passed": finite and relative_rms <= 0.02 and normalized_max <= 0.10,
            "relative_rms": relative_rms, "normalized_max": normalized_max,
            "relative_rms_limit": 0.02, "normalized_max_limit": 0.10, "absolute_floor": 2e-5,
            "max_abs_error": max_abs, "rmse": rmse,
            "legacy_allclose_passed": legacy_ratio <= 1.0, "legacy_max_tolerance_ratio": legacy_ratio}


class CpuReference:
    def __init__(self):
        from torch.utils.cpp_extension import load
        self.module = load(name="kda_cpu_reference", sources=[str(Path(__file__).with_name("kda_cpu_reference.cpp"))],
            extra_cflags=["-O3", "-fopenmp", "-fno-math-errno", "-march=native"], extra_ldflags=["-fopenmp"], verbose=False)
        self.previous = None

    def __call__(self, inputs, name):
        tensors = [x.detach().cpu().float().contiguous() for x in inputs[:6]]
        options = (inputs[6], inputs[7], inputs[8] is not None)
        reused = ""
        if self.previous is not None:
            old_name, old_tensors, old_options, expected = self.previous
            if options == old_options and all(torch.equal(a, b) for a, b in zip(tensors, old_tensors)):
                reused = old_name
            else:
                self.previous = None
                del old_tensors, expected
        if not reused:
            expected = self.module.reference(*tensors, *options)
        self.previous = (name, tensors, options, expected)
        return expected, reused


class KdaTorchExtension:
    """通过已安装的扩展调用；profiler 分别统计主 kernel 和 API 内的辅助设备任务。"""
    def __call__(self, q, k, v, g, beta, state, lengths, scale, a_log, outputs=None, layout="auto"):
        # CANN dense 128x128 在同核轮询多个 batch/head 且有初态时可能损坏最终状态。
        # 使用公共 TND + host cu_seqlens 路径，保持 tokens、head 和 chunk_size 均不变。
        multi_stream = q.shape[-1] == v.shape[-1] == 128 and len(lengths) * v.shape[-2] > 32
        dense = layout == "auto" and len(set(lengths)) == 1 and not multi_stream
        self.effective_layout = "BSND" if dense else "TND"
        if dense:
            q, k, v, g, beta = [x.view(len(lengths), lengths[0], *x.shape[1:])
                                for x in (q, k, v, g, beta)]
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        out = chunk_kda_fwd(q, k, v, g, beta, scale, initial_state=state,
            output_final_state=True, chunk_size=64, layout="BSND" if dense else "TND",
            cu_seqlens_cpu=None if dense else cumulative, state_v_first=True,
            safe_gate=(q.shape[-1], v.shape[-1]) == (128, 128), lower_bound=-5.0,
            use_gate_in_kernel=a_log is not None, A_log=a_log)
        torch.npu.synchronize()
        # attention 统一回到 RDV 的 TND 形状；其余输出保持公共接口定义。
        return [out[0].reshape(-1, v.shape[-2], v.shape[-1]), out[1], out[3], out[4]], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", default="")
    parser.add_argument("--mode", choices=["perf", "precision"], default="perf")
    parser.add_argument("--layout", choices=["auto", "TND"], default="auto")
    parser.add_argument("--pad-sequence-tails", action="store_true",
                        help="所有尾块均中性补齐到 64 token，避免尾块数值异常")
    parser.add_argument("--pad-generic-tail", action="store_true",
                        help="通用尾块补齐到 64 token，记录有效 shape；不改变原始 RDV 元数据")
    parser.add_argument("--defer-generic-tail", action="store_true",
                        help="非128维且存在尾chunk的用例留给独立进程复核，避免阻断全量采集")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    faulthandler.register(signal.SIGUSR1)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    cases = read_cases(args.cases)
    if args.names:
        names = set(args.names.split(","))
        cases = [case for case in cases if case["test_name"] in names]
        assert len(cases) == len(names)
    if args.mode == "precision":
        cases.sort(key=lambda case: (case["test_name"].replace("-NC", ""), case["contiguous"] == "False"))
    runtime = {"mode": args.mode, "warmup": args.warmup, "measured": args.measured,
        "seed": SEED, "per_dim_gate_seed": SEED + 1, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "device": str(torch.npu.get_device_properties(0)),
        "case_source": str(args.cases.resolve()), "case_source_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH"),
        "ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "case_count": len(cases), "pad_generic_tail": args.pad_generic_tail, "pad_sequence_tails": args.pad_sequence_tails,
        "layout_policy": "TND with host cu_seqlens for 128x128 and B*Nv>32; BSND otherwise", "pid": os.getpid(), "layout": args.layout,
        "state_v_first": True, "state_dtype": "float32", "use_gate_in_kernel": "has_g and Dk=Dv=128", "gate_mapping": "inverse safe sigmoid, A_log=0, lower_bound=-5; zero gate preactivated",
        "safe_gate": "Dk=Dv=128", "python_interface": "cann_ops_transformer.ops.chunk_kda_fwd",
        "extension_package": cann_ops_transformer.__file__, "required_outputs": ["attention", "final_state", "Aqk", "Akk"]}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (args.output / "case_manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    operation = KdaTorchExtension()
    cpu_reference = CpuReference() if args.mode == "precision" else None
    results = []
    for case in cases:
        name = case["test_name"]
        result = {"test_name": name, "case_index": case["case_index"], "mode": args.mode,
                  "precision_checked": False, "status": "PENDING", "error": ""}
        (args.output / "current_case.json").write_text(json.dumps(result) + "\n")
        reason = unsupported_reason(case)
        if reason:
            result.update(status="UNSUPPORTED_SHAPE", error=reason)
        elif (args.defer_generic_tail and (int(case["dk"]), int(case["dv"])) != (128, 128)
              and any(length % 64 for length in lengths_of(case))):
            result.update(status="DEFERRED_RUNTIME_PROBE", error="Generic tail path is tested in isolated processes")
        else:
            started = time.time()
            try:
                inputs = make_inputs(case, args.pad_generic_tail, args.pad_sequence_tails)
                output = None
                count = args.warmup + args.measured if args.mode == "perf" else 1
                for _ in range(count):
                    output, workspace_bytes = operation(*inputs, outputs=output, layout=args.layout)
                finite = [bool(torch.isfinite(x).all()) for x in output]
                if not all(finite):
                    raise AssertionError(f"nonfinite output: {finite}")
                result.update(status="PASSED", invocations=count, workspace_bytes=workspace_bytes,
                    effective_lengths=inputs[6], sequence_padding=inputs[6] != lengths_of(case),
                    effective_layout=operation.effective_layout,
                    output_shapes=[list(x.shape) for x in output],
                    input_shapes=[list(x.shape) for x in inputs[:6]],
                    input_strides=[list(x.stride()) for x in inputs[:6]], outputs_finite=True)
                if args.mode == "precision":
                    expected, reused = cpu_reference(inputs, name)
                    result["reference_reused_from"] = reused
                    if result["sequence_padding"]:
                        host_inputs = cpu_reference.previous[1]
                        b, padded_length, original_length = len(inputs[6]), inputs[6][0], lengths_of(case)[0]
                        original_tensors = [x.view(b, padded_length, *x.shape[1:])[:, :original_length]
                                            .reshape(b * original_length, *x.shape[1:]).contiguous()
                                            for x in host_inputs[:5]] + [host_inputs[5]]
                        original_expected = cpu_reference.module.reference(*original_tensors,
                            lengths_of(case), inputs[7], inputs[8] is not None)
                        unpadded_expected = expected[0].view(b, padded_length, *expected[0].shape[1:])[:, :original_length]
                        torch.testing.assert_close(unpadded_expected.flatten(0, 1), original_expected[0], rtol=1e-12, atol=1e-12)
                        torch.testing.assert_close(expected[1], original_expected[1], rtol=1e-12, atol=1e-12)
                        result["padding_reference_checked"] = True
                        del original_tensors, original_expected, unpadded_expected, host_inputs
                    metrics = [precision_metrics(a, e) for a, e in zip(output[:2], expected)]
                    result.update(precision_checked=True, precision_metrics=metrics)
                    del expected
                    if not all(x["passed"] for x in metrics):
                        result["status"] = "PRECISION_FAILED"
                del output, inputs
                gc.collect()
                torch.npu.empty_cache()
            except Exception as exc:
                result.update(status="FAILED", error=str(exc))
                raise
            finally:
                result["elapsed_wall_seconds"] = time.time() - started
                if result["status"] == "FAILED":
                    with (args.output / "progress.jsonl").open("a") as f:
                        f.write(json.dumps(result) + "\n")
        results.append(result)
        with (args.output / "progress.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"KDA_PROGRESS {len(results)}/{len(cases)} {name} {result['status']}", flush=True)
        if len(results) == 1:
            libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                if any(s in line for s in ("/vendors/", "libopapi", "liboptiling", "chunk_kda_fwd.so"))})
            (args.output / "loaded_libraries.txt").write_text("\n".join(libraries) + "\n")
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    success = all(x["status"] in ["PASSED", "UNSUPPORTED_SHAPE", "DEFERRED_RUNTIME_PROBE"] for x in results)
    (args.output / "runner_exit_code.txt").write_text("0\n" if success else "1\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

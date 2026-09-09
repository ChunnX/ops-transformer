"""通过 cannbot-dsl 公共 flash_kda 入口采集 RDV，保留输入适配和实际精度状态。"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[4]
CASES = REPO / "attention/chunk_gated_delta_rule/docs/rdv_perf/new_streaming/comparison.csv"
DSL_REPO = Path("/home/s00940320/cannbot-dsl")
SEED = 20260908
MARKER_BASE = 1000003


def read_cases():
    with CASES.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def unsupported(case):
    if (int(case["dk"]), int(case["dv"])) != (128, 128):
        return "flash_kda requires Dk=Dv=128; dimension padding is not applied"
    if int(case["nv"]) % int(case["nk"]):
        return "flash_kda requires Nv % Nk == 0"
    return ""


def make_inputs(case, layout):
    import torch
    b, s, nk, nv = [int(case[k]) for k in ("B", "seqlen", "nk", "nv")]
    padded_s = (s + 63) // 64 * 64
    torch.manual_seed(SEED)
    torch.npu.manual_seed(SEED)
    q = torch.rand((b * s, nk, 128), dtype=torch.bfloat16, device="npu")
    k = torch.rand_like(q)
    v = torch.rand((b * s, nv, 128), dtype=torch.bfloat16, device="npu")
    if case["has_g"] == "True":
        # 对齐既有 RDV 的随机数消费顺序；门控使用独立 seed。
        scalar_gate = -torch.rand((b * s, nv), device="npu")
        del scalar_gate
    beta = torch.rand((b * s, nv), dtype=torch.bfloat16, device="npu")
    q, k = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in (q, k)]
    state = torch.rand((b, nv, 128, 128), dtype=getattr(torch, case["state_dtype"]), device="npu").float()
    if case["contiguous"] == "False":
        storage = torch.zeros((b, nv, 129, 128), dtype=torch.float32, device="npu")
        storage[:, :, :128] = state
        state = storage[:, :, :128]
    source_stride = list(state.stride())
    # 公共接口要求连续；NC 来源先真实构造，再在计时范围外转换，不能当作原生 NC 支持。
    state = state.contiguous()
    torch.npu.manual_seed(SEED + 1)
    if case["has_g"] == "True":
        probability = (torch.rand((b * s, nv, 128), device="npu") / 5).clamp_min(2 ** -24)
        g = (probability.log() - torch.log1p(-probability)).bfloat16()
        lower_bound = -5.0
    else:
        g = torch.zeros((b * s, nv, 128), dtype=torch.bfloat16, device="npu")
        lower_bound = 0.0
    # FlashKDA 在内部对 beta 执行 sigmoid，先逆变换，随后 BF16 量化；参考计算使用实际 logits。
    probability = beta.float()
    beta = torch.where(probability == 0, -100.0,
                       probability.log() - torch.log1p(-probability)).bfloat16()
    values = []
    for index, x in enumerate((q, k, v, g, beta)):
        x = x.view(b, s, *x.shape[1:])
        if padded_s != s:
            fill = -100.0 if index in (3, 4) else 0.0
            padded = torch.full((b, padded_s, *x.shape[2:]), fill, dtype=x.dtype, device=x.device)
            padded[:, :s] = x
            x = padded
        if layout == "BNSD":
            x = x.transpose(1, 2).contiguous()
        values.append(x)
    a_log = torch.zeros(nv, dtype=torch.float32, device="npu")
    dt_bias = torch.zeros((nv, 128), dtype=torch.float32, device="npu")
    q, k, v, g, beta = values
    inputs = (q, k, v, g, beta, 128 ** -0.5, state, a_log, dt_bias, lower_bound, layout)
    metadata = dict(effective_seqlen=padded_s, sequence_padding=padded_s != s,
                    source_state_stride=source_stride, state_contiguous_copy=case["contiguous"] == "False",
                    input_shapes=[list(x.shape) for x in (*values, state, a_log, dt_bias)],
                    input_strides=[list(x.stride()) for x in (*values, state, a_log, dt_bias)],
                    input_dtypes=[str(x.dtype) for x in (*values, state, a_log, dt_bias)],
                    lower_bound=lower_bound, layout=layout)
    return inputs, metadata


def precision_metrics(actual, expected):
    import torch
    a, e = actual.detach().cpu().float().flatten(), expected.flatten()
    assert a.shape == e.shape
    block = 1 << 20
    error2 = reference2 = max_error = allclose_ratio = 0.0
    finite = True
    for offset in range(0, a.numel(), block):
        x, y = a[offset:offset + block].double(), e[offset:offset + block]
        error = (x - y).abs()
        finite = finite and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        error2 += error.square().sum().item()
        reference2 += y.square().sum().item()
        max_error = max(max_error, error.max().item())
        allclose_ratio = max(allclose_ratio, (error / (0.005 + 0.005 * y.abs())).max().item())
    rms = (reference2 / a.numel()) ** 0.5
    relative_rms = (error2 / a.numel()) ** 0.5 / max(rms, 2e-5)
    normalized_max = 0.0
    for offset in range(0, a.numel(), block):
        x, y = a[offset:offset + block].double(), e[offset:offset + block]
        normalized_max = max(normalized_max, ((x - y).abs() / y.abs().clamp_min(max(rms, 2e-5))).max().item())
    return dict(passed=finite and relative_rms <= 0.02 and normalized_max <= 0.10,
                finite=finite, relative_rms=relative_rms, normalized_max=normalized_max,
                max_abs_error=max_error, relative_rms_limit=0.02, normalized_max_limit=0.10,
                allclose_005_passed=finite and allclose_ratio <= 1.0,
                allclose_005_max_ratio=allclose_ratio)


class Reference:
    def __init__(self):
        from torch.utils.cpp_extension import load
        self.module = load(name="flash_kda_cpu_reference",
                           sources=[str(Path(__file__).with_name("kda_cpu_reference.cpp"))],
                           extra_cflags=["-O3", "-fopenmp", "-fno-math-errno", "-march=native"],
                           extra_ldflags=["-fopenmp"], verbose=False)
        self.previous = None

    def __call__(self, inputs, case):
        import torch
        q, k, v, g, beta, scale, state, _, _, lower_bound, layout = inputs
        values = [x.detach().cpu() for x in (q, k, v, g, beta)]
        if layout == "BNSD":
            values = [x.transpose(1, 2).contiguous() for x in values]
        q, k, v, g, beta = values
        b, s = q.shape[:2]
        # 对实际 BF16 输入执行公开接口定义的归一化/激活；递推用已有独立 CPU FP64 实现。
        q, k = [(x.float() / (x.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().float()
                for x in (q, k)]
        g = lower_bound * torch.sigmoid(g.float())
        beta = torch.sigmoid(beta.float())
        tensors = [x.flatten(0, 1).contiguous().float() for x in (q, k, v, g, beta)]
        tensors.append(state.cpu().float().contiguous())
        reused = ""
        if self.previous is not None:
            old_name, old_inputs, old_s, golden = self.previous
            if s == old_s and all(torch.equal(x, y) for x, y in zip(tensors, old_inputs)):
                reused = old_name
            else:
                self.previous = None
                del old_inputs, golden
        if not reused:
            golden = self.module.reference(*tensors, [s] * b, scale, False)
        self.previous = (case["test_name"], tensors, s, golden)
        padding_checked = False
        original_s = int(case["seqlen"])
        if original_s != s:
            original = [x.view(b, s, *x.shape[1:])[:, :original_s].flatten(0, 1).contiguous()
                        for x in tensors[:5]] + [tensors[5]]
            original_golden = self.module.reference(*original, [original_s] * b, scale, False)
            valid = golden[0].view(b, s, *golden[0].shape[1:])[:, :original_s].flatten(0, 1)
            torch.testing.assert_close(valid, original_golden[0], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(golden[1], original_golden[1], rtol=1e-12, atol=1e-12)
            padding_checked = True
        return golden, reused, padding_checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsl-repo", type=Path, default=DSL_REPO)
    parser.add_argument("--names", default="")
    parser.add_argument("--mode", choices=("manifest", "perf", "precision"), default="perf")
    parser.add_argument("--layout", choices=("BSND", "BNSD"), default="BSND")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    for index, case in enumerate(cases):
        case = dict(case, marker_elements=MARKER_BASE + index)
        cases[index] = case
    if args.names:
        names = set(args.names.split(","))
        cases = [x for x in cases if x["test_name"] in names]
        assert len(cases) == len(names)
    if args.mode == "precision":
        cases.sort(key=lambda x: (x["test_name"].replace("-NC", ""), x["contiguous"] == "False"))
    write_json(args.output / "case_manifest.json", cases)
    if args.mode == "manifest":
        write_json(args.output / "support_manifest.json", [dict(test_name=x["test_name"],
            eligible=not unsupported(x), reason=unsupported(x),
            effective_seqlen=(int(x["seqlen"]) + 63) // 64 * 64,
            state_contiguous_copy=x["contiguous"] == "False") for x in cases])
        return 0
    import torch
    import torch_npu
    import cannbotdsl
    sys.path.insert(0, str(args.dsl_repo / "samples/flash_kda"))
    import flash_kda as flash
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    faulthandler.register(signal.SIGUSR1)
    runtime = dict(mode=args.mode, python=sys.version, executable=sys.executable, torch=torch.__version__,
        torch_npu=torch_npu.__version__, cannbotdsl=importlib.metadata.version("cannbot-dsl"),
        cannbotdsl_module=cannbotdsl.__file__, flash_source=flash.__file__,
        flash_source_sha256=hashlib.sha256(Path(flash.__file__).read_bytes()).hexdigest(),
        case_source=str(CASES), case_source_sha256=hashlib.sha256(CASES.read_bytes()).hexdigest(),
        seed=SEED, gate_seed=SEED + 1, device=str(torch.npu.get_device_properties(0)),
        layout=args.layout, warmup=1, measured=5, compile_prerun=1, marker_base=MARKER_BASE,
        KDA_GROUP=os.environ.get("KDA_GROUP"), KDA_DV_BASE=os.environ.get("KDA_DV_BASE"),
        CANNBOTDSL_CACHE_DIR=os.environ.get("CANNBOTDSL_CACHE_DIR"),
        ASCEND_HOME_PATH=os.environ.get("ASCEND_HOME_PATH"), pid=os.getpid())
    assert not runtime["KDA_GROUP"] and not runtime["KDA_DV_BASE"], "基准使用样例默认 tiling"
    write_json(args.output / "runtime.json", runtime)
    reference = Reference() if args.mode == "precision" else None
    results = []
    verified_output = None
    for case in cases:
        name = case["test_name"]
        result = dict(test_name=name, case_index=case["case_index"], status="PENDING", error="",
                      mode=args.mode, precision_checked=False, marker_elements=case["marker_elements"])
        write_json(args.output / "current_case.json", result)
        if unsupported(case):
            result.update(status="UNSUPPORTED_SHAPE", error=unsupported(case))
        else:
            started = time.time()
            try:
                inputs, metadata = make_inputs(case, args.layout)
                result.update(metadata)
                b, nv = int(case["B"]), int(case["nv"])
                group = flash.get_group_config_bn(b * nv)
                result.update(group=group, dv_base=flash.get_dv_base_config(b * nv, metadata["effective_seqlen"]),
                              block_num=flash._device_block_num(inputs[0]))
                state_before = inputs[6].clone()
                # 首次 JIT/常量初始化独立执行，正式统计由两次独特长度 Add 标记包围。
                out = flash.flash_kda(*inputs)
                torch.npu.synchronize()
                if args.mode == "perf":
                    marker = torch.ones(case["marker_elements"], dtype=torch.float32, device="npu")
                    marker_out = torch.empty_like(marker)
                    torch.add(marker, marker, out=marker_out)
                    torch.npu.synchronize()
                    for _ in range(6):
                        out = flash.flash_kda(*inputs)
                        torch.npu.synchronize()
                    torch.add(marker, marker, out=marker_out)
                    torch.npu.synchronize()
                    del marker, marker_out
                assert all(bool(torch.isfinite(x).all()) for x in out), "nonfinite output"
                assert torch.equal(inputs[6], state_before), "initial_state was modified"
                result.update(status="PASSED", outputs_finite=True, initial_state_unchanged=True,
                              output_shapes=[list(x.shape) for x in out])
                if reference is not None:
                    # NC 在接口外转换后通常与连续变体完全相同。仅当全部实际输入、输出均逐位
                    # 相等时，复用已经完成 FP64 校验的误差；每条用例仍独立执行 NPU 算子。
                    input_tensors = (*inputs[:5], *inputs[6:9])
                    options = (inputs[5], inputs[9], inputs[10], int(case["seqlen"]))
                    same = (verified_output is not None and options == verified_output[2]
                            and all(torch.equal(x, y) for x, y in zip(input_tensors, verified_output[1]))
                            and all(torch.equal(x, y) for x, y in zip(out, verified_output[3])))
                    if same:
                        reused = verified_output[0]
                        metrics = [dict(x) for x in verified_output[4]]
                        padding_checked = verified_output[5]
                        method = "bitwise_equal_inputs_and_outputs_to_verified_case"
                    else:
                        verified_output = None
                        golden, reused, padding_checked = reference(inputs, case)
                        attention = out[0].transpose(1, 2) if args.layout == "BNSD" else out[0]
                        metrics = [precision_metrics(x, y) for x, y in zip((attention.flatten(0, 1), out[1]), golden)]
                        method = "cpu_fp64_recurrence"
                        del golden, attention
                    result.update(precision_checked=True, precision_metrics=metrics,
                                  precision_method=method, reference_reused_from=reused,
                                  padding_reference_checked=padding_checked)
                    verified_output = (name, input_tensors, options, out, metrics, padding_checked)
                    if not all(x["passed"] for x in metrics):
                        result["status"] = "PRECISION_FAILED"
                del inputs, out, state_before
                gc.collect()
                torch.npu.empty_cache()
            except Exception as exc:
                result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            finally:
                result["elapsed_wall_seconds"] = time.time() - started
        results.append(result)
        write_json(args.output / "results.json", results)
        with (args.output / "progress.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"FLASH_KDA_PROGRESS {len(results)}/{len(cases)} {name} {result['status']}", flush=True)
        if result["status"] == "FAILED":
            print(result["error"], flush=True)
            break
    success = len(results) == len(cases) and all(x["status"] in ("PASSED", "UNSUPPORTED_SHAPE") for x in results)
    (args.output / "runner_exit_code.txt").write_text("0\n" if success else "1\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

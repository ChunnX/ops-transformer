# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Original-arena equivalence and full-output numerical acceptance for FlashKDA."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ATOL = 5e-3
RTOL = 5e-3
ORIGINAL_SHA256 = "6dc4c436098784b314962c37a94edeb4cd16dbe9e61dbb2d02a03e8e659b996e"
MAIN_CASES = (
    ((1, 1, 1, 64), "BNSD", 0.0, 2.0),
    ((1, 1, 1, 8192), "BSND", -1.0, 1.01),
    ((2, 1, 1, 64), "BSND", 0.0, 1.25),
    ((1, 3, 3, 8192), "BNSD", -1.0, 128 ** -0.5),
    ((3, 2, 2, 128), "BNSD", -5.0, 1.25),
    ((2, 4, 8, 256), "BSND", 0.0, 0.5),
    ((3, 4, 8, 128), "BNSD", -1.0, 128 ** -0.5),
    ((1, 96, 96, 64), "BSND", -1.0, 128 ** -0.5),
)
PROBE_CASES = (
    ((1, 1, 1, 64), "BNSD", 0.0, 2.0),
    ((1, 1, 1, 128), "BSND", -5.0, 1.01),
)


def make_inputs(batch, key_heads, value_heads, sequence, seed, layout):
    generator = torch.Generator().manual_seed(seed)

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator).to(dtype)

    values = {
        "q": randn((batch, key_heads, sequence, 128)),
        "k": randn((batch, key_heads, sequence, 128)),
        "v": randn((batch, value_heads, sequence, 128)),
        "g": randn((batch, value_heads, sequence, 128)),
        "beta": randn((batch, value_heads, sequence)),
        "initial_state": randn((batch, value_heads, 128, 128), torch.float32) * 0.1,
        "A_log": torch.linspace(-1.0, 0.5, value_heads),
        "dt_bias": randn((value_heads, 128), torch.float32),
        "scale": 128 ** -0.5,
        "lower_bound": -1.0,
    }
    if layout == "BSND":
        for name in ("q", "k", "v", "g"):
            values[name] = values[name].transpose(1, 2).contiguous()
        values["beta"] = values["beta"].transpose(1, 2).contiguous()
    elif layout != "BNSD":
        raise ValueError(f"unsupported layout: {layout}")
    return values


def case_label(case, layout, lower_bound, scale):
    return ("B%d_Nk%d_Nv%d_S%d" % case) + f"_{layout}_gate{lower_bound:g}_scale{scale:g}"


def reference(inputs, layout):
    """Independent token recurrence; no DSL source or chunk implementation."""
    def bnsd(value):
        return value.transpose(1, 2) if layout == "BSND" else value

    q_input, k_input = bnsd(inputs["q"]), bnsd(inputs["k"])
    v_input, g_input = bnsd(inputs["v"]), bnsd(inputs["g"])
    beta_input = bnsd(inputs["beta"])
    heads = v_input.shape[1]
    repeats = heads // k_input.shape[1]
    q = F.normalize(q_input.float(), p=2, dim=-1).repeat_interleave(repeats, dim=1)
    k = F.normalize(k_input.float(), p=2, dim=-1).repeat_interleave(repeats, dim=1)
    gate = inputs["lower_bound"] * torch.sigmoid(
        inputs["A_log"].exp()[None, :, None, None]
        * (g_input.float() + inputs["dt_bias"][None, :, None, :])
    )
    beta = beta_input.float().sigmoid()
    state = inputs["initial_state"].transpose(-1, -2).contiguous()
    output = torch.empty_like(v_input, dtype=torch.float32)
    for token in range(q.shape[2]):
        key = k[:, :, token]
        state = state * gate[:, :, token].exp().unsqueeze(-1)
        delta = (
            v_input[:, :, token].float()
            - (key.unsqueeze(-2) @ state).squeeze(-2)
        ) * beta[:, :, token, None]
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, token] = (
            q[:, :, token].unsqueeze(-2) @ state
        ).squeeze(-2) * inputs["scale"]
    if layout == "BSND":
        output = output.transpose(1, 2).contiguous()
    return output, state.transpose(-1, -2).contiguous()


def error_stats(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "elements": actual.numel(),
        "max_abs_error": difference.max().item(),
        "rms_error": difference.square().mean().sqrt().item(),
        "atol": ATOL,
        "rtol": RTOL,
    }


def check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)
    return error_stats(actual, expected)


def assess(actual, expected):
    result = error_stats(actual, expected)
    result["finite"] = bool(torch.isfinite(actual).all())
    try:
        check(actual, expected)
    except AssertionError as error:
        result.update(passed=False, error=str(error))
    else:
        result["passed"] = True
    return result


def bitwise_equal(actual, expected):
    return bool(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))


def write_tensor(path, value):
    canonical = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
    canonical.copy_(value.cpu())
    canonical.view(torch.uint8).numpy().tofile(path)


def read_tensor(path, like):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=like.dtype).reshape(like.shape)


def _bound_names(node):
    names = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    return names


def verify_source_preservation(original_path, operator_path):
    original = original_path.read_bytes()
    candidate = operator_path.read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    assert original_sha == ORIGINAL_SHA256, (original_path, original_sha)
    assert candidate.startswith(original), "adapted source does not preserve the complete original byte prefix"
    original_tree = ast.parse(original, filename=str(original_path))
    candidate_tree = ast.parse(candidate, filename=str(operator_path))
    original_nodes = [ast.dump(node, include_attributes=False) for node in original_tree.body]
    candidate_prefix = [
        ast.dump(node, include_attributes=False)
        for node in candidate_tree.body[: len(original_tree.body)]
    ]
    assert candidate_prefix == original_nodes, "original top-level AST changed"
    original_names = set().union(*(_bound_names(node) for node in original_tree.body))
    appended_nodes = candidate_tree.body[len(original_tree.body):]
    appended_names = set().union(*(_bound_names(node) for node in appended_nodes))
    collisions = sorted(original_names & appended_names)
    assert not collisions, f"adapter overwrites original top-level declarations: {collisions}"
    return {
        "passed": True,
        "original": str(original_path.resolve()),
        "operator": str(operator_path.resolve()),
        "original_sha256": original_sha,
        "original_bytes": len(original),
        "original_top_level_nodes": len(original_tree.body),
        "appended_top_level_nodes": len(appended_nodes),
        "appended_names": sorted(appended_names),
        "overwritten_original_names": collisions,
    }


def load_operator(path):
    spec = importlib.util.spec_from_file_location("flash_kda_preserved_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import operator source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def original_overrides(group, dv_base):
    saved = {name: os.environ.get(name) for name in ("KDA_GROUP", "KDA_DV_BASE")}
    try:
        if group:
            os.environ["KDA_GROUP"] = str(group)
        else:
            os.environ.pop("KDA_GROUP", None)
        if dv_base:
            os.environ["KDA_DV_BASE"] = str(dv_base)
        else:
            os.environ.pop("KDA_DV_BASE", None)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_original(operator, values, layout, group, dv_base):
    with original_overrides(group, dv_base):
        result = operator.flash_kda(
            values["q"], values["k"], values["v"], values["g"], values["beta"],
            values["scale"], values["initial_state"], values["A_log"],
            values["dt_bias"], values["lower_bound"], layout, None,
        )
    torch.npu.synchronize()
    return tuple(value.cpu() for value in result)


def validate_constants(operator, device):
    reference_tensor = torch.empty((1,), dtype=torch.bfloat16, device=f"npu:{device}")
    expected = operator._stage1_constants_for(reference_tensor)
    masks = torch.empty_like(expected.lower_mask_tiles)
    identity = torch.empty_like(expected.identity16)
    operator._frozen_initialize_constants(masks, identity)
    torch.npu.synchronize(device)
    masks_equal = bitwise_equal(masks.cpu(), expected.lower_mask_tiles.cpu())
    identity_equal = bitwise_equal(identity.cpu(), expected.identity16.cpu())
    assert masks_equal and identity_equal
    return {
        "passed": True,
        "lower_mask_tiles_bitwise_equal": masks_equal,
        "identity16_bitwise_equal": identity_equal,
        "lower_mask_bytes": masks.numel() * masks.element_size(),
        "identity_bytes": identity.numel() * identity.element_size(),
    }


def entry_kwargs(values, layout, group, dv_base):
    return {
        **values,
        "is_bsnd": layout == "BSND",
        "group_override": group,
        "dv_base_override": dv_base,
    }


def reject_invalid_inputs(run, values, layout, group, dv_base):
    noncompact = values["q"].transpose(-1, -2).contiguous().transpose(-1, -2)
    assert noncompact.shape == values["q"].shape and not noncompact.is_contiguous()
    invalid_cases = {
        "lower_bound_above_domain": dict(values, lower_bound=0.01),
        "lower_bound_below_domain": dict(values, lower_bound=-5.01),
        "noncompact_layout": dict(values, q=noncompact),
        "dtype_mismatch": dict(values, q=values["q"].float()),
    }
    rejected = []
    for name, bad in invalid_cases.items():
        try:
            run(**entry_kwargs(bad, layout, group, dv_base))
        except (ValueError, TypeError):
            rejected.append(name)
        else:
            raise AssertionError("Unsupported input was accepted: " + name)
    return rejected


def compare_result(actual, original, expected):
    equivalence = {
        "output": assess(actual[0], original[0]),
        "state": assess(actual[1], original[1]),
        "output_bitwise_equal": bitwise_equal(actual[0], original[0]),
        "state_bitwise_equal": bitwise_equal(actual[1], original[1]),
    }
    assert equivalence["output"]["passed"] and equivalence["state"]["passed"], equivalence
    original_accuracy = {
        "output": assess(original[0], expected[0]),
        "state": assess(original[1], expected[1]),
    }
    entry_accuracy = {
        "output": assess(actual[0], expected[0]),
        "state": assess(actual[1], expected[1]),
    }
    original_passed = all(value["passed"] for value in original_accuracy.values())
    entry_passed = all(value["passed"] for value in entry_accuracy.values())
    if original_passed:
        assert entry_passed, entry_accuracy
        accuracy_status = "passed"
    else:
        accuracy_status = "failed_baseline_reproduced"
    return {
        "implementation_equivalence": equivalence,
        "reference_accuracy": {
            "status": accuracy_status,
            "original": original_accuracy,
            "entry": entry_accuracy,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path)
    mode.add_argument("--aclnn-executable", type=Path)
    mode.add_argument("--emit-aclnn-runner", action="store_true")
    mode.add_argument("--source-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operator-source", type=Path, default=Path(__file__).resolve().parents[1] / "flash_kda.py")
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-set", choices=("main", "probe", "override"), default="main")
    parser.add_argument("--case-filter", action="append", default=[], help="run cases whose label contains this text")
    parser.add_argument("--group-override", type=int, default=0)
    parser.add_argument("--dv-base-override", type=int, default=0)
    parser.add_argument("--negative-only", action="store_true")
    args = parser.parse_args()
    if args.emit_aclnn_runner:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(ACLNN_RUNNER_SOURCE)
        return
    if args.original_source is None:
        parser.error("--original-source is required for source and device equivalence")
    args.output.mkdir(parents=True, exist_ok=True)
    source_report = verify_source_preservation(args.original_source, args.operator_source)
    if args.source_only:
        report = {"status": "passed", "source_equivalence": source_report}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("SOURCE_EQUIVALENCE_PASS", json.dumps(report), flush=True)
        return

    global torch, F
    import torch
    import torch.nn.functional as F
    import torch_npu  # noqa: F401
    import cannbotdsl as cb
    from cannbotdsl.package.operator import OperatorBundle
    from cannbotdsl.jit_function import JitFunction

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_op_timeout_ms(120000)
    operator = load_operator(args.operator_source)
    report = {
        "status": "running",
        "execution": "frozen-bundle" if args.bundle else "aclnn",
        "source_equivalence": source_report,
        "constants_equivalence": validate_constants(operator, args.device),
        "group_override": args.group_override,
        "dv_base_override": args.dv_base_override,
        "cases": [],
    }
    run = None
    if args.bundle:
        bundle = OperatorBundle.load(args.bundle)
        properties = torch.npu.get_device_properties(args.device)
        if properties.cube_core_num < bundle.target.aic_count:
            raise ValueError("Deployment has fewer resident AICs than the frozen FlashKDA launch")
        run = bundle.runtime(target=bundle.target)
        report.update(bundle_identity=bundle.identity, runtime_module=cb.__file__)
    try:
        if args.negative_only:
            if run is None:
                parser.error("--negative-only requires --bundle")
            inputs = make_inputs(1, 1, 1, 64, seed=7, layout="BNSD")
            values = {key: value.npu(args.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            report["rejected_inputs"] = reject_invalid_inputs(run, values, "BNSD", args.group_override, args.dv_base_override)
            report.update(
                status="passed",
                implementation_equivalence={"status": "not_run"},
                reference_accuracy={"status": "not_run"},
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("NEGATIVE_VALIDATION_PASS", json.dumps(report["rejected_inputs"]), flush=True)
            return

        cases = PROBE_CASES if args.case_set in {"probe", "override"} else MAIN_CASES
        for case, layout, lower_bound, scale in cases:
            base_label = case_label(case, layout, lower_bound, scale)
            if args.case_filter and not any(pattern in base_label for pattern in args.case_filter):
                continue
            inputs = make_inputs(*case, seed=7, layout=layout)
            input_variant = "default"
            if args.case_set == "probe" and layout == "BSND" and case[1] == 1:
                transposed_q = inputs["q"]
                canonical_q = torch.empty(transposed_q.shape, dtype=transposed_q.dtype)
                canonical_q.copy_(transposed_q)
                assert canonical_q.is_contiguous() and transposed_q.is_contiguous()
                assert canonical_q.stride() != transposed_q.stride()
                inputs["q"] = canonical_q
                input_variant = "mixed_q_canonical_k_transposed"
            label = base_label + ("_mixed_strides" if input_variant != "default" else "")
            inputs.update(lower_bound=lower_bound, scale=scale)
            expected = reference(inputs, layout)
            values = {key: value.npu(args.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            original = run_original(operator, values, layout, args.group_override, args.dv_base_override)
            torch.testing.assert_close(values["initial_state"].cpu(), inputs["initial_state"], atol=0, rtol=0)
            result = {
                "case": label,
                "layout": layout,
                "input_variant": input_variant,
                "input_strides": {
                    name: list(value.stride())
                    for name, value in values.items() if isinstance(value, torch.Tensor)
                },
                "auto_group": operator.get_group_config_bn(case[0] * case[2]),
                "auto_dv_base": operator.get_dv_base_config(case[0] * case[2], case[3]),
                "iterations": [],
            }
            print("RUN", label, flush=True)
            if run:
                original_compile = JitFunction.compile
                actual_results = []
                try:
                    JitFunction.compile = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("frozen entry attempted JIT compilation"))
                    for _ in range(2):
                        actual_npu = run(**entry_kwargs(values, layout, args.group_override, args.dv_base_override))
                        torch.npu.synchronize(args.device)
                        actual = tuple(value.cpu() for value in actual_npu)
                        torch.testing.assert_close(values["initial_state"].cpu(), inputs["initial_state"], atol=0, rtol=0)
                        actual_results.append(actual)
                        result["iterations"].append(compare_result(actual, original, expected))
                finally:
                    JitFunction.compile = original_compile
                result["entry_repeatability"] = {
                    "output_bitwise_equal": bitwise_equal(actual_results[0][0], actual_results[1][0]),
                    "state_bitwise_equal": bitwise_equal(actual_results[0][1], actual_results[1][1]),
                }
                assert all(result["entry_repeatability"].values()), result["entry_repeatability"]
            else:
                directory = args.output / label
                directory.mkdir(exist_ok=True)
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        write_tensor(directory / (key + ".bin"), value)
                runner_command = [
                    str(args.aclnn_executable.resolve()), str(directory.resolve()),
                    *(str(value) for value in case), layout, str(args.device),
                    str(lower_bound), str(scale),
                ]
                if args.group_override or args.dv_base_override:
                    runner_command.extend((str(args.group_override), str(args.dv_base_override)))
                subprocess.run(runner_command, check=True, timeout=180)
                for iteration in range(2):
                    actual = (
                        read_tensor(directory / f"output_{iteration}.bin", inputs["v"]),
                        read_tensor(directory / f"state_{iteration}.bin", inputs["initial_state"]),
                    )
                    result["iterations"].append(compare_result(actual, original, expected))
                initial_after = read_tensor(directory / "initial_state_after.bin", inputs["initial_state"])
                torch.testing.assert_close(initial_after, inputs["initial_state"], atol=0, rtol=0)
                first_actual = (
                    read_tensor(directory / "output_0.bin", inputs["v"]),
                    read_tensor(directory / "state_0.bin", inputs["initial_state"]),
                )
                second_actual = (
                    read_tensor(directory / "output_1.bin", inputs["v"]),
                    read_tensor(directory / "state_1.bin", inputs["initial_state"]),
                )
                result["entry_repeatability"] = {
                    "output_bitwise_equal": bitwise_equal(first_actual[0], second_actual[0]),
                    "state_bitwise_equal": bitwise_equal(first_actual[1], second_actual[1]),
                }
                assert all(result["entry_repeatability"].values()), result["entry_repeatability"]
            report["cases"].append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            reference_statuses = sorted({
                iteration["reference_accuracy"]["status"]
                for iteration in result["iterations"]
            })
            print(
                "EQUIVALENCE_PASS", label,
                "reference_status=" + ",".join(reference_statuses),
                flush=True,
            )

        if not report["cases"]:
            raise ValueError("case filters selected no cases")
        if run:
            small_inputs = make_inputs(1, 1, 1, 64, seed=11, layout="BNSD")
            small_values = {key: value.npu(args.device) if isinstance(value, torch.Tensor) else value for key, value in small_inputs.items()}
            report["rejected_inputs"] = reject_invalid_inputs(run, small_values, "BNSD", args.group_override, args.dv_base_override)
        accuracy_statuses = {
            iteration["reference_accuracy"]["status"]
            for case in report["cases"] for iteration in case["iterations"]
        }
        report["implementation_equivalence"] = {"status": "passed", "cases": len(report["cases"])}
        report["reference_accuracy"] = {
            "status": "passed" if accuracy_statuses == {"passed"} else "failed_baseline_reproduced",
            "case_statuses": sorted(accuracy_statuses),
        }
        report["status"] = (
            "passed"
            if report["reference_accuracy"]["status"] == "passed"
            else "equivalent_with_baseline_accuracy_failure"
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if run:
            run.close()

# Transport fixture only: all cases, golden values and tolerances live above.
ACLNN_RUNNER_SOURCE = r'''
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>
#include "aclnn_FlashKda.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void Check(int result, const char* operation) {
    if (result != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
struct Tensor {
    aclTensor* tensor = nullptr;
    void* device = nullptr;
    size_t bytes = 0;
    Tensor(std::vector<int64_t> shape, aclDataType dtype, const std::string& input = "") {
        std::vector<int64_t> strides(shape.size());
        int64_t elements = 1;
        for (size_t i = shape.size(); i-- > 0;) { strides[i] = elements; elements *= shape[i]; }
        bytes = elements * (dtype == ACL_FLOAT ? 4 : 2);
        Check(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc tensor");
        if (!input.empty()) {
            std::vector<char> data(bytes);
            std::ifstream file(input, std::ios::binary);
            file.read(data.data(), bytes);
            if (file.gcount() != static_cast<std::streamsize>(bytes)) throw std::runtime_error("bad tensor file: " + input);
            Check(aclrtMemcpy(device, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        }
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), device);
        if (!tensor) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { if (tensor) aclDestroyTensor(tensor); if (device) aclrtFree(device); }
    void Dump(const std::string& path) const {
        std::vector<char> data(bytes);
        Check(aclrtMemcpy(data.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
        std::ofstream file(path, std::ios::binary);
        file.write(data.data(), bytes);
        if (!file) throw std::runtime_error("write failed: " + path);
    }
};

int main(int argc, char** argv) {
    if (argc != 10 && argc != 12) { std::cerr << "Usage: validate DIR B Nk Nv S LAYOUT DEVICE LOWER_BOUND SCALE [GROUP_OVERRIDE DV_BASE_OVERRIDE]\n"; return 2; }
    try {
        const std::string dir = argv[1];
        const int64_t b = std::stoll(argv[2]), nk = std::stoll(argv[3]), nv = std::stoll(argv[4]), s = std::stoll(argv[5]);
        const std::string layout = argv[6];
        if (layout != "BNSD" && layout != "BSND") throw std::runtime_error("layout must be BNSD or BSND");
        const bool is_bsnd = layout == "BSND";
        const int device = std::stoi(argv[7]);
        const double lower_bound = std::stod(argv[8]), scale = std::stod(argv[9]);
        const int64_t group_override = argc == 12 ? std::stoll(argv[10]) : 0;
        const int64_t dv_base_override = argc == 12 ? std::stoll(argv[11]) : 0;
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        Check(aclrtSetOpExecuteTimeOutWithMs(120000), "set kernel timeout");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        {
            const std::vector<int64_t> q_shape = is_bsnd ? std::vector<int64_t>{b,s,nk,128} : std::vector<int64_t>{b,nk,s,128};
            const std::vector<int64_t> v_shape = is_bsnd ? std::vector<int64_t>{b,s,nv,128} : std::vector<int64_t>{b,nv,s,128};
            const std::vector<int64_t> beta_shape = is_bsnd ? std::vector<int64_t>{b,s,nv} : std::vector<int64_t>{b,nv,s};
            Tensor q(q_shape, ACL_BF16, dir + "/q.bin"), k(q_shape, ACL_BF16, dir + "/k.bin");
            Tensor v(v_shape, ACL_BF16, dir + "/v.bin"), g(v_shape, ACL_BF16, dir + "/g.bin");
            Tensor beta(beta_shape, ACL_BF16, dir + "/beta.bin");
            Tensor initial({b,nv,128,128}, ACL_FLOAT, dir + "/initial_state.bin");
            Tensor a_log({nv}, ACL_FLOAT, dir + "/A_log.bin"), bias({nv,128}, ACL_FLOAT, dir + "/dt_bias.bin");
            Tensor output(v_shape, ACL_BF16), state({b,nv,128,128}, ACL_FLOAT);
            for (int iteration = 0; iteration < 2; ++iteration) {
                uint64_t bytes = 0;
                aclOpExecutor* executor = nullptr;
                Check(aclnnFlashKdaGetWorkspaceSize(q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                    initial.tensor, a_log.tensor, bias.tensor, output.tensor, state.tensor,
                    dv_base_override, group_override, is_bsnd, lower_bound, scale,
                    &bytes, &executor), "FlashKda GetWorkspaceSize");
                void* workspace = nullptr;
                if (bytes) Check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
                Check(aclnnFlashKda(workspace, bytes, executor, stream), "FlashKda Execute");
                Check(aclrtSynchronizeStream(stream), "synchronize");
                output.Dump(dir + "/output_" + std::to_string(iteration) + ".bin");
                state.Dump(dir + "/state_" + std::to_string(iteration) + ".bin");
                if (workspace) aclrtFree(workspace);
                std::cout << "ACLNN iteration " << iteration << " workspace=" << bytes << '\n';
            }
            initial.Dump(dir + "/initial_state_after.bin");
        }
        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtResetDevice(device), "reset device");
        Check(aclFinalize(), "finalize");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
'''

if __name__ == "__main__":
    main()

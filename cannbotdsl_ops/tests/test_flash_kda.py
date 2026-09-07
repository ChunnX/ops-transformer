# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Full-output numerical acceptance from frozen binaries and a CPU recurrence."""
import argparse
import json
from pathlib import Path
import subprocess



def make_inputs(batch, key_heads, value_heads, sequence, seed):
    generator = torch.Generator().manual_seed(seed)
    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator).to(dtype)
    return {
        "q": randn((batch, key_heads, sequence, 128)),
        "k": randn((batch, key_heads, sequence, 128)),
        "v": randn((batch, value_heads, sequence, 128)),
        "g": randn((batch, value_heads, sequence, 128)),
        "beta": randn((batch, value_heads, sequence)),
        "initial_state": randn((batch, value_heads, 128, 128), torch.float32) * 0.1,
        "a_log": torch.linspace(-1.0, 0.5, value_heads),
        "dt_bias": randn((value_heads, 128), torch.float32),
        "scale": 128 ** -0.5,
        "lower_bound": -1.0,
    }


def reference(inputs):
    """Independent token recurrence; no DSL source or chunk implementation."""
    heads = inputs["v"].shape[1]
    repeats = heads // inputs["k"].shape[1]
    q = F.normalize(inputs["q"].float(), p=2, dim=-1).repeat_interleave(repeats, dim=1)
    k = F.normalize(inputs["k"].float(), p=2, dim=-1).repeat_interleave(repeats, dim=1)
    gate = inputs["lower_bound"] * torch.sigmoid(
        inputs["a_log"].exp()[None, :, None, None] *
        (inputs["g"].float() + inputs["dt_bias"][None, :, None, :]))
    beta = inputs["beta"].float().sigmoid()
    state = inputs["initial_state"].transpose(-1, -2).contiguous()
    output = torch.empty_like(inputs["v"], dtype=torch.float32)
    for token in range(q.shape[2]):
        key = k[:, :, token]
        state = state * gate[:, :, token].exp().unsqueeze(-1)
        delta = (inputs["v"][:, :, token].float() -
                 (key.unsqueeze(-2) @ state).squeeze(-2)) * beta[:, :, token, None]
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, token] = (q[:, :, token].unsqueeze(-2) @ state).squeeze(-2) * inputs["scale"]
    return output, state.transpose(-1, -2).contiguous()


def check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=5e-3)
    difference = (actual.float() - expected.float()).abs()
    return {"elements": actual.numel(), "max_abs_error": difference.max().item(),
            "rms_error": difference.square().mean().sqrt().item(), "atol": 5e-3, "rtol": 5e-3}


def write_tensor(path, value):
    value.contiguous().view(torch.uint8).numpy().tofile(path)


def read_tensor(path, like):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=like.dtype).reshape(like.shape)


def reject_invalid_inputs(run, values):
    noncompact = values["q"].transpose(-1, -2).contiguous().transpose(-1, -2)
    assert noncompact.shape == values["q"].shape and not noncompact.is_contiguous()
    rejected = []
    invalid_cases = {
        "gate_outside_verified_domain": dict(values, lower_bound=0.0),
        "scale_outside_domain": dict(values, scale=1.01),
        "noncompact_layout": dict(values, q=noncompact),
        "dtype_mismatch": dict(values, q=values["q"].float()),
    }
    for name, bad in invalid_cases.items():
        try:
            run(**bad)
        except (ValueError, TypeError):
            rejected.append(name)
        else:
            raise AssertionError("Unsupported input was accepted: " + name)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path)
    mode.add_argument("--aclnn-executable", type=Path)
    mode.add_argument("--emit-aclnn-runner", action="store_true", help="Emit the minimal C++ fixture; no torch or DSL import")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--negative-only", action="store_true", help="Recheck frozen runtime input rejection without launching kernels")
    args = parser.parse_args()
    if args.emit_aclnn_runner:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(ACLNN_RUNNER_SOURCE)
        return
    global torch, F
    import torch
    import torch.nn.functional as F
    if args.negative_only and not args.bundle:
        parser.error("--negative-only requires --bundle")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"execution": "frozen-bundle" if args.bundle else "aclnn", "cases": []}
    run = None
    if args.bundle:
        import torch_npu  # noqa: F401
        import cannbotdsl as cb
        from cannbotdsl.package.operator import OperatorBundle
        from cannbotdsl.jit_function import JitFunction
        def forbidden_compile(*_args, **_kwargs):
            raise AssertionError("Frozen binary acceptance must not compile a JIT Provider")
        JitFunction.compile = forbidden_compile
        torch.npu.set_device(args.device)
        torch.npu.set_op_timeout_ms(20000)
        bundle = OperatorBundle.load(args.bundle)
        properties = torch.npu.get_device_properties(args.device)
        if properties.cube_core_num < bundle.target.aic_count:
            raise ValueError("Deployment has fewer resident AICs than the frozen FlashKDA launch")
        report.update(bundle_identity=bundle.identity, runtime_module=cb.__file__)
        run = bundle.runtime(target=bundle.target)
    try:
        if args.negative_only:
            inputs = make_inputs(1, 2, 2, 64, seed=7)
            values = {key: value.npu(args.device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            report["rejected_inputs"] = reject_invalid_inputs(run, values)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("PASS", json.dumps(report["rejected_inputs"]), flush=True)
            return
        cases = [((1,1,1,64), -1.0, 128 ** -0.5),
                 ((1,1,2,128), -1.0, 128 ** -0.5),
                 ((2,2,2,512), -1.0, 128 ** -0.5),
                 ((1,1,1,1024), -1.0, 128 ** -0.5),
                 ((1,1,2,1024), -1.0, 128 ** -0.5),
                 ((1,2,2,1024), -1.0, 128 ** -0.5),
                 ((1,1,2,128), -5.0, 1.0),
                 ((1,2,2,64), -1.0, 0.0)]
        for case, lower_bound, scale in cases:
            inputs = make_inputs(*case, seed=7)
            inputs.update(lower_bound=lower_bound, scale=scale)
            expected = reference(inputs)
            label = ("B%d_Nk%d_Nv%d_S%d" % case) + f"_gate{lower_bound:g}_scale{scale:g}"
            result = {"case": label, "iterations": []}
            print("RUN", label, flush=True)
            if run:
                values = {key: value.npu(args.device) if isinstance(value, torch.Tensor) else value
                          for key, value in inputs.items()}
                for iteration in range(2):
                    out, state = run(**values)
                    torch.npu.synchronize(args.device)
                    result["iterations"].append({"output": check(out.cpu(), expected[0]),
                                                  "state": check(state.cpu(), expected[1])})
                    torch.testing.assert_close(values["initial_state"].cpu(), inputs["initial_state"], atol=0, rtol=0)
            else:
                directory = args.output / label
                directory.mkdir(exist_ok=True)
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        write_tensor(directory / (key + ".bin"), value)
                subprocess.run([str(args.aclnn_executable.resolve()), str(directory.resolve()),
                                *(str(value) for value in case), str(args.device),
                                str(lower_bound), str(scale)], check=True, timeout=90)
                for iteration in range(2):
                    out = read_tensor(directory / f"output_{iteration}.bin", inputs["v"])
                    state = read_tensor(directory / f"state_{iteration}.bin", inputs["initial_state"])
                    result["iterations"].append({"output": check(out, expected[0]), "state": check(state, expected[1])})
                initial_after = read_tensor(directory / "initial_state_after.bin", inputs["initial_state"])
                torch.testing.assert_close(initial_after, inputs["initial_state"], atol=0, rtol=0)
            report["cases"].append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("PASS", label, json.dumps(result["iterations"]), flush=True)
        if run:
            report["rejected_inputs"] = reject_invalid_inputs(run, values)
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
    if (argc != 9) { std::cerr << "Usage: validate DIR B Nk Nv S DEVICE LOWER_BOUND SCALE\n"; return 2; }
    try {
        const std::string dir = argv[1];
        const int64_t b = std::stoll(argv[2]), nk = std::stoll(argv[3]), nv = std::stoll(argv[4]), s = std::stoll(argv[5]);
        const int device = std::stoi(argv[6]);
        const double lower_bound = std::stod(argv[7]), scale = std::stod(argv[8]);
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        Check(aclrtSetOpExecuteTimeOutWithMs(20000), "set kernel timeout");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        {
            Tensor q({b,nk,s,128}, ACL_BF16, dir + "/q.bin"), k({b,nk,s,128}, ACL_BF16, dir + "/k.bin");
            Tensor v({b,nv,s,128}, ACL_BF16, dir + "/v.bin"), g({b,nv,s,128}, ACL_BF16, dir + "/g.bin");
            Tensor beta({b,nv,s}, ACL_BF16, dir + "/beta.bin");
            Tensor initial({b,nv,128,128}, ACL_FLOAT, dir + "/initial_state.bin");
            Tensor a_log({nv}, ACL_FLOAT, dir + "/a_log.bin"), bias({nv,128}, ACL_FLOAT, dir + "/dt_bias.bin");
            Tensor output({b,nv,s,128}, ACL_BF16), state({b,nv,128,128}, ACL_FLOAT);
            for (int iteration = 0; iteration < 2; ++iteration) {
                uint64_t bytes = 0;
                aclOpExecutor* executor = nullptr;
                Check(aclnnFlashKdaGetWorkspaceSize(q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                    initial.tensor, a_log.tensor, bias.tensor, output.tensor, state.tensor,
                    lower_bound, scale, &bytes, &executor), "FlashKda GetWorkspaceSize");
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

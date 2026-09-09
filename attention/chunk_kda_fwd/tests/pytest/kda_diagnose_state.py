"""通过 torch_extension 重复检查 dense/TND 路径的最终状态，不修改 CANN 内核。"""
import argparse
import gc
import json
from pathlib import Path

import torch
from cann_ops_transformer.ops import chunk_kda_fwd

from kda_rdv_perf import CpuReference, DEFAULT_CASES, make_inputs, precision_metrics, read_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ARC-004")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    case = next(x for x in read_cases(DEFAULT_CASES) if x["test_name"] == args.case)
    inputs = make_inputs(case)
    q, k, v, g, beta, state, lengths, scale, a_log = inputs
    assert a_log is not None, "选择含 gate 的 128x128 用例"
    expected, _ = CpuReference()(inputs, args.case)
    batch, length = len(lengths), lengths[0]
    assert len(set(lengths)) == 1
    q, k, v, g, beta = [x.view(batch, length, *x.shape[1:]) for x in (q, k, v, g, beta)]
    configs = [
        ("dense_raw", True, True, True, {}),
        ("dense_raw_kv", True, True, False, {}),
        ("dense_preactivated", False, False, True, {}),
        ("dense_preactivated_safe", False, True, True, {}),
        ("dense_raw_h", True, True, True, {"return_intermediate_states": True}),
        ("dense_raw_all", True, True, True, {"disable_recompute": True}),
        ("tnd_raw", True, True, True, {"layout": "TND"}),
        ("chunk128_preactivated", False, False, True, {"chunk_size": 128}),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for label, raw_gate, safe_gate, v_first, options in configs:
        initial = state.clone() if v_first else state.transpose(-1, -2).contiguous()
        gate = g if raw_gate else -5 * torch.sigmoid(g)
        tensors = [q, k, v, gate, beta]
        if options.get("layout") == "TND":
            tensors = [x.flatten(0, 1) for x in tensors]
            options["cu_seqlens_cpu"] = [i * length for i in range(batch + 1)]
        for repeat in range(args.repeats):
            outputs = chunk_kda_fwd(*tensors, scale, initial_state=initial,
                output_final_state=True, state_v_first=v_first, use_gate_in_kernel=raw_gate,
                safe_gate=safe_gate, A_log=a_log if raw_gate else None, **options)
            torch.npu.synchronize()
            attention = outputs[0].reshape_as(inputs[2])
            final = outputs[1] if v_first else outputs[1].transpose(-1, -2)
            metrics = [precision_metrics(x, y) for x, y in zip((attention, final), expected)]
            result = dict(test_name=args.case, config=label, repeat=repeat, metrics=metrics,
                initial_state_unchanged=bool(torch.equal(initial, state if v_first else state.transpose(-1, -2))))
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            del outputs, attention, final
            gc.collect()
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()

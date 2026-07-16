"""Run one generated case through the installed torch_npu operator."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu

from case_utils import dump_json
from compare_result import compare_case
from golden_a2_a5 import OUTPUTS, read_tensor, save_tensor, torch_dtype


def load_inputs(case_dir, manifest):
    tensors = {}
    for name, description in manifest["inputs"].items():
        tensors[name] = read_tensor(
            case_dir / description["file"],
            description["shape"],
            description["dtype"],
        )
    return tensors


def run(case_dir: Path, device_id: int):
    manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    case = manifest["case"]
    tensors = load_inputs(case_dir, manifest)
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    logical_device_id = 0 if visible_devices else device_id
    torch.npu.set_device(logical_device_id)
    device = torch.device(f"npu:{logical_device_id}")
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(logical_device_id)
    allocated_before = torch.npu.memory_allocated(logical_device_id)
    device_tensors = {name: tensor.contiguous().to(device) for name, tensor in tensors.items()}
    optional = {}
    if case["rope"]:
        optional["query_rope"] = device_tensors["query_rope"]
        optional["key_rope"] = device_tensors["key_rope"]
    if case["layout"] == "TND":
        optional["actual_seq_qlen"] = manifest["metadata"]["actual_seq_qlen"]
        optional["actual_seq_klen"] = manifest["metadata"]["actual_seq_klen"]

    start = time.perf_counter()
    outputs = torch.ops.npu.npu_dense_lightning_indexer_grad_kl_loss(
        device_tensors["query"],
        device_tensors["key"],
        device_tensors["query_index"],
        device_tensors["key_index"],
        device_tensors["weight"],
        device_tensors["softmax_max"],
        device_tensors["softmax_sum"],
        device_tensors["softmax_max_index"],
        device_tensors["softmax_sum_index"],
        float(case.get("scale", 1.0 / int(case["d"]) ** 0.5)),
        layout=case["layout"],
        sparse_mode=int(case["sparse_mode"]),
        pre_tokens=int(case["pre_tokens"]),
        next_tokens=int(case["next_tokens"]),
        **optional,
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_allocated = torch.npu.max_memory_allocated(logical_device_id)
    peak_reserved = torch.npu.max_memory_reserved(logical_device_id)
    output_dir = case_dir / "outputs"
    for name, tensor in zip(OUTPUTS, outputs):
        save_tensor(output_dir / f"{name}.bin", tensor.cpu())
    report = compare_case(case_dir)
    report["elapsed_ms"] = elapsed_ms
    report["npu_memory"] = {
        "allocated_before_bytes": int(allocated_before),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
    }
    dump_json(case_dir / "run_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    report = run(args.case_dir, args.device_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

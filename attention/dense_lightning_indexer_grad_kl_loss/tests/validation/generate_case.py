"""Generate reproducible input, softmax-stat, golden, and manifest files."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import torch

from case_utils import dump_json, load_all_cases, select_cases, validate_case
from golden_a2_a5 import (
    OUTPUTS,
    TENSOR_INPUTS,
    compute_reference,
    dtype_name,
    make_inputs,
    save_tensor,
)


def tensor_record(relative_path: Path, tensor: torch.Tensor, sha256: str):
    return {
        "file": relative_path.as_posix(),
        "shape": list(tensor.shape),
        "dtype": dtype_name(tensor.dtype),
        "sha256": sha256,
    }


def generate(case, output_root: Path, golden_device: str):
    validate_case(case)
    seed = secrets.randbits(63)
    case_dir = output_root / str(case["id"])
    tensors, metadata = make_inputs(case, seed)
    stats, outputs = compute_reference(tensors, case, metadata, device=golden_device)
    tensors.update(stats)

    manifest = {
        "case": dict(case),
        "seed": seed,
        "metadata": metadata,
        "inputs": {},
        "outputs": {},
    }
    for name in TENSOR_INPUTS:
        tensor = tensors.get(name)
        if tensor is None:
            continue
        relative = Path("inputs") / f"{name}.bin"
        sha256 = save_tensor(case_dir / relative, tensor)
        manifest["inputs"][name] = tensor_record(relative, tensor, sha256)

    for name in OUTPUTS:
        tensor = outputs[name]
        relative = Path("golden") / f"{name}.bin"
        sha256 = save_tensor(case_dir / relative, tensor)
        manifest["outputs"][name] = {
            "golden_file": relative.as_posix(),
            "shape": list(tensor.shape),
            "dtype": dtype_name(tensor.dtype),
            "sha256": sha256,
        }
    dump_json(case_dir / "case_manifest.json", manifest)
    return case_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "typical", "full"), action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--golden-device", default="cpu")
    args = parser.parse_args()
    suites = args.suite or ["smoke"]
    cases = select_cases(args.root, suites, args.case)
    for case in cases:
        case_dir = generate(case, args.output_root, args.golden_device)
        print(f"generated {case['id']}: {case_dir}")


if __name__ == "__main__":
    main()

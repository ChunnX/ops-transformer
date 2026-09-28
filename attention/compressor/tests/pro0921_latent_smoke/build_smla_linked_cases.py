# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Tiny, directly runnable sparse_flash_mla (MLA) case for the delay0/delay1
# visible-length contract, wired to a *real* metadata tensor.
#
# What changed from the previous round:
#   * the case now declares the full 18-input MLA ABI and is emitted as a real
#     TTK case.csv (the previous script only produced .npy fixtures + a golden),
#     so `run_tiny_sim.sh <case-root> sparse_mla` can execute it directly;
#   * `metadata` is mandatory for the op (op_host/sparse_flash_mla_tiling.cpp
#     CheckSingleParaMetadata requires a (1024,) int32 tensor) and this script
#     refuses to fabricate one.  Supply the tensor produced by the real
#     SparseFlashMlaMetadata op via --metadata-npy or PRO0921_SMLA_METADATA_NPY.
#
# Environment limitation (source evidence), why the metadata cannot be produced
# here:
#   * attention/sparse_flash_mla_metadata has no `*_def.cpp`; its only kernel is
#     op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.{cpp,h} registered through
#     op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.json, whose opInfo sets
#     "engine": "DNN_VM_AICPU", "opKernelLib": "CUSTAICPUKernel".
#   * TTK kernel mode builds a testcase from the TBE op-info registry
#     (ttk/core_modules/operator/op_info_keeper.py reads
#     aic-<soc>-ops-info.json), which an AICPU-only op has no entry in.
#   * therefore CAMODEL (AscendC/TBE) can execute sparse_flash_mla itself but not
#     its AICPU metadata pre-op; the metadata must come from an aclnn/board run
#     of the operator, e.g. examples/test_torch_sparse_flash_mla_metadata.py or
#     aclnnSparseFlashMlaMetadata.
#
# Run:
#   python3 build_smla_linked_cases.py --out /tmp/pro0921_latent_mla \
#       --metadata-npy /path/from/real/run/metadata.npy
#   # per-delay metadata (one real tensor per delay case, never reused):
#   python3 build_smla_linked_cases.py --out /tmp/pro0921_latent_mla \
#       --metadata-dir /tmp/pro0921_latent_mla/meta
#
# Metadata sources, in priority order per delay case:
#   --metadata-npy-delay0 / --metadata-npy-delay1
#   --metadata-dir/metadata_delay<N>.npy
#   --metadata-npy (or $PRO0921_SMLA_METADATA_NPY) applied to both delays
# Each case dir always receives its own copy of the resolved tensor, and the
# source path is recorded in expected.json so reusing one delay's metadata for
# the other is visible, not silent.

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


HEAD = 512
RATIO = 4
Q_LEN = 1
ORI_WIN = 8
CMP_GROUPS = 3
METADATA_LIMIT = 1024

CSV_COLUMNS = [
    "testcase_name", "network_name", "op_name", "input_shapes", "input_dtypes",
    "input_formats", "output_shapes", "output_dtypes", "output_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision", "output_inplace_indexes", "is_enabled", "remark",
]

OP_INPUTS = (
    "q", "ori_kv", "cmp_kv", "ori_sparse_indices", "cmp_sparse_indices",
    "ori_block_table", "cmp_block_table", "cu_seqlens_q", "cu_seqlens_ori_kv",
    "cu_seqlens_cmp_kv", "seqused_q", "seqused_ori_kv", "seqused_cmp_kv",
    "cmp_residual_kv", "ori_topk_length", "cmp_topk_length", "sinks", "metadata",
)
INPUT_DTYPES = {
    "q": "bfloat16", "ori_kv": "bfloat16", "cmp_kv": "bfloat16",
    "ori_sparse_indices": "int32", "cmp_sparse_indices": "int32",
    "ori_block_table": "int32", "cmp_block_table": "int32",
    "cu_seqlens_q": "int32", "cu_seqlens_ori_kv": "int32", "cu_seqlens_cmp_kv": "int32",
    "seqused_q": "int32", "seqused_ori_kv": "int32", "seqused_cmp_kv": "int32",
    "cmp_residual_kv": "int32", "ori_topk_length": "int32", "cmp_topk_length": "int32",
    "sinks": "float32", "metadata": "int32",
}
OP_ATTRS = {
    "softmax_scale": 1.0 / (HEAD ** 0.5),
    "cmp_ratio": RATIO,
    "ori_mask_mode": 4,      # sliding window
    "cmp_mask_mode": 3,      # causal
    "ori_win_left": ORI_WIN - 1,
    "ori_win_right": 0,
    "layout_q": "TND",
    "layout_kv": "TND",
    "topk_value_mode": 1,
    "return_softmax_lse": False,
}


def _softmax(values, axis):
    values = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=axis, keepdims=True)


def cpu_attention(q, ori_kv, cmp_kv, visible_cmp_tokens):
    scores_ori = (q @ ori_kv.T) / (HEAD ** 0.5)
    scores_cmp = (q @ cmp_kv[:visible_cmp_tokens].T) / (HEAD ** 0.5)
    scores = np.concatenate([scores_ori, scores_cmp], axis=-1)
    weights = _softmax(scores, axis=-1)
    values = np.concatenate([ori_kv, cmp_kv[:visible_cmp_tokens]], axis=0)
    return (weights @ values).astype(np.float32)


def effective_tokens(actual_groups, delay):
    return max(actual_groups - delay, 0)


def load_metadata(path):
    array = np.load(path)
    array = np.asarray(array).reshape(-1).astype(np.int32)
    if array.size != METADATA_LIMIT:
        raise SystemExit(
            "FAIL: metadata must have %d int32 elements (sparse_flash_mla_tiling.h "
            "METADATA_LIMIT), got %d from %s" % (METADATA_LIMIT, array.size, path))
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/pro0921_latent_mla"))
    parser.add_argument("--metadata-npy", default=os.environ.get("PRO0921_SMLA_METADATA_NPY", ""))
    parser.add_argument("--metadata-npy-delay0", default="")
    parser.add_argument("--metadata-npy-delay1", default="")
    parser.add_argument("--metadata-dir", type=Path, default=None,
                        help="dir holding metadata_delay0.npy / metadata_delay1.npy")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_delay_override = {
        0: args.metadata_npy_delay0,
        1: args.metadata_npy_delay1,
    }

    def resolve_metadata(delay):
        """Return (tensor, source) for one delay; no cross-delay reuse."""
        explicit = per_delay_override.get(delay)
        if explicit:
            return load_metadata(Path(explicit)), str(Path(explicit).resolve())
        if args.metadata_dir is not None:
            candidate = Path(args.metadata_dir) / ("metadata_delay%d.npy" % delay)
            if candidate.exists():
                return load_metadata(candidate), str(candidate.resolve())
        if args.metadata_npy:
            return load_metadata(Path(args.metadata_npy)), str(Path(args.metadata_npy).resolve())
        return None, "missing"

    rng = np.random.default_rng(20260927)
    q = (rng.standard_normal((Q_LEN, HEAD)) * 0.02).astype(np.float32)
    ori_kv = (rng.standard_normal((ORI_WIN, HEAD)) * 0.02).astype(np.float32)
    cmp_kv = (rng.standard_normal((CMP_GROUPS, HEAD)) * 0.02).astype(np.float32)

    cases = []
    outputs = {}
    metadata_sources = {}
    for delay in (0, 1):
        visible = effective_tokens(CMP_GROUPS, delay)
        metadata, metadata_source = resolve_metadata(delay)
        metadata_sources[delay] = metadata_source
        payload = {
            "q": q.reshape(Q_LEN, 1, HEAD),
            "ori_kv": ori_kv.reshape(ORI_WIN, 1, HEAD),
            "cmp_kv": cmp_kv.reshape(CMP_GROUPS, 1, HEAD),
            "ori_sparse_indices": np.arange(ORI_WIN, dtype=np.int32).reshape(Q_LEN, 1, -1),
            "cmp_sparse_indices": np.arange(CMP_GROUPS, dtype=np.int32).reshape(Q_LEN, 1, -1),
            "ori_block_table": None,
            "cmp_block_table": None,
            "cu_seqlens_q": np.array([0, Q_LEN], dtype=np.int32),
            "cu_seqlens_ori_kv": np.array([0, ORI_WIN], dtype=np.int32),
            "cu_seqlens_cmp_kv": np.array([0, CMP_GROUPS], dtype=np.int32),
            "seqused_q": np.array([Q_LEN], dtype=np.int32),
            "seqused_ori_kv": np.array([ORI_WIN], dtype=np.int32),
            "seqused_cmp_kv": np.array([visible], dtype=np.int32),
            "cmp_residual_kv": np.zeros((1,), dtype=np.int32),
            "ori_topk_length": np.array([[ORI_WIN]], dtype=np.int32),
            "cmp_topk_length": np.array([[CMP_GROUPS]], dtype=np.int32),
            "sinks": np.array([-100.0], dtype=np.float32),
            "metadata": metadata,
        }
        golden = cpu_attention(q, ori_kv, cmp_kv, visible)
        outputs[delay] = golden
        name = "sparse_mla_b1_q1_delay%d" % delay
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        for key in OP_INPUTS:
            value = payload[key]
            if value is None:
                continue
            np.save(dest / (key + ".npy"), value)
        np.save(dest / "golden_0.npy", np.asarray(golden, dtype=np.float32).reshape(Q_LEN, 1, HEAD))
        (dest / "golden.json").write_text(json.dumps({
            "files": ["golden_0.npy", None], "dtypes": ["float32", None],
            "output_names": ["attn_out", "softmax_lse"],
        }, indent=2), encoding="utf-8")
        runnable = metadata is not None
        expected = {
            "checked_outputs": ["attn_out"] if runnable else [],
            "suppressed_outputs": ["softmax_lse"] if runnable else ["attn_out", "softmax_lse"],
            "metadata_source": metadata_source,
            "delay": delay,
            "visible_cmp_tokens": visible,
            "note": ("delay%d: visible compressed tokens = %d; attn_out compared against the "
                     "CPU golden once a real metadata tensor is supplied" % (delay, visible))
            if runnable else
            ("delay%d: visible compressed tokens = %d; NOT runnable without the metadata tensor "
             "from SparseFlashMlaMetadata (AICPU-only, no CAMODEL entry)" % (delay, visible)),
        }
        (dest / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
        row = {
            "testcase_name": name,
            "network_name": "deepseek_v4",
            "op_name": "sparse_flash_mla",
            "input_shapes": repr(tuple(
                None if payload[k] is None else tuple(int(v) for v in payload[k].shape)
                for k in OP_INPUTS)),
            "input_dtypes": repr(tuple(INPUT_DTYPES[k] for k in OP_INPUTS)),
            "input_formats": repr(tuple("ND" for _ in OP_INPUTS)),
            "output_shapes": repr(((Q_LEN, 1, HEAD), (Q_LEN, 1))),
            "output_dtypes": repr(("bfloat16", "float32")),
            "output_formats": repr(("ND", "ND")),
            "attributes": repr(OP_ATTRS),
            "input_data_ranges": repr(tuple((-0.1, 0.1) for _ in OP_INPUTS)),
            "precision_tolerances": repr(((0.005, 0.005), (1., 1.))),
            "absolute_precision": 0.001,
            "output_inplace_indexes": repr((None, None)),
            "is_enabled": True,
            "remark": "tiny MLA B1 Q1 delay%d metadata=%s" % (delay, metadata_source),
        }
        with (dest / "case.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        cases.append({"name": name, "dir": str(dest), "kind": "sparse_flash_mla",
                      "cmp_ratio": RATIO, "coff": 1, "x_cache_update": False,
                      "expected": expected, "runnable": runnable})
        print("built %s visible_cmp_tokens=%d runnable=%s -> %s"
              % (name, visible, runnable, dest))

    diff = float(np.abs(outputs[1] - outputs[0]).max())
    all_runnable = all(metadata_sources[d] != "missing" for d in (0, 1))
    manifest = {
        "cases": cases,
        "delay0_vs_delay1_max_abs_diff": diff,
        "delay_is_observable": diff > 0.0,
        "metadata_source": metadata_sources,
        "native_status": ("runnable: metadata supplied per delay" if all_runnable else
                          "blocked: metadata must be produced by the real "
                          "SparseFlashMlaMetadata AICPU op (no CAMODEL op-info entry); "
                          "generate it with aicpu_metadata_host/run_metadata_host.py and pass "
                          "--metadata-dir, or --metadata-npy-delay0/--metadata-npy-delay1"),
    }
    (args.out / "smla_cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plan = args.out / "smla_linked_plan.md"
    plan.write_text(
        "# Tiny MLA (metadata -> sparse_flash_mla) linked run\n\n"
        "1. Produce one metadata tensor per delay with the real op:\n"
        "   * op: `SparseFlashMlaMetadata`, engine `DNN_VM_AICPU`\n"
        "     (`attention/sparse_flash_mla_metadata/op_kernel_aicpu/"
        "sparse_flash_mla_metadata_aicpu.json`).\n"
        "   * it has no `*_def.cpp`, so TTK kernel mode (which builds cases from\n"
        "     `aic-<soc>-ops-info.json`, `ttk/core_modules/operator/op_info_keeper.py`)\n"
        "     cannot select it; CAMODEL can run `sparse_flash_mla` but not this pre-op.\n"
        "   * run the real kernel on the host with `aicpu_metadata_host/`:\n"
        "     `bash aicpu_metadata_host/gen_smla_metadata.sh <root>` (host g++ build,\n"
        "     real `SparseFlashMlaMetadataCpuKernel::Compute`, one tensor per delay).\n"
        "   * a board run of `test_torch_sparse_flash_mla_metadata.py` /\n"
        "     `aclnnSparseFlashMlaMetadata` with the same attrs is the board equivalent.\n"
        "2. Run the linked case in CAMODEL:\n"
        "   `python3 build_smla_linked_cases.py --out <root> --metadata-dir <meta>`\n"
        "   then `bash run_tiny_sim.sh <root> sparse_mla`.\n"
        "3. The delay0/delay1 variants differ only in `seqused_cmp_kv` (visible\n"
        "   compressed tokens); the CPU golden asserts the delay is observable.\n",
        encoding="utf-8")
    if not manifest["delay_is_observable"]:
        raise SystemExit("FAIL: delay0 and delay1 CPU outputs are identical")
    print("delay0 vs delay1 max abs diff = %.6g (observable)" % diff)
    for delay in (0, 1):
        print("metadata_source[delay%d]=%s" % (delay, metadata_sources[delay]))
    print("cases under %s" % args.out)


if __name__ == "__main__":
    main()

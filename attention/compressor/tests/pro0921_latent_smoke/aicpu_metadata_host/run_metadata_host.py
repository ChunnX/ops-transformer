# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Drive the host metadata driver for each built sparse_flash_mla case and emit
# one real metadata tensor per case.
#
# The metadata op is AICPU-only, so CAMODEL cannot run it.  This wrapper feeds
# the *same* parameters the MLA case declares (cmp_ratio / mask modes / window /
# layouts / cu_seqlens / seqused / residual / topk lengths) into the host driver
# that calls the real SparseFlashMlaMetadataCpuKernel::Compute.  Each delay case
# therefore gets its own metadata file; nothing is fabricated and nothing is
# reused across cases.
#
# Usage:
#   python3 run_metadata_host.py --mla-root /tmp/pro0921_latent_mla \
#       --out /tmp/pro0921_latent_mla/meta
#   # then rebuild the cases with real metadata:
#   python3 build_smla_linked_cases.py --out /tmp/pro0921_latent_mla \
#       --metadata-dir /tmp/pro0921_latent_mla/meta
#
# Attrs that the metadata op needs but the MLA case does not declare are
# explicit inputs here (num_heads_q / num_heads_kv / head_dim / ori_topk /
# cmp_topk); the defaults match the tiny fixture.  They are recorded per case in
# metadata_params_delay<N>.json so the numbers stay auditable.

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DRIVER = HERE / "build" / "metadata_host"

# The metadata op reads these at fixed input indices 0..8 (see
# ParamId in sparse_flash_mla_metadata_aicpu.h).
INPUT_ORDER = (
    "cu_seqlens_q", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv",
    "seqused_q", "seqused_ori_kv", "seqused_cmp_kv",
    "cmp_residual_kv", "ori_topk_length", "cmp_topk_length",
)

# Attribute names copied verbatim from the MLA case (same names / same meaning).
ATTR_COPY_INT = ("cmp_ratio", "ori_mask_mode", "cmp_mask_mode",
                 "ori_win_left", "ori_win_right")
ATTR_COPY_STR = ("layout_q", "layout_kv")

METADATA_ELEMS = 1024
FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72


def case_attributes(case_dir):
    row = next(csv.DictReader((case_dir / "case.csv").open(encoding="utf-8")))
    return ast.literal_eval(row["attributes"])


def discover_cases(mla_root):
    manifest_path = mla_root / "smla_cases.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = data.get("cases", data) if isinstance(data, dict) else data
        return [Path(e["dir"]) for e in entries]
    return sorted(Path(p) for p in mla_root.glob("sparse_mla_*") if p.is_dir())


def write_workdir(case_dir, workdir, params):
    workdir.mkdir(parents=True, exist_ok=True)

    attr_lines = []
    for name in ATTR_COPY_INT:
        attr_lines.append("int %s %d" % (name, int(params["attrs"][name])))
    for name in ATTR_COPY_STR:
        attr_lines.append("str %s %s" % (name, str(params["attrs"][name])))
    for name in ("num_heads_q", "num_heads_kv", "head_dim", "batch_size",
                 "max_seqlen_q", "max_seqlen_ori_kv", "max_seqlen_cmp_kv",
                 "ori_topk", "cmp_topk"):
        attr_lines.append("int %s %d" % (name, int(params[name])))
    attr_lines.append("str soc_version %s" % params["soc_version"])
    if params.get("aic_core_num") is not None:
        attr_lines.append("int aic_core_num %d" % int(params["aic_core_num"]))
    if params.get("aiv_core_num") is not None:
        attr_lines.append("int aiv_core_num %d" % int(params["aiv_core_num"]))
    (workdir / "attrs.txt").write_text("\n".join(attr_lines) + "\n", encoding="utf-8")

    input_lines = []
    for name in INPUT_ORDER:
        path = case_dir / (name + ".npy")
        if not path.exists():
            raise SystemExit("FAIL: %s is missing %s.npy (build the case first)" % (case_dir, name))
        array = np.load(path)
        if array.dtype != np.int32:
            array = array.astype(np.int32)
        dims = ",".join(str(int(v)) for v in array.shape)
        (workdir / (name + ".bin")).write_bytes(np.ascontiguousarray(array).tobytes())
        input_lines.append("%s int32 %s %s.bin" % (name, dims, name))
    (workdir / "inputs.txt").write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    (workdir / "output.txt").write_text(
        "int32 %d metadata.bin\n" % METADATA_ELEMS, encoding="utf-8")


def delay_of(case_dir):
    name = case_dir.name
    if "delay" in name:
        tail = name.split("delay", 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)
    return None


def validate_metadata(array):
    if array.dtype != np.int32:
        return "FAIL:metadata dtype %s, expected int32" % array.dtype
    if array.size != METADATA_ELEMS:
        return "FAIL:metadata size %d, expected %d (sparse_flash_mla_tiling.h METADATA_LIMIT)" % (
            array.size, METADATA_ELEMS)
    fa = array[:AIC_CORE_MAX_NUM * FA_METADATA_SIZE].reshape(AIC_CORE_MAX_NUM, FA_METADATA_SIZE)
    fd_flat = array[AIC_CORE_MAX_NUM * FA_METADATA_SIZE:
                    AIC_CORE_MAX_NUM * FA_METADATA_SIZE + AIV_CORE_MAX_NUM * FD_METADATA_SIZE]
    fd = fd_flat[:AIV_CORE_MAX_NUM * FD_METADATA_SIZE].reshape(AIV_CORE_MAX_NUM, FD_METADATA_SIZE)
    fa_enabled = int(np.count_nonzero(fa[:, 0]))
    fd_enabled = int(np.count_nonzero(fd[:, 0]))
    if fa_enabled == 0:
        return "FAIL:no FA core enabled in the metadata"
    return ("OK fa_enabled_cores=%d fd_enabled_cores=%d fa0=%s"
            % (fa_enabled, fd_enabled, fa[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mla-root", type=Path, required=True,
                        help="root of the built sparse_flash_mla cases")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for metadata_delay<N>.npy")
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--num-heads-q", type=int, default=1)
    parser.add_argument("--num-heads-kv", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--ori-topk", type=int, default=None,
                        help="metadata op ori_topk; default matches the sparse index width")
    parser.add_argument("--cmp-topk", type=int, default=None)
    parser.add_argument("--soc-version", default="Ascend950")
    parser.add_argument("--aic-core-num", type=int, default=None,
                        help="omit to use the op default (36)")
    parser.add_argument("--aiv-core-num", type=int, default=None,
                        help="omit to use the op default (72)")
    args = parser.parse_args()

    driver = args.driver
    if not os.access(driver, os.X_OK):
        raise SystemExit(
            "FAIL: metadata host driver %s is not executable; run build_metadata_host.sh first"
            % driver)

    cases = discover_cases(args.mla_root)
    if not cases:
        raise SystemExit("FAIL: no sparse_flash_mla cases under %s" % args.mla_root)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case_dir in cases:
        attrs = case_attributes(case_dir)
        seqused_q = np.load(case_dir / "seqused_q.npy").reshape(-1)
        seqused_ori = np.load(case_dir / "seqused_ori_kv.npy").reshape(-1)
        seqused_cmp = np.load(case_dir / "seqused_cmp_kv.npy").reshape(-1)
        cu_q = np.load(case_dir / "cu_seqlens_q.npy").reshape(-1)
        params = {
            "case_dir": str(case_dir.resolve()),
            "attrs": {name: attrs[name] for name in ATTR_COPY_INT + ATTR_COPY_STR},
            "num_heads_q": args.num_heads_q,
            "num_heads_kv": args.num_heads_kv,
            "head_dim": args.head_dim,
            "batch_size": int(cu_q.size - 1),
            "max_seqlen_q": int(seqused_q.max()),
            "max_seqlen_ori_kv": int(seqused_ori.max()),
            "max_seqlen_cmp_kv": int(seqused_cmp.max()),
            "ori_topk": args.ori_topk if args.ori_topk is not None else int(np.load(case_dir / "ori_sparse_indices.npy").shape[-1]),
            "cmp_topk": args.cmp_topk if args.cmp_topk is not None else int(np.load(case_dir / "cmp_sparse_indices.npy").shape[-1]),
            "soc_version": args.soc_version,
            "aic_core_num": args.aic_core_num,
            "aiv_core_num": args.aiv_core_num,
        }
        delay = delay_of(case_dir)
        tag = "delay%d" % delay if delay is not None else case_dir.name
        workdir = args.out / ("work_" + tag)
        write_workdir(case_dir, workdir, params)

        proc = subprocess.run([str(driver), str(workdir)], capture_output=True, text=True)
        log = (workdir / "driver.log")
        log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit("FAIL: metadata driver rc=%d for %s\n--- driver output ---\n%s"
                             % (proc.returncode, case_dir, proc.stdout + proc.stderr))

        meta_bytes = (workdir / "metadata.bin").read_bytes()
        array = np.frombuffer(meta_bytes, dtype=np.int32).copy()
        verdict = validate_metadata(array)
        if not verdict.startswith("OK"):
            raise SystemExit("FAIL: %s for %s\n--- driver output ---\n%s"
                             % (verdict, case_dir, proc.stdout + proc.stderr))

        out_npy = args.out / ("metadata_%s.npy" % tag)
        np.save(out_npy, array)
        (args.out / ("metadata_params_%s.json" % tag)).write_text(
            json.dumps(params, indent=2), encoding="utf-8")
        manifest.append({"tag": tag, "case_dir": str(case_dir.resolve()),
                         "metadata_npy": str(out_npy.resolve()),
                         "delay": delay, "params": params, "verdict": verdict})
        print("metadata %-10s -> %s (%s)" % (tag, out_npy, verdict))
        print(proc.stdout.strip())

    (args.out / "metadata_manifest.json").write_text(
        json.dumps({"entries": manifest}, indent=2), encoding="utf-8")
    print("wrote %d metadata tensor(s) under %s" % (len(manifest), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

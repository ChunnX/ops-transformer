#!/usr/bin/env python3
"""Run all RDV test cases and generate a report."""
import csv
import os
import time
import torch
import torch_npu
import cann_ops_transformer
from chunk_gated_delta_rule_main import run_chunk_gated_delta_rule_eager
from test_chunk_gated_delta_rule_paramset_rdv import ENABLED_PARAMS_RDV

FIXED_SEED = int(os.environ.get("TORCH_SEED", "0"))
if FIXED_SEED:
    torch.manual_seed(FIXED_SEED)
    torch_npu.npu.manual_seed(FIXED_SEED)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, "rdv_report_r4_r5_r7.csv")

results = []
total = len(ENABLED_PARAMS_RDV)
pass_cnt = 0
fail_cnt = 0
error_cnt = 0

for idx, params in enumerate(ENABLED_PARAMS_RDV):
    name = params["_name"][0]
    B = params["B"][0]
    seqlen = params["seqlen"][0]
    nk = params["nk"][0]
    nv = params["nv"][0]
    dk = params["dk"][0]
    dv = params["dv"][0]
    has_g = params["has_g"][0]
    dt = params["data_type"][0]
    sdt = params["state_data_type"][0]
    contig = params["is_contiguous"][0]
    cs = params["chunk_size"][0]

    desc = f"B={B} S={seqlen} nk={nk} nv={nv} dk={dk} dv={dv} g={has_g} {str(dt).split('.')[-1]}/{str(sdt).split('.')[-1]} c={contig}"
    print(f"\n[{idx+1}/{total}] {name}: {desc}", flush=True)

    if FIXED_SEED:
        torch.manual_seed(FIXED_SEED + idx)
        torch_npu.npu.manual_seed(FIXED_SEED + idx)

    t0 = time.time()
    try:
        ret = run_chunk_gated_delta_rule_eager(
            B, seqlen, nk, nv, dk, dv, cs,
            data_type=dt, state_data_type=sdt, has_g=has_g, is_contiguous=contig,
        )
        status = "PASS" if ret else "FAIL"
        if ret:
            pass_cnt += 1
        else:
            fail_cnt += 1
    except Exception as e:
        err_msg = str(e)[:120]
        status = f"ERROR:{err_msg}"
        error_cnt += 1
        ret = False
    elapsed = time.time() - t0

    results.append({
        "name": name, "B": B, "seqlen": seqlen, "nk": nk, "nv": nv,
        "dk": dk, "dv": dv, "has_g": has_g,
        "data_type": str(dt).split('.')[-1],
        "state_type": str(sdt).split('.')[-1],
        "contiguous": contig, "status": status, "time_s": f"{elapsed:.1f}",
    })
    print(f"  -> {status} ({elapsed:.1f}s)", flush=True)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name","B","seqlen","nk","nv","dk","dv","has_g","data_type","state_type","contiguous","status","time_s"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n{'='*60}")
print(f"Total: {total}, PASS: {pass_cnt}, FAIL: {fail_cnt}, ERROR: {error_cnt}")
print(f"Report: {csv_path}")

import csv, glob, os, sys

def mean_split(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    aic, aiv = [], []
    for r in rows:
        sub = (r.get("sub_block_id") or "").lower()
        if "cube" in sub:
            aic.append(r)
        elif "vector" in sub:
            aiv.append(r)
    def m(rows, k):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        return (sum(vals) / len(vals) if vals else None, len(vals))
    out = {}
    for k in ["aic_time(us)", "aic_cube_ratio", "aic_scalar_ratio", "aic_mte1_ratio",
              "aic_mte2_ratio", "aic_fixpipe_ratio"]:
        out[k] = m(aic, k)
    for k in ["aiv_time(us)", "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
              "aiv_mte3_ratio"]:
        out[k] = m(aiv, k)
    return out


def dur(basic):
    best = None
    with open(basic, newline="") as f:
        for r in csv.DictReader(f):
            d = r.get("Task Duration(us)") or ""
            try:
                v = float(d)
            except Exception:
                continue
            if best is None or v > best:
                best = v
    return best


def main():
    cases = sys.argv[1:]
    if not cases:
        cases = [
            "BSA_MXFP8_keep1|/home/npu_user5/xuxuepeng/ops-transformer_0703/attention/block_sparse_attention/tests/mxfp8_pytest/bsa_msprof_perf_mxfp8_keep1/S118800_keep1.0",
            "FIA_MXFP8_dense|/home/npu_user5/xuxuepeng/ops-transformer_0703/attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/fia_msprof_perf_mxfp8_sameseq/S118800_sp0",
        ]
    for item in cases:
        name, d = item.split("|", 1)
        pipes = glob.glob(os.path.join(d, "OPPROF_*", "PipeUtilization.csv"))
        basics = glob.glob(os.path.join(d, "OPPROF_*", "OpBasicInfo.csv"))
        print("====", name)
        if not pipes:
            print(" NO PIPE", d)
            continue
        p = mean_split(pipes[0])
        print(" duration_us", dur(basics[0]) if basics else None)
        for k, (mean, n) in p.items():
            if mean is None:
                continue
            print(f"  {k}: {mean:.4f} (n={n})")


if __name__ == "__main__":
    main()

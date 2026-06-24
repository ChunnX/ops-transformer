#!/usr/bin/env python3
# Generate 1000 kernel test cases for MhcPreSinkhornBackward.
# Layout: B=1, S=bs, n=4 (fixed), iters=20 (sum/norm first dim = 40), d<=7168 (mult of 128).
import csv, sys

N = 4
ITERS = 20
TWO_IT = 2 * ITERS
HC_MIX = N * N + 2 * N            # 24
MAX_ELEM = 96 * 1024 * 1024      # cap on x elements (bs*N*d) to stay memory-safe under --pc 2

# dtype profiles: (grad_hin/x/grad_x dtype)
PROFILES = ["bfloat16", "float16"]

# per-output (rtol, ptol); grad_x depends on profile, fp32 grads fixed
TOL_FP32 = (0.0001, 0.0001)
TOL_GRADX = {"bfloat16": (0.005, 0.005), "float16": (0.001, 0.001)}
ATOL = 1e-5

# bs spanning 1..16384 (mix of powers, primes, round numbers)
BS_VALUES = [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 24, 31, 32, 48, 50, 64, 96, 100, 128, 160,
             200, 256, 300, 333, 384, 512, 600, 700, 777, 900, 1024, 1234, 1500, 1777,
             2048, 2500, 3000, 3333, 4000, 4096, 5000, 6000, 6144, 7168, 7777, 8192,
             9000, 10000, 11000, 12288, 12345, 13000, 14336, 15000, 16000, 16384]
# d: emphasize 4096 & 7168; all multiples of 128, <=7168
D_EMPH = [4096, 7168]
D_OTHER = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048,
           2304, 2560, 2816, 3072, 3584, 4608, 5120, 5632, 6144, 6656]


def shapes(bs, d):
    nd = N * d
    inp = [
        (1, bs, d),            # grad_hin
        (1, bs, N),            # grad_h_post
        (1, bs, N, N),         # grad_h_res
        (1, bs, N, d),         # x
        (HC_MIX, nd),          # phi
        (3,),                  # alpha
        (HC_MIX,),             # bias
        (1, bs, N),            # h_pre
        (1, bs, HC_MIX),       # hc_before_norm
        (1, bs, 1),            # inv_rms
        (TWO_IT, 1, bs, N),    # sum_out
        (TWO_IT, 1, bs, N, N), # norm_out
    ]
    out = [(1, bs, N, d), (HC_MIX, nd), (3,), (HC_MIX,)]
    return inp, out


def dtypes(prof):
    p = prof
    inp = (p, 'float32', 'float32', p, 'float32', 'float32', 'float32',
           'float32', 'float32', 'float32', 'float32', 'float32')
    out = (p, 'float32', 'float32', 'float32')
    return inp, out


RANGES = "((-1,1),(-1,1),(-1,1),(-2,2),(-0.0078,0.0078),(-2,2),(-2,2),(0.01,1),(-2,2),(0.1,2),(0.5,2),(0.05,1))"


def tol_str(prof):
    gx = TOL_GRADX[prof]
    return f"({gx},{TOL_FP32},{TOL_FP32},{TOL_FP32})".replace(" ", "")


def tup(t):
    return "(" + ",".join(str(x) for x in t) + ("," if len(t) == 1 else "") + ")"


def shapes_str(lst):
    return "(" + ",".join(tup(s) for s in lst) + ")"


# Build (bs, d) pool with emphasis on 4096/7168, then both dtypes; cap by MAX_ELEM.
pairs = []
# emphasized d: pair with a broad bs sweep
for d in D_EMPH:
    for bs in BS_VALUES:
        if bs * N * d <= MAX_ELEM:
            pairs.append((bs, d))
# other d
for d in D_OTHER:
    for bs in BS_VALUES:
        if bs * N * d <= MAX_ELEM:
            pairs.append((bs, d))

rows = []
seen = set()
# interleave dtypes so both well represented; loop pairs repeatedly until 1000
i = 0
while len(rows) < 1000:
    bs, d = pairs[i % len(pairs)]
    prof = PROFILES[(i // 1) % 2] if (i % 2 == 0) else PROFILES[1 - ((i // 1) % 2)]
    prof = PROFILES[i % 2]
    key = (bs, d, prof)
    i += 1
    if key in seen:
        if i > len(pairs) * 4:   # exhausted unique combos
            break
        continue
    seen.add(key)
    inp_s, out_s = shapes(bs, d)
    inp_d, out_d = dtypes(prof)
    name = f"mpsb_{prof[:2]}_bs{bs}_d{d}"
    rows.append({
        "testcase_name": name,
        "op_name": "mhc_pre_sinkhorn_backward",
        "input_shapes": shapes_str(inp_s),
        "input_dtypes": str(inp_d).replace(" ", ""),
        "input_formats": "('ND',)",
        "output_shapes": shapes_str(out_s),
        "output_dtypes": str(out_d).replace(" ", ""),
        "output_formats": "('ND',)",
        "attributes": "{'hc_eps':1e-06}",
        "input_data_ranges": RANGES,
        "precision_tolerances": tol_str(prof),
        "absolute_precision": ATOL,
    })

cols = ["testcase_name", "op_name", "input_shapes", "input_dtypes", "input_formats",
        "output_shapes", "output_dtypes", "output_formats", "attributes",
        "input_data_ranges", "precision_tolerances", "absolute_precision"]
out_path = sys.argv[1] if len(sys.argv) > 1 else "cases.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# stats
import collections
ds = collections.Counter(r["testcase_name"].split("_d")[1] for r in rows)
ps = collections.Counter(r["testcase_name"].split("_")[1] for r in rows)
print(f"generated {len(rows)} cases -> {out_path}")
print("by dtype:", dict(ps))
print("d=4096:", ds.get('4096'), " d=7168:", ds.get('7168'))
bsset = sorted({int(r['testcase_name'].split('_bs')[1].split('_d')[0]) for r in rows})
print("bs range:", min(bsset), "..", max(bsset), f"({len(bsset)} distinct)")

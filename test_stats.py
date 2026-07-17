import numpy as np
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')
total_exact = 0; total_partial = 0; total_none = 0
total_count_mismatch = 0
for b in range(8192):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    if s == r: total_exact += 1
    elif s & r: total_partial += 1
    else: total_none += 1
    if len(s) != len(r): total_count_mismatch += 1
print(f"Exact: {total_exact}/8192")
print(f"Partial: {total_partial}/8192")
print(f"None: {total_none}/8192")
print(f"Count mismatch: {total_count_mismatch}/8192")

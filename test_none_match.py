import numpy as np
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')
none_batches = []
count_mismatch_batches = []
for b in range(8192):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    if len(s) != len(r):
        count_mismatch_batches.append((b, len(s), len(r)))
    if len(s & r) == 0 and len(s) > 0 and len(r) > 0:
        none_batches.append(b)
print(f"None match batches: {len(none_batches)}")
if none_batches:
    print(f"  First 10: {none_batches[:10]}")
    b = none_batches[0]
    s = sort[b,0][sort[b,0]!=-1]
    r = radix[b,0][radix[b,0]!=-1]
    print(f"  Batch {b}: sort={len(s)} valid, radix={len(r)} valid")
    print(f"    sort first 5: {s[:5].tolist()}")
    print(f"    radix first 5: {r[:5].tolist()}")

print(f"\nCount mismatch batches: {len(count_mismatch_batches)}")
if count_mismatch_batches:
    for b, s_cnt, r_cnt in count_mismatch_batches[:10]:
        print(f"  Batch {b}: sort={s_cnt}, radix={r_cnt}")

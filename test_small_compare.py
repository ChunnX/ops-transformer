import numpy as np

sort = np.load('/tmp/sort_small.npy')
radix = np.load('/tmp/radixtopk_small.npy')

print(f"Shape: sort={sort.shape}, radix={radix.shape}")
print(f"Sort valid: {len(sort[sort!=-1])}, Radix valid: {len(radix[radix!=-1])}")

exact = 0; partial = 0; none = 0; count_mismatch = 0
overlaps = []
for b in range(sort.shape[0]):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    ov = len(s & r)
    overlaps.append(ov)
    if s == r: exact += 1
    elif ov > 0: partial += 1
    else: none += 1
    if len(s) != len(r): count_mismatch += 1

overlaps = np.array(overlaps)
print(f"\n=== Overall (k={sort.shape[2]}) ===")
print(f"Exact: {exact}/{sort.shape[0]}")
print(f"Partial: {partial}/{sort.shape[0]}")
print(f"None: {none}/{sort.shape[0]}")
print(f"Count mismatch: {count_mismatch}/{sort.shape[0]}")
print(f"Overlap: min={overlaps.min()}, max={overlaps.max()}, mean={overlaps.mean():.1f}, median={np.median(overlaps):.0f}")

# By cuRealAcSeq range (cuRealAcSeq = batch_id + 1 for sparse_mode=3)
print(f"\n=== By cuRealAcSeq range ===")
k = sort.shape[2]
ranges = [(1, k), (k+1, 2*k), (2*k+1, 4*k), (4*k+1, 1024)]
for lo, hi in ranges:
    batches = [b for b in range(sort.shape[0]) if lo <= (b+1) <= hi]
    if not batches: continue
    ov = np.array([len(set(sort[b,0][sort[b,0]!=-1].tolist()) & set(radix[b,0][radix[b,0]!=-1].tolist())) for b in batches])
    ex = sum(1 for b in batches if set(sort[b,0][sort[b,0]!=-1].tolist()) == set(radix[b,0][radix[b,0]!=-1].tolist()))
    print(f"  [{lo:4d},{hi:4d}]: {len(batches):4d} batches, exact={ex:4d}, "
          f"overlap min={ov.min():3d} max={ov.max():3d} mean={ov.mean():6.1f}")

# Sample batches
print(f"\n=== Sample batches ===")
for b in [0, 50, 127, 128, 200, 500, 1000, 1023]:
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    ov = len(s & r)
    print(f"  batch {b:4d} (cuRealAcSeq={b+1:4d}): sort={len(s):3d}, radix={len(r):3d}, overlap={ov:3d}/{k}")

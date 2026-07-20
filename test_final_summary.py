import numpy as np

sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# Overall stats
exact = 0; partial = 0; none = 0; count_mismatch = 0
for b in range(8192):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    if s == r: exact += 1
    elif s & r: partial += 1
    else: none += 1
    if len(s) != len(r): count_mismatch += 1

print(f"=== Overall ===")
print(f"Exact: {exact}/8192")
print(f"Partial: {partial}/8192")
print(f"None: {none}/8192")
print(f"Count mismatch: {count_mismatch}/8192")

# By cuRealAcSeq range
print(f"\n=== By cuRealAcSeq range ===")
ranges = [(1, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192)]
for lo, hi in ranges:
    batches = [b for b in range(8192) if lo <= (b+1) <= hi]
    if not batches: continue
    overlaps = []
    for b in batches:
        s = set(sort[b,0][sort[b,0]!=-1].tolist())
        r = set(radix[b,0][radix[b,0]!=-1].tolist())
        overlaps.append(len(s & r))
    overlaps = np.array(overlaps)
    print(f"  [{lo:4d},{hi:4d}]: {len(batches):4d} batches, "
          f"overlap min={overlaps.min():3d} max={overlaps.max():3d} "
          f"mean={overlaps.mean():6.1f} median={np.median(overlaps):5.0f}")

# Detailed batch examples
print(f"\n=== Detailed batch examples ===")
for b in [0, 100, 511, 512, 600, 1000, 2000, 5000, 8191]:
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    ov = len(s & r)
    cuRealAcSeq = b + 1
    need_sort = cuRealAcSeq > 512
    print(f"  batch {b:4d} (cuRealAcSeq={cuRealAcSeq:4d}, sort_needed={need_sort}): "
          f"overlap={ov:3d}/512, sort_only={len(s-r):3d}, radix_only={len(r-s):3d}")

# Batch 600 detailed
print(f"\n=== Batch 600 detailed (cuRealAcSeq=601) ===")
s_set = set(sort[600,0][sort[600,0]!=-1].tolist())
r_set = set(radix[600,0][radix[600,0]!=-1].tolist())
s_only = sorted(s_set - r_set)
r_only = sorted(r_set - s_set)
print(f"  Sort: {len(s_set)} indices, {len(s_only)} unique to Sort")
print(f"  Radix: {len(r_set)} indices, {len(r_only)} unique to Radix")
print(f"  Overlap: {len(s_set & r_set)}")
print(f"  Sort-only (first 10): {s_only[:10]}")
print(f"  Radix-only (first 10): {r_only[:10]}")

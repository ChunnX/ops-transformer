import numpy as np
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# Exclude count-mismatch batches (sort=1)
valid_batches = []
for b in range(8192):
    s_cnt = len(sort[b,0][sort[b,0]!=-1])
    r_cnt = len(radix[b,0][radix[b,0]!=-1])
    if s_cnt == r_cnt and s_cnt > 0:
        valid_batches.append(b)

print(f"Valid batches: {len(valid_batches)}/8192\n")

# Categorize by cuRealAcSeq (which determines how many elements to choose from)
# cuRealAcSeq for TND attenMask mode 3: cuRealAcSeq = actS2 - (actS1 - s1Idx) + 1
# actS1 = actS2 = 8192, so cuRealAcSeq = 8192 - (8192 - s1Idx) + 1 = s1Idx + 1
# batch_id == s1Idx, so cuRealAcSeq = batch_id + 1

print("=== Overlap by cuRealAcSeq (= batch_id + 1) ===")
print(f"{'batch':>6s} {'cuRealAcSeq':>12s} {'need_sort':>10s} {'overlap':>8s} {'sort_only':>10s} {'radix_only':>10s} {'sort_first5':>40s} {'radix_first5':>40s}")

# Sample batches at different cuRealAcSeq levels
sample_batches = [0, 9, 49, 99, 199, 299, 399, 499, 510, 511, 512, 513, 520, 600, 1000, 2000, 5000, 8000, 8191]
for b in sample_batches:
    if b not in valid_batches: continue
    s_vals = sort[b,0][sort[b,0]!=-1]
    r_vals = radix[b,0][radix[b,0]!=-1]
    s_set = set(s_vals.tolist())
    r_set = set(r_vals.tolist())
    ov = len(s_set & r_set)
    cuRealAcSeq = b + 1
    need_sort = cuRealAcSeq > 512
    sort_only = len(s_set - r_set)
    radix_only = len(r_set - s_set)
    s5 = sorted(s_set)[:5]
    r5 = sorted(r_set)[:5]
    print(f"{b:6d} {cuRealAcSeq:12d} {'yes' if need_sort else 'no':>10s} {ov:8d} {sort_only:10d} {radix_only:10d}   {str(s5):>40s}   {str(r5):>40s}")

# Detailed look at a batch where cuRealAcSeq > 512 (actual sorting needed)
print("\n=== Batch 600 (cuRealAcSeq=601, need to select 512 from 601) ===")
b = 600
s_vals = sorted(set(sort[b,0][sort[b,0]!=-1].tolist()))
r_vals = sorted(set(radix[b,0][radix[b,0]!=-1].tolist()))
s_set = set(s_vals)
r_set = set(r_vals)
print(f"  Sort: {len(s_set)} valid, range [{min(s_vals)}, {max(s_vals)}]")
print(f"  Radix: {len(r_set)} valid, range [{min(r_vals)}, {max(r_vals)}]")
print(f"  Overlap: {len(s_set & r_set)}")
s_only = sorted(s_set - r_set)
r_only = sorted(r_set - s_set)
print(f"  Sort only ({len(s_only)}): {s_only[:20]}")
print(f"  Radix only ({len(r_only)}): {r_only[:20]}")

# Batch 1000
print("\n=== Batch 1000 (cuRealAcSeq=1001, select 512 from 1001) ===")
b = 1000
s_vals = sorted(set(sort[b,0][sort[b,0]!=-1].tolist()))
r_vals = sorted(set(radix[b,0][radix[b,0]!=-1].tolist()))
s_set = set(s_vals)
r_set = set(r_vals)
print(f"  Sort: {len(s_set)} valid, range [{min(s_vals)}, {max(s_vals)}]")
print(f"  Radix: {len(r_set)} valid, range [{min(r_vals)}, {max(r_vals)}]")
print(f"  Overlap: {len(s_set & r_set)}")
s_only = sorted(s_set - r_set)
r_only = sorted(r_set - s_set)
print(f"  Sort only ({len(s_only)}): {s_only[:20]}")
print(f"  Radix only ({len(r_only)}): {r_only[:20]}")

# Batch 5000
print("\n=== Batch 5000 (cuRealAcSeq=5001, select 512 from 5001) ===")
b = 5000
s_vals = sorted(set(sort[b,0][sort[b,0]!=-1].tolist()))
r_vals = sorted(set(radix[b,0][radix[b,0]!=-1].tolist()))
s_set = set(s_vals)
r_set = set(r_vals)
print(f"  Sort: {len(s_set)} valid, range [{min(s_vals)}, {max(s_vals)}]")
print(f"  Radix: {len(r_set)} valid, range [{min(r_vals)}, {max(r_vals)}]")
print(f"  Overlap: {len(s_set & r_set)}")
s_only = sorted(s_set - r_set)
r_only = sorted(r_set - s_set)
print(f"  Sort only ({len(s_only)}): {s_only[:20]}")
print(f"  Radix only ({len(r_only)}): {r_only[:20]}")

# Statistics by cuRealAcSeq range
print("\n=== Overlap stats by cuRealAcSeq range ===")
ranges = [(1, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192)]
for lo, hi in ranges:
    batch_in_range = [b for b in valid_batches if lo <= (b+1) <= hi]
    if not batch_in_range: continue
    overlaps = []
    for b in batch_in_range:
        s = set(sort[b,0][sort[b,0]!=-1].tolist())
        r = set(radix[b,0][radix[b,0]!=-1].tolist())
        overlaps.append(len(s & r))
    overlaps = np.array(overlaps)
    print(f"  cuRealAcSeq [{lo:4d}, {hi:4d}]: {len(batch_in_range):4d} batches, "
          f"overlap min={overlaps.min():3d} max={overlaps.max():3d} "
          f"mean={overlaps.mean():6.1f} median={np.median(overlaps):5.0f}")

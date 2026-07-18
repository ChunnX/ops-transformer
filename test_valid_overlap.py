import numpy as np
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# Exclude the 136 count-mismatch batches (sort=1, radix=512)
valid_batches = []
for b in range(8192):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    if len(s) == len(r) and len(s) > 0:
        valid_batches.append(b)

print(f"Valid batches (same count, both non-empty): {len(valid_batches)}/8192")

exact = 0; partial = 0; none = 0
overlaps = []
for b in valid_batches:
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    ov = len(s & r)
    overlaps.append(ov)
    if s == r: exact += 1
    elif ov > 0: partial += 1
    else: none += 1

overlaps = np.array(overlaps)
print(f"\n=== Match categories ===")
print(f"Exact: {exact}/{len(valid_batches)}")
print(f"Partial: {partial}/{len(valid_batches)}")
print(f"None: {none}/{len(valid_batches)}")

print(f"\n=== Overlap stats (out of 512) ===")
print(f"  min={overlaps.min()}, max={overlaps.max()}, mean={overlaps.mean():.1f}, median={np.median(overlaps):.0f}")
print(f"  Distribution:")
for threshold in [512, 511, 505, 500, 450, 400, 300, 200, 100, 50, 10, 0]:
    count = (overlaps >= threshold).sum()
    print(f"    >= {threshold}: {count}/{len(valid_batches)} ({count/len(valid_batches)*100:.1f}%)")

# Show some examples at different overlap levels
print(f"\n=== Examples by overlap level ===")
for target in [512, 511, 500, 400, 300, 200, 100, 50, 10]:
    for b in valid_batches:
        s = set(sort[b,0][sort[b,0]!=-1].tolist())
        r = set(radix[b,0][radix[b,0]!=-1].tolist())
        ov = len(s & r)
        if ov == target or (target < 512 and ov >= target and ov < target + 5):
            diff_s = sorted(s - r)[:5]
            diff_r = sorted(r - s)[:5]
            print(f"  Batch {b}: overlap={ov}/512, sort_only={diff_s}, radix_only={diff_r}")
            break

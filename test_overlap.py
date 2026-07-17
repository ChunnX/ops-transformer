import numpy as np
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# For each batch, compute overlap ratio
overlaps = []
for b in range(8192):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    overlap = len(s & r)
    overlaps.append(overlap)

overlaps = np.array(overlaps)
print(f"Overlap stats (out of 512):")
print(f"  min={overlaps.min()}, max={overlaps.max()}, mean={overlaps.mean():.1f}, median={np.median(overlaps):.0f}")
print(f"  Distribution:")
for threshold in [512, 500, 400, 300, 200, 100, 0]:
    count = (overlaps >= threshold).sum()
    print(f"    >= {threshold}: {count}/8192 ({count/8192*100:.1f}%)")

# Check score values for a mismatched batch
b = 512
s = set(sort[b,0][sort[b,0]!=-1].tolist())
r = set(radix[b,0][radix[b,0]!=-1].tolist())
diff_s = sorted(s - r)
diff_r = sorted(r - s)
print(f"\nBatch {b}: overlap={len(s&r)}, sort_only={diff_s}, radix_only={diff_r}")

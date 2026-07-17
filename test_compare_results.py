import numpy as np

radix = np.load('/tmp/radixtopk_result.npy')
sort = np.load('/tmp/sort_result.npy')

print(f"RadixTopK shape: {radix.shape}")
print(f"Sort shape: {sort.shape}")

# Flatten
radix_flat = radix.flatten()
sort_flat = sort.flatten()

# Overall comparison
radix_valid = radix_flat[radix_flat != -1]
sort_valid = sort_flat[sort_flat != -1]
print(f"\nRadixTopK: {len(radix_valid)} valid, min={radix_valid.min()}, max={radix_valid.max()}")
print(f"Sort:      {len(sort_valid)} valid, min={sort_valid.min()}, max={sort_valid.max()}")

# Per-batch comparison
total_batches = radix.shape[0] * radix.shape[1]
kValue = radix.shape[2]
match_count = 0
mismatch_count = 0
partial_match = 0

for b in range(total_batches):
    r_batch = set(radix_flat[b*kValue:(b+1)*kValue].tolist())
    s_batch = set(sort_flat[b*kValue:(b+1)*kValue].tolist())
    r_valid = r_batch - {-1}
    s_valid = s_batch - {-1}
    
    if r_valid == s_valid:
        match_count += 1
    elif len(r_valid & s_valid) > 0:
        partial_match += 1
    else:
        mismatch_count += 1

print(f"\nPer-batch comparison:")
print(f"  Exact match: {match_count}/{total_batches}")
print(f"  Partial match: {partial_match}/{total_batches}")
print(f"  No match: {mismatch_count}/{total_batches}")

# Detailed look at a few batches
for b in [0, 100, 500, 1000, 5000, 8000, 8191]:
    r_set = set(radix_flat[b*kValue:(b+1)*kValue].tolist()) - {-1}
    s_set = set(sort_flat[b*kValue:(b+1)*kValue].tolist()) - {-1}
    overlap = len(r_set & s_set)
    print(f"  batch {b}: radix={len(r_set)}, sort={len(s_set)}, overlap={overlap}/{max(len(r_set),len(s_set))}")

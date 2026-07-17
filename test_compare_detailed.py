import numpy as np

radix = np.load('/tmp/radixtopk_result.npy')
sort = np.load('/tmp/sort_result.npy')

radix_flat = radix.flatten()
sort_flat = sort.flatten()

total_batches = radix.shape[0] * radix.shape[1]
kValue = radix.shape[2]

# Categorize batches
exact_match = 0
partial_match = 0
no_match = 0
count_mismatch = 0

# Analyze valid_count difference
radix_valid_total = 0
sort_valid_total = 0
count_diff_batches = []

for b in range(total_batches):
    r_batch = radix_flat[b*kValue:(b+1)*kValue]
    s_batch = sort_flat[b*kValue:(b+1)*kValue]
    r_valid = r_batch[r_batch != -1]
    s_valid = s_batch[s_batch != -1]
    r_set = set(r_valid.tolist())
    s_set = set(s_valid.tolist())
    
    radix_valid_total += len(r_valid)
    sort_valid_total += len(s_valid)
    
    if len(r_valid) != len(s_valid):
        count_diff_batches.append((b, len(r_valid), len(s_valid)))
    
    if r_set == s_set:
        exact_match += 1
    elif len(r_set & s_set) > 0:
        partial_match += 1
    else:
        no_match += 1

print(f"RadixTopK valid: {radix_valid_total}")
print(f"Sort valid: {sort_valid_total}")
print(f"Difference: {sort_valid_total - radix_valid_total}")
print(f"\nExact match: {exact_match}/{total_batches}")
print(f"Partial match: {partial_match}/{total_batches}")
print(f"No match: {no_match}/{total_batches}")
print(f"Count mismatch batches: {len(count_diff_batches)}")

# Show count differences
if len(count_diff_batches) > 0:
    print(f"\nCount diff examples (first 20):")
    for b, r, s in count_diff_batches[:20]:
        print(f"  batch {b}: radix={r}, sort={s}, diff={s-r}")

# Analyze overlap by cuRealAcSeq range
# cuRealAcSeq = actS2 - (actS1 - s1Idx) + 1 = s1Idx + 1 (for actS1=actS2=8192)
print(f"\nOverlap by cuRealAcSeq range:")
ranges = [(1, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192)]
for lo, hi in ranges:
    match_count = 0
    total_count = 0
    overlap_sum = 0
    for b in range(total_batches):
        cuRealAcSeq = b + 1  # s1Idx = b, cuRealAcSeq = b + 1
        if cuRealAcSeq < lo or cuRealAcSeq > hi:
            continue
        total_count += 1
        r_set = set(radix_flat[b*kValue:(b+1)*kValue].tolist()) - {-1}
        s_set = set(sort_flat[b*kValue:(b+1)*kValue].tolist()) - {-1}
        if r_set == s_set:
            match_count += 1
        overlap_sum += len(r_set & s_set)
    if total_count > 0:
        avg_overlap = overlap_sum / total_count
        print(f"  cuRealAcSeq [{lo},{hi}]: {total_count} batches, exact={match_count}, avg_overlap={avg_overlap:.1f}/512")

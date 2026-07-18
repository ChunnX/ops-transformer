import numpy as np
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# Find all count mismatch and none match batches
count_mismatch = []
none_match = []
for b in range(8192):
    s = set(sort[b,0][sort[b,0]!=-1].tolist())
    r = set(radix[b,0][radix[b,0]!=-1].tolist())
    if len(s) != len(r):
        count_mismatch.append((b, len(s), len(r)))
    if len(s & r) == 0 and len(s) > 0 and len(r) > 0:
        none_match.append(b)

print(f"=== Count mismatch: {len(count_mismatch)} batches ===")
# Group by sort count
from collections import Counter
sort_counts = Counter(s for _, s, _ in count_mismatch)
print(f"Sort valid count distribution: {dict(sort_counts)}")
radix_counts = Counter(r for _, _, r in count_mismatch)
print(f"Radix valid count distribution: {dict(radix_counts)}")

print(f"\n=== Detailed: first 20 count mismatch batches ===")
for b, sc, rc in count_mismatch[:20]:
    s_vals = sort[b,0][sort[b,0]!=-1].tolist()
    r_vals = radix[b,0][radix[b,0]!=-1].tolist()
    print(f"  Batch {b}: sort={sc} valid {s_vals[:5]}, radix={rc} valid {r_vals[:5]}")

print(f"\n=== None match: {len(none_match)} batches ===")
print(f"  First 20: {none_match[:20]}")
# Check if none_match batches overlap with count_mismatch
cm_set = set(b for b, _, _ in count_mismatch)
nm_in_cm = [b for b in none_match if b in cm_set]
nm_not_in_cm = [b for b in none_match if b not in cm_set]
print(f"  In count_mismatch: {len(nm_in_cm)}")
print(f"  Not in count_mismatch: {len(nm_not_in_cm)}")

if nm_not_in_cm:
    print(f"\n=== None match but same count (first 5) ===")
    for b in nm_not_in_cm[:5]:
        s_vals = sort[b,0][sort[b,0]!=-1].tolist()
        r_vals = radix[b,0][radix[b,0]!=-1].tolist()
        print(f"  Batch {b}: sort={len(s_vals)} radix={len(r_vals)}")
        print(f"    sort first 10: {s_vals[:10]}")
        print(f"    radix first 10: {r_vals[:10]}")

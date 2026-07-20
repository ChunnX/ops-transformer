import numpy as np

sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# Load scores
scores_0_511 = np.load('/tmp/batch600_scores_0_511.npy')
scores_512_1023 = np.load('/tmp/batch600_scores_512_1023.npy')
all_scores = np.concatenate([scores_0_511, scores_512_1023])
valid_scores = all_scores[:601]
invalid_mask = valid_scores > 1e29
invalid_indices = set(np.where(invalid_mask)[0].tolist())

s_set = set(sort[600,0][sort[600,0]!=-1].tolist())
r_set = set(radix[600,0][radix[600,0]!=-1].tolist())

print(f"Invalid indices (score=+1e30): {len(invalid_indices)}")
print(f"Sort selected invalid: {len(s_set & invalid_indices)}")
print(f"Radix selected invalid: {len(r_set & invalid_indices)}")
print(f"Sort valid: {len(s_set - invalid_indices)}")
print(f"Radix valid: {len(r_set - invalid_indices)}")
print(f"Overlap (valid only): {len((s_set - invalid_indices) & (r_set - invalid_indices))}")

# Check: if we exclude invalid indices, what's the overlap?
s_valid = s_set - invalid_indices
r_valid = r_set - invalid_indices
print(f"\n=== After excluding invalid indices ===")
print(f"Sort valid count: {len(s_valid)}")
print(f"Radix valid count: {len(r_valid)}")
print(f"Overlap: {len(s_valid & r_valid)}")
print(f"Sort only: {len(s_valid - r_valid)}")
print(f"Radix only: {len(r_valid - r_valid)}")

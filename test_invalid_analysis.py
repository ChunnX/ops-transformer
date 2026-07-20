import numpy as np

sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

scores_0_511 = np.load('/tmp/batch600_scores_0_511.npy')
scores_512_1023 = np.load('/tmp/batch600_scores_512_1023.npy')
all_scores = np.concatenate([scores_0_511, scores_512_1023])
valid_scores = all_scores[:601]
invalid_mask = valid_scores > 1e29
invalid_indices = set(np.where(invalid_mask)[0].tolist())

s_set = set(sort[600,0][sort[600,0]!=-1].tolist())
r_set = set(radix[600,0][radix[600,0]!=-1].tolist())

# Check what indices RadixTopK selected that are in the invalid range [88~511]
r_in_invalid_range = {i for i in r_set if 88 <= i <= 511}
r_outside_invalid_range = {i for i in r_set if i < 88 or i > 511}
s_in_invalid_range = {i for i in s_set if 88 <= i <= 511}
s_outside_invalid_range = {i for i in s_set if i < 88 or i > 511}

print(f"=== RadixTopK ===")
print(f"  Total selected: {len(r_set)}")
print(f"  In invalid range [88~511]: {len(r_in_invalid_range)}")
print(f"  Outside invalid range: {len(r_outside_invalid_range)}")
print(f"  First 10 in invalid range: {sorted(r_in_invalid_range)[:10]}")
print(f"  Their scores: {[f'{valid_scores[i]:.2f}' for i in sorted(r_in_invalid_range)[:10]]}")

print(f"\n=== Sort ===")
print(f"  Total selected: {len(s_set)}")
print(f"  In invalid range [88~511]: {len(s_in_invalid_range)}")
print(f"  Outside invalid range: {len(s_outside_invalid_range)}")
print(f"  First 10 in invalid range: {sorted(s_in_invalid_range)[:10]}")
print(f"  Their scores: {[f'{valid_scores[i]:.2f}' for i in sorted(s_in_invalid_range)[:10]]}")

# Overlap outside invalid range
overlap_outside = len(r_outside_invalid_range & s_outside_invalid_range)
print(f"\n=== Overlap (outside invalid range) ===")
print(f"  Radix: {len(r_outside_invalid_range)}, Sort: {len(s_outside_invalid_range)}")
print(f"  Overlap: {overlap_outside}")

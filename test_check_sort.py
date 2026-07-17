import numpy as np

sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

# Compare batch 512
s_batch = sort[512, 0]
r_batch = radix[512, 0]

s_valid = set(s_batch[s_batch != -1].tolist())
r_valid = set(r_batch[r_batch != -1].tolist())

print(f"Sort valid: {len(s_valid)}, Radix valid: {len(r_valid)}")
print(f"In Sort but not Radix: {sorted(s_valid - r_valid)}")
print(f"In Radix but not Sort: {sorted(r_valid - s_valid)}")
print(f"Common: {len(s_valid & r_valid)}")

# The score for index 512 is large positive (10280463.0 as float)
# In bf16, this becomes 0x6B40 (approximately)
# Let's check what bf16 values the sort baseline has
# We need to check if multiple scores map to the same bf16 value

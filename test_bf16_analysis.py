import numpy as np

# Load scores
scores_0_511 = np.load('/tmp/batch600_scores_0_511.npy')
scores_512_1023 = np.load('/tmp/batch600_scores_512_1023.npy')
all_scores = np.concatenate([scores_0_511, scores_512_1023])

# cuRealAcSeq=601, valid indices are 0~600
valid_scores = all_scores[:601].copy()

# Mark +1e30 as invalid
invalid_mask = valid_scores > 1e29
n_invalid = invalid_mask.sum()
print(f"=== Batch 600 score analysis ===")
print(f"Total scores: {len(valid_scores)} (index 0~600)")
print(f"Invalid (+1e30): {n_invalid}")
print(f"Valid: {len(valid_scores) - n_invalid}")
print(f"Invalid indices: {np.where(invalid_mask)[0].tolist()[:20]}...")
print(f"Score range (valid): [{valid_scores[~invalid_mask].min():.4f}, {valid_scores[~invalid_mask].max():.4f}]")

# Step 2a: Float ground truth (select smallest 512 from valid scores)
# Replace +1e30 with +inf so they're never selected as "smallest"
float_scores = valid_scores.copy()
float_scores[invalid_mask] = np.inf
float_top512 = set(np.argsort(float_scores)[:512].tolist())
print(f"\n=== Float ground truth ===")
print(f"  Top-512 smallest indices: {len(float_top512)}")
print(f"  Contains invalid indices? {len(float_top512 & set(np.where(invalid_mask)[0].tolist()))}")

# Step 2b: BF16 ground truth
# Simulate Cast(float -> bf16, CAST_ROUND) then back to float
def float_to_bf16_to_float(f):
    """Simulate bf16 truncation (round to nearest even)"""
    bits = np.float32(f).view(np.uint32)
    # bf16 = top 16 bits of float32, with round-to-nearest-even
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)  # round to nearest even
    bf16_bits = (bits + rounding_bias) >> 16
    # Convert back to float32
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)

bf16_scores = np.array([float_to_bf16_to_float(s) for s in float_scores], dtype=np.float32)
bf16_top512 = set(np.argsort(bf16_scores)[:512].tolist())
print(f"\n=== BF16 ground truth ===")
print(f"  Top-512 smallest indices: {len(bf16_top512)}")
print(f"  Contains invalid indices? {len(bf16_top512 & set(np.where(invalid_mask)[0].tolist()))}")

# Step 3: Compare float vs bf16
overlap_float_bf16 = len(float_top512 & bf16_top512)
print(f"\n=== Float vs BF16 comparison ===")
print(f"  Overlap: {overlap_float_bf16}/512")
print(f"  In float but not bf16: {len(float_top512 - bf16_top512)}")
print(f"  In bf16 but not float: {len(bf16_top512 - float_top512)}")

# Step 4: Compare with RadixTopK actual output
# Need to load radixtopk_result.npy (clean, no debug)
# But current build has debug dump for batch 600
# We need to re-run without debug to get clean radix result
# For now, use the sort_result for comparison
sort = np.load('/tmp/sort_result.npy')
sort_top512 = set(sort[600, 0][sort[600, 0] != -1].tolist())
print(f"\n=== Sort result ===")
print(f"  Sort top-512 count: {len(sort_top512)}")

# Compare float vs sort
overlap_float_sort = len(float_top512 & sort_top512)
print(f"  Float vs Sort overlap: {overlap_float_sort}/512")

# Compare bf16 vs sort
overlap_bf16_sort = len(bf16_top512 & sort_top512)
print(f"  BF16 vs Sort overlap: {overlap_bf16_sort}/512")

# Check if sort selected any invalid indices
sort_invalid = sort_top512 & set(np.where(invalid_mask)[0].tolist())
print(f"  Sort contains invalid indices? {len(sort_invalid)}: {sorted(sort_invalid)[:10]}")

# Check sort scores
sort_scores = [float_scores[i] if i < len(float_scores) else 0 for i in sorted(sort_top512)]
print(f"  Sort score range: [{min(sort_scores):.4f}, {max(sort_scores):.4f}]")

import numpy as np

sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')

scores_0_511 = np.load('/tmp/batch600_scores_0_511.npy')
scores_512_1023 = np.load('/tmp/batch600_scores_512_1023.npy')
all_scores = np.concatenate([scores_0_511, scores_512_1023])
valid_scores = all_scores[:601]
invalid_mask = valid_scores > 1e29

s_set = set(sort[600,0][sort[600,0]!=-1].tolist())
r_set = set(radix[600,0][radix[600,0]!=-1].tolist())

# RadixTopK selected but Sort didn't
r_only = sorted(r_set - s_set)
# Sort selected but RadixTopK didn't
s_only = sorted(s_set - r_set)

print("=== RadixTopK selected but Sort didn't (31 indices) ===")
r_only_valid = [i for i in r_only if not invalid_mask[i]]
r_only_invalid = [i for i in r_only if invalid_mask[i]]
print(f"  Valid: {len(r_only_valid)}: {r_only_valid}")
print(f"  Invalid: {len(r_only_invalid)}: {r_only_invalid[:10]}...")
if r_only_valid:
    print(f"  Their scores: {[f'{valid_scores[i]:.2f}' for i in r_only_valid]}")

print(f"\n=== Sort selected but RadixTopK didn't (31 indices) ===")
s_only_valid = [i for i in s_only if not invalid_mask[i]]
s_only_invalid = [i for i in s_only if invalid_mask[i]]
print(f"  Valid: {len(s_only_valid)}: {s_only_valid}")
print(f"  Invalid: {len(s_only_invalid)}: {s_only_invalid[:10]}...")
if s_only_valid:
    print(f"  Their scores: {[f'{valid_scores[i]:.2f}' for i in s_only_valid]}")

# Which tile do the r_only_valid indices belong to?
print(f"\n=== Tile analysis of RadixTopK-only valid indices ===")
for i in r_only_valid:
    tile = 0 if i < 512 else 1
    print(f"  index {i:3d} (tile {tile}), score={valid_scores[i]:.4f}")

# Which tile do the s_only_invalid indices belong to?
print(f"\n=== Tile analysis of Sort-only invalid indices ===")
for i in s_only_invalid[:31]:
    tile = 0 if i < 512 else 1
    print(f"  index {i:3d} (tile {tile}), score={valid_scores[i]:.4f}")

# Check: Sort's virTopK parameter
# Sort selects virTopK per tile, then merges
# If virTopK < 512, some valid values in tile 0 might be dropped
print(f"\n=== Valid scores in tile 0 (index 0~87) ===")
tile0_valid = [(i, valid_scores[i]) for i in range(88) if not invalid_mask[i]]
print(f"  Count: {len(tile0_valid)}")
print(f"  Scores: {[f'{s:.2f}' for _, s in tile0_valid]}")

print(f"\n=== Valid scores in tile 1 (index 512~600) ===")
tile1_valid = [(i, valid_scores[i]) for i in range(512, 601) if not invalid_mask[i]]
print(f"  Count: {len(tile1_valid)}")
print(f"  First 10 scores: {[f'{s:.2f}' for _, s in tile1_valid[:10]]}")

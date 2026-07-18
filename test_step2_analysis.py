import numpy as np

# Load both score dumps
scores_0_511 = np.load('/tmp/batch600_scores_0_511.npy')
scores_512_1023 = np.load('/tmp/batch600_scores_512_1023.npy')
all_scores = np.concatenate([scores_0_511, scores_512_1023])

print(f"Total scores: {len(all_scores)} (index 0~1023)")
print(f"cuRealAcSeq = 601, so valid scores are index 0~600")

# Check which indices have +1e30 (invalid)
invalid_mask = all_scores > 1e29
print(f"\nInvalid scores (+1e30): {invalid_mask.sum()}")
print(f"  First invalid at index: {np.argmax(invalid_mask)}")
print(f"  Valid scores: {len(all_scores) - invalid_mask.sum()}")

# Check scores around index 500~610
print(f"\nScores around boundary (500~610):")
for i in range(500, 612):
    s = all_scores[i]
    tag = "INVALID" if s > 1e29 else ""
    if i % 10 == 0 or tag:
        print(f"  score[{i}] = {s:12.4f}  {tag}")

# Load sort and radix results
sort = np.load('/tmp/sort_result.npy')
radix = np.load('/tmp/radixtopk_result.npy')
s_set = set(sort[600,0][sort[600,0]!=-1].tolist())
r_set = set(radix[600,0][radix[600,0]!=-1].tolist())
print(f"\nSort batch 600: {len(s_set)} valid, first 5: {sorted(s_set)[:5]}")
print(f"Radix batch 600: {len(r_set)} valid, first 5: {sorted(r_set)[:5]}")
print(f"Overlap: {len(s_set & r_set)}")

# Check sort_only and radix_only scores
s_only = sorted(s_set - r_set)
r_only = sorted(r_set - s_set)
print(f"\nSort only ({len(s_only)}), first 10: {s_only[:10]}")
print(f"  Their scores: {[f'{all_scores[i]:.2f}' for i in s_only[:10] if i < 1024]}")
print(f"Radix only ({len(r_only)}), first 10: {r_only[:10]}")
print(f"  Their scores: {[f'{all_scores[i]:.2f}' for i in r_only[:10] if i < 1024]}")

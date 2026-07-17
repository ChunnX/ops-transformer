import numpy as np
sort = np.load('/tmp/sort_result.npy')
sort_flat = sort.flatten()
# batch 512
b = 512
batch = sort_flat[b*512:(b+1)*512]
valid = batch[batch != -1]
print(f"Sort batch {b}: {len(valid)} valid, min={valid.min()}, max={valid.max()}")
print(f"  first 20: {sorted(valid[:20].tolist())}")
# Check if indices are within 0-512 or 0-8192
print(f"  index range: [{valid.min()}, {valid.max()}]")

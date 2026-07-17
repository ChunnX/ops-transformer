import numpy as np
r = np.load('/tmp/sort_result.npy')
valid = r[r != -1]
print(f"Sort baseline: shape={r.shape}, valid={len(valid)}, min={valid.min()}, max={valid.max()}")

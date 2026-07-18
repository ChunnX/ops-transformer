import numpy as np
scores = np.load('/tmp/batch600_scores_0_511.npy')
print(f"Total: {len(scores)}")
print(f"Invalid (+1e30): {(scores > 1e29).sum()}")
print(f"First invalid at: {np.argmax(scores > 1e29)}")
print(f"score[88] = {scores[88]:.4f}")
print(f"score[507] = {scores[507]:.4f}")
print(f"score[508] = {scores[508]:.4f}")

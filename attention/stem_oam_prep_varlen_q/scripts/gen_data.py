# ============================================================================
# Test data generator for stem_oam_prep_varlen_q
# ============================================================================
#
# Small test: batch=1, H_q=1, q_len=128 (single stem block)
# Q: varlen BF16 tensor, qscale: all 1.0, kscale: 1.0
# ============================================================================

import numpy as np
import os
import math

from golden import stem_oam_prep_varlen_q_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- Test parameters ----
B = 128       # STEM_BLOCK_SIZE
S = 16        # STEM_STRIDE
D = 128       # DIM_QK
batch = 1
H_q = 1
q_len = 128   # single block
total_tokens = q_len  # batch=1

max_q_padded = math.ceil(q_len / B) * B   # 128
max_Qb = max_q_padded // B                 # 1

# ---- Generate input data ----
# Q: [total_tokens, H_q, D] BF16
np.random.seed(42)
q_fp32 = np.random.randn(total_tokens, H_q, D).astype(np.float32)
q_bf16 = q_fp32.astype(np.bfloat16)

# qscale: [batch, H_q, max_q_padded] FP32, all 1.0
qscale = np.ones((batch, H_q, max_q_padded), dtype=np.float32)

# kscale: scalar FP32, 1.0
kscale = np.array([1.0], dtype=np.float32)

# q_seq_lens: [batch] INT32
q_seq_lens = np.array([q_len], dtype=np.int32)

# cu_seqlens_q: [batch+1] INT32
cu_seqlens_q = np.array([0, q_len], dtype=np.int32)

# ---- Write input files ----
q_bf16.tofile("input/input_q.bin")
qscale.tofile("input/input_qscale.bin")
kscale.tofile("input/input_kscale.bin")
q_seq_lens.tofile("input/input_q_seq_lens.bin")
cu_seqlens_q.tofile("input/input_cu_seqlens_q.bin")

# ---- Compute and write golden ----
qflat_golden = stem_oam_prep_varlen_q_golden(
    q_bf16, qscale, kscale, q_seq_lens, cu_seqlens_q, B=B, S=S)

qflat_golden.tofile("output/golden.bin")

print(f"Generated test data: batch={batch}, H_q={H_q}, q_len={q_len}")
print(f"  input/input_q.bin:            shape={q_bf16.shape}, dtype={q_bf16.dtype}, size={q_bf16.nbytes}B")
print(f"  input/input_qscale.bin:       shape={qscale.shape}, dtype={qscale.dtype}, size={qscale.nbytes}B")
print(f"  input/input_kscale.bin:       shape={kscale.shape}, dtype={kscale.dtype}, size={kscale.nbytes}B")
print(f"  input/input_q_seq_lens.bin:   shape={q_seq_lens.shape}, dtype={q_seq_lens.dtype}")
print(f"  input/input_cu_seqlens_q.bin: shape={cu_seqlens_q.shape}, dtype={cu_seqlens_q.dtype}")
print(f"  output/golden.bin:            shape={qflat_golden.shape}, dtype={qflat_golden.dtype}, size={qflat_golden.nbytes}B")
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代一 穿刺2 (probe2_norm_multicore) — 测试数据 + golden 生成
# shape 必须与 op_host/ffn_wb_probe.asc 中的常量保持一致。
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- 固定 shape (与 ffn_wb_probe.asc 一致) ----
Y = 4096            # 4 sort 核 × 1024，触发多核 sort + 4 路 MrgSort 归并
H = 64              # hidden dim
EXPERT_NUM = 128

rng = np.random.default_rng(42)

# 全 token 有效 (expert_id ∈ [0, EXPERT_NUM))，含大量重复键 -> 检验多核归并稳定性
expert_ids = rng.integers(0, EXPERT_NUM, size=Y).astype(np.int32)
# token hidden states，逐行互异 -> 检验 gather 行搬运正确性
token_data = rng.standard_normal((Y, H)).astype(np.float16)
# id 表：取互异值，便于验证索引 gather 正确
idx = np.arange(Y, dtype=np.int32)
session_src = (idx * 3 + 1).astype(np.int32)
micro_batch_src = (idx * 7 + 2).astype(np.int32)
token_src = (idx * 11 + 5).astype(np.int32)

expert_ids.tofile("input/expert_ids.bin")
token_data.tofile("input/token_data.bin")
session_src.tofile("input/session_src.bin")
micro_batch_src.tofile("input/micro_batch_src.bin")
token_src.tofile("input/token_src.bin")

golden = compute_golden(expert_ids, token_data, session_src, micro_batch_src, token_src, EXPERT_NUM)
for name, arr in golden.items():
    arr.tofile(f"output/golden_{name}.bin")

print(f"[gen_data] Y={Y} H={H} EXPERT_NUM={EXPERT_NUM}")
print(f"  expert_ids: {expert_ids.shape} {expert_ids.dtype}, unique experts={len(np.unique(expert_ids))}")
print(f"  token_data: {token_data.shape} {token_data.dtype}")
print(f"  golden outputs: {list(golden.keys())}")
print(f"  actual_token_num(golden)={golden['actual_token_num'][0]}  group_count sum={golden['group_count'].sum()}")

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代二 穿刺2 (probe_iter2_boundary_maxshape) — 测试数据 + golden 生成
#
# shape 由 ub_params.txt (host --query-ub 写出) 决定，保证 host/kernel/golden 完全一致，
# 且 Y/SLM 由 NPU 运行时真实 UB 派生（禁写死 arch 常量，对齐 MEMORY「UB size no hardcode」）。
#   Y   = SORT_CORE_NUM * SLM,  SLM = ubKernel/32
# 语义: 全 token 有效 (k=N)；边界上界 shape 语义 (A=1024 / topK+1=64) 见 RESULT.md。
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import os

from golden import compute_golden, ID_PACK_WIDTH, ID_SESSION_COL, ID_MICRO_COL, ID_TOKEN_COL

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def read_ub_params(path="ub_params.txt"):
    params = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                params[parts[0]] = int(parts[1])
    return params


p = read_ub_params()
Y = p["Y"]
H = p["H"]
EXPERT_NUM = p["expertNum"]
SLM = p["slm"]
SORT_CORE_NUM = p["sortCoreNum"]
assert p["idPackWidth"] == ID_PACK_WIDTH, "idPackWidth mismatch host vs golden"

rng = np.random.default_rng(2026)

# 全 token 有效 (expert_id ∈ [0, EXPERT_NUM))，含大量重复键 -> 检验多核归并稳定性
expert_ids = rng.integers(0, EXPERT_NUM, size=Y).astype(np.int32)
# token hidden states，逐行互异 -> 检验 gather 行搬运正确性
token_data = rng.standard_normal((Y, H)).astype(np.float16)
# 打包 id 表 [Y, 8] int32：col0=session col1=micro col2=token，其余 pad=0；取互异值便于验证 gather
idx = np.arange(Y, dtype=np.int32)
id_pack = np.zeros((Y, ID_PACK_WIDTH), dtype=np.int32)
id_pack[:, ID_SESSION_COL] = idx * 3 + 1
id_pack[:, ID_MICRO_COL] = idx * 7 + 2
id_pack[:, ID_TOKEN_COL] = idx * 11 + 5

expert_ids.tofile("input/expert_ids.bin")
token_data.tofile("input/token_data.bin")
id_pack.tofile("input/id_pack_src.bin")

golden = compute_golden(expert_ids, token_data, id_pack, EXPERT_NUM)
for name, arr in golden.items():
    arr.tofile(f"output/golden_{name}.bin")

print(f"[gen_data] Y={Y} H={H} EXPERT_NUM={EXPERT_NUM} SLM={SLM} SORT_CORE_NUM={SORT_CORE_NUM}")
print(f"  边界语义: A*BS*(topK+1)=Y={Y} (上界 A=1024/topK+1=64 语义); sort 每核 tile=SLM={SLM} 打到物理 UB 上界")
print(f"  expert_ids: {expert_ids.shape} {expert_ids.dtype}, unique experts={len(np.unique(expert_ids))}")
print(f"  token_data: {token_data.shape} {token_data.dtype}  ({token_data.nbytes/1024/1024:.1f} MB)")
print(f"  id_pack: {id_pack.shape} {id_pack.dtype}")
print(f"  golden outputs: {list(golden.keys())}")
print(f"  actual_token_num(golden)={golden['actual_token_num'][0]}  group_count sum={golden['group_count'].sum()}")

#!/usr/bin/env python3
# ============================================================================
# 穿刺1 (RECV fp16) 数据 + CPU golden 生成
#  - 生成 device 输入: token_info(int32 [A,M,F]) / token_data(fp16 [A,M,BS,K,H])
#  - 生成 8 路 CPU golden(与 kernel RECV 单核 sort 语义严格一致, 二进制比对)
# 语义(照搬 arch22 ffn_wb RECV 路径, 见 op_kernel/ffn_wb_*.h):
#   1) 取 curMicroBatchID(=0) 下每 session 的 BS*K 个 expert_id; 无效 token(expert_id>=1e6)
#      被 mask 掉; 每 session 的 BS*K 段按 int32(=8 elem) 对齐补 pad(INT_MAX, 亦 mask)。
#   2) row_idx = a*(BS*K+pad) + (bs*K+k); 对 valid token 按 (expert_id 升序, row_idx 升序)
#      稳定排序 -> gather 序列。
#   3) 按 gather 序列搬 token_data(RECV: session=a, micro_batch=curMB), 生成 8 路输出。
# ============================================================================
import os
import numpy as np

# ---- shape（与 op_host/ffn_wb_recv_probe.asc 一致）----
A, M, BS, K, H = 4, 2, 2, 2, 8
EXPERT_NUM = 8
CUR_MB = 0
F = BS * K + 2                 # token_info 每 (a,m): [flag, layer_id, e0..e{BS*K-1}]
BSK = BS * K                   # 4
EXPERT_START = 1000000         # expert_id >= 此值视为无效(被 mask)
MASK = 2000000                 # 生产者置的无效 token 大值
INT32_MAX = 2147483647

def align_elem(n, bytes_per):  # 与 kernel Align() 一致
    return (n * bytes_per + 32 - 1) // 32 * 32 // bytes_per

PAD = align_elem(BSK, 4) - BSK             # 4
BSK_PAD = BSK + PAD                         # 8 (= bskProduct)
Y = A * BS * K                              # 16

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- 每 session micro-batch0 的 expert 分配（~50% mask + 跨 session 同专家做 batching）----
# slot = bs*K + k, slot in [0, BSK)
experts_mb0 = np.array([
    [2,    5,    MASK, MASK],   # a0: e2(s0) e5(s1)
    [2,    MASK, 7,    MASK],   # a1: e2(s0) e7(s2)
    [MASK, 5,    MASK, 0   ],   # a2: e5(s1) e0(s3)
    [MASK, MASK, MASK, 2   ],   # a3: e2(s3)
], dtype=np.int32)

# ---- token_info [A, M, F] ----
token_info = np.zeros((A, M, F), dtype=np.int32)
for a in range(A):
    # micro batch 0: flag=1, 有效路由
    token_info[a, 0, 0] = 1          # flag (scan 等待 sum(flags)==A)
    token_info[a, 0, 1] = 0          # layer_id
    token_info[a, 0, 2:2 + BSK] = experts_mb0[a]
    # micro batch 1: 未被扫描(curMB=0), 填 mask
    token_info[a, 1, 0] = 0
    token_info[a, 1, 2:2 + BSK] = MASK
token_info.tofile("input/token_info.bin")

# ---- token_data [A, M, BS, K, H] fp16 = arange(可精确表示)----
token_data = np.arange(A * M * BS * K * H, dtype=np.float16).reshape(A, M, BS, K, H)
token_data.tofile("input/token_data.bin")

# ---- CPU golden ----
# 收集 valid token, 保持 (a, bs, k) 迭代序(= row_idx 升序)
valid = []  # (expert_id, row_idx, a, bs, k)
for a in range(A):
    for bs in range(BS):
        for k in range(K):
            slot = bs * K + k
            e = int(experts_mb0[a, slot])
            if e < EXPERT_START:
                row_idx = a * BSK_PAD + slot
                valid.append((e, row_idx, a, bs, k))

exp_arr = np.array([v[0] for v in valid], dtype=np.int64)
order = np.argsort(exp_arr, kind="stable")   # 稳定: 同专家保持 row_idx 升序
V = len(valid)

y = np.zeros((Y, H), dtype=np.float16)
session_ids = np.zeros(Y, dtype=np.int32)
micro_batch_ids = np.zeros(Y, dtype=np.int32)
token_ids = np.zeros(Y, dtype=np.int32)
expert_offsets = np.zeros(Y, dtype=np.int32)
dynamic_scale = np.zeros(Y, dtype=np.float32)   # fp16: kernel 不写 -> 0

sorted_experts = []
for i, oi in enumerate(order):
    e, row_idx, a, bs, k = valid[oi]
    session_ids[i] = a
    micro_batch_ids[i] = CUR_MB
    token_ids[i] = bs
    expert_offsets[i] = k
    y[i, :] = token_data[a, CUR_MB, bs, k, :]
    sorted_experts.append(e)

# group_list [EXPERT_NUM, 2]: 升序去重 [expert_id, count], 末尾 [0,0], 其余 0
group_list = np.zeros((EXPERT_NUM, 2), dtype=np.int64)
row = 0
i = 0
while i < V:
    e = sorted_experts[i]
    j = i
    while j < V and sorted_experts[j] == e:
        j += 1
    group_list[row, 0] = e
    group_list[row, 1] = j - i
    row += 1
    i = j
# 末尾补 [0,0] terminator（未满 EXPERT_NUM 时；kernel 行为）
if row < EXPERT_NUM:
    group_list[row, 0] = 0
    group_list[row, 1] = 0

actual_token_num = np.array([V], dtype=np.int64)

# ---- 落盘 golden ----
y.tofile("output/y_golden.bin")
group_list.tofile("output/group_list_golden.bin")
session_ids.tofile("output/session_ids_golden.bin")
micro_batch_ids.tofile("output/micro_batch_ids_golden.bin")
token_ids.tofile("output/token_ids_golden.bin")
expert_offsets.tofile("output/expert_offsets_golden.bin")
dynamic_scale.tofile("output/dynamic_scale_golden.bin")
actual_token_num.tofile("output/actual_token_num_golden.bin")

print(f"[gen_data] A={A} M={M} BS={BS} K={K} H={H} expertNum={EXPERT_NUM} "
      f"BSK_PAD={BSK_PAD} Y={Y} validTokens={V}")
print(f"[gen_data] sorted_experts={sorted_experts}")
print(f"[gen_data] session_ids={session_ids[:V].tolist()}")
print(f"[gen_data] token_ids={token_ids[:V].tolist()} expert_offsets={expert_offsets[:V].tolist()}")
print(f"[gen_data] group_list=\n{group_list}")

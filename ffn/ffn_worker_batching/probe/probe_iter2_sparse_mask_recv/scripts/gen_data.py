#!/usr/bin/env python3
# ============================================================================
# 迭代二 穿刺4 (RECV fp16 稀疏 ~50% mask) 数据 + CPU golden 生成
#  验证目标(spec.extreme_inputs normal_sparse / PLAN 迭代二穿刺4):
#    - 部分 token 被 mask, 无效 token 初始化为极大值稳定排到序列末尾;
#    - 有效 token 按 expert_id 升序聚合, 同专家保持 row_idx 升序 (稳定排序);
#    - actual_token_num < Y (稀疏), 8 路输出与 kernel RECV 单核 sort 二进制一致。
#
#  相较迭代一穿刺1(Y=16,7 valid), 本穿刺加压排序稳定性:
#    A=8 (8 session, 跨 session 同专家), 精确 50% mask (16 valid / 32),
#    expert 直方图含重键 e5×6 / e2×4 (分散在非单调 row_idx 上), 逼近单核
#    scan-sort GatherMask 单 repeat 上限 totalLengthWithPad=64。
#
# 语义(照搬 arch22 ffn_wb RECV 路径, 见 op_kernel/ffn_wb_*.h):
#   1) 取 curMicroBatchID(=0) 下每 session 的 BS*K 个 expert_id; 无效 token
#      (expert_id>=1e6) 被 mask; 每 session 的 BS*K 段按 int32(=8 elem) 对齐
#      补 pad(INT_MAX, 亦 mask)。
#   2) row_idx = a*(BS*K+pad) + (bs*K+k); 对 valid token 按 (expert_id 升序,
#      row_idx 升序) 稳定排序 -> gather 序列。
#   3) 按 gather 序列搬 token_data(RECV: session=a, micro_batch=curMB), 生成 8 路输出。
# ============================================================================
import os
import numpy as np

# ---- shape（与 op_host/ffn_wb_sparse_recv_probe.asc 一致）----
A, M, BS, K, H = 8, 2, 2, 2, 8
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
Y = A * BS * K                              # 32

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- 每 session micro-batch0 的 expert 分配（精确 50% mask + 跨 session 同专家 batching）----
# slot = bs*K + k, slot in [0, BSK); MASK 表示无效 token(被 mask, 稳定排到末尾)。
# 有效专家直方图: e0×3, e2×4, e5×6, e7×3 = 16 (Y=32 的 50%)。
# e5/e2 的重复分散在多个 session 的不同 slot 上, 逼稳定排序按 row_idx 升序聚合。
experts_mb0 = np.array([
    [0,    MASK, 5,    MASK],   # a0: e0(s0k0) e5(s1k0)
    [MASK, 2,    MASK, 5   ],   # a1: e2(s0k1) e5(s1k1)
    [0,    MASK, MASK, 7   ],   # a2: e0(s0k0) e7(s1k1)
    [MASK, 2,    2,    MASK],   # a3: e2(s0k1) e2(s1k0)
    [5,    MASK, MASK, 5   ],   # a4: e5(s0k0) e5(s1k1)
    [MASK, 0,    7,    MASK],   # a5: e0(s0k1) e7(s1k0)
    [7,    MASK, MASK, 2   ],   # a6: e7(s0k0) e2(s1k1)
    [MASK, 5,    5,    MASK],   # a7: e5(s0k1) e5(s1k0)
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
      f"BSK_PAD={BSK_PAD} totalLengthWithPad={A*BSK_PAD} Y={Y} validTokens={V} "
      f"(sparsity={100*(Y-V)//Y}% masked)")
print(f"[gen_data] sorted_experts={sorted_experts}")
print(f"[gen_data] session_ids={session_ids[:V].tolist()}")
print(f"[gen_data] token_ids={token_ids[:V].tolist()} expert_offsets={expert_offsets[:V].tolist()}")
print(f"[gen_data] group_list=\n{group_list}")

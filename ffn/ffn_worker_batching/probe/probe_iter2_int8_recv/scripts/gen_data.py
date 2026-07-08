#!/usr/bin/env python3
# ============================================================================
# 迭代二 穿刺1 (probe_iter2_int8_recv) 数据 + CPU golden 生成
#   验证目标：P2 int8 + 32B FP32 scale 连续 gather 在 RECV(TilingKey 101) 路径。
#
#   组合来源：
#     - RECV 路径语义     ← probe1_recv_fp16（token_info 扫描 + 单核稳定 sort + pad + gather）
#     - int8 + 32B scale  ← probe3_norm_int8（HS = H(int8) + 尾部 32B FP32 scale 块，dynamic_scale 有效）
#
#   RECV 语义（照搬 arch22 ffn_wb RECV 路径，见 op_kernel/ffn_wb_scan_*.h + ffn_wb_gather_out_all.h）：
#     1) 取 curMicroBatchID(=0) 下每 session 的 BS*K 个 expert_id；无效 token(expert_id>=1e6)
#        被 mask；每 session 的 BS*K 段按 int32(=8 elem) 对齐补 pad(INT_MAX，亦 mask)。
#     2) row_idx = a*(BS*K+pad) + (bs*K+k)；对 valid token 按 (expert_id 升序, row_idx 升序) 稳定排序。
#     3) 按 gather 序列搬 token_data(RECV: session=a, micro_batch=curMB)，
#        int8 hidden 原样字节搬运 + 尾部 4B FP32 scale → dynamic_scale。生成 8 路输出。
#   精度口径 = 二进制一致 (np.array_equal, rtol=atol=0)。
# ============================================================================
import os
import numpy as np

# ---- shape（与 op_host/ffn_wb_recv_probe.asc 保持一致）----
A, M, BS, K, H = 8, 2, 2, 3, 128   # session / micro_batch_num / micro_batch_size / topK+1 / hidden(int8)
EXPERT_NUM = 16
CUR_MB = 0
TOKEN_DTYPE = 2                    # P2 dynamic_quant int8
HS = H + 32                        # attn_to_ffn_token_size(bytes) = int8 hidden(H) + 尾部 32B FP32 scale 块
F = BS * K + 2                     # token_info 每 (a,m): [flag, layer_id, e0..e{BS*K-1}]
BSK = BS * K                       # 6
EXPERT_START = 1000000             # expert_id >= 此值视为无效(被 mask)
MASK = 2000000                     # 生产者置的无效 token 大值
INT32_MAX = 2147483647

def align_elem(n, bytes_per):      # 与 kernel Align() 一致
    return (n * bytes_per + 32 - 1) // 32 * 32 // bytes_per

PAD = align_elem(BSK, 4) - BSK     # 8 - 6 = 2
BSK_PAD = BSK + PAD                # 8 (= bskProduct)
Y = A * BS * K                     # 48

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- 每 session micro-batch0 的 expert 分配（~50% mask + 跨 session 同专家做 batching）----
# slot = bs*K + k, slot in [0, BSK)。含重复专家(测稳定排序)，含 MASK(测稀疏剔除)。
experts_mb0 = np.array([
    [3,    MASK, 7,    MASK, 3,    MASK],   # a0: e3(s0) e7(s2) e3(s4)
    [MASK, 3,    MASK, 7,    MASK, 0   ],   # a1: e3(s1) e7(s3) e0(s5)
    [7,    MASK, MASK, 3,    MASK, MASK],   # a2: e7(s0) e3(s3)
    [MASK, 0,    3,    MASK, 7,    MASK],   # a3: e0(s1) e3(s2) e7(s4)
    [3,    MASK, MASK, MASK, 3,    7   ],   # a4: e3(s0) e3(s4) e7(s5)
    [MASK, 7,    MASK, 0,    MASK, MASK],   # a5: e7(s1) e0(s3)
    [0,    MASK, 3,    MASK, MASK, 7   ],   # a6: e0(s0) e3(s2) e7(s5)
    [MASK, MASK, 7,    MASK, 3,    MASK],   # a7: e7(s2) e3(s4)
], dtype=np.int32)
assert experts_mb0.shape == (A, BSK)

# ---- token_info [A, M, F] int32 ----
token_info = np.zeros((A, M, F), dtype=np.int32)
for a in range(A):
    # micro batch 0: flag=1（scan 等待 sum(flags)==A），有效路由
    token_info[a, CUR_MB, 0] = 1          # flag
    token_info[a, CUR_MB, 1] = 0          # layer_id
    token_info[a, CUR_MB, 2:2 + BSK] = experts_mb0[a]
    # 其余 micro batch：未被扫描(curMB=0)，flag=0 + mask 占位
    for mb in range(M):
        if mb == CUR_MB:
            continue
        token_info[a, mb, 0] = 0
        token_info[a, mb, 2:2 + BSK] = MASK
token_info.tofile("input/token_info.bin")

# ---- token_data [A, M, BS, K, HS] int8：hidden(H int8) + 尾部 4B FP32 scale ----
def gen_hidden(sess, mb, bs, k):
    j = np.arange(H, dtype=np.int64)
    vals = (sess * 37 + mb * 29 + bs * 17 + k * 11 + j * 3) % 251 - 125
    return vals.astype(np.int8)

def gen_scale(sess, mb, bs, k):
    return np.float32(sess * 1000.0 + mb * 100.0 + bs * 10.0 + k + 0.5)

def token_base(sess, mb, bs, k):
    return sess * M * BS * K * HS + mb * BS * K * HS + bs * K * HS + k * HS

token_data = np.zeros(A * M * BS * K * HS, dtype=np.int8)
for sess in range(A):
    for mb in range(M):
        for bs in range(BS):
            for k in range(K):
                base = token_base(sess, mb, bs, k)
                token_data[base:base + H] = gen_hidden(sess, mb, bs, k)
                sc = gen_scale(sess, mb, bs, k)
                token_data[base + H: base + H + 4] = np.frombuffer(sc.tobytes(), dtype=np.int8)
token_data.tofile("input/token_data.bin")

# ---- CPU golden（RECV 单核 sort 语义）----
# 收集 valid token（保持 (a, bs, k) 迭代序 = row_idx 升序），row_idx 含 pad
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
order = np.argsort(exp_arr, kind="stable")   # 稳定：同专家保持 row_idx 升序
V = len(valid)

y = np.zeros((Y, H), dtype=np.int8)
session_ids = np.zeros(Y, dtype=np.int32)
micro_batch_ids = np.zeros(Y, dtype=np.int32)
token_ids = np.zeros(Y, dtype=np.int32)
expert_offsets = np.zeros(Y, dtype=np.int32)
dynamic_scale = np.zeros(Y, dtype=np.float32)   # P2 关键路径：有效值

sorted_experts = []
for i, oi in enumerate(order):
    e, row_idx, a, bs, k = valid[oi]
    session_ids[i] = a          # RECV: sessionIndices = aIndices
    micro_batch_ids[i] = CUR_MB # RECV: microbatchIndices = curMicroBatchID
    token_ids[i] = bs
    expert_offsets[i] = k
    base = token_base(a, CUR_MB, bs, k)
    y[i, :] = token_data[base:base + H]
    dynamic_scale[i] = np.frombuffer(token_data[base + H:base + H + 4].tobytes(), dtype=np.float32)[0]
    sorted_experts.append(e)

# group_list [EXPERT_NUM, 2]：升序去重 [expert_id, count]，末尾补 [0,0]，其余 0
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
if row < EXPERT_NUM:               # 末尾补 [0,0] terminator（未满 EXPERT_NUM 时）
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

print(f"[gen_data] A={A} M={M} BS={BS} K={K} H={H} HS={HS} expertNum={EXPERT_NUM} "
      f"tokenDtype={TOKEN_DTYPE} curMB={CUR_MB}")
print(f"[gen_data] BSK_PAD={BSK_PAD} Y={Y} validTokens={V}")
print(f"[gen_data] sorted_experts={sorted_experts}")
print(f"[gen_data] session_ids={session_ids[:V].tolist()}")
print(f"[gen_data] token_ids={token_ids[:V].tolist()} expert_offsets={expert_offsets[:V].tolist()}")
print(f"[gen_data] dynamic_scale(head)={dynamic_scale[:min(8, V)].tolist()}")
print(f"[gen_data] group_list(nonzero rows)=\n{group_list[:row + 1]}")

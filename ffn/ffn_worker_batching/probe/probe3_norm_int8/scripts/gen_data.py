#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe3_norm_int8 数据生成 + 独立 CPU golden。

场景：FfnWorkerBatching NORM(TilingKey 100) 单核 sort + P2 dynamic_quant int8。
非计算类算子：sort 生成 gather 索引 -> gather 原样字节搬运 (int8 hidden + 尾部 32B FP32 scale)
-> group_listing 扫描计数。精度口径 = 二进制一致 (np.array_equal, rtol=atol=0)。

golden 按 spec.yaml math_semantics 三步语义 + arch22 参考实现 (KernelSortMaskOneCore /
KernelFfnWBGatherOutAll / KernelFfnWBGroupListing::ComputeOneCore) 的确定性行为独立复算：
  - sort：有效 token (expert_id < 1e6) 按 (expert_id, 原始索引) 升序稳定排序 (spec: 专家ID+稳定序)。
  - gather：y[out_i]=token_data hidden；dynamic_scale[out_i]=尾部 FP32 scale；同步产出 4 路 id。
  - group_listing：单核扫描排序后 expert_id 生成紧凑 [expert_id, count]，末尾补 [0,0]，其余 [0,0]。
输出行 [validLen, Y) 未被 gather 写入，与 device 侧 memset 0 对齐 → golden 置 0。
"""
import os
import struct
import numpy as np

# ---------------- 测试用例（单核 sort 小 shape；P2 int8）----------------
A          = 8    # session_num
M          = 2    # micro_batch_num
BS         = 2    # micro_batch_size
K          = 4    # selected_expert_num = topK+1
H          = 128  # hidden 维度 (int8)，<= maxBlockSize -> 单 hBlock
HS         = H + 32  # attn_to_ffn_token_size：int8 hidden(H) + 尾部 32B FP32 scale 块
EXPERT_NUM = 16
TOKEN_DTYPE = 2   # P2 dynamic_quant int8
OUT_NUM    = A    # 有效 attention session 数（全有效）

EXPERT_MASK_VALUE = 2000000  # >= expertStart_(1e6) 视为无效，稳定排到末尾并被剔除

Y = A * BS * K
TOTAL = OUT_NUM * BS * K
BSK = BS * K

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "input")
OUT_DIR = os.path.join(HERE, "..", "output")
# run.sh 在 build/ 下执行，脚本以 build 为 cwd 调用；用相对 cwd 的 input/output
IN_DIR = "./input"
OUT_DIR = "./output"
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def token_base(sess, mb, bs, k):
    return ((sess * M + mb) * BS + bs) * K * HS + (bs * 0) + k * HS  # = sess*M*BS*K*HS + mb*BS*K*HS + bs*K*HS + k*HS
# 上式展开显式写一遍，避免歧义
def token_base2(sess, mb, bs, k):
    return sess * M * BS * K * HS + mb * BS * K * HS + bs * K * HS + k * HS


def gen_hidden(sess, mb, bs, k):
    j = np.arange(H, dtype=np.int64)
    vals = (sess * 37 + mb * 29 + bs * 17 + k * 11 + j * 3) % 251 - 125
    return vals.astype(np.int8)


def gen_scale(sess, mb, bs, k):
    return np.float32(sess * 1000.0 + mb * 100.0 + bs * 10.0 + k + 0.5)


# ---------------- 生成输入 ----------------
# session_ids_buf / micro_batch_ids_buf: 长度 A，session 恒等映射，microbatch = a % M
session_ids_buf = np.arange(A, dtype=np.int32)
micro_batch_ids_buf = (np.arange(A, dtype=np.int32) % M).astype(np.int32)

# token_data: [A, M, BS, K, HS] int8
token_data = np.zeros(A * M * BS * K * HS, dtype=np.int8)
for sess in range(A):
    for mb in range(M):
        for bs in range(BS):
            for k in range(K):
                base = token_base2(sess, mb, bs, k)
                token_data[base:base + H] = gen_hidden(sess, mb, bs, k)
                sc = gen_scale(sess, mb, bs, k)
                token_data[base + H: base + H + 4] = np.frombuffer(sc.tobytes(), dtype=np.int8)

# expert_ids: 长度 TOTAL，布局 [OUT_NUM, BS, K]，含约 12.5% mask
expert_ids = np.zeros(TOTAL, dtype=np.int32)
for g in range(TOTAL):
    if g % 8 == 7:
        expert_ids[g] = EXPERT_MASK_VALUE
    else:
        expert_ids[g] = (g * 3) % EXPERT_NUM

# 写输入 bin
token_data.tofile(os.path.join(IN_DIR, "token_data.bin"))
session_ids_buf.tofile(os.path.join(IN_DIR, "session_ids_buf.bin"))
micro_batch_ids_buf.tofile(os.path.join(IN_DIR, "micro_batch_ids_buf.bin"))
# expert_ids_buf 分配大小 Y*4（host 侧读 Y 个），前 TOTAL 有效，其余填 0
expert_ids_full = np.zeros(Y, dtype=np.int32)
expert_ids_full[:TOTAL] = expert_ids
expert_ids_full.tofile(os.path.join(IN_DIR, "expert_ids_buf.bin"))

with open(os.path.join(IN_DIR, "params.txt"), "w") as f:
    f.write(f"{A} {M} {BS} {K} {H} {HS} {EXPERT_NUM} {TOKEN_DTYPE} {OUT_NUM}\n")

# ---------------- 独立 CPU golden ----------------
# 1) sort：有效 token 稳定升序 (expert_id, 原始索引)
valid = [g for g in range(TOTAL) if expert_ids[g] < 1000000]
gather_idx = sorted(valid, key=lambda g: (int(expert_ids[g]), g))
valid_len = len(gather_idx)
sorted_expert = [int(expert_ids[g]) for g in gather_idx]

# 2) gather + 4 路 id + dynamic_scale
y = np.zeros((Y, H), dtype=np.int8)
session_out = np.zeros(Y, dtype=np.int32)
microbatch_out = np.zeros(Y, dtype=np.int32)
token_out = np.zeros(Y, dtype=np.int32)
expert_offset_out = np.zeros(Y, dtype=np.int32)
dyn_scale = np.zeros(Y, dtype=np.float32)

for out_i in range(valid_len):
    g = gather_idx[out_i]
    a = g // BSK
    rem = g % BSK
    bs = rem // K
    k = rem % K
    sess = int(session_ids_buf[a])
    mb = int(micro_batch_ids_buf[a])
    base = token_base2(sess, mb, bs, k)
    y[out_i, :] = token_data[base:base + H]
    dyn_scale[out_i] = np.frombuffer(token_data[base + H:base + H + 4].tobytes(), dtype=np.float32)[0]
    session_out[out_i] = sess
    microbatch_out[out_i] = mb
    token_out[out_i] = bs
    expert_offset_out[out_i] = k

# 3) group_listing（复刻 ComputeOneCore 单核紧凑算法）
group_list = np.zeros((EXPERT_NUM, 2), dtype=np.int64)
cur_off = 0
if valid_len > 0:
    last_expert = sorted_expert[0]
    token_count = 0
    for i in range(valid_len):
        cur = sorted_expert[i]
        token_count += 1
        if cur > last_expert and cur_off < EXPERT_NUM:
            group_list[cur_off, 0] = last_expert
            group_list[cur_off, 1] = token_count - 1
            cur_off += 1
            token_count = 1
            last_expert = cur
    # 末专家
    if cur_off < EXPERT_NUM:
        group_list[cur_off, 0] = last_expert
        group_list[cur_off, 1] = token_count
        cur_off += 1
    # 补 [0,0]
    if cur_off < EXPERT_NUM:
        group_list[cur_off, 0] = 0
        group_list[cur_off, 1] = 0
        cur_off += 1

actual_token_num = np.array([valid_len], dtype=np.int64)

# ---------------- 写 golden bin ----------------
y.tofile(os.path.join(OUT_DIR, "golden_y.bin"))
group_list.tofile(os.path.join(OUT_DIR, "golden_group_list.bin"))
session_out.tofile(os.path.join(OUT_DIR, "golden_session_ids.bin"))
microbatch_out.tofile(os.path.join(OUT_DIR, "golden_micro_batch_ids.bin"))
token_out.tofile(os.path.join(OUT_DIR, "golden_token_ids.bin"))
expert_offset_out.tofile(os.path.join(OUT_DIR, "golden_expert_offsets.bin"))
dyn_scale.tofile(os.path.join(OUT_DIR, "golden_dynamic_scale.bin"))
# actual_token_num device 分配 32 字节；golden 也补齐到 32 字节做 bitwise 对齐
atn_pad = np.zeros(32 // 8, dtype=np.int64)
atn_pad[0] = valid_len
atn_pad.tofile(os.path.join(OUT_DIR, "golden_actual_token_num.bin"))

print(f"[gen_data] A={A} M={M} BS={BS} K={K} H={H} HS={HS} expertNum={EXPERT_NUM} "
      f"tokenDtype={TOKEN_DTYPE} outNum={OUT_NUM}")
print(f"[gen_data] Y={Y} TOTAL={TOTAL} valid_len={valid_len}")
print(f"[gen_data] sorted_expert(head)={sorted_expert[:16]}")
print(f"[gen_data] group_list(nonzero rows)=\n{group_list[:cur_off]}")
print(f"[gen_data] dyn_scale(head)={dyn_scale[:8]}")

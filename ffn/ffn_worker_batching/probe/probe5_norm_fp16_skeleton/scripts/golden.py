"""
probe5_norm_fp16_skeleton —— CPU golden（numpy 精确重排，非计算类，二进制一致口径）
迭代一骨架目标：NORM(TilingKey=100) / P0 fp16 / 单核 sort。复用主线 arch35 头。

复现主线 NORM 路径 (TilingKey 100) 三步语义：
  1) sort: 对 expert_ids 稳定升序排序（被 mask=MASK_VAL 的无效 token 排到末尾并丢弃），
           生成 gather_idx（原始 token 位置），validGatherIdxLength = 有效 token 数。
  2) gather: y[out_i] = token_data[session_tbl[a], microbatch_tbl[a], bs, k]（逐 uint16 元素/字节搬运）；
             session_ids/micro_batch_ids/token_ids(=bs)/expert_offsets(=k) 同步产出。
  3) group_listing: 压缩列表 [expert, count] 升序 + 末尾 [0,0] 终止符，其余槽 0。

参数与 op_host/probe_ffn_wb.asc 中常量必须严格一致。
"""
import numpy as np

MASK_VAL = 1000000  # = kernel expertStart_，被 mask token 的 expert_id

# ---- 形状常量（与 op_host/probe_ffn_wb.asc 一致）----
A = 4
M = 2
BS = 2
K = 3
H = 16
EXPERT_NUM = 8
TOKEN_DTYPE = 0          # fp16 (P0 骨架目标: NORM/100/fp16)
TOKEN_DTYPE_SIZE = 2
OUT_NUM = A
TOTAL_LENGTH = OUT_NUM * BS * K   # 24
Y = A * BS * K                    # 24


def gen_inputs(seed=42):
    """生成 4 路输入（确定性）。token_data 用 uint16 表示 bf16 原始 bit（byte gather 语义）。"""
    rng = np.random.RandomState(seed)

    # expert_ids: 24 个；随机 6 个 mask，其余取自 {0,1,3,4,6}（5 个专家 < EXPERT_NUM，触发 [0,0] 终止符 + 制造并列测稳定排序）
    valid_experts = np.array([0, 1, 3, 4, 6], dtype=np.int32)
    expert_ids = rng.choice(valid_experts, size=TOTAL_LENGTH).astype(np.int32)
    mask_pos = rng.choice(TOTAL_LENGTH, size=6, replace=False)
    expert_ids[mask_pos] = MASK_VAL

    # token_data: [A,M,BS,K,H] fp16（uint16 raw）。gather 走 tokenDtypeSize_=sizeof(half)=2B 纯字节搬运，
    # 与 bf16 同构（gather_out_all.h:47 判定 tokenDtype!=2 即取 sizeof(half)）。用真实 fp16 值。
    shape = (A, M, BS, K, H)
    vals = rng.uniform(-10.0, 10.0, size=shape).astype(np.float16)
    token_data_u16 = vals.view(np.uint16).copy()

    # session_tbl / microbatch_tbl（in-range：session∈[0,A), microbatch∈[0,M)）
    session_tbl = np.array([3, 1, 0, 2], dtype=np.int32)      # [A]
    microbatch_tbl = np.array([1, 0, 1, 0], dtype=np.int32)   # [A]

    return expert_ids, token_data_u16, session_tbl, microbatch_tbl


def compute_golden(expert_ids, token_data_u16, session_tbl, microbatch_tbl):
    bsk = BS * K

    # ---- step1 sort：稳定升序，丢弃 mask ----
    valid_idx = np.where(expert_ids < MASK_VAL)[0]
    order = np.argsort(expert_ids[valid_idx], kind="stable")  # 稳定：并列按原始索引升序
    gather_idx = valid_idx[order].astype(np.int64)
    valid_len = int(gather_idx.shape[0])
    sorted_experts = expert_ids[gather_idx]

    # ---- step2 gather（尾部 memset 0）----
    y = np.zeros((Y, H), dtype=np.uint16)
    session_ids = np.zeros((Y,), dtype=np.int32)
    micro_batch_ids = np.zeros((Y,), dtype=np.int32)
    token_ids = np.zeros((Y,), dtype=np.int32)
    expert_offsets = np.zeros((Y,), dtype=np.int32)

    for out_i in range(valid_len):
        p = int(gather_idx[out_i])
        a = p // bsk
        rem = p - a * bsk
        bs = rem // K
        k = rem - bs * K
        s = int(session_tbl[a])
        mb = int(microbatch_tbl[a])
        y[out_i, :] = token_data_u16[s, mb, bs, k, :]
        session_ids[out_i] = s
        micro_batch_ids[out_i] = mb
        token_ids[out_i] = bs
        expert_offsets[out_i] = k

    # ---- step3 group_listing：压缩 [expert,count] 升序 + [0,0]，其余 0 ----
    group_list = np.zeros((EXPERT_NUM, 2), dtype=np.int64)
    uniq, counts = np.unique(sorted_experts, return_counts=True)  # 升序
    off = 0
    for e, c in zip(uniq, counts):
        group_list[off, 0] = int(e)
        group_list[off, 1] = int(c)
        off += 1
    # off < EXPERT_NUM 时补 [0,0]（memset 已 0，此处显式无害）
    if off < EXPERT_NUM:
        group_list[off, 0] = 0
        group_list[off, 1] = 0

    dynamic_scale = np.zeros((Y,), dtype=np.float32)  # fp16 不用（仅 int8 动态量化产出）
    actual_token_num = np.array([valid_len], dtype=np.int64)

    return {
        "y": y,
        "group_list": group_list,
        "session_ids": session_ids,
        "micro_batch_ids": micro_batch_ids,
        "token_ids": token_ids,
        "expert_offsets": expert_offsets,
        "dynamic_scale": dynamic_scale,
        "actual_token_num": actual_token_num,
    }

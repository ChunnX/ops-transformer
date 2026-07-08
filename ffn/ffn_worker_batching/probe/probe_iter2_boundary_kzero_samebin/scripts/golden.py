"""
probe_iter2_boundary_kzero_samebin —— CPU golden（numpy 精确重排，非计算类，二进制一致口径）
迭代二穿刺3：验证迭代三难点边界 —— 全 mask(k=0) + same_bin(全同专家, expert_num=8192)。
复用主线 arch35 NORM(TilingKey=100)/P0 fp16/单核 sort 头，仅参数化 expert_ids/expert_num。

三步语义（与主线 NORM 一致，spec.math_semantics）：
  1) sort: 对 expert_ids 稳定升序排序（被 mask=MASK_VAL 的无效 token 排末尾并丢弃），
           生成 gather_idx，validGatherIdxLength=有效 token 数。
  2) gather: y[out_i]=token_data[session_tbl[a],microbatch_tbl[a],bs,k]（逐 uint16 字节搬运），
             session_ids/micro_batch_ids/token_ids(=bs)/expert_offsets(=k) 同步产出。
  3) group_listing: 压缩 [expert,count] 升序 + 末尾 [0,0] 终止符，其余槽 0。

golden 建模 spec.boundary_conditions 契约：
  - kzero(全 mask,k=0): actual_token_num=0，group_list 全 [0,0]，y/ids/scale 全 0。
  - samebin(全同专家): actual_token_num=Y，group_list[0]=[e,Y]、[1]=[0,0]、其余 0。

形状/常量必须与 op_host/probe_ffn_wb.asc 严格一致。
"""
import numpy as np

MASK_VAL = 1000000  # = kernel expertStart_，被 mask token 的 expert_id

# ---- 形状常量（与 op_host/probe_ffn_wb.asc 一致）----
A = 4
M = 2
BS = 2
K = 3
H = 16
TOKEN_DTYPE = 0          # fp16 (P0)
TOKEN_DTYPE_SIZE = 2
OUT_NUM = A
TOTAL_LENGTH = OUT_NUM * BS * K   # 24
Y = A * BS * K                    # 24

# same_bin 场景：全部有效 token 路由到同一专家 SAME_EXPERT（< expert_num）
SAME_EXPERT = 5

SESSION_TBL = np.array([3, 1, 0, 2], dtype=np.int32)     # [A], in-range [0,A)
MICROBATCH_TBL = np.array([1, 0, 1, 0], dtype=np.int32)  # [A], in-range [0,M)


def expert_num_of(scenario):
    return 8 if scenario == "kzero" else 8192


def gen_inputs(scenario, seed=42):
    """生成 4 路输入（确定性）。token_data 用 uint16 表示 fp16 原始 bit（byte gather 语义）。"""
    rng = np.random.RandomState(seed)

    if scenario == "kzero":
        # 全 mask：所有 token expert_id 置极大值 -> 排序后全丢弃 -> 有效 token=0
        expert_ids = np.full(TOTAL_LENGTH, MASK_VAL, dtype=np.int32)
    elif scenario == "samebin":
        # 全同专家：所有 token expert_id = SAME_EXPERT（无 mask，全有效，全落同一 bin）
        expert_ids = np.full(TOTAL_LENGTH, SAME_EXPERT, dtype=np.int32)
    else:
        raise ValueError(f"unknown scenario {scenario}")

    shape = (A, M, BS, K, H)
    vals = rng.uniform(-10.0, 10.0, size=shape).astype(np.float16)
    token_data_u16 = vals.view(np.uint16).copy()

    return expert_ids, token_data_u16, SESSION_TBL.copy(), MICROBATCH_TBL.copy()


def compute_golden(scenario, expert_ids, token_data_u16, session_tbl, microbatch_tbl):
    expert_num = expert_num_of(scenario)
    bsk = BS * K

    # ---- step1 sort：稳定升序，丢弃 mask ----
    valid_idx = np.where(expert_ids < MASK_VAL)[0]
    order = np.argsort(expert_ids[valid_idx], kind="stable")  # 稳定：并列按原始索引升序
    gather_idx = valid_idx[order].astype(np.int64)
    valid_len = int(gather_idx.shape[0])
    sorted_experts = expert_ids[gather_idx]

    # ---- step2 gather（尾部/无效槽 memset 0）----
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
    group_list = np.zeros((expert_num, 2), dtype=np.int64)
    if valid_len > 0:
        uniq, counts = np.unique(sorted_experts, return_counts=True)  # 升序
        off = 0
        for e, c in zip(uniq, counts):
            group_list[off, 0] = int(e)
            group_list[off, 1] = int(c)
            off += 1
        if off < expert_num:  # 末尾补 [0,0] 终止符（memset 已 0）
            group_list[off, 0] = 0
            group_list[off, 1] = 0
    # valid_len==0(kzero)：group_list 全 [0,0]（spec 契约期望）

    dynamic_scale = np.zeros((Y,), dtype=np.float32)  # fp16 不用
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

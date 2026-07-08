"""
FfnWorkerBatching —— TTK Kernel 模式 golden 插件（numpy，二进制精确口径）

算子类别：非计算类（Sort 排序 + IndexGather 索引 gather + Scan 分组计数），无浮点算术。
精度口径：二进制精确（np.array_equal，rtol=atol=0），取自 spec.yaml numerical_tolerance.per_dtype。

本 golden 复用已在 NPU 上验证过 byte-exact 的现成参考语义
（probe/probe4_norm_bf16/scripts/golden.py + tests/st/test_aclnn_ffn_worker_batching.cpp），
覆盖 NORM 路径（need_schedule=0，TilingKey 100）三步语义：
  1) sort   : 对 expert_id 稳定升序排序，expert_id >= MASK_VAL(1000000) 视无效并剔除，
              生成 gather_idx（原始 token 位置），actual_token_num = 有效 token 数。
  2) gather : y[out_i] = token_data[session_tbl[aI], microbatch_tbl[aI], bsI, kI]（逐字节原样搬运）；
              session_ids/micro_batch_ids/token_ids(=bsI)/expert_offsets(=kI) 同步查表产出；
              token_dtype=2 时每 token 尾部 32B 首 4B 为 FP32 dynamic_scale。
  3) group_listing : 升序打包 [expert_id, count] + 末尾 [0,0]，其余槽 0。

=====================================================================================
【接口说明 —— 重要】
本 golden 的输入签名对应 TTK 侧包装算子 `ffn_wb_norm_ttk`（4 输入显式暴露），
而非生产算子 `FfnWorkerBatching`（唯一输入 schedule_context 内嵌 4 个 GM 绝对地址，
TTK Kernel 模式无法注入设备地址，详见 docs/TTK_GUIDE.md 的“阻塞点”一节）。
包装算子输入顺序（与 probe4 kernel 入口一致）：
    expert_ids[int32, totalLength] , token_data[int8 raw bytes, TokenSlots*HS] ,
    session_tbl[int32, A] , microbatch_tbl[int32, A]
属性：expert_num, max_out_shape=[A,BS,K,H], token_dtype(0/1/2), need_schedule(=0), M。
=====================================================================================
"""
import numpy as np

MASK_VAL = 1000000          # kernel expertStart_，expert_id >= 视为无效 token
BLOCK_BYTES = 32            # int8 动态量化每 token 尾部 scale 块字节数

__golden__ = {
    "kernel": {
        "ffn_wb_norm_ttk": "ffn_wb_norm_ttk_golden",
    }
}


def _elem_size(token_dtype):
    # token_dtype: 0=fp16 / 1=bf16 (2B) ; 2=int8 (1B)
    return 1 if int(token_dtype) == 2 else 2


def _hs_bytes(token_dtype, H):
    # 每 token 字节步长：int8 尾部额外 BLOCK_BYTES 承载 FP32 scale
    es = _elem_size(token_dtype)
    return H + BLOCK_BYTES if int(token_dtype) == 2 else H * es


def ffn_wb_norm_ttk_golden(expert_ids, token_data, session_tbl, microbatch_tbl,
                           *, expert_num, max_out_shape, token_dtype=0, need_schedule=0,
                           M=2, **kwargs):
    """
    返回 8 路输出（顺序与算子 def 输出一致）：
        [y, group_list, session_ids, micro_batch_ids, token_ids,
         expert_offsets, dynamic_scale, actual_token_num]

    参数（**kwargs 接收 TTK 元信息：input_dtypes/output_dtypes/shapes/soc_version 等）：
      expert_ids     : int32 [totalLength]      每 token 的 expert_id
      token_data     : 原始字节缓冲（int8/uint8 视图），逻辑 [A,M,BS,K,HS]
      session_tbl    : int32 [A]                aI -> 真实 session id
      microbatch_tbl : int32 [A]                aI -> micro batch id
      max_out_shape  : [A, BS, K, H]
    """
    A, BS, K, H = (int(v) for v in max_out_shape)
    expert_num = int(expert_num)
    token_dtype = int(token_dtype)
    M = int(M)

    es = _elem_size(token_dtype)
    HS = _hs_bytes(token_dtype, H)
    payload = H * es
    bsk = BS * K
    Y = A * BS * K

    expert_ids = np.asarray(expert_ids).reshape(-1).astype(np.int64)
    session_tbl = np.asarray(session_tbl).reshape(-1).astype(np.int64)
    microbatch_tbl = np.asarray(microbatch_tbl).reshape(-1).astype(np.int64)
    # token_data 按原始字节处理（byte gather 语义，覆盖任意 bit-pattern）
    td_bytes = np.frombuffer(np.asarray(token_data).tobytes(), dtype=np.uint8)

    # ---- step1 sort：稳定升序，剔除无效 token ----
    valid_pos = np.where(expert_ids < MASK_VAL)[0]
    order = np.argsort(expert_ids[valid_pos], kind="stable")   # 稳定：并列按原始索引升序
    gather_idx = valid_pos[order]
    valid_len = int(gather_idx.shape[0])
    sorted_experts = expert_ids[gather_idx]

    # ---- 输出缓冲（全 Y 尺寸，尾部保持 0；有效区 [0,valid_len)）----
    y = np.zeros((Y, payload), dtype=np.uint8)
    session_ids = np.zeros((Y,), dtype=np.int32)
    micro_batch_ids = np.zeros((Y,), dtype=np.int32)
    token_ids = np.zeros((Y,), dtype=np.int32)
    expert_offsets = np.zeros((Y,), dtype=np.int32)
    dynamic_scale = np.zeros((Y,), dtype=np.float32)

    # ---- step2 gather（原样字节搬运 + 查表 id/scale）----
    for out_i in range(valid_len):
        p = int(gather_idx[out_i])
        aI = p // bsk
        rem = p - aI * bsk
        bsI = rem // K
        kI = rem - bsI * K
        sess = int(session_tbl[aI])
        mb = int(microbatch_tbl[aI])

        session_ids[out_i] = sess
        micro_batch_ids[out_i] = mb
        token_ids[out_i] = bsI
        expert_offsets[out_i] = kI

        # token_data 逻辑源 slot：[A,M,BS,K] 行优先，stride 单位 = HS 字节
        src_slot = (((sess * M + mb) * BS + bsI) * K + kI)
        src_byte = src_slot * HS
        y[out_i, :] = td_bytes[src_byte: src_byte + payload]
        if token_dtype == 2:
            sc = td_bytes[src_byte + H: src_byte + H + 4].tobytes()
            dynamic_scale[out_i] = np.frombuffer(sc, dtype=np.float32)[0]

    # ---- step3 group_listing：升序打包 [expert,count] + 末尾 [0,0] ----
    group_list = np.zeros((expert_num, 2), dtype=np.int64)
    row = 0
    i = 0
    while i < valid_len and row < expert_num:
        eid = int(sorted_experts[i])
        cnt = 0
        while i < valid_len and int(sorted_experts[i]) == eid:
            cnt += 1
            i += 1
        group_list[row, 0] = eid
        group_list[row, 1] = cnt
        row += 1
    # 未满 -> 末尾补 [0,0]（memset 已 0，显式无害）
    if row < expert_num:
        group_list[row, 0] = 0
        group_list[row, 1] = 0

    actual_token_num = np.array([valid_len], dtype=np.int64)

    # y 以算子输出 dtype 视图返回（fp16/bf16/int8），bit-pattern 不变
    out_dtypes = kwargs.get("output_dtypes")
    y_dtype = str(out_dtypes[0]).lower() if out_dtypes else ("int8" if token_dtype == 2 else "float16")
    y_np_dtype = {"float16": np.float16, "bfloat16": np.dtype("bfloat16") if _has_bf16() else np.uint16,
                  "bf16": np.dtype("bfloat16") if _has_bf16() else np.uint16,
                  "int8": np.int8}.get(y_dtype, np.uint16)
    y_view = np.frombuffer(y.tobytes(), dtype=y_np_dtype).reshape(Y, H)

    return [y_view, group_list, session_ids, micro_batch_ids, token_ids,
            expert_offsets, dynamic_scale, actual_token_num]


def _has_bf16():
    try:
        import ml_dtypes  # noqa: F401
        np.dtype("bfloat16")
        return True
    except Exception:
        return False

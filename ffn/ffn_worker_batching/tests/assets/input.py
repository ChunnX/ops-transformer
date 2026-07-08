"""
FfnWorkerBatching —— TTK Kernel 模式 input 插件（numpy，确定性构造）

为 TTK 侧包装算子 `ffn_wb_norm_ttk`（4 输入）生成与参考模型强绑定的确定性输入：
    expert_ids     : int32 [totalLength]   round-robin 分布到 [0,expert_num)，按 mask_stride 置无效
    token_data     : 原始字节缓冲          逻辑 [A,M,BS,K,HS]，字节值 = ByteVal(全局字节偏移)
    session_tbl    : int32 [A]             identity（合法映射到 token_data 的 A 维）
    microbatch_tbl : int32 [A]             a % M

确定性哈希 ByteVal 与 tests/st/test_aclnn_ffn_worker_batching.cpp 的 CPU 参考模型一致，
保证 golden（同一 numpy 语义）与 NPU 输出可做 byte-exact 对拍。

注：TTK 生成 4 个占位输入 ndarray（按 CSV shape/dtype）后调用本函数；本函数用确定性数据
    覆盖之，返回等长 numpy 数组列表。属性经 **kwargs 传入（expert_num/max_out_shape/... 及
    input_dtypes/soc_version 等元信息）。mask_stride 通过 attributes 传入（默认 5，约 20% mask）。
"""
import numpy as np

MASK_VAL = 1000000
BLOCK_BYTES = 32


def _elem_size(token_dtype):
    return 1 if int(token_dtype) == 2 else 2


def _hs_bytes(token_dtype, H):
    es = _elem_size(token_dtype)
    return H + BLOCK_BYTES if int(token_dtype) == 2 else H * es


def _byte_val(global_byte_pos):
    # 纯 mixing hash（与 gather 源位置强绑定），与 C++ 参考模型 ByteVal 逐位一致
    x = (np.uint64(global_byte_pos) * np.uint64(1103515245) + np.uint64(12345))
    return np.uint8((x >> np.uint64(16)) & np.uint64(0xFF))


__input__ = {
    "kernel": {
        "ffn_wb_norm_ttk": "ffn_wb_norm_ttk_input",
    }
}


def ffn_wb_norm_ttk_input(expert_ids, token_data, session_tbl, microbatch_tbl,
                          *, expert_num, max_out_shape, token_dtype=0, need_schedule=0,
                          M=2, mask_stride=5, out_num=None, **kwargs):
    A, BS, K, H = (int(v) for v in max_out_shape)
    expert_num = int(expert_num)
    token_dtype = int(token_dtype)
    M = int(M)
    mask_stride = int(mask_stride)
    out_num = A if out_num is None else int(out_num)

    HS = _hs_bytes(token_dtype, H)
    n_tok = out_num * BS * K
    token_slots = A * M * BS * K

    # expert_ids：round-robin，mask_stride 命中位置置无效
    eids = np.empty((n_tok,), dtype=np.int32)
    for p in range(n_tok):
        masked = mask_stride > 0 and (p % mask_stride == mask_stride - 1)
        eids[p] = MASK_VAL if masked else (p % expert_num)

    # session/micro batch 查表
    sess = np.arange(A, dtype=np.int32)
    micro = (np.arange(A, dtype=np.int32) % M).astype(np.int32)

    # token_data：确定性字节填充（原始字节缓冲）
    n_bytes = token_slots * HS
    idx = np.arange(n_bytes, dtype=np.uint64)
    x = idx * np.uint64(1103515245) + np.uint64(12345)
    td = ((x >> np.uint64(16)) & np.uint64(0xFF)).astype(np.uint8)

    # token_data 以算子输入 dtype 的 byte 视图返回（保持与占位 ndarray 元素数一致）
    td_out = _as_input_view(td, token_data)

    return [eids, td_out, sess, micro]


def _as_input_view(byte_arr, placeholder):
    """把裸字节数组重解释为 placeholder 的 dtype/shape（元素总字节数须一致）。"""
    ph = np.asarray(placeholder)
    need = ph.size * ph.dtype.itemsize if ph.size else byte_arr.size
    if byte_arr.size < need:
        byte_arr = np.concatenate([byte_arr, np.zeros(need - byte_arr.size, dtype=np.uint8)])
    byte_arr = byte_arr[:need]
    return np.frombuffer(byte_arr.tobytes(), dtype=ph.dtype).reshape(ph.shape) if ph.size else byte_arr

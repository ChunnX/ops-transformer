"""
LightningIndexer BSND+BSND TileLang-Ascend — v9

基于 v8 增加 P11 优化：固定核数，用满物理核

v8 → v9 改动:
  - 移除 _find_exact_core_num（要求 block_num 整除核数）
  - 改为固定 core_num = min(block_num, max_cores) 用满所有物理核
  - tasks_per_core 使用向上取整，尾部核通过 task guard 跳过多余任务
  - Workspace 按物理核数分配，常驻 L2 Cache
  - 预期收益: 核利用率从 16/20 提升到 20/20

保留部分 (v8):
  - P1: C/V per-iteration ping-pong
  - P2: L0C 双缓冲
  - P3: K DMA 预取
  - P7: Weights cache 复用
  - P8: L1/L0C 地址钉扎
  - S1 切分, Q 复用, Causal S2 裁剪
  - 段式 TopK + MergeSort, 配对 sync_all
"""
import tilelang
import tilelang.language as T
from tilelang import jit
import torch
from typing import Optional, Tuple

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _get_cube_core_num() -> int:
    try:
        import torch_npu as _tnpu
        props = _tnpu.npu.get_device_properties(0)
        return props.cube_core_num
    except Exception:
        return 20


def make_lightning_indexer_kernel(
    B: int, S1: int, S2: int, N1: int, D: int = 128, N2: int = 1,
    TOP_K: int = 2048, sparse_mode: int = 0, input_dtype: str = "float16",
    seg_size: int = 4096,
    block_n: Optional[int] = None,
    max_cores: Optional[int] = None,
):
    G = N1 // N2
    calc_dtype = "float"

    BLOCK_N = block_n if block_n is not None else 128
    BLOCK_M = 16
    BLOCK_K = D

    G_tiles = (G + BLOCK_M - 1) // BLOCK_M
    G_padded = G_tiles * BLOCK_M

    S2_padded = ((S2 + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    num_s2_blocks = S2_padded // BLOCK_N

    if G % 32 == 0:
        VECTOR_BASEG = 32
    elif G % 16 == 0:
        VECTOR_BASEG = 16
    elif G % 8 == 0:
        VECTOR_BASEG = 8
    else:
        VECTOR_BASEG = G

    TOP_K_ALIGNED = ((TOP_K + 63) // 64) * 64

    # 动态 SEG_SIZE：参考 AscendC arch22 实现，当 TopK 大且需要多段 merge 时，
    # 缩小 SEG_SIZE 以控制 TopK tmp 缓冲区（= SEG_SIZE * 16 bytes）不超出 UB。
    # TopK tmp 上限 32KB → SEG_SIZE 上限 2048（float32）。
    SEG_SIZE = seg_size
    if S2_padded <= SEG_SIZE:
        SEG_SIZE = S2_padded
    # 当 S2 > SEG_SIZE（需多段 merge）且 TopK tmp 过大时，缩小 SEG_SIZE
    _topk_tmp_bytes = SEG_SIZE * 16  # Sort 需要 4*alignedCount 个 float
    _topk_tmp_limit = 32768          # 32KB 上限，参考 arch22 的保守策略
    if S2_padded > SEG_SIZE and _topk_tmp_bytes > _topk_tmp_limit:
        SEG_SIZE = _topk_tmp_limit // 16  # = 2048
        # SEG_SIZE 不能小于 TOP_K_ALIGNED，否则 topk 无法找到足够元素
        SEG_SIZE = max(SEG_SIZE, TOP_K_ALIGNED)
    num_segs = (S2_padded + SEG_SIZE - 1) // SEG_SIZE  # ceiling division

    S1_BLOCK = 4 if TOP_K <= 2048 else 2
    s1_blocks = (S1 + S1_BLOCK - 1) // S1_BLOCK

    block_num = B * s1_blocks * N2 * num_s2_blocks

    if max_cores is None or max_cores <= 0:
        max_cores = _get_cube_core_num()

    # P11: 固定核数 — 用满所有物理核
    core_num = min(block_num, max_cores)
    tasks_per_core = (block_num + core_num - 1) // core_num  # ceiling division

    if tasks_per_core >= num_s2_blocks and tasks_per_core % num_s2_blocks == 0:
        s2_per_group = num_s2_blocks
        bsn_groups = tasks_per_core // num_s2_blocks
    else:
        s2_per_group = 1
        bsn_groups = tasks_per_core

    qk_ws_shape    = (core_num, 2, S1_BLOCK * G_padded, BLOCK_N)
    score_ws_shape = (B, S1, N2, S2_padded)

    SIG_L0C = 2

    # P7: Weight cache 动态决策
    num_g_groups = G // VECTOR_BASEG
    w_cache_bytes = S1_BLOCK * num_g_groups * VECTOR_BASEG * BLOCK_N * 4
    use_weight_cache = (w_cache_bytes <= 65536) and (s2_per_group > 1)

    # ===== P8: 地址布局计算 =====
    # 策略：pin L1 + L0C。对于 UB 使用率过高的 case（Phase 1 + Phase 2
    # 超过 UB 75%），MEMORY_PLANNING 会做激进的跨阶段复用导致数据污染，
    # 此时回退 BLOCK_N=64。
    def _align32(x):
        return ((x + 31) // 32) * 32

    UB_LIMIT = 196352

    # 估算 UB 使用率：Phase 1 (user) + Phase 2 (raw) + TopK tmp + stride2/offset
    _p1_user = (2 * VECTOR_BASEG * BLOCK_N * 4
                + BLOCK_N * 4
                + VECTOR_BASEG * 2 + VECTOR_BASEG * 4
                + VECTOR_BASEG * BLOCK_N * 4)
    _p2_raw = (SEG_SIZE * 4 * 2 + SEG_SIZE // 8
               + 2 * TOP_K * 4 * 2 + 4 * TOP_K * 4
               + TOP_K_ALIGNED * 4 * 3 + TOP_K_ALIGNED // 8
               + TOP_K_ALIGNED * 2)
    _topk_tmp_ub = SEG_SIZE * 16           # TopK Sort 临时缓冲区
    _p4_ub = 2 * TOP_K_ALIGNED * 4 * 2    # stride2 + offset
    _ub_ratio = (_p1_user + _p2_raw + _topk_tmp_ub + _p4_ub) / UB_LIMIT

    # 仅在 UB 使用率 >85% 且 G>=64（Phase 1 buffer 足够大导致
    # MEMORY_PLANNING 做激进的跨阶段复用）时回退 BN=64
    if block_n is None and _ub_ratio > 0.85 and G >= 64:
        BLOCK_N = 64
        S2_padded = ((S2 + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        num_s2_blocks = S2_padded // BLOCK_N
        SEG_SIZE = seg_size
        if S2_padded <= SEG_SIZE:
            SEG_SIZE = S2_padded
        _topk_tmp_bytes = SEG_SIZE * 16
        if S2_padded > SEG_SIZE and _topk_tmp_bytes > _topk_tmp_limit:
            SEG_SIZE = max(_topk_tmp_limit // 16, TOP_K_ALIGNED)
        num_segs = (S2_padded + SEG_SIZE - 1) // SEG_SIZE  # ceiling division

        # 重算依赖 BLOCK_N 的变量
        block_num = B * s1_blocks * N2 * num_s2_blocks
        core_num = min(block_num, max_cores)
        tasks_per_core = (block_num + core_num - 1) // core_num
        if tasks_per_core >= num_s2_blocks and tasks_per_core % num_s2_blocks == 0:
            s2_per_group = num_s2_blocks
            bsn_groups = tasks_per_core // num_s2_blocks
        else:
            s2_per_group = 1
            bsn_groups = tasks_per_core
        qk_ws_shape = (core_num, 2, S1_BLOCK * G_padded, BLOCK_N)
        score_ws_shape = (B, S1, N2, S2_padded)
        num_g_groups = G // VECTOR_BASEG
        w_cache_bytes = S1_BLOCK * num_g_groups * VECTOR_BASEG * BLOCK_N * 4
        use_weight_cache = (w_cache_bytes <= 65536) and (s2_per_group > 1)

    # L1 地址
    _q_l1_bytes = S1_BLOCK * G_tiles * BLOCK_M * BLOCK_K * 2   # f16
    _k_l1_bytes = 3 * BLOCK_N * BLOCK_K * 2                     # f16, triple buffer
    _q_l1_addr = 0
    _k_l1_addr = _align32(_q_l1_bytes)
    _l1_total = _k_l1_addr + _k_l1_bytes
    assert _l1_total <= 524032, f"L1 overflow: {_l1_total} > 524032"

    # L0C 地址
    _acc_l0c_addr = 0

    @jit(
        out_idx=[6, 7],
        workspace_idx=[3, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        },
    )
    def kernel_func():
        @T.prim_func
        def main(
            Query:          T.Tensor((B, S1, N1, D), input_dtype),
            Key:            T.Tensor((B, S2, N2, D), input_dtype),
            Weights:        T.Tensor((B, S1, N1), input_dtype),
            QK_Workspace:   T.Tensor(qk_ws_shape, calc_dtype),
            actual_q_len:   T.Tensor((B,), "int32"),
            actual_k_len:   T.Tensor((B,), "int32"),
            Out:            T.Tensor((B, S1, N2, TOP_K), "int32"),
            OutVal:         T.Tensor((B, S1, N2, TOP_K), input_dtype),
            Score_Workspace: T.Tensor(score_ws_shape, calc_dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ===== Cube 侧 =====
                q_l1    = T.alloc_L1((S1_BLOCK * G_tiles, BLOCK_M, BLOCK_K), input_dtype)
                k_l1    = T.alloc_L1((3, BLOCK_N, BLOCK_K), input_dtype)
                acc_l0c = T.alloc_L0C((2, BLOCK_M, BLOCK_N), calc_dtype)

                # ===== Vector 侧：阶段1 =====
                mm_res_ub     = T.alloc_ub((VECTOR_BASEG, BLOCK_N), calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, BLOCK_N), calc_dtype)
                reduce_g_ub   = T.alloc_ub(BLOCK_N, calc_dtype)

                # P7: Weight buffers — 全部无条件分配，MEMORY_PLANNING 优化未使用的
                w_raw_ub     = T.alloc_ub(VECTOR_BASEG, input_dtype)
                weight_ub    = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                weight_2d_ub = T.alloc_ub((VECTOR_BASEG, BLOCK_N), calc_dtype)
                w_cache_ub   = T.alloc_ub(
                    (S1_BLOCK * num_g_groups, VECTOR_BASEG, BLOCK_N), calc_dtype
                )

                # ===== Vector 侧：阶段2 =====
                score_seg_ub  = T.alloc_ub(SEG_SIZE, calc_dtype)
                topk_a_ub     = T.alloc_ub(2 * TOP_K_ALIGNED, calc_dtype)
                topk_b_ub     = T.alloc_ub(2 * TOP_K_ALIGNED, calc_dtype)
                merged_ub     = T.alloc_ub(4 * TOP_K_ALIGNED, calc_dtype)
                index_ub      = T.alloc_ub(SEG_SIZE, calc_dtype)
                mask_ub       = T.alloc_ub(SEG_SIZE // 8, "uint8")
                topk_index_ub = T.alloc_ub(TOP_K_ALIGNED, calc_dtype)
                output_ub     = T.alloc_ub(TOP_K_ALIGNED, "int32")
                output_val_ub = T.alloc_ub(TOP_K_ALIGNED, input_dtype)
                score_topk_ub = T.alloc_ub(TOP_K_ALIGNED, calc_dtype)
                mask_topk_ub  = T.alloc_ub(TOP_K_ALIGNED // 8, "uint8")
                # P4: stride-2 pattern for vectorized index offset
                stride2_topk_ub = T.alloc_ub(2 * TOP_K_ALIGNED, calc_dtype)
                offset_topk_ub  = T.alloc_ub(2 * TOP_K_ALIGNED, calc_dtype)

                # ===== P8: 固定地址布局 =====
                # 只 pin L1 + L0C
                T.annotate_address({
                    q_l1:    _q_l1_addr,
                    k_l1:    _k_l1_addr,
                    acc_l0c: _acc_l0c_addr,
                })

                # ===== C scope: 同 v6 =====
                with T.Scope("C"):
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_flag("FIX", "M", SIG_L0C + 1)

                    for bsn_off in T.serial(bsn_groups):
                        group_task_id = cid * tasks_per_core + bsn_off * s2_per_group
                        # P11: clamp for safe indexing on excess tasks
                        safe_gid = T.if_then_else(
                            group_task_id < block_num, group_task_id, 0)
                        n2_idx     = (safe_gid // num_s2_blocks) % N2
                        s1_blk_idx = (safe_gid // (num_s2_blocks * N2)) % s1_blocks
                        b_idx      = safe_gid // (num_s2_blocks * N2 * s1_blocks)
                        s1_start   = s1_blk_idx * S1_BLOCK

                        for s1_local in range(S1_BLOCK):
                            s1_idx = T.if_then_else(s1_start + s1_local < S1,
                                                    s1_start + s1_local, S1 - 1)
                            for g_tile in range(G_tiles):
                                T.copy(
                                    Query[b_idx, s1_idx,
                                          n2_idx * G + g_tile * BLOCK_M :
                                          n2_idx * G + (g_tile + 1) * BLOCK_M,
                                          0:D],
                                    q_l1[s1_local * G_tiles + g_tile, :, :],
                                )

                        first_s2_blk = safe_gid % num_s2_blocks
                        first_s2_start = first_s2_blk * BLOCK_N
                        T.copy(Key[b_idx, first_s2_start:first_s2_start + BLOCK_N,
                                   n2_idx, 0:D], k_l1[0, :, :])

                        # P3.5: 三缓冲 — 预加载第 2 个 K
                        if s2_per_group > 1:
                            second_s2_blk = (safe_gid + 1) % num_s2_blocks
                            second_s2_start = second_s2_blk * BLOCK_N
                            T.copy(Key[b_idx, second_s2_start:second_s2_start + BLOCK_N,
                                       n2_idx, 0:D], k_l1[1, :, :])

                        for s2_local in T.serial(s2_per_group):
                            pp = s2_local % 2
                            cur_k = s2_local % 3
                            nxt_k = (s2_local + 2) % 3  # 预取 2 步后的 K

                            T.wait_cross_flag(2 + pp)

                            s2_blk   = (safe_gid + s2_local) % num_s2_blocks
                            s2_start = s2_blk * BLOCK_N

                            if s2_local + 2 < s2_per_group:
                                next_s2_blk   = (safe_gid + s2_local + 2) % num_s2_blocks
                                next_s2_start = next_s2_blk * BLOCK_N
                                T.copy(Key[b_idx, next_s2_start:next_s2_start + BLOCK_N,
                                           n2_idx, 0:D], k_l1[nxt_k, :, :])

                            for s1_local in range(S1_BLOCK):
                                for g_tile in range(G_tiles):
                                    flat_idx = s1_local * G_tiles + g_tile
                                    side = flat_idx % 2

                                    T.wait_flag("FIX", "M", SIG_L0C + side)
                                    T.gemm_v0(
                                        q_l1[flat_idx, :, :], k_l1[cur_k, :, :],
                                        acc_l0c[side, :, :],
                                        transpose_B=True, init=True,
                                    )
                                    T.set_flag("M", "FIX", SIG_L0C + side)
                                    T.wait_flag("M", "FIX", SIG_L0C + side)
                                    T.copy(
                                        acc_l0c[side, :, :],
                                        QK_Workspace[cid, pp,
                                                     flat_idx * BLOCK_M :
                                                     (flat_idx + 1) * BLOCK_M,
                                                     0:BLOCK_N],
                                        enable_relu=True,
                                    )
                                    T.set_flag("FIX", "M", SIG_L0C + side)

                            T.set_cross_flag("FIX", pp)

                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.wait_flag("FIX", "M", SIG_L0C + 1)
                    T.sync_all()

                # ===== V scope =====
                with T.Scope("V"):
                    T.set_cross_flag("V", 2)
                    T.set_cross_flag("V", 3)

                    # P4: 初始化 stride-2 pattern [0,1,0,1,...,0,1]
                    # 偶数位=0(value不变), 奇数位=1(index加offset)
                    T.tile.fill(stride2_topk_ub, 0)
                    for _i in range(TOP_K_ALIGNED):
                        stride2_topk_ub[_i * 2 + 1] = T.cast(1, calc_dtype)

                    # Phase 1-V: reduce
                    for bsn_off in T.serial(bsn_groups):
                        group_task_id = cid * tasks_per_core + bsn_off * s2_per_group
                        # P11: clamp for safe indexing on excess tasks
                        safe_gid = T.if_then_else(
                            group_task_id < block_num, group_task_id, 0)
                        n2_idx     = (safe_gid // num_s2_blocks) % N2
                        s1_blk_idx = (safe_gid // (num_s2_blocks * N2)) % s1_blocks
                        b_idx      = safe_gid // (num_s2_blocks * N2 * s1_blocks)
                        s1_start   = s1_blk_idx * S1_BLOCK

                        if use_weight_cache:
                            # ★ P7: 预计算 weight_2d 缓存 ★
                            for s1_local in range(S1_BLOCK):
                                s1_idx = T.if_then_else(
                                    s1_start + s1_local < S1,
                                    s1_start + s1_local, S1 - 1,
                                )
                                for g_id in range(num_g_groups):
                                    cache_idx = s1_local * num_g_groups + g_id
                                    T.copy(
                                        Weights[b_idx, s1_idx,
                                                n2_idx * G + g_id * VECTOR_BASEG :
                                                n2_idx * G + (g_id + 1) * VECTOR_BASEG],
                                        w_raw_ub,
                                    )
                                    T.tile.cast(
                                        weight_ub, w_raw_ub,
                                        "CAST_NONE", VECTOR_BASEG,
                                    )
                                    T.tile.broadcast(
                                        w_cache_ub[cache_idx, :, :], weight_ub,
                                    )

                        for s2_local in T.serial(s2_per_group):
                            pp = s2_local % 2
                            T.wait_cross_flag(pp)

                            s2_blk   = (safe_gid + s2_local) % num_s2_blocks
                            s2_start = s2_blk * BLOCK_N

                            for s1_local in range(S1_BLOCK):
                                if s1_start + s1_local < S1:
                                    s1_idx = s1_start + s1_local

                                    s2_valid = actual_k_len[b_idx]
                                    if sparse_mode == 3:
                                        causal_limit = (
                                            actual_k_len[b_idx]
                                            - actual_q_len[b_idx] + s1_idx + 1
                                        )
                                        s2_valid = T.if_then_else(
                                            causal_limit > 0, causal_limit, 0
                                        )

                                    if s2_start < s2_valid:
                                        T.tile.fill(reduce_tmp_ub, 0)

                                        if use_weight_cache:
                                            # ★ P7: 使用缓存的 weight_2d ★
                                            for g_id in range(num_g_groups):
                                                qk_g_start = (
                                                    s1_local * G_padded
                                                    + g_id * VECTOR_BASEG
                                                )
                                                cache_idx = s1_local * num_g_groups + g_id
                                                T.copy(
                                                    QK_Workspace[cid, pp,
                                                                 qk_g_start :
                                                                 qk_g_start + VECTOR_BASEG,
                                                                 0:BLOCK_N],
                                                    mm_res_ub,
                                                )
                                                T.tile.mul(
                                                    mm_res_ub, mm_res_ub,
                                                    w_cache_ub[cache_idx, :, :],
                                                )
                                                T.tile.add(
                                                    reduce_tmp_ub, mm_res_ub,
                                                    reduce_tmp_ub,
                                                )
                                        else:
                                            # 非缓存模式: 逐次加载 (同 v6)
                                            for g_id in range(num_g_groups):
                                                qk_g_start = (
                                                    s1_local * G_padded
                                                    + g_id * VECTOR_BASEG
                                                )
                                                T.copy(
                                                    QK_Workspace[cid, pp,
                                                                 qk_g_start :
                                                                 qk_g_start + VECTOR_BASEG,
                                                                 0:BLOCK_N],
                                                    mm_res_ub,
                                                )
                                                T.copy(
                                                    Weights[b_idx, s1_idx,
                                                            n2_idx * G + g_id * VECTOR_BASEG :
                                                            n2_idx * G + (g_id + 1) * VECTOR_BASEG],
                                                    w_raw_ub,
                                                )
                                                T.tile.cast(
                                                    weight_ub, w_raw_ub,
                                                    "CAST_NONE", VECTOR_BASEG,
                                                )
                                                T.tile.broadcast(
                                                    weight_2d_ub, weight_ub,
                                                )
                                                T.tile.mul(
                                                    mm_res_ub, mm_res_ub,
                                                    weight_2d_ub,
                                                )
                                                T.tile.add(
                                                    reduce_tmp_ub, mm_res_ub,
                                                    reduce_tmp_ub,
                                                )

                                        T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)
                                        # P11: guard shared memory write for excess tasks
                                        if group_task_id < block_num:
                                            T.copy(
                                                reduce_g_ub,
                                                Score_Workspace[b_idx, s1_idx, n2_idx,
                                                                s2_start :
                                                                s2_start + BLOCK_N],
                                            )

                            T.set_cross_flag("V", 2 + pp)

                    T.sync_all()

                    # ===== Phase 2: TopK (同 v6) =====
                    for bsn_off in T.serial(bsn_groups):
                        group_task_id = cid * tasks_per_core + bsn_off * s2_per_group
                        # P11: skip excess tasks entirely
                        if group_task_id < block_num:
                          first_s2_blk = group_task_id % num_s2_blocks

                          if first_s2_blk == 0:
                            n2_idx     = (group_task_id // num_s2_blocks) % N2
                            s1_blk_idx = (
                                group_task_id // (num_s2_blocks * N2)
                            ) % s1_blocks
                            b_idx = group_task_id // (
                                num_s2_blocks * N2 * s1_blocks
                            )
                            s1_start = s1_blk_idx * S1_BLOCK

                            for s1_local in range(S1_BLOCK):
                                if s1_start + s1_local < S1:
                                    s1_idx = s1_start + s1_local

                                    T.tile.arith_progression(
                                        index_ub,
                                        T.cast(0, calc_dtype),
                                        T.cast(1, calc_dtype),
                                        SEG_SIZE,
                                    )

                                    # P4: 初始化 topk buffer 的 padding 区域为 -inf
                                    # 避免 merge_sort 时 padding 元素干扰结果
                                    if TOP_K_ALIGNED > TOP_K:
                                        T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))
                                        T.tile.fill(topk_b_ub, -T.infinity(calc_dtype))

                                    for seg in range(num_segs):
                                        T.copy(
                                            Score_Workspace[
                                                b_idx, s1_idx, n2_idx,
                                                seg * SEG_SIZE :
                                                (seg + 1) * SEG_SIZE],
                                            score_seg_ub,
                                        )

                                        if sparse_mode == 0:
                                            T.tile.compare(
                                                mask_ub, index_ub,
                                                actual_k_len[b_idx]
                                                - seg * SEG_SIZE,
                                                "LT",
                                            )
                                        else:
                                            causal_limit = (
                                                actual_k_len[b_idx]
                                                - actual_q_len[b_idx]
                                                + s1_idx + 1
                                            )
                                            clamped = T.if_then_else(
                                                causal_limit > 1,
                                                causal_limit, 1,
                                            )
                                            effective_s2 = T.if_then_else(
                                                clamped < actual_k_len[b_idx],
                                                clamped, actual_k_len[b_idx],
                                            )
                                            T.tile.compare(
                                                mask_ub, index_ub,
                                                effective_s2 - seg * SEG_SIZE,
                                                "LT",
                                            )
                                        T.tile.select(
                                            score_seg_ub, mask_ub, score_seg_ub,
                                            -T.infinity(calc_dtype),
                                            "VSEL_TENSOR_SCALAR_MODE",
                                        )

                                        T.tile.topk(
                                            topk_b_ub, score_seg_ub,
                                            TOP_K, SEG_SIZE,
                                        )

                                        if num_segs > 1:
                                            # P4: 向量化 index offset（替代标量循环）
                                            T.tile.mul(
                                                offset_topk_ub,
                                                stride2_topk_ub,
                                                T.cast(seg * SEG_SIZE, calc_dtype),
                                            )
                                            T.tile.add(
                                                topk_b_ub, topk_b_ub, offset_topk_ub,
                                            )

                                        if seg == 0:
                                            T.copy(topk_b_ub, topk_a_ub)
                                        else:
                                            T.tile.merge_sort(
                                                merged_ub, topk_a_ub, topk_b_ub
                                            )
                                            T.copy(
                                                merged_ub[0 : 2 * TOP_K],
                                                topk_a_ub,
                                            )

                                    T.tile.gather_mask(
                                        topk_index_ub, topk_a_ub, "P1010"
                                    )
                                    T.tile.fill(score_topk_ub, 0)
                                    T.tile.gather_mask(
                                        score_topk_ub, topk_a_ub, "P0101"
                                    )
                                    T.tile.compare(
                                        mask_topk_ub, score_topk_ub,
                                        T.cast(-1e30, calc_dtype), "GT",
                                    )
                                    T.tile.select(
                                        topk_index_ub, mask_topk_ub,
                                        topk_index_ub,
                                        T.cast(-1.0, calc_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.tile.cast(
                                        output_ub, topk_index_ub,
                                        "CAST_ROUND", TOP_K,
                                    )

                                    if s1_idx >= actual_q_len[b_idx]:
                                        T.tile.fill(output_ub, -1)

                                    T.copy(
                                        output_ub,
                                        Out[b_idx, s1_idx, n2_idx, 0:TOP_K],
                                    )

                                    T.tile.gather_mask(
                                        topk_index_ub, topk_a_ub, "P0101"
                                    )
                                    T.tile.cast(
                                        output_val_ub, topk_index_ub,
                                        "CAST_RINT", TOP_K,
                                    )
                                    T.copy(
                                        output_val_ub,
                                        OutVal[b_idx, s1_idx, n2_idx, 0:TOP_K],
                                    )

        return main
    return kernel_func()


# ============================================================
# Wrapper
# ============================================================
def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 0,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    return_value: bool = False,
    seg_size: int = 4096,
    block_n: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert layout_query == "BSND" and layout_key == "BSND"
    B, S1, N1, D = query.shape
    _, S2, N2, _ = key.shape
    assert D == 128
    input_dtype = "float16" if query.dtype == torch.float16 else "bfloat16"

    if actual_seq_lengths_query is None:
        actual_seq_lengths_query = torch.full(
            (B,), S1, dtype=torch.int32, device=query.device
        )
    if actual_seq_lengths_key is None:
        actual_seq_lengths_key = torch.full(
            (B,), S2, dtype=torch.int32, device=query.device
        )

    max_cores = _get_cube_core_num()

    func = make_lightning_indexer_kernel(
        B=B, S1=S1, S2=S2, N1=N1, D=D, N2=N2,
        TOP_K=sparse_count, sparse_mode=sparse_mode, input_dtype=input_dtype,
        seg_size=seg_size, block_n=block_n, max_cores=max_cores,
    )
    indices, values = func(
        query, key, weights,
        actual_seq_lengths_query, actual_seq_lengths_key,
    )
    return indices, values


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import sys

    sys.path.insert(
        0,
        "/home/y00936642/ops-transformer/attention/lightning_indexer/tests/pytest",
    )
    from result_compare_method import check_result
    import torch_npu

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()

    test_cases = [
        (2, 16, 256, 16, 1, 128, 64, 0, "basic: S2=256, G=16"),
        (2, 16, 512, 32, 1, 128, 128, 0, "basic: S2=512, G=32"),
        (2, 1, 2137, 16, 1, 128, 64, 0, "non-aligned: S2=2137"),
        (2, 4, 3072, 24, 1, 128, 1774, 0, "non-aligned G=24: S2=3072"),
        (2, 1, 5248, 16, 1, 128, 308, 0, "long non-aligned: S2=5248"),
        (2, 1, 8192, 8, 1, 128, 2048, 0, "long: S2=8192, G=8"),
        (2, 3, 16384, 64, 1, 128, 2048, 0, "long: S2=16384, G=64"),
        (2, 1, 32768, 16, 1, 128, 2048, 0, "long: S2=32768, G=16"),
        (28, 1, 2137, 16, 1, 128, 2048, 0, "excel: B=28, S2=2137"),
        (56, 1, 4095, 32, 1, 128, 2048, 0, "excel: B=56, S2=4095, G=32"),
        (36, 1, 5248, 16, 1, 128, 308, 0, "excel: B=36, S2=5248, TopK=308"),
        (12, 2, 8192, 24, 1, 128, 2048, 0, "excel: B=12, S1=2, S2=8192, G=24"),
        (6, 4, 16384, 64, 1, 128, 2048, 0, "excel: B=6, S1=4, S2=16384, G=64"),
        (4, 1, 32768, 8, 1, 128, 2048, 3, "excel: B=4, S2=32768, G=8"),
        (2, 4, 4096, 16, 1, 128, 64, 0, "varlen_k: act_k<S2"),
        (2, 8, 2048, 16, 1, 128, 64, 0, "varlen_q: act_q<S1"),
        (2, 16, 256, 16, 1, 128, 64, 3, "causal: S2=256, G=16"),
        (2, 4, 2048, 32, 1, 128, 128, 3, "causal: S2=2048, G=32"),
        (2, 1, 8192, 8, 1, 128, 2048, 3, "causal: S2=8192, G=8, long"),
    ]

    all_passed = True
    for tc_idx, (B, S1, S2, N1, N2, D, TOP_K, s_mode, desc) in enumerate(
        test_cases
    ):
        G = N1 // N2
        BLOCK_N = 128
        S2_padded = ((S2 + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        num_s2_blocks = S2_padded // BLOCK_N
        S1_BLOCK = 4 if TOP_K <= 2048 else 2
        s1_blocks_tc = (S1 + S1_BLOCK - 1) // S1_BLOCK
        block_num = B * s1_blocks_tc * N2 * num_s2_blocks

        # 计算 weight cache 状态
        if G % 32 == 0: VBG = 32
        elif G % 16 == 0: VBG = 16
        elif G % 8 == 0: VBG = 8
        else: VBG = G
        bn = BLOCK_N
        w_bytes = S1_BLOCK * G * bn * 4
        s2pg = num_s2_blocks if (block_num // min(block_num, 20)) >= num_s2_blocks and (block_num // min(block_num, 20)) % num_s2_blocks == 0 else 1
        w_cached = "cache" if (w_bytes <= 65536 and s2pg > 1) else "no-cache"

        print(f"\n{'='*70}")
        print(f"[TC{tc_idx+1}] {desc}")
        print(
            f"  B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, "
            f"TopK={TOP_K}, G={G}, sparse_mode={s_mode}"
        )
        print(
            f"  block_num={block_num} "
            f"(s1_blocks={s1_blocks_tc}, num_s2_blocks={num_s2_blocks})"
            f" weight={w_cached}({w_bytes}B)"
        )
        print(f"{'='*70}")

        query = torch.randn((B, S1, N1, D), dtype=torch.float16)
        key = torch.randn((B, S2, N2, D), dtype=torch.float16)
        weights = torch.randn((B, S1, N1), dtype=torch.float16).abs()

        act_q_list = [S1] * B
        act_k_list = [S2] * B
        actual_seq_lengths_query = None
        actual_seq_lengths_key = None

        if "varlen_k" in desc:
            act_k_list = [S2 - 500 - (i * 200) % 1000 for i in range(B)]
            actual_seq_lengths_key = torch.tensor(
                act_k_list, dtype=torch.int32, device=query.device
            )

        if "varlen_q" in desc:
            act_q_list = [S1 - 2 - i for i in range(B)]
            actual_seq_lengths_query = torch.tensor(
                act_q_list, dtype=torch.int32, device=query.device
            )

        torch.npu.synchronize()

        # Golden: AscendC 算子
        golden_kwargs = dict(
            sparse_count=TOP_K, sparse_mode=s_mode,
            layout_query="BSND", layout_key="BSND",
        )
        if actual_seq_lengths_key is not None:
            golden_kwargs["actual_seq_lengths_key"] = actual_seq_lengths_key
        if actual_seq_lengths_query is not None:
            golden_kwargs["actual_seq_lengths_query"] = actual_seq_lengths_query
        cann_idx, cann_value = torch_npu.npu_lightning_indexer(
            query, key, weights, **golden_kwargs
        )
        torch.npu.synchronize()
        print("  Golden done.")

        # 手动计算 topk_value 用于 check_result 的 value 比较
        q_f, k_f, w_f = query.float(), key.float(), weights.float()
        topk_value = torch.zeros(
            (B, N2, S1, S2), dtype=torch.float32, device=query.device
        )
        for b in range(B):
            act_k = act_k_list[b]
            act_q = act_q_list[b]
            for s1 in range(S1):
                for n2 in range(N2):
                    q_s = q_f[b, s1, n2 * G : (n2 + 1) * G, :]
                    k_s = k_f[b, :act_k, n2, :]
                    score = torch.relu(q_s @ k_s.T)
                    score = (
                        score * w_f[b, s1, n2 * G : (n2 + 1) * G].unsqueeze(-1)
                    ).sum(dim=0)
                    if s_mode == 3:
                        causal_len = min(act_k, act_k - act_q + s1 + 1)
                        score[causal_len:] = 0
                    topk_value[b, n2, s1, :act_k] = score
        torch.npu.synchronize()

        print("  Compiling TileLang...")
        try:
            tl_indices, tl_values = lightning_indexer(
                query, key, weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                sparse_count=TOP_K, sparse_mode=s_mode, return_value=True,
            )
            torch.npu.synchronize()
            print("  Kernel done.")
        except Exception as e:
            print(f"  *** ERROR: {e} ***")
            import traceback

            traceback.print_exc()
            all_passed = False
            continue

        params = (
            B, S1, S2, 0, 0, N1, N2, D, 0, 0,
            query.dtype, weights.dtype, torch.int32,
            act_q_list, act_k_list, "BSND", "BSND", TOP_K, s_mode,
            None, None, None, True,
        )
        result, pct = check_result(
            cann_idx, tl_indices, topk_value, tl_values, params
        )
        print(f"  Result: {result}, fulfill: {pct:.2f}%")
        if result != "Pass":
            all_passed = False

    print(f"\n{'='*70}")
    print("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED!")
    print(f"{'='*70}")

import math
import os
import sys
import threading
import torch
from enum import IntEnum

from cannbotdsl.channel import Channel
from cannbotdsl.constexpr import const_expr, range_constexpr
from cannbotdsl.delay_line import DelayLineGroup
from cannbotdsl.arch import get_subblock_id, get_block_idx
from cannbotdsl import select as dyn_select
from cannbotdsl.integer import Int32, Int64
from cannbotdsl._mlir.dialects import arith, ascvec
from cannbotdsl.ascir import extract_buffer
from cannbotdsl.core.scalar_conversion import coerce_scalar_value
from cannbotdsl import dtypes
from cannbotdsl import Dim, TensorSpec
from cannbotdsl.float import Float8E4M3FN
from cannbotdsl.jit_function import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import add as vadd
from cannbotdsl.math import cast, log as vlog, matmul
from cannbotdsl.types import ChannelKind, MemLoc, PIPE, Tensor
from cannbotdsl.tensor import (
    ceil_div,
    partition_view,
    local_slice,
    make_bounded_tiler,
    tile_view,
    make_copy_engine,
    make_partition_tiler,
    mem_copy,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl._mlir import ir
from cannbotdsl.vf import vf
from cannbotdsl.sync import (
    cube_get_buf,
    cube_raw_l1_to_l0a,
    cube_raw_l1_to_l0b,
    cube_release_buf,
    cube_sync_intra_wait,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_intra_arrive,
)
from cannbotdsl import raw_reg as rr

# ---- 性能/实验环境开关（默认值均经板上 A/B 验证，勿随意翻转；见 dsl_impl_report.md §13）----
_DYN_TILE = os.environ.get("MQSMLA_DYN_TILE") == "1"   # GM→L1 源视图运行时行数实验（只影响搬运）
_NZ17 = os.environ.get("MQSMLA_NZ17", "1") == "1"      # vec0 直出 NZ 分形 + identity GM→L1（默认开）
_PV_P_ONCE = os.environ.get("MQSMLA_PV_P_ONCE", "1") == "1"  # bmm2 的 P 单次装载 L0A、4 mmad 共读
_PV_WIDE = os.environ.get("MQSMLA_PV_WIDE", "0") == "1"      # E2：4 mmad 攒宽 L0C + 单笔 fixpipe（已证伪）
_L0C_SHARED = os.environ.get("MQSMLA_L0C_SHARED", "0") == "1"  # E3：QK/PV 共享 L0C 环（存档实现）
_PV_HALF = os.environ.get("MQSMLA_PV_HALF", "0") == "1"       # E4：半宽两批 fixpipe（持平，证伪事务数假设）
# NZ17：UB 内每个 16×16 face 后加 1 行 padding，改变相邻 face 的 bank 映射。
# _NZ_CHUNK 是 face 间距（BF16 元素数）；padding 只存在于 UB，copy-out 时跳过。
_NZ_ROW = 16
_NZ_PAD_ROWS = 17
_NZ_CHUNK = _NZ_PAD_ROWS * _NZ_ROW

_DYN_KN = os.environ.get("MQSMLA_DYN_KN") == "1"

if _DYN_KN:
    import importlib

    _mdh = importlib.import_module("mqsmla.mmad_dyn_hack")

    def _patch_mmad_rewrite():
        import cannbotdsl.core.compiler.passes as _passes
        if getattr(_passes, "_mqsmla_mmad_dyn_kn_patched", False):
            return
        _orig = _passes._run_cannir_passes

        def _wrapped(module):
            _orig(module)
            _mdh.rewrite(module)

        _passes._run_cannir_passes = _wrapped
        _passes._mqsmla_mmad_dyn_kn_patched = True

    _patch_mmad_rewrite()

def _to_index(value):
    return coerce_scalar_value(value, ir.IndexType.get())

# ---- 编译期形状契约（冻结值，见接口文档「约束说明」）----
D = 512                  # 逻辑 head dim：nope 448 + rope 64（计算序 nope 在前）
N1 = 64                  # Q 头数唯一取值（SPLIT_G=128 与 runtime_n1 已删，2026-09-07）
N2 = 1                   # KV 头数（G=N1/N2=64）
TILE_N = 128             # n-tile：一个 s2 任务的 KV token 数
WS_DEPTH = 3             # GM ws 三深环：复用前借 softmax(T−3) 确认 loadQK(T−3) 已完成
D_ROPE = 64              # rope 段元素数（物理存储在行首）
D_NOPE = 448             # nope 段元素数
GROUP_SIZE_WIN = 32
GROUP_SIZE_CMP = 16
NUM_GROUPS_WIN = D // GROUP_SIZE_WIN
NUM_GROUPS_CMP = D // GROUP_SIZE_CMP

# ---- per-token-group 物理布局 ----
# 物理字节序 = rope[64] ‖ nope[448]；win 每 32 元素、cmp 每 16 元素共享 1 个 BF16 scale。
# scale 内联在数据段尾、无 pad 字段，均以 uint8 字节视图传入：
#   win_kv (FP8 E4M3)    544 B/token = 512B fp8 数据 + 32B scale
#   cmp_kv (FP4 E2M1 x2) 320 B/token = 256B fp4x2 数据（偶元素低 nibble）+ 64B scale
KV_ROW_BYTES_WIN = 544
KV_ROW_BYTES_CMP = 320
KV_SCALE_BYTES_WIN = NUM_GROUPS_WIN * 2
KV_SCALE_BYTES_CMP = NUM_GROUPS_CMP * 2
ROW_BF16_WIN = KV_ROW_BYTES_WIN // 2      # win 池 bf16 视图行宽 272
ROW_BF16_CMP = KV_ROW_BYTES_CMP // 2      # cmp 池 bf16 视图行宽 160
SCALE_BF16_OFF_WIN = 256                  # scale 段在 win 行内的 bf16 偏移（512B/2）
SCALE_BF16_OFF_CMP = 128                  # scale 段在 cmp 行内的 bf16 偏移（256B/2）

class QuantMode(IntEnum):
    CONTIGUOUS = 1

class Scenario(IntEnum):
    ORI_SPARSE = 0
    ORI_CMP_SPARSE = 1

METADATA_ROWS = 2
WIN_TOKENS_PER_QUERY = 128
CMP_TOKENS_PER_QUERY = 512

def _query_cost(has_cmp):
    # metadata 分核的均摊负载模型：每 query 负载恒定，折成 tile_n=128 的 tile 数作权重
    # （win 128 token = 1 tile；ORI_CMP_SPARSE 再加 cmp 512 token = 4 tile）。
    cost = -(-WIN_TOKENS_PER_QUERY // TILE_N)
    if has_cmp:
        cost += -(-CMP_TOKENS_PER_QUERY // TILE_N)
    return cost

def _balance_from_costs(costs, blocks):
    # host 贪心分核（照抄 cannbotdsl flash_attn 样例形态）：前缀和越过 (core+1)*target
    # 即换核，产出每核一个**连续区间** [start, start+count)；空核 start 停表尾、count=0。
    # 等权重（本算子的均摊模型）下退化为均分。
    total = len(costs)
    target = sum(costs) / blocks if blocks else 0
    starts = [total] * blocks
    counts = [0] * blocks
    cumulative = 0
    core = 0
    for i in range(total):
        if core < blocks - 1 and cumulative >= (core + 1) * target:
            core += 1
        if counts[core] == 0:
            starts[core] = i
        counts[core] += 1
        cumulative += costs[i]
    return starts, counts

def _ws_rows(blocks):
    # GM workspace 的 v0 区行数：核数 × ws_depth(3) × tile_n(128) 行，每行 512 bf16。
    return blocks * WS_DEPTH * TILE_N

# win 侧派生偏移（kernel 侧缺省口径；cmp 侧在 `_vec0_body` 内按池分派）：
_ROW_BF16 = ROW_BF16_WIN          # ch_batch 槽行距 272 bf16（两池统一按 win 的 544B 槽行距）
_NOPE_FP8_OFF = D_ROPE            # nope 数据段在 fp8 字节视图内的起始偏移（rope 在前）
_SCALE_BF16_OFF = SCALE_BF16_OFF_WIN  # scale 段在 win 槽行内的 bf16 偏移

_VL = 128                  # vec0 反量化的单笔向量宽度（bf16 lanes）
_DEFAULT_BLOCK_SIZE = 64
_TILE_ROWS = 16            # vec0 子块行数：一个 128-token tile 切 8 个 16 行子块（分形 face 高）

def _i64(gm, *idx):
    raw = gm[idx] if len(idx) > 1 else gm[idx[0]]
    ext = arith.extsi(ir.IntegerType.get_signless(64), raw)
    return Int64(ext)

class Kvcache:
    """PA_BBND 两级寻址（blockTable 值是全局物理块号，batch 只在下标上）：
    blk = s2_idx // blockSize; off = s2_idx % blockSize;
    phys_row = blockTable[bo][blk] * blockSize + off
    仅在 blockSize 非 2 的幂一侧被逐 token 标量调用（2 幂侧走前置 pass 的中转表）。
    """

    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
        self.block_size = int(block_size)

    def get_key_offset(self, gm_block_table, s2_idx, bo_idx=0, block_size=None):
        bs = Int64(self.block_size) if block_size is None else block_size
        phys_blk, off = self._pa_blk_off(gm_block_table, s2_idx, bo_idx, bs)
        return phys_blk * bs + off

    def _pa_blk_off(self, gm_block_table, s2_idx, bo_idx, bs):
        blk = s2_idx // bs
        off = s2_idx % bs
        return _i64(gm_block_table, Int64(bo_idx), blk), off

TILE = 128
D_TILES = D // TILE          # D=512 切 4 个 128 列 chunk（bmm1 的 K 轴 / bmm2 的 N 轴）

# ---- 每 AIC 的 L1 预算：K̃ 384 KiB + Q 96 KiB + p_l1 32 KiB = 512 KiB ----
# GM ws 与 L1 K̃ 都用 tick%3，但它们是不同存储：ws 在 loadQK 后即可复用，
# L1 K̃ 还要保留到 bmm2 装载结束。Q 用 m_seq 轮转，在同一 query 的所有 n-tile 间复用。
_KV_RING = 3                 # K̃ 三深环：同一代数据在 lag1 写/lag2 bmm1 读/lag3 bmm2 读三拍都活着
_KV_CHUNK_ELEMS = TILE * TILE
_KV_SLOT_ELEMS = _KV_CHUNK_ELEMS * D_TILES
_KV_L0B_FRACTAL = dict(m_start=0, k_start=0, m_step=8, k_step=8, src_stride=8, dst_stride=8)
_KV_BUF_NUM = 6              # 6 个子 buffer credit（3 代 × 2 半 D），MTE2 写前阻塞取、bmm2 读完还

_Q_RING = 3                  # 3 个半 D 槽；一份 Q 占环上相邻 2 槽，槽号为 (2*m_seq+h)%3
_Q_SLOT_ELEMS = 64 * (D // 2)
_Q_CHUNK_ELEMS = 64 * TILE
_Q_L0A_FRACTAL = dict(m_start=0, k_start=0, m_step=4, k_step=8, src_stride=4, dst_stride=4)

_P_L1_DEPTH = 2              # p_l1 跨核 Channel 深度（AIV store_p → AIC bmm2）
_Q_BUF_NUM = _Q_RING

# ---- ChannelArena id 纪律（烧过三次的教训：手挑 id 曾「不报错、只间歇挂死」）----
# sync_id 全空间 [0,15]：0..4 跨核 Channel 自取；本侧向 arena 申请并烧号到地板 5 让开；
# 保留段 {11..14}（AscendC 运行时）与 {10}（板上负面证据）整段避让。
# block-sync flag 必须在所有 Channel 构造之后申请（先申请会顶走 Channel 的 sync_id）。
_PLANNER_SYNC_ID_FLOOR = 5
_PLANNER_BUF_ID_FLOOR = 0

def _burn_to_floor(alloc, floor, *alloc_args):
    while True:
        got = alloc(*alloc_args)
        if min(got) >= floor:
            return got

def _alloc_kv_buf_ids():
    from cannbotdsl.channel_arena import _current_channel_arena
    arena = _current_channel_arena()
    return _burn_to_floor(arena.alloc_buf_ids, _PLANNER_BUF_ID_FLOOR,
                          MemLoc.L1, _KV_BUF_NUM)

def _alloc_q_buf_ids():
    from cannbotdsl.channel_arena import _current_channel_arena
    arena = _current_channel_arena()
    return _burn_to_floor(arena.alloc_buf_ids, _PLANNER_BUF_ID_FLOOR,
                          MemLoc.L1, _Q_BUF_NUM)

_TND_B_MAX = 256
_TND_BSEARCH_ROUNDS = 8
assert 2 ** _TND_BSEARCH_ROUNDS >= _TND_B_MAX, "二分轮数不足以覆盖 B 的上界"

_AIV1_ID_OFFSET = 16

def _alloc_ws_intra_ids():
    from cannbotdsl.channel_arena import _current_channel_arena
    arena = _current_channel_arena()
    return _burn_to_floor(arena.alloc_sync_ids, _PLANNER_SYNC_ID_FLOOR, 1)

_PHY_S2_PER_LOOP = 128      # 前置 pass 每趟处理的 token 数
_PHY_S2_PER_REG = 64        # 每寄存器（128-lane 窗口按 64 半窗）处理的 token 数
_PHY_OUT_PER_LOOP = 256
_PHY_OUT_PER_REG = 128
_ASCENDC_RESERVED_FLAG_IDS = (11, 12, 13, 14)   # AscendC 运行时保留段，整段禁碰

_BOARD_BAD_FLAG_IDS = (10,)                     # 板上负面证据：同组只差此号即确定性挂死，避让

def _alloc_block_sync_flag(what):
    from cannbotdsl.channel_arena import _current_channel_arena
    arena = _current_channel_arena()
    blocked = set(_ASCENDC_RESERVED_FLAG_IDS) | set(_BOARD_BAD_FLAG_IDS)
    while True:
        try:
            (fid,) = arena.alloc_sync_ids(1)
        except RuntimeError as e:
            raise RuntimeError(
                f"{what} 要不到 block-sync flag id：arena 已耗尽，且 [0,15] 里被占的号是 "
                f"{sorted(blocked)}（AscendC 运行时保留 {list(_ASCENDC_RESERVED_FLAG_IDS)} + "
                f"板上负面 {list(_BOARD_BAD_FLAG_IDS)}）。"
                f"原始错误：{e}") from e
        if fid not in blocked:
            return fid

_KV_NUM_PER_LOOP = 128
_TILE_SIZE = 64
_COMBINE_DIM = _ROW_BF16 * 2

def _clamp_nonneg(value):
    if isinstance(value, int):
        return max(0, value)
    is_neg = value < 0
    return Int64(arith.select(is_neg, Int64(0), value))

def _rt_min(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    a64 = a if not isinstance(a, int) else Int64(a)
    b64 = b if not isinstance(b, int) else Int64(b)
    return Int64(dyn_select(a64 < b64, a64, b64))

class Matmul:
    """AIC 侧：把 Q/K̃ 装入 L1，完成 QK、PV 两次矩阵乘并经 fixpipe 送 AIV。

    q_wide 有 3 个 [64,256] BF16 槽（96 KiB），一份 Q 占 2 槽。
    Q 在首个 n-tile 取读 credit、最后一个 n-tile 释放；随后才预取下一 query 的 Q。
    kv_wide 保存 3 份 [128,512] BF16 K̃（384 KiB），同一份数据依次用于 QK 和 PV，
    直到 PV 的 L1→L0B 装载结束才归还 credit。NZ 路径按 (D/16, token, 16) 存储，
    ND 路径由 nd2nz 搬运转换；两者供 L0 装载的分形内容一致。
    默认 QK/PV 使用独立 L0C Channel；其他 L0C 分支仅用于保留的环境开关实验。
    """

    def __init__(self, tile_cube_m, tile_vec_m, tile_n):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D

        self.nd2nz = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.identity = make_copy_engine(format_transform="identity", dtype=dtypes.bfloat16)
        self._kv_load_engine = self.identity if _NZ17 else self.nd2nz
        self.fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

        # Buffer 只提供物理存储，Q/K̃ 的复用同步由下方 _q_* / _kv_* 显式维护。
        self.q_wide = Buffer(MemLoc.L1, (tile_cube_m, _Q_RING * (D // 2)), dtypes.bfloat16)
        self._q_ids = _alloc_q_buf_ids()
        self.kv_wide = (Buffer(MemLoc.L1, (_KV_RING * D // 16, tile_n, 16), dtypes.bfloat16,
                               data_format="nd")
                        if _NZ17 else
                        Buffer(MemLoc.L1, (tile_n, _KV_RING * D), dtypes.bfloat16))
        self._kv_ids = _alloc_kv_buf_ids()
        self.l0a = Channel(MemLoc.L0A, shape=(tile_cube_m, TILE), dtype=dtypes.bfloat16, depth=2)
        self.l0a_p = Channel(MemLoc.L0A, shape=(tile_cube_m, tile_n), dtype=dtypes.bfloat16, depth=2)
        self.l0b = Channel(MemLoc.L0B, shape=(TILE, TILE), dtype=dtypes.bfloat16, depth=2)
        if const_expr(_L0C_SHARED):
            self.l0c_qk = None
            self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, self.tile_d),
                               dtype=dtypes.float32, depth=2)
        else:
            self.l0c_qk = Channel(MemLoc.L0C, shape=(tile_cube_m, TILE), dtype=dtypes.float32, depth=2)
            if const_expr(_PV_HALF):
                self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, self.tile_d // 2),
                                   dtype=dtypes.float32, depth=2)
            elif const_expr(_PV_WIDE):
                self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, self.tile_d),
                                   dtype=dtypes.float32, depth=1)
            else:
                self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, TILE), dtype=dtypes.float32,
                                   depth=2)

    @jit
    def load_q_wide(self, q_tile_gm, m_seq: Int64, rows=None):
        # Q GM → L1 Q 环的一代装载：两条半 D（各 256 列）nd2nz 搬运，搬前阻塞取写 credit。
        for h in range_constexpr(2):
            self._q_acquire_w(m_seq, h)
            dst = tile_view(self.q_wide, (self.tile_cube_m, D // 2),
                            (0, self._q_slot(m_seq, h)))
            if const_expr(_DYN_TILE and rows is not None):
                rows_a = (rows + Int64(15)) // Int64(16) * Int64(16)
                src = tile_view(
                    q_tile_gm,
                    make_bounded_tiler((rows_a, D // 2),
                                       (self.tile_cube_m, D // 2), (16, 16)),
                    (Int64(0), Int64(h)))
            else:
                src = tile_view(q_tile_gm, (self.tile_cube_m, D // 2), (0, h))
            mem_copy(dst, src, engine=self.nd2nz)
            self._q_release_w(m_seq, h)

    @jit
    def _q_slot_base(self, m_seq: Int64):
        return (m_seq * Int64(2)) % Int64(_Q_RING)

    @jit
    def _q_slot(self, m_seq: Int64, half):
        return (self._q_slot_base(m_seq) + Int64(half)) % Int64(_Q_RING)

    @jit
    def _q_chunk_off(self, m_seq: Int64, j):
        # j 是 128 列 chunk 号：j//2 选择半 D 槽，j%2 选择槽内的前/后 chunk。
        return (self._q_slot(m_seq, j // 2) * Int64(_Q_SLOT_ELEMS)
                + Int64((j % 2) * _Q_CHUNK_ELEMS))

    @jit
    def _q_id(self, m_seq: Int64, half):
        return Int64(self._q_ids[0]) + self._q_slot(m_seq, half)

    @jit
    def _q_acquire_w(self, m_seq: Int64, half):
        # 写前等可复用，写后通知 MTE1；读前等就绪，读完归还 MTE2。
        # 此处 get/release(mode=0) 是写侧的配对指令，数据就绪在 _q_release_w 发布。
        b = self._q_id(m_seq, half)
        cube_get_buf(PIPE.MTE2, b, 0)
        cube_release_buf(PIPE.MTE2, b, 0)

    @jit
    def _q_release_w(self, m_seq: Int64, half):
        b = self._q_id(m_seq, half)
        cube_get_buf(PIPE.MTE2, b, 1)
        cube_release_buf(PIPE.MTE2, b, 1)

    @jit
    def _q_acquire_r(self, m_seq: Int64):
        for h in range_constexpr(2):
            b = self._q_id(m_seq, h)
            cube_get_buf(PIPE.MTE1, b, 0)
            cube_release_buf(PIPE.MTE1, b, 0)

    @jit
    def _q_release_r(self, m_seq: Int64):

        for h in range_constexpr(2):
            b = self._q_id(m_seq, h)
            cube_get_buf(PIPE.MTE1, b, 1)
            cube_release_buf(PIPE.MTE1, b, 1)

    @jit
    def _q_acquire_r_first(self, m_seq: Int64, n):
        if n == Int64(0):
            self._q_acquire_r(m_seq)

    @jit
    def _q_release_r_last(self, m_seq: Int64, is_last):
        if is_last == Int64(1):
            self._q_release_r(m_seq)

    @jit
    def _kv_chunk_off(self, slot: Int64, j):
        # 每 slot 是 4 个 [128,128] chunk；偏移以 BF16 元素计，不是字节偏移。
        return slot * Int64(_KV_SLOT_ELEMS) + Int64(j * _KV_CHUNK_ELEMS)

    @jit
    def _kv_id_base(self, slot: Int64):
        return Int64(self._kv_ids[0]) + (slot * Int64(2)) % Int64(_KV_BUF_NUM)

    @jit
    def _kv_sub(self, slot: Int64, half):
        return self._kv_id_base(slot) + Int64(half)

    @jit
    def _kv_acquire_w(self, slot: Int64):
        for h in range_constexpr(2):
            b = self._kv_sub(slot, h)
            cube_get_buf(PIPE.MTE2, b, 0)
            cube_release_buf(PIPE.MTE2, b, 0)

    @jit
    def _kv_release_w(self, slot: Int64, half):
        b = self._kv_sub(slot, half)
        cube_get_buf(PIPE.MTE2, b, 1)
        cube_release_buf(PIPE.MTE2, b, 1)

    @jit
    def _kv_acquire_r(self, slot: Int64, half):
        b = self._kv_sub(slot, half)
        cube_get_buf(PIPE.MTE1, b, 0)
        cube_release_buf(PIPE.MTE1, b, 0)

    @jit
    def _kv_release_r(self, slot: Int64):
        for h in range_constexpr(2):
            b = self._kv_sub(slot, h)
            cube_get_buf(PIPE.MTE1, b, 1)
            cube_release_buf(PIPE.MTE1, b, 1)

    @jit
    def _bmm1_chunk_fanout(self, qk_slot, slot: Int64, j, m_seq: Int64, init,
                           actual_n=None):

        cube_raw_l1_to_l0a(self.l0a, self.q_wide,
                           l1_offset_elems=self._q_chunk_off(m_seq, j),
                           **_Q_L0A_FRACTAL)
        cube_raw_l1_to_l0b(self.l0b, self.kv_wide,
                           l1_offset_elems=self._kv_chunk_off(slot, j),
                           transpose=False, **_KV_L0B_FRACTAL)
        if const_expr(_DYN_KN):
            matmul(qk_slot, self.l0a, self.l0b, init=init, unit_flag=actual_n)
        else:
            matmul(qk_slot, self.l0a, self.l0b, init=init)
        if const_expr(_DYN_KN):
            _mdh.register("qk")

    def load_qk(self, k_tile_gm, row_tile, slot, rows=None):
        # lag1：GM ws 槽 → L1 K̃ 环，每次搬 D/2=256 列，搬完一半即发布该半的读就绪。
        # NZ17 路径的 GM 数据已去除 UB padding，按 [32,128,16] 分形直接 identity 拷入。
        self._kv_acquire_w(slot)
        for h in range_constexpr(2):
            if const_expr(_NZ17):
                dst = tile_view(self.kv_wide, (D // 32, self.tile_n, 16),
                                (slot * Int64(2) + Int64(h), 0, 0))
            else:
                dst = tile_view(self.kv_wide, (self.tile_n, D // 2),
                                (0, slot * Int64(2) + Int64(h)))
            if const_expr(_DYN_TILE and rows is not None):
                rows_a = (rows + Int64(15)) // Int64(16) * Int64(16)
                bounded = make_bounded_tiler(
                    (rows_a, D // 2), (self.tile_n, D // 2), (16, 16))
                if const_expr(row_tile is None):
                    src = tile_view(k_tile_gm, bounded, (Int64(0), Int64(h)))
                else:
                    src = tile_view(k_tile_gm, bounded,
                                    (row_tile, Int64(h)))
            elif const_expr(_NZ17):
                ws3 = k_tile_gm.view(
                    k_tile_gm.shape[0] // self.tile_n * (D // 16), self.tile_n, 16)
                rt = Int64(0) if row_tile is None else row_tile
                src = tile_view(ws3, (D // 32, self.tile_n, 16),
                                (rt * Int64(2) + Int64(h), 0, 0))
            elif const_expr(row_tile is None):
                src = tile_view(k_tile_gm, (self.tile_n, D // 2), (0, h))
            else:
                src = tile_view(k_tile_gm, (self.tile_n, D // 2), (row_tile, Int64(h)))
            mem_copy(dst, src, engine=self._kv_load_engine)
            self._kv_release_w(slot, h)

    def bmm1_fanout_q(self, slot, m_seq: Int64, n, is_last, actual_n=None):
        # lag2：S[64,128] = Q[64,512] @ K̃[128,512]^T，FP32 L0C 累加。
        # D 切为 4 个 128 宽的归约 chunk：首 chunk init=True，随后三个累加到同一 S。
        # Q 的读权限覆盖整个 query；K̃ 按半 D 交错等待，前两个 chunk 可先于后半就绪计算。
        # 此处不归还 K̃ 的读权限，因为 lag3 的 PV 还要读取同一 slot。
        if const_expr(_L0C_SHARED):
            sh = self.l0c
            qk_dst = tile_view(sh, (self.tile_cube_m, TILE), (0, 0))
        else:
            qk_dst = self.l0c_qk
        self._q_acquire_r_first(m_seq, n)
        self._kv_acquire_r(slot, 0)
        self._bmm1_chunk_fanout(qk_dst, slot, 0, m_seq, init=True,
                                actual_n=actual_n)
        self._bmm1_chunk_fanout(qk_dst, slot, 1, m_seq, init=False,
                                actual_n=actual_n)
        self._kv_acquire_r(slot, 1)
        self._bmm1_chunk_fanout(qk_dst, slot, 2, m_seq, init=False,
                                actual_n=actual_n)
        self._bmm1_chunk_fanout(qk_dst, slot, 3, m_seq, init=False,
                                actual_n=actual_n)
        self._q_release_r_last(m_seq, is_last)

    def store_s(self, qk_ub_ch, partition):
        # S 从 L0C 经 fixpipe（dual_dst、split-M partition）跨核送到 AIV 的 qk_ub。
        # split-M 将 [64,128] 的 64 个头对半分给两个 AIV。
        # AIV0 处理 head[0:32]，AIV1 处理 head[32:64]，store_p/finalize 保持同一划分。
        if const_expr(_L0C_SHARED):
            sh = self.l0c
            mem_copy(qk_ub_ch, tile_view(sh, (self.tile_cube_m, TILE), (0, 0)),
                     engine=self.fixpipe, partition=partition)
        else:
            mem_copy(qk_ub_ch, self.l0c_qk,
                     engine=self.fixpipe, partition=partition)

    def compute_pv_fanout(self, p_l1_ch, pv_ub_ch, partition, slot,
                          actual_n=None):
        # lag3：PV[64,512] = P[64,128] @ K̃[128,512]；此时 P 是 exp(score−running_max)，
        # 尚未除以分母。D 切为 4 个独立的 128 列输出 chunk，各自 init=True。
        # K̃ 复用 QK 的 L1 slot，通过 L0B 的 transpose=True 改变矩阵乘的读取方向。
        # 默认 P 只装入 L0A 一次并供 4 个 chunk 共读；每个 chunk 算完即 fixpipe 到
        # pv_ub 的对应列，全部列写完后才 commit，AIV 随后可累加完整 [32,512] 半块。
        # 最后归还 K̃ 两个半 D 的 credit 与 p_l1 槽，允许上游覆盖。
        p_slot = p_l1_ch
        out = pv_ub_ch.acquire()
        if const_expr(_PV_HALF):
            a_slot = self.l0a_p
            mem_copy(a_slot, p_slot)

            a_view = self.l0a_p
            for g in range_constexpr(2):
                c_slot = self.l0c
                for n in range_constexpr(g * (D_TILES // 2), (g + 1) * (D_TILES // 2)):
                    cube_raw_l1_to_l0b(self.l0b, self.kv_wide,
                                       l1_offset_elems=self._kv_chunk_off(slot, n),
                                       transpose=True, **_KV_L0B_FRACTAL)
                    if const_expr(_DYN_KN):
                        matmul(tile_view(c_slot, (self.tile_cube_m, TILE),
                                         (0, n - g * (D_TILES // 2))),
                               a_view, self.l0b, init=True, unit_flag=actual_n)
                    else:
                        matmul(tile_view(c_slot, (self.tile_cube_m, TILE),
                                         (0, n - g * (D_TILES // 2))),
                               a_view, self.l0b, init=True)
                    if const_expr(_DYN_KN):
                        _mdh.register("pv")
                c_view = self.l0c
                part = make_partition_tiler((self.tile_cube_m, self.tile_d // 2),
                                            (self.tile_cube_m, self.tile_d // 2))
                mem_copy(tile_view(out, (self.tile_vec_m, self.tile_d // 2), (0, g)),
                         c_view, engine=self.fixpipe, partition=part)

        elif const_expr(_PV_WIDE or _L0C_SHARED):
            c_slot = self.l0c
            a_slot = self.l0a_p
            mem_copy(a_slot, p_slot)

            a_view = self.l0a_p
            for n in range(D_TILES):
                cube_raw_l1_to_l0b(self.l0b, self.kv_wide,
                                   l1_offset_elems=self._kv_chunk_off(slot, n),
                                   transpose=True, **_KV_L0B_FRACTAL)
                if const_expr(_DYN_KN):
                    matmul(tile_view(c_slot, (self.tile_cube_m, TILE), (0, n)),
                           a_view, self.l0b, init=True, unit_flag=actual_n)
                else:
                    matmul(tile_view(c_slot, (self.tile_cube_m, TILE), (0, n)),
                           a_view, self.l0b, init=True)
                if const_expr(_DYN_KN):
                    _mdh.register("pv")
            c_view = self.l0c
            part = make_partition_tiler((self.tile_cube_m, self.tile_d),
                                        (self.tile_cube_m, self.tile_d))
            mem_copy(out, c_view, engine=self.fixpipe, partition=part)

        elif const_expr(_PV_P_ONCE):
            a_slot = self.l0a_p
            mem_copy(a_slot, p_slot)

            a_view = self.l0a_p
            for n in range(D_TILES):
                cube_raw_l1_to_l0b(self.l0b, self.kv_wide,
                                   l1_offset_elems=self._kv_chunk_off(slot, n),
                                   transpose=True, **_KV_L0B_FRACTAL)
                if const_expr(_DYN_KN):
                    matmul(self.l0c, a_view, self.l0b, init=True,
                           unit_flag=actual_n)
                else:
                    matmul(self.l0c, a_view, self.l0b, init=True)
                if const_expr(_DYN_KN):
                    _mdh.register("pv")
                col = tile_view(out, (self.tile_vec_m, TILE), (0, n))
                mem_copy(col, self.l0c, engine=self.fixpipe, partition=partition)

        else:
            for n in range(D_TILES):
                mem_copy(self.l0a_p, p_slot)
                cube_raw_l1_to_l0b(self.l0b, self.kv_wide,
                                   l1_offset_elems=self._kv_chunk_off(slot, n),
                                   transpose=True, **_KV_L0B_FRACTAL)
                if const_expr(_DYN_KN):
                    matmul(self.l0c, self.l0a_p, self.l0b, init=True,
                           unit_flag=actual_n)
                else:
                    matmul(self.l0c, self.l0a_p, self.l0b, init=True)
                if const_expr(_DYN_KN):
                    _mdh.register("pv")
                col = tile_view(out, (self.tile_vec_m, TILE), (0, n))
                mem_copy(col, self.l0c, engine=self.fixpipe, partition=partition)
        self._kv_release_r(slot)
        pv_ub_ch.commit(out)

class Vector:
    """AIV 侧：每核负责同一 query 的 32 个头，完成 online softmax 与输出累加。

    对一个头，设旧状态为 (m,l,o)，当前 tile 的缩放后 score 为 s：
      m' = max(m, tile_max)，alpha = exp(m−m')，p_j = exp(s_j−m')；
      l' = alpha*l + Σ_j p_j，o' = alpha*o + BF16(p) @ K̃。
    仅有效前缀参与指数与求和；残块的 tile_max 还可能包含 masked score 的零占位。
    sink 播种 (m=sink,l=1,o=0)，末 tile 才输出 BF16(o/l) 与 log(l)+m。
    sink 是未乘 softmax_scale 的 logit；传 0 仍贡献 exp(0)=1 的分母项。

    max/sum 属于 query（m_seq%2），alpha 属于 KV tile（tick%2）。两套双槽分别防止
    下一 query 的 softmax 覆盖上一 query 收尾所需的分母，以及相邻 tile 覆盖 alpha。
    res_o 只用一块 UB：lag3 按任务顺序消费，上一 query finalize 完才开始下一 query。
    """

    def __init__(self, tile_vec_m, tile_n, tile_d, subblock_idx, sinks_span):
        self.sinks_span = sinks_span
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # max/sum 按 query、exp 按任务轮转；索引均为 %2，各分配两槽。
        self.sm_max_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        self.sm_sum_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        self.sm_exp_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.lse_ub = Channel(MemLoc.UB, shape=(tile_vec_m, 1), dtype=dtypes.float32, depth=1)
        self.out_ub = Channel(MemLoc.UB, shape=(8, tile_d), dtype=dtypes.bfloat16, depth=1)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)

        p_n1_pad = 32 // 2
        self.p_ub = Channel(
            MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.bfloat16, depth=2,
            data_format="nz", n1_pad=p_n1_pad,
        )
        self.sinks_ub = Channel(MemLoc.UB, shape=(1, self.sinks_span), dtype=dtypes.float32, depth=1)

    def _nz_params(self):
        # 从 p_ub 的真实物理 stride 推导落点，包含 Channel 的 NZ padding；
        # 此布局与 vec0 的 NZ17 由不同对象管理，不能直接套用 _NZ_CHUNK。
        s = self.p_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return m0, s_m1, s_m0, s_n1 // n0

    def _softmax_fold_row(self, qk_ch, base, nz_off, max_brc_buf, row,
                          ve_mask, vo_mask, b16, b16_full, block_stride):
        # softmax 第二遍（行内）：exp(s−max) 后把 f32 偶/奇 lane 各 cast bf16（RegLayout
        # ZERO/ONE 分别落偶/奇 b16 lanes），vbitwise_or 合并成 128 连续 bf16 = P，
        # 按 NZ 分形 vstore_strided 存 p_ub。返回 (ve, vo) 供行求和。
        # 指数掩码按偶/奇列数 ceil(actual_n/2)、floor(actual_n/2) 分开，兼容奇数尾。
        # 最后用 b16_full 写满 128 列，将尾部 P=0 一并写入，避免复用槽里的旧 P 残留。
        mx = rr.vload_broadcast(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, dtypes.bfloat16, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, dtypes.bfloat16, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16)
        rr.vstore_strided(self.p_ub, nz_off, merged, b16_full, block_stride=block_stride, repeat_stride=0)
        return ve, vo

    def _pass_a_row(self, qk_ch, scale, sm_max_dst, row, row_stride, VL_T,
                    half0_mask, half1_mask, full_mask, half_only=False):
        # 第一遍：有效 score 乘 scale 并原位回写，同时求用于稳定指数的行 max。
        # 每行分成前/后各 64 列；vmuls 的非活跃 lane 置零，而 max 用 full_mask 归约，
        # 因此残块的零也可能抬高 max（不等价于用 −inf 填尾）。后续指数和 P 仍按
        # actual_n 屏蔽尾列。half_only 只在 actual_n 为编译期整数且 ≤64 时裁掉后半。
        base = row * row_stride
        if const_expr(half_only):
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rmax = rr.vreduce_max(v0, mask=full_mask)
        else:
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            v1 = rr.vmuls(rr.vload(qk_ch, base + VL_T), scale, mask=half1_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rr.vstore(qk_ch, base + VL_T, v1, half1_mask)
            rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
        rr.vstore_first(sm_max_dst, row, rmax)

    @staticmethod
    def _half_only(actual_n, VL_T):
        return isinstance(actual_n, int) and actual_n <= VL_T

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        # 用本 tile 的新 max 合并归一化分母：alpha = exp(old_max−new_max)，
        # new_sum = alpha * old_sum + Σ_j exp(score_j−new_max)。
        # alpha 存入本任务的 sm_exp 槽，供 lag3 将旧的 O 分子缩放到同一个指数基准。
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)

    @jit
    def softmax_rest(self, qk_ch, scale, m_axis_triple: Int64, tile_triple: Int64, actual_n):
        # 合并当前 tile 与已保存的 running 状态；有 sinks 时首 tile 也走此函数。
        # win→cmp 时 m_seq 不变、n 不归零，继续使用同一 max/sum 槽，不做分池 softmax。
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]
        VL_T = 2048 // 32
        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        half_only = self._half_only(actual_n, VL_T)
        with vf(mode="raw"):
            rows = qk_ch.shape[0]
            src_row_stride = qk_ch.stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                if const_expr(half_only):
                    half1_mask = None
                else:
                    half1_mask, _ = rr.update_mask(_clamp_nonneg(actual_n - VL_T), elem_bits=32)
                self._pass_a_row(qk_ch, scale, self.tmp_new_max, row, src_row_stride, VL_T,
                                 half0_mask, half1_mask, full, half_only=half_only)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                b16_full, _ = rr.update_mask(N, elem_bits=16)
                ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
                vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
                b16, _ = rr.update_mask(actual_n, elem_bits=16)
                base = row * src_row_stride
                nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
                ve, vo = self._softmax_fold_row(
                    qk_ch, base, nz_off, self.tmp_new_max, row,
                    ve_mask, vo_mask, b16, b16_full, block_stride
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
                rr.vstore_first(self.tmp_sum, row, rsum)
            rr.vmem_bar("vst_vld")
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def seed_from_sinks(self, sinks_view, m_axis_triple: Int64):
        # sinks 播种（仅第一个 n-tile）：max ← sink[head]、sum ← 1.0（exp(sink−sink)=1），
        # 此后每 tile 走 softmax_rest 的 update 路径 —— sink 只吸收分母质量、无 value 贡献。
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        span = self.sinks_span
        with vf(mode="raw"):
            full, _ = rr.update_mask(2048 // 32, elem_bits=32)
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            for row in tuple(range(self.tile_vec_m)):
                rr.vstore_first(sm_max, row, rr.vload_broadcast(sinks_view, row % span))
                rr.vstore_first(sm_sum, row, one)
            rr.vmem_bar("vst_vld")

    def store_p(self, p_l1_ch, partition):
        # P 从 p_ub 跨核送 p_l1：每 AIV 填自己的 32 头半块，L1 两槽合计 32 KiB。
        piece = partition_view(p_l1_ch, partition, self.subblock_idx)
        mem_copy(piece, self.p_ub)

    @jit
    def init_o(self, pv_ch):
        # 首 tile 的旧分子为 0（sink 无 Value），直接复制 PV，无需读未初始化的 res_o。
        mem_copy(self.res_o, pv_ch)

    @jit
    def update_o(self, pv_ch, exp_idx: Int64):
        # res_o 保存尚未除分母的 FP32 分子：new_o = alpha * old_o + PV。
        # alpha 由同一 tick 的 softmax 产生；PV 中的 P 已转 BF16，矩阵乘累加为 FP32。
        # full_mask 复用于 D=512 的八个 64-lane chunk，D 的整除约束在构造时检查。
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def update_o_last(self, pv_ch, exp_idx: Int64, sum_idx: Int64):
        # 末 tile：累加后在 FP32 域除 sum，随后 finalize 才转 BF16，避免先舍入再归一化。
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=full)
                    o = rr.vdiv(o, sum_b, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def init_o_last(self, pv_ch, sum_idx: Int64):
        # 整个 query 只有一个 KV tile：没有跨 tile 累加，直接用 PV/最终分母初始化 O。
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    cur = rr.vload(pv_ch, off)
                    rr.vstore(self.res_o, off, rr.vdiv(cur, sum_b, mask=full), full)

    @jit
    def _finalize_lse_vf(self, lse_gm, lse_base, sum_idx: Int64):
        # lse = log(sum) + max；独立 Channel 管理 V 写/MTE3 读及槽复用。
        sm_max_buf = self.sm_max_tb[sum_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        vlog(self.tmp_sum, sm_sum_buf)
        vadd(self.lse_ub, self.tmp_sum, sm_max_buf)
        mem_copy(tile_view(lse_gm, (self.tile_vec_m,), (lse_base,)),
                 local_slice(self.lse_ub, (self.tile_vec_m,)))

    def finalize_o(self, o_tile_gm, sum_idx: Int64,
                   lse_gm=None, lse_base=None):
        # init_o_last/update_o_last 已完成除分母；此处计算可选 LSE，并转 BF16 写出。
        # 用独立8行BF16 Channel写出，不与FP32累加器别名。
        split_m = make_partition_tiler(o_tile_gm.shape, (self.tile_m, self.tile_d))
        half = partition_view(o_tile_gm, split_m, self.subblock_idx)

        if const_expr(lse_gm is not None):
            self._finalize_lse_vf(lse_gm, lse_base, sum_idx)

        # 独立8行输出槽：V cast→MTE3 DMA，避免与FP32累加器原地别名。
        for row in range_constexpr(self.tile_vec_m // 8):
            cast(self.out_ub, tile_view(self.res_o, (8, self.tile_d), (row, 0)))
            mem_copy(tile_view(half, (8, self.tile_d), (row, 0)), self.out_ub)

def _validate_reference_contract(*, cmp_topk=0, ori_topk=0,
                                 pa_block_size=64, pa_cmp_block_size=0):
    if int(ori_topk) <= 0:
        raise ValueError("ori_topk must be positive")
    if int(cmp_topk) < 0:
        raise ValueError("cmp_topk must be non-negative")
    for bs in (pa_block_size, pa_cmp_block_size or pa_block_size):
        if not (1 <= int(bs) <= 1024):
            raise ValueError("PA block size must be in 1..1024")

@kernel
class mqsmla_b1_attn_kernel:
    def __init__(self, tile_cube_m, tile_vec_m, tile_n,
                 win_page_stride, cmp_page_stride, n_heads=0,
                 s2_len=0, s2c_len=0,
                 cmp_topk=0, ori_topk=0,
                 pa_block_size=64, pa_cmp_block_size=0, ws_depth=3, blocks=0,
                 batch_size=1, softmax_scale=None):
        self.n_heads = n_heads
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D
        self.s2_len = int(s2_len)

        self._scale = (float(softmax_scale) if softmax_scale is not None
                       else 1.0 / math.sqrt(D))
        self._b_rt = Int64(int(batch_size))
        self._win_page_stride = Int64(int(win_page_stride))
        self._cmp_page_stride = Int64(int(cmp_page_stride))

        self.s2c_len = int(s2c_len)
        self.cmp_topk = int(cmp_topk)
        self.ori_topk = int(ori_topk)
        self.has_cmp = self.cmp_topk > 0
        self.ori_tiles_cap = -(-self.ori_topk // tile_n)
        self.cmp_tiles = 0
        if const_expr(self.has_cmp):
            self.cmp_tiles = -(-self.cmp_topk // tile_n)
        self.qk_ub = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32,
                             depth=2, kind=ChannelKind.CrossCore)
        self.pv_ub = Channel(MemLoc.UB, shape=(tile_vec_m, self.tile_d), dtype=dtypes.float32,
                             depth=1, kind=ChannelKind.CrossCore)
        self.p_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_n), dtype=dtypes.bfloat16,
                            depth=_P_L1_DEPTH, kind=ChannelKind.CrossCore)

        # 三条跨核数据链：QK(AIC→AIV，depth=2)、P(AIV→AIC，depth=2)、PV(AIC→AIV，depth=1)。
        # qk_ub/pv_ub 的 shape 是每 AIV 半块，p_l1 的 shape 是 AIC 接收的完整 64 头块。

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.sinks_span = tile_vec_m

        self.matmul = Matmul(tile_cube_m, tile_vec_m, tile_n)
        self.vector = Vector(tile_vec_m, tile_n, self.tile_d, self.subblock_idx,
                             sinks_span=self.sinks_span)

        self.ws_depth = int(ws_depth)
        self.blocks = int(blocks)
        (self._ws_ready_id,) = _alloc_ws_intra_ids()
        self.kvcache = Kvcache(block_size=pa_block_size)
        self.pa_cmp_block_size = int(pa_cmp_block_size or pa_block_size)
        self._phy_vf_cmp = (self.pa_cmp_block_size & (self.pa_cmp_block_size - 1)) == 0
        self._phy_vf_ori = (int(pa_block_size) & (int(pa_block_size) - 1)) == 0
        _bs = int(self.pa_cmp_block_size)
        self._phy_shift = 0
        while _bs > 1:
            _bs >>= 1
            self._phy_shift += 1
        _bs_ori = int(pa_block_size)
        self._phy_shift_ori = 0
        while _bs_ori > 1:
            _bs_ori >>= 1
            self._phy_shift_ori += 1
        self._phy_k2cap = self.cmp_tiles * tile_n
        self._phy_ori_k2cap = self.ori_tiles_cap * tile_n
        self._phy_max_kcap = max(self._phy_k2cap, self._phy_ori_k2cap)
        self._phy_btcap = -(-self.s2c_len // int(self.pa_cmp_block_size))
        self._phy_btcap_ori = -(-self.s2_len // int(pa_block_size))
        ch_rows = _TILE_ROWS
        self.ch_batch = Channel(MemLoc.UB,
                                shape=(ch_rows,
                                       _ROW_BF16),
                                dtype=dtypes.bfloat16, depth=2)
        self.ch_batch_bytes = self.ch_batch.reinterpret(
            dtypes.uint8,
            shape=(ch_rows, _COMBINE_DIM))
        # BF16 视图读取 scale；uint8 视图同时负责 DMA 写入和 VF 字节读取，保证同步配对。
        # win 数据在 VF 内解释为 FP8，cmp 拆出 FP4 码点；两视图不另占 UB。
        self.n_sub = tile_n // _TILE_ROWS

    # ---- 坐标与有效宽度：m_seq 是核内 query 序号，m_idx 是全局 query token 序号 ----
    # n 是同一 query 上 win→cmp 串接的 tile 序号；tick 则跨 query 连续编号。
    # 每侧 w = topk_length（None 时为 K），tiles = ceil(w/128)，actual_n = min(128,w−128*n)。
    # gather 的有效前缀与 softmax 的有效列数必须同源；切到 cmp 后须先减去 win tile 数。
    # 内部 ori_* 对应公共接口 win_*；*_s2_start 是稀疏索引表列偏移，不是逻辑 token 值。

    def _m_idx(self, m_seq: Int64):
        # metadata 连续区间映射：m_start 在 __call__ 里读一次存 SSA，这里只做单次加法。
        return self._m_start + m_seq

    def _batch_of(self, m_idx: Int64):
        # 在 cu_seqlens_q 的结束前缀中找首个 > m_idx 的位置，得到该 query 的 batch。
        # 固定 8 轮二分按 B≤256 的既定上界编写，更新用 select；相同 SSA query 坐标只在
        # 同一 MLIR block 内复用结果，避免跨控制流 region 引用不可见的 SSA 值。
        blk = ir.InsertionPoint.current.block
        if const_expr(self._bo_key is m_idx and self._bo_blk == blk):
            return self._bo_val
        lo, hi = Int64(0), self._b_rt
        for _ in range(_TND_BSEARCH_ROUNDS):
            mid = (lo + hi) // Int64(2)
            take = _i64(self._cu_q, Int64(0), mid + Int64(1)) <= m_idx
            lo = Int64(dyn_select(take, mid + Int64(1), lo))
            hi = Int64(dyn_select(take, hi, mid))
        self._bo_key, self._bo_val, self._bo_blk = m_idx, lo, blk
        return lo

    def _cmp_tiles_rt(self, m_idx: Int64):
        w = self._csa_valid_w(m_idx)
        return Int64((w + Int64(self.tile_n - 1)) // Int64(self.tile_n))

    def _ori_valid_w(self, m_idx: Int64):
        # win 侧有效宽度：min(K1, win_topk_length[m_idx])。
        w = Int64(self.ori_topk)
        if const_expr(self._ori_topk_length is not None):
            w = _rt_min(Int64(self.ori_topk), _i64(self._ori_topk_length, m_idx))
        return w

    def _cmp_topk_len(self, m_idx: Int64):
        if const_expr(self._cmp_topk_length is not None):
            return _rt_min(Int64(self.cmp_topk), _i64(self._cmp_topk_length, m_idx))
        return Int64(self.cmp_topk)

    def _ori_tiles_rt(self, m_idx: Int64):
        w = self._ori_valid_w(m_idx)
        return Int64((w + Int64(self.tile_n - 1)) // Int64(self.tile_n))

    def _n_end_rt(self, m_idx: Int64):
        # 本 query 的 n-tile 趟数（运行时）：ori_tiles + cmp_tiles。
        ori_tiles = self._ori_tiles_rt(m_idx)
        if const_expr(not self.has_cmp):
            return ori_tiles
        return Int64(ori_tiles + self._cmp_tiles_rt(m_idx))

    def _actual_n(self, m_idx: Int64, n):
        # 根据 n 所在的池返回当前 tile 的有效 KV 列数，供 softmax 的指数/P 掩码使用。
        # 只有被选中的池内坐标有效；dyn_select 未选中的候选宽度不代表实际任务。
        ori_w = self._ori_width(m_idx, n)
        if const_expr(not self.has_cmp):
            return ori_w
        ori_tiles = self._ori_tiles_rt(m_idx)
        c = n - (Int64(ori_tiles) if isinstance(ori_tiles, int) else ori_tiles)
        cmp_w = self._cmp_valid(m_idx, c)
        ori_w = ori_w if not isinstance(ori_w, int) else Int64(ori_w)
        cmp_w = cmp_w if not isinstance(cmp_w, int) else Int64(cmp_w)
        return Int64(dyn_select(n >= (Int64(ori_tiles) if isinstance(ori_tiles, int)
                                      else ori_tiles), cmp_w, ori_w))

    def _ori_width(self, m_idx: Int64, n):
        w = self._ori_valid_w(m_idx)
        return _rt_min(self.tile_n, w - n * Int64(self.tile_n))

    def _csa_valid_w(self, m_idx: Int64):
        return self._cmp_topk_len(m_idx)

    def _cmp_valid(self, m_idx: Int64, c):
        return _rt_min(self.tile_n, self._csa_valid_w(m_idx) - c * Int64(self.tile_n))

    def _csa_phyaddr_vf(self, w: Int64, tiles: int, bt_ub, sp_ub,
                        bs: int, shift: int):
        # block size 为 2 的幂时，用 shift 替代除法：逻辑 idx → block_table[batch,idx>>shift]
        # → physical_row = physical_block * BS + (idx % BS)。block_table 已按 batch 搬入 UB。
        # mask_counter(w) 仅激活前 w 个稀疏位置；idx<0 被钳为 0，不是在此删除该位置。
        # 有效前缀须由调用方保证合法；尾部生成的占位值不能视为有效 KV 地址。
        # 每地址写两个 int32（低字=物理行号，高字=0），每趟写 128 对；kcap 按 128 对齐。
        BS = bs
        SH = shift
        with vf(mode="raw"):
            vzero = rr.vdups(0, dtypes.int32)
            cnt = rr.mask_counter(w)
            for i in tuple(range(tiles)):
                for r in tuple(range(_PHY_S2_PER_LOOP // _PHY_S2_PER_REG)):
                    mk, cnt = rr.update_mask(cnt, elem_bits=32)
                    src = i * _PHY_S2_PER_LOOP + r * _PHY_S2_PER_REG
                    idx = rr.vload(sp_ub, src)
                    idx = rr.vmaxs(idx, 0, mask=mk)
                    blk = rr.vshr(idx, SH, mask=mk)
                    tmp = rr.vmuls(blk, BS, mask=mk)
                    off = rr.vsub(idx, tmp, mask=mk)
                    pblk = rr.vgather(bt_ub, rr.vreinterpret(blk, dtypes.uint32), mask=mk)
                    lo = rr.vmuls(pblk, BS, mask=mk)
                    lo = rr.vadd(lo, off, mask=mk)
                    hi = vzero
                    dst = i * _PHY_OUT_PER_LOOP + r * _PHY_OUT_PER_REG
                    rr.vstore_interleave(self._phy_out_ub, dst, lo, hi)

    @jit
    def _csa_phyaddr_query(self, m_idx: Int64, is_ori: bool):
        # 一条 query：复制 K 个稀疏索引 → VF 生成地址 → 写出本侧对齐后的 kcap 个地址。
        # GM 每行统一 pitch 为 2*max_kcap；win 写 row=m_idx，cmp 写 row=T1+m_idx。
        # 三块UB由Channel管理DMA/VF交接；indices读写使用同一K列视图，
        # 避免完整容量别名与DMA局部视图被推导成不一致的事务。
        if const_expr(is_ori):
            idx_tab = self._ori_idx
            k = self.ori_topk
            kcap = self.ori_tiles_cap * self.tile_n
            tiles = self.ori_tiles_cap
            bt_ub = self._phy_bt_ub
            bs = int(self.kvcache.block_size)
            shift = self._phy_shift_ori
            w_fn = self._ori_valid_w
            phy_off = Int64(0)
        else:
            idx_tab = self._cmp_idx
            k = self.cmp_topk
            kcap = self.cmp_tiles * self.tile_n
            tiles = self.cmp_tiles
            bt_ub = self._phy_bt_ub
            bs = int(self.pa_cmp_block_size)
            shift = self._phy_shift
            w_fn = self._csa_valid_w
            phy_off = self._phy_cmp_row_off
        sp_view = tile_view(self._phy_sp_ub, (1, k), (Int64(0), Int64(0)))
        mem_copy(sp_view,
                 tile_view(idx_tab, (1, k), (m_idx, Int64(0))))
        self._csa_phyaddr_vf(w_fn(m_idx), tiles, bt_ub, sp_view, bs, shift)
        phy_row = m_idx if phy_off is None else m_idx + phy_off
        mem_copy(tile_view(self._phy_gm, (1, 2 * kcap), (phy_row, Int64(0))),
                 tile_view(self._phy_out_ub, (1, 2 * kcap), (Int64(0), Int64(0))))

    @jit
    def _csa_phyaddr_precompute(self, m_count: Int64, is_ori: bool):
        # 主流水之前在 AIV 预计算地址；仅对 block size 为 2 的幂的一侧执行。
        # 每个 MIX block 只处理 metadata 分到的 query 区间；其中 w>0 的 query 按顺序
        # 分给两个 AIV，保证每个地址表行只有一个写者。主流水的 gather 则让两个 AIV
        # 分别读取同一 query 的前/后 64 个 KV，因此地址表可能由另一个 AIV 生产。
        # 外层全 AIV 屏障使 GM 地址表先写完再进入 gather，空闲 block 也必须参与。
        # 非 2 幂的一侧跳过此 pass，在 gather 中用 get_key_offset 做标量除法/取模。
        vec_core_num = Int64(2)
        vec_core_idx = self.subblock_idx
        total_valid = Int64(0)
        for m_seq in range(Int64(0), m_count):
            w0 = self._ori_valid_w(self._m_idx(m_seq)) if const_expr(is_ori) \
                else self._csa_valid_w(self._m_idx(m_seq))
            total_valid = total_valid + Int64(dyn_select(w0 > Int64(0), Int64(1), Int64(0)))
        per_core = total_valid // vec_core_num
        s1_tail = total_valid % vec_core_num
        cur_start = per_core * vec_core_idx + _rt_min(vec_core_idx, s1_tail)
        cur_count = per_core + Int64(dyn_select(vec_core_idx < s1_tail, Int64(1), Int64(0)))
        valid_counter = Int64(0)
        for m_seq in range(Int64(0), m_count):
            m_idx = self._m_idx(m_seq)
            w = (self._ori_valid_w(m_idx) if const_expr(is_ori)
                 else self._csa_valid_w(m_idx))
            if w > Int64(0):
                if valid_counter >= cur_start:
                    if valid_counter < cur_start + cur_count:
                        btcap = self._phy_btcap_ori if const_expr(is_ori) else self._phy_btcap
                        bt_gm = self._pa_bt if const_expr(is_ori) else self._pa_bt_cmp
                        mem_copy(tile_view(self._phy_bt_ub, (1, btcap),
                                           (Int64(0), Int64(0))),
                                 tile_view(bt_gm, (1, btcap),
                                           (self._batch_of(m_idx), Int64(0))))
                        self._csa_phyaddr_query(m_idx, is_ori)
                valid_counter = valid_counter + Int64(1)

    def _split_m(self):
        return make_partition_tiler(
            (self.tile_cube_m, self.tile_n), (self.tile_cube_m, self.tile_n),
        )

    @staticmethod
    def _scale_idx(grp_lane, base: int, m16):
        # scale gather 的索引向量：idx[lane m] = 组基号 base + m//group_size（u16）。
        # ★ u16 运算必须用 b16 粒度掩码：用 b32 的 full_mask 会把移位结果整寄存器清零
        # ⇒ gather 恒取 srow[0] ⇒ 全列小误差（0.018 量级，板证教训）。
        return rr.vadd(grp_lane, rr.vdups(base, dtypes.uint16), mask=m16)

    def _antiquant_vf_fp8_g32(self, kv_in, kv_fp8, out):
        # win：物理字节序 rope64|nope448|scale16 → BF16 计算序 nope448|rope64。
        # 解码链路：一次 unpack 读 64B，将每个 FP8 字节放入一个 32-bit lane；
        # reinterpret 只改位解释，FP8→FP32→BF16 的 cast 才做数值转换。
        # cast 后有效值在偶 BF16 lane；deinterleave(b0, b1) 提取两组有效值，拼成连续 128 列。
        # scale 组号始终按原物理序：nope 三段 128 列分别用组 2..5、6..9、10..13，
        # nope 尾 64 列用组 14..15，rope 64 列用组 0..1。重排输出不能重排错 scale。
        # 每行 load scale 后用 idx=组基号+lane//32 gather，再做一笔 BF16 乘法。
        # srow 的有效 scale 仅在前 16 lane；尾段只存前 64 lane，屏蔽其余 gather 结果。
        # 输入数据偏移以字节计 i*544，scale 偏移以 BF16 元素计 i*272+256。
        SUB = _TILE_ROWS
        with vf(mode="raw"):
            m32, _ = rr.update_mask(_TILE_SIZE, elem_bits=32)
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            m16h, _ = rr.update_mask(_TILE_SIZE, elem_bits=16)
            m8 = rr.full_mask()
            grp_lane = rr.vshr(rr.varange(0, dtypes.uint16), 5, mask=m16)
            nope_off, rope_off = _NOPE_FP8_OFF, 0
            row_bf16, row_fp8 = _ROW_BF16, _COMBINE_DIM

            # 主循环：j=0/1/2 各处理 128 维 NoPE，合计前 384 维；i 遍历子块内 token。
            for j in tuple(range(D_NOPE // _KV_NUM_PER_LOOP)):
                src_j = j * _KV_NUM_PER_LOOP
                dst_j = j * _KV_NUM_PER_LOOP
                idx_j = self._scale_idx(grp_lane, 4 * j + 2, m16)
                for i in range(SUB):
                    s = nope_off + i * row_fp8 + src_j
                    u0 = rr.vload_unpack(kv_fp8, s, unpack_mode="b8_to_b32")
                    u1 = rr.vload_unpack(kv_fp8, s + _TILE_SIZE,
                                         unpack_mode="b8_to_b32")
                    e0 = rr.vreinterpret_lanes(u0, Float8E4M3FN)
                    e1 = rr.vreinterpret_lanes(u1, Float8E4M3FN)
                    f0 = rr.vcast(e0, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                    f1 = rr.vcast(e1, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                    b0 = rr.vcast(f0, dtypes.bfloat16, mask=m32)
                    b1 = rr.vcast(f1, dtypes.bfloat16, mask=m32)
                    raw, _ = rr.vdeinterleave(b0, b1)
                    srow = rr.vload(kv_in, _SCALE_BF16_OFF + i * row_bf16)
                    svec = rr.vgather_reg(srow, idx_j)
                    packed = rr.vmul(raw, svec, mask=m16)
                    self._v0_store(out, i, dst_j, packed, m16)

            # 收尾：NoPE 剩余 64 维与 RoPE 64 维分别解码，写到输出列 384/448。
            # 每段只需一笔 64B load；deinterleave(b, zero) 将有效值压到前 64 个 BF16 lane，
            # 后 64 lane 补零，m16h 仅允许前 64 lane 参与乘法和写出。
            zero = rr.vdups(0.0, dtypes.bfloat16)
            tail_j = D_NOPE // _KV_NUM_PER_LOOP
            src_t = tail_j * _KV_NUM_PER_LOOP
            dst_t = tail_j * _KV_NUM_PER_LOOP
            rope_dst = D_NOPE
            idx_t = self._scale_idx(grp_lane, 14, m16)
            idx_r = self._scale_idx(grp_lane, 0, m16)
            for i in range(SUB):
                ut = rr.vload_unpack(
                    kv_fp8, nope_off + i * row_fp8 + src_t,
                    unpack_mode="b8_to_b32")
                et = rr.vreinterpret_lanes(ut, Float8E4M3FN)
                ft = rr.vcast(et, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                bt = rr.vcast(ft, dtypes.bfloat16, mask=m32)
                raw_t, _ = rr.vdeinterleave(bt, zero)
                srow_t = rr.vload(kv_in, _SCALE_BF16_OFF + i * row_bf16)
                packed_t = rr.vmul(raw_t, rr.vgather_reg(srow_t, idx_t),
                                   mask=m16h)
                self._v0_store(out, i, dst_t, packed_t, m16h)
                ur = rr.vload_unpack(
                    kv_fp8, rope_off + i * row_fp8,
                    unpack_mode="b8_to_b32")
                er = rr.vreinterpret_lanes(ur, Float8E4M3FN)
                fr = rr.vcast(er, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                br = rr.vcast(fr, dtypes.bfloat16, mask=m32)
                raw_r, _ = rr.vdeinterleave(br, zero)
                srow_r = rr.vload(kv_in, _SCALE_BF16_OFF + i * row_bf16)
                packed_r = rr.vmul(raw_r, rr.vgather_reg(srow_r, idx_r),
                                   mask=m16h)
                self._v0_store(out, i, rope_dst, packed_r, m16h)

    def _fp4_lut_reg(self, m16u):
        # 用整数位运算在寄存器中构造 16 项 BF16 LUT：
        # code 0..7 → [0,0.5,1,1.5,2,3,4,6]，code 8..15 为对应负值（含 −0）。
        # 这些 E2M1 数值都能用 BF16 精确表示；这里直接构造 BF16 位模式供拆码后查表。
        # 幅值 t>=2 时 bits=(t<<6)+0x3F00；t==1 时减 0x0040 得 0x3F00（+0.5），
        # t==0 时清零；最后将 code 的 bit3 左移 12 位成为 BF16 符号位。
        # LUT 每 16 lane 重复，实际 code 只索引前 16 项，不需要额外的 UB 常量表。
        ar = rr.varange(0, dtypes.uint16)
        c7 = rr.vdups(7, dtypes.uint16)
        c8 = rr.vdups(8, dtypes.uint16)
        t = rr.vbitwise_and(ar, c7, mask=m16u)
        b = rr.vadd(rr.vshl(t, 6, mask=m16u),
                    rr.vdups(0x3F00, dtypes.uint16), mask=m16u)
        one_m = rr.veq(t, rr.vdups(1, dtypes.uint16), mask=m16u)
        b_minus = rr.vadd(b, rr.vdups(0xFFC0, dtypes.uint16), mask=m16u)
        b1 = rr.vselect(b_minus, b, cond_mask=one_m)
        zero_m = rr.veq(t, rr.vdups(0, dtypes.uint16), mask=m16u)
        b2 = rr.vselect(rr.vdups(0, dtypes.uint16), b1, cond_mask=zero_m)
        sign = rr.vshl(rr.vbitwise_and(ar, c8, mask=m16u), 12, mask=m16u)
        bits = rr.vbitwise_or(b2, sign, mask=m16u)
        return rr.vreinterpret_lanes(bits, dtypes.bfloat16)

    def _fp4_codes(self, u, m32):
        # 将 unpack 后的 64 个字节拆成 128 个 uint16 编码（每项 0..15），保持元素顺序。
        # 每字节低 4 位是偶元素、高 4 位是奇元素，例如 0xA5 → [5, 10]。
        # lo/hi 各占 64 个 u32 lane；reinterpret 成 u16 后编码落在偶 lane。
        # 两次 deinterleave 分别压紧低/高编码，再 interleave 成 [lo0, hi0, lo1, hi1, ...]。
        # 返回的是 LUT 整数索引，尚未转换成浮点值（探针逐值验证见 quant_change_design.md §4.4）。
        m04 = rr.vdups(0x0F, dtypes.uint32)
        lo = rr.vbitwise_and(u, m04, mask=m32)
        hi = rr.vshr(u, 4, mask=m32)
        lo16 = rr.vreinterpret_lanes(lo, dtypes.uint16)
        hi16 = rr.vreinterpret_lanes(hi, dtypes.uint16)
        ce, _ = rr.vdeinterleave(lo16, lo16)
        co, _ = rr.vdeinterleave(hi16, hi16)
        d0, _ = rr.vinterleave(ce, co)
        return d0

    def _antiquant_vf_fp4_g16(self, kv_in, kv_fp8, out):
        # cmp：物理字节序 rope32B|nope224B|scale64B → BF16 计算序 nope448|rope64。
        # 反量化链路：加载打包字节 → 拆出低/高 4-bit 编码 → 查 BF16 LUT → 乘 group16 scale。
        # kv_fp8 只是共用的字节加载视图，实际承载 FP4x2；此路径不调用 FP4 数值 cast。
        # 一笔 unpack 读 64B，拆出 128 个 FP4 值；scale 仍按重排前 rope|nope 的组号取值。
        # GM 每 token 为 320B，但 gather 后 UB 行距仍为 544B（272 个 BF16 元素）；
        # scale 在行内字节偏移 256，即 BF16 偏移 128。packed 是乘 scale 后的 BF16 结果。
        SUB = _TILE_ROWS
        with vf(mode="raw"):
            m32, _ = rr.update_mask(_TILE_SIZE, elem_bits=32)
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            m16h, _ = rr.update_mask(_TILE_SIZE, elem_bits=16)
            lut_reg = self._fp4_lut_reg(m16)
            # lane//16：每 16 个连续元素共用一项 scale，随后加各段的组基号。
            grp_lane = rr.vshr(rr.varange(0, dtypes.uint16), 4, mask=m16)
            row_fp8 = _COMBINE_DIM
            scale_off = SCALE_BF16_OFF_CMP

            # 主循环：p=0/1/2 从字节 32/96/160 各读 64B，处理 NoPE 前 384 维。
            # 每段输出 128 维，分别使用 scale 组 4..11 / 12..19 / 20..27；i 遍历 token。
            for p in tuple(range(D_NOPE // _KV_NUM_PER_LOOP)):
                src_p = D_ROPE // 2 + p * _TILE_SIZE
                dst_p = p * _KV_NUM_PER_LOOP
                idx_p = self._scale_idx(grp_lane, 8 * p + 4, m16)
                for i in range(SUB):
                    u = rr.vload_unpack(kv_fp8, src_p + i * row_fp8,
                                        unpack_mode="b8_to_b32")
                    vals = rr.vgather_reg(lut_reg, self._fp4_codes(u, m32))
                    srow = rr.vload(kv_in, scale_off + i * _ROW_BF16)
                    packed = rr.vmul(vals, rr.vgather_reg(srow, idx_p),
                                     mask=m16)
                    self._v0_store(out, i, dst_p, packed, m16)

            # NoPE 尾 64 维：从字节 224 读 64B，前 32B 是数据，后 32B 是 scale 段的前半部分。
            # 全部字节都会拆码查表，但 m16h 只允许前 64 个值乘 scale[28..31] 并写到列 384。
            tail_p = D_NOPE // _KV_NUM_PER_LOOP
            src_t = D_ROPE // 2 + tail_p * _TILE_SIZE
            dst_t = tail_p * _KV_NUM_PER_LOOP
            idx_t = self._scale_idx(grp_lane, 8 * tail_p + 4, m16)
            for i in range(SUB):
                u = rr.vload_unpack(kv_fp8, src_t + i * row_fp8,
                                    unpack_mode="b8_to_b32")
                vals = rr.vgather_reg(lut_reg, self._fp4_codes(u, m32))
                srow_t = rr.vload(kv_in, scale_off + i * _ROW_BF16)
                packed_t = rr.vmul(vals, rr.vgather_reg(srow_t, idx_t),
                                   mask=m16h)
                self._v0_store(out, i, dst_t, packed_t, m16h)

            # RoPE 64 维：从字节 0 读 64B，后 32B 属于 NoPE，同样由 m16h 屏蔽。
            # 前 64 个值乘 scale[0..3]，写到列 448，完成 nope|rope 重排。
            idx_r = self._scale_idx(grp_lane, 0, m16)
            for i in range(SUB):
                u = rr.vload_unpack(kv_fp8, i * row_fp8,
                                    unpack_mode="b8_to_b32")
                vals = rr.vgather_reg(lut_reg, self._fp4_codes(u, m32))
                srow_r = rr.vload(kv_in, scale_off + i * _ROW_BF16)
                packed_r = rr.vmul(vals, rr.vgather_reg(srow_r, idx_r),
                                   mask=m16h)
                self._v0_store(out, i, D_NOPE, packed_r, m16h)

    def _v0_store(self, out, i, col, val, mask):
        # i 为子块内 token 行，col 为重排后的 D 列，所有偏移均以 BF16 元素计。
        # NZ17 的 UB 地址 = (col//16)*(17*16) + i*16 + col%16；vstore_strided
        # 每写 16 列跨到下一 face，间距 17 个 32B block。第 17 行是 padding。
        # ND 回退路径地址为 i*512+col；两路径的数值与计算序均为 nope448|rope64。
        if const_expr(_NZ17):
            rr.vstore_strided(out, (col // 16) * _NZ_CHUNK + i * _NZ_ROW,
                              val, mask, block_stride=_NZ_PAD_ROWS,
                              repeat_stride=0)
        else:
            rr.vstore(out, i * D + col, val, mask)

    @jit
    def _vec0_body(self, kv_pool, blk_table, blk_size, max_blk, s2_start: Int64, slot: Int64,
                   n_real, idx_tab=None, idx_row=None, batch=None,
                   phy_row_offset=None, scalar_phyaddr=False, is_cmp=False):
        # lag0：128-token tile 分成 8 个 16 行子块；AIV0 负责前 4 个，AIV1 负责后 4 个。
        # 当前每子块依次执行以下操作（不是性能记录中的 prime/replenish 发射方式）：
        #   1. 先对 8 对物理行发成对 DMA，按地址排序后放入 UB 的相邻两行。
        #   2. 再无条件逐行 DMA 全部 16 行：有效行恢复稀疏索引顺序，尾行钳到
        #      最后一个有效索引。因此最终反量化输入由第二轮决定，整块也会重读。
        #      两轮 DMA 的索引都在寻址前钳到有效前缀，避免读取尾部 -1。
        #   3. 双视图 wait 后反量化：win FP8 / cmp FP4，scale 都在同一行内。
        #   4. V→MTE3 就绪事件后，将 BF16 数据写入本核 GM ws 的 tick%3 槽。
        # 两池 UB 行距统一为 544B；cmp 每行只搬入前 320B，反量化仅使用有效数据/scale。
        # 输出 _v0_out0/1 按 lt%2 交替复用，MTE3→V 事件阻止 VF 覆盖尚未搬完的源槽。
        # 只处理 sub_row<n_real 的子块；完全无效的子块不写，ws 保留零初值或前一代内容。
        # 对已处理残块的尾行重读，可避免未初始化值进入 PV 的 0*Value 路径。
        SUB = _TILE_ROWS
        if const_expr(is_cmp):
            _ROW_BYTES = KV_ROW_BYTES_CMP
            page_stride = self._cmp_page_stride
        else:
            _ROW_BYTES = KV_ROW_BYTES_WIN
            page_stride = self._win_page_stride
        n_rt = n_real if not isinstance(n_real, int) else Int64(n_real)
        phy_off = phy_row_offset
        # 显式覆盖从首字节到末页末字节的物理 storage（含页间空洞），不是 token 折叠。
        # mem_copy 保留 Channel 可识别的生产者，raw DMA 与其共用同一字节地址。
        storage_bytes = ((kv_pool.shape[0] - Int64(1)) * page_stride + blk_size * Int64(_ROW_BYTES))
        kv_bytes = kv_pool.view(storage_bytes)

        part = self.n_sub // 2
        t0 = self.subblock_idx * Int64(part)
        for lt in range_constexpr(part):
            sub_b = t0 + Int64(lt)
            sub_row = sub_b * Int64(SUB)
            if sub_row < n_rt:
                buf = self.ch_batch_bytes.acquire()
                bo_sp = Int64(0) if batch is None else batch
                gm_buf, gm_off = extract_buffer(kv_pool, access="read")
                ub_buf, ub_off = extract_buffer(buf, access="write")
                # 两轮 DMA 都钳位到有效前缀，避免尾部 -1 索引参与 PA 寻址。
                last_off = n_rt - Int64(1) - sub_row
                for ii in range_constexpr(0, SUB, 2):
                    pos0 = s2_start + sub_row + _rt_min(Int64(ii), last_off)
                    pos1 = s2_start + sub_row + _rt_min(Int64(ii + 1), last_off)
                    if const_expr(scalar_phyaddr):
                        tok0 = _i64(idx_tab, idx_row, pos0)
                        tok1 = _i64(idx_tab, idx_row, pos1)
                        phys0 = self.kvcache.get_key_offset(
                            blk_table, tok0, bo_idx=bo_sp, block_size=blk_size)
                        phys1 = self.kvcache.get_key_offset(
                            blk_table, tok1, bo_idx=bo_sp, block_size=blk_size)
                    else:
                        phys_row0 = idx_row if phy_off is None else idx_row + phy_off
                        phys_row1 = idx_row if phy_off is None else idx_row + phy_off
                        phys0 = _i64(self._phy_gm, phys_row0, pos0 * Int64(2))
                        phys1 = _i64(self._phy_gm, phys_row1, pos1 * Int64(2))
                    # 预计算表和标量路径均给出逻辑物理行；在 DMA 前转换为字节地址。
                    # PA 页距来自 host Tensor.stride(0)，不能由 block_size 推断。
                    addr0 = (phys0 // blk_size * page_stride
                             + phys0 % blk_size * Int64(_ROW_BYTES))
                    addr1 = (phys1 // blk_size * page_stride
                             + phys1 % blk_size * Int64(_ROW_BYTES))
                    lower = Int64(dyn_select(addr0 < addr1, addr0, addr1))
                    upper = Int64(dyn_select(addr0 < addr1, addr1, addr0))
                    # GM/UB 都用 uint8 载体，offset、burst 和 stride 均以字节计。
                    src_stride = upper - lower
                    gm_start = arith.addi(
                        gm_off, _to_index(lower))
                    ub_start = arith.addi(
                        ub_off, _to_index(Int64(ii) * Int64(KV_ROW_BYTES_WIN)))
                    ascvec.copy_gm2ub(
                        ub_buf, ub_start, gm_buf, gm_start,
                        Int32(2), Int32(_ROW_BYTES),
                        src_stride, Int32(KV_ROW_BYTES_WIN),
                        Int32(0), Int32(0))
                # 当前实现对整块也重读 16 行；残块中 ii>last_off 的行改读最后一个有效 token。
                for ii in range_constexpr(SUB):
                    off = Int64(dyn_select(Int64(ii) <= last_off, Int64(ii), last_off))
                    pos = s2_start + sub_row + off
                    if const_expr(scalar_phyaddr):
                        tok = _i64(idx_tab, idx_row, pos)
                        phys_row = self.kvcache.get_key_offset(
                            blk_table, tok, bo_idx=bo_sp, block_size=blk_size)
                    else:
                        phys_row = _i64(self._phy_gm,
                                        (idx_row if phy_off is None
                                         else idx_row + phy_off),
                                        pos * Int64(2))
                    byte_offset = (phys_row // blk_size * page_stride
                                   + phys_row % blk_size * Int64(_ROW_BYTES))
                    kv_row = kv_bytes[byte_offset:byte_offset + Int64(_ROW_BYTES),].view(
                        Int64(1), _ROW_BYTES)
                    mem_copy(tile_view(buf, (1, _ROW_BYTES), (ii, 0)), kv_row)
                self.ch_batch_bytes.commit(buf)
                # 显式等待两种别名后才让 VF 使用它们；释放也成对，保持同一槽代数。
                rslot = self.ch_batch.wait()
                rslot_bytes = self.ch_batch_bytes.wait()

                v0_out = self._v0_out
                if const_expr(is_cmp):
                    self._antiquant_vf_fp4_g16(rslot, rslot_bytes, v0_out)
                else:
                    self._antiquant_vf_fp8_g32(rslot, rslot_bytes, v0_out)
                self.ch_batch.release(rslot)
                self.ch_batch_bytes.release(rslot_bytes)
                if const_expr(_NZ17):
                    gm_buf, gm_off = extract_buffer(self._ws, access="write")
                    ub_buf, ub_off = extract_buffer(v0_out, access="read")
                    # 去掉 UB 的第 17 行，将 32 个 face 写入 GM 的 [32,128,16] 槽：
                    # GM BF16 offset = slot*128*512 + (col//16)*128*16 + token*16 + col%16。
                    # 每笔 burst 搬 16×16×2=512B；相邻 face 的 GM/UB 起点间距为 4096/544B。
                    tile_base = (slot * Int64(self.tile_n * D)
                                 + sub_b * Int64(SUB * 16))
                    ascvec.copy_ub2gm(
                        gm_buf, arith.addi(gm_off, _to_index(tile_base)),
                        ub_buf, ub_off,
                        Int32(D // 16), Int32(SUB * 16 * 2),
                        Int64(self.tile_n * 16 * 2), Int32(_NZ_CHUNK * 2))
                else:
                    mem_copy(
                        tile_view(self._ws, (SUB, D),
                                  (slot * Int64(self.tile_n // SUB) + sub_b, Int64(0))),
                        v0_out)

    @jit
    def _stage_vec0(self, tick: Int64, m_seq: Int64, n):
        # 按 n 选择 win/cmp 的池、索引表和有效宽度，写同一条 ws 任务环。
        # cmp 的局部 tile 号 c=n−ori_tiles，从索引表 c*128 列开始取数；其地址表行偏移为 T1。
        # 两 AIV 即使本 tile 无子块可处理，也都在末尾发布 ws-ready，供 AIC 成对等待。
        m_idx = self._m_idx(m_seq)
        batch = self._batch_of(m_idx)
        max_blk = self._pa_bt.shape[1]
        max_blk_cmp = self._pa_bt_cmp.shape[1] if self.has_cmp else None
        slot = self._ws_core_slot0 + tick % Int64(self.ws_depth)

        if const_expr(not self.has_cmp):
            self._vec0_body(self._pa_phys, self._pa_bt, self._pa_bs_rt, max_blk,
                            n * Int64(self.tile_n), slot, self._ori_width(m_idx, n),
                            idx_tab=self._ori_idx, idx_row=m_idx, batch=batch,
                            scalar_phyaddr=not self._phy_vf_ori, is_cmp=False)
        else:
            ori_tiles = self._ori_tiles_rt(m_idx)
            ori_bound = Int64(ori_tiles) if isinstance(ori_tiles, int) else ori_tiles
            if n >= ori_bound:
                c = n - ori_bound
                self._vec0_body(self._pa_phys_cmp, self._pa_bt_cmp, self._pa_cmp_bs_rt,
                                max_blk_cmp, c * Int64(self.tile_n), slot,
                                self._cmp_valid(m_idx, c),
                                idx_tab=self._cmp_idx, idx_row=m_idx, batch=batch,
                                phy_row_offset=self._phy_cmp_row_off,
                                scalar_phyaddr=not self._phy_vf_cmp, is_cmp=True)
            else:
                self._vec0_body(self._pa_phys, self._pa_bt, self._pa_bs_rt, max_blk,
                                n * Int64(self.tile_n), slot, self._ori_width(m_idx, n),
                                idx_tab=self._ori_idx, idx_row=m_idx, batch=batch,
                                scalar_phyaddr=not self._phy_vf_ori, is_cmp=False)

        self._ws_ready_arrive()

    @jit
    def _ws_ready_arrive(self):
        # ws-ready 握手（AIV MTE3 写完槽 → arrive）；走 mode=4 指令路径不过 FFTS。
        vec_sync_intra_arrive(PIPE.MTE3, self._ws_ready_id)

    @jit
    def _ws_ready_wait(self):
        # AIC MTE2 分别等 id 和 id+16，确保两 AIV 的 MTE3 都已写完本任务的 ws。
        # ws 无反向归还事件：AIV 开始 vec0(T) 前，上一轮 softmax(T−3) 已等到 QK(T−3)，
        # 从而确认 AIC 已将旧 ws 槽搬入 L1。故深度 3 下可覆盖同槽，尚未完成的 PV
        # 读取的是 L1 K̃；L1 的复用另由 _kv_* credit 保护，不能与 ws 就绪混为一谈。
        cube_sync_intra_wait(PIPE.MTE2, self._ws_ready_id)
        cube_sync_intra_wait(PIPE.MTE2, self._ws_ready_id + _AIV1_ID_OFFSET)

    @jit
    def _stage_load_q_first(self, tick: Int64, m_seq: Int64):
        if tick == Int64(0):
            self._load_q_gen(m_seq)

    @jit
    def _stage_prefetch_q_next(self, m_seq: Int64, is_last, m_count: Int64):
        # 调用点在 bmm1 已释放当前 query 的 Q credit 之后，三个半 D 槽按此顺序轮转。
        if is_last == Int64(1):
            if m_seq + Int64(1) < m_count:
                self._load_q_gen(m_seq + Int64(1))

    @jit
    def _load_q_gen(self, m_seq: Int64):
        m_idx = self._m_idx(m_seq)
        q_tile_gm = tile_view(self._q, (self.tile_cube_m, self.tile_d),
                              (m_idx, 0))
        self.matmul.load_q_wide(q_tile_gm, m_seq)

    @jit
    def _stage_loadqk(self, tick: Int64, m_seq: Int64, n):
        # lag1：tick==0 时装入首份 Q，再等 ws-ready，将本任务 K̃ 从 GM 搬入 L1。
        # 后续 Q 由上一 query 最后一次 bmm1 后预取，不在每个 KV tile 重新装载。
        # ws slot 带 block_idx 私有区基址；L1 slot 是本 AIC 内的 tick%3，无需跨核偏移。
        slot = tick % Int64(_KV_RING)
        self._stage_load_q_first(tick, m_seq)
        self._ws_ready_wait()
        row_tile = self._ws_core_slot0 + tick % Int64(self.ws_depth)
        self.matmul.load_qk(
            self._ws, row_tile, slot,
            rows=(self._actual_n(self._m_idx(m_seq), n)
                  if const_expr(_DYN_TILE) else None))

    def _stage_bmm1_fanout(self, tick: Int64, m_seq: Int64, n, is_last=None,
                           m_count=None):
        # lag2 cube：S = Q@K̃ᵀ（4 chunk）→ fixpipe 跨核送 qk_ub；末尾预取下一代 Q。
        actual_n = (self._actual_n(self._m_idx(m_seq), n)
                    if const_expr(_DYN_KN) else None)
        self.matmul.bmm1_fanout_q(tick % Int64(_KV_RING), m_seq, n, is_last,
                                  actual_n=actual_n)
        self._stage_prefetch_q_next(m_seq, is_last, m_count)
        self.matmul.store_s(self.qk_ub, self._split_m())

    @jit
    def _stage_softmax(self, tick: Int64, m_seq: Int64, n):
        # lag2 AIV：Channel 自动管理 qk_ub 的消费事务 →
        # online softmax（首个 tile 由 sinks 播种）→ store_p 送 p_l1。
        # 槽轮转：max/sum 按 m_seq%2、exp 按 tick%2（承重，见 Vector 类注释）。
        m_idx = self._m_idx(m_seq)
        actual_n = self._actual_n(m_idx, n)
        sinks_bo = self._batch_of(m_idx)
        split_m = self._split_m()
        actual_vec_m = self.tile_vec_m

        m_axis_triple = m_seq % Int64(2)
        tile_triple = tick % Int64(2)

        qk_slot = self.qk_ub
        qk_view = local_slice(qk_slot, (actual_vec_m, self.tile_n), stride=(self.tile_n, 1))
        is_first = (n == Int64(0))
        sinks_coord = self.subblock_idx
        if is_first:
            sk = self.vector.sinks_ub
            mem_copy(sk, tile_view(self._sinks, (1, self.sinks_span),
                                   (sinks_bo, sinks_coord)))
            skr = self.vector.sinks_ub
            self.vector.seed_from_sinks(skr, m_axis_triple)
        self.vector.softmax_rest(qk_view, self._scale, m_axis_triple, tile_triple, actual_n)

        self.vector.store_p(self.p_l1, split_m)

    @jit
    def _stage_pv_fanout(self, tick: Int64, m_seq=None, n=None):
        # lag3 cube：O_tile = P@K̃（P 读 p_l1，K̃ 转置读同一代 L1 环）→ fixpipe 送 pv_ub。
        split_m = make_partition_tiler(
            (self.tile_cube_m, TILE), (self.tile_cube_m, self.tile_n),
        )
        actual_n = (self._actual_n(self._m_idx(m_seq), n)
                    if const_expr(_DYN_KN) else None)
        self.matmul.compute_pv_fanout(self.p_l1, self.pv_ub, split_m,
                                      tick % Int64(_KV_RING),
                                      actual_n=actual_n)

    @jit
    def _stage_update(self, tick: Int64, m_seq: Int64, n, is_last: Int64):
        # lag3 按任务顺序消费 PV；exp_idx=tick%2，sum_idx=m_seq%2，与 softmax 写槽一致。
        # (首,末)=(1,1)：PV/sum；(1,0)：复制 PV；(0,0)：alpha*O+PV；
        # (0,1)：(alpha*O+PV)/sum。只有末 tile 才 finalize 写 O/LSE。
        # 两侧宽度同时为 0 时 n_end=0，无任务到达这里，输出保持预填值；空 query 未支持。
        exp_idx = tick % Int64(2)
        sum_idx = m_seq % Int64(2)

        is_first = (n == Int64(0))

        m_idx = self._m_idx(m_seq)
        o_row_tile = m_idx
        lse_tile_base = (
            o_row_tile * Int64(self.tile_cube_m // self.tile_vec_m)
            + self.subblock_idx) if const_expr(self._lse is not None) else None
        # lse_tile_base 是 32 元素 tile 的编号；对应扁平 LSE 元素偏移 m_idx*64+aiv_id*32。
        if is_first:
            if is_last == Int64(1):
                self.vector.init_o_last(self.pv_ub, sum_idx)
                o_tile_gm = tile_view(self._out, (self.tile_cube_m, self.tile_d),
                                      (o_row_tile, 0))
                self.vector.finalize_o(o_tile_gm, sum_idx,
                                       lse_gm=self._lse, lse_base=lse_tile_base)
            else:
                self.vector.init_o(self.pv_ub)
        else:
            if is_last == Int64(1):
                self.vector.update_o_last(self.pv_ub, exp_idx, sum_idx)
                o_tile_gm = tile_view(self._out, (self.tile_cube_m, self.tile_d),
                                      (o_row_tile, 0))
                self.vector.finalize_o(o_tile_gm, sum_idx,
                                       lse_gm=self._lse, lse_base=lse_tile_base)
            else:
                self.vector.update_o(self.pv_ub, exp_idx)

    def __call__(self, out_gm: Tensor, q_gm: Tensor,
                 sinks_gm: Tensor,
                 pa_phys_gm: Tensor = None, pa_bt_gm: Tensor = None, ws_gm: Tensor = None,
                 pa_phys_cmp_gm: Tensor = None, pa_bt_cmp_gm: Tensor = None,
                 cmp_idx_gm: Tensor = None,
                 ori_idx_gm: Tensor = None,
                 cu_seqlens_q: Tensor = None,
                 softmax_lse: Tensor = None,
                 ori_topk_length: Tensor = None, cmp_topk_length: Tensor = None,
                 metadata_gm: Tensor = None):
        v0_rows = self.blocks * self.ws_depth * self.tile_n
        v0_bytes = v0_rows * D * 2
        phy_rows = q_gm.shape[0] // self.n_heads
        if const_expr(self.has_cmp):
            phy_rows = phy_rows * 2
        phy_addr_gm = ws_gm.view(dtypes.int32)[v0_bytes // 4:,].view(
            phy_rows, 2 * self._phy_max_kcap)
        ws_gm = ws_gm.view(dtypes.bfloat16)[:v0_bytes // 2,].view(v0_rows, D)
        self._q = q_gm
        self._sinks = sinks_gm
        self._out = out_gm
        self._lse = softmax_lse
        self._ori_topk_length = ori_topk_length
        self._cmp_topk_length = cmp_topk_length

        self._bo_key, self._bo_val, self._bo_blk = None, None, None
        self._cu_q = cu_seqlens_q

        self._pa_phys, self._pa_bt, self._ws = pa_phys_gm, pa_bt_gm, ws_gm
        self._ori_idx = ori_idx_gm
        self._pa_bs_rt = Int64(self.kvcache.block_size)
        if const_expr(self.has_cmp):
            self._pa_phys_cmp, self._pa_bt_cmp = pa_phys_cmp_gm, pa_bt_cmp_gm
            self._pa_cmp_bs_rt = Int64(self.pa_cmp_block_size)
            self._cmp_idx = cmp_idx_gm
        self._phy_gm = phy_addr_gm
        self._ws_core_slot0 = self.block_idx * Int64(self.ws_depth)
        SUB = _TILE_ROWS
        _v0_shape = (D // 16, _NZ_PAD_ROWS, 16) if _NZ17 else (SUB, D)
        self._v0_out = Channel(MemLoc.UB, shape=_v0_shape, dtype=dtypes.bfloat16, depth=2)
        self._phyaddr_bar_flag = _alloc_block_sync_flag("P2-a")
        btcap_w = max(self._phy_btcap_ori, self._phy_btcap)
        self._phy_bt_ub = Channel(MemLoc.UB, shape=(1, btcap_w), dtype=dtypes.int32, depth=1)
        self._phy_sp_ub = Channel(MemLoc.UB, shape=(1, self._phy_max_kcap), dtype=dtypes.int32, depth=1)
        self._phy_out_ub = Channel(MemLoc.UB, shape=(1, 2 * self._phy_max_kcap), dtype=dtypes.int32, depth=1)

        if const_expr(self.has_cmp):
            M_TILES = ceil_div(q_gm.shape[0], Int64(self.n_heads))
            self._phy_cmp_row_off = Int64(M_TILES)
        else:
            self._phy_cmp_row_off = Int64(0)

        self._m_start = _i64(metadata_gm, Int64(0), self.block_idx)
        m_count = _i64(metadata_gm, Int64(1), self.block_idx)
        # DelayLineGroup 只延迟任务坐标，不保存张量；深度 4 对应最大 lag=3。
        # g 是本核流水时钟，s.t 是被取出的任务编号；每级必须用 s.t 选择该任务的环槽。
        # 例：g=3 时 vec0(3)、loadQK(2)、QK/softmax(1)、PV/update(0)。
        # m/n/last 随任务一起传递，跨 query 边界也不能拿当前循环坐标替代延迟坐标。
        lag_load = 1
        lag_qk = lag_load
        lag_b = lag_load + 1
        lag_c = lag_load + 2
        DEPTH = lag_c + 1
        DRAIN = lag_c
        dl = DelayLineGroup(DEPTH, 'm', 'n', 't', 'last')
        g = Int64(0)

        # ---- 前置 pass：本核 query 地址预计算 + 全 AIV 会合（同一行写者可能为另一 AIV）----
        _vf_ori = self._phy_vf_ori
        _vf_cmp = self.has_cmp and self._phy_vf_cmp
        if const_expr(_vf_ori):
            self._csa_phyaddr_precompute(m_count, True)
        if const_expr(_vf_cmp):
            self._csa_phyaddr_precompute(m_count, False)
        if const_expr(_vf_ori or _vf_cmp):
            vec_sync_block_arrive(PIPE.MTE3, self._phyaddr_bar_flag, mode=0)
            vec_sync_block_wait(PIPE.S, self._phyaddr_bar_flag, mode=0)
        # ---- warmup / steady：每核 g 从 0 递增，跨 query 及 win→cmp 边界不重置 ----
        # g>=lag 才存在该级任务；每级按 Channel 依赖分别发往 AIC/AIV，Python 调用顺序
        # 不等于所有硬件串行执行。每个 query 仅 n 从 0 重新计数，触发 softmax/O 初始化。
        # m_count=0 的核不发任务，但仍参加地址预计算的跨AIV屏障。
        for m_seq in range(Int64(0), m_count):
            n_end = self._n_end_rt(self._m_idx(m_seq))
            n_end = n_end if not isinstance(n_end, int) else Int64(n_end)
            for n in range(Int64(0), n_end):
                is_last = Int64(1) if n == n_end - Int64(1) else Int64(0)
                dl.push(m=m_seq, n=n, t=g, last=is_last)

                self._stage_vec0(g, m_seq, n)
                if g >= Int64(lag_qk):
                    s0 = dl.tap(lag_qk)
                    self._stage_loadqk(s0.t, s0.m, s0.n)
                if g >= Int64(lag_b):
                    s1 = dl.tap(lag_b)
                    self._stage_bmm1_fanout(s1.t, s1.m, s1.n, s1.last, m_count)
                    self._stage_softmax(s1.t, s1.m, s1.n)
                if g >= Int64(lag_c):
                    s2 = dl.tap(lag_c)
                    self._stage_pv_fanout(s2.t, s2.m, s2.n)
                    self._stage_update(s2.t, s2.m, s2.n, s2.last)

                dl.advance()
                g = g + Int64(1)
        # ---- drain：不再 push 新任务，固定推进 3 拍，让末尾坐标走完所有后级 ----
        # r<lag 排除该级已排空的拍；g>=lag 排除任务数过少时尚未填充的 tap。
        # 即使只发 1 个任务，也会依次补跑 loadQK、QK/softmax、PV/update；空核各守卫均不通过。
        for r in range_constexpr(DRAIN):
            if const_expr(r < lag_qk):
                if g >= Int64(lag_qk):
                    s0 = dl.tap(lag_qk)
                    self._stage_loadqk(s0.t, s0.m, s0.n)
            if const_expr(r < lag_b):
                if g >= Int64(lag_b):
                    s1 = dl.tap(lag_b)
                    self._stage_bmm1_fanout(s1.t, s1.m, s1.n, s1.last, m_count)
                    self._stage_softmax(s1.t, s1.m, s1.n)
            if const_expr(r < lag_c):
                if g >= Int64(lag_c):
                    s2 = dl.tap(lag_c)
                    self._stage_pv_fanout(s2.t, s2.m, s2.n)
                    self._stage_update(s2.t, s2.m, s2.n, s2.last)
            dl.advance()
            g = g + Int64(1)

mqsmla = sys.modules[__name__]

def _phy_pitch(k2_cap):
    # 先将 token 容量向上对齐到 128，再乘每地址的两个 int32；返回的是元素数而非字节数。
    return 2 * (-(-int(k2_cap) // TILE_N) * TILE_N)

def _get_cube_core_num(device=None):
    # 与 arena matmul / flash_attn 使用相同的 torch_npu 属性接口；延迟到 host 调用时查询。
    blocks = int(torch.npu.get_device_properties(device).cube_core_num)
    if blocks <= 0:
        raise ValueError(f"device cube_core_num must be positive, got {blocks}")
    return blocks

def mixed_quant_sparse_flash_mla_metadata(
    win_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_win_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_win_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_win_kv=True,
    has_cmp_kv=True,
):
    """Generate flat core partitions; unused specification fields remain reserved."""
    for name, value in (("win_topk_length", win_topk_length),
                        ("cmp_topk_length", cmp_topk_length)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int32
                or value.dim() != 2 or value.shape[1] != num_heads_kv
                or value.shape[0] <= 0):
            raise ValueError(f"{name} must be an int32 tensor of shape [T1, N2]")
    if (win_topk_length.shape != cmp_topk_length.shape
            or win_topk_length.device != cmp_topk_length.device):
        raise ValueError("topk_length tensors must have the same shape and device")
    if (num_heads_q, num_heads_kv, head_dim, quant_mode) != (N1, N2, D, 1):
        raise ValueError("metadata supports num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1")
    if layout_q != "TND" or layout_kv != "PA_BBND" or not has_win_kv:
        raise ValueError("metadata supports TND / PA_BBND with has_win_kv=True")
    _require_npu(win_topk_length, "win_topk_length")
    device = win_topk_length.device
    if cu_seqlens_q is not None:
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.dim() != 1:
            raise ValueError("cu_seqlens_q must be int32 [B+1]")
        _require_npu(cu_seqlens_q, "cu_seqlens_q")
        values = cu_seqlens_q.detach().cpu()
        if (values.numel() < 2 or int(values[0]) != 0
                or bool((values[1:] < values[:-1]).any())):
            raise ValueError("cu_seqlens_q must be a non-decreasing prefix sum starting at zero")
        rows = int(values[-1])
        if cu_seqlens_q.device != device:
            raise ValueError("cu_seqlens_q must be on the topk_length device")
    elif seqused_q is not None:
        if seqused_q.dtype != torch.int32 or seqused_q.dim() != 1:
            raise ValueError("seqused_q must be int32 [B]")
        _require_npu(seqused_q, "seqused_q")
        values = seqused_q.detach().cpu()
        if bool((values < 0).any()):
            raise ValueError("seqused_q must be non-negative")
        rows = int(values.to(torch.int64).sum())
        if seqused_q.device != device:
            raise ValueError("seqused_q must be on the topk_length device")
    elif batch_size is not None and max_seqlen_q is not None:
        if int(batch_size) <= 0 or int(max_seqlen_q) <= 0:
            raise ValueError("batch_size and max_seqlen_q must be positive")
        rows = int(batch_size) * int(max_seqlen_q)
    else:
        rows = int(win_topk_length.shape[0])
    if rows != int(win_topk_length.shape[0]):
        raise ValueError("query token count must match topk_length.shape[0]")
    # KV lengths, topk fields and KV maximum lengths are reserved; the current
    # load model still uses fixed per-query costs (win 128 + optional cmp 512).
    blocks = _get_cube_core_num(device)
    starts, counts = _balance_from_costs([_query_cost(bool(has_cmp_kv))] * rows, blocks)
    return torch.tensor(starts + counts, dtype=torch.int32, device=device or "npu")

def _require_npu(t, name):
    if t.device.type not in ("npu", "privateuseone"):
        raise ValueError(
            f"{name} must already live on the NPU device: the operator never "
            f"moves or re-lays-out its inputs; got device {t.device}")

class MqsmlaTiling:
    """host 侧解析与挡截（只做校验并派生 kernel 构造参数）。
    """

    def __init__(
        self,
        q,
        *,
        win_kv=None,
        cmp_kv=None,
        win_sparse_indices=None,
        cmp_sparse_indices=None,
        win_block_table=None,
        cmp_block_table=None,
        cu_seqlens_q=None,
        seqused_q=None,
        seqused_win_kv=None,
        seqused_cmp_kv=None,
        win_topk_length=None,
        cmp_topk_length=None,
        sinks=None,
        metadata=None,
        quant_mode,
        softmax_scale=None,
        layout_q="TND",
        layout_kv="PA_BBND",
    ):
        if str(layout_q).upper() != "TND" or str(layout_kv).upper() != "PA_BBND":
            raise ValueError("mixed_quant_sparse_flash_mla supports only "
                             "layout_q='TND' and layout_kv='PA_BBND'")
        if int(quant_mode) != 1:
            raise ValueError("mixed_quant_sparse_flash_mla supports only "
                             "quant_mode=1")

        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        _require_npu(q, "q")
        if q.dtype != torch.bfloat16 or q.dim() != 3 or int(q.shape[-1]) != D:
            raise ValueError("q must be a BF16 TND tensor with shape [T1, N1, 512]")
        rows, n1 = int(q.shape[0]), int(q.shape[1])
        if n1 != N1:
            raise ValueError(f"N1 must be {N1} (the only supported query-head "
                             f"count), got {n1}")
        if rows <= 0:
            raise ValueError("q must contain at least one token")

        if cu_seqlens_q is None:
            raise ValueError(
                "cu_seqlens_q is required for TND varlen and is never synthesized: "
                "pass an int32 [B+1] prefix-sum tensor ending at q.shape[0]")
        if (not isinstance(cu_seqlens_q, torch.Tensor)
                or cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.dim() != 1):
            raise ValueError("cu_seqlens_q must be an int32 tensor of shape (B+1,)")
        _require_npu(cu_seqlens_q, "cu_seqlens_q")
        cu_cpu = cu_seqlens_q.detach().cpu()
        if int(cu_cpu.numel()) < 2:
            raise ValueError("cu_seqlens_q must describe at least one batch")
        if int(cu_cpu[0]) != 0 or int(cu_cpu[-1]) != rows:
            raise ValueError("cu_seqlens_q must be a prefix sum ending at q.shape[0]")
        if bool((cu_cpu[1:] < cu_cpu[:-1]).any()):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        b = int(cu_cpu.numel()) - 1

        if seqused_q is not None:
            if not isinstance(seqused_q, torch.Tensor):
                raise TypeError("seqused_q must be a torch.Tensor when provided")
            used_cpu = seqused_q.detach().cpu().reshape(-1)
            if int(used_cpu.numel()) != b:
                raise ValueError("seqused_q must have one length per cu_seqlens_q batch")
            spans = cu_cpu[1:] - cu_cpu[:-1]
            if bool((used_cpu < 0).any()) or bool((used_cpu > spans).any()):
                raise ValueError("seqused_q must not exceed its cu_seqlens_q span")
            if bool((used_cpu != spans).any()):
                raise ValueError("padding query rows from seqused_q are unsupported")

        for name, seqused in (("seqused_win_kv", seqused_win_kv),
                              ("seqused_cmp_kv", seqused_cmp_kv)):
            if seqused is None:
                continue
            if not isinstance(seqused, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor when provided")
            if seqused.dtype != torch.int32 or seqused.dim() != 1:
                raise ValueError(f"{name} must be a 1-D int32 tensor of shape (B,)")
            if int(seqused.numel()) != b:
                raise ValueError(f"{name} must have one entry per batch element (B={b})")

        has_cmp = (cmp_kv is not None or cmp_sparse_indices is not None
                   or cmp_block_table is not None)
        if has_cmp:
            if (cmp_kv is None or cmp_sparse_indices is None
                    or cmp_block_table is None):
                raise ValueError("ORI_CMP_SPARSE requires cmp_kv, cmp_sparse_indices, "
                                 "cmp_block_table together")
        elif cmp_topk_length is not None:
            raise ValueError("cmp inputs are not valid for ORI_SPARSE")

        pa_win = self._parse_pa_side(win_kv, win_block_table, name="win_kv",
                                     row_bytes=KV_ROW_BYTES_WIN)
        k1 = self._parse_indices(win_sparse_indices, name="win_sparse_indices",
                                 rows=rows)
        len_win_cpu = self._parse_lengths(win_topk_length, name="win_topk_length",
                                          rows=rows, cap=k1)
        self._check_active_indices(win_sparse_indices, len_win_cpu,
                                   name="win_sparse_indices",
                                   s2_cap=pa_win["s2_cap"], k=k1)

        if has_cmp:
            pa_cmp = self._parse_pa_side(cmp_kv, cmp_block_table, name="cmp_kv",
                                         row_bytes=KV_ROW_BYTES_CMP)
            k2 = self._parse_indices(cmp_sparse_indices, name="cmp_sparse_indices",
                                     rows=rows)
            len_cmp_cpu = self._parse_lengths(cmp_topk_length, name="cmp_topk_length",
                                              rows=rows, cap=k2)
            self._check_active_indices(cmp_sparse_indices, len_cmp_cpu,
                                       name="cmp_sparse_indices",
                                       s2_cap=pa_cmp["s2_cap"], k=k2)
        else:
            pa_cmp, k2, len_cmp_cpu = None, 0, None

        if sinks is None:
            raise ValueError(
                "sinks is required and is never synthesized: pass an FP32 [B, N1] "
                "(or [N1]) sink tensor (zeros when no sink is wanted)")
        if not isinstance(sinks, torch.Tensor) or sinks.dtype != torch.float32:
            raise ValueError("sinks must be an FP32 tensor with shape [B, N1] (or [N1])")
        _require_npu(sinks, "sinks")
        sinks_dim = int(sinks.dim())
        if sinks_dim not in (1, 2) or int(sinks.numel()) != b * n1:
            raise ValueError("sinks must be FP32 with shape [B, N1] (or [N1])")
        if sinks_dim == 2 and int(sinks.shape[1]) != n1:
            raise ValueError("sinks must be FP32 with shape [B, N1] (or [N1])")

        if metadata is None:
            raise ValueError(
                "metadata is required and is never synthesized: generate it with "
                "mixed_quant_sparse_flash_mla_metadata(win_topk_length, cmp_topk_length, cu_seqlens_q=..., num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1, has_cmp_kv=...) and pass "
                "the result in (two-stage call)")
        if not isinstance(metadata, torch.Tensor):
            raise TypeError("metadata must be a torch.Tensor")
        self.blocks = _get_cube_core_num(q.device)
        _require_npu(metadata, "metadata")
        if (metadata.dtype != torch.int32
                or tuple(metadata.shape) != (METADATA_ROWS * self.blocks,)):
            raise ValueError(f"metadata must be int32 with shape ({METADATA_ROWS * self.blocks},) "
                             "matching the device Cube core count")

        self.rows = rows
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.bs = pa_win["block_size"]
        self.bs2 = (pa_cmp["block_size"] if pa_cmp is not None else self.bs)
        self.win_page_stride = pa_win["page_stride"]
        self.cmp_page_stride = pa_cmp["page_stride"] if pa_cmp is not None else 0
        self.s2_cap = pa_win["s2_cap"]
        self.s2c_cap = (pa_cmp["s2_cap"] if pa_cmp is not None else 0)
        self.sinks_dim = sinks_dim
        self.scale = (float(softmax_scale) if softmax_scale is not None
                      else 1.0 / math.sqrt(D))
        self.ws_depth = 3
        _validate_reference_contract(
            cmp_topk=self.k2, ori_topk=self.k1,
            pa_block_size=self.bs, pa_cmp_block_size=self.bs2)

    @staticmethod
    def _parse_pa_side(kv, block_table, *, name, row_bytes):
        if not isinstance(kv, torch.Tensor) or not isinstance(block_table, torch.Tensor):
            raise TypeError(f"{name} and {name}_block_table must be torch.Tensor")
        _require_npu(kv, name)
        _require_npu(block_table, f"{name}_block_table")
        if (kv.dtype != torch.uint8 or kv.dim() != 4
                or int(kv.shape[2]) != N2 or int(kv.shape[-1]) != row_bytes):
            side = "FP8 (win)" if name != "cmp_kv" else "FP4 (cmp)"
            raise ValueError(
                f"{name} must be a uint8 [blocknum, blocksize, 1, {row_bytes}] "
                f"tensor ({side} byte view); got shape {tuple(kv.shape)}, "
                f"dtype {kv.dtype}")
        if int(kv.shape[0]) <= 0 or int(kv.shape[1]) <= 0:
            raise ValueError(f"{name} must be non-empty")
        if (block_table.dtype != torch.int32 or block_table.dim() != 2
                or int(block_table.shape[0]) <= 0 or int(block_table.shape[1]) <= 0):
            raise ValueError(f"{name}_block_table must be a non-empty int32 "
                             f"[batch, num_blocks] tensor")
        bs = int(kv.shape[1])
        strides = tuple(kv.stride())
        page_stride = int(kv.stride(0))  # uint8 元素数即字节数；独立于页内 token 数。
        if strides[1:] != (N2 * row_bytes, row_bytes, 1):
            raise ValueError(f"{name} only supports axis-0 non-contiguity; "
                             f"page dimensions must be contiguous, got strides {strides}")
        if page_stride < bs * N2 * row_bytes:
            raise ValueError(f"{name} page stride must be at least {bs * N2 * row_bytes} "
                             "bytes (positive, non-overlapping pages)")
        nlb = int(block_table.shape[1])
        return {"block_size": bs, "s2_cap": nlb * bs, "page_stride": page_stride}

    @staticmethod
    def _parse_indices(idx, *, name, rows):
        if not isinstance(idx, torch.Tensor) or idx.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 torch.Tensor")
        _require_npu(idx, name)
        if (idx.dim() != 3 or int(idx.shape[0]) != rows
                or int(idx.shape[1]) != N2):
            raise ValueError(f"{name} must be int32 with shape [T1, 1, K] "
                             f"(one row per query token, {rows} rows)")
        k = int(idx.shape[2])
        if k <= 0:
            raise ValueError(f"{name} must have at least one index column")
        return k

    @staticmethod
    def _parse_lengths(x, *, name, rows, cap):
        if x is None:
            return None
        if not isinstance(x, torch.Tensor) or x.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 torch.Tensor")
        _require_npu(x, name)
        if x.dim() != 2 or int(x.shape[0]) != rows or int(x.shape[1]) != N2:
            raise ValueError(f"{name} must be int32 with shape [T1, 1]")
        vals = x.detach().cpu()
        if bool((vals < 0).any()) or bool((vals > int(cap)).any()):
            raise ValueError(f"{name} values must lie in [0, {cap}]")
        return vals

    @staticmethod
    def _check_active_indices(idx, len_cpu, *, name, s2_cap, k):
        idx_cpu = idx.detach().cpu()
        if len_cpu is None:
            # 未传 lengths 时全部 K 列有效，不构造默认长度 Tensor。
            vals = idx_cpu
        else:
            active = torch.arange(int(k)) < len_cpu.unsqueeze(-1)
            vals = idx_cpu[active]
        if int(vals.numel()) and (bool((vals < 0).any()) or bool((vals >= int(s2_cap)).any())):
            raise ValueError(f"{name} contains an out-of-range active token index")

def construct_workspace(rows, k1, k2, blocks):
    # 单块字节存储：[BF16 v0 三深环 | int32 物理地址表]，由设备 kernel 切视图。
    # v0 每核 3×128×512×2B；地址区每侧 rows 行、每行 2*align128(max(K1,K2)) 个 int32。
    # 当前使用 zeros；vec0 跳过的完整尾子块首次被读取时为零，后续复用时保留旧槽内容。
    v0_bytes = _ws_rows(blocks) * D * 2
    phy_bytes = (2 if k2 else 1) * int(rows) * _phy_pitch(max(k1, k2)) * 4
    return torch.zeros(v0_bytes + phy_bytes, dtype=torch.uint8, device="npu")

class MqsmlaKernel:
    """kernel 侧入口：把 MqsmlaTiling 的解析结果转成 kernel 构造参数（编译期契约），
    `@jit run` 原样直收接口张量后发射。
    """

    def __init__(self, tiling: MqsmlaTiling):
        self.tiling = tiling
        self.n_heads = N1
        self.blocks = tiling.blocks
        self.tile_cube_m = N1
        self.tile_vec_m = 32
        self.tile_n = TILE_N
        self.s2_len = tiling.s2_cap
        self.s2c_len = tiling.s2c_cap
        self.cmp_topk = tiling.k2
        self.ori_topk = tiling.k1
        self.pa_block_size = tiling.bs
        self.pa_cmp_block_size = tiling.bs2
        self.win_page_stride = tiling.win_page_stride
        self.cmp_page_stride = tiling.cmp_page_stride
        self.ws_depth = tiling.ws_depth
        self.batch_size = tiling.b
        self.softmax_scale = tiling.scale

    def _mk_kernel(self):
        return mqsmla_b1_attn_kernel(
            tile_cube_m=self.tile_cube_m, tile_vec_m=self.tile_vec_m, tile_n=self.tile_n,
            n_heads=self.n_heads, s2_len=self.s2_len,
            s2c_len=self.s2c_len,
            cmp_topk=self.cmp_topk, ori_topk=self.ori_topk,
            pa_block_size=self.pa_block_size,
            pa_cmp_block_size=self.pa_cmp_block_size,
            win_page_stride=self.win_page_stride,
            cmp_page_stride=self.cmp_page_stride,
            ws_depth=self.ws_depth, blocks=self.blocks,
            batch_size=self.batch_size, softmax_scale=self.softmax_scale,
        )

    @jit
    def run(self, q_gm: Tensor, win_kv_gm: Tensor, win_idx_gm: Tensor,
            win_bt_gm: Tensor, win_len_gm: Tensor, out_gm: Tensor,
            sinks_gm: Tensor, cu_seqlens_q: Tensor,
            metadata_gm: Tensor, workspace_gm: Tensor,
            cmp_kv_gm: Tensor = None, cmp_idx_gm: Tensor = None,
            cmp_bt_gm: Tensor = None, cmp_len_gm: Tensor = None,
            lse_gm: Tensor = None):
        # trace 期只变 DSL Tensor 视图，不做 torch 变形或数值转换：
        # Q/out [T1,64,512]→[T1*64,512]，故设备 m-tile 行号就是全局 query token 号；
        # KV 保留四维 uint8 原视图；两侧 page stride 由 tiling 读取并在 kernel init 时设置；
        # indices [T1,1,K]→[T1,K]；lengths [T1,1]→[T1]；cu [B+1]→[1,B+1]；
        # sinks [64]→[1,64]（二维输入保留）；LSE [1,T1,64]→[T1*64]。
        # KV 按字节寻址，避免把页间 padding 当成 token；vec0 搬到 UB 后按 FP8/FP4 解码。
        m_rows = q_gm.shape[0] * q_gm.shape[1]
        q_flat = q_gm.view(m_rows, D)
        out_flat = out_gm.view(m_rows, D)
        win_pool = win_kv_gm
        win_idx_flat = win_idx_gm.view(
            win_idx_gm.shape[0] * win_idx_gm.shape[1], win_idx_gm.shape[2])
        win_len_flat = None
        if const_expr(win_len_gm is not None):
            win_len_flat = win_len_gm.view(
                win_len_gm.shape[0] * win_len_gm.shape[1])
        cu_q = cu_seqlens_q.view(Int64(1), cu_seqlens_q.shape[0])
        if const_expr(self.tiling.sinks_dim == 1):
            sinks_k = sinks_gm.view(Int64(1), self.n_heads)
        else:
            sinks_k = sinks_gm
        lse_flat = None
        if const_expr(lse_gm is not None):
            lse_flat = lse_gm.view(lse_gm.shape[1] * lse_gm.shape[2])
        cmp_pool = None
        cmp_idx_flat = None
        cmp_len_flat = None
        if const_expr(cmp_kv_gm is not None):
            cmp_pool = cmp_kv_gm
            cmp_idx_flat = cmp_idx_gm.view(
                cmp_idx_gm.shape[0] * cmp_idx_gm.shape[1], cmp_idx_gm.shape[2])
            if const_expr(cmp_len_gm is not None):
                cmp_len_flat = cmp_len_gm.view(
                    cmp_len_gm.shape[0] * cmp_len_gm.shape[1])
        op = self._mk_kernel()
        op[self.blocks](out_flat, q_flat, sinks_k,
                        win_pool, win_bt_gm, workspace_gm,
                        cmp_pool, cmp_bt_gm, cmp_idx_flat,
                        win_idx_flat, cu_q, lse_flat, win_len_flat,
                        cmp_len_flat, metadata_gm.view(METADATA_ROWS, self.blocks))

_COMPILED_KERNELS = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _get_compiled_kernel(tiling, tensors):
    """Compile once per static contract; token/page counts stay dynamic.

    Keep every tiling field except rows in the key: K, page stride, block-table
    capacity, batch size and scale are still captured by the device kernel.
    TensorSpec also distinguishes optional tensors and sink/LSE ranks.
    """
    rows = Dim("T1", min=1)
    specs = []
    row_axes = {0: 0, 2: 0, 4: 0, 5: 0, 11: 0, 13: 0, 14: 1}
    dtype_map = {torch.bfloat16: dtypes.bfloat16,
                 torch.float32: dtypes.float32, torch.int32: dtypes.int32,
                 torch.uint8: dtypes.uint8}
    for index, tensor in enumerate(tensors):
        if tensor is None:
            specs.append(None)
            continue
        shape = list(tensor.shape)
        stride = None
        if index in row_axes:
            shape[row_axes[index]] = rows
        elif index in (1, 10):
            shape[0] = Dim("WIN_PAGES" if index == 1 else "CMP_PAGES", min=1)
            # Page padding is part of the static addressing contract.
            stride = tuple(tensor.stride())
        elif index == 9:
            shape[0] = Dim("WORKSPACE_BYTES", min=1)
        specs.append(TensorSpec(tuple(shape), dtype_map[tensor.dtype], stride=stride))
    config = tuple(sorted((name, value) for name, value in vars(tiling).items()
                          if name != "rows"))
    key = (tensors[0].device, config, tuple(specs))
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is None:
            compiled = MqsmlaKernel(tiling).run.compile(*specs)
            # Publish only after compilation succeeds; failures can be retried.
            _COMPILED_KERNELS[key] = compiled
    return compiled


def mixed_quant_sparse_flash_mla(
    q,
    *,
    win_kv=None,
    cmp_kv=None,
    win_sparse_indices=None,
    cmp_sparse_indices=None,
    win_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_win_kv=None,
    seqused_cmp_kv=None,
    win_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
):
    tiling = MqsmlaTiling(
        q,
        win_kv=win_kv,
        cmp_kv=cmp_kv,
        win_sparse_indices=win_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        win_block_table=win_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_win_kv=seqused_win_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        win_topk_length=win_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    for name, tensor, shape, dtype in (
        ("out", out, (tiling.rows, N1, D), torch.bfloat16),
        ("lse", lse, (N2, tiling.rows, N1), torch.float32),
    ):
        if name == "lse" and not return_softmax_lse:
            shape = (0,)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a caller-allocated tensor")
        _require_npu(tensor, name)
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device != q.device or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on {q.device}")
    workspace = construct_workspace(tiling.rows, tiling.k1, tiling.k2, tiling.blocks)
    tensors = (
        q, win_kv, win_sparse_indices, win_block_table, win_topk_length,
        out, sinks, cu_seqlens_q, metadata, workspace,
        cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    compiled = _get_compiled_kernel(tiling, tensors)
    # AOT fixes absent Tensor parameters at compile time and removes them from
    # the runtime signature. Remaining Tensor arguments keep their order.
    compiled(*(tensor for tensor in tensors if tensor is not None))
    torch.npu.synchronize()

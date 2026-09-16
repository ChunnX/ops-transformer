# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AddLora kernel (ops-transformer binary delivery).

y_out[b, y_offset:y_offset+s] = rint_fp16(y[b, ..] + scale * lora[b, ..]), where
lora = (x[b] . A[indices[b], layer_idx]^T) . B[indices[b], layer_idx]^T when
weightA is present (addLoraFlag=1, R <= 128), or lora = x[b] .
B[indices[b], layer_idx]^T (BGMV, R == H1) when weightA is absent
(addLoraFlag=0). Rows with out-of-range indices and columns outside the slice
keep their original y values.

Single ``@pl.jit`` MIX kernel with three accumulated modules:
  * Module 1 ``group_gather_repack`` (section_vector) — routing scan,
    offsets/m_start/inv_map tables, grouped row repack x -> x_g.
  * Module 2 ``grouped_dual_matmul`` (section_cube) — MM1 t = x_g . A^T
    (addLoraFlag=1) -> GM relay -> MM2 lora = t . B^T; BGMV path reads x_g
    directly with K=R=H1 blocking.
  * Module 3 ``scale_accumulate_epilogue`` (section_vector) — per-row
    y += scale * lora with a single FP32 add and one FP16 rounding (ruling 6).

Source of truth: the verified Stage-4 delivery ``custom/addlora/test_addlora.py``
(52/52 PASS); DESIGN.md in that directory is the design authority. The kernel
body below is carried over VERBATIM from it except for the declared ops
adaptations A1-A4; all mechanism-level deviations D1-D15 verified on device in
Stage 4 are preserved unchanged.

ops-transformer adaptations (declared; none touches the math / interface /
support range):

  A1. *kernel symbol rename.* ``addlora_kernel`` -> ``add_lora``. The binary
      build derives ``opInterface.value`` from ``opFile.value`` (= "add_lora"),
      and ``pypto_compile_op`` locates the kernel via ``getattr(module,
      "add_lora")`` — the @pl.jit symbol must carry the op-file name (same
      convention as the dense_lightning_indexer_softmax_lse_v2 sample).
  A2. *signature reordered to the op I/O contract.* Params are the op inputs in
      proto order (y/x/weight_b/indices/weight_a), then the output ``y_out``,
      then ``workspace`` and ``tiling``. The generated entry binds the op's
      runtime arguments in that order and routes the raw workspace through
      ``AscendC::GetUserWorkspace`` (workspace is the second-to-last entry
      param). ``y_out`` is the in-place output bound to y's address by the
      caller (y_out == y, single-writer per row); every original reference to
      the ``y`` pointer in the kernel body now goes through ``y_out`` —
      read-modify-write semantics at the same address, equivalent to the
      verified in-place form.
  A3. *tbl folded into workspace.* The original standalone ``tbl`` tensor
      parameter is removed; the table region lives in the workspace at byte
      offset ``tiling.ws_tbl_off`` (19th TilingData field, filled by the host
      tiling). Kernel-internal base: ``tbl = pl.addptr(workspace,
      tiling.ws_tbl_off)``; table layout/length/mechanics unchanged
      (offsets[W+1] | m_start[W+1] | inv_map[B], 2*(W+1)+B int32 values).
  A4. *no host-side python.* Module imports only pypto_pro (no torch / no test
      code); TilingData filling, parameter checking and workspace sizing moved
      to ``op_host/add_lora_tiling.cpp`` mirroring the verified wrapper math.

Host-side contract (mirrored by op_host/add_lora_tiling.cpp):
  * blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum) = 28 on ascend950PR
    (MIX AIC_1_2 task type; kernel-side AIC get_block_num() == 28 and
    AIV get_block_num() * get_subblock_num() == 56).
  * TilingKey: addLoraFlag = weightA present ? 1 : 0.
  * Workspace user segments (offsets relative to the user workspace base, each
    32B aligned; lora segment 64B aligned per D8):
        x_g    [B, H1] FP16        @ ws_xg_off = 0
        t      [B, R]  FP16        @ ws_t_off = align32(B*H1*2)
        lora_g [B, H2] FP32        @ ws_lora_off = align64(ws_t_off + B*R*2)
        cnt_seg [nseg_cap, W_MAX] i32 @ ws_cnt_off = align32(ws_lora_off + B*H2*4)
        segbase [nseg_cap, W_MAX] i32 @ ws_segbase_off = align32(ws_cnt_off + nseg_cap*W_MAX*4)
        rowpos  [B] i32            @ ws_rowpos_off = align32(ws_segbase_off + nseg_cap*W_MAX*4)
        tbl     2*(W+1)+B i32      @ ws_tbl_off = align32(ws_rowpos_off + B*4)
    with nseg_cap = 2 * blockDim and W_MAX = 32 (SPEC upper bound, not the
    runtime W). Total user workspace = align32(ws_tbl_off + (2*(W+1)+B)*4).

Declared implementation deviations from DESIGN.md (carried over VERBATIM from
the Stage-4 delivery; math / interface / support range unchanged —
mechanism-level only, all evidenced by the on-device probes referenced):

  D1. *flat-transposed weight loads.* The DESIGN §1 transposed views
      (``[H1, R]``/``[R, H2]`` with default order) silently read wrong data on
      this install (probe maxdiff 1209). Implemented as flat row-major views
      ``[W*L*R, H1]`` / ``[W*L*H2, R]`` + ``order=[1, 0]`` loads with the
      (w, layer_idx) byte offset folded into the load row coordinate (probe
      maxdiff 0.0).
  D2. *tbl staged through UB.* V-pipe scalar GM writes (setval) are not
      visible cross-core; the Module-1 tables are built in the ``tbl_ub`` UB
      tile and flushed with one MTE3 ``pl.store`` so the AIC side observes
      them (verified all cases).
  D3. *total task count from tbl.* ``total = tbl_c[2*W+1, 0]`` = m_start[W].
  D4. *asw round-based task dispatch* (``task = core_id + r * core_num`` under
      a ``total > 0`` guard) instead of a strided range, plus the
      pre-initialized SSA phi form for the task-table lookup (Module-1
      verified forms).
  D5. *compact L0 moves.* All L0A/L0B move destinations declare ``compact=1``
      and pre-set their dst window before the move (matmul consumes the same
      window); compact=0/None moves by physical shape ignore set_validshape.
  D5b. *bar_all after bar_fix for the t relay.* Same-core FIX-written GM is
      stale to same-core MTE2 readers with only ``bar_fix``; a full-pipe
      ``bar_all`` drain is required and probe-verified exact.
  D7. (superseded by D8) original 32B-aligned lora base triggered a
      hardware store-loss class at base ≡ 32 (mod 64).
  D8. *lora segment 64B alignment.* With the lora base ≡ 32 (mod 64) and the
      store window covering the physically last row, the trailing 32B (last 8
      FP32 of the last valid row) were not written (26/26 probe grid). The
      lora segment base is promoted to 64B alignment
      (``ws_lora_off = align64(ws_t_off + B*R*2)``), eliminating the class;
      H2 being a multiple of 16 keeps B*H2*4 a multiple of 64B so the
      following segments stay equivalently spaced (chain stays monotonic and
      non-overlapping; total grows by at most 32B).
  D9. *scale operand.* DESIGN §1 S4a names ``vf.muls(reg, scale, preg)`` with
      a runtime float scalar. No kernel-scope float scalar exists on this
      install (scale arrives as TilingData ``scale_bits`` int32 bit pattern),
      and the muls scalar operand has no runtime-Scalar form that could carry
      a bit-restored float. Implemented as the broadcast FP32 register restore
      (``vf.full(scale_bits -> INT32 broadcast)`` -> ``store_align`` UB int32
      tile -> ``load_align(same tile, dtype=DT_FP32)`` byte-reinterpret ->
      broadcast FP32 register) + ``vf.mul`` — identical FP32 multiply
      numerics. The restore runs ONCE per AIV before all tile loops.
  D10. *round-mode spelling.* DESIGN ruling 6 names ``CAST_RINT``; the codegen
      maps it to ``ROUND_N``, which the installed bisheng 15.0.5 does not
      declare (``error: use of undeclared identifier 'ROUND_N'``). The
      compiler's documented nearest-even constant is ``ROUND_R``
      (tcvt_common.hpp); ``CAST_ROUND`` emits it. Bitwise probe (crafted
      midpoints) confirms CAST_ROUND == RNE on this path.
  D11. *saturation realized as an FP32 pre-clamp.* The hardware ``RS_ENABLE``
      does NOT clamp FP32->FP16 overflow (sums of ±80000 produced ±inf, not
      ±65504). Implemented as ``vf.mins/vf.maxs`` clamp to ±65504 in FP32
      before the cast — bit-identical to the golden's
      ``clamp(-65504, 65504).to(fp16)``; ``saturate=ON`` retained on the
      astype (no-op post-clamp).
  D12. *y widening mechanism.* DESIGN §1 S4b widens y with a single FP16 load
      + ``astype(..., layout=CastLayout.ONE)`` for the odd half; on this
      install the ONE-widening returns all zeros (semantics probe). Implemented
      with two ``UNPK_B16``-dist loads (64 elems each at even 16-bit slots) +
      the default ZERO widening -> consecutive FP32 halves; the
      ``interleave`` step is dropped. Same elements, same rounding points.
  D13. *unaligned auxiliary path.* DESIGN ruling 1 specifies a SIMT scalar
      path; SIMT is unavailable (Module 1 round). The tile-level
      gather/scatter alternative is unusable on this install (``vgather2``
      UB reads observe stale tile content deterministically; ``vscatter``
      tile writes never reach the subsequent TSTORE). Implemented as the
      scalar-staged chain: per-row one-dimensional views + scalar GM reads
      staged into a UB FP16 scratch via ``pl.setval`` -> the same ruling-6 VF
      chain on the scratch -> ``getval`` read-back -> scalar GM writes.
      aiv-0 serial (Module 1 topology deviation class). correctness-only.
  D14. *cube-side bar_all before V5* — draining the FIX pipe so the AIV-side
      MTE2 reads of lora_g observe the AIC-written FP32 values (D5b evidence
      class: FIX-written GM was stale to same-core MTE2 readers even after
      bar_fix).
  D15. *inv_map read path.* The aiv-0 same-core "MTE3-written tbl -> V/S-pipe
      Tensor scalar read" (both the aligned main path and the unaligned
      auxiliary path) observed STALE GM content (stale value = the address'
      previous-launch value; cross-launch predictions all hit; failure shapes
      included a real inv0=-1 overwritten by a stale >=0 value and a real
      inv0>=0 skipped by a stale -1). Neither the 5x sync_all(MIX) arrival
      barriers nor a writer-side bar_all() fix it — a read-path coherence
      issue. Fix: the Module-3 inv_map lookup relays through the verified
      path — ``pl.load`` the tbl segment into the tbl_ub UB tile via MTE2 and
      read ``pos = tbl_ub[0, 2*(W+1)+b]`` (UB tile scalar-read form), for BOTH
      paths. Post-fix probe matrix (R1 x3 fresh processes, seal chains,
      invalid-row-0, valid-row-skip, big-GEMM, unaligned offsets) all healed.

Also carried over unchanged: the frozen tile groups / addresses / mutex ids
(DESIGN §2/§3), the 5 unconditional ``sync_all(MIX)`` barriers per side in the
same order, the single-kernel / single-launch structure, and the D8 workspace
layout (with the tbl segment appended per A3).

Historical note: the Stage-4 ``AddLoraTiling`` dataclass was counted as
"17-field" in the module-1 staged docstring (typo; actually 18 pure-int
fields); this delivery adds the 19th field ``ws_tbl_off`` (ops adaptation A3).
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # official layernorm-sample import
from pypto_pro.runtime.tilingkey import TilingKeyField

# ============================================================================
# 模块级常量（DESIGN.md §2.1 — coder 必须逐字使用）
# ============================================================================
TM = 64       # Cube M 块（组行块）：m_start 任务表步长
KL1 = 256     # K 外层窗口（GM→L1 宽读）
KL0 = 64      # K 内层子块（L1→L0A/L0B）
TN = 64       # MM2 N 块（H2 方向）
XC = 2048     # Module 1 行拷贝列窗（FP16 [1, XC] = 4KB）
EC = 2048     # Module 3 epilogue 列窗 — Module 3 使用
# ── 表容量上界（SPEC 约束，编译期常量；DESIGN §1 静态视图约束）──
B_MAX = 4096      # B ∈ [1, 4096]
W_MAX = 32        # W, L ≤ 32
NSEG_MAX = 56     # AIV 数上界 = 2 × 28（block_dim=28，DESIGN §5）
TBL_MAX = 4192    # 2*(W_MAX+1) + B_MAX = 4162 → 32B 对齐取整
H1_MAX_1D = 131072  # 非对齐辅路径一维视图常量 — Module 3 使用

# host 侧 block_dim（DESIGN §5：950PR 平台文件 28 AIC / 56 AIV；AIC:AIV=1:2）
BLOCK_DIM = 28

# tbl staging tile（偏离 D2）：UB 0x04000，[1, TBL_MAX] INT32 = 16.5KB；
# 不与 §3 冻结的 UB tile 重叠（x_row 0x00000 / lora_row 0x01000 / y_row 0x03000
# 均为 Module 1/3 量），mutex [17] 位于 Vector 命名空间冻结段 14–16 之后。
UB_TBL_OFF = 0x04000
UB_TBL_MUTEX = 17

# Module 3 UB tiles（§3 冻结：lora_row 0x01000 / y_row 0x03000；其余为本轮新增，
# 均位于 tbl_ub 段（0x04000..0x08180）之后、32B 对齐、互不重叠；mutex 18–22
# 位于 Vector 命名空间冻结段 14–17 之后，全部 ≤ 31 且互异）
UB_BITS_OFF = 0x08200       # scale_bits [1,64] INT32（D9 位还原中转）
UB_SCALE_OFF = 0x08300      # scale    [1,64] FP32（还原结果，epilogue 逐 chunk 读）
UB_YSCR_OFF = 0x08400       # y 切片中转 [1,128] FP16（D13 非对齐路径 S->V 中转）
UB_OSCR_OFF = 0x08500       # 输出中转 [1,128] FP16（D13 V->S 中转）
UB_LORAUA_OFF = 0x08600     # lora 列窗 [1,128] FP32（D13 非对齐路径对齐装载）
UB_BITS_MUTEX = 18
UB_SCALE_MUTEX = 19
UB_YSCR_MUTEX = 20
UB_LORAUA_MUTEX = 21
UB_OSCR_MUTEX = 22



# ============================================================================
# Module 3 VF 函数（@pl.vector_function，模块级；机制链全部上机验证——见文件头）
# ============================================================================
@pl.vector_function
def restore_scale_vf(scale_bits: pl.DT_INT64, bits_tile, f32_tile):
    """scale 一次性位还原（D9）：full(运行时 int -> INT32 广播) -> store_align
    UB INT32 tile -> load_align(同 tile, dtype=DT_FP32) 字节重解释（load_align
    dtype 参数的文档化用法）-> 全 64 lane 即 [scale]*64，store_align 到 FP32
    tile 供 epilogue 逐 chunk 读取。每 AIV 一次、位于全部 tile 循环之外。"""
    preg_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    preg_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    bits_reg = vf.full(scale_bits, preg_i32, dtype=pl.DT_INT32)
    vf.store_align(bits_tile, bits_reg, preg_i32)
    scale_reg = vf.load_align(bits_tile, 0, dtype=pl.DT_FP32)
    vf.store_align(f32_tile, scale_reg, preg_f32)


@pl.vector_function
def epilogue_aligned_vf(lora_row, y_row, scale_f32_tile, valid_c: pl.DT_INT64):
    """裁定 6 主链（对齐路径，单个 EC=2048 列窗；probe P2 位级验证）：128 元素
    chunk 循环内——lora 2×64 连续 FP32 load；y 2×UNPK_B16 load + 默认 ZERO 宽化
    （D12）得连续 FP32 半区；inc = mul(lora, scale 广播寄存器)（D9，FP32 不落
    FP16）；恰一次 FP32 add；FP32 预钳位 ±65504（D11）；astype(FP16, CAST_ROUND
    = ROUND_R/RNE, SAT_ON, layout=ZERO)×2（D10）+ de_interleave 打包（FA 官方
    模式）；store_align 写回同一 y_row tile（FA max_tile 读后写同类）。尾 lane
    以 preg_a/preg_b（FP32 通路）与 preg16（FP16 通路）各自 update_mask 屏蔽
    （DESIGN §7 / vec-mask-width R01：两通路独立建 mask，不跨宽度复用）。"""
    scale_reg = vf.load_align(scale_f32_tile, 0)
    n_chunks = (valid_c + 127) // 128
    for ch in pl.range(0, n_chunks, 1):
        base = ch * 128
        n = pl.min(128, valid_c - base)
        n_a = pl.min(64, n)
        n_b = n - n_a
        preg_a = vf.update_mask(n_a, dtype=pl.DT_FP32)
        preg_b = vf.update_mask(n_b, dtype=pl.DT_FP32)
        preg16 = vf.update_mask(n, dtype=pl.DT_FP16)
        lora_a = vf.load_align(lora_row, base)
        lora_b = vf.load_align(lora_row, base + 64)
        y_a16 = vf.load_align(y_row, base, dist=pl.LoadDist.UNPK_B16)
        y_b16 = vf.load_align(y_row, base + 64, dist=pl.LoadDist.UNPK_B16)
        y_lo = vf.astype(y_a16, preg_a, dtype=pl.DT_FP32)
        y_hi = vf.astype(y_b16, preg_b, dtype=pl.DT_FP32)
        inc_a = vf.mul(lora_a, scale_reg, preg_a)
        inc_b = vf.mul(lora_b, scale_reg, preg_b)
        sum_a = vf.add(y_lo, inc_a, preg_a)
        sum_b = vf.add(y_hi, inc_b, preg_b)
        sum_a = vf.mins(sum_a, 65504.0, preg_a)
        sum_a = vf.maxs(sum_a, -65504.0, preg_a)
        sum_b = vf.mins(sum_b, 65504.0, preg_b)
        sum_b = vf.maxs(sum_b, -65504.0, preg_b)
        e16 = vf.astype(sum_a, preg_a, dtype=pl.DT_FP16,
                        layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_ROUND,
                        saturate=pl.SaturateMode.ON)
        o16 = vf.astype(sum_b, preg_b, dtype=pl.DT_FP16,
                        layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_ROUND,
                        saturate=pl.SaturateMode.ON)
        packed, _pb = vf.de_interleave(e16, o16)
        vf.store_align(y_row, packed, preg16, base)


@pl.vector_function
def epilogue_unaligned_vf(lora_ua, yscr, oscr, scale_f32_tile,
                          n: pl.DT_INT64):
    """裁定 6 链（非对齐辅路径，D13：y 值已由标量循环 setval 中转进 yscr、结果
    经 oscr 由标量循环写回；计算链与对齐路径逐指令一致，lora 走对齐 lora_ua
    tile 装载）。correctness-only。"""
    scale_reg = vf.load_align(scale_f32_tile, 0)
    n_a = pl.min(64, n)
    n_b = n - n_a
    preg_a = vf.update_mask(n_a, dtype=pl.DT_FP32)
    preg_b = vf.update_mask(n_b, dtype=pl.DT_FP32)
    preg16 = vf.update_mask(n, dtype=pl.DT_FP16)
    lora_a = vf.load_align(lora_ua, 0)
    lora_b = vf.load_align(lora_ua, 64)
    y_a16 = vf.load_align(yscr, 0, dist=pl.LoadDist.UNPK_B16)
    y_b16 = vf.load_align(yscr, 64, dist=pl.LoadDist.UNPK_B16)
    y_lo = vf.astype(y_a16, preg_a, dtype=pl.DT_FP32)
    y_hi = vf.astype(y_b16, preg_b, dtype=pl.DT_FP32)
    inc_a = vf.mul(lora_a, scale_reg, preg_a)
    inc_b = vf.mul(lora_b, scale_reg, preg_b)
    sum_a = vf.add(y_lo, inc_a, preg_a)
    sum_b = vf.add(y_hi, inc_b, preg_b)
    sum_a = vf.mins(sum_a, 65504.0, preg_a)
    sum_a = vf.maxs(sum_a, -65504.0, preg_a)
    sum_b = vf.mins(sum_b, 65504.0, preg_b)
    sum_b = vf.maxs(sum_b, -65504.0, preg_b)
    e16 = vf.astype(sum_a, preg_a, dtype=pl.DT_FP16,
                    layout=pl.CastLayout.ZERO,
                    round_mode=pl.VFRoundMode.CAST_ROUND,
                    saturate=pl.SaturateMode.ON)
    o16 = vf.astype(sum_b, preg_b, dtype=pl.DT_FP16,
                    layout=pl.CastLayout.ZERO,
                    round_mode=pl.VFRoundMode.CAST_ROUND,
                    saturate=pl.SaturateMode.ON)
    packed, _pb = vf.de_interleave(e16, o16)
    vf.store_align(oscr, packed, preg16, 0)


# ============================================================================
# TilingKey（DESIGN.md §1 — SPEC 冻结的唯一字段）
# ============================================================================
class AddLoraTilingKey:
    """唯一字段：addLoraFlag（SPEC 冻结）。Σbits = 1 ≤ 64。

    weightA 存在（非 None）→ 1（双矩阵乘路径，R ≤ 128）；
    weightA 缺省 → 0（BGMV 单矩阵乘路径，R == H1）。
    """

    addLoraFlag = TilingKeyField(bits=1, values=[0, 1])


# ============================================================================
# TilingData（DESIGN.md §1 — 字段顺序即设备侧结构体布局，host/device 同一定义）
# ============================================================================
@dataclass
class AddLoraTiling:
    # ── 规模（host 从 tensor 元数据读取填充）──
    B: int            # batch 行数 ∈ [1, 4096]
    W: int            # 适配器数 ∈ [1, 32]
    L: int            # 层堆叠数 ∈ [1, 32]
    R: int            # LoRA 秩（addLoraFlag=1: ≤128；=0: == H1）
    H1: int           # x 列宽（16 对齐, ≤131072）
    H2: int           # lora 列宽（16 对齐, ≤131072）
    H3: int           # y 列宽（16 对齐, ≤131072）
    # ── 属性（规约后）──
    layer_idx: int    # 0 ≤ layer_idx < L
    scale_bits: int   # scale 的 IEEE-754 binary32 位模式（int32 承载）
    y_offset: int     # 0 ≤ y_offset < H3
    s: int            # y_slice_size 规约值（-1 → H2；0 ≤ s ≤ H2）
    # ── 派生规模 ──
    nseg_cap: int     # cnt_seg/segbase 段表容量 = 2 * block_dim
    # ── workspace 段偏移（字节，host 计算，各段 32B 对齐；x_g 为第一段）──
    ws_xg_off: int    # x_g   [B, H1] FP16
    ws_t_off: int     # t     [B, R]  FP16（仅 addLoraFlag=1 使用）
    ws_lora_off: int  # lora_g[B, H2] FP32
    ws_cnt_off: int   # cnt_seg [nseg_cap, W_MAX] INT32（段计数，行 stride=W_MAX）
    ws_segbase_off: int  # segbase [nseg_cap, W_MAX] INT32（段基址）
    ws_rowpos_off: int   # rowpos [B] INT32（段内 rank）
    ws_tbl_off: int   # tbl 段基址（ops 适配 A3：host 填充；2*(W+1)+B 个 INT32）


# ============================================================================
# Kernel（单 kernel 铁律：唯一 @pl.jit；Module 1..k 在同一 kernel 内累积）
# ============================================================================
@pl.jit(auto_mutex=True, tiling_key=AddLoraTilingKey)
def add_lora(
    y: pl.Ptr[pl.DT_UINT8],
    x: pl.Ptr[pl.DT_UINT8],
    weight_b: pl.Ptr[pl.DT_UINT8],
    indices: pl.Ptr[pl.DT_UINT8],
    weight_a: pl.Ptr[pl.DT_UINT8],
    y_out: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: AddLoraTiling,
):
    """AddLora MIX kernel — staged to Module 3 (full operator:
    group_gather_repack + grouped_dual_matmul +
    scale_accumulate_epilogue). Section structure and the V1..V5
    sync_all(MIX) barrier pair are frozen by DESIGN §4/§6 and stay identical
    in every staged file of this chain. Module 3 lives in section_vector
    after the V5 barrier (aligned multi-AIV main path / unaligned aiv-0
    auxiliary path, deviations D9–D14 in the module docstring).
    """
    # ── 视图构造区（make_tensor/addptr；两 Section 共用的 GM 视图，DESIGN §0 维度契约）──
    B = tiling.B
    W = tiling.W
    L = tiling.L
    R = tiling.R
    H1 = tiling.H1
    H2 = tiling.H2
    H3 = tiling.H3
    # x [B, H1] FP16
    x_v = pl.make_tensor(x, [B, H1], [H1, 1], dtype=pl.DT_FP16)
    # x_g [B, H1] FP16（GM workspace 第一段）
    xg_v = pl.make_tensor(
        pl.addptr(workspace, tiling.ws_xg_off), [B, H1], [H1, 1], dtype=pl.DT_FP16
    )
    # t [B, R] FP16（MM1 产物，GM 中转；偏离 D1：平铺视图）
    t_v = pl.make_tensor(
        pl.addptr(workspace, tiling.ws_t_off), [B, R], [R, 1], dtype=pl.DT_FP16
    )
    # lora_g [B, H2] FP32（MM2 产物）
    lora_v = pl.make_tensor(
        pl.addptr(workspace, tiling.ws_lora_off), [B, H2], [H2, 1], dtype=pl.DT_FP32
    )
    # 权重平铺 2D 视图（偏离 D1：[W·L·R, H1] / [W·L·H2, R] 行主序 + order=[1,0] 转置搬运；
    # (w, layer_idx) 的 addptr 字节偏移折入 load offsets 的运行时行坐标）
    wa_flat = pl.make_tensor(weight_a, [W * L * R, H1], [H1, 1], dtype=pl.DT_FP16)
    wb_flat = pl.make_tensor(weight_b, [W * L * H2, R], [R, 1], dtype=pl.DT_FP16)
    # tbl 基址（ops 适配 A3：原独立 tensor 形参折叠进 workspace 段——基址 =
    # workspace + tiling.ws_tbl_off，由 host 侧 add_lora_tiling.cpp 按 wrapper
    # _ws_offsets 同款数学填充；段长 2*(W+1)+B 个 INT32，布局/机制不变）
    tbl = pl.addptr(workspace, tiling.ws_tbl_off)
    # tbl 静态上界视图（AIC 标量读；MTE3 写入后跨核可见——偏离 D2/probe2 ch1）
    tbl_c = pl.make_tensor(pl.addptr(tbl, 0), [TBL_MAX, 1], [1, 1], dtype=pl.DT_INT32)
    # tbl 一维存储视图（vector 侧 MTE3 整段写；运行时长度 2*(W+1)+B）
    tbl_len = 2 * (W + 1) + B
    tbl_flat_v = pl.make_tensor(tbl, [1, tbl_len], [tbl_len, 1], dtype=pl.DT_INT32)
    # y [B, H3] FP16（Module 3 epilogue 的 in-place 读改写视图，DESIGN §0 维度契约）
    y_v = pl.make_tensor(y_out, [B, H3], [H3, 1], dtype=pl.DT_FP16)

    # ── section_cube：Module 2 grouped_dual_matmul（DESIGN §1；偏离 D1–D6）──
    with pl.section_cube():
        core_id = pl.get_block_idx()    # AIC 上 get_subblock_num()==1（DESIGN §1）
        core_num = pl.get_block_num()   # 运行时实际核数（教程铁律）

        # ── Cube tile groups（DESIGN §2 Tile 属性表 / §3 地址映射表，逐字）──
        # L1(Mat)：xg_wide 0x00000(2×32KB) | waw 0x10000(2×64KB) | t_l1 0x30000 | wbl 0x34000
        xg_wide = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, KL1], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
            ),
            addrs=0x00000, mutex_ids=[0, 1],
        )
        waw = pl.make_tile_group(
            type=pl.TileType(
                shape=[KL1, 128], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
            ),
            addrs=0x10000, mutex_ids=[2, 3],
        )
        t_l1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, 128], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
            ),
            addrs=0x30000, mutex_ids=[9],
        )
        wbl_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, TN], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
            ),
            addrs=0x34000, mutex_ids=[11],
        )
        # L0A(Left)：a_l0a 0x0000(2×8KB) | t_l0a 0x4000（compact=1，偏离 D5）
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, KL0], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1,
            ),
            addrs=0x0000, mutex_ids=[4, 5],
        )
        t_l0a_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, 128], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1,
            ),
            addrs=0x4000, mutex_ids=[10],
        )
        # L0B(Right)：wab 0x0000(2×16KB) | wb_r 0x8000（compact=1，偏离 D5）
        wab = pl.make_tile_group(
            type=pl.TileType(
                shape=[KL0, 128], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right, layout=pl.ZN, compact=1,
            ),
            addrs=0x0000, mutex_ids=[6, 7],
        )
        wb_r_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, TN], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right, layout=pl.ZN, compact=1,
            ),
            addrs=0x8000, mutex_ids=[12],
        )
        # L0C(Acc)：acc_t 0x0000 | acc_l 0x8000
        acc_t_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, 128], valid_shape=[-1, -1], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc, fractal=1024,
            ),
            addrs=0x0000, mutex_ids=[8],
        )
        acc_l_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN], valid_shape=[-1, -1], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc, fractal=1024,
            ),
            addrs=0x8000, mutex_ids=[13],
        )

        # persistent/resident 句柄（DESIGN §2 槽位访问表：单槽 current()，不轮转）
        acc_t_cur = acc_t_g.current()
        acc_l_cur = acc_l_g.current()
        t_l0a_cur = t_l0a_g.current()
        wb_r_cur = wb_r_g.current()

        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V1（空转等待 Vector 0a）
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V2
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V3
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V4（x_g / tbl 全局就绪）

        # ── Module 2：task=(w, m_blk) 跨步分发（偏离 D4：asw round-based 形态）──
        total = tbl_c[2 * W + 1, 0]     # = m_start[W] = total_tasks（偏离 D3）
        k1_tiles_h1 = (H1 + KL1 - 1) // KL1   # MM1 K=H1 窗口数
        k1_tiles_r = (R + KL1 - 1) // KL1     # bgmv MM2' K=R=H1 窗口数
        n2_tiles = (H2 + TN - 1) // TN        # MM2 N=H2 块数

        if total > 0:
            round_cnt = (total + core_num - 1) // core_num
            for r in pl.range(0, round_cnt, 1):
                task = core_id + r * core_num
                if task < total:
                    # 任务查表：线性扫 m_start（DESIGN §1）。偏离 D4：查表结果经
                    # 预初始化变量泄漏到扫描外（Module 1 已验证的 SSA phi 形态），
                    # 任务主体整体位于 if task < total 层级——全部 GM store 保持
                    # 在 t-store 已验证的嵌套深度（更深的分支内 store 在本运行时
                    # 存在部分/完全丢失现象，Module 1 轮偏离 #3 同类）
                    w = 0
                    pos0 = 0
                    m0 = 0
                    Bw = 0
                    for w0 in pl.range(0, W, 1):
                        ms0 = tbl_c[W + 1 + w0, 0]   # m_start[w0]
                        ms1 = tbl_c[W + 2 + w0, 0]   # m_start[w0+1]
                        if (task >= ms0) and (task < ms1):
                            w = w0
                            pos0 = tbl_c[w0, 0]                     # offsets[w0]
                            m0 = (task - ms0) * TM
                            Bw = tbl_c[w0 + 1, 0] - tbl_c[w0, 0]    # 组行数
                    valid_m = pl.min(TM, Bw - m0)           # M 尾（§7）

                    # ============ addLoraFlag == 1（编译期折叠，FA 模式）============
                    if addLoraFlag == 1:
                        # ---- MM1: t = x_g · Aᵀ[w, layer_idx]（K=H1 分块）----
                        base1 = (w * L + tiling.layer_idx) * R   # wa_flat 行基
                        for ko in pl.range(0, k1_tiles_h1, 1):
                            k_off = ko * KL1
                            valid_k1 = pl.min(KL1, H1 - k_off)
                            xg_slot = xg_wide.next()          # L1 [TM, KL1] 双缓冲
                            pl.set_validshape(xg_slot, [valid_m, valid_k1])
                            pl.load(xg_slot, xg_v, [pos0 + m0, k_off])
                            wa_slot = waw.next()              # L1 [KL1, 128] 双缓冲
                            pl.set_validshape(wa_slot, [valid_k1, R])
                            pl.load(wa_slot, wa_flat, [base1, k_off], order=[1, 0])
                            n_sub = (valid_k1 + KL0 - 1) // KL0
                            for ki in pl.range(0, n_sub, 1):
                                sub_off = ki * KL0
                                valid_k0 = pl.min(KL0, valid_k1 - sub_off)
                                gsub = ko * (KL1 // KL0) + ki
                                # asw 子块序列 + compact 纪律（偏离 D5：
                                # dst 窗口移动前预设，matmul 消费同一窗口）
                                a_l = a_l0a.next()
                                pl.set_validshape(a_l, [TM, valid_k0])
                                pl.move(a_l, xg_slot, offset=[0, sub_off])
                                b_r1 = wab.next()
                                pl.set_validshape(b_r1, [valid_k0, R])
                                pl.move(b_r1, wa_slot, offset=[sub_off, 0])
                                if gsub == 0:
                                    pl.matmul(acc_t_cur, a_l, b_r1)
                                else:
                                    pl.matmul_acc(acc_t_cur, acc_t_cur, a_l, b_r1)
                        # L0C FP32 → GM t FP16（FIX 随路转换，裁定 3）
                        pl.set_validshape(acc_t_cur, [valid_m, R])
                        pl.store(t_v, acc_t_cur, [pos0 + m0, 0])
                        pl.system.bar_fix()   # 同核 FIX→MTE2 顺序（bar_fix.md）
                        # 偏离 D5b：同核 FIX 写 GM → MTE2 读在本运行时不可见
                        # （probe15/16：仅 bar_fix 时 MTE2 读回零值）；bar_all
                        # （等待核内全部流水）实测精确通过
                        pl.system.bar_all()
                        # ---- t 中转：GM → L1 → L0A（裁定 4；compact 纪律）----
                        t_l1s = t_l1_g.current()
                        pl.set_validshape(t_l1s, [valid_m, R])
                        pl.load(t_l1s, t_v, [pos0 + m0, 0])
                        pl.set_validshape(t_l0a_cur, [TM, R])
                        pl.move(t_l0a_cur, t_l1s)
                        # ---- MM2: lora = t · Bᵀ[w, layer_idx]（K=R 单块）----
                        base2 = (w * L + tiling.layer_idx) * H2  # wb_flat 行基
                        for n2 in pl.range(0, n2_tiles, 1):
                            n_off = n2 * TN
                            valid_n = pl.min(TN, H2 - n_off)
                            wb_slot = wbl_g.next()            # L1 [128, TN]
                            pl.set_validshape(wb_slot, [R, valid_n])
                            pl.load(wb_slot, wb_flat, [base2 + n_off, 0], order=[1, 0])
                            pl.set_validshape(wb_r_cur, [R, valid_n])
                            pl.move(wb_r_cur, wb_slot)
                            pl.matmul(acc_l_cur, t_l0a_cur, wb_r_cur)
                            pl.set_validshape(acc_l_cur, [valid_m, valid_n])
                            pl.store(lora_v, acc_l_cur, [pos0 + m0, n_off])
                    else:
                        base2 = (w * L + tiling.layer_idx) * H2
                        for n2 in pl.range(0, n2_tiles, 1):
                            n_off = n2 * TN
                            valid_n = pl.min(TN, H2 - n_off)
                            for ko in pl.range(0, k1_tiles_r, 1):
                                k_off = ko * KL1
                                valid_k1 = pl.min(KL1, R - k_off)
                                xg_slot = xg_wide.next()
                                pl.set_validshape(xg_slot, [valid_m, valid_k1])
                                pl.load(xg_slot, xg_v, [pos0 + m0, k_off])
                                wbw_slot = waw.next()          # waw 复用（§2）
                                pl.set_validshape(wbw_slot, [valid_k1, valid_n])
                                pl.load(wbw_slot, wb_flat, [base2 + n_off, k_off],
                                        order=[1, 0])
                                n_sub = (valid_k1 + KL0 - 1) // KL0
                                for ki in pl.range(0, n_sub, 1):
                                    sub_off = ki * KL0
                                    valid_k0 = pl.min(KL0, valid_k1 - sub_off)
                                    gsub = ko * (KL1 // KL0) + ki
                                    a_l = a_l0a.next()
                                    pl.set_validshape(a_l, [TM, valid_k0])
                                    pl.move(a_l, xg_slot, offset=[0, sub_off])
                                    pl.set_validshape(wb_r_cur, [valid_k0, valid_n])
                                    pl.move(wb_r_cur, wbw_slot, offset=[sub_off, 0])
                                    if gsub == 0:
                                        pl.matmul(acc_l_cur, a_l, wb_r_cur)
                                    else:
                                        pl.matmul_acc(acc_l_cur, acc_l_cur, a_l,
                                                      wb_r_cur)
                            pl.set_validshape(acc_l_cur, [valid_m, valid_n])
                            pl.store(lora_v, acc_l_cur, [pos0 + m0, n_off])

        # 偏离 D14（本轮）：FIX 管线全排空后再进 V5——D5b 已证本运行时 FIX 写入
        # 的 GM 对 MTE2 读侧存在陈旧视图（同核 bar_fix 不足，需 bar_all）；Module 3
        # 的 AIV 侧 MTE2 跨核读 lora_g 属同类暴露面，全局屏障前先排空写侧管线
        pl.system.bar_all()
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V5（lora_g 就绪）

    # ── section_vector：Module 1（本 staged 文件承载；偏离 D2：tbl 经 UB 中转）──
    # 执行拓扑偏离声明（Module 1 轮已验证，见 test_addlora_module1.py docstring）：
    # 本安装实测多 AIV 并发执行「标量 GM 读改写」结构时存在逐核不确定的写丢失；
    # 故 Module 1 主体收敛为 aiv 0 单核串行。数学输出与 DESIGN §1 逐位等价。
    with pl.section_vector():
        aiv_id = pl.get_block_idx()                                # 全局 AIV 编号

        # M1 行拷贝窗 tile（DESIGN §2/§3：[1, XC] FP16 UB 单槽，addr 0x00000，mutex [14]）
        x_row_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, XC],
                valid_shape=[-1, -1],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=0x00000,
            mutex_ids=[14],
        )
        x_row = x_row_g.current()

        # tbl 中转 tile（偏离 D2：[1, TBL_MAX] INT32 UB 单槽 @0x04000，mutex [17]）
        tbl_ub_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, TBL_MAX],
                valid_shape=[-1, -1],
                dtype=pl.DT_INT32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=UB_TBL_OFF,
            mutex_ids=[UB_TBL_MUTEX],
        )
        tbl_ub = tbl_ub_g.current()

        # M3 lora 行窗（§3 冻结：[1, EC] FP32 @0x01000，mutex [15]）
        lora_row_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, EC], valid_shape=[-1, -1], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=0x01000,
            mutex_ids=[15],
        )
        lora_row = lora_row_g.current()
        # M3 y 行窗（§3 冻结：[1, EC] FP16 @0x03000，mutex [16]；读-改-写同 tile，
        # FA max_tile 官方读后写同类）
        y_row_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, EC], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=0x03000,
            mutex_ids=[16],
        )
        y_row = y_row_g.current()
        # M3 新增 tile（D9/D13；地址/mutex 见常量区注释，均不与冻结 tile 重叠）
        bits_g3 = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 64], valid_shape=[-1, -1], dtype=pl.DT_INT32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=UB_BITS_OFF, mutex_ids=[UB_BITS_MUTEX],
        )
        scale_f32_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 64], valid_shape=[-1, -1], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=UB_SCALE_OFF, mutex_ids=[UB_SCALE_MUTEX],
        )
        yscr_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 128], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=UB_YSCR_OFF, mutex_ids=[UB_YSCR_MUTEX],
        )
        oscr_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 128], valid_shape=[-1, -1], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=UB_OSCR_OFF, mutex_ids=[UB_OSCR_MUTEX],
        )
        lora_ua_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 128], valid_shape=[-1, -1], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=UB_LORAUA_OFF, mutex_ids=[UB_LORAUA_MUTEX],
        )
        yscr = yscr_g.current()
        oscr = oscr_g.current()
        lora_ua = lora_ua_g.current()

        if aiv_id == 0:
            # 静态上界视图（DESIGN §1「静态视图约束」；一维化 flat 视图，
            # 标量访问保持 FA work_ranges 的 [runtime, 0] 下标形态）
            idx_v = pl.make_tensor(indices, [B_MAX, 1], [1, 1], dtype=pl.DT_INT32)
            rowpos_v = pl.make_tensor(
                pl.addptr(workspace, tiling.ws_rowpos_off),
                [B_MAX, 1], [1, 1], dtype=pl.DT_INT32,
            )
            cntseg_v = pl.make_tensor(
                pl.addptr(workspace, tiling.ws_cnt_off),
                [W_MAX, 1], [1, 1], dtype=pl.DT_INT32,
            )

            # Pass 1: 路由扫描（全 B 行；路由判定先于任何索引使用，越界行
            # rowpos = -1；组计数器驻留 cnt_seg 段首行——本核独占）
            for w0 in pl.range(0, W, 1):
                pl.setval(cntseg_v, w0, 0)
            for b in pl.range(0, B, 1):
                w = idx_v[b, 0]
                if (w >= 0) and (w < W):
                    r = cntseg_v[w, 0]
                    pl.setval(rowpos_v, b, r)
                    pl.setval(cntseg_v, w, r + 1)
                else:
                    pl.setval(rowpos_v, b, -1)

            # Pass 2: 前缀和（单写者：offsets[w+1]、m_start[w+1]；偏离 D2：写 UB tile）
            pl.setval(tbl_ub, 0, 0)          # offsets[0] = 0
            pl.setval(tbl_ub, W + 1, 0)      # m_start[0] = 0
            off = 0
            ms = 0
            for w0 in pl.range(0, W, 1):
                base = cntseg_v[w0, 0]
                pl.setval(tbl_ub, w0 + 1, off + base)             # offsets[w0+1]
                ntask = (base + TM - 1) // TM
                pl.setval(tbl_ub, W + 2 + w0, ms + ntask)         # m_start[w0+1]
                pl.setval(cntseg_v, w0, off)      # offsets[w0] 本地副本
                off = off + base
                ms = ms + ntask

            # Pass 3a: 逆映射（pos = offsets[w] + rowpos[b]；越界行 -1）——
            # 写入 rowpos 段覆盖原 rank（rowpos 为 Module 1 内部量，下游不读）
            for b in pl.range(0, B, 1):
                w = idx_v[b, 0]
                r = rowpos_v[b, 0]
                if (w >= 0) and (w < W) and (r >= 0):
                    pl.setval(rowpos_v, b, cntseg_v[w, 0] + r)
                else:
                    pl.setval(rowpos_v, b, -1)

            # Pass 3b: inv_map 无条件搬运（rowpos → tbl_ub；偏离 D2）
            inv_base = 2 * (W + 1)
            for b in pl.range(0, B, 1):
                pl.setval(tbl_ub, inv_base + b, rowpos_v[b, 0])

            # Pass 3c: 行重排（tile 搬运；列尾块 valid_shape 收窄（DESIGN §7））
            for b in pl.range(0, B, 1):
                pos = rowpos_v[b, 0]
                if pos >= 0:
                    for xc in pl.range(0, H1, XC):
                        valid_xc = pl.min(XC, H1 - xc)
                        pl.set_validshape(x_row, [1, valid_xc])
                        pl.load(x_row, x_v, [b, xc])     # GM → UB [1, XC] FP16
                        pl.store(xg_v, x_row, [pos, xc])  # UB → GM x_g（MTE3）

            # tbl 整段 MTE3 写出（偏离 D2：单次 pl.store，AIC 侧可见）
            pl.set_validshape(tbl_ub, [1, tbl_len])
            pl.store(tbl_flat_v, tbl_ub, [0, 0])

        # ── V1–V5 屏障序列（DESIGN §4/§6 冻结结构：两侧各 5 次、无条件、同序）──
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V1
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V2
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V3
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V4：x_g / tbl 全局就绪
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # V5：等待 lora（Module 2）

        # ══════════ Module 3：scale_accumulate_epilogue（DESIGN §1 裁定 6）══════════
        # scale 一次性位还原（D9；每 AIV 一次，先于全部 tile 循环）
        restore_scale_vf(tiling.scale_bits, bits_g3.current(),
                         scale_f32_g.current())
        s = tiling.s                                 # y_slice_size 规约值
        p = tiling.y_offset - (tiling.y_offset // 16) * 16   # y_offset % 16
        aiv_num = pl.get_block_num() * pl.get_subblock_num()  # §4/§5 全 AIV 分工
        rows_per = (B + aiv_num - 1) // aiv_num
        seg_start = aiv_id * rows_per
        # ── 对齐主路径（y_offset % 16 == 0）：AIV 连续行段（DESIGN §5）──
        if (p == 0) and (s > 0) and (seg_start < B):
            # 偏离 D15（m3 修复轮）：inv_map 改经 MTE2 装载到 UB tile 后标量读。
            # 根因（verifier module-check FAIL + 探针 5/5 复现）：aiv 0 同核
            # 「MTE3 写 tbl → V/S pipe Tensor 标量读」读到该 GM 地址陈旧内容
            # （陈旧值=上一轮 launch 的 inv_map[0]，跨 launch 预测逐一命中；
            # 新页=0）。写侧 bar_all() 全流水排空已探针证伪（R1 仍 60/9472
            # 超差）——5×sync_all(MIX) 核到达屏障与流水排空均不排序该可见性，
            # 属读通路一致性问题。MTE2 搬运走 L2（Module 2 AIC 标量读已验证
            # 的同源通路，D2/probe2 ch1 实证），UB tile 标量读（D13 已验证
            # tile[0, i] 形态）不经过陈旧读窗口。修复后探针矩阵全绿。
            pl.set_validshape(tbl_ub, [1, tbl_len])
            pl.load(tbl_ub, tbl_flat_v, [0, 0])
            seg_len = pl.min(rows_per, B - seg_start)
            for b in pl.range(seg_start, seg_start + seg_len, 1):
                pos = tbl_ub[0, 2 * (W + 1) + b]     # inv_map（UB tile 标量读）
                if pos >= 0:                          # 越界行跳过（y 保持原值）
                    for c in pl.range(0, s, EC):      # EC=2048 列窗（§7 尾块收窄）
                        valid_c = pl.min(EC, s - c)
                        pl.set_validshape(lora_row, [1, valid_c])
                        pl.load(lora_row, lora_v, [pos, c])
                        pl.set_validshape(y_row, [1, valid_c])
                        pl.load(y_row, y_v, [b, tiling.y_offset + c])
                        epilogue_aligned_vf(lora_row, y_row,
                                            scale_f32_g.current(), valid_c)
                        pl.set_validshape(y_row, [1, valid_c])
                        pl.store(y_v, y_row, [b, tiling.y_offset + c])
        # ── 非对齐辅路径（y_offset % 16 != 0，correctness-only）：D13 标量中转链，
        #    aiv 0 串行（Module 1 拓扑偏离类：多 AIV 并发标量 GM 写不可靠）──
        if (p != 0) and (s > 0) and (aiv_id == 0):
            # 偏离 D15：非对齐辅路径同暴露面（aiv 0 同核标量读 tbl），
            # 与对齐主路径同款 MTE2 装载 + UB tile 标量读
            pl.set_validshape(tbl_ub, [1, tbl_len])
            pl.load(tbl_ub, tbl_flat_v, [0, 0])
            for b in pl.range(0, B, 1):
                pos = tbl_ub[0, 2 * (W + 1) + b]
                if pos >= 0:
                    # 每行一维视图（DESIGN §1 定稿段：addptr 运行时行基 + 静态上界）
                    row_y = pl.make_tensor(
                        pl.addptr(y_out, (b * H3) * 2),
                        [1, H1_MAX_1D], [1, 1], dtype=pl.DT_FP16,
                    )
                    n_chunks = (s + 127) // 128
                    for ch in pl.range(0, n_chunks, 1):
                        base = ch * 128
                        n = pl.min(128, s - base)
                        # 1) 标量读 y 切片 -> UB 中转 tile（S->V 边已探针验证）
                        for i in pl.range(0, n, 1):
                            yval = row_y[0, tiling.y_offset + base + i]
                            pl.setval(yscr, i, yval)
                        # 2) lora 对齐装载（[pos, base] 128 列窗）
                        pl.set_validshape(lora_ua, [1, n])
                        pl.load(lora_ua, lora_v, [pos, base])
                        # 3) 裁定 6 链（与主路径逐指令一致）
                        epilogue_unaligned_vf(lora_ua, yscr, oscr,
                                              scale_f32_g.current(), n)
                        # 4) 标量写回非对齐 GM 列（V->S 边已探针验证）
                        for i in pl.range(0, n, 1):
                            oval = oscr[0, i]
                            row_y[0, tiling.y_offset + base + i] = oval

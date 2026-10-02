// sa2_attention_loop.h — SageAttention2 注意力流水线（AIC Cube 引擎 + AIV Vector 引擎）。
#if !defined(__AICORE__) && !defined(__CCE_KT_TEST__)
#error "op_kernel/ is strictly reserved for Ascend NPU device compilation (__AICORE__ or __CCE_KT_TEST__)! Host simulation must use sim/sa2_sim_engine.h instead."
#endif

#ifndef SA2_ATTENTION_LOOP_H_
#define SA2_ATTENTION_LOOP_H_

#include <cstdint>
#include <cstring>

#include "sa2_spec.h"
#include "sa2_planner.h"
#include "sa2_ub_plan.h"
#include "sa2_events.h"

#if __has_include("kernel_operator.h")
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "tla/tensor.hpp"
// catlass.hpp 只聚含 detail/*；AIC 轨用到的 TileCopyTla / TileMmadTla 需显式引入，
// 且必须包含 ascend950/ 特化件（主模板只有 static_assert 哨兵，特化全在 arch 子目录）：
// L1→L0A（zN→zN）、L1→L0B（zN→nZ 与 MX scale 三参档）、L0C→UB（fixpipe 封装参考件）。
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#elif __has_include("sa2_ascendc_hoststub.h")
#include "sa2_ascendc_hoststub.h"
#endif

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

namespace sa2 {

// =============================================================================
// M2b-0 VF 计算核上下文（设备/宿主双轨共用；__ubuf__ 在宿主桩轨展开为空）。
// 共合 scratch 布局（kScratchBase 起 1344B；函数间不交错，同流串行复用）：
//   softmax：+0/+256 = alpha(parity1，GetStateAlpha 契约)、
//            +512 = qh f32 展开行(256B；AIV 后端禁标量位强转，half→f32 展开由 wrapper 完成，
//                    VF 内只允许「直接 float 装载喂 Muls」——迭代 2 探针 p33e/p33g 实测)、
//            +768 = kScaleStage0 (256B, ping 槽，KScaleBase(0))、
//            +1024 = kScaleStage1 (256B, pong 槽，KScaleBase(1))、
//            +1280 = 归约标量提取槽(16B)
//   finish ：+0 = FP32 暂存行(256B)、+256 = vMean 行(512B)、+768 = LSE m 装配(GetStateMStage)
struct Sa2VfSoftmaxArgs {
    __ubuf__ float *rowStat;
    __ubuf__ float *denomSlot;
    __ubuf__ float *oAcc;
    __ubuf__ float *alphaSlot;
    __ubuf__ uint8_t *pSlotUb;   // P 槽 ND 基址（行内偏移 = r * kKvStep）
    __ubuf__ half *scoreWarp;    // per_warp：deqScore 行基址（per_thread 传 nullptr）
    __ubuf__ float *scoreThread; // per_thread：score 行基址（per_warp 传 nullptr）
    __ubuf__ half *kScaleRow;    // per_warp：kh 行 MTE 暂存
    __ubuf__ half *qScaleRow;    // qh half 位型源（64 half，wrapper 自 L1 标量暂存）
    __ubuf__ float *qhF32;       // qh f32 展开行（wrapper 由 qScaleRow 位运算展开；per_thread 未用）
    __ubuf__ float *scalarSlot;  // 归约标量提取槽（[0]=max、[1]=sum、[2]=alpha 位型提取）
    int64_t validRows;
    int64_t m0;
    int64_t kvTokenStart;
    int64_t sk;
    int64_t dp;
    float lnPDomain;
    bool isThreadGran;
};

struct Sa2VfFoldArgs {
    // 非 const：MicroAPI LoadAlign 的 T 模板形参拒绝 const 限定（dav_3510 datacopy 实测）
    __ubuf__ float *pvSlot0;
    __ubuf__ float *pvSlot1;        // 配对折叠：step1 槽；单步折叠传 nullptr
    __ubuf__ float *oAcc;
    __ubuf__ float *alpha1;         // 配对折叠：step1 的 alpha 行（nullptr = 单步）
    int64_t validRows;
    int64_t dp;
    bool fp16Accum;
    bool paired;
};

struct Sa2VfFinishArgs {
    __ubuf__ float *oAcc;           // 原位归一化（O 驻 UB，D-R2；GM 写回由 wrapper 完成）
    __ubuf__ float *denom;          // 非 const：LoadAlign 拒绝 const T（同 FoldArgs 注记）
    __ubuf__ float *vMeanRow;       // smooth_v 行（wrapper 标量暂存；nullptr = 关闭）
    __ubuf__ float *mStage;         // LSE m 装配行（nullptr = returnLse 关闭）
    __ubuf__ float *corr;
    __ubuf__ float *stageRow;       // LSE 结果暂存行（GM 写回由 wrapper 完成）
    int64_t validRows;
    int64_t d;                      // 输出有效列宽（归一化范围）
    int64_t dp;                     // oAcc 行距
    float smScale;
};

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || defined(__CCE_KT_TEST__)
// Fixpipe L0C→UB 行主序配置（catlass copy_l0c_to_ub.hpp 同款常量）
constexpr AscendC::FixpipeConfig kSa2CfgRowMajorUb = {AscendC::CO2Layout::ROW_MAJOR, true};

// M2c-B1 设备映射：Sa2L2CacheHint（spec 策略层）→ AscendC::CacheMode（950 真实枚举
// CACHE_MODE_NORMAL/DISABLE/LAST/PERSISTENT，tensor_api/utils/utils.h）。语义对位：
// NORMAL_LAST_VICTIM → CACHE_MODE_LAST（次选逐出 = 常驻倾向）、NOTALLOC_KEEP →
// CACHE_MODE_DISABLE（不分配 L2 行）；精确逐出代际行为 = M2b 真机核实项（D-R5）。
SA2_AICORE constexpr AscendC::CacheMode Sa2CacheModeOf(Sa2L2CacheHint hint)
{
    return (hint == CACHE_HINT_NOTALLOC_KEEP || hint == CACHE_HINT_NOTALLOC_CLEAN)
               ? AscendC::CacheMode::CACHE_MODE_DISABLE
               : AscendC::CacheMode::CACHE_MODE_LAST;
}
#endif

#if __has_include("kernel_operator.h")
// =============================================================================
// M2b-0 设备指令序列常量（CANN 9.2.0-beta.1 dav-c310 实测口径）：
//   向量寄存器宽 256B ⇒ RegTensor<float> = 64 lane（catlass FLOAT_REP_SIZE=64 同值）、
//   RegTensor<half> = 128 lane、RegTensor<uint8_t> = 256 lane。
//   一次 RegTensor 装载/存储 = 单条 vld/vst；本文件 AIV 段一律以 kVlF32=64 为列分块单位。
// CastTrait 与 CANN ops load_store_utils.h / catlass tile_perblock_quant.hpp 同款：
//   窄化一律 CAST_RINT（RNE，与宿主 oracle aicNearbyInt / FloatToFp8E4M3 舍入一致）+ SAT
//  （softmax 概率 ∈ [0,κ]，κ ≤ 448 不触饱和，SAT 仅对齐 E4M3 封顶语义）；
//   宽化 UNKNOWN（无数值含义）。RegLayout ZERO/ONE = 源寄存器半区选择（ccec dav_3510
//   SupportEnum 仅放行 ZERO/ONE，见 asc/impl/.../dav_3510/kernel_reg_compute_vec_vconv_impl.h）。
using vapi::CastTrait;
using vapi::MaskMergeMode;
using vapi::RegLayout;
using vapi::SatMode;
constexpr int64_t kVlF32 = 64;   // RegTensor<float> lane 数
constexpr int64_t kVlF16 = 128;  // RegTensor<half> lane 数
constexpr vapi::CastTrait kCastB162F32Zero = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr vapi::CastTrait kCastB162F32One = {
    RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr vapi::CastTrait kCastF322F16Zero = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr vapi::CastTrait kCastF322Fp8 = {
    RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// =============================================================================
// M2b-0 VF 计算核（__simd_vf__ 自由函数）。ccec dav-c310 实测约束：
//   1. __simd_vf__ 函数必须为自由函数或 static 成员函数（普通成员 = 硬错误）；
//   2. 指针形参必须 __ubuf__ 地址空间限定（普通/GM/L1 指针形参 = 硬错误）；
//   3. 函数体内不得调用 __aicore__ 普通助手（事件 Notify*/Await*、aicMin/aicMax/aicExp
//      等）——事件与 GM/L1/MTE 搬运、L1 NZ 散写全部留在成员 wrapper 侧（非 VF 函数无此限）；
//   4. UB 槽位指针由 wrapper 解析后经上下文结构（值传递，定义见上方双轨共用段）携带。
// 数值语义 = 宿主 oracle（sim/）逐位镜像，映射注记随函数。
// =============================================================================

// 指令序 ↔ 宿主 oracle 映射（逐位语义冻结；位级对拍挂 M2b 真机出口）：
//   列分块 kKvStep=128 = 2 × RegTensor<float>(64 lane)；per_warp 经 Cast(B162F32, ZERO/ONE)
//   宽化（ZERO/ONE = 源寄存器低/高半区，catlass tile_perblock_quant 同款 trait 背书）；
//   ReduceMax/ReduceSum 按段掩码归约（掩码外 lane 不参与 = oracle activeMask 同构），双段
//   Max/Add 合并；归约结果 lane0 经寄存器 Duplicate 广播（CANN softmax_flash_impl.h 同款，
//   迭代 2 探针 p33b 实测循环内寄存器 Duplicate 合法）；mNew/alpha 状态数学全向量 1-lane
//   掩码（m1）执行 —— AIV 后端拒绝一切标量浮点 ALU/fcmp/exp 调用（迭代 2 探针矩阵钉死，
//   见 tests/aiv_backend_probe.cpp）：
//     mNew = max(mOld, mBlk − θ)（θ-skip 判定条件 mBlk − mOld ≤ θ ⇒ mBlk − θ ≤ mOld
//     ⇒ mNew = mOld 恒成立，与 oracle skip 覆盖逐位一致）；
//     alpha = exp(mOld − mNew)（skip ⇒ diff=0 ⇒ exp(0)=1.0f 位精确 = oracle 强制 1.0f；
//     非 skip 与 oracle 同公式。设备向量 ExpSub vs oracle std::exp 的 ulp 级差异 = 既有
//     真机对拍项，非本轮引入）；
//     O 重缩放条件改为 alpha 位型 ≠ 0x3F800000（= 1.0f），与 oracle「!skip && alpha≠1」
//     在 skip（位型恰 1.0f）/非 skip（同一 alpha）两支均逐位等价。
//   ExpSub(ZEROING) 掩码外产 0 = oracle ExpSub+Select(zero) 序等价；
//   FP8 打包镜像 catlass tile_perblock_quant.hpp 量产序列（Interleave + Cast(ZERO)×2 +
//   StoreAlign(DIST_PACK4_B32)×2 ⇒ UB 行内码序 = 列升序；Interleave 与 ZERO 取码的合成序
//   = 真机对拍确认点，误差局部化于本段）。
//   vloop 降低附加约束（迭代 2 探针实测）：带标量参的向量指令（Duplicate/Muls 的标量形参）
//   在行循环内必须循环不变（p33d）或来自「直接标量装载」（p33e）；标量位强转（bitcast of
//   计算值）在 VF 内不可选中（p33f/p33g）——故 qh 由 wrapper 预展开为 f32 行，VF 只做直接
//   float 装载（p33e 合法）。
template <bool IsThreadGran, bool IsCausal>
__simd_vf__ inline void Sa2VfSoftmaxRun(Sa2VfSoftmaxArgs a)
{
    constexpr int64_t kVL = 64;  // RegTensor<float> lane 数（256B 寄存器 / 4B）
    vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    uint32_t oneLane = 1u;
    vapi::MaskReg m1 = vapi::UpdateMask<uint32_t>(oneLane);
    // 循环不变量预广播（vloop 降低要求标量参指令可外提，探针 p33d/p33 实测）
    vapi::RegTensor<float> thetaV;
    vapi::Duplicate(thetaV, a.lnPDomain);
    vapi::RegTensor<float> zeroV;
    vapi::Duplicate(zeroV, 0.0f);
    vapi::RegTensor<float> negMaxV;
    vapi::Duplicate(negMaxV, kFp16NegMax);
    for (int64_t r = 0; r < a.validRows; ++r) {
        int64_t qTokenIdx = a.m0 + r;

        // 1. 向量化加载 QK 分数并结合 Q/K 尺度执行向量缩放
        vapi::RegTensor<float> s0;
        vapi::RegTensor<float> s1;
        if constexpr (!IsThreadGran) {
            // PerWarp：deqScore（half×128 单寄存器）宽化为双段，乘 kh（双段）与行标量 qh
            __ubuf__ half *scoreRow = a.scoreWarp + r * kKvStep;
            vapi::RegTensor<half> deqHalfReg;
            vapi::LoadAlign(deqHalfReg, scoreRow);
            vapi::Cast<float, half, kCastB162F32Zero>(s0, deqHalfReg, m64);
            vapi::Cast<float, half, kCastB162F32One>(s1, deqHalfReg, m64);

            vapi::RegTensor<half> khHalfReg;
            vapi::LoadAlign(khHalfReg, a.kScaleRow);
            vapi::RegTensor<float> kh0;
            vapi::RegTensor<float> kh1;
            vapi::Cast<float, half, kCastB162F32Zero>(kh0, khHalfReg, m64);
            vapi::Cast<float, half, kCastB162F32One>(kh1, khHalfReg, m64);

            vapi::Mul(s0, s0, kh0, m64);
            vapi::Mul(s1, s1, kh1, m64);

            // qh：wrapper 已按 half 位型展开为 f32 行（AIV 禁标量位强转，见文件头注记），
            // 此处直接 float 装载喂 Muls（探针 p33e 合法形态）
            float qh = a.qhF32[r];
            vapi::Muls(s0, s0, qh, m64);
            vapi::Muls(s1, s1, qh, m64);
        } else {
            // PerThread：分段反量化点积与 (QH · KH) 分段尺度严格组合（§4.2 组合式闭环）
            __ubuf__ float *scoreRow = a.scoreThread + r * kKvStep;
            vapi::LoadAlign(s0, scoreRow);
            vapi::LoadAlign(s1, scoreRow + kVL);
        }

        // 因果与边界掩码判定：有效列为 [0, maxActiveCol)，双段各自 UpdateMask（左值入参）
        int64_t maxActiveCol = a.sk - a.kvTokenStart;
        if constexpr (IsCausal) {
            int64_t causalLimit = (qTokenIdx + 1) - a.kvTokenStart;
            if (causalLimit < maxActiveCol) {
                maxActiveCol = causalLimit;
            }
        }
        if (maxActiveCol < 0) maxActiveCol = 0;
        if (maxActiveCol > kKvStep) maxActiveCol = kKvStep;

        uint32_t len0 = (maxActiveCol >= kVL) ? static_cast<uint32_t>(kVL)
                                              : static_cast<uint32_t>(maxActiveCol);
        uint32_t len1 = (maxActiveCol > kVL) ? static_cast<uint32_t>(maxActiveCol - kVL) : 0u;
        uint32_t cnt0 = len0;
        uint32_t cnt1 = len1;
        vapi::MaskReg act0 = vapi::UpdateMask<uint32_t>(cnt0);
        vapi::MaskReg act1 = vapi::UpdateMask<uint32_t>(cnt1);

        // 2. 行最大值：双段掩码归约 + 合并，lane0 经寄存器 Duplicate 广播（oracle mBlk）
        vapi::RegTensor<float> mBlkV;
        vapi::Duplicate(mBlkV, negMaxV, m64);
        if (maxActiveCol > 0) {
            vapi::RegTensor<float> max0;
            vapi::ReduceMax(max0, s0, act0);
            if (maxActiveCol > kVL) {
                vapi::RegTensor<float> max1;
                vapi::ReduceMax(max1, s1, act1);
                vapi::Max(max0, max0, max1, m64);
            }
            vapi::Duplicate(mBlkV, max0, m64);
        }

        // 3. online softmax 状态与 FlashMLA θ-skip（全向量 1-lane 掩码；语句序 = oracle 镜像）：
        //    mOld = rowStat lane0（状态行 64-lane 窗口装载，仅 lane0 参与后续 m1 掩码运算）
        vapi::RegTensor<float> stateV;
        vapi::LoadAlign(stateV, a.rowStat + r * 2);
        vapi::RegTensor<float> mBlkShift;
        vapi::Sub(mBlkShift, mBlkV, thetaV, m1);   // mBlk − θ
        vapi::RegTensor<float> mNewV;
        vapi::Max(mNewV, stateV, mBlkShift, m1);   // skip ⇒ mBlk−θ ≤ mOld ⇒ mNew = mOld
        vapi::RegTensor<float> diff;
        vapi::Sub(diff, stateV, mNewV, m1);        // mOld − mNew
        vapi::RegTensor<float> alphaV;
        vapi::ExpSub(alphaV, diff, zeroV, m1);     // skip ⇒ exp(0) = 1.0f 位精确

        // 4. ExpSub 硬件单指令：mNew lane0 广播后双段 exp(s - m)；
        //    ZEROING 使掩码外 lane 产 0（与 oracle ExpSub+Select(zero) 序等价）
        vapi::RegTensor<float> mNewB;
        vapi::Duplicate(mNewB, mNewV, m64);
        vapi::RegTensor<float> p0;
        vapi::RegTensor<float> p1;
        vapi::ExpSub(p0, s0, mNewB, act0);
        vapi::ExpSub(p1, s1, mNewB, act1);

        // 向量累加求行和（分段归约 + lane0 广播；分段次序差 = 归约树形状差，挂真机对拍）
        vapi::RegTensor<float> sum0;
        if (maxActiveCol > 0) {
            vapi::ReduceSum(sum0, p0, act0);
            if (maxActiveCol > kVL) {
                vapi::RegTensor<float> sum1;
                vapi::ReduceSum(sum1, p1, act1);
                vapi::Add(sum0, sum0, sum1, m64);
            }
            vapi::Duplicate(sum0, sum0, m64);
        } else {
            vapi::Duplicate(sum0, zeroV, m64);
        }

        // denom = alpha·denom + sumP8（lane0 掩码读改写；探针 p23 形态）
        vapi::RegTensor<float> denomV;
        vapi::LoadAlign(denomV, a.denomSlot + r);
        vapi::Mul(denomV, denomV, alphaV, m1);
        vapi::Add(denomV, denomV, sum0, m1);
        vapi::StoreAlign(a.denomSlot + r, denomV, m1);

        // 状态回写（m1 = 仅 lane0；rowStat lane1 旗标不受扰动，alpha 落 parity 槽供配对折叠）
        vapi::StoreAlign(a.rowStat + r * 2, mNewV, m1);
        vapi::StoreAlign(a.alphaSlot + r, alphaV, m1);

        // O 累加器向量重缩放：alpha 位型 ≠ 1.0f 才执行（位型比较 = 整数指令，等价原
        // 「!skip && alpha != 1.0f」；skip 支路 alpha 位型恰 0x3F800000 ⇒ 自动跳过）。
        // 位型提取 = FIRST_ELEMENT 单指令落 UB + 整数标量装载（VF 禁标量 fcmp）。
        if (maxActiveCol > 0) {
            vapi::StoreAlign<float, vapi::StoreDist::DIST_FIRST_ELEMENT_B32>(a.scalarSlot + 2, alphaV, m64);
            uint32_t alphaBits = *reinterpret_cast<__ubuf__ uint32_t *>(a.scalarSlot + 2);
            if (alphaBits != 0x3F800000u) {
                vapi::RegTensor<float> alphaB;
                vapi::Duplicate(alphaB, alphaV, m64);
                for (int64_t c = 0; c < a.dp; c += kVL) {
                    int64_t chunkLen = (a.dp - c < kVL) ? (a.dp - c) : kVL;
                    uint32_t oCnt = static_cast<uint32_t>(chunkLen);
                    vapi::MaskReg oMask = vapi::UpdateMask<uint32_t>(oCnt);
                    vapi::RegTensor<float> oReg;
                    vapi::LoadAlign(oReg, a.oAcc + r * a.dp + c);
                    vapi::Mul(oReg, oReg, alphaB, oMask);
                    vapi::StoreAlign(a.oAcc + r * a.dp + c, oReg, oMask);
                }
            }
        }

        // 5. FP8 E4M3FN 离散化（oracle 语义：RINT 舍入 + 448 封顶）写入 UB P 槽（ND 行）
        vapi::Interleave(p0, p1, p0, p1);
        vapi::RegTensor<float8_e4m3_t> q0;
        vapi::RegTensor<float8_e4m3_t> q1;
        vapi::Cast<float8_e4m3_t, float, kCastF322Fp8>(q0, p0, m64);
        vapi::Cast<float8_e4m3_t, float, kCastF322Fp8>(q1, p1, m64);
        vapi::MaskReg full8 = vapi::CreateMask<uint8_t, vapi::MaskPattern::ALL>();
        __ubuf__ uint8_t *pRowUb = a.pSlotUb + r * kKvStep;
        vapi::StoreAlign<uint8_t, vapi::StoreDist::DIST_PACK4_B32>(
            pRowUb, reinterpret_cast<vapi::RegTensor<uint8_t> &>(q0), full8);
        vapi::StoreAlign<uint8_t, vapi::StoreDist::DIST_PACK4_B32>(
            pRowUb + kVL, reinterpret_cast<vapi::RegTensor<uint8_t> &>(q1), full8);
    }
}

// PV 折叠（fp32 / fp32+fp32 / fp32+fp16）。fp16 往返 Cast 显式携带 RINT trait：
// 窄化 ZERO / 宽化 ZERO 半区选择对称 ⇒ 逐元素 f32→f16→f32 往返与 oracle .half() 语义一致
//（位级对拍挂 M2b 真机出口）。
// step1 alpha 条件乘（配对折叠）：AIV 后端禁标量 fcmp 且禁「计算值/装载值」的标量位强转
//（迭代 2 探针 p33f/p33g + 内核实测 GVN 会把 u32/f32 双装载合并回不可选中的 bitcast）
// ⇒ 改全向量形态：条件 = 位型 u32 直接装载 + 整数比较（0x3F800000 = 1.0f 位型唯一，
// 与 oracle「a1 != 1.0f」逐位等价）；乘法 = alpha1 行向量装载 + 寄存器 Duplicate 广播
// lane0 + 向量 Mul（p33b 合法形态，CANN softmax_flash_impl.h 状态广播同款）。
// 语句序：o = o + a1·f16(pv0) → o = o + f16(pv1)（P1-2 配对调度，
// sim/sa2_sim_loop.h StepVectorPvFoldPair 同提交镜像）。
__simd_vf__ inline void Sa2VfFoldRun(Sa2VfFoldArgs a)
{
    constexpr int64_t kVL = 64;
    vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    const bool hasAlpha1 = (a.alpha1 != nullptr);
    for (int64_t r = 0; r < a.validRows; ++r) {
        vapi::RegTensor<float> a1B;
        bool scalePv0 = false;
        if (hasAlpha1) {
            uint32_t a1Bits = *reinterpret_cast<__ubuf__ uint32_t *>(a.alpha1 + r);
            scalePv0 = (a1Bits != 0x3F800000u);
            if (scalePv0) {
                vapi::RegTensor<float> a1V;
                vapi::LoadAlign(a1V, a.alpha1 + r);
                vapi::Duplicate(a1B, a1V, m64);  // 广播 lane0 = alpha1[r]
            }
        }
        for (int64_t c = 0; c < a.dp; c += kVL) {
            int64_t chunkLen = (a.dp - c < kVL) ? (a.dp - c) : kVL;
            uint32_t cnt = static_cast<uint32_t>(chunkLen);
            vapi::MaskReg mask = vapi::UpdateMask<uint32_t>(cnt);

            vapi::RegTensor<float> oReg;
            vapi::LoadAlign(oReg, a.oAcc + r * a.dp + c);
            if (a.paired) {
                vapi::RegTensor<float> pvReg0;
                vapi::RegTensor<float> pvReg1;
                vapi::LoadAlign(pvReg0, a.pvSlot0 + r * a.dp + c);
                vapi::LoadAlign(pvReg1, a.pvSlot1 + r * a.dp + c);
                if (a.fp16Accum) {
                    vapi::RegTensor<half> pvHalfReg0;
                    vapi::RegTensor<half> pvHalfReg1;
                    vapi::Cast<half, float, kCastF322F16Zero>(pvHalfReg0, pvReg0, mask);
                    vapi::Cast<float, half, kCastB162F32Zero>(pvReg0, pvHalfReg0, mask);
                    vapi::Cast<half, float, kCastF322F16Zero>(pvHalfReg1, pvReg1, mask);
                    vapi::Cast<float, half, kCastB162F32Zero>(pvReg1, pvHalfReg1, mask);
                }
                // step1 发生非 θ-skip 重缩放（alpha1 位型 ≠ 1.0f）时，pvReg0 随路缩放
                if (scalePv0) {
                    vapi::Mul(pvReg0, pvReg0, a1B, mask);
                }
                vapi::Add(oReg, oReg, pvReg0, mask);
                vapi::Add(oReg, oReg, pvReg1, mask);
            } else {
                vapi::RegTensor<float> pvReg;
                vapi::LoadAlign(pvReg, a.pvSlot0 + r * a.dp + c);
                if (a.fp16Accum) {
                    vapi::RegTensor<half> pvHalfReg;
                    vapi::Cast<half, float, kCastF322F16Zero>(pvHalfReg, pvReg, mask);
                    vapi::Cast<float, half, kCastB162F32Zero>(pvReg, pvHalfReg, mask);
                }
                vapi::Add(oReg, oReg, pvReg, mask);
            }
            vapi::StoreAlign(a.oAcc + r * a.dp + c, oReg, mask);
        }
    }
}

// Epilogue：O 原位归一化 + smooth_v 补偿 + LSE 闭环 ln(denom) + m + s·corr → stageRow。
// GM 侧写回（标量、含 dtype 转换）全部在 wrapper（VF 形参禁 GM 指针）。
__simd_vf__ inline void Sa2VfFinishRun(Sa2VfFinishArgs a)
{
    constexpr int64_t kVL = 64;
    // invDenom 行预计算（标量 fdiv 为 AIV loop 不支持指令，实测；改向量 Div 一次算满
    // validRows 行，行循环内改读 UB invDenom 行）。scratch +1280 起 256B 段未占用。
    if (a.validRows > 0) {
        uint32_t validRowsU = static_cast<uint32_t>(a.validRows);
        vapi::MaskReg rowMaskAll = vapi::UpdateMask<uint32_t>(validRowsU);
        vapi::RegTensor<float> dReg, oneReg, invReg;
        vapi::LoadAlign(dReg, a.denom);
        vapi::Duplicate(oneReg, 1.0f);
        vapi::Div(invReg, dReg, oneReg, rowMaskAll);
        __ubuf__ float *invRow = reinterpret_cast<__ubuf__ float *>(
            reinterpret_cast<uint64_t>(a.stageRow) + 1024);  // scratch +1024..1279（256B 空段）
        vapi::StoreAlign(invRow, invReg, rowMaskAll);
    }
    for (int64_t r = 0; r < a.validRows; ++r) {
        float invDenom = a.stageRow[256 + r];  // +1024B = +256 float 槽位
        for (int64_t c = 0; c < a.d; c += kVL) {
            int64_t chunkLen = (a.d - c < kVL) ? (a.d - c) : kVL;
            uint32_t cnt = static_cast<uint32_t>(chunkLen);
            vapi::MaskReg mask = vapi::UpdateMask<uint32_t>(cnt);
            vapi::RegTensor<float> oReg;
            vapi::LoadAlign(oReg, a.oAcc + r * a.dp + c);
            vapi::Muls(oReg, oReg, invDenom, mask);
            if (a.vMeanRow != nullptr) {
                vapi::RegTensor<float> vMeanReg;
                vapi::LoadAlign(vMeanReg, a.vMeanRow + c);
                vapi::Add(oReg, oReg, vMeanReg, mask);
            }
            vapi::StoreAlign(a.oAcc + r * a.dp + c, oReg, mask);
        }
    }

    if (a.mStage != nullptr && a.validRows > 0) {
        uint32_t rowCnt = static_cast<uint32_t>(a.validRows);
        vapi::MaskReg rowMask = vapi::UpdateMask<uint32_t>(rowCnt);
        vapi::RegTensor<float> denomReg;
        vapi::RegTensor<float> mReg;
        vapi::RegTensor<float> corrReg;
        vapi::RegTensor<float> lseReg;
        vapi::LoadAlign(denomReg, a.denom);
        vapi::LoadAlign(mReg, a.mStage);
        vapi::LoadAlign(corrReg, a.corr);
        vapi::Ln(lseReg, denomReg, rowMask);
        vapi::Add(lseReg, lseReg, mReg, rowMask);
        vapi::RegTensor<float> scaledCorr;
        vapi::Muls(scaledCorr, corrReg, a.smScale, rowMask);
        vapi::Add(lseReg, lseReg, scaledCorr, rowMask);
        vapi::StoreAlign(a.stageRow, lseReg, rowMask);
    }
}

// =============================================================================
// VF 调用 thunk（自由函数）。ccec dav-c310 实测：__simd_vf__ 函数的调用点若位于
// 类非静态成员函数体内，AIV pass 后端报 "Unsupported scalar instruction in AIV loop"
//（成员上下文的调用序列不被 AIV loop 降低接受）；自由函数调用点同形验证可过
//（本文件 VF 探针对照实测）。故成员 wrapper 一律经本组 thunk 间接调用。
__aicore__ inline void Sa2VfSoftmaxInvoke(Sa2VfSoftmaxArgs a)
{
    if (a.isThreadGran) {
        Sa2VfSoftmaxRun<true, false>(a);
    } else {
        Sa2VfSoftmaxRun<false, false>(a);
    }
}

template <bool IsCausal>
__aicore__ inline void Sa2VfSoftmaxInvokeC(Sa2VfSoftmaxArgs a)
{
    if (a.isThreadGran) {
        Sa2VfSoftmaxRun<true, IsCausal>(a);
    } else {
        Sa2VfSoftmaxRun<false, IsCausal>(a);
    }
}

__aicore__ inline void Sa2VfFoldInvoke(Sa2VfFoldArgs a) { Sa2VfFoldRun(a); }

__aicore__ inline void Sa2VfFinishInvoke(Sa2VfFinishArgs a) { Sa2VfFinishRun(a); }

#else
// =============================================================================
// 宿主桩轨（hoststub 仿真）标量镜像：语义 = 设备 VF 序列逐位同构（oracle 镜像）。
// softmax 镜像注记（迭代 2）：设备已改为「无 skip 覆盖」的全公式形态——
//   mNew = max(mOld, mBlk − θ)、alpha = exp(mOld − mNew)、O 重缩放条件 = alpha ≠ 1.0f；
//   θ-skip 在设备上以「skip ⇒ mBlk − θ ≤ mOld ⇒ mNew = mOld（Max 恒等）+ exp(0)=1.0f
//   位精确」恒等式实现，与本镜像的「先草算再覆盖」在全部输入上逐位一致（证明见设备侧
//   Sa2VfSoftmaxRun 头注）。故本镜像语句序保持覆盖形态不改，位一致由该恒等式背书。
// 仅用于宿主语法检查/仿真构建（真机数值由上方 __simd_vf__ 分支承载）；无 vapi 依赖。
// =============================================================================
template <bool IsThreadGran, bool IsCausal>
inline void Sa2VfSoftmaxRun(Sa2VfSoftmaxArgs a)
{
    for (int64_t r = 0; r < a.validRows; ++r) {
        int64_t qTokenIdx = a.m0 + r;
        float s[kKvStep];
        for (int64_t c = 0; c < kKvStep; ++c) {
            if (!IsThreadGran) {
                s[c] = static_cast<float>(a.scoreWarp[r * kKvStep + c]) *
                       static_cast<float>(a.kScaleRow[c]) * a.qhF32[r];
            } else {
                s[c] = a.scoreThread[r * kKvStep + c];
            }
        }
        int64_t maxActiveCol = a.sk - a.kvTokenStart;
        if constexpr (IsCausal) {
            int64_t causalLimit = (qTokenIdx + 1) - a.kvTokenStart;
            if (causalLimit < maxActiveCol) maxActiveCol = causalLimit;
        }
        if (maxActiveCol < 0) maxActiveCol = 0;
        if (maxActiveCol > kKvStep) maxActiveCol = kKvStep;

        float mBlk = -65504.0f;
        if (maxActiveCol > 0) {
            mBlk = s[0];
            for (int64_t c = 1; c < maxActiveCol; ++c) {
                if (s[c] > mBlk) mBlk = s[c];
            }
        }
        float mOld = a.rowStat[r * 2 + 0];
        float mBlkShift = mBlk - a.lnPDomain;
        float mNew = (mOld > mBlkShift) ? mOld : mBlkShift;
        float alpha = std::exp(mOld - mNew);
        if ((mBlk - mOld) <= a.lnPDomain) {
            alpha = 1.0f;
            mNew = mOld;
        }
        float sumP8 = 0.0f;
        for (int64_t c = 0; c < kKvStep; ++c) {
            float pv = (c < maxActiveCol) ? std::exp(s[c] - mNew) : 0.0f;
            if (c < maxActiveCol) sumP8 += pv;
            a.pSlotUb[r * kKvStep + c] = FloatToFp8E4M3(pv);
        }
        a.denomSlot[r] = alpha * a.denomSlot[r] + sumP8;
        a.rowStat[r * 2 + 0] = mNew;
        a.alphaSlot[r] = alpha;
        if (alpha != 1.0f) {
            for (int64_t c = 0; c < a.dp; ++c) {
                a.oAcc[r * a.dp + c] *= alpha;
            }
        }
    }
}

inline void Sa2VfFoldRun(Sa2VfFoldArgs a)
{
    for (int64_t r = 0; r < a.validRows; ++r) {
        float a1 = (a.alpha1 != nullptr) ? a.alpha1[r] : 1.0f;
        for (int64_t c = 0; c < a.dp; ++c) {
            const int64_t k = r * a.dp + c;
            float pv0 = a.pvSlot0[k];
            if (a.fp16Accum) pv0 = static_cast<float>(static_cast<half>(pv0));
            if (a.paired) {
                float pv1 = a.pvSlot1[k];
                if (a.fp16Accum) pv1 = static_cast<float>(static_cast<half>(pv1));
                a.oAcc[k] = a.oAcc[k] + pv0 * a1 + pv1;
            } else {
                a.oAcc[k] = a.oAcc[k] + pv0;
            }
        }
    }
}

inline void Sa2VfFinishRun(Sa2VfFinishArgs a)
{
    for (int64_t r = 0; r < a.validRows; ++r) {
        float invDenom = (a.denom[r] > 0.0f) ? (1.0f / a.denom[r]) : 0.0f;
        for (int64_t c = 0; c < a.d; ++c) {
            float v = a.oAcc[r * a.dp + c] * invDenom;
            if (a.vMeanRow != nullptr) v += a.vMeanRow[c];
            a.oAcc[r * a.dp + c] = v;
        }
    }
    if (a.mStage != nullptr && a.validRows > 0) {
        for (int64_t r = 0; r < a.validRows; ++r) {
            a.stageRow[r] = std::log(a.denom[r]) + a.mStage[r] + a.smScale * a.corr[r];
        }
    }
}

__aicore__ inline void Sa2VfSoftmaxInvoke(Sa2VfSoftmaxArgs a)
{
    if (a.isThreadGran) {
        Sa2VfSoftmaxRun<true, false>(a);
    } else {
        Sa2VfSoftmaxRun<false, false>(a);
    }
}
template <bool IsCausal>
__aicore__ inline void Sa2VfSoftmaxInvokeC(Sa2VfSoftmaxArgs a)
{
    if (a.isThreadGran) {
        Sa2VfSoftmaxRun<true, IsCausal>(a);
    } else {
        Sa2VfSoftmaxRun<false, IsCausal>(a);
    }
}
__aicore__ inline void Sa2VfFoldInvoke(Sa2VfFoldArgs a) { Sa2VfFoldRun(a); }
__aicore__ inline void Sa2VfFinishInvoke(Sa2VfFinishArgs a) { Sa2VfFinishRun(a); }

#endif  // __has_include("kernel_operator.h")

// =============================================================================
// Softmax 归约助手（v27 诚实化重注记，todo §一.3）：
// 当前实现 = 宿主标量仿真（行 max / exp / FP8 离散化 / Kahan 补偿求和），是算法正确性
// 载体与差分 oracle；"TROWMAX/TROWSUM 同引擎零屏障 / 10^-9"是 M2b 向量化阶段的落地纪律
//（D-R9：同一 V 引擎内零 pipe_barrier，归约走 AscendC 2D Tile 指令），非本层已实现事实；
// 真机误差量级 = M5.1 实测项。AIV 设备分支 = vapi MicroAPI 指令层（宿主由
// sa2_ascendc_hoststub.h 同构仿真），AIC 侧 = 宿主标量仿真（P0-4 双轨纪律）。
// =============================================================================
// =============================================================================
// M2.7.1 同引擎零停顿 —— AIV 侧计算序列同步审计表（迭代 5 落地物，同步拓扑唯一陈述）。
// 分类纪律：[免屏障] = 纯 PIPE_V 计算段，DaVinci 同队列指令发射保序，RAW/WAR 由硬件按序保证；
//           [方向切换] = UB 装载/写回方向切换（V 计算 ↔ MTE 搬运），需 pipe_barrier/LocalMemBar 收口；
//           [跨引擎边] = AIC ↔ AIV，一律走 sa2_events 计数旗号对，禁止以私加屏障替代。
//
// 阶段一 KV 预处理（sa2_kv_prep.h，全 AIV、KV head 单写者分头，无事件无屏障）：
//   K1 ProcessKMeanHost  ：GM K 读 → V 归累 → GM 写回 wsKMean            | 纯 V 计算段 [免屏障]
//   K2 ProcessKQuantHost ：K 中心化 → amax/127 量化 → L1/GM 写回          | 纯 V [免屏障]（依赖 K1 程序序）
//   K3 ProcessVQuantHost ：V MX 组 amax → ComputeE8m0Exponent 位技巧 → FP8 | 纯 V [免屏障]
//   阶段收口（跨阶段边）：AIV 写 GM 队列排空 PipeBarrier(PIPE_ALL) → SyncAll()
//   （sa2_kernel.cpp：全栅格 AIC+AIV 驻留同步，全局唯一屏障，必要非冗余）
//
// 阶段二 注意力主循环（本文件，AIV 流水线 V1 → (V2 → V3)×steps → V4）：
//   段 V1 StepVectorQQuant：GM q 读 → UB 状态初始化（corr/rowStat/denom/oAcc）+ L1 q8/qScale 写
//       [计算+写回，仿真单队列免屏障]
//       出口跨引擎边：kQ8Ok NotifyV(PIPE_V 生产) → AIC StepCubeQk step==0 AwaitMte1（消费管道 MTE1：L1→L0）
//   段 V2 StepVectorSoftmax（宿主标量 / vapi 设备双轨同构）：
//       入口跨引擎边：kScoreOk AwaitFix（AIC PIPE_FIX 生产 → PIPE_V 消费，CrossEnginePending 类型态门）
//       段内：UB score 装载 → 缩放/因果掩码 → ReduceMax → θ-skip(mNew=mOld) → ExpSub/ReduceSum →
//             P 双槽写（UB ND + L1 NZ），全程纯 PIPE_V ⇒ [免屏障]（D-R9 零 pipe_barrier 纪律段）
//       出口跨引擎边：kQkFree NotifyV → AIC AwaitV(PIPE_FIX 重填 score 槽)；
//                     kPTaken NotifyMte3 → AIC AwaitMte1（MTE1 取 P 入 L0A）
//       ⚠ M2b 真机落地注记：P 由 PIPE_V 产出而旗号走 PIPE_MTE3（C.4 口径 = "UB→L1 搬运退休后"）；
//         真机若以 DataCopy(UB→L1, MTE3) 产 P 的 L1 副本，须在 SetFlag 前插入 V→MTE3 方向收口
//         （[方向切换] 点）；当前双轨仿真直写 L1 槽、无 MTE 段，桩序已保证消费可见性。
//   段 V3 StepVectorPvFold（宿主 / vapi 双轨）：
//       入口跨引擎边：kPvOk AwaitFix（AIC FIX → V）
//       段内：UB pv 装载 →（fp32+fp16 档 half 往返折叠）→ Add → oAcc 写回，纯 PIPE_V ⇒ [免屏障]
//       出口跨引擎边：kPvFree NotifyV → AIC AwaitV(PIPE_FIX 复用 PV 槽)
//   段 V4 FinishTileVector（宿主 / vapi 双轨）：
//       段内：UB 状态读 → 归一化/smooth_v → GM out 写回 + 向量化 LSE 闭环；任务末无旗号（流尾自然对齐）
//
// AIC 对偶（本文件下半，供对账）：C1 StepCubeQk 与 C2 StepCubePv 各有宿主/设备双轨，事件对对称——
//   AwaitMte1 kQ8Ok@step0 / AwaitV kQkFree@step≥2 → 产 UB score → NotifyFix kScoreOk；
//   AwaitMte1 kPTaken / AwaitV kPvFree@step≥2 → 产 UB pv → NotifyFix kPvOk；
//   C3 FinishTileCube drain kQkFree/kPvFree ×min(steps,2)。
//   AIC 设备分支 = CATLASS/TLA 真实多队列流水：每 GEMM 步 5 处队列切换收口
//   （MTE1 搬 L0A → MTE2 预取 L1 → MTE1 搬 L0B → M 矩乘 → FIX 写 UB；各 PipeBarrier 为
//   [方向切换] 类队列间收口，必要非冗余——AIC 单核内跨队列必须显式排空）。
//
// 审计结论（迭代 5，grep 全量对账，对应最终态：宿主/vapi AIV + CATLASS AIC 双轨）：
//   1. AIV 侧同引擎段内（V2/V3/V4 与 sa2_kv_prep.h K1~K3）0 个 pipe_barrier/LocalMemBar —— 零屏障
//      纪律成立，无同引擎冗余屏障可删；
//   2. 屏障全量清单 = sa2_kernel.cpp SyncAll 前 PIPE_ALL（跨阶段收口）+ AIC 设备分支每 GEMM 步
//      5 处队列切换收口（见上，[方向切换] 类）——全部必要，无同引擎冗余屏障；
//   3. 跨引擎边 6 旗号（kQ8Ok/kScoreOk/kQkFree/kPvOk/kPvFree/kPTaken）每任务 1:1 计数守恒，
//      生产/消费管道配对符合 sa2_events.h 附录 C.4 口径（V 产 V 旗 / FIX 产 FIX 旗 / MTE3 产 MTE1 收）；
//   4. 唯一遗留注记 = V2 出口 kPTaken 的 V→MTE3 方向收口（M2b 真机落地项，见上 ⚠）。
// =============================================================================



template <typename Dtype, Gran G, bool IsCausal>
class Sa2AttentionLoop {
public:
    __aicore__ inline Sa2AttentionLoop()
    {
        pipe_.InitBuffer(ubBuf_, Sa2DeviceRes<G>::kUbEnd);
        // M2b-0（拆分编译实测）：L1/L0 TPipe 分配仅 AIC 合法——AIV pass 对 L1/L0C TBuf
        // InitBuffer 直接触发后端硬错误（"Unsupported scalar instruction in AIV loop" /
        // "error pointer address space cast"，-O0/-O3 同现）。AIV 侧本地内存只有 UB；
        // L1/L0A/L0B/L0C 张量构建是 AIC 专属路径，编译期按核型裁剪。
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(l1Buf_, kL1End);
            pipe_.InitBuffer(l0aBuf_, kL0ABBytes);
            pipe_.InitBuffer(l0bBuf_, kL0ABBytes);
            pipe_.InitBuffer(l0cBuf_, kL0CQuadTotal);
        }
        BindLocalBases();
        mxScaleCache_.Init(l1_.GetMxScaleCacheBase(), kL1MxScaleCacheBytes);
    }

    __aicore__ inline void Init(
        GM_ADDR qGmBase, GM_ADDR outGmBase, GM_ADDR lseGmBase,
        GM_ADDR userWs, const Sa2TilingData &td)
    {
        qGmBase_ = qGmBase;
        outGmBase_ = outGmBase;
        lseGmBase_ = lseGmBase;
        userWs_ = userWs;
        td_ = &td;

        pipe_.InitBuffer(ubBuf_, Sa2DeviceRes<G>::kUbEnd);
        // M2b-0（拆分编译实测）：L1/L0 TPipe 分配仅 AIC 合法——AIV pass 对 L1/L0C TBuf
        // InitBuffer 直接触发后端硬错误（"Unsupported scalar instruction in AIV loop" /
        // "error pointer address space cast"，-O0/-O3 同现）。AIV 侧本地内存只有 UB；
        // L1/L0A/L0B/L0C 张量构建是 AIC 专属路径，编译期按核型裁剪。
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(l1Buf_, kL1End);
            pipe_.InitBuffer(l0aBuf_, kL0ABBytes);
            pipe_.InitBuffer(l0bBuf_, kL0ABBytes);
            pipe_.InitBuffer(l0cBuf_, kL0CQuadTotal);
        }
        BindLocalBases();
        mxScaleCache_.Init(l1_.GetMxScaleCacheBase(), kL1MxScaleCacheBytes);
    }

    // M2b-0：真实 LocalTensor 无 GetData()（hoststub 假想 API）——
    // UB 基址经 GetPhyAddr() 取物理地址后转 __ubuf__ 限定指针（MicroAPI LoadAlign/StoreAlign
    // 与标量 UB 访问的唯一合法入口）；L1/L0C 基址转普通指针（仅标量访问/偏移运算载体，
    // 真实张量一律经下方 L0*Tensor()/ubTensor() 走 TBuf 链构建）。
    __aicore__ inline void BindLocalBases()
    {
        ub_.SetBase(reinterpret_cast<__ubuf__ uint8_t *>(ubBuf_.template Get<uint8_t>().GetPhyAddr()));
        l1_.SetBase(reinterpret_cast<uint8_t *>(l1Buf_.template Get<uint8_t>().GetPhyAddr()));
        l0c_.SetBase(reinterpret_cast<uint8_t *>(l0cBuf_.template Get<uint8_t>().GetPhyAddr()));
    }

    // AIC 侧 LocalTensor 构建（MTE1/MMA 消费）：TBuf::Get()[字节偏移].ReinterpretCast<T>()，
    // 位置（A1/A2/B1/B2/CO1/VECCALC）由 TBuf 模板参数携带，满足 catlass TileCopyTla/
    // TileMmadTla 对 tla::Tensor<LocalTensor<E>, Layout, Coord, TPosition::x> 的特化匹配。
    __aicore__ inline AscendC::LocalTensor<int8_t> L1TensorI8(int64_t byteOffset)
    {
        return l1Buf_.template Get<uint8_t>()[static_cast<uint32_t>(byteOffset)].template ReinterpretCast<int8_t>();
    }
    __aicore__ inline AscendC::LocalTensor<float8_e4m3_t> L1TensorF8(int64_t byteOffset)
    {
        return l1Buf_.template Get<uint8_t>()[static_cast<uint32_t>(byteOffset)]
            .template ReinterpretCast<float8_e4m3_t>();
    }
    __aicore__ inline AscendC::LocalTensor<float8_e8m0_t> L1TensorF8E8M0(int64_t byteOffset)
    {
        return l1Buf_.template Get<uint8_t>()[static_cast<uint32_t>(byteOffset)]
            .template ReinterpretCast<float8_e8m0_t>();
    }
    __aicore__ inline AscendC::LocalTensor<int8_t> L0aTensorI8()
    {
        return l0aBuf_.template Get<uint8_t>().template ReinterpretCast<int8_t>();
    }
    __aicore__ inline AscendC::LocalTensor<float8_e4m3_t> L0aTensorF8()
    {
        return l0aBuf_.template Get<uint8_t>().template ReinterpretCast<float8_e4m3_t>();
    }
    __aicore__ inline AscendC::LocalTensor<int8_t> L0bTensorI8()
    {
        return l0bBuf_.template Get<uint8_t>().template ReinterpretCast<int8_t>();
    }
    __aicore__ inline AscendC::LocalTensor<float8_e4m3_t> L0bTensorF8()
    {
        return l0bBuf_.template Get<uint8_t>().template ReinterpretCast<float8_e4m3_t>();
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> L0cTensorI32(int64_t byteOffset)
    {
        return l0cBuf_.template Get<uint8_t>()[static_cast<uint32_t>(byteOffset)]
            .template ReinterpretCast<int32_t>();
    }
    __aicore__ inline AscendC::LocalTensor<float> L0cTensorF32(int64_t byteOffset)
    {
        return l0cBuf_.template Get<uint8_t>()[static_cast<uint32_t>(byteOffset)]
            .template ReinterpretCast<float>();
    }
    // UB 槽张量（Fixpipe 目的侧）：kUbEnd 全量 TBuf 内按字节偏移取槽。
    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> UbTensor(int64_t byteOffset)
    {
        return ubBuf_.template Get<uint8_t>()[static_cast<uint32_t>(byteOffset)].template ReinterpretCast<T>();
    }

    // ============================ AIV Vector 引擎流水线 ============================

    /**
     * @brief AIV 负责：Q 片上量化（常驻 L1）、corr = q·μ 顺路计算、Softmax 状态初始化
     */
    __aicore__ inline void StepVectorQQuant(const Sa2Task &task, int64_t streamId)
    {
        StepVectorQQuantDevice(task, streamId);
    }

    __aicore__ inline void StepVectorQQuantDevice(const Sa2Task &task, int64_t streamId)
    {
        const Sa2Params &p = td_->params;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const int64_t validRows = QTileValidRows(p, task.m0);

        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        __gm__ const float *wsKMean = reinterpret_cast<__gm__ const float *>(
            userWs_ + td_->kMean + kvView.wsKMean);
        const int64_t qElemOffset = QTileOffset(p, task.b, task.qh, task.m0);
        __gm__ const Dtype *qPtr = reinterpret_cast<__gm__ const Dtype *>(qGmBase_) + qElemOffset;

        __ubuf__ float *qFloatBuf = ub_.GetPvSlot(0);

        __ubuf__ float *corrSlot = ub_.GetStateCorr(streamId);
        for (int64_t r = 0; r < kStreamRows; ++r) {
            float rowCorr = 0.0f;
            if (r < validRows) {
                for (int64_t c = 0; c < p.dp; ++c) {
                    float qVal = (c < p.d) ? Sa2ScalarToFloat(qPtr[r * p.qStride[2] + c]) : 0.0f;
                    qFloatBuf[r * p.dp + c] = qVal;
                    rowCorr += qVal * wsKMean[c];
                }
            } else {
                for (int64_t c = 0; c < p.dp; ++c) {
                    qFloatBuf[r * p.dp + c] = 0.0f;
                }
            }
            corrSlot[r] = rowCorr;
        }

        int8_t *q8L1 = l1_.GetQ8Slot(streamId);
        half *qScaleL1 = l1_.GetQScaleSlot(streamId);

        if constexpr (!isThreadGran) {
            for (int64_t r = 0; r < kStreamRows; ++r) {
                float amax = 0.0f;
                if (r < validRows) {
                    for (int64_t c = 0; c < p.d; ++c) {
                        float absVal = aicAbs(qFloatBuf[r * p.dp + c]);
                        if (absVal > amax) amax = absVal;
                    }
                }
                const Sa2QuantScaleReciprocal qRecip(amax, p.qFactor);

                for (int64_t c = 0; c < p.dp; ++c) {
                    if (r < validRows && c < p.d) {
                        q8L1[r * p.dp + c] = qRecip.Quantize(qFloatBuf[r * p.dp + c]);
                    } else {
                        q8L1[r * p.dp + c] = 0;
                    }
                }
                qScaleL1[r] = Sa2ScalarFromFloat<half>(qRecip.scaledFactor);
            }
        } else {
            const int64_t numSegs = p.dp / kSegCols;
            for (int64_t r = 0; r < kStreamRows; ++r) {
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    float amax = 0.0f;
                    int64_t segStart = seg * kSegCols;
                    int64_t segEnd = aicMin(segStart + kSegCols, p.d);

                    if (r < validRows && segStart < p.d) {
                        for (int64_t c = segStart; c < segEnd; ++c) {
                            float absVal = aicAbs(qFloatBuf[r * p.dp + c]);
                            if (absVal > amax) amax = absVal;
                        }
                    }
                    const Sa2QuantScaleReciprocal qRecip(amax, p.qFactor);

                    for (int64_t c = segStart; c < segStart + kSegCols; ++c) {
                        if (r < validRows && c < p.d) {
                            q8L1[r * p.dp + c] = qRecip.Quantize(qFloatBuf[r * p.dp + c]);
                        } else {
                            q8L1[r * p.dp + c] = 0;
                        }
                    }
                    qScaleL1[r * numSegs + seg] = Sa2ScalarFromFloat<half>(qRecip.scaledFactor);
                }
            }
        }

        __ubuf__ float *rowStat = ub_.GetStateRowStat(streamId);
        __ubuf__ float *denomSlot = ub_.GetStateDenom(streamId);
        __ubuf__ float *oAcc = ub_.GetOAcc();
        for (int64_t r = 0; r < kStreamRows; ++r) {
            rowStat[r * 2 + 0] = kFp16NegMax;
            rowStat[r * 2 + 1] = 0.0f;
            denomSlot[r] = 0.0f;
        }

        for (int64_t i = 0; i < kStreamRows * p.dp; ++i) {
            oAcc[i] = 0.0f;
        }

        // P1-2 任务切换竞态闭环：q8/qScale 已落 L1（含 O 状态复位完成），放行 AIC 该任务首步 QK。
        // 计数对账：每任务 1 set（此处）/1 wait（StepCubeQk step==0），FinishTile 无需 drain。
        NotifyV(streamId, Sa2CrossFlag::kQ8Ok);
    }


    /**
     * @brief M2c-B3 AIV 负责：K-scale 双缓冲异步预取（与 kEvScale0/1 本地事件映射）
     */
    __aicore__ inline void PrefetchKScale(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        PrefetchKScaleDevice(task, step, streamId);
    }

    __aicore__ inline void PrefetchKScaleDevice(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        (void)streamId;
        if (step >= task.steps) {
            return;
        }
        int64_t parity = step % 2;
        if (step >= 2) {
            AwaitScaleConsumed(parity);
        }
        const Sa2Params &p = td_->params;
        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        __gm__ half *wsKScaleBase = reinterpret_cast<__gm__ half *>(
            userWs_ + td_->kScale + kvView.wsKScale);
        __gm__ half *kScaleStep = wsKScaleBase + step * kKvStep * p.segWidth;

        AscendC::LocalTensor<half> kScaleUb =
            UbTensor<half>(Sa2DeviceRes<G>::KScaleBase(parity));
        AscendC::GlobalTensor<half> gmKScale;
        gmKScale.SetGlobalBuffer(kScaleStep);
        AscendC::DataCopy(kScaleUb, gmKScale, static_cast<uint32_t>(kKvStep * p.segWidth));
        AscendC::PipeBarrier<PIPE_MTE2>();
        NotifyScaleLoaded(parity);
    }

    /**
     * @brief AIV 负责：等待 AIC QK 完成，执行在线 Softmax（含 θ 阈值跳过），生成 P 写入 L1 并通知 AIC
     */
    __aicore__ inline void StepVectorSoftmax(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        StepVectorSoftmaxDevice(task, step, streamId);
    }

    /**
     * @brief AIV 负责：等待 AIC QK 完成，执行在线 Softmax（含 θ 阈值跳过），生成 P 写入 L1 并通知 AIC。
     * M2b-0 wrapper：事件、GM/MTE 搬运（kh 行）、L1 标量暂存（qScale）与 L1 NZ 散写在此侧，
     * 纯向量指令序列下沉至自由函数 Sa2VfSoftmaxRun（__simd_vf__ 约束见其注记）。
     */
    __aicore__ inline void StepVectorSoftmaxDevice(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        int64_t parity = step % 2;
        CrossEnginePending<void, Sa2CrossFlag::kScoreOk> pendingScore(nullptr, 0, streamId);
        auto readyScore = aicMove(pendingScore).Await();
        (void)readyScore;

        const Sa2Params &p = td_->params;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const int64_t validRows = QTileValidRows(p, task.m0);
        const int64_t kvTokenStart = step * kKvStep;
        const half *qScalePtr = l1_.GetQScaleSlot(streamId);

        constexpr int64_t kScr = Sa2DeviceRes<G>::kScratchBase;
        __ubuf__ uint8_t *ubBase = ub_.base;
        __ubuf__ float *scalarSlot = reinterpret_cast<__ubuf__ float *>(
            reinterpret_cast<uint64_t>(ubBase) + kScr + 1280);
        __ubuf__ half *kScaleStage = ub_.GetKScaleSlot(parity);
        // qh f32 展开行（scratch +512，256B；见 Sa2VfSoftmaxArgs 注记——AIV 后端禁标量位
        // 强转，half 位型 → f32 的展开只能在 wrapper（普通 __aicore__ 语境）完成）
        __ubuf__ float *qhF32Stage = reinterpret_cast<__ubuf__ float *>(
            reinterpret_cast<uint64_t>(ubBase) + kScr + 512);

        // qScale 行 L1→UB 标量暂存（VF 形参禁 L1 指针；AIV 直读 L1 的硬件语义 = 950PR 核实项）。
        // qh 以 f32 位型经 u32 通道写入（AIV 单元禁「计算值位强转」，见 Sa2HalfBitsToF32Bits
        // 注记）；VF 侧按 float 直接装载重建（bitcast(load) 可选中，探针 p33e）。
        __ubuf__ uint32_t *qhBitsStage = reinterpret_cast<__ubuf__ uint32_t *>(qhF32Stage);
        for (int64_t r = 0; r < kStreamRows; ++r) {
            half h = qScalePtr[r];
            uint16_t qhHalfBits;
            std::memcpy(&qhHalfBits, &h, sizeof(qhHalfBits));
            qhBitsStage[r] = Sa2HalfBitsToF32Bits(qhHalfBits);
        }

        // 等待异步 PrefetchKScaleDevice 将 Scale 加载完成
        AwaitScaleLoaded(parity);

        Sa2VfSoftmaxArgs args{};
        args.rowStat = ub_.GetStateRowStat(streamId);
        args.denomSlot = ub_.GetStateDenom(streamId);
        args.oAcc = ub_.GetOAcc();
        args.alphaSlot = ub_.GetStateAlpha(streamId, parity);
        args.pSlotUb = ub_.GetPSlot(parity);
        args.scoreWarp = isThreadGran ? nullptr : ub_.GetScoreSlotWarp(parity);
        args.scoreThread = isThreadGran ? ub_.GetScoreSlotThread(parity) : nullptr;
        args.kScaleRow = kScaleStage;
        args.qScaleRow = reinterpret_cast<__ubuf__ half *>(qhF32Stage);
        args.qhF32 = qhF32Stage;
        args.scalarSlot = scalarSlot;
        args.validRows = validRows;
        args.m0 = task.m0;
        args.kvTokenStart = kvTokenStart;
        args.sk = p.sk;
        args.dp = p.dp;
        args.lnPDomain = p.lnPDomain;
        args.isThreadGran = isThreadGran;

        // VF 调用点必须落在自由函数（成员上下文调用 = AIV pass 后端硬错误，见 thunk 注记）
        if constexpr (IsCausal) {
            Sa2VfSoftmaxInvokeC<true>(args);
        } else {
            Sa2VfSoftmaxInvokeC<false>(args);
        }

        // Vector 消费 Scale 完成，释放该 slot 供后续步预取
        NotifyScaleConsumed(parity);

        // L1 NZ 分形打包写入：16×16 分形、行块主序，4 行块 × 8 列块 = 32 块；
        // 分形行 32B = 16 码 + 16B pad，以 16 字节对齐块（两个 64 位无符号整数字）直接搬运，
        // 彻底根除逐字节标量寻址开销，与 PNzFractalIndex 冻结映射数学等价。
        __ubuf__ const uint8_t *pSlotUbR = ub_.GetPSlot(parity);
        uint8_t *pSlotL1 = l1_.GetPNzSlot(streamId);
        for (int64_t r = 0; r < validRows; ++r) {
            __ubuf__ const uint8_t *pRow = pSlotUbR + r * kKvStep;
            const int64_t rowBlk = r / 16;
            const int64_t inRow = r % 16;
            for (int64_t colBlk = 0; colBlk < 8; ++colBlk) {
                const int64_t dstOffset = ((rowBlk * 8) + colBlk) * 512 + inRow * 32;
                *reinterpret_cast<uint64_t *>(pSlotL1 + dstOffset) =
                    *reinterpret_cast<const uint64_t *>(pRow + colBlk * 16);
                *reinterpret_cast<uint64_t *>(pSlotL1 + dstOffset + 8) =
                    *reinterpret_cast<const uint64_t *>(pRow + colBlk * 16 + 8);
            }
        }

        NotifyV(streamId, Sa2CrossFlag::kQkFree);
        NotifyMte3(streamId, Sa2CrossFlag::kPTaken);
    }

    /**
     * @brief AIV 负责：等待 AIC PV 完成，将部分和折叠进 O 累加器并释放 PV 槽
     */
    __aicore__ inline void StepVectorPvFold(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        StepVectorPvFoldDevice(task, step, streamId);
    }

    /**
     * @brief AIV 负责：等待 AIC PV 完成，将部分和折叠进 O 累加器并释放 PV 槽。
     * M2b-0 wrapper：类型状态槽（AcquireSlot/SlotData/ReleaseSlot，迭代 9 示范迁移）在
     * wrapper 侧，纯向量折叠下沉至 Sa2VfFoldRun（单步档）。
     */
    __aicore__ inline void StepVectorPvFoldDevice(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        int64_t parity = step % 2;
        const Sa2Params &p = td_->params;

        // M2.7.4 类型状态全周期示范迁移（迭代 9，唯一示范点；其余调用点迁移登记 todo M2b）：
        //   Empty --AcquireSlot<kPvOk>（内部 AwaitFix(kPvOk)，AIC FIX 生产 → V 消费，C.4 口径，
        //   事件语义与位置 = 原 CrossEnginePending::Await 不变）--> Full；
        //   数据访问唯一原语 = SlotData（Empty 态无访问器，编译期杜绝未同步读）；
        //   消费完毕 ReleaseSlot<kPvFree>（内部 NotifyV(kPvFree)，= 原函数尾散置 Notify，位置不变）
        //   --> Empty。旗号对账不变：每步 1×kPvOk Await + 1×kPvFree Notify（1:1 守恒）。
        //   其余跨核调用点（Softmax score 槽、AIC Cube 轨散置 Await/Notify、FinishTileCube
        //   drain）渐进迁移挂 M2b；宿主仿真轨（sim/，单线程零事件）无需槽位状态。
        auto pvFull = AcquireSlot<Sa2CrossFlag::kPvOk>(
            reinterpret_cast<void *>(reinterpret_cast<uint64_t>(ub_.GetPvSlot(parity))),
            p.dp * kStreamRows * static_cast<int64_t>(sizeof(float)), streamId);
        // 槽体在 AIV UB（AIC fixpipe 目的侧）；SlotData 擦除地址空间，消费点按 UB 重限定
        __ubuf__ float *pvSlot = reinterpret_cast<__ubuf__ float *>(
            reinterpret_cast<uint64_t>(SlotData<float>(pvFull)));

        Sa2VfFoldArgs args{};
        args.pvSlot0 = pvSlot;
        args.pvSlot1 = nullptr;
        args.oAcc = ub_.GetOAcc();
        args.alpha1 = nullptr;
        args.validRows = QTileValidRows(p, task.m0);
        args.dp = p.dp;
        args.fp16Accum = (p.accum == static_cast<int64_t>(PvAccum::kFp32Fp16));
        args.paired = false;
        Sa2VfFoldInvoke(args);

        ReleaseSlot<Sa2CrossFlag::kPvFree>(aicMove(pvFull));
    }

    /**
     * @brief AIV 负责：M2c-A2 配对 PV 聚合（Paired PV Aggregation 机制）
     * 将相邻两个 PV Tile 步 (step0, step1) 配对，在 Vector 核心内部执行融合加法折叠，
     * 使 Vector 核心的 O 累加与归约迭代次数减少 50%。
     * 完整保持全部规格优势：支持 per_thread 与 per_warp 双粒度、三档累加（fp32, fp32+fp32, fp32+fp16 κ=2.25 封顶）、
     * FlashMLA θ-skip 阈值跳过与闭环 LSE。
     */
    __aicore__ inline void StepVectorPvFoldPairDevice(
        const Sa2Task &task, int64_t step0, int64_t step1, int64_t streamId)
    {
        int64_t parity0 = step0 % 2;
        int64_t parity1 = step1 % 2;
        const Sa2Params &p = td_->params;

        // 依序获取相邻两步的 PV 槽（AcquireSlot 消费 kPvOk 跨核旗号；地址经整数过桥保 UB 值）
        auto pvFull0 = AcquireSlot<Sa2CrossFlag::kPvOk>(
            reinterpret_cast<void *>(reinterpret_cast<uint64_t>(ub_.GetPvSlot(parity0))),
            p.dp * kStreamRows * static_cast<int64_t>(sizeof(float)), streamId);
        auto pvFull1 = AcquireSlot<Sa2CrossFlag::kPvOk>(
            reinterpret_cast<void *>(reinterpret_cast<uint64_t>(ub_.GetPvSlot(parity1))),
            p.dp * kStreamRows * static_cast<int64_t>(sizeof(float)), streamId);

        __ubuf__ float *pvSlot0 = reinterpret_cast<__ubuf__ float *>(
            reinterpret_cast<uint64_t>(SlotData<const float>(pvFull0)));
        __ubuf__ float *pvSlot1 = reinterpret_cast<__ubuf__ float *>(
            reinterpret_cast<uint64_t>(SlotData<const float>(pvFull1)));

        Sa2VfFoldArgs args{};
        args.pvSlot0 = pvSlot0;
        args.pvSlot1 = pvSlot1;
        args.oAcc = ub_.GetOAcc();
        args.alpha1 = ub_.GetStateAlpha(streamId, parity1);
        args.validRows = QTileValidRows(p, task.m0);
        args.dp = p.dp;
        args.fp16Accum = (p.accum == static_cast<int64_t>(PvAccum::kFp32Fp16));
        args.paired = true;
        Sa2VfFoldInvoke(args);

        // 分别释放两步的 PV 槽，各回退 1 个 kPvFree 信用，维持严格 1:1 计数守恒
        ReleaseSlot<Sa2CrossFlag::kPvFree>(aicMove(pvFull0));
        ReleaseSlot<Sa2CrossFlag::kPvFree>(aicMove(pvFull1));
    }

    /**
     * @brief AIV 负责：Epilogue 输出归一化、smooth_v 补偿及 GM 写回与 LSE 闭环
     */
    __aicore__ inline void FinishTileVector(const Sa2Task &task, int64_t streamId)
    {
        FinishTileVectorDevice(task, streamId);
    }

    /**
     * @brief AIV 负责：Epilogue 输出归一化、smooth_v 补偿及 GM 写回与 LSE 闭环。
     * M2b-0 wrapper：GM 标量写回（含 dtype 转换与 LSE 落地）在此侧——GM 无向量写通路
     *（StoreAlign 仅 UB；MTE DataCopy 需 32B 对齐，裁剪尾 d=48 不满足），设备标量 dtype
     * 转换同为 RNE，与 oracle 的 vapi::Cast(CAST_RINT)/static_cast<Dtype> 逐元素语义一致；
     * 纯向量归一化/LSE 闭环下沉至 Sa2VfFinishRun。
     */
    __aicore__ inline void FinishTileVectorDevice(const Sa2Task &task, int64_t streamId)
    {
        const Sa2Params &p = td_->params;
        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        __gm__ const float *wsVMean = reinterpret_cast<__gm__ const float *>(
            userWs_ + td_->vMean + kvView.wsVMean);
        const int64_t validRows = QTileValidRows(p, task.m0);
        const bool smoothVActive = p.smoothV && (p.accum == static_cast<int64_t>(PvAccum::kFp32));

        const int64_t outElemOffset = OutTileOffset(p, task.b, task.qh, task.m0);
        __gm__ Dtype *outPtr = reinterpret_cast<__gm__ Dtype *>(outGmBase_) + outElemOffset;

        constexpr int64_t kScr = Sa2DeviceRes<G>::kScratchBase;
        __ubuf__ uint8_t *ubBase = ub_.base;
        __ubuf__ float *stageRow =
            reinterpret_cast<__ubuf__ float *>(reinterpret_cast<uint64_t>(ubBase) + kScr);
        __ubuf__ float *vMeanStage =
            reinterpret_cast<__ubuf__ float *>(reinterpret_cast<uint64_t>(ubBase) + kScr + 256);
        __ubuf__ float *vMeanRow = nullptr;
        if (smoothVActive && wsVMean != nullptr) {
            // smooth_v 均值行 GM→UB 标量暂存（d ≤ 128 floats = 512B，VF 形参禁 GM 指针）
            for (int64_t c = 0; c < p.d; ++c) {
                vMeanStage[c] = wsVMean[c];
            }
            vMeanRow = vMeanStage;
        }

        __ubuf__ float *mStage = nullptr;
        if (p.returnLse && lseGmBase_ != nullptr && validRows > 0) {
            // 跨步 rowStat（lane 步长 2）装配为连续行供单寄存器装载
            __ubuf__ float *mStageW = ub_.GetStateMStage(streamId);
            __ubuf__ float *rowStat = ub_.GetStateRowStat(streamId);
            for (int64_t r = 0; r < validRows; ++r) {
                mStageW[r] = rowStat[r * 2 + 0];
            }
            mStage = mStageW;
        }

        Sa2VfFinishArgs args{};
        args.oAcc = ub_.GetOAcc();
        args.denom = ub_.GetStateDenom(streamId);
        args.vMeanRow = vMeanRow;
        args.mStage = mStage;
        args.corr = ub_.GetStateCorr(streamId);
        args.stageRow = stageRow;
        args.validRows = validRows;
        args.d = p.d;
        args.dp = p.dp;
        args.smScale = p.smScale;
        Sa2VfFinishInvoke(args);

        // GM 向量化块写回：O 归一化结果以 16 字节（8 元素）块猝发写 GM，彻底消除逐元素标量访存
        __ubuf__ float *oAcc = ub_.GetOAcc();
        for (int64_t r = 0; r < validRows; ++r) {
            __gm__ Dtype *outRow = outPtr + r * p.qStride[2];
            __ubuf__ float *oRow = oAcc + r * p.dp;
            int64_t c = 0;
            for (; c + 8 <= p.d; c += 8) {
                Dtype tmp[8];
                for (int64_t i = 0; i < 8; ++i) {
                    tmp[i] = Sa2ScalarFromFloat<Dtype>(oRow[c + i]);
                }
                *reinterpret_cast<__gm__ uint64_t *>(outRow + c) =
                    *reinterpret_cast<const uint64_t *>(tmp);
                *reinterpret_cast<__gm__ uint64_t *>(outRow + c + 4) =
                    *reinterpret_cast<const uint64_t *>(tmp + 4);
            }
            for (; c < p.d; ++c) {
                outRow[c] = Sa2ScalarFromFloat<Dtype>(oRow[c]);
            }
        }

        if (mStage != nullptr) {
            const int64_t lseOffset = LseOffset(p, task.b, task.qh, task.m0);
            __gm__ float *lsePtr = reinterpret_cast<__gm__ float *>(lseGmBase_) + lseOffset;
            int64_t r = 0;
            for (; r + 4 <= validRows; r += 4) {
                *reinterpret_cast<__gm__ uint64_t *>(lsePtr + r) =
                    *reinterpret_cast<const uint64_t *>(stageRow + r);
                *reinterpret_cast<__gm__ uint64_t *>(lsePtr + r + 2) =
                    *reinterpret_cast<const uint64_t *>(stageRow + r + 2);
            }
            for (; r < validRows; ++r) {
                lsePtr[r] = stageRow[r];
            }
        }
    }

    // ============================ AIC Cube 引擎流水线 ============================

    /**
     * @brief AIC 负责：调度单个 step 的 INT8 QK^T MMAD GEMM，经 Fixpipe 写入 UB score 槽
     */
    __aicore__ inline void StepCubeQk(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        StepCubeQkDevice(task, step, streamId);
    }

    __aicore__ inline void StepCubeQkDevice(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        if (step == 0) {
            // P1-2：等 AIV QQuant(t) 完成（kQ8Ok），杜绝读到上一任务陈旧 q8 或半写状态
            AwaitMte1(streamId, Sa2CrossFlag::kQ8Ok);
        }
        if (step >= 2) {
            AwaitV(streamId, Sa2CrossFlag::kQkFree);
        }

        const Sa2Params &p = td_->params;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        int64_t parity = step % 2;

        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        __gm__ int8_t *wsK8Base = reinterpret_cast<__gm__ int8_t *>(
            userWs_ + td_->k8 + kvView.wsK8);
        __gm__ int8_t *k8Step = wsK8Base + step * kKvStep * p.dp;

        // 1. 搬运 Q8 从 L1 到 L0A (MTE1)。M2b-0：真实 LocalTensor 无 (ptr,size,pos) 构造
        //（hoststub 假想 API），一律经 TBuf::Get()[字节偏移].ReinterpretCast<T>() 构建，
        // 位置语义由 TBuf 模板参数携带；tla 布局 = 标签制 MakeLayout<E, zN>(rows, cols)。
        AscendC::LocalTensor<int8_t> l1QTensor = L1TensorI8(Sa2DeviceRes<G>::L1Q8Base(streamId));
        auto layoutQInL1 = tla::MakeLayout<int8_t, Catlass::layout::zN>(kStreamRows, p.dp);
        auto tensorQInL1 = tla::MakeTensor(l1QTensor, layoutQInL1, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL1{});

        AscendC::LocalTensor<int8_t> l0aQTensor = L0aTensorI8();
        auto layoutQInL0A = tla::MakeLayout<int8_t, Catlass::layout::zN>(kStreamRows, p.dp);
        auto tensorQInL0A = tla::MakeTensor(l0aQTensor, layoutQInL0A, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL0A{});

        Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, decltype(tensorQInL1), decltype(tensorQInL0A)> copyQToL0A;
        copyQToL0A(tensorQInL0A, tensorQInL1);
        AscendC::PipeBarrier<PIPE_MTE1>();

        // 2. 搬运 K8：GM -> L1 (MTE2) -> L0B (MTE1)（应用 M2c-B1 L2 Cache Hint 常驻策略）。
        //    GlobalTensor 无 (ptr,size) 构造：默认构造 + SetGlobalBuffer(__gm__ 裸指针直入，
        //    禁止跨地址空间 const/reinterpret 转换）。
        AscendC::GlobalTensor<int8_t> gmKTensor;
        gmKTensor.SetGlobalBuffer(k8Step, static_cast<uint64_t>(kKvStep * p.dp));
        gmKTensor.SetL2CacheHint(Sa2CacheModeOf(Sa2BufferCachePolicy::kK8Hint));
        AscendC::LocalTensor<int8_t> l1KTensor =
            L1TensorI8(parity == 0 ? kL1KStep0 : kL1KStep1);
        AscendC::DataCopy(l1KTensor, gmKTensor, static_cast<uint32_t>(kKvStep * p.dp));
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::LocalTensor<int8_t> l0bKTensor = L0bTensorI8();
        auto layoutKInL1 = tla::MakeLayout<int8_t, Catlass::layout::zN>(kKvStep, p.dp);
        auto tensorKInL1 = tla::MakeTensor(l1KTensor, layoutKInL1, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL1{});
        auto layoutKInL0B = tla::MakeLayout<int8_t, Catlass::layout::nZ>(kKvStep, p.dp);
        auto tensorKInL0B = tla::MakeTensor(l0bKTensor, layoutKInL0B, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL0B{});

        Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, decltype(tensorKInL1), decltype(tensorKInL0B)> copyKToL0B;
        copyKToL0B(tensorKInL0B, tensorKInL1);
        AscendC::PipeBarrier<PIPE_MTE1>();

        // 3. L0C 象限映射与 MMAD 乘加
        AscendC::LocalTensor<int32_t> l0cTensor =
            L0cTensorI32(Sa2DeviceRes<G>::L0cQuadBase(streamId, parity));
        auto layoutL0C = tla::MakeLayoutL0C(kStreamRows, kKvStep);
        auto tensorL0C = tla::MakeTensor(l0cTensor, layoutL0C, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL0C{});

        Catlass::Gemm::Tile::TileMmadTla<Catlass::Arch::Ascend950, int8_t, Catlass::layout::zN> tileMmad;
        tileMmad(tensorL0C, tensorQInL0A, tensorKInL0B,
                 static_cast<uint32_t>(kStreamRows), static_cast<uint32_t>(kKvStep), static_cast<uint32_t>(p.dp), true);
        AscendC::PipeBarrier<PIPE_M>();

        // 4. Fixpipe 硬件反量化重排写入 UB（catlass copy_l0c_to_ub.hpp 同款三模板参 +
        //    FixpipeConfig{ROW_MAJOR, isToUB=true} 形态）
        if constexpr (!isThreadGran) {
            AscendC::LocalTensor<half> ubScoreTensor =
                UbTensor<half>(Sa2DeviceRes<G>::ScoreBase(parity));
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.mSize = static_cast<uint16_t>(kStreamRows);
            fixpipeParams.nSize = static_cast<uint16_t>(kKvStep);
            fixpipeParams.srcStride = static_cast<uint16_t>(kKvStep / 16);
            fixpipeParams.dstStride = static_cast<uint32_t>(kKvStep);
            fixpipeParams.quantPre = QuantMode_t::DEQF16;
            // ⚠ M2b 核实项（todo：Fixpipe DEQ 模式枚举核实，D-R13-(1) 前提）：真实
            // deqScalar 载体为 uint64_t（hoststub 假想为 float），950 fixpipe 的 DEQ 标量
            // 编码（fp32 位型/双 fp16 打包/移位格式）未经真机核实，暂以 2^-14 的 fp32 位型装入。
            const float kSa2DeqScalar = 1.0f / 16384.0f;
            fixpipeParams.deqScalar =
                static_cast<uint64_t>(*reinterpret_cast<const uint32_t *>(&kSa2DeqScalar));
            AscendC::Fixpipe<half, int32_t, kSa2CfgRowMajorUb>(ubScoreTensor, l0cTensor, fixpipeParams);
        } else {
            // per_thread 硬件反量化重排写入 UB 分段槽（§4.2 组合式落地）：
            // 施加标量 deqScalar = 2^-14，输出至 UB 分段槽 (kUbSegSlot / ScoreSlot)
            // 由 AIV Vector 核心在 Softmax 前段按 16 列分段融合 (qh_seg * kh_seg) 严格累进
            AscendC::LocalTensor<float> ubScoreTensor =
                UbTensor<float>(Sa2DeviceRes<G>::ScoreBase(parity));
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.mSize = static_cast<uint16_t>(kStreamRows);
            fixpipeParams.nSize = static_cast<uint16_t>(kKvStep);
            fixpipeParams.srcStride = static_cast<uint16_t>(kKvStep / 16);
            fixpipeParams.dstStride = static_cast<uint32_t>(kKvStep);
            AscendC::LocalTensor<float> l0cTensorAsFloat = l0cTensor.template ReinterpretCast<float>();
            const float kSa2DeqScalar = 1.0f / 16384.0f;
            fixpipeParams.deqScalar =
                static_cast<uint64_t>(*reinterpret_cast<const uint32_t *>(&kSa2DeqScalar));
            AscendC::Fixpipe<float, float, kSa2CfgRowMajorUb>(ubScoreTensor, l0cTensorAsFloat, fixpipeParams);
        }
        AscendC::PipeBarrier<PIPE_FIX>();

        NotifyFix(streamId, Sa2CrossFlag::kScoreOk);
    }

    /**
     * @brief AIC 负责：等待 P 矩阵就绪，调度单个 step 的 FP8 PV MMAD GEMM，写入 UB PV 槽
     */
    __aicore__ inline void StepCubePv(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        StepCubePvDevice(task, step, streamId);
    }

    __aicore__ inline void StepCubePvDevice(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        int64_t parity = step % 2;
        AwaitMte1(streamId, Sa2CrossFlag::kPTaken);
        if (step >= 2) {
            AwaitV(streamId, Sa2CrossFlag::kPvFree);
        }

        const Sa2Params &p = td_->params;
        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        __gm__ int8_t *wsV8Base = reinterpret_cast<__gm__ int8_t *>(
            userWs_ + td_->v8 + kvView.wsV8);
        __gm__ int16_t *wsVScaleBase = reinterpret_cast<__gm__ int16_t *>(
            userWs_ + td_->vScale + kvView.wsVScale);
        __gm__ int8_t *v8Step = wsV8Base + step * kKvStep * p.dp;

        // 1. 搬运 P (FP8 E4M3) 从 L1 到 L0A (MTE1)。M2b-0：张量构建走 TBuf 链（见 StepCubeQk）。
        AscendC::LocalTensor<float8_e4m3_t> l1PTensor =
            L1TensorF8(streamId == 0 ? kL1PNz0 : kL1PNz1);
        auto layoutPInL1 = tla::MakeLayout<float8_e4m3_t, Catlass::layout::zN>(kStreamRows, kKvStep);
        auto tensorPInL1 = tla::MakeTensor(l1PTensor, layoutPInL1, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL1{});

        AscendC::LocalTensor<float8_e4m3_t> l0aPTensor = L0aTensorF8();
        auto layoutPInL0A = tla::MakeLayout<float8_e4m3_t, Catlass::layout::zN>(kStreamRows, kKvStep);
        auto tensorPInL0A = tla::MakeTensor(l0aPTensor, layoutPInL0A, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL0A{});

        Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, decltype(tensorPInL1), decltype(tensorPInL0A)> copyPToL0A;
        copyPToL0A(tensorPInL0A, tensorPInL1);
        AscendC::PipeBarrier<PIPE_MTE1>();

        // 2. 搬运 V8 与 UE8M0 VScale：GM -> L1 (MTE2) -> L0B (MTE1，带 MX 尺度展开)
        // 应用 M2c-B1 L2 Cache Hint 常驻策略（CACHE_HINT_NORMAL_LAST_VICTIM）
        // MX 伴尺度搬运（copy_l1_to_l0b 三参特化）的类型契约：TensorSrc(L1) = 普通 fp8 族、
        // TensorDst(L0B) = mx_* 族、scale = float8_e8m0_t nN；L1 侧保持 float8_e4m3_t，
        // L0B 侧改用 mx_fp8_e4m3_t（硬件 MX LoadData 在 L0B 展开尺度）。
        AscendC::GlobalTensor<float8_e4m3_t> gmVTensor;
        gmVTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e4m3_t *>(v8Step),
                                  static_cast<uint64_t>(kKvStep * p.dp));
        gmVTensor.SetL2CacheHint(Sa2CacheModeOf(Sa2BufferCachePolicy::kV8Hint));
        AscendC::LocalTensor<float8_e4m3_t> l1VTensor =
            L1TensorF8(parity == 0 ? kL1VStep0 : kL1VStep1);
        AscendC::DataCopy(l1VTensor, gmVTensor, static_cast<uint32_t>(kKvStep * p.dp));

        AscendC::LocalTensor<float8_e8m0_t> l1VScaleTensor =
            L1TensorF8E8M0(parity == 0 ? kL1VScale0 : kL1VScale1);

        // M2c-B4 V MX Scale L1 缓存命中检查：
        const int64_t totalScaleBytes = (p.kp * p.dp) / 32;
        const bool canCacheHead = (totalScaleBytes <= kL1MxScaleCacheBytes);
        const int64_t stepScaleOffset = step * (kKvStep / 32) * p.dp;

        if (canCacheHead && mxScaleCache_.IsHit(task.b, task.kh, wsVScaleBase)) {
            // L1 缓存命中：直接从 Head 级 L1 缓存区复制当前 step 的尺度张量到双缓冲槽
            AscendC::LocalTensor<float8_e8m0_t> cachedScaleTensor =
                L1TensorF8E8M0(kL1MxScaleCacheBase + stepScaleOffset);
            AscendC::DataCopy(l1VScaleTensor, cachedScaleTensor, static_cast<uint32_t>(kL1VScaleSlot));
        } else {
            // 未命中或超容量：从 GM 加载并附带 CACHE_HINT_NORMAL_LAST_VICTIM
            AscendC::GlobalTensor<float8_e8m0_t> gmVScaleTensor;
            gmVScaleTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e8m0_t *>(wsVScaleBase),
                                          static_cast<uint64_t>(kL1VScaleSlot));
            gmVScaleTensor.SetL2CacheHint(Sa2CacheModeOf(Sa2BufferCachePolicy::kScaleHint));
            AscendC::DataCopy(l1VScaleTensor, gmVScaleTensor, static_cast<uint32_t>(kL1VScaleSlot));

            if (canCacheHead && step == 0) {
                // 首步预热本 Head 的全部 L1 缓存
                AscendC::GlobalTensor<float8_e8m0_t> gmFullScaleTensor;
                gmFullScaleTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e8m0_t *>(wsVScaleBase),
                                                  static_cast<uint64_t>(totalScaleBytes));
                gmFullScaleTensor.SetL2CacheHint(Sa2CacheModeOf(Sa2BufferCachePolicy::kScaleHint));
                AscendC::LocalTensor<float8_e8m0_t> cacheBufTensor =
                    L1TensorF8E8M0(kL1MxScaleCacheBase);
                AscendC::DataCopy(cacheBufTensor, gmFullScaleTensor, static_cast<uint32_t>(totalScaleBytes));
                mxScaleCache_.MarkCached(task.b, task.kh, wsVScaleBase, totalScaleBytes);
            }
        }
        AscendC::PipeBarrier<PIPE_MTE2>();

        auto layoutVInL1 = tla::MakeLayout<float8_e4m3_t, Catlass::layout::zN>(kKvStep, p.dp);
        auto tensorVInL1 = tla::MakeTensor(l1VTensor, layoutVInL1, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL1{});

        int64_t mxScaleK = kKvStep / 32;
        auto layoutVScaleL1 = tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::nN, true>(mxScaleK, p.dp);
        auto tensorVScaleL1 = tla::MakeTensor(l1VScaleTensor, layoutVScaleL1, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL1{});

        // L0B 侧保持普通 fp8：MMAD(mad) 的 3 参 dtype 契约 = Tuple<float, fp8, fp8>（dav_3510
        // MmadCal isMx 表之外的普通路径），MX 尺度由 LoadData2DMxParams 随搬运写入 L0B 专用
        // MX 存储区（load_cbuf_to_cb_mx），数据体不需 mx_* 类型声明。
        AscendC::LocalTensor<float8_e4m3_t> l0bVTensor = L0bTensorF8();
        auto layoutVInL0B = tla::MakeLayout<float8_e4m3_t, Catlass::layout::nZ>(kKvStep, p.dp);
        auto tensorVInL0B = tla::MakeTensor(l0bVTensor, layoutVInL0B, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL0B{});

        // M2b-0：V8 + MX scale 的 L1→L0B 搬运按 catlass copy_l1_to_l0b.hpp 三参特化首分支
        // 蓝图直调 LoadData（该特化在本 catlass 版本不可达：类级 enable_if 限定 dst ∈ B8 族，
        // 其 3 参算子 static_assert 又要求 dst ∈ mx 族，模板选择互斥；首分支恒可走——
        // L0N = dp ∈ {64,128} 均 32 整除）。元素契约循蓝图：src(L1 zN) = float8_e4m3_t、
        // dst(L0B nZ) = mx_fp8_e4m3_t（硬件在 L0B 侧展开 MX 尺度）、scale(L1 nN) = e8m0。
        {
            const uint32_t l0k = static_cast<uint32_t>(tla::get<0, 0>(tensorVInL0B.shape())) *
                                 static_cast<uint32_t>(tla::get<0, 1>(tensorVInL0B.shape()));
            const uint32_t l0n = static_cast<uint32_t>(tla::get<1, 0>(tensorVInL0B.shape())) *
                                 static_cast<uint32_t>(tla::get<1, 1>(tensorVInL0B.shape()));
            const uint32_t srcOuterStrideCol =
                static_cast<uint32_t>(tla::get<1, 1>(tensorVInL1.stride()));
            const uint32_t dstOuterStrideRow =
                static_cast<uint32_t>(tla::get<0, 1>(tensorVInL0B.stride()));
            auto vSrcCoord = tensorVInL1.coord();
            (void)vSrcCoord;
            auto vDstOffset = tensorVInL0B.layout()(tensorVInL0B.coord());
            auto vScaleCoord = tensorVScaleL1.coord();

            AscendC::LoadData2DMxParams loadDataMxParams;
            loadDataMxParams.xStartPosition =
                tla::get<1>(vScaleCoord);
            loadDataMxParams.yStartPosition =
                tla::get<0>(vScaleCoord);
            loadDataMxParams.xStep = static_cast<uint32_t>(tla::get<1, 1>(tensorVScaleL1.shape()));
            loadDataMxParams.yStep = static_cast<uint32_t>(tla::get<0, 1>(tensorVScaleL1.shape()));
            loadDataMxParams.srcStride =
                (static_cast<uint32_t>(tla::get<1, 1>(tensorVScaleL1.stride())) + 31) / 32;  // CeilDiv<BYTE_PER_C0=32>
            loadDataMxParams.dstStride = static_cast<uint32_t>(tla::get<0, 1>(tensorVScaleL1.shape()));

            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = l0k / 16;  // CeilDiv<C0_NUM_PER_FRACTAL>(L0K)，C0=16 恒整除（128/64 行块）
            loadDataParams.kStep = l0n / 32;  // CeilDiv<ELE_NUM_PER_C0>(L0N)，fp8 C0=32 恒整除（dp 64/128）
            loadDataParams.srcStride = srcOuterStrideCol / 512;  // CeilDiv<ELE_NUM_PER_FRACTAL=512>
            loadDataParams.dstStride = dstOuterStrideRow / 512;
            loadDataParams.ifTranspose = true;

            AscendC::LoadData(tensorVInL0B.data()[vDstOffset], tensorVInL1.data(), tensorVScaleL1.data(),
                              loadDataParams, loadDataMxParams);
        }
        AscendC::PipeBarrier<PIPE_MTE1>();

        // 3. L0C 象限映射与 FP8 PV MMAD 乘加
        AscendC::LocalTensor<float> l0cTensor =
            L0cTensorF32(Sa2DeviceRes<G>::L0cQuadBase(streamId, parity));
        auto layoutL0C = tla::MakeLayoutL0C(kStreamRows, p.dp);
        auto tensorL0C = tla::MakeTensor(l0cTensor, layoutL0C, tla::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)), Catlass::Arch::PositionL0C{});

        Catlass::Gemm::Tile::TileMmadTla<Catlass::Arch::Ascend950, float8_e4m3_t, Catlass::layout::zN> tileMmad;
        tileMmad(tensorL0C, tensorPInL0A, tensorVInL0B,
                 static_cast<uint32_t>(kStreamRows), static_cast<uint32_t>(p.dp), static_cast<uint32_t>(kKvStep), true);
        AscendC::PipeBarrier<PIPE_M>();

        // 4. Fixpipe 拷贝至 UB PV 槽（三模板参 + FixpipeConfig 形态，同 StepCubeQk）
        AscendC::LocalTensor<float> ubPvTensor =
            UbTensor<float>(Sa2DeviceRes<G>::PvBase(parity));
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = static_cast<uint16_t>(kStreamRows);
        fixpipeParams.nSize = static_cast<uint16_t>(p.dp);
        fixpipeParams.srcStride = static_cast<uint16_t>(p.dp / 16);
        fixpipeParams.dstStride = static_cast<uint32_t>(p.dp);
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<float, float, kSa2CfgRowMajorUb>(ubPvTensor, l0cTensor, fixpipeParams);
        AscendC::PipeBarrier<PIPE_FIX>();

        NotifyFix(streamId, Sa2CrossFlag::kPvOk);
    }

    /**
     * @brief AIC 负责：Tile 任务结束时归还并平账未消费的旗号（严格 1:1 计数守恒，防硬件死锁）
     */
    __aicore__ inline void FinishTileCube(const Sa2Task &task, int64_t streamId)
    {
        FinishTileCubeDevice(task, streamId);
    }

    __aicore__ inline void FinishTileCubeDevice(const Sa2Task &task, int64_t streamId)
    {
        int64_t drain = aicMin(task.steps, static_cast<int64_t>(2));
        for (int64_t d = 0; d < drain; ++d) {
            AwaitV(streamId, Sa2CrossFlag::kQkFree);
            AwaitV(streamId, Sa2CrossFlag::kPvFree);
        }
    }

    // ============================ 完整 Tile 执行入口 ============================

    /**
     * @brief 设备 AIV 执行单个 query tile 的完整流水线（纯 AscendC Vector MicroAPI 硬件通路）
     * 采用 M2c-A2 配对 PV 聚合机制：将相邻两个 PV Tile 步配对，O 累加与归约迭代次数减少 50%
     * 结合 M2c-B3 K-scale 双缓冲流水预取：消除 Scale 搬运延迟
     */
    __aicore__ inline void ProcessTileVector(const Sa2Task &task, int64_t streamId)
    {
        if (!task.valid || task.steps <= 0) {
            return;
        }
        StepVectorQQuant(task, streamId);

        // M2c-B3: Seed 预取 step 0 与 step 1 双缓冲槽
        PrefetchKScaleDevice(task, 0, streamId);
        if (task.steps > 1) {
            PrefetchKScaleDevice(task, 1, streamId);
        }

        int64_t step = 0;
        for (; step + 1 < task.steps; step += 2) {
            StepVectorSoftmaxDevice(task, step, streamId);
            PrefetchKScaleDevice(task, step + 2, streamId);

            StepVectorSoftmaxDevice(task, step + 1, streamId);
            PrefetchKScaleDevice(task, step + 3, streamId);

            StepVectorPvFoldPairDevice(task, step, step + 1, streamId);
        }
        for (; step < task.steps; ++step) {
            StepVectorSoftmaxDevice(task, step, streamId);
            PrefetchKScaleDevice(task, step + 2, streamId);

            StepVectorPvFoldDevice(task, step, streamId);
        }
        FinishTileVectorDevice(task, streamId);
    }

    /**
     * @brief 兼容执行入口：分发至 Vector 流程
     */
    __aicore__ inline void ProcessTile(const Sa2Task &task, int64_t streamId)
    {
        ProcessTileVector(task, streamId);
    }

private:
    GM_ADDR qGmBase_{nullptr};
    GM_ADDR outGmBase_{nullptr};
    GM_ADDR lseGmBase_{nullptr};
    GM_ADDR userWs_{nullptr};
    const Sa2TilingData *td_{nullptr};

    AscendC::TPipe pipe_{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    AscendC::TBuf<AscendC::TPosition::A1> l1Buf_{};
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf_{};
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf_{};
    AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf_{};

    Sa2UbBuffer<G> ub_{};
    Sa2L1Buffer l1_{};
    Sa2L0cBuffer l0c_{};
    Sa2MxScaleL1Cache mxScaleCache_{};
};

} // namespace sa2

#endif // SA2_ATTENTION_LOOP_H_

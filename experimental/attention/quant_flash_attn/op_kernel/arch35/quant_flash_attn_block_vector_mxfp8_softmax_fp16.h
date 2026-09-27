/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_block_vector_mxfp8_softmax_fp16.h
 * \brief MxFP8 Softmax FP16 Vector Block：V1（softmax→fp8）+ V2（累积→除→转置→GM 输出）
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_

// ===== vf 头文件（编译冒烟：让编译器解析已实现的 vf 代码，提前暴露问题）=====
#include "vf/vf_common_def.h"
#include "vf/vf_calc_rescale_factor.h"
#include "vf/vf_calc_pscale.h"
#include "vf/vf_flashupdate_acc.h"
#include "vf/vf_div_cast_bf16.h"
#include "vf/vf_softmax_fp16_to_fp8.h"
// ===== GM 输出公共助手（镜像 cube 侧引用方式；copy_ub_to_gm.h 传递引入 fa_ub_tensor/gm_coord 等）=====
#include "../../../common/op_kernel/memcopy/copy_ub_to_gm.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/parser.h"

namespace QFA_KERNEL {

template <typename QUANT_T, typename SCALE_T, typename OUT_T>
class QuantFlashAttnBlockVectorMxfp8SoftmaxFp16 {
public:
    // ===== 类型别名（对齐 cube 侧命名）=====
    using MM_OUT_T = float;       // mm2Res fp32
    using SOFTMAX_T = half;       // 状态 half（log2 整数域）

    // ===== 编译期常量（基础尺寸，对齐 common_def）=====
    static constexpr uint32_t s1BaseSize = S1_BASE_SIZE;        // 256（S1 总宽，V0/V1 各持一半）
    static constexpr uint32_t s2SubLoopSize = S2_SUB_LOOP_SIZE; // 128（每 subLoop 的 mm1Res/P 行数，即 S2 方向每次处理的行数）
    static constexpr uint32_t dVBaseSize = DV_BASE_SIZE;        // 128（V2 输出的 DV 方向行数）
    static constexpr half MIN_HALF_VALUE = static_cast<half>(-65504.0f);  // fp16 最小有限负值

private:
    // ===== UB 尺寸常量（vector 侧自有；mm1Res/mm2Res 直接用 common_def 跨核契约）=====
    static constexpr uint32_t UB_VEC1P_SIZE = 2 * s2SubLoopSize * (s1BaseSize / 2); // 2 slots × [128,128] fp8
    static constexpr uint32_t UB_STAGE2OUT_SIZE = dVBaseSize * (s1BaseSize / 2); // [128,128] fp32
    static constexpr uint32_t UB_ACCROWSUM_SIZE = s1BaseSize / 2;           // [128] fp32
    static constexpr uint32_t UB_SMAX_SLOT = s1BaseSize / 2;                // [128] half
    static constexpr uint32_t UB_SMAX_BUFCNT = 3;                           // mloop%3
    static constexpr uint32_t UB_SEXP_BUFCNT = 3;                           // loop%3
    static constexpr uint32_t UB_PSCALE_SUBLOOPS = 2;                       // subLoopUsedMaxUB 槽数
    // pscale 网格影像（共享常量同源 vf_common_def，B1 教训）：[ring][768B]，ring 周期 6
    static constexpr uint32_t UB_PSCALE_SLOT = QfaVectorApi::QFA_UB_PSCALE_SLOT;    // 768B/槽
    static constexpr uint32_t UB_PSCALE_RINGS = QfaVectorApi::QFA_UB_PSCALE_RINGS;  // 6

    // ===== UB BASE 常量链（mm1Res/mm2Res 的 BASE 用 common_def 值，不自引用）=====
    static constexpr uint32_t UB_VEC1P_BASE = UB_MM2RES_BASE + UB_MM2RES_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_STAGE2OUT_BASE = UB_VEC1P_BASE + UB_VEC1P_SIZE * sizeof(QUANT_T);
    static constexpr uint32_t UB_ACCROWSUM_BASE = UB_STAGE2OUT_BASE + UB_STAGE2OUT_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SMAX_BASE = UB_ACCROWSUM_BASE + UB_ACCROWSUM_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SEXP_BASE = UB_SMAX_BASE + UB_SMAX_SLOT * UB_SMAX_BUFCNT * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_BSMAX_BASE = UB_SEXP_BASE + UB_SMAX_SLOT * UB_SEXP_BUFCNT * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SUBMAX_BASE = UB_BSMAX_BASE + UB_SMAX_SLOT * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_PSCALE_BASE = UB_SUBMAX_BASE + UB_SMAX_SLOT * UB_PSCALE_SUBLOOPS * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_NZIDX_BASE = UB_PSCALE_BASE + UB_PSCALE_SLOT * UB_PSCALE_RINGS;
    // 零批输出清零源（2026-09-27 seq_used=0 泛化，DEAL_ZERO TODO 落地）：
    // [64 行 × dVBaseSize 列] OUT_T 零源，分块整行清零 GM（colCount=dVBaseSize → srcStride=0 纯连续）
    static constexpr uint32_t UB_ZERO_OUT_ROWS = 64;
    static constexpr uint32_t UB_ZERO_OUT_SIZE = UB_ZERO_OUT_ROWS * dVBaseSize;   // OUT_T 元素
    static constexpr uint32_t UB_ZERO_OUT_BASE = UB_NZIDX_BASE + 256;
    // pNzIdxUB 索引表 256B 计入总量（原实现漏计）
    static constexpr uint32_t UB_TOTAL = UB_ZERO_OUT_BASE + UB_ZERO_OUT_SIZE * sizeof(OUT_T);
    static_assert(UB_TOTAL <= 256 * 1024, "V-core UB budget (256KB) exceeded");

    // ===== UB Tensor 成员（对齐设计 §4 表）=====
    LocalTensor<SOFTMAX_T> mm1ResUB_;     // [2][128,128] half，C1 Fixpipe 目标（loop%2）
    LocalTensor<MM_OUT_T> mm2ResUB_;      // [129,128] fp32，C2 Fixpipe 目标（V2 输入 + divOut/outputT 别名）
    LocalTensor<QUANT_T> vec1PUb_;        // [2][128,128] fp8，V1 P 的 NZ 源布局暂存
    LocalTensor<MM_OUT_T> stage2OutUB_;   // [128,128] fp32，V2 跨块累积分子 [DV,S1]
    LocalTensor<MM_OUT_T> accRowsumUB_;   // [128] fp32，V2 跨块累积分母 per S1
    LocalTensor<SOFTMAX_T> softmaxMaxUB_; // [3][128] half，accMax log2 整数 k（mloop%3）
    LocalTensor<MM_OUT_T> softmaxExpUB_;  // [3][128] fp32，跨块因子 fp32 位模式（loop%3）
    LocalTensor<SOFTMAX_T> blockStartMaxUB_;   // [128] half，块首 K^{blkStart} 快照
    LocalTensor<SOFTMAX_T> subLoopUsedMaxUB_;  // [2][128] half，各 subLoop k_i（供末 subLoop 算 pscale）
    LocalTensor<SCALE_T> pscaleGridUB_;  // [6][768B] pscale L1 网格影像（[ring]，VF 散布 + 连续整槽发 L1）
    LocalTensor<uint8_t> pNzIdxUB_;      // P NZ 打包 Gather 索引表（pscale 改字节对后唯一消费者是 P 存储）
    LocalTensor<OUT_T> zeroOutUB_;       // 零批输出清零源 [64,128]（DEAL_ZERO 行整段清零 GM）

    // ===== L1 共享区视图成员（AIC L1，V1 经 MTE3 写入 P/pscale 的目标地址）=====
    // P 槽核间交错布局（common_def L1_PSLOT 宏）下双核槽不连续——统一走宽视图 + 计算基址
    LocalTensor<QUANT_T> l1SharedTensor_;       // L1 共享区整域视图（P 数据，交错槽）
    LocalTensor<SCALE_T> pScaleL1V0Tensor_;    // V0 的 pscale（[0, 4608B) 窗口内）
    LocalTensor<SCALE_T> pScaleL1V1Tensor_;    // V1 的 pscale（[4608, 9216B) 窗口内）

    // ===== 依赖注入 =====
    const ConstInfo &constInfo_;          // 核信息（subBlockIdx、scaleValue 等）

    // ===== GM 输出（设计 §6.2；BNSD 布局 + S1_ONLY 行拷贝——realGSize=1 语义下 GS1 退化为 S1）=====
    GlobalTensor<OUT_T> attentionOutGm_;                                            // GM 输出 tensor
    FaGmTensor<OUT_T, GmFormat::BNGSD, int32_t> outGmTensor_;                       // + 偏移计算器
    CopyAttenOutUbToGm<OUT_T, GmFormat::BNGSD, UbFormat::S1_ONLY> attenOutUbToGm_;  // UB→GM 拷贝仿函数

    // ===== 外部 pScale（静态 headroom 因子，InitTensorsVec 时 cast half 存成员）=====
    half pScaleHalf_ = static_cast<half>(1.0f);

    // ===== 核内事件 ID（对齐设计 §4 事件表）=====
    static constexpr uint32_t EVT_P_COPY = 0;   // V↔MTE3：P/pscale MTE3 前后 + 顶部等待
    static constexpr uint32_t EVT_GM_OUT = 1;   // V→MTE3：GM 输出前
    static constexpr uint32_t EVT_ZERO_OUT = 2; // V→MTE3：零批清零源 Duplicate 就绪后放行

public:
    __aicore__ inline explicit QuantFlashAttnBlockVectorMxfp8SoftmaxFp16(ConstInfo &constInfo)
        : constInfo_(constInfo) {}

    /*
     * GM 输出初始化（kernel InitInput 阶段调用）
     * 注意：parser 必须已 Init 完成后再调用（OffsetCalculator 内部为拷贝语义，早调会拷到空 parser）
     */
    __aicore__ inline void
    InitInput(__gm__ uint8_t *attentionOut,
              const ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> &parser)
    {
        attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_T *>(attentionOut));
        outGmTensor_.gmTensor = attentionOutGm_;
        outGmTensor_.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                           constInfo_.s1Size, constInfo_.dSizeV, parser);
    }

    __aicore__ inline void InitTensorsVec()
    {
        // ① UB 静态分配（编译期 BASE 常量链，对齐设计 §4 表）
        //    mm1Res/mm2Res 用 common_def 跨核契约；其余按 BASE 链推导
        //    addr = 字节地址，size = 元素数
        mm1ResUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_MM1RES_BASE,
                                           UB_MM1RES_SLOT * UB_MM1RES_BUFCNT);
        mm2ResUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_MM2RES_BASE, UB_MM2RES_SIZE);
        vec1PUb_ = LocalTensor<QUANT_T>(TPosition::VECCALC, UB_VEC1P_BASE, UB_VEC1P_SIZE);
        stage2OutUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_STAGE2OUT_BASE, UB_STAGE2OUT_SIZE);
        accRowsumUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_ACCROWSUM_BASE, UB_ACCROWSUM_SIZE);
        softmaxMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_SMAX_BASE,
                                              UB_SMAX_SLOT * UB_SMAX_BUFCNT);
        softmaxExpUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_SEXP_BASE,
                                              UB_SMAX_SLOT * UB_SEXP_BUFCNT);
        blockStartMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_BSMAX_BASE, UB_SMAX_SLOT);
        subLoopUsedMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_SUBMAX_BASE,
                                                  UB_SMAX_SLOT * UB_PSCALE_SUBLOOPS);
        pscaleGridUB_ = LocalTensor<SCALE_T>(TPosition::VECCALC, UB_PSCALE_BASE,
                                             UB_PSCALE_SLOT * UB_PSCALE_RINGS);
        pNzIdxUB_ = LocalTensor<uint8_t>(TPosition::VECCALC, UB_NZIDX_BASE,
                                          256);  // 索引表 256B（实现期可调）
        zeroOutUB_ = LocalTensor<OUT_T>(TPosition::VECCALC, UB_ZERO_OUT_BASE, UB_ZERO_OUT_SIZE);

        // ② L1 共享区视图（按 common_def 常量构造，不分配、不初始化）。
        //    P 槽核间交错布局下双核槽不连续——整域宽视图 + L1_PSLOT(core, ring) 计算基址
        l1SharedTensor_ = LocalTensor<QUANT_T>(TPosition::A1, 0, L1_SHARED_REGION_SIZE);
        pScaleL1V0Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V0_BASE,
                                                 L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE);
        pScaleL1V1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V1_BASE,
                                                 L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE);

        // ③ Gather 解交织索引表初始化（2026-09-20 定稿，设计 §9.1-③：无 Gather 方案作废，此表转正）
        //    公式 idx[i*128+j] = i + 2j（i∈{0,1}，j∈[0,128)）——一张表两个消费者：
        //    P 存储：Cast+Or 交织产物 [a0,b0,a1,b1,…] → Gather → [行i | 行i+1]（解交织）
        //    pscale：单 Cast 产物（偶位=值，奇位=零）→ Gather → [顺序128字节 | 零]（压缩）
        for (int32_t i = 0; i < 2; i++) {
            for (int32_t j = 0; j < 128; j++) {
                pNzIdxUB_.SetValue(i * 128 + j, static_cast<uint8_t>(i + 2 * j));
            }
        }

        // ④ 核内事件初始置位：EVT_P_COPY 的首次消费保护（镜像 DN L1082 惯用法）。
        //    ComputeVec1 顶部 Wait<MTE3_V> 先于任何 Set 执行（Set 在首任务末 subLoop 的
        //    epilogue 才发）——不预置则首任务挂死（2026-09-20 首跑发现）。
        //    跨核 flag 的初始释放由 kernel 层 Process() 负责（分层约定），此处只管核内事件。
        SetFlag<HardEvent::MTE3_V>(EVT_P_COPY);
    }

    __aicore__ inline void ComputeVec1(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx)
    {
        // S1 尾行（actMSize ≤ 128）时本核（V1）没有自己的 S1 列，直接返回：
        // S1 按 V0=前半 / V1=后半分给两个核，行尾块的 S1 总数不足 256 时 V1 分不到列。
        // 此时若继续执行，会在上一轮的残留数据（stale mm1Res）上空算一遍，
        // 产出的 P/pscale 虽无消费者但浪费流水。外层 ExecuteTask 的跨核 flag
        // 是无条件 Set 的，这里直接 return 不会引起死锁。
        if (runInfo.actVecS1Size == 0U) {
            return;
        }

        uint32_t c1v1Loop = AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, s2SubLoopSize);  // ≤2
        bool isLastSub = (subLoopIdx + 1U == c1v1Loop);
        uint32_t validRows = isLastSub
            ? (runInfo.actSingleLoopS2Size - subLoopIdx * s2SubLoopSize)   // 1..128
            : s2SubLoopSize;
        uint32_t mmSlot = subLoopIdx % 2U;                    // 对齐 cube FixpipeMm1 ring
        uint32_t stateSlot = runInfo.softmaxStateSlot;          // mloop % 3
        uint32_t taskSlot = runInfo.loop % 3U;                  // exp/pscale/pSlot 共用

        // 上轮任务的 P/pscale MTE3 排空后才允许 vf 覆写 vec1PUb（DN L279 同款）。
        // ⚠️ 仅任务首（subLoop 0）等待：Set 在每任务末 subLoop 的 epilogue 才发一次，
        //    若每个 subLoop 都等，子循环 1 将无 Set 可消费 → 死锁（2026-09-20 审计发现）。
        //    子循环 1 写槽 1 与上任务子循环 0 的 MTE3（槽 0）不同槽，且队列有序已传递保护。
        if (subLoopIdx == 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVT_P_COPY);
        }

        // 行首复位 accMax：新 [bN2,gS1] 行的槽位含 3 行前旧值，必须复位为 MIN
        if (runInfo.isFirstS2Loop && subLoopIdx == 0U) {
            Duplicate(softmaxMaxUB_[stateSlot * UB_SMAX_SLOT], MIN_HALF_VALUE, UB_SMAX_SLOT);
        }

        // 块首快照 m_{k-1}：非首块的 subLoop 0 时拷贝 accMax → blockStartMaxUB
        if (subLoopIdx == 0U && !runInfo.isFirstS2Loop) {
            DataCopy(blockStartMaxUB_, softmaxMaxUB_[stateSlot * UB_SMAX_SLOT], UB_SMAX_SLOT);
        }

        // softmax 主体：max 归约 → 量化合并 → exp → cast fp8 → NZ 打包
        // 尾 subLoop 先清零本槽：任意 S2 泛化（2026-09-24）——softmax 只写 validRows 行，
        // 行 [validRows, 128) 为陈旧数据；C2 的 Mmad k 已对齐到 64（MmadMx 约束），扩展行
        // 会参与累加，必须为 0（0×陈旧 V=0，rowsum 同免）。整槽 16KB 清零后由 softmax
        // 覆写前 validRows 行，全宽 subLoop（validRows==128）无需清零
        if (validRows < s2SubLoopSize) {
            constexpr uint32_t pSlotSize = s2SubLoopSize * (s1BaseSize / 2);   // 16KB（与 VfSoftmax 槽宽一致）
            Duplicate(vec1PUb_.template ReinterpretCast<uint8_t>()[subLoopIdx * pSlotSize],
                      static_cast<uint8_t>(0U), pSlotSize);
        }
        QfaVectorApi::VfSoftmaxFp16ToFp8(vec1PUb_, softmaxMaxUB_, subLoopUsedMaxUB_, pNzIdxUB_,
                                            mm1ResUB_, pScaleHalf_, validRows, subLoopIdx,
                                            mmSlot, stateSlot);

        // P → AIC L1（MTE3；subLoop 粒度槽：ring = loop×2+subLoopIdx，核间交错 L1_PSLOT）
        SetFlag<HardEvent::V_MTE3>(EVT_P_COPY);
        WaitFlag<HardEvent::V_MTE3>(EVT_P_COPY);
        // L1 P 槽（16KB）与 vec1PUb 暂存槽（16KB）布局完全同构：列组主序 [4 组 × 128 行]，
        // 组 j @ j×4KB、行 r @ r×32B → 一次 16KB 纯连续拷贝。
        // 尾块 S2∈(129,192] 时写满 16KB 也安全（槽定长不出界；垃圾行被 Mmad k=actSubKSize 截断）
        // ——结构性消除旧跳写公式 subLoopIdx×4KB 越过 s2Align64=192 列组边界的 bug
        uint32_t pRing = L1_PRING(runInfo.loop, subLoopIdx);
        constexpr uint32_t pSlotSize = s2SubLoopSize * (s1BaseSize / 2);   // 16KB（= L1_P_SINGLE_SLOT_SIZE）
        DataCopy(l1SharedTensor_[L1_PSLOT(constInfo_.subBlockIdx, pRing)],
                 vec1PUb_[subLoopIdx * pSlotSize], pSlotSize);

        // ===== 末 subLoop epilogue：跨块因子 + pscale 网格影像直发 =====
        if (isLastSub) {
            // 跨块 rescale 因子（非首块才算）
            if (!runInfo.isFirstS2Loop) {
                QfaVectorApi::VfCalcRescaleFactor(softmaxExpUB_, blockStartMaxUB_, softmaxMaxUB_,
                                                   taskSlot, stateSlot);
            }
            // pscale：所有 subLoop 统一计算 + 网格直落（末 subLoop 的 k_i = K_final——
            // accMax 单调累积 → Δk 恒 0 → 127，走通用路径无需特判；读侧每个被读的 y 行
            // 都由本任务的 V1 写过，无陈旧读取，不需要 cube 侧预填）
            LocalTensor<SCALE_T> &pScaleL1Target =
                (constInfo_.subBlockIdx == 0U) ? pScaleL1V0Tensor_ : pScaleL1V1Tensor_;
            LocalTensor<uint8_t> pscaleGridU8 = pscaleGridUB_.template ReinterpretCast<uint8_t>();
            for (uint32_t i = 0; i < c1v1Loop; i++) {
                uint32_t psRing = L1_PRING(runInfo.loop, i);
                QfaVectorApi::VfCalcPScale(pscaleGridU8, subLoopUsedMaxUB_, softmaxMaxUB_,
                                           psRing, i, stateSlot);
            }
            // 一次连续整槽拷贝进 L1（768B count-form——唯一实测可靠的 AIV MTE3 UB→L1 形态；
            // strided 直发实测 scrambled，故镜像 DN/权威两段式：VF 散布 UB 网格 + 连续整槽发）
            SetFlag<HardEvent::V_MTE3>(EVT_P_COPY);
            WaitFlag<HardEvent::V_MTE3>(EVT_P_COPY);
            for (uint32_t i = 0; i < c1v1Loop; i++) {
                uint32_t psRing = L1_PRING(runInfo.loop, i);
                DataCopy(pScaleL1Target.template ReinterpretCast<uint8_t>()
                             [psRing * L1_PSCALE_SINGLE_SLOT_SIZE],
                         pscaleGridU8[psRing * UB_PSCALE_SLOT], L1_PSCALE_SINGLE_SLOT_SIZE);
            }
            // 为下一任务的顶部等待补 Set（全部 P/pscale MTE3 排空后触发）
            SetFlag<HardEvent::MTE3_V>(EVT_P_COPY);
        }
    }

    __aicore__ inline void ComputeVec2(RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        // V1 半无有效列（actVecS1Size==0）：C2 由 cube 侧防御跳过，V2 同步早退
        //（外层 C2_V2 flag 无条件 Set，早退不破坏跨核配平）
        if (runInfo.actVecS1Size == 0U) {
            return;
        }

        // ===== (1) 跨 S2 分块累积（设计 §6.2：首块 = vf 纯拷贝起步，消费者统一 V 管道）=====
        uint32_t taskSlot = runInfo.loop % 3;
        if (runInfo.isFirstS2Loop) {
            QfaVectorApi::VfFlashUpdateAcc<false>(stage2OutUB_, accRowsumUB_, mm2ResUB_,
                                                   softmaxExpUB_, taskSlot);
        } else {
            QfaVectorApi::VfFlashUpdateAcc<true>(stage2OutUB_, accRowsumUB_, mm2ResUB_,
                                                   softmaxExpUB_, taskSlot);
        }

        // ===== (2) 末块：除 + cast + 转置写 + GM 输出（设计 §6.2 步骤(2)）=====
        if (runInfo.isLastS2Loop) {
            // 踩踏闸：累积步骤读尽 mm2ResUB 后，才允许转置写以别名区覆写它
            PipeBarrier<PIPE_V>();
            // 转置带 pad 输出（T 后缀 = Transposed），别名落 mm2Res 区 [0, ~36KB)
            LocalTensor<bfloat16_t> outputT = mm2ResUB_.template ReinterpretCast<bfloat16_t>();
            QfaVectorApi::VfDivAndCast(outputT, stage2OutUB_, accRowsumUB_);

            // ===== GM 输出（V 管道写尽 outputT 后才允许 MTE3 搬运）=====
            SetFlag<HardEvent::V_MTE3>(EVT_GM_OUT);
            WaitFlag<HardEvent::V_MTE3>(EVT_GM_OUT);
            // colCount=144 = 128 数据 + 16 pad：助手 srcStride 公式算得 1 个 32B 单位，
            // 每行末尾恰好跳过填充块（无需压实拷贝）
            FaUbTensor<OUT_T> ubTensor{outputT, runInfo.actVecS1Size, 144U};
            GmCoordS1Only gmCoord{runInfo.bIdx, runInfo.realN2Idx, 0U,
                                  runInfo.gS1Idx + runInfo.vecS1BaseIdx, 0U,
                                  runInfo.actVecS1Size, constInfo_.dSizeV};
            attenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
        }
    }

    /*
     * 零批输出清零（2026-09-27 seq_used=0 泛化：DEAL_ZERO TODO 落地，消费 metadata
     * needInitOutput 语义）。batch 的 seqused_q/kv 为 0 时该 bN2 行无任何计算任务，
     * 输出必须为 0——kv=0 批 q 有效行为严格对比区（原实现不写=未初始化内存，
     * PctRlt 随内存状态漂移：B2_KVS0x256 实测 50%）。整行 [0, s1Size) 清零
     *（含 padding 尾行，语义完备）；行范围按 V0/V1 对半分摊（镜像 S1 半语义）；
     * 64 行/块循环支持任意 s1Size。
     * 流水安全：零源为专用 UB 槽（与流水线缓冲无交集），[Dup,Set]@V / [Wait,Copy]@MTE3
     * 成对追加于队尾——既有跨管道 Wait 均由更早已入队的 Set 满足，无环（EVT_GM_OUT 同款论证）
     */
    __aicore__ inline void ZeroOutputRows(uint32_t bIdx, uint32_t realN2Idx)
    {
        uint32_t half = (constInfo_.s1Size + 1U) / 2U;
        uint32_t rowStart = (constInfo_.subBlockIdx == 0U) ? 0U : half;
        uint32_t rowCount = (constInfo_.subBlockIdx == 0U) ? half : constInfo_.s1Size - half;
        if (rowCount == 0U) {
            return;
        }

        // 零源整备（V 管道 Duplicate）→ V→MTE3 事件放行
        Duplicate(zeroOutUB_, static_cast<OUT_T>(0.0f), UB_ZERO_OUT_SIZE);
        SetFlag<HardEvent::V_MTE3>(EVT_ZERO_OUT);
        WaitFlag<HardEvent::V_MTE3>(EVT_ZERO_OUT);

        // 分块整行清零：colCount=dVBaseSize、dDealSize=dSizeV ≤ dVBaseSize → 源侧紧凑
        while (rowCount > 0U) {
            uint32_t chunkRows = AttentionCommon::Min(rowCount, UB_ZERO_OUT_ROWS);
            FaUbTensor<OUT_T> ubTensor{zeroOutUB_, chunkRows, dVBaseSize};
            GmCoordS1Only gmCoord{bIdx, realN2Idx, 0U, rowStart, 0U, chunkRows, constInfo_.dSizeV};
            attenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
            rowStart += chunkRows;
            rowCount -= chunkRows;
        }
    }

    __aicore__ inline void ReleaseTensorsVec()
    {
        // 空壳即最终形态：UB 用裸 LocalTensor 寻址（无 TPipe 队列需释放）、事件用固定 ID（无需回收）；
        // 跨核 flag 配平归 kernel 层（架构裁决，corrections #4/#8 链路）
    }
};

} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于 Tensor API + mad_mx 硬件的 MX-FP8 矩阵乘计算阶段
 * 替换 MC2MXQuantMatmul + MatMulASWKernel
 *
 * 核心流程（硬件自动 dequant，无需手动 Cast+Mul）：
 *   GM→L1: A(fp8/Nz) + ScaleA(fp8_e8m0/Zz) + B(fp8/Zn) + ScaleB(fp8_e8m0/Nn)
 *   L1→L0A: A(Nz) + ScaleA(Zz) via load_cbuf_to_ca_mx
 *   L1→L0B: B(Zn) + ScaleB(Nn) via load_cbuf_to_cb_mx
 *   MMAD: mad_mx via MmadMx (硬件将 data*scale 后做矩阵乘)
 *   L0C→GM: fixpipe
 *
 * 依赖: 样例版 tensor_api（/mc2/3rd/tensor_api/）已内置 fp8_e8m0_t 支持
 */

#ifndef TENSOR_MX_MATMUL_STAGE_H
#define TENSOR_MX_MATMUL_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../../common/op_kernel/mc2_templates/computation/matmul/matmul_base.h"
#include "../../common/op_kernel/mc2_templates/computation/matmul/mc2_mx_quant_matmul.h"
#include "tensor_matmul_tiling_data.h"
#include "tensor.h"

// ═══════════════════════════════════════════════════════════════════
// 自定义 MX 尺度 L1→L0A copy atom（源自 matmul_story 样例）
// ═══════════════════════════════════════════════════════════════════

namespace Mc2Mx {

static constexpr uint32_t MXFP_GROUP_SIZE = 32;
static constexpr uint32_t MXFP_DIVISOR_SIZE = 64;

template <typename T>
__aicore__ inline T CeilDiv(T a, T b) { return (b == 0) ? a : (a + b - 1) / b; }

struct CopyL12L0MxScaleA3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        uint16_t mStartPos = CeilDiv<uint16_t>(AscendC::Std::get<0>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPos = AscendC::Std::get<1>(coord) / MXFP_DIVISOR_SIZE;
        auto mStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_ca_mx(mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()),
                           mStartPos, kStartPos, mStep, kStep, srcStride, dstStride);
    }
};

struct CopyL12L0MxScaleB3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        uint16_t nStartPos = CeilDiv<uint16_t>(AscendC::Std::get<1>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPos = AscendC::Std::get<0>(coord) / MXFP_DIVISOR_SIZE;
        auto nStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_cb_mx(mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()),
                           nStartPos, kStartPos, nStep, kStep, srcStride, dstStride);
    }
};

// ═══════════════════════════════════════════════════════════════════
// 自定义 MmadMx 操作（源自 matmul_story 样例的 tile_mmad_mx.h）
// ═══════════════════════════════════════════════════════════════════

struct MmadMx {
    template <typename Tp, const Tp& traits, typename T, typename U, typename S>
    __aicore__ inline static void Mad(
        const T& dst, const U& fm, const S& filter, uint16_t m, uint16_t k, uint16_t n,
        uint8_t unitFlagCtrl, bool btBuffCtrl, bool initCMatrixCtrl)
    {
        mad_mx(dst.Data().Get(), fm.Data().Get(), filter.Data().Get(),
               m, k, n, unitFlagCtrl, true, btBuffCtrl, initCMatrixCtrl);
    }
};

} // namespace Mc2Mx

// ── 注册到 tensor API 的 CopyTraits 系统 ──
template <>
struct AscendC::Te::CopyTraits<::Mc2Mx::CopyL12L0MxScaleA3510>
    : public AscendC::Te::CopyTraits<::Mc2Mx::CopyL12L0MxScaleA3510, AscendC::Te::LoadDataTraitDefault,
                                     ::Mc2Mx::CopyL12L0MxScaleA3510, AscendC::Te::LoadDataTraitDefault> {};

template <>
struct AscendC::Te::CopyTraits<::Mc2Mx::CopyL12L0MxScaleB3510>
    : public AscendC::Te::CopyTraits<::Mc2Mx::CopyL12L0MxScaleB3510, AscendC::Te::LoadDataTraitDefault,
                                     ::Mc2Mx::CopyL12L0MxScaleB3510, AscendC::Te::LoadDataTraitDefault> {};

// ── 注册 MmadMx 到 tensor API 的 MmadTraits 系统 ──
namespace AscendC { namespace Te {
template <typename Opration, typename TraitStruct>
struct MmadTraits<Opration, TraitStruct> {
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;
    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args&... args) const {
        Opration::template Mad<TraitType, trait, Args...>(args..., m, k, n, unitFlagCtrl, btBuffCtrl, initCMatrixCtrl);
    }
    uint16_t m = 0; uint16_t k = 0; uint16_t n = 0;
    uint8_t unitFlagCtrl = 0; bool btBuffCtrl = false; bool initCMatrixCtrl = false;
};
template <>
struct MmadTraits<::Mc2Mx::MmadMx>
    : public MmadTraits<::Mc2Mx::MmadMx, MmadTraitDefault, ::Mc2Mx::MmadMx, MmadTraitDefault> {};
}}


// ═══════════════════════════════════════════════════════════════════
// TensorMXQuantMatmulStage
// ═══════════════════════════════════════════════════════════════════

namespace Mc2Kernel {

static constexpr uint32_t MX_BASE_N = 128;
static constexpr uint32_t MX_BASE_K = 128;
static constexpr uint64_t MX_L1_HALF = 256UL * 1024;

template <typename TilingType, bool X2TRANSPOSE>
class TensorMXQuantMatmulStage {
public:
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using ScaleType = AscendC::fp8_e8m0_t;
    using CType = DTYPE_Y;
    using AccumType = float;

    __aicore__ inline TensorMXQuantMatmulStage(AscendC::TPipe *tPipe) : tPipePtr_(tPipe) {}
    __aicore__ inline void Init() {}
    __aicore__ inline MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::MXQuantMMAdditionalData, TilingType> *GetContextPtr() { return &MMcontext_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
    MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::MXQuantMMAdditionalData, TilingType> MMcontext_;
    AscendC::TPipe *tPipePtr_;
};


template <typename TilingType, bool X2TRANSPOSE>
__aicore__ inline void TensorMXQuantMatmulStage<TilingType, X2TRANSPOSE>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV { return; }

    auto& base = MMcontext_.baseData;
    auto& add  = MMcontext_.additionalData;
    GM_ADDR aGM = base.aGM + taskIndex * base.aOffset;
    GM_ADDR bGM = base.bGM + taskIndex * base.bOffset;
    GM_ADDR cGM = base.cGM + taskIndex * base.cOffset;
    GM_ADDR x1ScaleGM = add.x1Scale + taskIndex * add.x1ScaleOffset;
    GM_ADDR x2ScaleGM = add.x2Scale;

    if (taskIndex == 0) {
        AscendC::printf("[TENSOR-MXMUL-FP8] Process enter task=0\n");
    }

    tPipePtr_->Reset();
    auto* tl = MMcontext_.tilingDataPtr;
    uint32_t m_ = tl->tileM, k_ = tl->fullK, n_ = tl->fullN;

    uint32_t kSGrp = (k_ + Mc2Mx::MXFP_GROUP_SIZE - 1) / Mc2Mx::MXFP_GROUP_SIZE;
    uint32_t kL1 = MX_BASE_K;

    // GM 视图
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGM)),
                                        AscendC::Te::MakeNDLayout<AType>(m_, k_));
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(bGM)),
                                        AscendC::Te::MakeNDLayout<BType>(k_, n_));
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM)),
                                        AscendC::Te::MakeNDLayout<CType>(m_, n_));
    auto gmSA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleType*>(x1ScaleGM)),
                                         AscendC::Te::MakeScaleANDLayout<ScaleType>(m_, kSGrp));
    auto gmSB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleType*>(x2ScaleGM)),
                                         AscendC::Te::MakeScaleBDNLayout<ScaleType>(kSGrp, n_));

    // L1 布局工厂
    using MkA = AscendC::Te::NzLayoutFormat<AType>;
    using MkB = AscendC::Te::ZnLayoutFormat<BType>;
    using MkSA = AscendC::Te::ZzLayoutFormat<ScaleType>;
    using MkSB = AscendC::Te::NnLayoutFormat<ScaleType>;

    auto pk = [](uint32_t s) -> uint32_t { return ((s + 1) / 2) * 2; };
    uint32_t kLPk = pk((kL1 + Mc2Mx::MXFP_GROUP_SIZE - 1) / Mc2Mx::MXFP_GROUP_SIZE);
    uint64_t aL1Sz = (uint64_t)m_ * kL1 * sizeof(AType);
    uint64_t bL1Sz = (uint64_t)kL1 * MX_BASE_N * sizeof(BType);
    uint64_t saL1Sz = (uint64_t)m_ * kLPk * sizeof(ScaleType);
    uint64_t sbL1Sz = (uint64_t)kLPk * MX_BASE_N * sizeof(ScaleType);

    uint64_t l1AO[2], l1SAO[2], l1BO[2], l1SBO[2];
    for (int b = 0; b < 2; ++b) {
        uint64_t h = b * MX_L1_HALF;
        l1AO[b] = h; l1SAO[b] = h + aL1Sz;
        l1BO[b] = h + aL1Sz + saL1Sz; l1SBO[b] = h + aL1Sz + saL1Sz + bL1Sz;
    }

    uint64_t ab = 0;

    // N-tiling
    for (uint32_t no = 0; no < n_; no += MX_BASE_N) {
        uint32_t cN = (no + MX_BASE_N <= n_) ? MX_BASE_N : (n_ - no);
        auto gmBC = gmC(AscendC::Te::MakeCoord(0, no), AscendC::Te::MakeShape(m_, cN));
        auto tL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<AccumType>(0),
                                             AscendC::Te::MakeL0CLayout(m_, cN));
        ab = 0;

        uint32_t kIt = (k_ + kL1 - 1) / kL1;
        for (uint32_t ik = 0; ik < kIt; ++ik) {
            uint64_t id = ab & 1;
            uint32_t ko = ik * kL1;
            uint32_t cK = (ko + kL1 <= k_) ? kL1 : (k_ - ko);
            uint32_t kSo = ko / Mc2Mx::MXFP_GROUP_SIZE;
            uint32_t cKPk = pk((cK + Mc2Mx::MXFP_GROUP_SIZE - 1) / Mc2Mx::MXFP_GROUP_SIZE);

            auto CtGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

            // A(fp8) GM→L1 Nz
            auto tA = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(l1AO[id]), MkA{}(m_, cK));
            AscendC::Te::Copy(CtGM2L1, tA, gmA(AscendC::Te::MakeCoord(0, ko), AscendC::Te::MakeShape(m_, cK)));

            // ScaleA(fp8_e8m0) GM→L1 Zz (新 tensor_api 已支持)
            auto tSA = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<ScaleType>(l1SAO[id]), MkSA{}(m_, cKPk));
            AscendC::Te::Copy(CtGM2L1, tSA, gmSA(AscendC::Te::MakeCoord(0, kSo), AscendC::Te::MakeShape(m_, cKPk)));

            // B(fp8) GM→L1 Zn
            auto tB = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<BType>(l1BO[id]), MkB{}(cK, cN));
            AscendC::Te::Copy(CtGM2L1, tB, gmB(AscendC::Te::MakeCoord(ko, no), AscendC::Te::MakeShape(cK, cN)));

            // ScaleB(fp8_e8m0) GM→L1 Nn
            auto tSB = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<ScaleType>(l1SBO[id]), MkSB{}(cKPk, cN));
            AscendC::Te::Copy(CtGM2L1, tSB, gmSB(AscendC::Te::MakeCoord(kSo, no), AscendC::Te::MakeShape(cKPk, cN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(id);

            // K 内循环
            uint32_t k0It = (cK + MX_BASE_K - 1) / MX_BASE_K;
            uint64_t l0p = 0;
            for (uint32_t ik0 = 0; ik0 < k0It; ++ik0) {
                uint32_t k0o = ik0 * MX_BASE_K;
                uint32_t cK0 = (k0o + MX_BASE_K <= cK) ? MX_BASE_K : (cK - k0o);
                uint64_t l0id = l0p & 1;
                uint64_t l0Off = (64UL * 1024 / 2) * l0id;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0id);
                auto CtL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});

                // L1→L0A: A
                auto tAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<AType>(l0Off),
                                                     AscendC::Te::MakeNzLayout<AType>(m_, cK0));
                AscendC::Te::Copy(CtL12L0, tAL0, tA(AscendC::Te::MakeCoord(0, k0o), AscendC::Te::MakeShape(m_, cK0)));

                // L1→L0A: ScaleA (load_cbuf_to_ca_mx)
                auto CtMxSA = AscendC::Te::MakeCopy(::Mc2Mx::CopyL12L0MxScaleA3510{});
                uint32_t k0So = k0o / Mc2Mx::MXFP_GROUP_SIZE;
                uint32_t cK0Pk = pk((cK0 + Mc2Mx::MXFP_GROUP_SIZE - 1) / Mc2Mx::MXFP_GROUP_SIZE);
                auto tSAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<ScaleType>(l0Off),
                                                      AscendC::Te::MakeZzLayout<ScaleType>(m_, cK0Pk));
                AscendC::Te::Copy(CtMxSA, tSAL0, tSA(AscendC::Te::MakeCoord(0, k0So), AscendC::Te::MakeShape(m_, cK0Pk)),
                                  AscendC::Te::MakeCoord(0, k0o));

                // L1→L0B: B
                auto tBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<BType>(l0Off),
                                                     AscendC::Te::MakeZnLayout<BType>(cK0, cN));
                AscendC::Te::Copy(CtL12L0, tBL0, tB(AscendC::Te::MakeCoord(k0o, 0), AscendC::Te::MakeShape(cK0, cN)));

                // L1→L0B: ScaleB (load_cbuf_to_cb_mx)
                auto CtMxSB = AscendC::Te::MakeCopy(::Mc2Mx::CopyL12L0MxScaleB3510{});
                auto tSBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<ScaleType>(l0Off),
                                                      AscendC::Te::MakeNnLayout<ScaleType>(cK0Pk, cN));
                AscendC::Te::Copy(CtMxSB, tSBL0, tSB(AscendC::Te::MakeCoord(k0So, 0), AscendC::Te::MakeShape(cK0Pk, cN)),
                                  AscendC::Te::MakeCoord(k0o, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0id);

                // mad_mx
                bool fin = (ik + 1 == kIt && ik0 + 1 == k0It);
                bool ini = (ik == 0 && ik0 == 0);
                uint32_t cKa = ((cK0 + Mc2Mx::MXFP_DIVISOR_SIZE - 1) / Mc2Mx::MXFP_DIVISOR_SIZE) * Mc2Mx::MXFP_DIVISOR_SIZE;
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<::Mc2Mx::MmadMx>>{}.with(
                        static_cast<uint16_t>(m_), static_cast<uint16_t>(cKa), static_cast<uint16_t>(cN),
                        static_cast<uint8_t>(fin ? 3U : 2U), false, ini),
                    tL0C, tAL0, tBL0);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0id);
                l0p++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(id);
            ab++;
        }
        // fixpipe
        auto CtFp = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CtFp, gmBC, tL0C, AscendC::Te::FixpipeParams{static_cast<uint8_t>(3)});
    }
}

} // namespace Mc2Kernel

#endif

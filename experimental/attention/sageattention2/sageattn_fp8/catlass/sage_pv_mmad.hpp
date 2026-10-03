#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

#include <cstdint>
#include "../ascendc/tiling.h"
#include "../ascendc/accum_reduce.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"
#include "../../quant/quant_common.h"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace Catlass::SageAttn {

/**
 * FP8 PV 矩阵乘分块与硬件特征
 * 负责 P(FP8) x V(FP8) -> O(FP32) 的 MMAD 运算与 FixPipe 裸搬交接
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 128,
    uint32_t TileK = 64
>
struct SagePvMmadTraits {
    using ElementA = int8_t;  // Softmax 概率矩阵 S (FP8 e4m3)
    using ElementB = int8_t;  // Value 矩阵 V (FP8 e4m3)
    using ElementC = float;   // L0C 内部累加格式 (FP32)

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;
};

/**
 * FP8 PV 矩阵乘执行器与 Epilogue 封装
 * 委托 sage_tile.hpp 中的通用 LoadTile2D / ExecuteTileMmad / CopyL0CToUbSplitM 原语执行，
 * 实现一处定义与消除重复样板代码。
 */
template <class Traits = SagePvMmadTraits<64, 128, 64>>
class SagePvMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;

    __aicore__ SagePvMmad() = default;

    /**
     * 加载 Softmax 概率矩阵 S 分块至 L0A（从 UB 奇偶槽搬运）
     */
    __aicore__ void LoadL0A(uint32_t s_addr, uint32_t l0a_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
        LoadTile2D<ElementA, AscendC::TPosition::A2, AscendC::TPosition::VECIN>(l0a_addr, s_addr, m, k);
    }

    /**
     * 加载 V 分块至 L0B (从 L1 Bank 1 搬运)
     */
    __aicore__ void LoadL0B(uint32_t l1_v_addr, uint32_t l0b_addr, uint32_t k, uint32_t n) {
        k_ = k;
        n_ = n;
        LoadTile2D<ElementB, AscendC::TPosition::B2, AscendC::TPosition::B1>(l0b_addr, l1_v_addr, k, n);
    }

    /**
     * 执行 Cube FP8 MMAD 计算：O += S x V
     *
     * @param init 是否重置 L0C 累加器 (两级累加新周期起始为 true)
     */
    __aicore__ void Compute(bool init = false) {
        ExecuteTileMmad<ElementC, ElementA, ElementB, true>(m_, n_, k_, init);
    }

    /**
     * FixPipe Epilogue：L0C FP32 部分和裸搬至 UB 交接槽（A7，task/12 D1 方案甲 + D2）
     * 委托 CopyL0CToUbSplitM 统一实现。
     */
    __aicore__ void ExecuteEpilogue(
        uint32_t l0c_src_offset,
        uint32_t ub_dst_offset,
        [[maybe_unused]] const sage_attn::FixpipeConfig& fixpipe_cfg) {
        CopyL0CToUbSplitM<ElementC>(l0c_src_offset, ub_dst_offset, m_, n_);
    }

private:
    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
};

#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
/**
 * A7（task/12 D2）V1 消费侧单行合并原语：pv_partial 交接槽行 → O_ACCUM 行。
 * 轨迹 = 宿主 twin::QuantizePAndAccumulatePvRow（AccumulatePvBlockTwin 调用）的设备段
 * 实例：o_row[c] += partial[r][c] · v_scale[c]，其中 partial 为 Cube PV MMAD 的 FP32
 * 部分和（P8 码 × V8 码，P 侧 448· 放大域与 v_scale 的 V 通道域拆分同宿主 v_at 分解）。
 * - fp32+fp16 档（fp16_round=1，档位判据 AccumReduceController::GetFixpipeConfig
 *   .enable_inline_quant）：并入前按 QuantizeToFp16 唯一口径做 FP16 RNE 舍回——
 *   块内求和已由 PV MMAD 承担（设备段粒度替代 twin k_step 子块，L-A 容差域，task/12）；
 * - fp32 / fp32+fp32 档：FP32 直通并入。
 * 调用方须保证 O 行经 S 槽 / dot32 同款握手串行化（FLAG_PV_PARTIAL_READY/FREE）。
 */
__simd_vf__ inline void MergePvPartialRowVF(__ubuf__ float* partial_row, __ubuf__ float* o_row,
                                            __ubuf__ float* v_scale_ptr, uint32_t d,
                                            uint32_t fp16_round) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> partial, vs, o;
        LoadAlign(partial, partial_row + c);
        LoadAlign(vs, v_scale_ptr + c);
        Mul(partial, partial, vs, full);   // 贡献 = partial · v_scale[c]
        if (fp16_round != 0u) {
            // D2：FP16 RNE 舍回定在 V1 合并处（quant::Fp32ToFp16/Fp16ToFp32 同口径 RNE）
            RegTensor<half> rounded;
            Cast<half, float, toFp16>(rounded, partial, full);
            Cast<float, half, toFp32>(partial, rounded, full);
        }
        LoadAlign(o, o_row + c);
        Add(o, o, partial, full);
        StoreAlign(o_row + c, o, full);
    }
}
#endif

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

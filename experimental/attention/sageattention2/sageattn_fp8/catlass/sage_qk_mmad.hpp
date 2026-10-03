#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

#include <cstdint>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace Catlass::SageAttn {

/**
 * INT8 QK^T 矩阵乘分块与调度策略
 * 负责 L1 -> L0A/L0B -> MMAD -> L0C 的标准硬件通路封装
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 64,
    uint32_t TileK = 128
>
struct SageQkMmadTraits {
    using ElementA = int8_t;    // Query 元素类型 (INT8)
    using ElementB = int8_t;    // Key 元素类型 (INT8)
    using ElementC = int32_t;   // MMAD 累加类型 (INT32)

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;
};

/**
 * INT8 QK^T 矩阵乘执行器
 * 委托 sage_tile.hpp 中的通用 LoadTile2D / ExecuteTileMmad / CopyL0CToUbSplitM 原语执行，
 * 实现一处定义与消除重复样板代码。
 */
template <class Traits = SageQkMmadTraits<64, 64, 128>>
class SageQkMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;

    __aicore__ SageQkMmad() = default;

    /**
     * 加载 Q 分块：从 L1 Bank 0 搬入 L0A
     */
    __aicore__ void LoadL0A(uint32_t l1_q_addr, uint32_t l0a_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
        LoadTile2D<ElementA, AscendC::TPosition::A2, AscendC::TPosition::A1>(l0a_addr, l1_q_addr, m, k);
    }

    /**
     * 加载 K 分块：从 L1 Bank 1 搬入 L0B
     */
    __aicore__ void LoadL0B(uint32_t l1_k_addr, uint32_t l0b_addr, uint32_t n, uint32_t k) {
        n_ = n;
        k_ = k;
        LoadTile2D<ElementB, AscendC::TPosition::B2, AscendC::TPosition::B1>(l0b_addr, l1_k_addr, n, k);
    }

    /**
     * 执行 Cube MMAD 计算：P = Q x K^T，累加于 L0C
     *
     * @param init 是否清零 L0C 累加器 (新分块起始为 true，迭代累加为 false)
     */
    __aicore__ void Compute(bool init = true) {
        ExecuteTileMmad<ElementC, ElementA, ElementB, false>(m_, n_, k_, init);
    }

    /**
     * A7（task/12_device_data_path.md D1 方案甲）：QK^T 的 L0C INT32 结果裸搬至 UB
     * logits 区（dav-3510 FixPipe，quantPre DISABLE = 位型透传，无精度转换）。
     * 反量化不在本层，由 V1 侧 DequantMaskLogitsRowVF 承担。
     * 委托 CopyL0CToUbSplitM 统一实现。
     */
    __aicore__ void EmitL0CToUbRaw(uint32_t ub_dst_half0, [[maybe_unused]] uint32_t ub_dst_half1,
                                   uint32_t m, uint32_t n) {
        CopyL0CToUbSplitM<ElementC>(0U, ub_dst_half0, m, n);
    }

private:
    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

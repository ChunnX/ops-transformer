#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

#include <cstdint>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"


#if SAGE_ATTN_KERNEL_API
#include "catlass/arch/arch.hpp"
#endif

#if defined(__has_include) && __has_include("tla/tensor.hpp")
#include "tla/tensor.hpp"
#endif

#if SAGE_ATTN_DEVICE_COMPILE
#if defined(__has_include) && __has_include("catlass/gemm/tile/tile_mmad.hpp")
#include "catlass/gemm/tile/tile_mmad.hpp"
#endif
#endif

namespace Catlass::SageAttn {

/**
 * INT8 QK^T 矩阵乘分块与调度策略（TileM/N/K 为分块缺省形状）：
 * L1 -> L0A/L0B -> MMAD(官方 TileMmadTla) -> L0C 的硬件通路封装。
 * 元素类型契约：A/B=INT8 码、C=INT32 累加（反量化在 V1 侧，见 EmitL0CToUbRaw）。
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 64,
    uint32_t TileK = 128
>
struct SageQkMmadTraits {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;
};

/**
 * INT8 QK^T 矩阵乘执行器：加载/搬出委托 sage_tile.hpp 的 LoadTile2D /
 * CopyL0CToUbSplitM 原语，Mmad 委托官方 Gemm::Tile::TileMmadTla——一处定义。
 */
template <class Traits = SageQkMmadTraits<64, 64, 128>>
class SageQkMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;

    __aicore__ SageQkMmad() = default;

    /**
     * 加载 Q 分块：从 L1 Bank 0 搬入 L0A（ND 行布局）。
     * 搬运不走官方 TileCopyTla L1→L0A 特化——其 EnableIf 强约束 L1 侧 zN 分形布局
     * （copy_l1_to_l0a.hpp），与本算子 V0 产出的 ND 行 + Data/Scale 相邻 L1 布局不匹配
     * （布局契约见 sage_tile.hpp 头注释与设计文档"CATLASS 接入边界"节），
     * 故保留 LoadTile2D 单点实现。L0 寻址恒为基址 0（本核 L0A/L0B 单缓冲无分片，
     * 原显式 l0a/l0b 基址参数全调用域恒 0，随死参清理删除）。
     */
    __aicore__ void LoadL0A(uint32_t l1_q_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
        LoadTile2D<ElementA, AscendC::TPosition::A2, AscendC::TPosition::A1>(0U, l1_q_addr, m, k);
    }

    /**
     * 加载 K 分块：从 L1 Bank 1 搬入 L0B（ND 行布局，理由同 LoadL0A）
     */
    __aicore__ void LoadL0B(uint32_t l1_k_addr, uint32_t n, uint32_t k) {
        n_ = n;
        k_ = k;
        LoadTile2D<ElementB, AscendC::TPosition::B2, AscendC::TPosition::B1>(0U, l1_k_addr, n, k);
    }

    /**
     * 执行 Cube MMAD：P = Q x K^T，累加于 L0C（INT32）。
     * CATLASS 组件采用：Gemm::Tile::TileMmadTla（官方 tile_mmad.hpp）承担 Mmad 发射——
     * m/n/k/cmatrixInitVal 逐参等价于原手写 ExecuteTileMmad（已删）；差异仅两点，
     * 均对验收域无行为影响：
     * 1. 3510 分支官方置 disableGemv=true（原 MmadParams 缺省 false=自动）：本算子
     *    M ∈ [64,128]（m_per_core 为 64 的倍数），恒不触发 M=1 gemv 路径；
     * 2. (m/16)*(n/16)<10 时官方追加 PipeBarrier<PIPE_M>（纯 M 管序栅栏，无数值效应，
     *    不触碰跨核 flag 握手——握手走 S/M/FIX 管显式事件）。
     */
    __aicore__ void Compute([[maybe_unused]] bool init = true) {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::TBuffAddr l0a_desc{};
        l0a_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        l0a_desc.dataLen = m_ * k_ * sizeof(ElementA);
        AscendC::LocalTensor<ElementA> l0a_tensor;
        l0a_tensor.SetAddr(l0a_desc);

        AscendC::TBuffAddr l0b_desc{};
        l0b_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        l0b_desc.dataLen = n_ * k_ * sizeof(ElementB);
        AscendC::LocalTensor<ElementB> l0b_tensor;
        l0b_tensor.SetAddr(l0b_desc);

        AscendC::TBuffAddr l0c_desc{};
        l0c_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
        l0c_desc.dataLen = m_ * n_ * sizeof(ElementC);
        AscendC::LocalTensor<ElementC> l0c_tensor;
        l0c_tensor.SetAddr(l0c_desc);

        auto layout_a = tla::MakeLayout(tla::MakeShape(m_, k_), tla::MakeStride(k_, tla::Int<1>{}));
        auto layout_b = tla::MakeLayout(tla::MakeShape(n_, k_), tla::MakeStride(k_, tla::Int<1>{}));
        // L0C 布局官方单点（tla/layout.hpp MakeLayoutL0C）：zN 分形 shape/stride +
        // originShape=(rows,cols) 由官方一处派生，取代手写行主 MakeLayout 装配；
        // 本处 TileMmadTla 的显式 (m,n,k) 重载仅消费 tensor.data()，位元零变化。
        auto layout_c = tla::MakeLayoutL0C(m_, n_);

        auto tensor_l0a = tla::MakeTensor(l0a_tensor, layout_a, Catlass::Arch::PositionL0A{});
        auto tensor_l0b = tla::MakeTensor(l0b_tensor, layout_b, Catlass::Arch::PositionL0B{});
        auto tensor_l0c = tla::MakeTensor(l0c_tensor, layout_c, Catlass::Arch::PositionL0C{});

        Catlass::Gemm::Tile::TileMmadTla<Catlass::Arch::Ascend950, ElementA, Catlass::layout::RowMajor> tile_mmad;
        tile_mmad(tensor_l0c, tensor_l0a, tensor_l0b, m_, n_, k_, init);
#endif
    }

    /**
     * A7（task/12_device_data_path.md D1 方案甲）：QK^T 的 L0C INT32 结果裸搬至 UB
     * logits 区（dav-3510 FixPipe，quantPre DISABLE = 位型透传，无精度转换）。
     * 反量化不在本层，由 V1 侧 DequantMaskLogitsRowVF 承担。
     * Fixpipe 走 CopyL0CToUbSplitM 单点：官方 CopyL0CToUBTla 特化基于
     * FixpipeParamsC310（nSize 按 32B 粒度 RoundUp、unitFlag 语义）且 SPLIT_M 档用
     * dualDstCtl 硬件双目的地模式，与本核已真机验证的 Arch3510 手动分段发射
     * （m>64 双 Fixpipe、整幅 zN 行距）不同构，不替换（golden 位级锁定域）。
     * 原 ub_dst_half1 死参已删（dot32 单缓冲，第二半槽地址无消费者）。
     */
    __aicore__ void EmitL0CToUbRaw(uint32_t ub_dst_offset, uint32_t m, uint32_t n) {
        CopyL0CToUbSplitM<ElementC>(0U, ub_dst_offset, m, n);
    }

private:
    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

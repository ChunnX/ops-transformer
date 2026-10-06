#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
#pragma once

#include <cstdint>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"
#include "../../quant/quant_common.h"

#if SAGE_ATTN_DEVICE_COMPILE
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
 * FP8 PV 矩阵乘分块与硬件特征（TileM/N/K 为分块缺省形状）：
 * P(FP8 e4m3) x V(FP8 e4m3) -> O(FP32) 的 MMAD 运算与 FixPipe 裸搬交接。
 * 注意 Traits 的 ElementA/B 以 int8_t 承载字节码（FP8 与 INT8 同为 1 字节），
 * MMAD 发射时按 mx_fp8_e4m3_t 位型解释（见 Compute）——与真机验证过的
 * ReinterpretCast 形态逐位一致。
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 128,
    uint32_t TileK = 64
>
struct SagePvMmadTraits {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = float;

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;
};

/**
 * FP8 PV 矩阵乘执行器与 Epilogue 封装：加载/搬出委托 sage_tile.hpp 的 LoadTile2D /
 * CopyL0CToUbSplitM 原语，Mmad 委托官方 Gemm::Tile::TileMmadTla——一处定义。
 */
template <class Traits = SagePvMmadTraits<64, 128, 64>>
class SagePvMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;

    __aicore__ inline __attribute__((always_inline)) SagePvMmad() = default;

    /**
     * 加载 Softmax 概率矩阵 S(P) 分块至 L0A（从 L1 Bank 0 奇偶双缓冲槽搬运至 L0A）。
     */
    __aicore__ inline __attribute__((always_inline)) void LoadL0A(uint32_t s_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
        // L0 寻址恒为基址 0（单缓冲无分片，原 l0a/l0b 基址参数全调用域恒 0，死参清理）
        LoadTile2D<ElementA, AscendC::TPosition::A2, AscendC::TPosition::A1>(0U, s_addr, m, k, false);
    }

    /**
     * 加载 V 分块至 L0B（从 L1 Bank 1 槽搬运，zN -> nZ 硬件转置）
     */
    __aicore__ inline __attribute__((always_inline)) void LoadL0B(uint32_t l1_v_addr, uint32_t k, uint32_t n) {
        k_ = k;
        n_ = n;
        LoadTile2D<ElementB, AscendC::TPosition::B2, AscendC::TPosition::B1>(0U, l1_v_addr, k, n, true);
    }

    /**
     * 执行 Cube FP8 MMAD：O += S x V，累加于 L0C（FP32）。
     * CATLASS 组件采用：Gemm::Tile::TileMmadTla（官方 tile_mmad.hpp）承担 Mmad 发射。
     * 位型等价要点：L0A/L0B 张量按 AscendC::mx_fp8_e4m3_t 构造（与原手写路径的
     * ReinterpretCast<mx_fp8_e4m3_t> 同一 Mmad 重载，FP8 Cube 管线），ElementC=FP32
     * 直传——TileMmadTla::operator() 仅透传 tensor.data()，位型由张量元素类型决定。
     * m/n/k/cmatrixInitVal 逐参等价；disableGemv=true 与 PipeBarrier<PIPE_M> 差异
     * 的无行为影响论证见 sage_qk_mmad.hpp Compute 注释（两执行器同一组件）。
     */
    __aicore__ inline __attribute__((always_inline)) void Compute([[maybe_unused]] bool init = false) {
#if SAGE_ATTN_DEVICE_COMPILE
        using Fp8Code = ::float8_e4m3_t;

        AscendC::TBuffAddr l0a_desc{};
        l0a_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        l0a_desc.dataLen = m_ * k_ * sizeof(ElementA);
        AscendC::LocalTensor<Fp8Code> l0a_tensor;
        l0a_tensor.SetAddr(l0a_desc);

        AscendC::TBuffAddr l0b_desc{};
        l0b_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        l0b_desc.dataLen = k_ * n_ * sizeof(ElementB);
        AscendC::LocalTensor<Fp8Code> l0b_tensor;
        l0b_tensor.SetAddr(l0b_desc);

        AscendC::TBuffAddr l0c_desc{};
        l0c_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
        l0c_desc.dataLen = m_ * n_ * sizeof(ElementC);
        l0c_desc.bufferAddr = HardwareResourceQuota::O_CHAIN_L0C_OFFSET;
        AscendC::LocalTensor<ElementC> l0c_tensor;
        l0c_tensor.SetAddr(l0c_desc);

        auto layout_a = tla::MakeLayout(tla::MakeShape(m_, k_), tla::MakeStride(k_, tla::Int<1>{}));
        auto layout_b = tla::MakeLayout(tla::MakeShape(k_, n_), tla::MakeStride(n_, tla::Int<1>{}));
        // L0C 布局官方单点（tla/layout.hpp MakeLayoutL0C），理由同 sage_qk_mmad.hpp：
        // 显式 (m,n,k) 重载仅消费 tensor.data()，位元零变化。
        auto layout_c = tla::MakeLayoutL0C(m_, n_);

        auto tensor_l0a = tla::MakeTensor(l0a_tensor, layout_a, Catlass::Arch::PositionL0A{});
        auto tensor_l0b = tla::MakeTensor(l0b_tensor, layout_b, Catlass::Arch::PositionL0B{});
        auto tensor_l0c = tla::MakeTensor(l0c_tensor, layout_c, Catlass::Arch::PositionL0C{});

        Catlass::Gemm::Tile::TileMmadTla<Catlass::Arch::Ascend950, ElementA, Catlass::layout::RowMajor> tile_mmad;
        tile_mmad(tensor_l0c, tensor_l0a, tensor_l0b, m_, n_, k_, init);
#endif
    }

    /**
     * FixPipe Epilogue：L0C FP32 部分和裸搬至 UB 交接槽（A7，task/12 D1 方案甲 + D2）。
     * 委托 CopyL0CToUbSplitM 单点实现（不走官方 CopyL0CToUBTla 的理由同
     * sage_qk_mmad.hpp EmitL0CToUbRaw：C310 参数面 + dualDstCtl 硬件分段与已验证
     * 的 Arch3510 手动分段不同构，golden 位级锁定域不替换）。
     * fixpipe_cfg 死参数链已删（P11）：随路量化不在 Fixpipe——fp32+fp16 档的
     * FP16 RNE 舍回定在 V1 合并处（FusedRescaleMergeVF，档位直读 tiling）。
     * l0c_src_offset 参数已删：现已映射至 HardwareResourceQuota::O_CHAIN_L0C_OFFSET。
     *
     * SPLIT_M 考证结论（task/todo.md 复用 5，独立评审通过，2026-10-04）：
     * 官方 CopyL0CToUBMode::SPLIT_M（dualDstCtl=1）是把 M/2*N 并行写入两个
     * Vector SubCore 各自 UB 的双目的地模式（仓内佐证：git/catlass
     * examples/46_ascend950_matmul_fixpipe_opti/README.md:15、迁移指南
     * .agents/skills/catlass-ascend950-migration/README.zh.md:353-355；特化实现
     * gemm/tile/ascend950/copy_l0c_to_ub.hpp:184-189 仅 ROW_MAJOR+NoQuant、
     * :210 mSize 偶数约束、:218 dualDstCtl=1）。其收益前提是两个 SubCore 对称
     * 各消费半幅（block_epilogue_fixpipe.hpp:44-46 `if (!SPLIT_M &&
     * GetSubBlockIdx() > 0) return;`；examples/49 fai_kernel.h:576/:365/:442
     * 双 AIV 半幅偏移互证）。本核 V0 专职量化、从不消费 Cube 产出：dot32 槽
     * 唯一消费者 attention_vec1_softmax.h:210，pv_partial 槽唯一消费者
     * attention_vec1_output.h:69，subCore 分派见 attention_vec1_output.h:471-480。
     * 若启用 dualDstCtl=1，上半 M/2 行将按 V1 槽偏移写进 V0 UB，与 V0 现役
     * QUANT_WORKSPACE/NZ_OUTPUT 工作区（sage_resource.hpp:14-16,27,92-95）
     * 正面碰撞，属活性覆写而非死数据——SPLIT_M 在本 1C2V 单消费者拓扑不可落地。
     * 现行 dualDstCtl=0 + subBlockId=true（sage_tile.hpp:155-156）即官方
     * NO_SPLIT 单目的地等效档（gemm/tile/ascend950/copy_l0c_to_ub.hpp:91-92）：
     * 整幅单指令硬件直通，零软件切分、零复制、零 GM 往返；双 SubCore 吞吐匹配
     * 预期由 V0 量化 / V1 Softmax 职责分工结构性达成（msprof 待证）。
     * 真机跟进候选：m=128 单条发射省一次双段发射（官方 gemm/tile/ascend950/
     * copy_l0c_to_ub.hpp:210 提示可行，未经真机 msprof 实测不改动）。
     */
    __aicore__ inline __attribute__((always_inline)) void ExecuteEpilogue(uint32_t ub_dst_offset) {
        CopyL0CToUbSplitM<ElementC>(HardwareResourceQuota::O_CHAIN_L0C_OFFSET, ub_dst_offset, m_, n_);
    }

private:

    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

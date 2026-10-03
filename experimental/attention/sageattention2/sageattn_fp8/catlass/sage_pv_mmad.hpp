#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

#include <cstdint>
#include <type_traits>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"
#include "../../quant/quant_common.h"

#if SAGE_ATTN_KERNEL_API
#include "catlass/arch/arch.hpp"
// 官方位型映射表单点（catlass/gemm/helper.hpp）：
// - ElementAccumulatorSelector：fp8×fp8→FP32 累加位型（Arch3510 表项）；
// - GetL0Element：fp8 码 → L0 MMAD 寄存器位型（mx_fp8_e4m3_t，CATLASS_ARCH=3510）。
// 两表均随官方架构支持自动跟随；依赖 kernel_operator.h 的 AscendC 位型，宿主门禁
// （KERNEL_API=0）不引入，Traits 走下方显式 fallback。构建侧（run.sh kernel、
// build_opp.sh、tools/sim 设备步）统一注入 -DCATLASS_ARCH=3510。
#include "catlass/gemm/helper.hpp"
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
#if SAGE_ATTN_KERNEL_API
    // 官方累加位型表（Catlass::Gemm::helper，helper.hpp 单点）：语义位型
    // fp8_e4m3×fp8_e4m3→FP32（Arch3510 表项；A/B 以 int8_t 字节码承载，见下），
    // 随官方 Arch 支持自动跟随，不再手写位型
    using ElementC = typename Catlass::Gemm::helper::ElementAccumulatorSelector<
        float8_e4m3_t, float8_e4m3_t>::ElementAccumulator;
#else
    // 宿主门禁（无 kernel_operator.h，helper.hpp 不可解析）：显式 FP32，与
    // 设备态官方表项一致（tests/test_kernel_twin 宿主分支位级锁定域）
    using ElementC = float;
#endif

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

    __aicore__ SagePvMmad() = default;

    /**
     * 加载 Softmax 概率矩阵 S 分块至 L0A（从 UB 奇偶槽搬运，ND 行布局）。
     * 不走官方 TileCopyTla——其 L1 侧 zN 分形布局强约束与本算子 V1 产出的
     * UB 行布局不匹配（布局契约见 sage_tile.hpp 头注释），保留 LoadTile2D 单点。
     */
    __aicore__ void LoadL0A(uint32_t s_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
        // L0 寻址恒为基址 0（单缓冲无分片，原 l0a/l0b 基址参数全调用域恒 0，死参清理）
        LoadTile2D<ElementA, AscendC::TPosition::A2, AscendC::TPosition::VECIN>(0U, s_addr, m, k);
    }

    /**
     * 加载 V 分块至 L0B（从 L1 Bank 1 槽搬运，(n,k)=(d, block_n) 记账见
     * op_kernel ProcessCube 的 DEFECT-M-2 注释）
     */
    __aicore__ void LoadL0B(uint32_t l1_v_addr, uint32_t k, uint32_t n) {
        k_ = k;
        n_ = n;
        LoadTile2D<ElementB, AscendC::TPosition::B2, AscendC::TPosition::B1>(0U, l1_v_addr, k, n);
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
    __aicore__ void Compute([[maybe_unused]] bool init = false) {
#if SAGE_ATTN_DEVICE_COMPILE
        // FP8 位型载体（字节码按 e4m3 解释；与 INT8 Traits 字段解耦）：
        // 官方 GetL0Element 表单点（helper.hpp，CATLASS_ARCH=3510 表项 fp8→mx_fp8），
        // 随官方 L0 位型映射自动跟随；static_assert 使构建漏配 -DCATLASS_ARCH=3510
        // 时在编译期显式失败（而非静默退回非 MX 位型）
        using Fp8Code = typename Catlass::Gemm::helper::GetL0Element<float8_e4m3_t, true>::Element;
        static_assert(std::is_same<Fp8Code, AscendC::mx_fp8_e4m3_t>::value,
                      "GetL0Element<fp8,true> requires -DCATLASS_ARCH=3510 (mx_fp8_e4m3_t L0 mapping)");

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
     * FP16 RNE 舍回定在 V1 合并处（MergePvPartialRowVF，档位直读 tiling）。
     * l0c_src_offset 参数已删：L0C 单缓冲，全调用域恒 0（死参清理）。
     */
    __aicore__ void ExecuteEpilogue(uint32_t ub_dst_offset) {
        CopyL0CToUbSplitM<ElementC>(0U, ub_dst_offset, m_, n_);
    }

private:
    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
};

#if SAGE_ATTN_DEVICE_COMPILE
/**
 * A7（task/12 D2）V1 消费侧单行合并原语：pv_partial 交接槽行 → O_ACCUM 行。
 * 轨迹 = 宿主 twin::AccumulatePvRow（AccumulatePvBlockTwin 调用）的设备段
 * 实例：o_row[c] += partial[r][c] · v_scale[c]，其中 partial 为 Cube PV MMAD 的 FP32
 * 部分和（P8 码 × V8 码，P 侧 448· 放大域与 v_scale 的 V 通道域拆分同宿主 v_at 分解）。
 * - fp32+fp16 档（fp16_round=1，档位判据由调用方直读 tiling.pv_accum_dtype）：
 *   并入前做 FP16 RNE 舍回（与宿主 quant::Fp32ToFp16/Fp16ToFp32 同 RNE 口径——
 *   accum_reduce.h 的 QuantizeToFp16 仿真壳已随死代码清理删除，此处 Cast 链
 *   即其位级等价）——
 *   块内求和已由 PV MMAD 承担（设备段粒度替代 twin k_step 子块，L-A 容差域，task/12）；
 * - fp32 / fp32+fp32 档：FP32 直通并入。
 * 调用方须保证 O 行经 S 槽 / dot32 同款握手串行化（FLAG_PV_PARTIAL_READY/FREE）。
 * 本原语为 AscendC VF 域（任务书分工：Vector 侧逻辑不走 CATLASS 模板）；
 * 函数标记采用官方设备函数宏 Catlass::CATLASS_DEVICE_SIMD_VF
 *（catlass/detail/macros.hpp：__simd_vf__ static，与官方 FA epilogue 同款）。
 */
CATLASS_DEVICE_SIMD_VF void MergePvPartialRowVF(__ubuf__ float* partial_row, __ubuf__ float* o_row,
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
        Mul(partial, partial, vs, full);
        if (fp16_round != 0u) {
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

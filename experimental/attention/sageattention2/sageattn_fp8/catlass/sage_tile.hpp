#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP
#pragma once

#include <cstdint>
#include "../ascendc/tiling.h"

/**
 * 官方 CATLASS 组件接入（cann/catlass 仓，CATLASS_ROOT/include 经构建接线注入 -I，
 * 全部编译场景——宿主门禁/OPP 打包/BiSheng 设备与 sim 混编——均带该 -I，故
 * catlass.hpp 无条件直含，不再保留本地字面值回退分支）：
 * 分形粒度常量以官方 catlass.hpp 为唯一定义源（BYTE_PER_C0=32、
 * C0_NUM_PER_FRACTAL=16、BYTE_PER_FRACTAL=512），本文件一律经官方常量派生
 * （如 LOADDATA2D_REPEAT_BYTES = BYTE_PER_FRACTAL），官方演进自动跟随，
 * 不做字面量锁定。
 * 设备态热路径的搬运原语（LoadTile2D、CopyUbToL1zN）：L1 全面统一为 zN 分形布局，
 * 硬件直连 LoadTile2D（位级对齐 CATLASS copy_l1_to_l0a.hpp / copy_l1_to_l0b.hpp）与
 * CopyUbToL1zN（位级对齐 CATLASS copy_ub_to_l1_tla.hpp），架构豁免项见设计文档。
 */
#include "catlass/catlass.hpp"
// 官方对齐/极值组件显式声明（catlass/detail/alignment.hpp，全局 ::RoundUp/::RoundDown/
// ::CeilDiv/::Min/::Max，CATLASS_HOST_DEVICE constexpr）——catlass.hpp 已带入，此处
// 显式 include 使本文件对全局对齐原语的依赖单点可见。
#include "catlass/detail/alignment.hpp"
#if SAGE_ATTN_DEVICE_COMPILE
#include "kernel_operator.h"
#else
namespace AscendC {

enum class TPosition : uint8_t {
    GM = 0,
    A1 = 1,
    A2 = 2,
    B1 = 3,
    B2 = 4,
    C1 = 5,
    C2 = 6,
    CO1 = 7,
    CO2 = 8,
    VECIN = 9,
    VECOUT = 10,
    VECCALC = 11,
    TSCM = 12,
    LCM = 13
};

// 宿主门禁（DEVICE_COMPILE=0）HardEvent 影子枚举：编号逐项镜像 dav-3510 设备头
//（cann aarch64-linux/asc/impl/basic_api/kernel_event.h，3510 无 FIX_V/V_FIX 档）。
// 宿主分支该枚举仅作 Catlass::SageAttn::Sync<E> 的模板标签（事件体编译期剔除），
// 不触发硬件语义；编号对齐消除双域枚举漂移隐患（旧影子枚举为自造编号，已废弃）。
enum class HardEvent : uint8_t {
    MTE2_MTE1, MTE1_MTE2, MTE1_M, M_MTE1, MTE2_V, V_MTE2, MTE3_V, V_MTE3,
    M_V, V_M, V_V, MTE3_MTE1, MTE1_MTE3, MTE1_V, MTE2_M, M_MTE2, V_MTE1,
    M_FIX, FIX_M, MTE3_MTE2, MTE2_MTE3, S_V, V_S, S_MTE2, MTE2_S, S_MTE3,
    MTE3_S, MTE2_FIX, FIX_MTE2, FIX_S, M_S, FIX_MTE3, MTE1_FIX, FIX_MTE1, FIX_FIX,
};
} // namespace AscendC
#endif

// 官方架构资源组件（Arch::Ascend950 片上容量）与官方 layout 构件（tla/layout.hpp 的
// MakeLayout/MakeShape/MakeStride/crd2offset）——Bank 几何与 GM 布局单点经官方组件派生。
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"

namespace Catlass::SageAttn {

/**
 * 昇腾硬件事件同步模板（对齐竞品 PR#12430 / common.h 原语，消除冗余配对代码）
 */
template <AscendC::HardEvent E>
__aicore__ inline __attribute__((always_inline)) void Sync() {
#if SAGE_ATTN_DEVICE_COMPILE
    AscendC::SetFlag<E>(0);
    AscendC::WaitFlag<E>(0);
#endif
}

/**
 * LoadData 2D 单 repeat 搬运字节数（512B，CANN LoadData2DParams.repeatTimes 的计量单位）
 * 直接引用官方 CATLASS BYTE_PER_FRACTAL 常量（32 * 16 = 512B，一处定义）
 */
constexpr uint32_t LOADDATA2D_REPEAT_BYTES = Catlass::BYTE_PER_FRACTAL;

/**
 * 统一分形坐标映射原语（ZnIndex）：
 * 对形状为 (M, K) 的矩阵，按昇腾硬件 zN 分形布局计算线性字节偏移。
 * 以 32 字节为一个分形列块（C0）：
 * col_block = col / 32, col_in = col % 32
 * m_aligned = (total_rows + 15u) & ~15u
 * offset = (col_block * m_aligned + row) * 32 + col_in
 */
__aicore__ inline __attribute__((always_inline)) constexpr uint32_t ZnIndex(
    uint32_t row, uint32_t col, uint32_t total_rows)
{
    const uint32_t m_aligned = (total_rows + 15u) & ~15u;
    const uint32_t col_block = col / 32u;
    const uint32_t col_in = col % 32u;
    return (col_block * m_aligned + row) * 32u + col_in;
}

/**
 * 通用 2D 分块搬运原语：从片上源位置搬入目标寄存器/缓冲（L1 -> L0A/L0B，基于 LoadData2DParamsV2）
 * 官方 CATLASS 最佳实践：
 * - L1 统一保存为 zN 分形布局；
 * - 载入 L0A：if_transpose = false，硬件直载 zN；
 * - 载入 L0B：if_transpose = true，硬件 MTE1 自动转置分形内元素为 nZ（支持 int8 与 fp8）。
 */
template <typename Element, AscendC::TPosition DstPos, AscendC::TPosition SrcPos>
__aicore__ inline __attribute__((always_inline)) void LoadTile2D(
    uint32_t dst_addr, uint32_t src_addr, uint32_t rows, uint32_t cols, bool if_transpose = false)
{
#if SAGE_ATTN_DEVICE_COMPILE
    const uint32_t total_bytes = rows * cols * static_cast<uint32_t>(sizeof(Element));
    AscendC::TBuffAddr dst_desc{};
    dst_desc.logicPos = static_cast<uint8_t>(DstPos);
    dst_desc.bufferHandle = nullptr;
    dst_desc.bufferAddr = dst_addr;
    dst_desc.dataLen = total_bytes;
    AscendC::LocalTensor<Element> dst_tensor;
    dst_tensor.SetAddr(dst_desc);

    AscendC::TBuffAddr src_desc{};
    src_desc.logicPos = static_cast<uint8_t>(SrcPos);
    src_desc.bufferHandle = nullptr;
    src_desc.bufferAddr = src_addr;
    src_desc.dataLen = total_bytes;
    AscendC::LocalTensor<Element> src_tensor;
    src_tensor.SetAddr(src_desc);

    AscendC::LoadData2DParamsV2 load_params;
    load_params.mStartPosition = 0;
    load_params.kStartPosition = 0;
    load_params.mStep = static_cast<uint16_t>(::RoundUp<2>(::CeilDiv<16>(rows)));
    load_params.kStep = static_cast<uint16_t>(::CeilDiv<32>(cols));
    load_params.ifTranspose = if_transpose;
    if (if_transpose) {
        // 官方 copy_l1_to_l0b.hpp B8/B4 特化（单装载分支，RoundUp<16>(N) % 32 == 0）：
        // 源 zN 外步长 srcOuterStrideCol = rowsRound * ELE_NUM_PER_C0 = rowsRound * 32
        // srcStride = srcOuterStrideCol / 512 = rowsRound / 16
        // 目的 nZ 外步长 dstOuterStrideRow = colsRound * ELE_NUM_PER_C0 = colsRound * 32
        // dstStride = dstOuterStrideRow / 512 = colsRound / 16
        // 本仓只取官方单装载分支：block_n ∈ {64,128}、d_pad ∈ {64,128} 恒为 32 的倍数，
        // N ≡ 16 (mod 32) 的 split 分支（mStep=2 循环装载，官方 169-182 行）不可达；
        // 域锁见 tests/test_tiling_struct.cpp Test 10，扩域前必须先移植 split 分支。
        load_params.srcStride = static_cast<int32_t>(::CeilDiv<16>(rows));
        load_params.dstStride = static_cast<uint16_t>(::CeilDiv<16>(cols));
    } else {
        // 官方 copy_l1_to_l0a.hpp 直载特化：
        load_params.srcStride = static_cast<int32_t>(load_params.mStep);
        load_params.dstStride = static_cast<uint16_t>(load_params.mStep);
    }
    AscendC::LoadData(dst_tensor, src_tensor, load_params);
#else
    (void)dst_addr;
    (void)src_addr;
    (void)rows;
    (void)cols;
    (void)if_transpose;
#endif
}

/**
 * 通用 2D 分块搬运原语（L1 nZ -> L0B nZ 直载，基于 LoadData2DParamsV2）：
 * 对齐官方 CATLASS copy_l1_to_l0b.hpp Ascend950 B8 nZ -> nZ 偏特化（354-362 行）：
 * 配合数学恒等式 zN(block_n, d) ≡ nZ(d, block_n)，V0 写入 L1 的 K 矩阵已天然为 nZ([d, block_n])。
 * 载入 L0B 时直接发起硬件直载（ifTranspose = false）：
 * - load_params.mStep = CeilDiv<16>(cols);
 * - load_params.kStep = CeilDiv<32>(rows);
 * - load_params.srcStride = load_params.mStep;
 * - load_params.dstStride = load_params.mStep;
 * 零标量转置，零 Vector ALU 开销，零中间 UB 缓冲区。
 * 域锁契约：
 * - 官方 copy_l1_to_l0b.hpp 354-362 行 nZ -> nZ 偏特化原生无 split 分支；
 * - 本算子 K 路径 cols (block_n) ∈ {64, 128}、rows (d) ∈ {64, 128}，恒为 16/32 整数倍；
 * - 域锁参数经 tests/test_tiling_struct.cpp Test 10 d) 段以官方字面式交叉验证锁定；
 * - 对应 V 路径继续由 LoadTile2D（if_transpose = true，单装载分支，N ≡ 0 mod 32）严密覆盖。
 */
template <typename Element, AscendC::TPosition DstPos, AscendC::TPosition SrcPos>
__aicore__ inline __attribute__((always_inline)) void LoadTile2DNzToNz(
    uint32_t dst_addr, uint32_t src_addr, uint32_t rows, uint32_t cols)
{
#if SAGE_ATTN_DEVICE_COMPILE
    const uint32_t total_bytes = rows * cols * static_cast<uint32_t>(sizeof(Element));
    AscendC::TBuffAddr dst_desc{};
    dst_desc.logicPos = static_cast<uint8_t>(DstPos);
    dst_desc.bufferHandle = nullptr;
    dst_desc.bufferAddr = dst_addr;
    dst_desc.dataLen = total_bytes;
    AscendC::LocalTensor<Element> dst_tensor;
    dst_tensor.SetAddr(dst_desc);

    AscendC::TBuffAddr src_desc{};
    src_desc.logicPos = static_cast<uint8_t>(SrcPos);
    src_desc.bufferHandle = nullptr;
    src_desc.bufferAddr = src_addr;
    src_desc.dataLen = total_bytes;
    AscendC::LocalTensor<Element> src_tensor;
    src_tensor.SetAddr(src_desc);

    AscendC::LoadData2DParamsV2 load_params;
    load_params.mStartPosition = 0;
    load_params.kStartPosition = 0;
    load_params.mStep = static_cast<uint16_t>(::CeilDiv<16>(cols));
    load_params.kStep = static_cast<uint16_t>(::CeilDiv<32>(rows));
    load_params.srcStride = static_cast<int32_t>(load_params.mStep);
    load_params.dstStride = static_cast<uint16_t>(load_params.mStep);
    load_params.ifTranspose = false;
    AscendC::LoadData(dst_tensor, src_tensor, load_params);
#else
    (void)dst_addr;
    (void)src_addr;
    (void)rows;
    (void)cols;
#endif
}

/**
 * 通用 Fixpipe L0C -> UB 裸搬原语（SPLIT_M 分段发射）：
 * dav-3510 Fixpipe 透传搬出（quantPre DISABLE，精确位型透传，zN -> ND）。
 * m > 64 时按 64 行双段发射，两段同分区写入 UB。
 * 不采用官方 Gemm::Tile::CopyL0CToUBTla：其 Ascend950 特化基于 FixpipeParamsC310
 * 参数面（nSize 按 32B 粒度 RoundUp、unitFlag 语义），SPLIT_M 档走 dualDstCtl 硬件
 * 双目的地模式；与本核真机 L-A 验证过的 Arch3510 参数面 + 手动双段发射（整幅 zN
 * 行距 srcStride=m、UB ND 行距 dstStride=n）不同构——golden 位级锁定域，不替换。
 */
template <typename ElementC>
__aicore__ inline __attribute__((always_inline)) void CopyL0CToUbSplitM(
    uint32_t l0c_src_offset,
    uint32_t ub_dst_offset,
    uint32_t m,
    uint32_t n)
{
#if SAGE_ATTN_DEVICE_COMPILE
    constexpr uint32_t M_SPLIT = 64;
    constexpr uint32_t elem_size = sizeof(ElementC);
    static constexpr auto format = AscendC::CO2Layout::ROW_MAJOR;
    static constexpr AscendC::FixpipeConfig cfg{format, true};

    AscendC::FixpipeParamsArch3510<format> fp;
    fp.nSize = static_cast<uint16_t>(n);
    fp.srcStride = static_cast<uint16_t>(m);   // L0C zN 行距（整幅行数）
    fp.dstStride = static_cast<uint16_t>(n);   // UB ND 行距
    fp.params.ndNum = 1;
    fp.params.srcNdStride = 0;
    fp.params.dstNdStride = 0;
    fp.quantPre = QuantMode_t::NoQuant;
    fp.dualDstCtl = 0;
    fp.subBlockId = true;

    AscendC::TBuffAddr l0c_desc{};
    l0c_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
    l0c_desc.bufferHandle = nullptr;
    l0c_desc.bufferAddr = l0c_src_offset;
    l0c_desc.dataLen = static_cast<uint32_t>(m * n * elem_size);
    AscendC::LocalTensor<ElementC> l0c_tensor;
    l0c_tensor.SetAddr(l0c_desc);

    AscendC::TBuffAddr ub0_desc{};
    ub0_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub0_desc.bufferHandle = nullptr;
    ub0_desc.bufferAddr = ub_dst_offset;
    ub0_desc.dataLen = static_cast<uint32_t>(m * n * elem_size);
    AscendC::LocalTensor<ElementC> ub_dst0;
    ub_dst0.SetAddr(ub0_desc);

    if (m <= M_SPLIT) {
        fp.mSize = static_cast<uint16_t>(m);
        AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst0, l0c_tensor, fp);
    } else {
        fp.mSize = static_cast<uint16_t>(M_SPLIT);
        AscendC::TBuffAddr ub1_desc = ub0_desc;
        ub1_desc.bufferAddr = ub_dst_offset + M_SPLIT * n * elem_size;
        ub1_desc.dataLen = static_cast<uint32_t>((m - M_SPLIT) * n * elem_size);
        AscendC::LocalTensor<ElementC> ub_dst1;
        ub_dst1.SetAddr(ub1_desc);
        // 第二段 L0C 源偏移：zN 内分形行粒度 = C0_NUM_PER_FRACTAL（16 行/分形），
        // 64 行 = 4 个分形行，元素偏移 64*16（官方 catlass.hpp 常量，一处定义）
        AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst0, l0c_tensor, fp);
        AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst1, l0c_tensor[M_SPLIT * Catlass::C0_NUM_PER_FRACTAL], fp);
    }
#else
    (void)l0c_src_offset;
    (void)ub_dst_offset;
    (void)m;
    (void)n;
#endif
}

/**
 * UB 行 → L1 行单块搬入（ND 行布局，V0 量化产出写入 L1 Data/Scale 相邻槽）：
 * 收敛 LoadAndQuantizeQ / QuantizeAndStageKey / QuantizeAndStageValue 三处相同的
 * DataCopyParams 样板（blockCount=1、blockLen 以 32B 块计 = 官方 BYTE_PER_BLK、
 * 行内连续零步距）。row_bytes 须为 BYTE_PER_BLK 整数倍（d ∈ {64,128} 字节码恒满足）。
 */
#if SAGE_ATTN_DEVICE_COMPILE
template <typename Element>
__aicore__ inline __attribute__((always_inline)) void CopyUbToL1(AscendC::LocalTensor<Element> l1_dst,
                                                                 AscendC::LocalTensor<Element> ub_src,
                                                                 uint32_t total_bytes)
{
    AscendC::DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = total_bytes / Catlass::BYTE_PER_BLK;
    cp.srcStride = 0;
    cp.dstStride = 0;
    AscendC::DataCopy(l1_dst, ub_src, cp);
}

template <typename Element>
__aicore__ inline __attribute__((always_inline)) void CopyUbRowToL1(AscendC::LocalTensor<Element> l1_dst,
                                                                    AscendC::LocalTensor<Element> ub_src,
                                                                    uint32_t row_bytes)
{
    CopyUbToL1(l1_dst, ub_src, row_bytes);
}

/**
 * 官方 CATLASS CopyUb2L1Tla 最佳实践原语（UB RowMajor -> L1 zN）：
 * UB 侧 RowMajor 矩阵经 MTE3 硬件 2D DMA 分形重组写入 L1 侧 zN 布局。
 * 对于每个 32 字节列块（cols / 32），发起一次跨步 DMA：
 * blockCount = m_aligned, blockLen = 1 (32B), srcStride = (cols / 32) - 1, dstStride = 0
 * 零 Vector ALU 算力占用，零标量循环，满线速硬件内存重组。
 */
template <typename Element>
__aicore__ inline __attribute__((always_inline)) void CopyUbToL1zN(
    AscendC::LocalTensor<Element> l1_dst,
    AscendC::LocalTensor<Element> ub_src,
    uint32_t rows,
    uint32_t cols)
{
    const uint32_t m_aligned = (rows + 15u) & ~15u;
    const uint32_t col_blocks = cols / 32u;
    AscendC::DataCopyParams cp;
    cp.blockCount = static_cast<uint16_t>(m_aligned);
    cp.blockLen = 1;
    cp.srcStride = static_cast<uint16_t>(col_blocks - 1u);
    cp.dstStride = 0;
    for (uint32_t c = 0; c < col_blocks; ++c) {
        AscendC::DataCopy(l1_dst[c * m_aligned * 32u], ub_src[c * 32u], cp);
    }
}

/**
 * 通用 2D 分块从 GM 搬入 UB 原语（利用 Ascend 950 NDDMA 2D/Strided DMA 硬件指令）：
 * 统一封装 DataCopyParams，支持 HND 稠密连续搬运与 NHD 跨步连续搬运。
 * blockLen / srcStride 以 32 字节块计（BYTE_PER_BLK = 32）。
 * ub_dst: 目标 UB 本地张量（连续紧凑存放，dstStride = 0）
 * gm_src_base: GM 源起始地址张量
 * rows: 搬运块数（行数，如 cur_q_len 或 block_k）
 * d: 每行元素数（列数，half 类型，d * sizeof(Element) 需为 32B 对齐）
 * stride_32b_blocks: GM 源地址相邻行之间的间隙（以 32 字节块计）。HND 时为 0；NHD 时为 (H - 1) * (d * sizeof(Element) / 32)
 */
template <typename Element>
__aicore__ inline __attribute__((always_inline)) void LoadTile2DFromGm(
    AscendC::LocalTensor<Element> ub_dst,
    AscendC::GlobalTensor<Element> gm_src_base,
    uint32_t rows,
    uint32_t d,
    uint32_t stride_32b_blocks = 0)
{
    if (rows == 0) return;
    const uint32_t block_len = (d * static_cast<uint32_t>(sizeof(Element))) / Catlass::BYTE_PER_BLK;
    if (stride_32b_blocks == 0 || rows <= 1) {
        AscendC::DataCopy(ub_dst, gm_src_base, rows * d);
    } else {
        AscendC::DataCopyParams cp;
        cp.blockCount = static_cast<uint16_t>(rows);
        cp.blockLen = static_cast<uint16_t>(block_len);
        cp.srcStride = static_cast<uint16_t>(stride_32b_blocks);
        cp.dstStride = 0;
        AscendC::DataCopy(ub_dst, gm_src_base, cp);
    }
}

template <typename Element>
__aicore__ inline __attribute__((always_inline)) void LoadTile2DFromGm(
    AscendC::LocalTensor<Element> ub_dst,
    __gm__ Element *gm_src_ptr,
    uint32_t rows,
    uint32_t d,
    uint32_t stride_32b_blocks = 0)
{
    if (rows == 0) return;
    AscendC::GlobalTensor<Element> gm_src;
    gm_src.SetGlobalBuffer(gm_src_ptr);
    gm_src.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    LoadTile2DFromGm(ub_dst, gm_src, rows, d, stride_32b_blocks);
}
#endif

/**
 * 计算 GM 2D 搬运相邻行之间的间隙（以 32 字节块计，供 DataCopyParams.srcStride 使用）：
 * - HND 布局：同一 Head 内相邻 Token 在物理内存上严格连续，stride = 0；
 * - NHD 布局 ([B, S, H, D])：同一 Head 内相邻 Token 的地址间隔为 H * D * sizeof(Element)，
 *   相邻行间隙为 (H - 1) * D * sizeof(Element) 字节，换算为 32 字节块数即：
 *   srcStride = (total_heads - 1) * (d * sizeof(Element) / 32)。
 */
[[nodiscard]] __aicore__ inline __attribute__((always_inline)) constexpr uint32_t CalcGm2dStride32B(
    uint32_t total_heads,
    uint32_t d,
    uint32_t elem_size,
    uint32_t tensor_layout)
{
    if (tensor_layout == sage_attn::LAYOUT_HND || total_heads <= 1u) {
        return 0u;
    }
    return (total_heads - 1u) * ((d * elem_size) / Catlass::BYTE_PER_BLK);
}


namespace detail {

// GM (B,H,S,D) 布局类型标签（运行时形状/步距；步距 int64 元素单位 + D 连续）
using GmBhsdShapeTag = decltype(tla::MakeShape(uint32_t{}, uint32_t{}, uint32_t{}, uint32_t{}));
using GmBhsdStrideTag = decltype(tla::MakeStride(int64_t{}, int64_t{}, int64_t{}, tla::Int<1>{}));

} // namespace detail

/**
 * GM 四维张量布局的 tla 单点表达（任务书分工：CATLASS 负责 Layout）：
 * - HND（layout_hnd != 0）：B 步距 H·S·D、H 步距 S·D、S 步距 D、D 连续；
 * - NHD（layout_hnd == 0）：B 步距 S·H·D、H 步距 D、S 步距 H·D、D 连续。
 * 步距统一 int64 元素单位（64 位 GM 寻址纪律），坐标折叠由 tla::crd2offset 承担
 * （逐维乘加，与手写 b·sb + h·sh + s·ss 位元一致，宿主孪生位级锁定域已验证）。
 * Q/K/V 输入与 O 输出的布局推导此前在核内 8 处各写一份 if (LAYOUT_HND) 步距四元组，
 * 全部收敛至此单点（布局/步距/重复次数逐项对齐原手写式，test_tiling_struct 锁值）。
 */
using GmBhsdLayout = decltype(tla::MakeLayout(detail::GmBhsdShapeTag{}, detail::GmBhsdStrideTag{}));

[[nodiscard]] __aicore__ inline __attribute__((always_inline)) constexpr GmBhsdLayout MakeGmBhsdLayout(uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                                                uint32_t layout_hnd)
{
    if (layout_hnd != 0u) {
        return tla::MakeLayout(tla::MakeShape(b, h, s, d),
                               tla::MakeStride(static_cast<int64_t>(h) * s * d,
                                               static_cast<int64_t>(s) * d,
                                               static_cast<int64_t>(d), tla::Int<1>{}));
    }
    return tla::MakeLayout(tla::MakeShape(b, h, s, d),
                           tla::MakeStride(static_cast<int64_t>(s) * h * d,
                                           static_cast<int64_t>(d),
                                           static_cast<int64_t>(h) * d, tla::Int<1>{}));
}

namespace detail {

using GmBhsShapeTag = decltype(tla::MakeShape(uint32_t{}, uint32_t{}, uint32_t{}));
using GmBhsStrideTag = decltype(tla::MakeStride(int64_t{}, int64_t{}, tla::Int<1>{}));

} // namespace detail

/**
 * GM 三维稠密布局（B,H,S 元素连续，与 tensor_layout 无关）：
 * LSE / lseCorrection float 张量按 (b, hq, sq) 稠密排布，坐标折叠
 * b·(H·S) + h·S + s 与手写嵌套乘加位元一致。
 */
using GmBhsLayout = decltype(tla::MakeLayout(detail::GmBhsShapeTag{}, detail::GmBhsStrideTag{}));

[[nodiscard]] __aicore__ inline __attribute__((always_inline)) constexpr GmBhsLayout MakeGmBhsLayout(uint32_t b, uint32_t h, uint32_t s)
{
    (void)b; // 步距自 H 起，b 不参与折叠（稠密布局形状仅作文档）
    return tla::MakeLayout(tla::MakeShape(b, h, s),
                           tla::MakeStride(static_cast<int64_t>(h) * s,
                                           static_cast<int64_t>(s), tla::Int<1>{}));
}

/**
 * Data + Scale 相邻排布契约（FlashMLA 硬件并发核心优化）：
 * 将 INT8/FP8 矩阵数据与对应块的 FP32 Scale 在物理内存中紧密相邻排布，
 * 使得每 token 的两次 MTE2 拷贝合并为一次、请求数量直接减半，
 * 彻底化解 16 项 MTE2 Issue Queue 溢出导致的硬件挂起。
 * 本算子的槽内实际摆布唯一定义点为下方 L1DoubleBankContract（Bank 0 的
 * Q_DATA→Q_SCALE 与 Bank 1 KV 槽的 [K Data][K Scale] 相邻排布均由其承载；
 * 对齐走官方 Catlass::RoundUp，32B = BYTE_PER_BLK，一个 MTE 块粒度）。
 * 原 KvSlotLayoutContract 模板已随 2026-10-03 死代码清扫删除：生产链路
 * （kernel / host tiling）零实例化，且其 RoundUp 槽内 Scale 偏移与核内实际
 * 摆布（block_n·d 直排）描述不一致，留存反而构成误导性契约。
 */

/**
 * L1 双 Bank 物理配平契约
 * Bank 0 (0~256KB): Q 矩阵与 Q Scale 常驻
 * Bank 1 (256~512KB): K/V 矩阵及相邻 Scale 双缓冲 Ping-Pong 滚动
 * 彻底消除 Cube MMAD 读取矩阵 A 与矩阵 B 时的 Bank 读通道争用
 */
struct L1DoubleBankContract {
    static constexpr uint32_t BANK0_START = sage_attn::L1_BANK0_OFFSET; // 0 KB
    static constexpr uint32_t BANK1_START = sage_attn::L1_BANK1_OFFSET; // 256 KB
    static constexpr uint32_t BANK1_SIZE  = sage_attn::L1_BANK_SIZE;    // 256 KB

    // Bank 0 分配
    static constexpr uint32_t Q_DATA_OFFSET = BANK0_START;
    static constexpr uint32_t Q_MAX_BYTES   = 128 * 128 * sizeof(int8_t); // 16 KB
    static constexpr uint32_t Q_SCALE_OFFSET = Q_DATA_OFFSET + Q_MAX_BYTES; // Q 数据区后紧邻 scale（Data+Scale 相邻）

    // P 矩阵在 Bank 0 中的双缓冲槽位（供 Vector 1 写入，Cube 读取；单槽最大 128x128 = 16KB）
    static constexpr uint32_t P_SLOT_BYTES   = 16 * 1024;
    static constexpr uint32_t P_PING_OFFSET  = BANK0_START + 64 * 1024; // 64 KB
    static constexpr uint32_t P_PONG_OFFSET  = P_PING_OFFSET + P_SLOT_BYTES; // 80 KB

    // Bank 1 分配 (Ping-Pong 双缓冲)：单槽 128KB = K 子区 64KB (INT8) + V 子区 64KB (FP8)
    static constexpr uint32_t KV_SLOT_BYTES   = 128 * 1024;
    static constexpr uint32_t KV_V_SUB_OFFSET = KV_SLOT_BYTES / 2;          // 槽内 V 子区偏移 (K 居槽首)
    static constexpr uint32_t KV_PING_OFFSET  = BANK1_START;
    static constexpr uint32_t KV_PONG_OFFSET  = BANK1_START + KV_SLOT_BYTES;
    static constexpr uint32_t V_PING_OFFSET   = KV_PING_OFFSET + KV_V_SUB_OFFSET;
    static constexpr uint32_t V_PONG_OFFSET   = KV_PONG_OFFSET + KV_V_SUB_OFFSET;
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP

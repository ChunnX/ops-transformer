#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include "../ascendc/tiling.h"

/**
 * 官方 CATLASS 组件接入（cann/catlass 仓，CATLASS_ROOT/include 经构建接线注入 -I，
 * 全部编译场景——host 门禁/内核语法门/OPP 打包/BiSheng 设备——均带该 -I，故
 * catlass.hpp 无条件直含，不再保留本地字面值回退分支）：
 * 分形粒度常量以官方 catlass.hpp 为唯一定义源（BYTE_PER_C0=32、
 * C0_NUM_PER_FRACTAL=16、BYTE_PER_FRACTAL=512），下方 static_assert 锁定官方值，
 * 官方漂移即编译期报错。
 * 设备态热路径的 LoadData2D/Fixpipe 搬运为 ND 布局 + Data/Scale 相邻等本算子
 * 自定义契约（官方 TileCopyTla 组件要求 L1 侧 zN 分形布局，不匹配本算子 V0 产出
 * 布局——该数据通路重构须随 camodel 仿真/真机 L-A 循环推进，见设计文档模块边界）。
 */
#include "catlass/catlass.hpp"

#if SAGE_ATTN_KERNEL_API
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
    C2PIPE2GM = 13,
};

enum class HardEvent : uint16_t {
    V_MTE2 = 0,
    MTE2_V = 1,
    V_MTE3 = 2,
    MTE3_V = 3,
    MTE2_MTE1 = 4,
    MTE1_MTE2 = 5,
    MTE1_M = 6,
    M_MTE1 = 7,
    V_S = 8,
    S_V = 9,
    FIX_S = 10,
    S_FIX = 11,
};
} // namespace AscendC
#endif

namespace Catlass::SageAttn {

/**
 * 昇腾硬件事件同步模板（对齐竞品 PR#12430 / common.h 原语，消除冗余配对代码）
 */
template <AscendC::HardEvent E>
__aicore__ inline void Sync() {
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
 * 昇腾 950 DaVinci 硬件分形基本粒度常量
 * 直接引用官方 CATLASS 分形基元常量：
 * - M0/N0: C0_NUM_PER_FRACTAL = 16
 * - K0: BYTE_PER_C0 = 32（INT8/FP8 下 1 字节元素单 C0 包含 32 个元素）
 */
constexpr uint32_t FRACTAL_M0 = Catlass::C0_NUM_PER_FRACTAL;
constexpr uint32_t FRACTAL_N0 = Catlass::C0_NUM_PER_FRACTAL;
constexpr uint32_t FRACTAL_K0_INT8 = Catlass::BYTE_PER_C0;
constexpr uint32_t FRACTAL_K0_FP8 = Catlass::BYTE_PER_C0;

static_assert(FRACTAL_M0 == 16 && FRACTAL_N0 == 16, "fractal MN granularity must be 16x16");
static_assert(FRACTAL_K0_INT8 == 32 && FRACTAL_K0_FP8 == 32,
              "fractal K0 must be 32 for 1-byte elements");
static_assert(LOADDATA2D_REPEAT_BYTES == 512, "LoadData2D repeat unit must be 512B");

/**
 * 通用 2D 分块搬运原语：从片上源位置搬入目标寄存器/缓冲（L1/UB -> L0A/L0B）
 * 统一收敛 LoadData2D 样板代码，杜绝多处重复 TBuffAddr 描述符装配。
 */
template <typename Element, AscendC::TPosition DstPos, AscendC::TPosition SrcPos>
__aicore__ inline void LoadTile2D(uint32_t dst_addr, uint32_t src_addr, uint32_t rows, uint32_t cols, bool if_transpose = false) {
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

    AscendC::LoadData2DParams load_params;
    load_params.repeatTimes = static_cast<uint16_t>((total_bytes + LOADDATA2D_REPEAT_BYTES - 1) / LOADDATA2D_REPEAT_BYTES);
    load_params.srcStride = 0;
    load_params.ifTranspose = if_transpose;
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
 * 通用 Fixpipe L0C -> UB 裸搬原语（SPLIT_M 分段发射）：
 * dav-3510 Fixpipe 透传搬出（quantPre DISABLE，精确位型透传，zN -> ND）。
 * m > 64 时按 64 行双段发射，两段同分区写入 UB。
 */
template <typename ElementC>
__aicore__ inline void CopyL0CToUbSplitM(
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
    fp.subBlockId = false;

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
        AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst0, l0c_tensor, fp);
        AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst1, l0c_tensor[M_SPLIT * 16], fp);
    }
#else
    (void)l0c_src_offset;
    (void)ub_dst_offset;
    (void)m;
    (void)n;
#endif
}

/**
 * Data + Scale 相邻排布契约（FlashMLA 硬件并发核心优化）
 * 将 INT8/FP8 矩阵数据与对应块的 FP32 Scale 在物理内存中紧密相邻排布，
 * 使得每 token 的两次 MTE2 拷贝合并为一次、请求数量直接减半，
 * 彻底化解 16 项 MTE2 Issue Queue 溢出导致的硬件挂起。
 */
template <typename DataType, uint32_t BlockN, uint32_t HeadDim>
struct KvSlotLayoutContract {
    static constexpr uint32_t DATA_ELEMENTS = BlockN * HeadDim;
    static constexpr uint32_t DATA_BYTES = DATA_ELEMENTS * sizeof(DataType);

    // 32 字节对齐
    static constexpr uint32_t DATA_BYTES_ALIGNED = ((DATA_BYTES + 31U) / 32U) * 32U;

    // per_thread 模式下每 64 token 块对应 4 个 FP32 scale，per_warp 为 1 个
    static constexpr uint32_t PER_THREAD_SCALE_COUNT = (BlockN / 64U) * 4U;
    static constexpr uint32_t PER_WARP_SCALE_COUNT = (BlockN / 64U) * 1U;

    static constexpr uint32_t PER_THREAD_SCALE_BYTES = PER_THREAD_SCALE_COUNT * sizeof(float);
    static constexpr uint32_t PER_THREAD_SCALE_BYTES_ALIGNED = ((PER_THREAD_SCALE_BYTES + 31U) / 32U) * 32U;

    static constexpr uint32_t TOTAL_SLOT_BYTES_PER_THREAD = DATA_BYTES_ALIGNED + PER_THREAD_SCALE_BYTES_ALIGNED;

    // Scale 在 Slot 内的相对偏移即为 Data 对齐后的大小
    static constexpr uint32_t SCALE_OFFSET = DATA_BYTES_ALIGNED;
};

/**
 * L1 双 Bank 物理配平契约
 * Bank 0 (0~256KB): Q 矩阵与 Q Scale 常驻
 * Bank 1 (256~512KB): K/V 矩阵及相邻 Scale 双缓冲 Ping-Pong 滚动
 * 彻底消除 Cube MMAD 读取矩阵 A 与矩阵 B 时的 Bank 读通道争用
 */
struct L1DoubleBankContract {
    static constexpr uint32_t BANK0_START = sage_attn::L1_BANK0_OFFSET; // 0 KB
    static constexpr uint32_t BANK0_SIZE  = sage_attn::L1_BANK_SIZE;    // 256 KB
    static constexpr uint32_t BANK1_START = sage_attn::L1_BANK1_OFFSET; // 256 KB
    static constexpr uint32_t BANK1_SIZE  = sage_attn::L1_BANK_SIZE;    // 256 KB

    // Bank 0 分配
    static constexpr uint32_t Q_DATA_OFFSET = BANK0_START;
    static constexpr uint32_t Q_MAX_BYTES   = 128 * 128 * sizeof(int8_t); // 16 KB
    static constexpr uint32_t Q_SCALE_OFFSET = Q_DATA_OFFSET + Q_MAX_BYTES; // Q 数据区后紧邻 scale（Data+Scale 相邻）

    // Bank 1 分配 (Ping-Pong 双缓冲)：单槽 128KB = K 子区 64KB (INT8) + V 子区 64KB (FP8)
    static constexpr uint32_t KV_SLOT_BYTES   = 128 * 1024;
    static constexpr uint32_t KV_V_SUB_OFFSET = KV_SLOT_BYTES / 2;          // 槽内 V 子区偏移 (K 居槽首)
    static constexpr uint32_t KV_PING_OFFSET  = BANK1_START;
    static constexpr uint32_t KV_PONG_OFFSET  = BANK1_START + KV_SLOT_BYTES;
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP

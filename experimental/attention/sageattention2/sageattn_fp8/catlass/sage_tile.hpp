#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_TILE_HPP

#include <cstdint>
#include <cstddef>
#include "../ascendc/tiling.h"

#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#endif

namespace Catlass::SageAttn {


/**
 * LoadData 2D 单 repeat 搬运字节数（512B，CANN LoadData2DParams.repeatTimes 的计量单位）
 */
constexpr uint32_t LOADDATA2D_REPEAT_BYTES = 512;

/**
 * 昇腾 950 DaVinci 硬件分形基本粒度常量
 * INT8 / FP8 下 K 维分形步长为 32 (32B)，FP16 下为 16 (32B)
 */
constexpr uint32_t FRACTAL_M0 = 16;
constexpr uint32_t FRACTAL_N0 = 16;
constexpr uint32_t FRACTAL_K0_INT8 = 32;
constexpr uint32_t FRACTAL_K0_FP8 = 32;
constexpr uint32_t FRACTAL_K0_FP16 = 16;

/**
 * 分形布局标签定义（对齐 CATLASS 规范与 950PR 专用分形）
 */
struct Layout_zZ {
    static constexpr const char* name = "zZ";
};

struct Layout_nZ {
    static constexpr const char* name = "nZ";
};

struct Layout_zN {
    static constexpr const char* name = "zN";
};

struct Layout_ND {
    static constexpr const char* name = "ND";
};

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

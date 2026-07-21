/*!
 * \file engram_fetch_train.h
 * \brief engram_fetch_train 算子公共头文件
 */

#ifndef ENGRAM_FETCH_TRAIN_H
#define ENGRAM_FETCH_TRAIN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

constexpr uint32_t ENGRAM_TRAIN_MAX_QP_SIZE = 1024;
struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[ENGRAM_TRAIN_MAX_QP_SIZE];
    uint64_t hcommHandle_[ENGRAM_TRAIN_MAX_QP_SIZE];
};

namespace Mc2Kernel {
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t TILE_BYTES = 32U * 1024U;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t STATE_OFFSET = 32U;  // win 状态区/计数区每 sender slot 32B
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t NUM_SLOTS = 4U;       // quad-buffer 槽位数
constexpr uint32_t INDICES_RATIO = 50U;  // indices 区占数据区的 1/50

using namespace AscendC;
} // namespace Mc2Kernel

#endif

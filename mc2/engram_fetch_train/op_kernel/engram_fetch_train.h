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
#include "../../engram_fetch/op_kernel/engram_fetch_utils.h"

namespace Mc2Kernel {
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t STATE_OFFSET = 32U;  // win 状态区/计数区每 sender slot 32B
constexpr uint32_t NUM_SLOTS = 4U;       // quad-buffer 槽位数
constexpr uint32_t INDICES_RATIO = 50U;  // indices 区占数据区的 1/50
constexpr uint32_t WIN_REGION_COUNT = 8U;  // win 区固定区域数
constexpr uint64_t UB_RESERVED_SIZE = 8U * 1024U;  // UB 预留空间

using namespace AscendC;
} // namespace Mc2Kernel

#endif

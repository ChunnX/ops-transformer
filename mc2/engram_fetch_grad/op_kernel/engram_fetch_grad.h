/*!
 * \file engram_fetch_grad.h
 * \brief engram_fetch_grad 算子公共头文件
 */

#ifndef ENGRAM_FETCH_GRAD_H
#define ENGRAM_FETCH_GRAD_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../engram_fetch/op_kernel/engram_fetch_utils.h"

namespace Mc2Kernel {
constexpr uint32_t STATE_OFFSET = 32U;  // win 状态区/计数区每 sender slot 32B
constexpr uint32_t NUM_SLOTS = 4U;       // quad-buffer 槽位数

using namespace AscendC;
} // namespace Mc2Kernel

#endif

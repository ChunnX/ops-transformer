/**
 * CANN 8.5 的 asc/include/basic_api 下无 kernel_basic_intf.h（CANN 9+ 提供）。
 * 与 9.x 一致：先 Cube 后 Vec，保证 TPipe 等类型完整定义（勿用 kernel_operator_intf 单头替代）。
 */
#ifndef ASCENDC_MODULE_BASIC_INTERFACE_H
#define ASCENDC_MODULE_BASIC_INTERFACE_H

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include "stub_def.h"
#include "stub_fun.h"
#endif

#include "kernel_cube_intf.h"
#include "kernel_vec_intf.h"

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Vendored from CANN 9.x asc/include/basic_api for CANN 8.5 (file absent there).
 */
#ifndef ASCENDC_MODULE_CUBE_INTERFACE_H
#define ASCENDC_MODULE_CUBE_INTERFACE_H

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include "stub_def.h"
#include "stub_fun.h"
#endif

#include "kernel_type.h"
#include "kernel_tpipe.h"
#include "kernel_tensor.h"
#include "kernel_prof_trace_intf.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_fixpipe_intf.h"
#include "kernel_operator_dump_tensor_intf.h"
#include "kernel_operator_mm_intf.h"
#include "kernel_operator_gemm_intf.h"
#include "kernel_operator_conv2d_intf.h"
#include "kernel_operator_common_intf.h"
#include "kernel_operator_scalar_intf.h"
#include "kernel_operator_limits_intf.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_atomic_intf.h"
#include "kernel_operator_set_atomic_intf.h"
#include "kernel_operator_cache_intf.h"
#include "kernel_operator_utils_intf.h"

#endif

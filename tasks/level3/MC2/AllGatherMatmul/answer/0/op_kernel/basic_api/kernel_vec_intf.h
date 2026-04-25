/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Vendored from CANN 9.x asc/include/basic_api for CANN 8.5 (file absent there).
 */
#ifndef ASCENDC_MODULE_VEC_INTERFACE_H
#define ASCENDC_MODULE_VEC_INTERFACE_H

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include "stub_def.h"
#include "stub_fun.h"
#endif

#include "kernel_type.h"
#include "kernel_tpipe.h"
#include "kernel_tensor.h"
#include "kernel_prof_trace_intf.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_dump_tensor_intf.h"
#include "kernel_operator_common_intf.h"
#include "kernel_operator_vec_binary_intf.h"
#include "kernel_operator_vec_binary_scalar_intf.h"
#include "kernel_operator_vec_duplicate_intf.h"
#include "kernel_operator_vec_gather_mask_intf.h"
#include "kernel_operator_vec_vconv_intf.h"
#include "kernel_operator_scalar_intf.h"
#include "kernel_operator_vec_reduce_intf.h"
#include "kernel_operator_proposal_intf.h"
#include "kernel_operator_determine_compute_sync_intf.h"
#include "kernel_operator_vec_transpose_intf.h"
#include "kernel_operator_vec_gather_intf.h"
#include "kernel_operator_vec_scatter_intf.h"
#include "kernel_operator_vec_brcb_intf.h"
#include "kernel_operator_vec_cmpsel_intf.h"
#include "kernel_operator_vec_mulcast_intf.h"
#include "kernel_operator_vec_bilinearinterpolation_intf.h"
#include "kernel_operator_vec_createvecindex_intf.h"
#include "kernel_operator_vec_ternary_scalar_intf.h"
#include "kernel_operator_vec_unary_intf.h"
#include "kernel_operator_vec_vpadding_intf.h"
#include "kernel_operator_limits_intf.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_atomic_intf.h"
#include "kernel_operator_set_atomic_intf.h"
#include "kernel_operator_cache_intf.h"
#include "kernel_operator_utils_intf.h"

#endif

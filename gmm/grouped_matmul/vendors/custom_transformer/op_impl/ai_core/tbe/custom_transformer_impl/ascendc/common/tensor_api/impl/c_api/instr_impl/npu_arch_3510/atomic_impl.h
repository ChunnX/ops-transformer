/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_ATOMIC_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_ATOMIC_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_float16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_float_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_bfloat_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_float_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_none_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_get_store_atomic_config_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_store_atomic_config_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_float_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_float16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_bfloat_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_bfloat_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_float16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int8_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int8_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int8_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int16_impl.h"

// ==========asc_set_atomic_add_float16==========
__aicore__ inline void asc_set_atomic_add_float16() { asc_set_atomic_add_float16_impl(); }

// ==========asc_set_atomic_max_float==========
__aicore__ inline void asc_set_atomic_max_float() { asc_set_atomic_max_float_impl(); }
//===========asc_set_atomic_bf16===========
__aicore__ inline void asc_set_atomic_add_bfloat() { asc_set_atomic_add_bfloat_impl(); }

//===========asc_set_atomic_min_float===========
__aicore__ inline void asc_set_atomic_min_float() { asc_set_atomic_min_float_impl(); }

// ==========asc_set_atomic_min_bfloat==========
__aicore__ inline void asc_set_atomic_min_bfloat() { asc_set_atomic_min_bfloat_impl(); }

[[deprecated("NOTICE: asc_set_store_atomic_config_v2 is deprecated."
             "Please use asc_atomic_add instead for atomic add operation.")]] __aicore__ inline void
asc_set_store_atomic_config_v2(uint16_t type, uint16_t op)
{
    asc_set_store_atomic_config_v2_impl(type, op);
}

__aicore__ inline void asc_set_atomic_add_float() { asc_set_atomic_add_float_impl(); }

__aicore__ inline void asc_set_atomic_max_float16() { asc_set_atomic_max_float16_impl(); }
//===========asc_set_atomic_none===========
__aicore__ inline void asc_set_atomic_none() { asc_set_atomic_none_impl(); }

//===========asc_get_store_atomic_config===========
[[deprecated("NOTICE: asc_get_store_atomic_config is deprecated on Ascend 950PR/Ascend 950DT."
             "Please use asc_atomic_add instead for atomic add operation.")]] __aicore__ inline void
asc_get_store_atomic_config(asc_store_atomic_config& config)
{
    asc_get_store_atomic_config_impl(config);
}

// ==========asc_set_atomic_max_bfloat==========
__aicore__ inline void asc_set_atomic_max_bfloat() { asc_set_atomic_max_bfloat_impl(); }

// ==========asc_set_atomic_min_float16==========
__aicore__ inline void asc_set_atomic_min_float16() { asc_set_atomic_min_float16_impl(); }

// ==========asc_set_atomic_add_int==========
__aicore__ inline void asc_set_atomic_add_int() { asc_set_atomic_add_int_impl(); }

// ==========asc_set_atomic_add_int8==========
__aicore__ inline void asc_set_atomic_add_int8() { asc_set_atomic_add_int8_impl(); }

// ==========asc_set_atomic_add_int16==========
__aicore__ inline void asc_set_atomic_add_int16() { asc_set_atomic_add_int16_impl(); }

// ==========asc_set_atomic_max_int==========
__aicore__ inline void asc_set_atomic_max_int() { asc_set_atomic_max_int_impl(); }

// ==========asc_set_atomic_max_int8==========
__aicore__ inline void asc_set_atomic_max_int8() { asc_set_atomic_max_int8_impl(); }

// ==========asc_set_atomic_max_int16==========
__aicore__ inline void asc_set_atomic_max_int16() { asc_set_atomic_max_int16_impl(); }

// ==========asc_set_atomic_min_int==========
__aicore__ inline void asc_set_atomic_min_int() { asc_set_atomic_min_int_impl(); }

// ==========asc_set_atomic_min_int8==========
__aicore__ inline void asc_set_atomic_min_int8() { asc_set_atomic_min_int8_impl(); }

// ==========asc_set_atomic_min_int16==========
__aicore__ inline void asc_set_atomic_min_int16() { asc_set_atomic_min_int16_impl(); }

#endif

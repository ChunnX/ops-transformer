/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_combine_gates_struct.h
 * \brief MoeTutelCombineGates template argument (TPL) declaration.
 */

#ifndef MOE_TUTEL_COMBINE_GATES_STRUCT_H_
#define MOE_TUTEL_COMBINE_GATES_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros, GET_TPL_TILING_KEY

// ---------------------------------------------------------------------------
// dtype mode constants for TPL template dispatch
//   Values are identical to the reference implementation's BF16_MODE / FP32_MODE
//   / FP16_MODE (op_host/moe_tutel_combine_gates_tiling.cpp:24-28) — A10 keeps the
//   historical tilingKey semantics (0 = bf16 default, 1 = fp32, 2 = fp16).
// ---------------------------------------------------------------------------
#define MOE_TUTEL_COMBINE_GATES_BF16_MODE 0
#define MOE_TUTEL_COMBINE_GATES_FP32_MODE 1
#define MOE_TUTEL_COMBINE_GATES_FP16_MODE 2

// ---------------------------------------------------------------------------
// ASCENDC_TPL_ARGS_DECL — declares the compile-time template argument
//
//   ASCENDC_TPL_UINT_DECL(DTYPE_MODE, 2, ASCENDC_TPL_UI_LIST, ...)
//     → template parameter "DTYPE_MODE", 2 bits wide, list-valued
//     → allowed values: 0 (bf16) / 1 (fp32) / 2 (fp16)
//
//   Declaration order = bit-position order; with a single parameter it occupies
//   the lowest 2 bits, and because the list is [0, 1, 2] the encoded key equals
//   the value (TilingKey.md 注 1). Changing the
//   list order would silently change the tilingKey numbers and break A10.
// ---------------------------------------------------------------------------
ASCENDC_TPL_ARGS_DECL(MoeTutelCombineGates,
                      ASCENDC_TPL_UINT_DECL(DTYPE_MODE, 2, ASCENDC_TPL_UI_LIST, MOE_TUTEL_COMBINE_GATES_BF16_MODE,
                                            MOE_TUTEL_COMBINE_GATES_FP32_MODE, MOE_TUTEL_COMBINE_GATES_FP16_MODE));

// ---------------------------------------------------------------------------
// ASCENDC_TPL_SEL — generates template specialization selectors
//
// Each ASCENDC_TPL_ARGS_SEL line creates a specialization for one DTYPE_MODE
// value; the compiler generates a separate kernel binary per specialization and
// the runtime selects it with the host-side tiling key.
// ---------------------------------------------------------------------------
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, MOE_TUTEL_COMBINE_GATES_BF16_MODE)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, MOE_TUTEL_COMBINE_GATES_FP32_MODE)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, MOE_TUTEL_COMBINE_GATES_FP16_MODE)));

#endif // MOE_TUTEL_COMBINE_GATES_STRUCT_H_

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_arch35_reuse.h
 * \brief arch35(A5) 复用清单:以下 kernel 头与 A2 逻辑完全一致,直接复用 A2 实现,经命名空间桥接进
 *        FfnWbBatchingArch35(A2/A3 零改动)。arch35 仅保留有真实代次差异的 *_arch35.h 独立实现。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_ARCH35_REUSE_H
#define OP_KERNEL_ARCH35_FFN_WB_ARCH35_REUSE_H
#include "../ffn_wb_common.h"
#include "../ffn_wb_sort_base.h"
#include "../ffn_wb_sort_mrgsort.h"
#include "../ffn_wb_sort_mrgsort_out.h"
#include "../ffn_wb_sort_one_core.h"
#include "../ffn_wb_get_schedule_context.h"
#include "../ffn_wb_gather_out_all.h"
#include "../ffn_wb_scan_token_info.h"
#include "../ffn_wb_scan_sort_one_core.h"
#include "../ffn_wb_scan_get_valid_experts.h"
namespace FfnWbBatchingArch35 {
using namespace FfnWbBatching;
// op 辅助函数与 AscendC 同名,using-声明取优先解析,消除二义。
using FfnWbBatching::Align;
using FfnWbBatching::Ceil;
using FfnWbBatching::CeilDiv;
using FfnWbBatching::Max;
using FfnWbBatching::Min;
using FfnWbBatching::PowerOfFourCeil;
using FfnWbBatching::SetWaitFlag;
} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_ARCH35_REUSE_H

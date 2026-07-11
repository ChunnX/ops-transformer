/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is part of the CANN Open Software (see LICENSE).
 */
// arch35(A5) 复用清单:以下 11 个 kernel 头与 A2 逻辑完全一致,直接复用 A2 实现,
// 经命名空间桥接进 FfnWbBatchingArch35(A2/A3 零改动)。arch35 仅保留有真实代次差异的
// 4 个文件(*_arch35.h)独立实现。
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
#include "../ffn_wb_scan_group_listing_one_core.h"
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
}
#endif // OP_KERNEL_ARCH35_FFN_WB_ARCH35_REUSE_H

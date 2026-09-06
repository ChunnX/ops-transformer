/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_type.h
 * \brief qfa 算子 traits 与变长序列工具（对齐 flash_attn utils/flash_attn_type.h 的组织方式；
 *        计算原理为 qfa 自有：NTD/PA 分支、变长 KV ListTensorDesc）
 */

#ifndef QUANT_FLASH_ATTN_TYPE_H_
#define QUANT_FLASH_ATTN_TYPE_H_

#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"

namespace BaseApi {

// 对齐 fa FAType：单 traits 携带 kernel/block/Dummy 全部模板参数
template <typename INPUT_T, typename OUTPUT_T, LayOutTypeEnum LAYOUT_T, LayOutTypeEnum LAYOUT_KV,
          LayOutTypeEnum LAYOUT_OUT, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KV_LAYOUT_TYPE = 0, bool HAS_MASK = false,
          bool IS_FD = false, bool USE_DN = false, bool IS_D_ALIGNED = true>
struct QFAType {
    using inputType = INPUT_T;
    using outputType = OUTPUT_T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT_T;
    static constexpr LayOutTypeEnum kvLayout = LAYOUT_KV;
    static constexpr LayOutTypeEnum attnOutLayout = LAYOUT_OUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr uint8_t kvLayoutType = KV_LAYOUT_TYPE;
    static constexpr bool pageAttention = (KV_LAYOUT_TYPE > 0);
    static constexpr bool hasMask = HAS_MASK;
    static constexpr bool isFd = IS_FD;
    static constexpr bool useDn = USE_DN;
    static constexpr bool isDAligned = IS_D_ALIGNED;
};

// 对齐 fa SeqLensTool：持双 parser + 编译期按 layout 分派（qfa 计算原理：NTD/PA 分支语义）
// INCLUDE_NTD：NTD 布局是否走 cuParser（路径间计算原理差异：fp8 为 TND||NTD，mxfp8/hif8 仅 TND）
template <LayOutTypeEnum LAYOUT_T, bool IS_PA, bool INCLUDE_NTD, typename SEQLEN_T, ActualSeqLensMode PARSER_MODE>
class QFASeqLensTool {
public:
    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT_T == LayOutTypeEnum::LAYOUT_TND || (INCLUDE_NTD && LAYOUT_T == LayOutTypeEnum::LAYOUT_NTD));
    // PA+TND 时 cuParser 按 BY_BATCH 形态初始化（flag=false），与原 kernel 侧类型一致
    ActualSeqLensParser<PARSER_MODE, SEQLEN_T, (IS_TND_LAYOUT && !IS_PA)> cuSeqLensParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, SEQLEN_T> seqUsedParser;

    __aicore__ inline void Init(__gm__ uint8_t *cuSeqLensGmAddr, uint32_t cuSeqLensDims, __gm__ uint8_t *seqUsedGmAddr,
                                uint32_t seqUsedDims, uint64_t defaultSeqUsedVal)
    {
        if constexpr (IS_TND_LAYOUT) {
            if constexpr (IS_PA) {
                // PA+TND: 序列长度取自 seqUsed（按 batch）
                cuSeqLensParser.Init(seqUsedGmAddr, seqUsedDims, defaultSeqUsedVal);
            } else {
                // TND/NTD: 累积和解析（+1 含末元素）
                cuSeqLensParser.Init(cuSeqLensGmAddr, cuSeqLensDims + 1, seqUsedGmAddr, seqUsedDims);
            }
        } else {
            seqUsedParser.Init(seqUsedGmAddr, seqUsedDims, defaultSeqUsedVal);
        }
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx)
    {
        if constexpr (IS_TND_LAYOUT) {
            return cuSeqLensParser.GetActualSeqLength(bIdx);
        } else {
            return seqUsedParser.GetActualSeqLength(bIdx);
        }
    }
};

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_TYPE_H_

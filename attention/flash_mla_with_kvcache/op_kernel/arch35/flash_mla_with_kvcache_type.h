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
 * \file flash_mla_with_kvcache_type.h
 * \brief arch35 flash_mla_with_kvcache 编译期模板参数聚合类型（镜像 flash_attn 的
 *        utils/flash_attn_type.h FAType 组织模式，语义命名保留 MLA 语境）。
 *        cube/vec/fd 三类 block 参数集不一致（vec 多 OUTPUT_T/outLayout/hasMask/isFd，
 *        cube 无 mask/out 概念），故分别为每类建聚合类型；isFd 是 kernel 级模板参数
 *        （tiling key 第 5 字段），FD block 无 isFd（FD 运行时由 metadata mLen>0 决定），
 *        因此 FATypeFd 不含 flashDecode 成员。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TYPE_H_
#define FLASH_MLA_WITH_KVCACHE_TYPE_H_

#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"

namespace BaseApi {

// Cube block 参数聚合（8 参：INPUT_T、T、layout、s1、s2、d、dv、KvLayoutType）。
// PAGE_ATTENTION 派生规则与 block 内 (KvLayoutType > 0) 一致（四件套 KV_FORMAT 同源）。
template <typename INPUT_T, typename T, LayOutTypeEnum LAYOUT = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
struct FATypeCube {
    using inputType = INPUT_T;
    using mmType = T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
};

// Vec block 参数聚合（12 参：INPUT_T、T、OUTPUT_T、layout、outLayout、s1、s2、d、dv、
// hasMask、KvLayoutType、flashDecode）。isFd 在该处仅作为 vec 块内 FLASH_DECODE 常量来源，
// kernel 级 isFd 模板参数仍独立保留（见文件头注释）。
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum LAYOUT = LayOutTypeEnum::None,
          LayOutTypeEnum OUT_LAYOUT = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool HAS_MASK = false, uint8_t KvLayoutType = 0,
          bool FLASH_DECODE = false>
struct FATypeVec {
    using inputType = INPUT_T;
    using mmType = T;
    using outputType = OUTPUT_T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT;
    static constexpr LayOutTypeEnum attnOutLayout = OUT_LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
    static constexpr bool flashDecode = FLASH_DECODE;
};

// FD (flashdecode) block 参数聚合（11 参，无 flashDecode：FD 由 metadata mLen>0 运行时使能，
// kernel 级 isFd 模板参数不进 FD block / 聚合类型）。
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum LAYOUT = LayOutTypeEnum::None,
          LayOutTypeEnum OUT_LAYOUT = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool HAS_MASK = false, uint8_t KvLayoutType = 0>
struct FATypeFd {
    using inputType = INPUT_T;
    using mmType = T;
    using outputType = OUTPUT_T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT;
    static constexpr LayOutTypeEnum attnOutLayout = OUT_LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
};

} // namespace BaseApi

#endif // FLASH_MLA_WITH_KVCACHE_TYPE_H_

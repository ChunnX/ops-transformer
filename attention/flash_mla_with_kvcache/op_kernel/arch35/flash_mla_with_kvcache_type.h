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
#include "../../../common/op_kernel/memcopy/parser.h"
#include "flash_mla_with_kvcache_gm_format.h"

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

// kernel 持有的 seq-lens 解析工具（镜像 flash_attn utils/flash_attn_type.h:134-155 SeqLensTool
// 的"kernel 持有工具、block 构造收引用"组织方式）。与 FA 的关键差异（Oracle 约束）：
// 1) 保留 q 侧 seqused_q==nullptr 双分支——FA 的 SeqLensTool 无 null-seqused 概念，直接移植
//    会错误合并空指针分支并改变 seqused=NULL/0 槽位边角语义；
// 2) q 侧 ACCUM 带首零头（WITH_ZERO_HEAD=true，flash_attn cu_seqlens 约定）；
// 3) kv 侧 BY_BATCH（cache_seqlens 非累加）。
// seq-lens 为 INT32 接口（R2/D26），ACTLEN_T=uint32_t；cu_seqlens_q [b+1] / seqused_q [b] / cache_seqlens [b]。
template <ActualSeqLensMode Q_MODE, ActualSeqLensMode KV_MODE>
class FlashMlaSeqLensTool {
public:
    using SEQLEN_T = uint32_t;

    // cu_seqlens_q / cache_seqlens 的 GM 张量（offsetCalculator 直接消费；kernel Init 内绑定
    // 后各 block 只读引用，替代 block 各自持副本）
    GlobalTensor<SEQLEN_T> actualSeqLengthsGmQ;
    GlobalTensor<SEQLEN_T> actualSeqLengthsGm;

    ActualSeqLensParser<Q_MODE, SEQLEN_T, true> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE, SEQLEN_T, false> kvActSeqLensParser;

    __aicore__ inline void InitQ(__gm__ uint8_t *cuSeqlensQAddr, __gm__ uint8_t *sequsedQAddr,
                                 uint32_t actualSeqLenSize, uint64_t s1Size)
    {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ SEQLEN_T *)cuSeqlensQAddr, actualSeqLenSize);
        if (sequsedQAddr == nullptr) {
            // 双分支 1（保留，不压平）：无 seqused_q → cu_seqlens_q 差分语义（含首零头累加）
            qActSeqLensParser.Init(actualSeqLengthsGmQ, actualSeqLenSize, s1Size);
        } else {
            // 双分支 2：有 seqused_q → 按逐 batch 实际长度取值（seqused 长度 = b = actualSeqLenSize - 1）
            qActSeqLensParser.Init((__gm__ uint8_t *)cuSeqlensQAddr, actualSeqLenSize,
                                   (__gm__ uint8_t *)sequsedQAddr, actualSeqLenSize - 1);
        }
    }

    __aicore__ inline void InitKv(__gm__ uint8_t *cacheSeqlensAddr, uint32_t actualSeqLenKVSize, uint64_t s2Size)
    {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ SEQLEN_T *)cacheSeqlensAddr, actualSeqLenKVSize);
        kvActSeqLensParser.Init(actualSeqLengthsGm, actualSeqLenKVSize, s2Size);
    }
};

} // namespace BaseApi

#endif // FLASH_MLA_WITH_KVCACHE_TYPE_H_

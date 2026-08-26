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
 * \file flash_mla_with_kvcache.cpp
 * \brief flash_mla_with_kvcache Kernel 唯一入口；按 tiling key 编译期推导模板参数，
 *        分发 FlashAttentionNoQuantMlaKernel（由 fia_kernel_noquant_mla.h 复制改名）。
 *        结构照 flash_attn.cpp：__global__ 层直接解析 tiling key 四模板参数
 *        (InOutLayoutType, KvLayoutType, HasAttenMask, Config) → q/out 布局与模板块大小，
 *        无 isFd（FD 能力恒实例化，运行时由 metadata 各 section mLen>0 驱动）。
 *        q 仅支持 BSND/BNSD/TND 三布局，不支持输出转置（out 布局恒等于 q）；
 *        rope 已并入 q/k_cache 576 宽单张量，无独立 rope 输入。
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"

#include "arch35/flash_mla_with_kvcache_template_tiling_key.h"
// 入口直连 include 面保持最小：公共符号头（util_regbase.h / infer_flash_attention_comm.h，
// S1/S2/DTemplateType、LayOutTypeEnum 等）、软名前缀头（vf_* / attenmask_gs1.h）与
// memory_copy_arch35.h（CopyQueryGmToL1 等）均由实际消费它们的 block/type 头随用随含；
// 与 flash_attn.cpp 同构——FA 亦只直接 include template_tiling_key + 共享 common，其余经 block 头传递。
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#else
#include "../common/arch35/flash_attention_score_common_regbase_arch35.h"
#endif
#include "adv_api/activation/softmax.h"
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_tiling_regbase_arch35.h")
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase_arch35.h"
#else
#include "../common/arch35/flash_attention_score_tiling_regbase_arch35.h"
#endif
#include "arch35/flash_mla_with_kvcache_type.h"
using namespace optiling; // tiling_regbase 提供；kernel 头 static_assert 需在解析期可见 optiling 域（与 FA 入口同款）
#include "arch35/flash_mla_with_kvcache_kernel_noquant_mla.h"

using namespace FaVectorApi; // 命名空间声明仅在 vf 前缀头（经 kernel 头传递）之后可用

using namespace AscendC;
using namespace AttentionCommon; // flash_mla_with_kvcache_kernel_noquant_mla.h 内 FLASH_MLA_WITH_KVCACHE_LAYOUT
                                 // 等符号（继承自 fia 结构）

// ============ 入口 layout 推导（flash_attn.cpp 风格：if-constexpr 直接映射，无表查找）============
// InOutLayoutType → q 布局。ql out 布局恒等于 q（MLA 不支持输出转置），无独立 GetOutLayout。
template <uint8_t inOutLayoutType>
__aicore__ inline constexpr LayOutTypeEnum GetQueryLayoutMla()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_TND),
                  "GetQueryLayoutMla fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND) {
        return LayOutTypeEnum::LAYOUT_BSH;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD) {
        return LayOutTypeEnum::LAYOUT_BNSD;
    } else { // InOutLayoutType_TND
        return LayOutTypeEnum::LAYOUT_TND;
    }
}

// KvLayoutType → 是否 PA 分页（NO_PA=0 不实例化，真值仅为静态约束）
template <uint8_t kvLayoutType>
__aicore__ inline constexpr bool IsPageAttentionMla()
{
    static_assert(kvLayoutType <= KvLayoutType_PA_NZ, "IsPageAttentionMla fail, kvLayoutType is incorrect");
    return (kvLayoutType != KvLayoutType_NO_PA);
}

// kv 存储/块内格式推导（PA 三格式；layout 仅用于 NO_PA 连续分支，此处 PA 恒真）
template <LayOutTypeEnum layout, uint8_t kvLayoutType, bool isPa>
__aicore__ inline constexpr GmFormat GetKvLayoutMla()
{
    return BaseApi::GetKVGmFormat<layout, kvLayoutType, isPa>();
}

// —— 实例化组合语义互锁（tiling key SEL 与块消费一致性；flash_attn 同款静态约束）——
static_assert(GetQueryLayoutMla<InOutLayoutType_BSND>() == LayOutTypeEnum::LAYOUT_BSH);
static_assert(GetQueryLayoutMla<InOutLayoutType_BNSD>() == LayOutTypeEnum::LAYOUT_BNSD);
static_assert(GetQueryLayoutMla<InOutLayoutType_TND>() == LayOutTypeEnum::LAYOUT_TND);
static_assert(IsPageAttentionMla<KvLayoutType_PA_NZ>() == true);
static_assert(IsPageAttentionMla<KvLayoutType_NO_PA>() == false);
static_assert(GetKvLayoutMla<LayOutTypeEnum::LAYOUT_TND, KvLayoutType_PA_NZ, true>() == GmFormat::PA_NZ);

// 解析 tiling key，分发 MLA 四件套（镜像 flash_attn.cpp 的四参解析：ConfigValue[config] 查表
// 得 s1/s2/d/dv 模板块大小；InOutLayoutType/KvLayoutType 由上述函数推导）
template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint8_t config>
inline __aicore__ void run_flash_mla_with_kvcache_noquant_kernel(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cacheSeqlens, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *metadata,
    __gm__ uint8_t *tiling)
{
    fa_base_matmul::idCounterNum = 0;
    constexpr LayOutTypeEnum inputLayoutType = GetQueryLayoutMla<inOutLayoutType>();
    constexpr LayOutTypeEnum outputLayoutType = inputLayoutType; // MLA 不支持输出转置
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    // —— 模板参数聚合（FAType 式，镜像 flash_attn utils/flash_attn_type.h:113-132；
    // cube/vec/fd 参数集不一致 → 各建聚合类型，见 arch35/flash_mla_with_kvcache_type.h）——
    using FA_T_Cube = BaseApi::FATypeCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                          dTemplateType, dVTemplateType, KvLayoutType>;
    using CubBlockNormal = BaseApi::FlashMlaWithKvcacheNoQuantMlaBlockCube<FA_T_Cube>;
    using CubBlockDummy = BaseApi::FlashMlaWithKvcacheNoQuantMlaBlockCubeDummy<FA_T_Cube>;

    // VecFaBlockType（无 isFd；FD 运行时由 metadata mLen>0 决定）
    using FA_T_Vec = BaseApi::FATypeVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                        s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;
    using VecFaBlockNormal = BaseApi::FlashMlaWithKvcacheNoQuantMlaBlockVec<FA_T_Vec>;
    using VecFaBlockDummy = BaseApi::FlashMlaWithKvcacheNoQuantMlaBlockVecDummy<FA_T_Vec>;

    // VecFdBlockType（FD 恒实例化；运行时 FD 使能 = metadata mLen > 0；无 FD 专属 Dummy ——
    // AIC TU 的 FD 槽复用 vec 哑块，与 flash_attn.cpp:166 同构）
    using FA_T_Fd = BaseApi::FATypeFd<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                      s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;
    using VecFdBlockNormal = BaseApi::FlashMlaWithKvcacheBlockVecFlashDecodeMla<FA_T_Fd>;

    // 编译期 core-type 选型（与 flash_attn.cpp:150-169 的 #ifdef __DAV_C310_CUBE__ 同构）：
    // 该宏由 950 编译在 AIC TU 定义、AIV TU 不定义（mla 编译产物实测确认），语义与
    // g_coreType==AscendC::AIC/AIV 等价 —— AIC 核取 cube 实块 + vec 哑块（FD 槽复用，
    // 与 flash_attn 的 KernelNd<..., VecDummy, VecDummy> 一致），AIV 核反之。
#ifdef __DAV_C310_CUBE__
    using CubBlock = CubBlockNormal;
    using VecFaBlock = VecFaBlockDummy;
    using VecFdBlock = VecFaBlockDummy;
#else
    using CubBlock = CubBlockDummy;
    using VecFaBlock = VecFaBlockNormal;
    using VecFdBlock = VecFdBlockNormal;
#endif

    // KernelType
    using Kernel = BaseApi::FlashAttentionNoQuantMlaKernel<CubBlock, VecFaBlock, VecFdBlock>;

    // 根因（静态图）：固定 shape 下 tiling 是编译期常量字节数组而非 __gm__ buffer，
    // 不能直接 reinterpret_cast 成结构体指针访问，必须先拷贝到栈局部结构体对象再取指针。
    // 该宏由编译器(generate_tiling_code.py)按动态/静态图自动生成两套展开：
    //   动态图 → (__gm__ FlashMlaWithKvcacheTilingData*)tiling 直接强转，零拷贝；
    //   静态图 → convert_from_bytes 把常量数组拷成栈局部对象，dst_ptr 取其地址。
    GET_TILING_DATA_MEMBER(optiling::FlashMlaWithKvcacheTilingData, baseTiling, baseTilingIn, tiling);
    const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;

    Kernel op;
    // 无独立 queryRope/keyRope；q/k_cache 为 576 宽单张量（nope512+rope64 合并）。
    // value 与 key 同一 GM 指针（k_cache 的 nope 段即 V，k==v 语义）。
    // metadata（GM 输入，AICPU 多 section 布局）直接透传，section 循环在 Process 内。
    // 静态 tensor 模型：block 内自管 buffer（与 flash_attn Init 15 参风格一致，入口仅接地址）
    op.Init(query, key, value, attenMask, cuSeqlensQ, sequsedQ, cacheSeqlens, blockTable, softmaxLse, attentionOut,
            workspace, metadata, tilingData);
    op.Process();
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
// 形参序必须吃齐 def 输入序 cache_seqlens(3)/cu_seqlens_q(4)/seqused_q(5)（运行时按 def 序绑槽位；
// 曾用 FIA 内部序致 cacheSeqlens 收到 seqused_q=NULL 槽 → kv 长度解析解引用 GM 0 越界）。
__global__ __aicore__ void flash_mla_with_kvcache(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                  __gm__ uint8_t *blockTable, __gm__ uint8_t *cacheSeqlens,
                                                  __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                  __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
                                                  __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse,
                                                  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashMlaWithKvcacheTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // 编译宏名须对齐 asc_opc 按输入名派生的 ORIG_DTYPE_{Q,K_CACHE,ATTN_OUT}
    // （fia 内部名 ORIG_DTYPE_QUERY/KEY/ATTENTION_OUT 未被定义 → 恒走 bf16 分支，fp16 .o 实为 bf16）。
    // DT_* 在 device TU 为枚举非宏，故比数值（FP16==1/BF16==27，实测确认）。
#define MLA_DT_FLOAT16 1
#define MLA_DT_BF16 27
#if (defined(ORIG_DTYPE_Q) && ORIG_DTYPE_Q == MLA_DT_FLOAT16 && defined(ORIG_DTYPE_K_CACHE) && \
     ORIG_DTYPE_K_CACHE == MLA_DT_FLOAT16 && defined(ORIG_DTYPE_ATTN_OUT) && ORIG_DTYPE_ATTN_OUT == MLA_DT_FLOAT16)
    using INPUT_T = half;
    using OUT_T = half;
#elif (defined(ORIG_DTYPE_Q) && ORIG_DTYPE_Q == MLA_DT_BF16 && defined(ORIG_DTYPE_K_CACHE) && \
       ORIG_DTYPE_K_CACHE == MLA_DT_BF16 && defined(ORIG_DTYPE_ATTN_OUT) && ORIG_DTYPE_ATTN_OUT == MLA_DT_BF16)
    using INPUT_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#else
#error "flash_mla_with_kvcache: ORIG_DTYPE_* macros must be all fp16 or all bf16"
#endif

    run_flash_mla_with_kvcache_noquant_kernel<INPUT_T, OUT_T, inOutLayoutType, KvLayoutType, hasAttenMask, config>(
        query, key, key, attnMask, cuSeqlensQ, sequsedQ, cacheSeqlens, blockTable, softmaxLse, attnOut, user, metadata,
        tiling);

    AscendC::PipeBarrier<PIPE_ALL>();
}

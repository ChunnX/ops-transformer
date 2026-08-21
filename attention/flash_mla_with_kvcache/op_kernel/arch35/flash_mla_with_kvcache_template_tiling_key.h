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
 * \file flash_mla_with_kvcache_template_tiling_key.h
 * \brief flash_mla_with_kvcache TilingKey 定义（非量化 MLA，仅 FP16/BF16）
 *
 * 模板实例化组合继承 fused_infer_attention_score/op_kernel/arch35 的 MLA 分发宏集
 * （fia_arch35_noquant.h 中 dTemplateType == Aligned576 路径 → run_fia_noquant_mla_kernel 实例化集），
 * 并按 R1（Oracle BLOCKER 1）限定到唯一被 fia*mla 四件套验证过的路由：
 *   q = TND（varlen, cu_seqlens_q）、kv = PA_NZ（block_table + cache_seqlens）、out = NTD（运行时 tiling 字段）。
 * **不实例化 BSND/BNSD query、连续/非-NZ KV、非-NTD 输出组合**（checker 拒绝，FIA MLA 从未运行过）。
 *
 * 实例化集合（SEL 笛卡尔积 × 2 dtype）：
 *   InOutLayoutType ∈ {InOutLayoutType_TND_TND}            (1)
 *   Config          ∈ {Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512} (1, MLA 唯一配置)
 *   HasAttenMask    ∈ {false, true}                        (2)
 *   KvLayoutType    ∈ {KvLayoutType_PA_NZ}                 (1)
 *   IsFd            ∈ {false, true}                        (2)
 *   ⇒ 4 个组合/ dtype × INPUT_T ∈ {half, bfloat16_t}（ORIG_DTYPE_* 编译宏） = 8 个 kernel 实例。
 * 枚举/常量数值与 fia_arch35_template_tiling_key_enum.h 逐值一致（只读参考，不改动）。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TEMPLATE_TILING_KEY_H_
#define FLASH_MLA_WITH_KVCACHE_TEMPLATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#include "flash_mla_with_kvcache_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10

#if (__CCE_AICORE__ == 310)
// 解析 tiling key 字段（数值对齐 fia_arch35_template_tiling_key_enum.h 的 PARSE_PARAMS_NoQuant，
// 仅保留 flash_mla_with_kvcache 采用的部分：pse 固定 PSE_NONE，quant 固定 NoQuant，无独立 hasRope 位——rope 已并入 576 宽输入）
#define PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...) \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]); \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1); \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2); \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d); \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv)
#endif

enum class inferFaLayOutTypeEnum {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6
};
enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[6][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_NTD},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_TND},
};

#define InOutLayoutType_BNSD_BNSD 0
#define InOutLayoutType_BSH_BSH 1
#define InOutLayoutType_TND_TND 2
#define InOutLayoutType_BNSD_BSND 3
#define InOutLayoutType_NTD_NTD 4
#define InOutLayoutType_NTD_TND 5

// 布尔 template 参数值必须带数字 token：CANN ASCENDC_TPL_PRE 预编译产出的
// `@@ASCENDC_TPL_BOOL_DECL_<name>@@ = {<values>}` 由 template_tiling.py 的
// extract_num(re.findall(r'\d+', ...)) 解析（pkg_inc/op_common/atvoss 的 C0/C1 同款约定）。
// 字面 "false"/"true" 无数字 → "values of ASCENDC_TPL_BOOL_DECL <name> is empty!"。
// （Todo 14 实测：本文件原声明用 false,true 导致 tbe opc 编译失败；此为最小修复，
// 真实 bug 记录见 learnings。）
#define FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE 0
#define FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE 1

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

static constexpr ConfigParams ConfigValue[] = {
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 0
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // 1
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 2
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // 3
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned192,
     inferDTemplateType::Aligned128}, // 4
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned128}, // 5
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // 6
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512,
     inferDTemplateType::Aligned512}, // 7
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 8
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned576,
     inferDTemplateType::Aligned512}, // 9 (MLA: mBase=64, s2Base=128, D=576=nope512+rope64, DV=512)
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // 10
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned512,
     inferDTemplateType::Aligned512}, // 11
    {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned1024, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 12
    {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // 13
    {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // 14
    {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512,
     inferDTemplateType::Aligned512}, // 15
    {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 16
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // 17
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // 18
    {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 19
    {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // 20
    {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // 21
    {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512,
     inferDTemplateType::Aligned512}, // 22
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned64}, // 23
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned128}, // 24
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned64}, // 25
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned128}, // 26
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // 27
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // 28
};

// 只实例化 MLA 恒真配置（R1/活动范围限定；数值对齐 fia 同名宏）
#define Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 9
#define PSE_MODE_PSE_NONE_TYPE 9
#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_NZ 3
#define NoQuantMode 31

ASCENDC_TPL_ARGS_DECL(FlashMlaWithKvcache,
                      //    bit 7-0 InOutLayoutType（MLA 限定：仅 2 = InOutLayoutType_TND_TND）
                      //    0: InOutLayoutType_BNSD_BNSD   （不实例化，R1）
                      //    1: InOutLayoutType_BSH_BSH     （不实例化，R1）
                      //    2: InOutLayoutType_TND_TND     （q=TND；out 布局走 tiling 运行时字段）
                      //    3: InOutLayoutType_BNSD_BSND   （不实例化，R1）
                      //    4: InOutLayoutType_NTD_NTD     （不实例化，R1）
                      //    5: InOutLayoutType_NTD_TND     （不实例化，R1）
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      //    bit 17-8 Config(S1,S2,D,DV)：MLA 唯一配置
                      //    9: Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      //    bit 18 HasAttenMask
                      //    0: false
                      //    1: true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE),
                      //    bit 20-19 KvLayoutType（MLA 限定：仅 3 = KvLayoutType_PA_NZ）
                      //    0: KvLayoutType_NO_PA          （不实例化：连续 KV 从未被 fia MLA 跑过，R1）
                      //    1: KvLayoutType_PA_BBH         （不实例化，R1）
                      //    2: KvLayoutType_PA_BNBD        （不实例化，R1）
                      //    3: KvLayoutType_PA_NZ
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      //    bit 21 IsFd
                      //    0: false
                      //    1: true
                      ASCENDC_TPL_BOOL_DECL(IsFd, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE),);
// tiling key 编码（权威 = AscendC::FastEncodeTilingKeyDirect，DECL 序累积 bitWidth、UINT 记 DECL 值表索引）：
// key = InOutLayoutType | Config<<8 | HasAttenMask<<18 | KvLayoutType<<19 | IsFd<<21
//   = 2 | 9<<8 | m<<18 | 3<<19 | f<<21  ⇒  4 个组合 key：1575170(m0f0) / 1837314(m1f0) / 3672322(m0f1) / 3934466(m1f1)
// 注意：desc 注释的 "bit 8-1" 等是 FIA 旧文案（其 DECL 参数位域不同），实际编码以本行为准
//（Todo 14 实测：按旧注释推导的 key 被 asc_opc "None of the given tiling keys are in the supported list" 拒绝）。

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_TND_TND),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashMlaWithKvcacheTilingData)), );

#endif // FLASH_MLA_WITH_KVCACHE_TEMPLATE_TILING_KEY_H_
/*
 * probe4_norm_bf16 —— Tiling 结构体（纯 C/C++，kernel 与 host 共用）
 *
 * 说明：穿刺工程绕过 1024B schedule_context 逐字节解析，直接把 host 计算好的标量参数
 *       以扁平 struct 传给 kernel，由 kernel 组装 ScheduleContextInfo 后驱动
 *       真实 A2 三步链（sort -> gather -> group_listing）。
 *       字段全部来自运行时/属性推导，无任何 arch UB/核数常量写死（ubSize/coreNum
 *       由 host 经 PlatformAscendCManager 运行时查询后填入）。
 */
#pragma once

#include <cstdint>

struct ProbeTilingData {
    uint32_t A;          // attention session num
    uint32_t M;          // micro batch num
    uint32_t BS;         // batch size
    uint32_t K;          // topK + 1
    uint32_t HS;         // per-token 字节步长（= H * tokenDtypeSize）
    uint32_t H;          // hidden size（元素个数）
    uint32_t Y;          // A * BS * K
    uint32_t outNum;     // 有效 session 数（NORM 用），totalLength = outNum*BS*K
    uint32_t tokenDtype; // 0:fp16 1:bf16 2:int8   —— 本穿刺 = 1
    uint32_t expertNum;  // 专家总数
    int64_t  coreNum;            // 运行时 aivNum（launch blockDim 与之一致）
    int64_t  ubSize;             // 运行时 GetCoreMemSize(UB) 返回值
    int64_t  sortLoopMaxElement; // 单次可排序 int32 数（UB 派生）
    int64_t  sortNumWorkSpace;   // = Y
};

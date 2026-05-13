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
 * \file test_aclnn_moe_distribute_dispatch_setup.cpp
 * \brief aclnn直调测试脚本
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <getopt.h>
#include <cstring>
#include <sys/wait.h>
#include <bits/stdc++.h>
#include <unistd.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_setup.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_teardown.h"

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapesize = 1;
    for (auto i : shape) {
        shapesize *= i;
    }
    return shapesize;
}

static int64_t GetCeil(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream dispatchsetupstream;
    aclrtStream dispatchteardownstream;
    aclrtContext context;
};

// 全局参数
int64_t g_bs = 8;
int64_t g_h = 7168;
int64_t g_k = 8;
int64_t g_expertSharedType = 0;
int64_t g_sharedExpertNum = 0;
int64_t g_sharedExpertRankNum = 0;
int64_t g_moeExpertNum = 32;
int64_t g_quantMode = 0;
int64_t g_expertTokenNumsType = 1;
int64_t g_outDtype = 0;
int64_t g_commQuantMode = 0;

// 量化场景下使用参数
// 静态量化场景下，是不是所有专家共享scales
// 0：所有专家不共享scale
// 1：所有专家共享scales，且维度为（h，）
// 2：所有专家共享scales，且维度为（1，）
int64_t g_isStaticSharedScales = 0;
int64_t g_staticSharedScalesOne = 1;
int64_t g_staticSharedScalesTwo = 2;

int64_t g_groupListType = 1;
int64_t g_rankId = 0;
int64_t g_epWorldSize = 2;
int64_t g_tpWorldSize = 1;
int64_t g_hasSmoothScale = 0;
int64_t FIRST_RANK_ID = 0;
int64_t commType = 2;

int64_t globalBS = g_bs * g_epWorldSize;
int64_t localExpertNum = g_moeExpertNum / (g_epWorldSize - g_sharedExpertRankNum);
int64_t localToken = globalBS * (localExpertNum < g_k ? localExpertNum : g_k);
uint64_t tokenMsgSize = 0;
uint64_t expandIdxOutSize = 16;
uint64_t assistInfoForCombineOutSize = localToken * 128;
bool hasSmoothScale = g_hasSmoothScale != 0;

int64_t g_timeOut = 100000000;
int64_t g_hcclBufferSize = 12000;
std::string g_xDtype = "fp16";
std::string g_caseName = "./";
std::string g_ranktablePath = "./";
std::string g_scaleType = "fp32";
std::string g_expandxDtype = "fp16";
std::string g_yOutDtype = "fp16";

/* 根据当前场景，构造device侧输入输出变量 */
// 声明device侧输入输出变量
void *xDeviceAddr = nullptr;
void *expertIdsDeviceAddr = nullptr;
void *scalesDeviceAddr = nullptr;
void *expertScalesDeviceAddr = nullptr;
void *expandXDeviceAddr = nullptr;
void *dynamicScalesDeviceAddr = nullptr;
void *expandIdxDeviceAddr = nullptr;
void *expertTokenNumsDeviceAddr = nullptr;
void *epRecvCountsDeviceAddr = nullptr;
void *tpRecvCountsDeviceAddr = nullptr;
void *expandScalesDeviceAddr = nullptr;
void *yOutDeviceAddr = nullptr;
void *assistInfoForCombineOutDeviceAddr = nullptr;
void *quantExpandXOutDeviceAddr = nullptr;
void *xOutDeviceAddr = nullptr;
void *commCmdInfoOutDeviceAddr = nullptr;
void *commCmdInfoOutShapeforcombineDeviceAddr = nullptr;

aclTensor *x = nullptr;
aclTensor *expertIds = nullptr;
aclTensor *scalesOptional = nullptr;
aclTensor *xActiveMaskOptional = nullptr;
aclTensor *yOut = nullptr;
aclTensor *xOut = nullptr;
aclTensor *sharedExpertXOptional = nullptr;
aclTensor *scales = nullptr;
aclTensor *expertScales = nullptr;
aclTensor *expandX = nullptr;
aclTensor *dynamicScales = nullptr;
aclTensor *expandIdx = nullptr;
aclTensor *expertTokenNums = nullptr;
aclTensor *epRecvCounts = nullptr;
aclTensor *tpRecvCounts = nullptr;
aclTensor *expandScales = nullptr;
aclTensor *assistInfoForCombineOut = nullptr;
aclTensor *quantExpandXOut = nullptr;
aclTensor *commCmdInfoOut = nullptr;
aclTensor *commCmdInfoOutforCombine = nullptr;

// 定义当前场景下各变量维度
std::vector<int64_t> xShape{g_bs, g_h};
std::vector<int64_t> expertIdsShape{g_bs, g_k};
std::vector<int64_t> scalesShape{(g_sharedExpertRankNum > 0) ? 1 + g_moeExpertNum : g_moeExpertNum, g_h};
std::vector<int64_t> scaleshif8Shape{1};
std::vector<int64_t> scalesCommonShape_h{g_h};
std::vector<int64_t> scalesCommonShape_1{1};
std::vector<int64_t> expertScalesShape{g_bs, g_k};
std::vector<int64_t> expandXShape{g_tpWorldSize * localToken, g_h};
std::vector<int64_t> scalesmodel2Shape{g_bs, g_h};
std::vector<int64_t> yOutShape{g_bs * (g_k + g_sharedExpertNum), 7168};

std::vector<int64_t> dynamicScalesShape_H{g_tpWorldSize * localToken, g_h};
std::vector<int64_t> dynamicScalesShape_ceil32{g_tpWorldSize * localToken, GetCeil(g_h, 32)};
std::vector<int64_t> dynamicScalesShape_ceil128{g_tpWorldSize * localToken, GetCeil(g_h, 128)};
std::vector<int64_t> dynamicScalesShape{g_tpWorldSize * localToken};
std::vector<int64_t> dynamicModel2ScalesShape{localToken * g_h};

std::vector<int64_t> expandIdxShape{g_bs * g_k};
std::vector<int64_t> expertTokenNumsShape{localExpertNum};
std::vector<int64_t> epRecvCountsShape{g_tpWorldSize * localExpertNum * g_epWorldSize};
std::vector<int64_t> tpRecvCountsShape{g_tpWorldSize * localExpertNum};
std::vector<int64_t> expandScalesShape{localToken};
std::vector<int64_t> assistInfoForCombineOutShape{assistInfoForCombineOutSize};
std::vector<int64_t> quantExpandXOutShape{g_tpWorldSize * localToken, 10752};
std::vector<int64_t> commCmdInfoOutShape{(g_bs * (g_k + g_sharedExpertNum) + g_epWorldSize * localExpertNum) * 16};
std::vector<int64_t> commCmdInfoOutShapeforcombine{1056};
std::vector<int64_t> xOutShape{g_bs, g_h};
long long xShapeSize = GetShapeSize(xShape);
int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
int64_t scalesShapeSize = GetShapeSize(scalesShape);
int64_t scalesmodel2ShapeSize = GetShapeSize(scalesmodel2Shape);
int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
int64_t expandXShapeSize = GetShapeSize(expandXShape);
int64_t yOutShapeSize = GetShapeSize(yOutShape);

int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
int64_t dynamicModel2ScalesShapeSize = GetShapeSize(dynamicModel2ScalesShape);
int64_t dynamicScalesShapeSizeH = GetShapeSize(dynamicScalesShape_H);
int64_t dynamicScalesShapeSizeCeil32 = GetShapeSize(dynamicScalesShape_ceil32);
int64_t dynamicScalesShapeSizeCeil128 = GetShapeSize(dynamicScalesShape_ceil128);

int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
int64_t assistInfoForCombineOutShapeSize = GetShapeSize(assistInfoForCombineOutShape);
int64_t commCmdInfoOutShapeSize = GetShapeSize(commCmdInfoOutShape);
int64_t quantExpandXOutShapeSize = GetShapeSize(quantExpandXOutShape);
int64_t xOutShapeSize = GetShapeSize(xOutShape);
int64_t commCmdInfoOutShapeSizeforcombine = GetShapeSize(commCmdInfoOutShapeforcombine);

// 构造host侧变量
std::vector<int16_t> xHostMode1Data(xShapeSize, 1);
std::vector<int8_t> xHostMode2Data(xShapeSize, 1);
std::vector<int32_t> expertIdsHostData(expertIdsShapeSize, 0);
std::vector<float> scalesModel2FloatHostData(scalesmodel2ShapeSize, 0);
std::vector<int8_t> scalesMode2Int8HostData(scalesmodel2ShapeSize, 0);
std::vector<float> scalesHostData(scalesShapeSize, 0);
std::vector<int8_t> scalesMxfp8HostData(scalesShapeSize, 0);
std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);

std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
std::vector<float> dynamicModel2ScalesHostDataFloat(dynamicModel2ScalesShapeSize, 0);
std::vector<int8_t> dynamicModel2ScalesHostDataInt8(dynamicModel2ScalesShapeSize, 0);
std::vector<float> dynamicScalesHostDataH(dynamicScalesShapeSizeH, 0);
std::vector<int8_t> dynamicScalesHostDataCeil32(dynamicScalesShapeSizeCeil32, 0);
std::vector<int8_t> dynamicScalesHostDataCeil128(dynamicScalesShapeSizeCeil128, 0);
std::vector<int8_t> dynamicScalesMxfp8HostData(dynamicScalesShapeSize, 0);

std::vector<float> expandIdxHostData(expandIdxShapeSize, 0);
std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
std::vector<int8_t> expandXQuantHostData(expandXShapeSize, 0);
std::vector<float> yOutHostData(yOutShapeSize, 0);
std::vector<int32_t> AssistInfoForCombineOutHostData(assistInfoForCombineOutShapeSize, 0);
std::vector<int32_t> commCmdInfoOutHostData(commCmdInfoOutShapeSize, 0);
std::vector<int8_t> quantExpandXOutHostData(quantExpandXOutShapeSize, 0);
std::vector<float> xOutHostData(xOutShapeSize, 0);
std::vector<int32_t> commCmdInfoOutforCombineHostData(commCmdInfoOutShapeSizeforcombine, 0);

// 0: dispatch算子非量化场景
// 1: 静态量化
// 2: pertoken动态量化
// 3: pergroup动态量化
// 4: MX量化
int64_t g_noQuant = 0;
int64_t g_staticQuant = 1;
int64_t g_pertokenQuant = 2;
int64_t g_pergroupQuant = 3;
int64_t g_mxQuant = 4;

/* 声明算子执行必需变量 */
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
void *workspaceAddr = nullptr;

std::map<std::string, aclDataType> x_dtypeIn = {
    {"bf16", ACL_BF16},     {"fp16", ACL_FLOAT16}, {"fp8_e5m2", ACL_FLOAT8_E5M2}, {"fp8_e4m3fn", ACL_FLOAT8_E4M3FN},
    {"hif8", ACL_HIFLOAT8},
};

std::map<std::string, aclDataType> expandx_dtypeOut = {
    {"bf16", ACL_BF16},
    {"fp16", ACL_FLOAT16},
    {"int8", ACL_INT8},
    {"fp8_e5m2", ACL_FLOAT8_E5M2},
    {"fp8_e4m3fn", ACL_FLOAT8_E4M3FN},
    {"hif8", ACL_HIFLOAT8},
};

std::map<std::string, aclDataType> scaleIn = {
    {"fp32", ACL_FLOAT},
    {"fp8_e8m0", ACL_FLOAT8_E8M0},
};

std::vector<int32_t> generate_random_vector(int64_t g_bs, int64_t g_k, int seed)
{
    // 为expert_id生成一个值随机为0或1的vec
    // 总大小
    int total_size = g_bs * g_k;

    // 使用随机数引擎和分布
    std::mt19937 gen(seed);                    // Mersenne Twister 引擎
    std::uniform_int_distribution<> dis(0, 1); // 生成 0 或 1

    // 创建并填充向量
    std::vector<int32_t> result;
    result.reserve(total_size); // 预分配内存，提升性能

    for (int i = 0; i < total_size; ++i) {
        result.push_back(dis(gen));
    }

    return result;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, uint32_t &rankId, const std::string &name)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorX(Args &args)
{
    int ret = 0;
    if (g_quantMode == g_noQuant && (g_xDtype == "fp8_e5m2" || g_xDtype == "fp8_e4m3fn" || g_xDtype == "hif8")) {
        ret = CreateAclTensor(xHostMode2Data, xShape, &xDeviceAddr, x_dtypeIn[g_xDtype], &x, args.rankId, "x");
    } else {
        ret = CreateAclTensor(xHostMode1Data, xShape, &xDeviceAddr, x_dtypeIn[g_xDtype], &x, args.rankId, "x");
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorExpertIds(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds,
                          args.rankId, "expertIds");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorScales(Args &args)
{
    int ret = 0;
    if (g_quantMode == g_noQuant && (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        scales = nullptr;
    } else if ((g_quantMode == g_noQuant) && (x_dtypeIn[g_xDtype] == ACL_HIFLOAT8)) {
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales,
                              args.rankId, "scales");
    } else if ((g_quantMode == g_noQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_FLOAT8_E4M3FN || x_dtypeIn[g_xDtype] == ACL_FLOAT8_E5M2)) {
        if (scaleIn[g_scaleType] == ACL_FLOAT) {
            ret = CreateAclTensor(scalesModel2FloatHostData, scalesmodel2Shape, &scalesDeviceAddr,
                                  aclDataType::ACL_FLOAT, &scales, args.rankId, "scales");
        } else if (scaleIn[g_scaleType] == ACL_FLOAT8_E8M0) {
            ret = CreateAclTensor(scalesMode2Int8HostData, scalesmodel2Shape, &scalesDeviceAddr,
                                  aclDataType::ACL_FLOAT8_E8M0, &scales, args.rankId, "scales");
        }
    } else if ((g_quantMode == g_staticQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        if (g_isStaticSharedScales == g_staticSharedScalesOne) {
            ret = CreateAclTensor(scalesHostData, scalesCommonShape_h, &scalesDeviceAddr, aclDataType::ACL_FLOAT,
                                  &scales, args.rankId, "scales");
        } else if (g_isStaticSharedScales == g_staticSharedScalesTwo) {
            ret = CreateAclTensor(scalesHostData, scalesCommonShape_1, &scalesDeviceAddr, aclDataType::ACL_FLOAT,
                                  &scales, args.rankId, "scales");
        } else {
            ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales,
                                  args.rankId, "scales");
        }
    } else if ((g_quantMode == g_pertokenQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales,
                              args.rankId, "scales");
    } else if ((g_quantMode == g_pergroupQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales,
                              args.rankId, "scales");
    } else if ((g_quantMode == g_mxQuant) && (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        scales = nullptr;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorExpertScales(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScales, args.rankId, "expertScales");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorExpandX(Args &args)
{
    int ret = 0;
    if (g_quantMode > 0) {
        ret = CreateAclTensor(expandXQuantHostData, expandXShape, &expandXDeviceAddr, expandx_dtypeOut[g_expandxDtype],
                              &expandX, args.rankId, "expandX");
    } else if ((g_quantMode == g_noQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_HIFLOAT8 || x_dtypeIn[g_xDtype] == ACL_FLOAT8_E4M3FN ||
                x_dtypeIn[g_xDtype] == ACL_FLOAT8_E5M2)) {
        ret = CreateAclTensor(expandXQuantHostData, expandXShape, &expandXDeviceAddr, expandx_dtypeOut[g_expandxDtype],
                              &expandX, args.rankId, "expandX");
    } else {
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, expandx_dtypeOut[g_expandxDtype],
                              &expandX, args.rankId, "expandX");
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorDynamicScales(Args &args)
{
    int ret = 0;
    if (g_quantMode == g_noQuant && (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr,
                              aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
    } else if ((g_quantMode == g_noQuant) && (x_dtypeIn[g_xDtype] == ACL_HIFLOAT8)) {
        ret = CreateAclTensor(dynamicModel2ScalesHostDataFloat, dynamicModel2ScalesShape, &dynamicScalesDeviceAddr,
                              aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
    } else if ((g_quantMode == g_noQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_FLOAT8_E4M3FN || x_dtypeIn[g_xDtype] == ACL_FLOAT8_E5M2)) {
        if (scaleIn[g_scaleType] == ACL_FLOAT) {
            ret = CreateAclTensor(dynamicModel2ScalesHostDataFloat, dynamicModel2ScalesShape, &dynamicScalesDeviceAddr,
                                  aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
        } else if (scaleIn[g_scaleType] == ACL_FLOAT8_E8M0) {
            ret = CreateAclTensor(dynamicModel2ScalesHostDataInt8, dynamicModel2ScalesShape, &dynamicScalesDeviceAddr,
                                  aclDataType::ACL_FLOAT8_E8M0, &dynamicScales, args.rankId, "dynamicScales");
        } else {
            ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr,
                                  aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
        }
    } else if ((g_quantMode == g_staticQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        if (expandx_dtypeOut[g_expandxDtype] == ACL_INT8) {
            ret = CreateAclTensor(dynamicScalesHostDataH, dynamicScalesShape_H, &dynamicScalesDeviceAddr,
                                  aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
        } else if (expandx_dtypeOut[g_expandxDtype] == ACL_HIFLOAT8) {
            ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr,
                                  aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
        }
    } else if ((g_quantMode == g_pertokenQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr,
                              aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
    } else if ((g_quantMode == g_pergroupQuant) &&
               (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        ret = CreateAclTensor(dynamicScalesHostDataCeil128, dynamicScalesShape_ceil128, &dynamicScalesDeviceAddr,
                              aclDataType::ACL_FLOAT, &dynamicScales, args.rankId, "dynamicScales");
    } else if ((g_quantMode == g_mxQuant) && (x_dtypeIn[g_xDtype] == ACL_BF16 || x_dtypeIn[g_xDtype] == ACL_FLOAT16)) {
        ret = CreateAclTensor(dynamicScalesHostDataCeil32, dynamicScalesShape_ceil32, &dynamicScalesDeviceAddr,
                              aclDataType::ACL_FLOAT8_E8M0, &dynamicScales, args.rankId, "dynamicScales");
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorExpandIdx(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx,
                          args.rankId, "expandIdx");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorExpertTokenNums(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr,
                          aclDataType::ACL_INT64, &expertTokenNums, args.rankId, "expertTokenNums");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorEpRecvCounts(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &epRecvCounts, args.rankId, "epRecvCounts");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorTpRecvCounts(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &tpRecvCounts, args.rankId, "tpRecvCounts");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorExpandScales(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expandScales, args.rankId, "expandScales");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorYOut(Args &args)
{
    int ret = 0;
    ret =
        CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT16, &yOut, args.rankId, "yOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorAssistInfoForCombineOut(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(AssistInfoForCombineOutHostData, assistInfoForCombineOutShape,
                          &assistInfoForCombineOutDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombineOut,
                          args.rankId, "assistInfoForCombineOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorcommCmdInfoOut(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(commCmdInfoOutHostData, commCmdInfoOutShape, &commCmdInfoOutDeviceAddr,
                          aclDataType::ACL_INT32, &commCmdInfoOut, args.rankId, "commCmdInfoOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorquantExpandXOut(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(quantExpandXOutHostData, quantExpandXOutShape, &quantExpandXOutDeviceAddr,
                          aclDataType::ACL_INT8, &quantExpandXOut, args.rankId, "quantExpandXOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclTensorxOut(Args &args)
{
    int ret = 0;
    ret =
        CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT16, &xOut, args.rankId, "xOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}

int CreateAclcommCmdInfoOutforCombine(Args &args)
{
    int ret = 0;
    ret = CreateAclTensor(commCmdInfoOutforCombineHostData, commCmdInfoOutShapeforcombine,
                          &commCmdInfoOutShapeforcombineDeviceAddr, aclDataType::ACL_INT32, &commCmdInfoOutforCombine,
                          args.rankId, "commCmdInfoOutforCombine");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ret;
}


int ProcessDispatchSetup(char *hcomEpName, char *hcomTpName, Args &args)
{
    int ret = 0;
    /******************************先调用dispatch_setup********************************************/
    // 调用dispatch_setup算子第一阶段接口
    ret = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, (hasSmoothScale ? scales : nullptr), nullptr, hcomEpName, g_epWorldSize, args.epRankId,
        g_moeExpertNum, g_expertSharedType, g_sharedExpertNum, g_sharedExpertRankNum, g_quantMode, globalBS, commType,
        nullptr, yOut, expandIdx, commCmdInfoOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchSetupGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    // 根据dispatch算子第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] DispatchSetup aclrtMalloc failed. ret = %d\n", ret);
                  return ret);
    }

    // 调用dispatch算子第二阶段接口
    ret = aclnnMoeDistributeDispatchSetup(workspaceAddr, workspaceSize, executor, args.dispatchsetupstream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchSetup failed. ret = %d \n", ret);
              return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchsetupstream, g_timeOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchSetup execute successfully.\n", args.rankId);
    return ret;
}

int ProcessDispatchTeardown(char *hcomEpName, char *hcomTpName, Args &args)
{
    int ret = 0;
    /******************************调用dispatch_teardown********************************************/
    // 调用dispatch_teardown算子第一阶段接口
    ret = aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
        x, yOut, expertIds, commCmdInfoOut, hcomEpName, g_epWorldSize, args.epRankId, g_moeExpertNum,
        g_expertSharedType, g_sharedExpertNum, g_sharedExpertRankNum, g_quantMode, globalBS, g_expertTokenNumsType,
        commType, nullptr, expandX, dynamicScales, assistInfoForCombineOut, expertTokenNums, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchTeardownGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    // 根据dispatch算子第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] DispatchTeardown aclrtMalloc failed. ret = %d\n", ret);
                  return ret);
    }

    // 调用dispatch_teardown算子第二阶段接口
    ret = aclnnMoeDistributeDispatchTeardown(workspaceAddr, workspaceSize, executor, args.dispatchteardownstream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchTeardown failed. ret = %d \n", ret);
              return ret);

    // 等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchteardownstream, g_timeOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchTeardown execute successfully.\n", args.rankId);
    return ret;
}

void ReleaseTensor()
{
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (scalesOptional != nullptr) {
        aclDestroyTensor(scalesOptional);
    }
    if (xActiveMaskOptional != nullptr) {
        aclDestroyTensor(xActiveMaskOptional);
    }
    if (yOut != nullptr) {
        aclDestroyTensor(yOut);
    }
    if (xOut != nullptr) {
        aclDestroyTensor(xOut);
    }
    if (sharedExpertXOptional != nullptr) {
        aclDestroyTensor(sharedExpertXOptional);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (expertScales != nullptr) {
        aclDestroyTensor(expertScales);
    }
    if (expandX != nullptr) {
        aclDestroyTensor(expandX);
    }
    if (dynamicScales != nullptr) {
        aclDestroyTensor(dynamicScales);
    }
    if (expandIdx != nullptr) {
        aclDestroyTensor(expandIdx);
    }
    if (expertTokenNums != nullptr) {
        aclDestroyTensor(expertTokenNums);
    }
    if (epRecvCounts != nullptr) {
        aclDestroyTensor(epRecvCounts);
    }
    if (tpRecvCounts != nullptr) {
        aclDestroyTensor(tpRecvCounts);
    }
    if (expandScales != nullptr) {
        aclDestroyTensor(expandScales);
    }
    if (assistInfoForCombineOut != nullptr) {
        aclDestroyTensor(assistInfoForCombineOut);
    }
    if (quantExpandXOut != nullptr) {
        aclDestroyTensor(quantExpandXOut);
    }
    if (commCmdInfoOut != nullptr) {
        aclDestroyTensor(commCmdInfoOut);
    }
    if (commCmdInfoOutforCombine != nullptr) {
        aclDestroyTensor(commCmdInfoOutforCombine);
    }
}

void ReleaseAddr()
{
    // 释放device资源
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (dynamicScalesDeviceAddr != nullptr) {
        aclrtFree(dynamicScalesDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (expertTokenNumsDeviceAddr != nullptr) {
        aclrtFree(expertTokenNumsDeviceAddr);
    }
    if (epRecvCountsDeviceAddr != nullptr) {
        aclrtFree(epRecvCountsDeviceAddr);
    }
    if (tpRecvCountsDeviceAddr != nullptr) {
        aclrtFree(tpRecvCountsDeviceAddr);
    }
    if (expandScalesDeviceAddr != nullptr) {
        aclrtFree(expandScalesDeviceAddr);
    }
    if (yOutDeviceAddr != nullptr) {
        aclrtFree(yOutDeviceAddr);
    }
    if (assistInfoForCombineOutDeviceAddr != nullptr) {
        aclrtFree(assistInfoForCombineOutDeviceAddr);
    }
    if (quantExpandXOutDeviceAddr != nullptr) {
        aclrtFree(quantExpandXOutDeviceAddr);
    }
    if (xOutDeviceAddr != nullptr) {
        aclrtFree(xOutDeviceAddr);
    }
    if (commCmdInfoOutDeviceAddr != nullptr) {
        aclrtFree(commCmdInfoOutDeviceAddr);
    }
    if (commCmdInfoOutShapeforcombineDeviceAddr != nullptr) {
        aclrtFree(commCmdInfoOutShapeforcombineDeviceAddr);
    }
}

void ReleaseResources()
{
    ReleaseTensor();
    ReleaseAddr();
}

void ProcessLocalExpertNumAndLocalToken(Args &args)
{
    if (args.epRankId < g_sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        localToken = globalBS / g_sharedExpertRankNum;
    } else {
        // Moe专家卡
        localExpertNum = g_moeExpertNum / (g_epWorldSize - g_sharedExpertRankNum);
        localToken = globalBS * (localExpertNum < g_k ? localExpertNum : g_k);
    }

    expandXShape = {g_tpWorldSize * localToken, g_h};
    dynamicScalesShape_H = {g_tpWorldSize * localToken, g_h};
    dynamicScalesShape_ceil32 = {g_tpWorldSize * localToken, GetCeil(g_h, 32)};
    dynamicScalesShape_ceil128 = {g_tpWorldSize * localToken, GetCeil(g_h, 128)};
    dynamicScalesShape = {g_tpWorldSize * localToken};
    dynamicModel2ScalesShape = {localToken * g_h};
    expertTokenNumsShape = {localExpertNum};
    epRecvCountsShape = {g_tpWorldSize * localExpertNum * g_epWorldSize};
    tpRecvCountsShape = {g_tpWorldSize * localExpertNum};
    expandScalesShape = {localToken};

    expandXShapeSize = GetShapeSize(expandXShape);
    dynamicScalesShapeSizeH = GetShapeSize(dynamicScalesShape_H);
    dynamicScalesShapeSizeCeil32 = GetShapeSize(dynamicScalesShape_ceil32);
    dynamicScalesShapeSizeCeil128 = GetShapeSize(dynamicScalesShape_ceil128);
    dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    dynamicModel2ScalesShapeSize = GetShapeSize(dynamicModel2ScalesShape);
    expandScalesShapeSize = GetShapeSize(expandScalesShape);
    expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);

    expandXHostData.resize(expandXShapeSize, 0);
    expandXQuantHostData.resize(expandXShapeSize, 0);
    dynamicScalesHostDataH.resize(dynamicScalesShapeSizeH, 0);
    dynamicScalesHostDataCeil32.resize(dynamicScalesShapeSizeCeil32, 0);
    dynamicScalesHostDataCeil128.resize(dynamicScalesShapeSizeCeil128, 0);
    dynamicScalesHostData.resize(dynamicScalesShapeSize, 0);
    dynamicScalesMxfp8HostData.resize(dynamicScalesShapeSize, 0);
    dynamicModel2ScalesHostDataFloat.resize(dynamicModel2ScalesShapeSize, 0);
    dynamicModel2ScalesHostDataInt8.resize(dynamicModel2ScalesShapeSize, 0);
    expandScalesHostData.resize(expandScalesShapeSize, 0);
    expertTokenNumsHostData.resize(expertTokenNumsShapeSize, 0);
    epRecvCountsHostData.resize(epRecvCountsShapeSize, 0);
    tpRecvCountsHostData.resize(tpRecvCountsShapeSize, 0);
}

int LaunchOneProcess(Args &args)
{
    ProcessLocalExpertNumAndLocalToken(args);
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomEpName[128] = {0}; // ep通信域名称
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return ret);
    char hcomTpName[128] = {0};

    auto destroyFunc = [&args]() {
        std::cout << "== begin to destroy " << std::endl;
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchsetupstream);
        aclrtDestroyStream(args.dispatchteardownstream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
    };

    // 构造device侧变量
    ret = CreateAclTensorX(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorExpertIds(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorScales(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorExpertScales(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorExpandX(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorDynamicScales(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorExpandIdx(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorExpertTokenNums(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorEpRecvCounts(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorTpRecvCounts(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorExpandScales(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorYOut(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorAssistInfoForCombineOut(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorcommCmdInfoOut(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorquantExpandXOut(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclTensorxOut(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor failed. ret: %d\n", ret); return ret);
    ret = CreateAclcommCmdInfoOutforCombine(args);

    LOG_PRINT("----Dispatch Setup aclnn start----");
    ProcessDispatchSetup(hcomEpName, hcomTpName, args);
    LOG_PRINT("----Dispatch Teardown aclnn start----");
    ProcessDispatchTeardown(hcomEpName, hcomTpName, args);
    ReleaseResources();
    return 0;
}

int RunInProcess(int rank, int rankSize)
{
    int ret = aclInit(nullptr);
    LOG_PRINT("aclInit: %d \n", ret);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
    aclrtStream dispatchsetupstream;
    aclrtStream dispatchteardownstream;
    aclrtContext context;
    ret = aclrtSetDevice(rank);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
    ret = aclrtCreateContext(&context, rank);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
    ret = aclrtCreateStream(&dispatchsetupstream);
    ret = aclrtCreateStream(&dispatchteardownstream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);

    HcclComm commsEp;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = g_hcclBufferSize;
    strncpy(config.hcclCommName, "hccl_comm_test", COMM_NAME_MAX_LENGTH - 1);
    std::string rankTableFile = std::getenv("RANK_TABLE_FILE");
    std::cout << "rankTableFilePath is :" << rankTableFile << std::endl;
    int rank_id = rank + FIRST_RANK_ID;

    ret = HcclCommInitClusterInfoConfig(rankTableFile.c_str(), rank_id, &config, &commsEp);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig ep world %d failed. ret = %d\n", g_rankId, ret);
              return ret);

    Args args;
    uint32_t epRankId = rank_id / g_tpWorldSize;
    uint32_t tpRankId = rank_id % g_tpWorldSize;
    args.rankId = rank;
    args.epRankId = epRankId;
    args.tpRankId = tpRankId;
    args.hcclEpComm = commsEp;
    args.dispatchsetupstream = dispatchsetupstream;
    args.dispatchteardownstream = dispatchteardownstream;
    args.context = context;
    int res = LaunchOneProcess(args);
    if (res != ACL_SUCCESS) {
        std::cout << "run LaunchOneProcess failed, ret = " << res << std::endl;
        return res;
    }
    LOG_PRINT("[INFO] aclFinalize success\n");
    return res;
}

int main(int argc, char *argv[])
{
    char *env_rankID = std::getenv("FIRST_RANK_ID");
    if (!env_rankID) {
        std::cerr << "FIRST_RANK_ID环境变量未设置！\n";
        return 1;
    }
    FIRST_RANK_ID = std::stoi(std::string(env_rankID));
    std::cout << "FIRST_RANK_ID is: " << FIRST_RANK_ID << std::endl;

    const int processCount = 2;
    pid_t pids[processCount];

    for (int i = 0; i < processCount; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            // 子进程，完成任务后退出
            int ret = RunInProcess(i, processCount);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RunInProcess failed. ret = %d\n", ret); return ret);
            exit(0);
        }
    }

    // 父进程等待所有子进程完成
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], nullptr, 0);
    }

    return 0;
}
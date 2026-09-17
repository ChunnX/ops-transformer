/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_allto_all_quant_matmul.cpp
 * \brief aclnnAlltoAllQuantMatmul mx量化场景(MXFP8/MXFP4)测试样例
 *
 * 本样例基于Ascend 950PR/Ascend 950DT实现，必须在Ascend 950PR/Ascend 950DT上运行，主要演示：
 * 1. MXFP8场景：x1/x2为FLOAT8_E4M3FN(也支持FLOAT8_E5M2)，x1Scale/x2Scale为FLOAT8_E8M0；
 *    MXFP4场景：x1/x2为FLOAT4_E2M1，存储时2个元素打包为1字节(偶数下标元素占低4bit，奇数下标占高4bit)，
 *    x1Scale/x2Scale同为FLOAT8_E8M0；
 *    两种场景均为x1QuantMode=x2QuantMode=6(mx量化)，groupSize=4295032864(对应[gsM,gsN,gsK]=[1,1,32])，
 *    transposeX2=true(x2存储shape为(N, H*rankSize))，每个E8M0 scale覆盖K方向连续32个元素，数值为2^(码值-127)。
 * 2. 使用HcclCommInitClusterInfoConfig接口，基于rank table文件创建通信域(每个进程对应一个rank)。
 * 3. 将算子的输入输出数据保存为bin文件。
 *
 * 使用步骤(以2卡为例)：
 * 1. 准备rank table文件rank_table_2p.json：
 *    {
 *        "status": "completed",
 *        "version": "1.0",
 *        "server_count": "1",
 *        "server_list": [
 *            {
 *                "server_id": "127.0.0.1",
 *                "device": [
 *                    {"device_id": "0", "device_ip": "192.168.100.101", "rank_id": "0"},
 *                    {"device_id": "1", "device_ip": "192.168.100.102", "rank_id": "1"}
 *                ]
 *            }
 *        ]
 *    }
 * 2. 启动(每个rank对应一个进程，MXFP4场景将--quant_mode替换为mxfp4即可)：
 *    export RANK_TABLE_FILE=$(pwd)/rank_table_2p.json
 *    ./test_aclnn_allto_all_quant_matmul --rank_id 0 --quant_mode mxfp8 &
 *    ./test_aclnn_allto_all_quant_matmul --rank_id 1 --quant_mode mxfp8
 *
 * 输入输出bin文件保存在<output_dir>/<quant_mode>/rank<rank_id>/目录下：
 *   输入：input_x1.bin、input_x2.bin、input_bias.bin、input_x1_scale.bin、input_x2_scale.bin
 *   输出：output_out.bin、output_alltoall_out.bin
 * 其中MXFP4场景的input_x1.bin/input_x2.bin/output_alltoall_out.bin为fp4x2打包数据(2个元素1字节)。
 */
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <acl/acl.h>
#include <hccl/hccl.h>
#include <securec.h>
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_allto_all_quant_matmul.h"

// mx量化场景矩阵规格：BS必须整除rankSize，H必须整除64
constexpr int64_t g_bs = 256; // BS
constexpr int64_t g_h = 64;   // H(每卡K轴长度)
constexpr int64_t g_n = 256;  // N
// mx量化参数：groupSize=4295032864对应[gsM,gsN,gsK]=[1,1,32]
constexpr int64_t g_mxGroupSize = 4295032864;
// mx量化模式
constexpr int64_t g_mxQuantMode = 6;

enum class MxQuantMode {
    MXFP8,
    MXFP4,
};

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// 逐级创建目录，目录已存在时忽略报错
static void MakeDirs(const std::string &dirPath)
{
    std::string curDir;
    for (auto c : dirPath) {
        curDir.push_back(c);
        if (c == '/') {
            (void)mkdir(curDir.c_str(), 0755);
        }
    }
    (void)mkdir(dirPath.c_str(), 0755);
}

// 将host侧数据保存为bin文件
static int SaveBinFile(const std::string &fileName, const void *data, size_t size)
{
    std::ofstream outFile(fileName, std::ios::binary);
    CHECK_RET(outFile.is_open(), LOG_PRINT("[ERROR] open file %s failed.\n", fileName.c_str()); return -1);
    outFile.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    CHECK_RET(outFile.good(), LOG_PRINT("[ERROR] write file %s failed.\n", fileName.c_str()); return -1);
    LOG_PRINT("[INFO] save %s (%zu bytes) success\n", fileName.c_str(), size);
    return 0;
}

// 生成FP8数据(FLOAT8_E4M3FN/FLOAT8_E5M2)
// 码值{0x30,0x38,0x3C,0x40,0x44,0x48}在E4M3FN下对应{0.5,1.0,1.5,2.0,2.5,4.0}，均为有限值
static void FillFp8HostData(std::vector<uint8_t> &hostData)
{
    static const uint8_t fp8Codes[] = {0x30, 0x38, 0x3C, 0x40, 0x44, 0x48};
    constexpr uint8_t codeNum = sizeof(fp8Codes) / sizeof(fp8Codes[0]);
    for (size_t i = 0; i < hostData.size(); i++) {
        hostData[i] = fp8Codes[i % codeNum];
    }
}

// 生成FP4打包数据(FLOAT4_E2M1)：2个元素打包为1字节，偶数下标元素占低4bit，奇数下标占高4bit
// 4bit码值1~6对应数值{0.5,1.0,1.5,2.0,3.0,4.0}
static void FillFp4PackedHostData(std::vector<uint8_t> &hostData, int64_t elemNum)
{
    for (int64_t i = 0; i < elemNum; i += 2) {
        const uint8_t evenCode = static_cast<uint8_t>(1 + (i % 6));
        // 元素个数为奇数时，最后一字节的高4bit补0
        const uint8_t oddCode = (i + 1 < elemNum) ? static_cast<uint8_t>(1 + ((i + 1) % 6)) : 0;
        hostData[i / 2] = static_cast<uint8_t>(evenCode | (oddCode << 4));
    }
}

// 生成E8M0 scale数据，数值为2^(码值-127)，码值{126,127,128}对应{0.5,1.0,2.0}
static void FillE8M0ScaleHostData(std::vector<uint8_t> &hostData)
{
    static const uint8_t e8m0Codes[] = {127, 128, 126};
    constexpr uint8_t codeNum = sizeof(e8m0Codes) / sizeof(e8m0Codes[0]);
    for (size_t i = 0; i < hostData.size(); i++) {
        hostData[i] = e8m0Codes[i % codeNum];
    }
}

// 按显式字节数创建aclTensor(FP8/FP4/E8M0等非2字节对齐或打包存储的场景)
static int CreateAclTensorWithBytes(const void *hostData, size_t byteSize, const std::vector<int64_t> &shape,
                                    void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    return CreateAclTensorWithBytes(hostData.data(), hostData.size() * sizeof(T), shape, deviceAddr, dataType, tensor);
}

struct Args {
    uint32_t rankId;
    uint32_t rankSize;
    MxQuantMode quantMode;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    std::string outputDir;
};

int LaunchOneRankAlltoAllMxQuantMatmul(const Args &args)
{
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);

    const int64_t k = g_h * args.rankSize; // K = H * rankSize
    // transposeX2=true时x2存储shape为(N, K)
    std::vector<int64_t> x1Shape = {g_bs, g_h};
    std::vector<int64_t> x2Shape = {g_n, k};
    std::vector<int64_t> biasShape = {g_n};
    // mx量化场景scale shape：x1Scale为(BS, ceil(H/64), 2)，x2Scale为(N, ceil(K/64), 2)
    std::vector<int64_t> x1ScaleShape = {g_bs, g_h / 64, 2};
    std::vector<int64_t> x2ScaleShape = {g_n, k / 64, 2};
    std::vector<int64_t> outShape = {g_bs / args.rankSize, g_n};
    std::vector<int64_t> allToAllOutShape = {g_bs / args.rankSize, k};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *x1ScaleDeviceAddr = nullptr;
    void *x2ScaleDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *allToAllOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *x1Scale = nullptr;
    aclTensor *x2Scale = nullptr;
    aclTensor *commScaleOptional = nullptr;
    aclTensor *x1OffsetOptional = nullptr;
    aclTensor *x2OffsetOptional = nullptr;
    aclTensor *out = nullptr;
    aclTensor *allToAllOut = nullptr;

    const char *modeStr = (args.quantMode == MxQuantMode::MXFP8) ? "mxfp8" : "mxfp4";
    LOG_PRINT("[INFO] rank %d quant mode: %s, x1(%ld,%ld), x2(%ld,%ld), x1Scale(%ld,%ld,%ld), x2Scale(%ld,%ld,%ld)\n",
              args.rankId, modeStr, x1Shape[0], x1Shape[1], x2Shape[0], x2Shape[1], x1ScaleShape[0], x1ScaleShape[1],
              x1ScaleShape[2], x2ScaleShape[0], x2ScaleShape[1], x2ScaleShape[2]);

    int64_t a2aAxes[2] = {-2, -1};
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
    uint64_t workspaceSize = 0;
    // mx量化场景量化参数：x1QuantMode=x2QuantMode=6，commQuantMode/commQuantDtype为预留参数，
    // x1QuantDtype在mx量化场景不生效配置0，groupSize=4295032864([gsM,gsN,gsK]=[1,1,32])
    const int64_t x1QuantMode = g_mxQuantMode;
    const int64_t x2QuantMode = g_mxQuantMode;
    const int64_t commQuantMode = 0;
    const int64_t commQuantDtype = -1;
    const int64_t x1QuantDtype = 0;
    const int64_t groupSize = g_mxGroupSize;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    const int64_t x1ElemNum = GetShapeSize(x1Shape);
    const int64_t x2ElemNum = GetShapeSize(x2Shape);
    const int64_t outElemNum = GetShapeSize(outShape);
    const int64_t allToAllOutElemNum = GetShapeSize(allToAllOutShape);
    // MXFP8场景1字节/元素，MXFP4场景2个元素打包为1字节
    const size_t x1Bytes = (args.quantMode == MxQuantMode::MXFP8) ? x1ElemNum : x1ElemNum / 2;
    const size_t x2Bytes = (args.quantMode == MxQuantMode::MXFP8) ? x2ElemNum : x2ElemNum / 2;
    const size_t allToAllOutBytes =
        (args.quantMode == MxQuantMode::MXFP8) ? allToAllOutElemNum : allToAllOutElemNum / 2;
    const size_t outBytes = outElemNum * sizeof(op::fp16_t);
    // x1/x2/alltoallout数据类型必须一致(MXFP4场景x1/x2必须同为FLOAT4_E2M1)
    const aclDataType xDtype =
        (args.quantMode == MxQuantMode::MXFP8) ? aclDataType::ACL_FLOAT8_E4M3FN : aclDataType::ACL_FLOAT4_E2M1;

    // 构造host侧数据
    std::vector<uint8_t> x1HostData(x1Bytes);
    std::vector<uint8_t> x2HostData(x2Bytes);
    std::vector<float> biasHostData(g_n, 1.0f);
    // E8M0 scale每个元素1字节，每个scale覆盖K方向连续32个元素
    std::vector<uint8_t> x1ScaleHostData(GetShapeSize(x1ScaleShape));
    std::vector<uint8_t> x2ScaleHostData(GetShapeSize(x2ScaleShape));
    std::vector<op::fp16_t> outHostData(outElemNum, 0);
    std::vector<uint8_t> allToAllOutHostData(allToAllOutBytes, 0);
    if (args.quantMode == MxQuantMode::MXFP8) {
        FillFp8HostData(x1HostData);
        FillFp8HostData(x2HostData);
    } else {
        FillFp4PackedHostData(x1HostData, x1ElemNum);
        FillFp4PackedHostData(x2HostData, x2ElemNum);
    }
    FillE8M0ScaleHostData(x1ScaleHostData);
    FillE8M0ScaleHostData(x2ScaleHostData);

    // 创建tensor
    ret = CreateAclTensorWithBytes(x1HostData.data(), x1Bytes, x1Shape, &x1DeviceAddr, xDtype, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorWithBytes(x2HostData.data(), x2Bytes, x2Shape, &x2DeviceAddr, xDtype, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorWithBytes(x1ScaleHostData.data(), x1ScaleHostData.size(), x1ScaleShape, &x1ScaleDeviceAddr,
                                   aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorWithBytes(x2ScaleHostData.data(), x2ScaleHostData.size(), x2ScaleShape, &x2ScaleDeviceAddr,
                                   aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorWithBytes(allToAllOutHostData.data(), allToAllOutBytes, allToAllOutShape, &allToAllOutDeviceAddr,
                                   xDtype, &allToAllOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 保存输入数据为bin文件
    const std::string rankDir = args.outputDir + "/" + modeStr + "/rank" + std::to_string(args.rankId);
    MakeDirs(rankDir);
    ret = SaveBinFile(rankDir + "/input_x1.bin", x1HostData.data(), x1Bytes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = SaveBinFile(rankDir + "/input_x2.bin", x2HostData.data(), x2Bytes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = SaveBinFile(rankDir + "/input_bias.bin", biasHostData.data(), biasHostData.size() * sizeof(float));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = SaveBinFile(rankDir + "/input_x1_scale.bin", x1ScaleHostData.data(), x1ScaleHostData.size());
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = SaveBinFile(rankDir + "/input_x2_scale.bin", x2ScaleHostData.data(), x2ScaleHostData.size());
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一段接口(mx量化场景transposeX2必须为true)
    ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(
        x1, x2, bias, x1Scale, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional, hcomName,
        alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype, groupSize, false,
        true, out, allToAllOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 100000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnAlltoAllQuantMatmul(%s) execute success \n", args.rankId, modeStr);

    // 拷贝输出数据到host侧并保存为bin文件
    ret = aclrtMemcpy(outHostData.data(), outBytes, outDeviceAddr, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out to host failed. ERROR: %d\n", ret); return ret);
    ret = SaveBinFile(rankDir + "/output_out.bin", outHostData.data(), outBytes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(allToAllOutHostData.data(), allToAllOutBytes, allToAllOutDeviceAddr, allToAllOutBytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy alltoallout to host failed. ERROR: %d\n", ret); return ret);
    ret = SaveBinFile(rankDir + "/output_alltoall_out.bin", allToAllOutHostData.data(), allToAllOutBytes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 打印首个输出数值，便于快速观察
    LOG_PRINT("[INFO] rank %d out[0] = %f, out[1] = %f\n", args.rankId, static_cast<float>(outHostData[0]),
              static_cast<float>(outHostData[1]));

    // 释放device资源，需要根据具体API的接口定义修改
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (x1Scale != nullptr) {
        aclDestroyTensor(x1Scale);
    }
    if (x2Scale != nullptr) {
        aclDestroyTensor(x2Scale);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (allToAllOut != nullptr) {
        aclDestroyTensor(allToAllOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (x1ScaleDeviceAddr != nullptr) {
        aclrtFree(x1ScaleDeviceAddr);
    }
    if (x2ScaleDeviceAddr != nullptr) {
        aclrtFree(x2ScaleDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (allToAllOutDeviceAddr != nullptr) {
        aclrtFree(allToAllOutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (alltoAllAxesOptional != nullptr) {
        aclDestroyIntArray(alltoAllAxesOptional);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Ascend 950PR/Ascend 950DT实现，必须在Ascend 950PR/Ascend 950DT上运行
    uint32_t rankId = 0;
    uint32_t rankSize = 2;
    std::string quantModeStr = "mxfp8";
    std::string outputDir = "./result";
    static const struct option longOptions[] = {{"rank_id", 1, 0, 'r'},
                                                {"rank_size", 1, 0, 's'},
                                                {"quant_mode", 1, 0, 'q'},
                                                {"output_dir", 1, 0, 'o'},
                                                {0, 0, 0, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "r:s:q:o:", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'r':
                rankId = static_cast<uint32_t>(atoi(optarg));
                break;
            case 's':
                rankSize = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'q':
                quantModeStr = optarg;
                break;
            case 'o':
                outputDir = optarg;
                break;
            default:
                LOG_PRINT("[WARN] unknown option: %d\n", opt);
                break;
        }
    }
    MxQuantMode quantMode;
    if (quantModeStr == "mxfp8") {
        quantMode = MxQuantMode::MXFP8;
    } else if (quantModeStr == "mxfp4") {
        quantMode = MxQuantMode::MXFP4;
    } else {
        LOG_PRINT("[ERROR] invalid quant_mode: %s, only support mxfp8/mxfp4\n", quantModeStr.c_str());
        return -1;
    }
    // mx量化场景约束：BS必须整除rankSize，rankSize支持2/4/8/16卡
    CHECK_RET(rankSize >= 2, LOG_PRINT("[ERROR] rank_size %u must be >= 2\n", rankSize); return -1);
    CHECK_RET(g_bs % rankSize == 0, LOG_PRINT("[ERROR] bs %ld must be divisible by rank_size %u\n", g_bs, rankSize);
              return -1);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    aclrtStream stream;
    aclrtContext context;
    HcclComm hcclComm = nullptr;
    // 单进程单rank模型，进程内rankId与deviceId一致
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    // 使用HcclCommInitClusterInfoConfig接口初始化通信域
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    // 算子默认支持确定性计算
    commConfig.hcclDeterministic = 1;
    // 设置通信缓冲区大小(单位MB)
    commConfig.hcclBufferSize = 200;
    // 设置通信域名称
    ret = strcpy_s(commConfig.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "hccl_a2a_quant_matmul");
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] hcclCommName strcpy failed. ret = %d\n", ret); return ret);
    // 从环境变量获取rank table文件路径
    const char *rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr,
              LOG_PRINT("[ERROR] get RANK_TABLE_FILE failed, please export RANK_TABLE_FILE first.\n"); return -1);
    ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret = %d\n", ret);
              return ret);

    Args args;
    args.rankId = rankId;
    args.rankSize = rankSize;
    args.quantMode = quantMode;
    args.hcclComm = hcclComm;
    args.stream = stream;
    args.context = context;
    args.outputDir = outputDir;
    ret = LaunchOneRankAlltoAllMxQuantMatmul(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LaunchOneRankAlltoAllMxQuantMatmul failed. ret = %d\n", ret);
              return ret);

    // 释放通信域和运行时资源
    HcclCommDestroy(hcclComm);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(rankId);
    aclFinalize();
    LOG_PRINT("[INFO] rank %d test_aclnn_allto_all_quant_matmul(%s) run success\n", rankId, quantModeStr.c_str());
    return 0;
}

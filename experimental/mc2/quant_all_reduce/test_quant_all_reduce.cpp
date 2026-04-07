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
 * \file test_aclnn_quant_all_reduce.cpp
 * \brief <<<>>>测试样例
 */
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <fstream>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "./kernel/quant_all_reduce_tiling_data.h"
#include "./quant_all_reduce.h"

using namespace std;
using namespace AscendC;
// 校验返回值
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t BLOCK_NUM = 64;
constexpr int64_t DEV_NUM = 4;
constexpr const char* PATH_PREFIX = "./golden/quantallreduce_";
int streamWithTimeout = 10000;
bool use_golden = false;
// 获取Shape对应的lengthSize
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
// 将filePath路径的文件将bufferSize大小的内容读出到buffer为起始的地址
int ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    // 定义文件状态结构体，用于存储stat获取的文件属性（大小、类型、权限等）
    struct stat sBuf;
    // 调用stat函数获取文件属性，filePath.data()与c_str()等效，返回const char*路径
    int fileStatus = stat(filePath.data(), &sBuf);
    // 校验stat执行结果：失败返回-1，说明文件不存在/无访问权限等
    if (fileStatus == -1) {
        printf("Failed to get file\n");
        return -1;
    }
    // 校验文件类型：S_ISREG判断是否为普通文件，排除目录、设备文件、管道等非普通文件
    if (S_ISREG(sBuf.st_mode) == 0) {
        printf("%s is not a file, please enter a file.\n", filePath.c_str());
        return -1;
    }
    // 定义C++文件输入流对象，用于二进制方式读取文件
    std::ifstream file;
    // 以二进制模式打开文件：ios::binary避免文本模式的换行符/EOF自动转换，保证数据原始性
    file.open(filePath, std::ios::binary);
    // 校验文件是否成功打开：失败可能是权限不足、文件被占用等
    if (!file.is_open()) {
        printf("Open file failed. path = %s.\n", filePath.c_str());
        return -1;
    }
    // 获取文件流的底层文件缓冲区对象指针，用于精准控制文件读写/偏移
    std::filebuf *buf = file.rdbuf();
    // 将文件读指针移动到文件末尾，获取文件总字节数（pubseekoff：偏移操作）
    // 参数：0-偏移量，ios::end-偏移基准（文件末尾），ios::in-操作模式（读）
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    // 校验文件是否为空：空文件无数据可读，直接返回失败
    if (size == 0) {
        printf("File size is 0\n");
        file.close();
        return -1;
    }
    // 校验缓冲区大小是否足够：避免缓冲区溢出，保证数据完整存储
    if (size > bufferSize) {
        printf("File size is larger than buffer size.\n");
        file.close();
        return -1;
    }
    // 将文件读指针重置到文件开头，准备读取全部数据
    // pubseekpos：绝对位置定位，0-文件开头，ios::in-读模式
    buf->pubseekpos(0, std::ios::in);
    // 从文件开头读取fileTotalSize字节数据到缓冲区
    // sgetn：二进制批量读取，强转char*是因为流操作默认基于字符类型，void*需显式转换 
    buf->sgetn(static_cast<char *>(buffer), size);
    // 关闭文件流：释放底层文件描述符和缓冲区资源，ifstream析构也会自动关闭，显式关闭更规范
    file.close();
    return ACL_SUCCESS;
} 
// 将以buffer为起始地址size大小内容写入到filePath所在路径的文件
int WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    // 校验数据缓冲区指针：空指针直接返回写入失败
    if (buffer == nullptr) {
        printf("Write file failed. Buffer is nullptr.\n");
        return -1;
    }
    // 打开/创建文件：O_RDWR-读写模式 | O_CREAT-文件不存在则创建；0666-文件默认权限
    // open成功返回非负文件描述符fd，失败返回-1
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        printf("Open file failed. path = %s\n", filePath.c_str());
        return -1;
    }
    // 获取文件独占锁,flock失败返回-1，需及时关闭文件描述符避免资源泄漏
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << std::system_error(errno, std::generic_category()).what()
                 << std::endl;
        close(fd);
        return -1;
    }
    // 设置文件写入偏移量：SEEK_SET-以文件开头为基准，移动offset字节
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << std::system_error(errno, std::generic_category()).what() 
                << std::endl;
        close(fd);
        return -1;
    }
    // 执行文件写入：将buffer中size字节数据写入fd，static_cast强转为char*（write要求字符指针）
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << std::system_error(errno, std::generic_category()).what()
                 << std::endl;
    }
    // 释放资源：先释放文件锁，再关闭文件描述符（顺序不可颠倒）
    flock(fd, LOCK_UN);
    close(fd);
    return ACL_SUCCESS;
}
template<typename T>
int CreateDeviceTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template<typename T>
int LoadDeviceTensorFromGolden(const std::string &fileName, const std::vector<int64_t> &shape, void **deviceAddr, int rankId)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    
    uint8_t *hostData;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&hostData), size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMallocHost failed. ret: %d\n", ret); return ret);
    
    std::string filePath = std::string(PATH_PREFIX) + fileName + "_" + std::to_string(rankId) + ".bin";
    ret = ReadFile(filePath, hostData, size);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[WARN] Failed to read golden file %s, using zeros\n", filePath.c_str());
        memset(hostData, 0, size);
    }
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    
    ret = aclrtFreeHost(hostData);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtFreeHost failed. ret: %d\n", ret); return ret);
    return ACL_SUCCESS;
}
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

int CreatTilingDataAndContext(const char* hcomName, aclrtStream stream, void **deviceTilingAddr, void **deviceContextAddr,
                               int64_t bs, int64_t hiddenSize, int64_t hcclBufferSize) 
{
    QuantAllReduceTilingData *tilingData = new QuantAllReduceTilingData();
    if (tilingData == nullptr) {
        LOG_PRINT("[ERROR] tilingData is nullptr\n");
        return -1;
    } 
    Mc2InitTiling mc2InitTiling; 
    Mc2CcTiling mc2CcTiling;
    tilingData->mc2InitTiling = mc2InitTiling;
    tilingData->mc2CcTiling = mc2CcTiling;
    tilingData->quantAllReduceTilingInfo.bs = bs;
    tilingData->quantAllReduceTilingInfo.hiddenSize = hiddenSize;
    tilingData->quantAllReduceTilingInfo.scaleHiddenSize = hiddenSize;
    tilingData->quantAllReduceTilingInfo.aivNum = BLOCK_NUM;
    tilingData->quantAllReduceTilingInfo.totalWinSize = hcclBufferSize;
    
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);
    int ret = mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2InitTiling failed. ret = %d\n", ret); return ret);
    ret = mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2CcTiling failed. ret = %d\n", ret); return ret);
    
    int tilingSize = sizeof(struct QuantAllReduceTilingData);
    ret = aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc TilingData failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceTilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy TilingData failed. ret: %d\n", ret); return ret);
    
    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(hcomName, &commHandle);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcomGetCommHandleByGroup failed. ret = %d\n", ret); return ret);    
    void *mc2Context = nullptr;
    ret = HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclAllocComResourceByTiling failed. ret = %d\n", ret); return ret);
    if (mc2Context == nullptr) {
        LOG_PRINT("[ERROR] mc2Context is nullptr\n");
        return -1;
    }  
    
    int64_t mc2ContextSize = 576;
    ret = aclrtMalloc(deviceContextAddr, mc2ContextSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc Mc2Context failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceContextAddr, mc2ContextSize, mc2Context, mc2ContextSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy Mc2Context failed. ret: %d\n", ret); return ret);  
    return ACL_SUCCESS;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

// 分配设备内存
int AllocDeviceMemory(Args& args, const char* hcomName, void** xDevice, void** scalesDevice, void** workspace,
                       void** tiling, void** mc2Context, int64_t bs, int64_t hiddenSize, int64_t hcclBufferSize,
                       int64_t workSpaceSize, bool useGolden) 
{
    int ret = CreatTilingDataAndContext(hcomName, args.stream, tiling, mc2Context, bs, hiddenSize, hcclBufferSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreatTilingDataAndContext failed. ret = %d\n", ret); return ret);    
    
    std::vector<int64_t> xShape = {bs, hiddenSize};
    std::vector<int64_t> scalesShape = {bs, hiddenSize/128};
    
    if (workSpaceSize > 0) {
        ret = aclrtMalloc(workspace, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }           
    
    if (useGolden) {
        ret = LoadDeviceTensorFromGolden<int8_t>("input_x", xShape, xDevice, args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LoadDeviceTensorFromGolden X failed. ret: %d\n", ret); return ret);
        
        ret = LoadDeviceTensorFromGolden<int32_t>("input_scale", scalesShape, scalesDevice, args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LoadDeviceTensorFromGolden scales failed. ret: %d\n", ret); return ret);
    } else {
        long long xShapeSize = GetShapeSize(xShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        std::vector<int8_t> xHostData(xShapeSize, 0);
        std::vector<int32_t> scalesHostData(scalesShapeSize, 0);
        
        ret = CreateDeviceTensor(xHostData, xShape, xDevice);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateDeviceTensor X failed. ret: %d\n", ret); return ret);
        
        ret = CreateDeviceTensor(scalesHostData, scalesShape, scalesDevice);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateDeviceTensor scales failed. ret: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

int freeDeviceMemory(Args& args, void *xDeviceAddr, void *scalesDeviceAddr, void *workspaceAddr,
                      void *tilingAddr, void *mc2ContextAddr, void *outputDeviceAddr, int64_t workSpaceSize) 
{
    if (xDeviceAddr != nullptr)  aclrtFree(xDeviceAddr);
    if (scalesDeviceAddr != nullptr)  aclrtFree(scalesDeviceAddr);
    if (outputDeviceAddr != nullptr) aclrtFree(outputDeviceAddr);
    if (mc2ContextAddr != nullptr) aclrtFree(mc2ContextAddr);
    if (tilingAddr != nullptr) aclrtFree(tilingAddr);   
    int ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
    return ACL_SUCCESS;
}

int LaunchOneThreadQuantAllReduce(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);
    char hcomName[128] = {0}; 
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    
    int64_t bs = 16;
    int64_t hiddenSize = 8192;
    int64_t hcclBufferSize = 200;
    int64_t workSpaceSize = 256 * 1024 * 1024;
    
    void *xDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *tilingAddr = nullptr;    
    void *mc2ContextAddr = nullptr; 
    void *outputDeviceAddr = nullptr;
    ret = AllocDeviceMemory(args, hcomName, &xDeviceAddr, &scalesDeviceAddr, &workspaceAddr,
                            &tilingAddr, &mc2ContextAddr, bs, hiddenSize, hcclBufferSize, workSpaceSize, use_golden);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] AllocDeviceMemory failed. ret = %d \n", ret); return ret);
    
    std::vector<int64_t> outputShape = {bs, hiddenSize};
    long long outputShapeSize = GetShapeSize(outputShape) * sizeof(int16_t);
    ret = aclrtMalloc(&outputDeviceAddr, outputShapeSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc outputDevice failed. ret: %d\n", ret); return ret);    
    
    quant_all_reduce_demo(BLOCK_NUM, args.stream, (uint8_t*)xDeviceAddr, (uint8_t*)scalesDeviceAddr, (uint8_t*)outputDeviceAddr,
                            (uint8_t*)workspaceAddr, (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] device_%d aclnnQuantAllReduce execute successfully.\n", args.rankId);
    
    int16_t *output_host;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&output_host), outputShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMallocHost output failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(output_host, outputShapeSize, outputDeviceAddr, outputShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::string outputFile = std::string(PATH_PREFIX) + "output_npu_" + std::to_string(args.rankId) + ".bin";
    ret = WriteFile(outputFile, output_host, outputShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] WriteFile output failed. ret: %d\n", ret); return ret);
    ret = aclrtFreeHost(output_host);    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtFreeHost failed. ret: %d\n", ret); return ret);
    ret = freeDeviceMemory(args, xDeviceAddr, scalesDeviceAddr, workspaceAddr, tilingAddr, mc2ContextAddr, outputDeviceAddr, workSpaceSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] freeDeviceMemory failed. ret = %d\n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc > 1 && std::string(argv[1]) == "--use-golden") {
        use_golden = true;
        LOG_PRINT("[INFO] Using golden data from files\n");
    } else {
        use_golden = false;
        LOG_PRINT("[INFO] Using zero-initialized data\n");
    }
    
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);
    
    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadQuantAllReduce, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
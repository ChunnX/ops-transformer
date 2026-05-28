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
* \file test_aclnn_moe_distribute_combine_v2.cpp
* \brief
*/

#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <cstdio>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
#include "aclnnop/aclnn_moe_distribute_combine_v2.h"

#include "mpi.h"
#include "acl/acl.h"
#include "shmem.h"
#include <iterator>
#include <unistd.h>

int g_npus  = 8;
const char*ipport;
static const char *g_master_ip = "tcp://127.0.0.1:18899";

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream dispatchV2Stream;
    aclrtStream combineV2Stream;
    aclrtContext context;
};

const uint32_t MACHINE_NUM = 1;
const char* env_dev_num = std::getenv("ENV_DEV_NUM");

uint32_t EP_WORLD_SIZE = 2;
uint32_t TP_WORLD_SIZE = 1;
uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

const bool IS_TEST_A2 = false;
const bool IS_TEST_A3A5 = true;
const uint32_t EP_WORLD_SIZE_A2 = 8;
const uint32_t TP_WORLD_SIZE_A2 = 1;
const uint32_t DEV_NUM_A2 = EP_WORLD_SIZE_A2 * TP_WORLD_SIZE_A2;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, 
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
    );
    return 0;
}

void DestroyTensor(aclTensor *tensor) {
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

void FreeDeviceAddr(void *deviceAddr) {
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int launchOneThreadDispatchV2AndCombineV2_A3A5(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    LOG_PRINT(
        "[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
        args.rankId, hcomEpName, hcomTpName, args.dispatchV2Stream, args.combineV2Stream, args.context
    );

    // 设置场景
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 1;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
    int64_t quantMode = 0;
    int64_t globalBS = BS * EP_WORLD_SIZE;
    int64_t expertTokenNumsType = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        A = globalBS / sharedExpertRankNum;
    } else { 
        // Moe专家卡
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        A = globalBS * (localExpertNum < K ? localExpertNum : K);
    }
    std::string commAlg = "";

    /* 根据当前场景，构造device侧输入输出变量*/
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

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *expertTokenNums = nullptr;
    aclTensor *epRecvCounts = nullptr;
    aclTensor *tpRecvCounts = nullptr;
    aclTensor *expandScales = nullptr;
    
    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> expandXShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A, H};
    std::vector<int64_t> dynamicScalesShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A};
    std::vector<int64_t> expandIdxShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * localExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1};
    std::vector<int64_t> expandScalesShape{A};

    long long xShapeSize = GetShapeSize(xShape);
    long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
    long long scalesShapeSize = GetShapeSize(scalesShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

    // 构造host侧变量
    std::vector<op::fp16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        // 每个token发给moe专家{0, 1, ... k - 1}
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> scalesHostData(scalesShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    // ####### Shmem Hack START #######
    int pe_id = aclshmem_my_pe();
    int n_pes = aclshmem_n_pes();

    uint32_t trans_size = 16;
    uint8_t* custom_ptr = static_cast<uint8_t*>(aclshmem_malloc(n_pes * trans_size * sizeof(int32_t) + 16 * 1024 * 1024));
    std::cout << "[HOST custom_ptr] " << (uint64_t)custom_ptr << std::endl;

    // 准备shmem host侧数据
    const int num10 = 10;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = num10 + pe_id * 100;
    }

    aclrtMemcpy(
        custom_ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), input.data(),
        trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);  
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); 
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    /* 声明算子执行必需变量 */
    uint64_t dispatchV2WorkspaceSize = 0;
    aclOpExecutor *dispatchV2Executor = nullptr;
    void *dispatchV2WorkspaceAddr = nullptr;

    uint64_t combineV2WorkspaceSize = 0;
    aclOpExecutor *combineV2Executor = nullptr;
    void *combineV2WorkspaceAddr = nullptr;   

    /* 依次执行dispatchV2及combineV2算子 */
    // 这里修改成仅执行combine算子
    uint64_t shmemBuffer = (uint64_t)custom_ptr;
    // 调用combineV2算子第一阶段接口
    ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, shmemBuffer, outDtype, commQuantMode, groupListType,
        commAlg.c_str(), x, &combineV2WorkspaceSize, &combineV2Executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
    );
    // 根据combineV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (combineV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    printf("[HOST TIME] Launch Kernel Start !!\n");
    // 调用combineV2算子第二阶段接口
    ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2 failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
    CHECK_RET(
        ret == ACL_SUCCESS, 
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
    );

    LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributecombineV2 execute successfully.\n", args.rankId);

    // 当前同步有问题，先验证功能
    sleep(5);

    printf("[HOST TIME] Kernel Run Success !!\n");
    std::vector<int64_t> time_vec(4, 0);
    ret = aclrtMemcpy(time_vec.data(), 32, expertIdsDeviceAddr, 32, ACL_MEMCPY_DEVICE_TO_HOST);
    printf("[HOST TIME] pe=%d, total wqe: %lld, total wqe+pollcq: %lld, \n", pe_id, time_vec[0], time_vec[1]);

    // shmem validate result
    int32_t* y_host;
    size_t input_size = n_pes * trans_size * sizeof(int32_t);
    aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    aclrtMemcpy(y_host, input_size, custom_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<int> output((int*)y_host, (int*)(y_host + n_pes * trans_size));
    std::cout << "[Output] pe=" << pe_id << ": ";
    for (int i = 0; i < n_pes; i++) {
        std::cout << y_host[trans_size * i] << " ";
    }
    std::cout << std::endl;

    aclshmem_free(custom_ptr);

    // 释放device资源
    if (dispatchV2WorkspaceSize > 0) {
        aclrtFree(dispatchV2WorkspaceAddr);
    }
    if (combineV2WorkspaceSize > 0) {
        aclrtFree(combineV2WorkspaceAddr);
    }
    DestroyTensor(x);
    DestroyTensor(expertIds);
    DestroyTensor(scales);
    DestroyTensor(expertScales);
    DestroyTensor(expandX);
    DestroyTensor(dynamicScales);
    DestroyTensor(expandIdx);
    DestroyTensor(expertTokenNums);
    DestroyTensor(epRecvCounts);
    DestroyTensor(tpRecvCounts);
    DestroyTensor(expandScales);

    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(expertIdsDeviceAddr);
    FreeDeviceAddr(scalesDeviceAddr);
    FreeDeviceAddr(expertScalesDeviceAddr);
    FreeDeviceAddr(expandXDeviceAddr);
    FreeDeviceAddr(dynamicScalesDeviceAddr);
    FreeDeviceAddr(expandIdxDeviceAddr);
    FreeDeviceAddr(expertTokenNumsDeviceAddr);
    FreeDeviceAddr(epRecvCountsDeviceAddr);
    FreeDeviceAddr(expandScalesDeviceAddr);
    FreeDeviceAddr(tpRecvCountsDeviceAddr);
    
    return 0;
}

int run_example_on_A3A5()
{
    MPI_Init(nullptr, nullptr);

    int pe_id;
    int pe_size;

    MPI_Request request;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe_id);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);

    // EP就是全局通信域，相应tiling要跟着变化
    EP_WORLD_SIZE = pe_size;

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    int device_id = pe_id % g_npus;
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);

    // Hcomm通信域初始化
    HcclRootInfo rootInfo;
    int root_rank = 0;
    if (pe_id == 0) {
        HcclResult result = HcclGetRootInfo(&rootInfo);
        if (result == HCCL_SUCCESS) {
            MPI_Ibcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        } else {
            char send_str[] = "invalid";
            MPI_Ibcast(send_str, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        }
        MPI_Wait(&request, &status);
        if (result != HCCL_SUCCESS) {
            printf("Process %d Failed ! \n", root_rank);
            return result;
        }
    } else {
        MPI_Ibcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
    }

    HcclComm hcclEpComm = nullptr;
    HcclCommInitRootInfo(pe_size, &rootInfo, pe_id, &hcclEpComm);

    aclrtStream dispatchV2Stream;
    aclrtStream combineV2Stream;
    aclrtContext context;

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
    ret = aclrtCreateStream(&dispatchV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    ret = aclrtCreateStream(&combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);

    // Shmem通信域初始化
    aclshmemx_uniqueid_t default_flag_uid;
    aclshmemx_init_attr_t attributes;

    attributes.my_pe = pe_id;
    attributes.n_pes = pe_size;
    snprintf(attributes.ip_port, ACLSHMEM_MAX_IP_PORT_LEN, "%s", g_master_ip);
    attributes.local_mem_size = 1024UL * 1024UL * 1024;
    attributes.option_attr = {
        (1 << 16) + sizeof(aclshmemx_init_attr_t),
        ACLSHMEM_DATA_OP_MTE,
        DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes.comm_args = reinterpret_cast<void *>(&default_flag_uid);

    int shmem_status = 0;
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    printf("aclshmemx_init_attr start \n");
    shmem_status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    printf("aclshmemx_init_attr success \n");

    // Dispatch算子执行
    Args arg;
    std::cout << "We in rank: " << pe_id << std::endl;
    arg.rankId = pe_id;
    arg.epRankId = pe_id;
    arg.tpRankId = 0;
    arg.hcclEpComm = hcclEpComm;
    arg.hcclTpComm = nullptr;
    arg.dispatchV2Stream = dispatchV2Stream;
    arg.combineV2Stream = combineV2Stream;
    arg.context = context;
    launchOneThreadDispatchV2AndCombineV2_A3A5(arg);

    MPI_Barrier(MPI_COMM_WORLD);
    shmem_status = aclshmem_finalize();
    printf("aclshmem_finalize success \n");

    HcclCommDestroy(hcclEpComm);
    HcclCommDestroy(nullptr);
    aclrtDestroyStream(dispatchV2Stream);
    aclrtDestroyStream(combineV2Stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(pe_id);

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    MPI_Barrier(MPI_COMM_WORLD);
 	MPI_Finalize();
    return 0;
}


int main(int argc, char *argv[])
{
    // 解析shmem必要入参
    if (argc > 2)   {
        g_master_ip = argv[1];
        g_npus = atoi(argv[2]);
    }
    run_example_on_A3A5();
    return 0;
}
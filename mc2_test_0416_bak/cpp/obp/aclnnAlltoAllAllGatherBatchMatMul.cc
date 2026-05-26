#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <cmath>
#include <map>
#include <unistd.h>
#include <sys/stat.h>
#include <getopt.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_matmul_all_reduce.h"
#include "aclnnop/aclnn_matmul_all_reduce.h"
#include "aclnnop/aclnn_matmul_all_reduce_v2.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"
#include "aclnnop/aclnn_all_gather_matmul.h"
#include "aclnnop/aclnn_all_to_all_all_gather_batch_matmul.h"
#include "common.h"

using namespace std;

// 定义全局变量
std::string g_platform = "1971";
std::string g_bin_path = "./";
std::string g_is_quant = "uquant";
// 输入
std::string g_input_tensor_dtype[128] = {};
std::string g_input_tensor_format[128] = {};
std::vector<int64_t> g_x1_shape = {};
std::vector<int64_t> g_x2_shape = {};
std::vector<int64_t> g_bias_shape = {};
// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::vector<int64_t> g_gaoutput_tensor_shape = {};
std::vector<int64_t> g_bmmoutput_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
int g_allgatherout_flag = 0;
int g_bmmout_flag = 0;
// 转置开关
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
// EP域和TP域大小
int ep_size = 4;
int tp_size = 2;
// 其余attr
int act_type = 0;
int shard_type = 1; //当前只支持1

void GetOption(int argc, char **argv) {
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] =
                {
                    {"is_quant",               1, 0, 'a'},
                    {"x1_shape",               1, 0, 'b'},
                    {"x2_shape",               1, 0, 'c'},
                    {"bias_shape",             1, 0, 'd'},
                    {"ep_size",                1, 0, 'e'},
                    {"tp_size",                1, 0, 'f'},
                    {"act_type",               1, 0, 'g'},
                    {"shard_type",             1, 0, 'h'},
                    {"input_tensor_dtype",     1, 0, 'i'},
                    {"input_tensor_format",    1, 0, 'j'},
                    {"output_shape",           1, 0, 'k'},
                    {"output_dtype",           1, 0, 'l'},
                    {"output_format",          1, 0, 'm'},
                    {"n_dev",                  1, 0, 'n'},
                    {"platform",               1, 0, 'o'},
                    {"bias_flag",              1, 0, 'p'},
                    {"transpose_b_flag",       1, 0, 'q'},
                    {"bin_path",               1, 0, 'r'},
                    {"loop_cnt",               1, 0, 's'},
                    {"gatherout_shape",        1, 0, 't'},
                    {"need_activation_feature",1, 0, 'u'},
                    {"need_allgather_out",     1, 0, 'v'},
                    {"bmmout_shape",           1, 0, 'w'},
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "v:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'a':
                g_is_quant = string(optarg);
                break;
            case 'b': {
                std::vector<std::string> x1_shape;
                split(optarg, x1_shape, ',');
                for (uint32_t i = 0; i < x1_shape.size(); ++i) {
                    g_x1_shape.push_back(atoi(x1_shape[i].c_str()));
                }
            }
                break;
            case 'c': {
                std::vector<std::string> x2_shape;
                split(optarg, x2_shape, ',');
                for (uint32_t i = 0; i < x2_shape.size(); ++i) {
                    g_x2_shape.push_back(atoi(x2_shape[i].c_str()));
                }
            }
                break;
            case 'd': {
                std::vector<std::string> bias_shape;
                split(optarg, bias_shape, ',');
                for (uint32_t i = 0; i < bias_shape.size(); ++i) {
                    g_bias_shape.push_back(atoi(bias_shape[i].c_str()));
                }
            }
                break;
            case 'e': 
                ep_size = atoi(optarg);
                break;
            case 'f': 
                tp_size = atoi(optarg);
                break;
            case 'g': 
                act_type = atoi(optarg);
                break;
            case 'h': 
                shard_type = atoi(optarg);
                break;
            case 'i': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'j': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];

                }
            }
                break;
            case 'k': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(atoi(output_shape[i].c_str()));
                }
            }
                break;
            case 'l':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'm':
                g_output_tensor_format = string(optarg);
                break;
            case 'n':
                g_ndev = atoi(optarg);
                break;
            case 'o':
                g_platform = string(optarg);
                break;
            case 'p':
                g_bias_flag = atoi(optarg);
                break;
            case 'q':
                g_transpose_b_flag = atoi(optarg);
                break;
            case 'r':
                g_bin_path = string(optarg);
                break;
            case 's':
                g_loop_cnt = atoi(optarg);
                break;
            case 't':{
                std::vector<std::string> gatherout_shape;
                split(optarg, gatherout_shape, ',');
                for (uint32_t i = 0; i < gatherout_shape.size(); ++i) {
                    g_gaoutput_tensor_shape.push_back(atoi(gatherout_shape[i].c_str()));
                }
            }
                break;
            case 'u':
                g_bmmout_flag = atoi(optarg);   
                break;
            case 'v':
                g_allgatherout_flag = atoi(optarg);
                break;
            case 'w':{
                std::vector<std::string> bmmout_shape;
                split(optarg, bmmout_shape, ',');
                for (uint32_t i = 0; i < bmmout_shape.size(); ++i) {
                    g_bmmoutput_tensor_shape.push_back(atoi(bmmout_shape[i].c_str()));
                }
            }
                break;
            default:
                break;
        }
    }
}



int launchOneThread(Args &args) {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomEpName[128] = {0};;
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, stream = %p, context = %p\n", args.rankId,
        hcomEpName, hcomTpName, args.stream, args.context);
    timespec tp1, tp2;
    long cost;

    std::vector<int64_t> x1Shape = g_x1_shape;
    std::vector<int64_t> x2Shape = g_x2_shape;
    std::vector<int64_t> biasShape = g_bias_shape;
    std::vector<int64_t> OutShape = g_output_tensor_shape;
    std::vector<int64_t> gaOutShape = g_gaoutput_tensor_shape;
    std::vector<int64_t> bmmOutShape = g_bmmoutput_tensor_shape;
    std::vector<int64_t> transposeX2Shape;
    transposeX2Shape.push_back(x2Shape[0]);
    transposeX2Shape.push_back(x2Shape[2]);
    transposeX2Shape.push_back(x2Shape[1]);
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *transposeX2DeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *gaoutDeviceAddr = nullptr;
    void *bmmoutDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    // 转置
    void *transposeWorkspaceAddr = nullptr;
    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *y1Out = nullptr;
    aclTensor *y2Out = nullptr;
    aclTensor *y3Out = nullptr;
    aclTensor *transposeX2 = nullptr;
    uint64_t workspaceSize = 0;
    // 转置
    uint64_t transposeWorkspaceSize = 0;
    aclOpExecutor *executor;

    int64_t xShardType = shard_type;
    int64_t actType = act_type;

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2");
    if (g_transpose_b_flag) {
        std::vector<int64_t> dimsData = {0, 2, 1};
        // 创建dims aclIntArray
        aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
        // 创建转置后的tensor
        CreateTensorWithoutHostData(transposeX2Shape, aclFormat::ACL_FORMAT_ND, &transposeX2DeviceAddr, &transposeX2, x2Dtype);
        // 调用aclnnPermute第一段接口
        ret = aclnnPermuteGetWorkspaceSize(x2, dims, transposeX2, &transposeWorkspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnPermuteGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (transposeWorkspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&transposeWorkspaceAddr, transposeWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用aclnnPermute第二段接口
        ret = aclnnPermute(transposeWorkspaceAddr, transposeWorkspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnPermute failed. ret = %d \n", ret); return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        printf("[INFO] transpose x2 success\n");
    }
    // }
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    auto OutSize = GetShapeSize(OutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &y1Out, g_output_tensor_dtype);
    auto gaOutSize = GetShapeSize(gaOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    if (g_allgatherout_flag){
        CreateTensorWithoutHostData(gaOutShape, aclFormat::ACL_FORMAT_ND, &gaoutDeviceAddr, &y2Out, g_output_tensor_dtype);
    }
    auto bmmOutSize = GetShapeSize(bmmOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    if (g_bmmout_flag){
        CreateTensorWithoutHostData(bmmOutShape, aclFormat::ACL_FORMAT_ND, &bmmoutDeviceAddr, &y3Out, g_output_tensor_dtype);
    }
    // 内存校验
    aclrtMemAttr attr = ACL_HBM_MEM;
    size_t mem_free;
    size_t mem_total;
    auto ret_before = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_before = (mem_total - mem_free) / 1024 / 1024;
    printf(" before RunGraph : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n", mem_free / 1024 / 1024,
           mem_total / 1024 / 1024, mem_used_before, ret_before);

    for (int i = 0; i < g_loop_cnt; i++) {
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
         // 调用第一阶段接口
        if (g_transpose_b_flag) {
            ret = aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x1, transposeX2, bias, hcomEpName, hcomTpName, ep_size,
                tp_size, xShardType, act_type, y1Out, y2Out, y3Out, &workspaceSize, &executor);
        } else {
            ret = aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x1, x2, bias, hcomEpName, hcomTpName, ep_size,
                tp_size, xShardType, act_type, y1Out, y2Out, y3Out, &workspaceSize, &executor);
        }
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        ret = aclnnAlltoAllAllGatherBatchMatMul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMul failed. ret = %d \n", ret);
            return ret);
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        clock_gettime(CLOCK_MONOTONIC, &tp2);
        if (tp2.tv_sec != tp1.tv_sec) {
            cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
        } else {
            cost = tp2.tv_nsec - tp1.tv_nsec;
        }
        printf("[INFO] device:%d step:%d cost_time: %lu, \n", args.rankId, i, cost);
    }
    printf("[INFO] aclnnAlltoAllAllGatherBatchMatmul workspaceSize = %lld\n", workspaceSize);
    // 内存校验
    auto ret_after = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;
    printf("last run : free :%lld M,  total:%lld M, used :%lld M, ret :%d \n",
           mem_free / 1024 / 1024, mem_total / 1024 / 1024, mem_used_after, ret_after);
    printf("rank %d op mc2 cost memory :%lld M \n", args.rankId, mem_used_after - mem_used_before);
    //保存结果
    saveOutPut(g_bin_path,OutSize,args.rankId,outDeviceAddr,"output_npu.bin");

    if (g_allgatherout_flag){
        saveOutPut(g_bin_path,gaOutSize,args.rankId,gaoutDeviceAddr,"gatherout_npu.bin");
    }

    if (g_bmmout_flag){
        saveOutPut(g_bin_path,bmmOutSize,args.rankId,bmmoutDeviceAddr,"bmmout_npu.bin");  
    }
    // 释放资源
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (transposeX2 != nullptr) {
        aclDestroyTensor(transposeX2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (y1Out != nullptr) {
        aclDestroyTensor(y1Out);
    }
    if (y2Out != nullptr) {
        aclDestroyTensor(y2Out);
    }
    if (y3Out != nullptr) {
        aclDestroyTensor(y3Out);
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
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (gaoutDeviceAddr != nullptr) {
        aclrtFree(gaoutDeviceAddr);
    }
    if (bmmoutDeviceAddr != nullptr) {
        aclrtFree(bmmoutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    // 转置
    if (transposeWorkspaceSize > 0) {
        aclrtFree(transposeWorkspaceAddr);
    }
    ACL_CHECK(aclrtDestroyStreamForce(args.stream));
    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    ACL_CHECK(aclrtDestroyContext(args.context));
    ACL_CHECK(aclrtResetDevice(args.rankId));
    return 0;
}


int main(int argc, char *argv[]) {
    GetOption(argc, argv);
    
    int ret;
    ACL_CHECK(aclInit(nullptr));
    Args args[g_ndev];

    aclrtStream stream[g_ndev];
    aclrtContext context[g_ndev];
    int32_t devices[g_ndev];
    for (int i = 0; i < g_ndev; i++) {
        devices[i] = i;
    }
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // 初始化集合通信域
    HcclComm comms[g_ndev];
    ret = HcclCommInitAll(g_ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    // 初始化ep域  ep = 4  {0,1,2,3} {4,5,6,7}
    HcclComm commsEp[g_ndev];
    for (int i = 0; i < g_ndev / ep_size; i++) {
        ret = HcclCommInitAll(ep_size, &devices[i * ep_size], &commsEp[i * ep_size]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d \n", i, ret);
            return ret);
    }
    // 初始化tp域 tp=2 {0,4,1,5,2,6,3,7}  {0,4} {1,5} {2,6} {3,7}
    int32_t devicesTp[g_ndev];
    for (int i = 0; i < g_ndev - 1; i++) { // Tp init按照顺序init，但同域是device id配对为[0,4],[1,5],...,[3,7].
        devicesTp[i] = devices[g_ndev / tp_size * i % (g_ndev - 1)];
        std::cout << "test devices order " << i << " = " << devicesTp[i] << std::endl;
    }
    devicesTp[g_ndev - 1] = devices[g_ndev - 1];
    std::cout << "test devices order " << g_ndev - 1 << " = " << devicesTp[g_ndev - 1] << std::endl;
    HcclComm commsTp[g_ndev];
    for (int i = 0; i < g_ndev / tp_size; i++) {
        ret = HcclCommInitAll(tp_size, &devicesTp[i * tp_size], &commsTp[i * tp_size]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d \n", i, ret);
            return ret);
    }

    std::vector<std::unique_ptr<std::thread>> threads(g_ndev);
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclEpComm = commsEp[rankId];
        // Tp init按照顺序init，但同域是device id配对为[0,4],[1,5],...,[3,7].
        args[rankId].hcclTpComm = commsTp[tp_size * rankId % (g_ndev - 1)];
        if (rankId == (g_ndev - 1)) {
            args[rankId].hcclTpComm = commsTp[rankId];
            std::cout << "test devices id " << rankId << " = " << devicesTp[rankId] << std::endl;
        }
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        threads[rankId]->join();
    }
    ACL_CHECK(aclFinalize());
    return 0;
}



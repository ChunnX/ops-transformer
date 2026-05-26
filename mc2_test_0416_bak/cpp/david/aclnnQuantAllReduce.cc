#include <bits/stdc++.h>
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
// #include "acl/acl.h"
#include "aclnnop/aclnn_quant_all_reduce.h"
#include "common.h"
using namespace std;

// 定义全局变量
std::string g_platform = "1982";
std::string g_bin_path = "./";

std::vector<int64_t> g_x_shape = {};
std::vector<int64_t> g_scales_shape = {};
std::string g_x_tensor_dtype = "fp16";
std::string g_scales_tensor_dtype = "fp32";

// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::string g_output_tensor_format = "ND";

// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int rankId = 0;

// 保存卡id的结果
int save_rankid = -1;
int g_is_mxfp = 0;
int64_t g_hcclBufferSize = 200;
void GetOption(int argc, char **argv) {
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] =
                {
                    {"x_shape",                1, 0, 'a'},
                    {"scales_shape",           1, 0, 'b'},
                    {"x_dtype",                1, 0, 'c'},
                    {"scales_dtype",           1, 0, 'd'},
                    {"output_shape",           1, 0, 'e'},
                    {"output_dtype",           1, 0, 'f'},
                    {"n_dev",                  1, 0, 'g'},
                    {"bin_path",               1, 0, 'h'},
                    {"loop_cnt",               1, 0, 'i'},
                    {"rankId",                 1, 0, 'j'},
                    {"is_mxfp",                1, 0, 'k'},
                    // {"save_rankid",            1, 0, 'n'},
                    {0,                        0, 0, 0}
                };
        int c = getopt_long(argc, argv, "C:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'a': {
                std::vector<std::string> x_shape;
                split(optarg, x_shape, ',');
                for (uint32_t i = 0; i < x_shape.size(); ++i) {
                    g_x_shape.push_back(atoi(x_shape[i].c_str()));
                }
            }
                break;
            case 'b': {
                std::vector<std::string> scales_shape;
                split(optarg, scales_shape, ',');
                for (uint32_t i = 0; i < scales_shape.size(); ++i) {
                    g_scales_shape.push_back(atoi(scales_shape[i].c_str()));
                }
            }
                break;
            case 'c': {
                g_x_tensor_dtype = string(optarg);
            }
                break;
            case 'd': {
                g_scales_tensor_dtype = string(optarg);
            }
                break;
            case 'e': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(atoi(output_shape[i].c_str()));
                }
            }
                break;
            case 'f':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'g':
                g_ndev = atoi(optarg);
                break;
            case 'h':
                g_bin_path = string(optarg);
                break;
            case 'i':
                g_loop_cnt = atoi(optarg);
                break;
            case 'j':
                rankId = atoi(optarg);
                break;
            case 'k':
                g_is_mxfp = atoi(optarg);
                break;
            // case 'n':
            //     save_rankid = atoi(optarg);
            //     break;
            default:
                break;
        }
    }
}
int launchOneThread(Args &args, aclrtStream &stream) {
    printf("----Use aclnnQuantReduceScatter\n");
    int ret = aclrtSetCurrentContext(args.context);;
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);

    std::vector<int64_t> xShape = g_x_shape;
    std::vector<int64_t> scalesShape = g_scales_shape;
    std::vector<int64_t> OutputShape = g_output_tensor_shape;

    // input
    void *xDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    // mc2 out
    void *outputDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    aclOpExecutor *executor;
    std::string xDtype = g_x_tensor_dtype;
    std::string scalesDtype = g_scales_tensor_dtype;
    std::string xFormat = "ND";
    std::string scalesFormat = "ND";

    aclTensor *x = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *Output = nullptr;
    uint64_t workspaceSize = 0;

    // 创建 tensor
    CreateTensor(xShape, aclFormat::ACL_FORMAT_ND, &xDeviceAddr, &x, xDtype, g_bin_path, args.rankId, "x");
    CreateTensor(scalesShape, aclFormat::ACL_FORMAT_ND, &scalesDeviceAddr, &scales, scalesDtype, g_bin_path, args.rankId, "scales");

    auto OutputSize = GetShapeSize(OutputShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(OutputShape, aclFormat::ACL_FORMAT_ND, &outputDeviceAddr, &Output, g_output_tensor_dtype);

    for (int i = 0; i < g_loop_cnt; i++) {
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        printf("[INFO] aclnnQuantAllReduceGetWorkspaceSize start\n");
        // 调用第一段接口
        ret = aclnnQuantAllReduceGetWorkspaceSize(
            x, scales, hcomName, "sum", Output, &workspaceSize, &executor);
        printf("[INFO] aclnnQuantAllReduceGetWorkspaceSize end\n");
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantAllReduce failed. ret = %d \n", ret); return ret);
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        printf("[INFO] aclnnQuantAllReduce start\n");
        ret = aclnnQuantAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
        printf("[INFO] aclnnQuantAllReduce end\n");
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantAllReduce failed. ret = %d \n", ret); return ret); 
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));
    }
    printf("[INFO] allreduce workspaceSize = %ld\n", workspaceSize);
    saveOutPut(g_bin_path,OutputSize,args.rankId,outputDeviceAddr,"output_npu.bin");
    if (x != nullptr) { aclDestroyTensor(x);}
    if (scales != nullptr) {aclDestroyTensor(scales);}
    if (Output != nullptr) {aclDestroyTensor(Output);}
    if (xDeviceAddr != nullptr) {aclrtFree(xDeviceAddr);}
    if (scalesDeviceAddr != nullptr) {aclrtFree(scalesDeviceAddr);}
    if (outputDeviceAddr != nullptr) {aclrtFree(outputDeviceAddr);}
    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
    DestoryRes(args);
    return 0;
}

int main(int argc, char *argv[]) {
    GetOption(argc, argv);
    int ret;
    int32_t devices[g_ndev];
    for (int i = 0; i < g_ndev; i++) {
        devices[i] = i;
    }
    ACL_CHECK(aclInit(nullptr));
    // 启动多线程
    Args args;
    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    // 初始化集合通信域
    //HcclRootInfo *c = new HcclRootInfo;
    //for (int i=0; i<10; i++) {
        //c->internal[i] = 'a';
    //}
    //c->internal[10] = '\0';
    //LOG_PRINT("before HcclCommInitRootInfo g_ndev: %d, rankId: %d\n", g_ndev, rankId);
    //ret = HcclCommInitRootInfo(g_ndev, c, rankId, &comms);
    //CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitRootInfo failed. ERROR: %d\n", ret); return ret);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = g_hcclBufferSize;
    strncpy(config.hcclCommName, "hccl_comm_test", COMM_NAME_MAX_LENGTH - 1);
    string rankTableFile = getenv("RANK_TABLE_FILE");
    std::cout << "rankTableFilePath is :" << rankTableFile << std::endl;
    // int rank_id = rankId + FIRST_RANK_ID;
    ret = HcclCommInitClusterInfoConfig(rankTableFile.c_str(), rankId, &config, &comms);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig ep world %d failed. ret = %d\n", rankId, ret); return ret
    );

    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    int re = launchOneThread(args, stream);
    LOG_PRINT("end launchOneThread re = %d", re);
    ACL_CHECK(aclFinalize());
    return 0;
}

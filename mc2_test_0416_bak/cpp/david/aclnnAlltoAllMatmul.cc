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
#include "aclnnop/aclnn_allto_all_matmul.h"  // 补充，加上正确头文件名
#include "common.h"

using namespace std;

// 定义全局变量
std::string g_platform = "1982";
std::string g_bin_path = "./";
// 输入
std::string g_input_tensor_dtype[128] = {};
std::string g_input_tensor_format[128] = {};
std::vector<int64_t> g_x1_shape = {};
std::vector<int64_t> g_x2_shape = {};
std::vector<int64_t> g_bias_shape = {};
// 输出
std::string g_output_tensor_dtype = "fp16";
std::string g_alltoalloutput_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::vector<int64_t> g_alltoall_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
int g_alltoall_flag = 0;
// 转置开关
int g_transpose_x1_flag = 0;
int g_transpose_x2_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int rankId = 0;
int64_t g_hcclBufferSize = 200;
void GetOption(int argc, char **argv) {
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] =
                {
                        {"x1_shape",               1, 0, 'a'},
                        {"x2_shape",               1, 0, 'b'},
                        {"bias_shape",             1, 0, 'c'},
                        {"input_tensor_dtype",     1, 0, 'd'},
                        {"input_tensor_format",    1, 0, 'e'},
                        {"output_shape",           1, 0, 'f'},
                        {"output_dtype",           1, 0, 'g'},
                        {"alltoall_dtype",         1, 0, 'h'},
                        {"output_format",          1, 0, 'i'},
                        {"n_dev",                  1, 0, 'j'},
                        {"bias_flag",              1, 0, 'k'},
                        {"transpose_x1_flag",      1, 0, 'l'},
                        {"transpose_x2_flag",      1, 0, 'm'},
                        {"bin_path",               1, 0, 'n'},
                        {"loop_cnt",               1, 0, 'o'},
                        {"alltoall_shape",         1, 0, 'p'},
                        {"alltoall_flag",          1, 0, 'q'},
                        {"rankId",                 1, 0, 'r'},
                        {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "E:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'a': {
                std::vector<std::string> x1_shape;
                split(optarg, x1_shape, ',');
                for (uint32_t i = 0; i < x1_shape.size(); ++i) {
                    g_x1_shape.push_back(atoi(x1_shape[i].c_str()));
                }
            }
                break;
            case 'b': {
                std::vector<std::string> x2_shape;
                split(optarg, x2_shape, ',');
                for (uint32_t i = 0; i < x2_shape.size(); ++i) {
                    g_x2_shape.push_back(atoi(x2_shape[i].c_str()));
                }
            }
                break;
            case 'c': {
                std::vector<std::string> bias_shape;
                split(optarg, bias_shape, ',');
                for (uint32_t i = 0; i < bias_shape.size(); ++i) {
                    g_bias_shape.push_back(atoi(bias_shape[i].c_str()));
                }
            }
                break;
            case 'd': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'e': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];

                }
            }
                break;
            case 'f': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(atoi(output_shape[i].c_str()));
                }
            }
                break;
            case 'g':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'h':
                g_alltoalloutput_tensor_dtype = string(optarg);
                break;
            case 'i':
                g_output_tensor_format = string(optarg);
                break;
            case 'j':
                g_ndev = atoi(optarg);
                break;
            case 'k':
                g_bias_flag = atoi(optarg);
                break;
            case 'l':
                g_transpose_x1_flag = atoi(optarg);
                break;
            case 'm':
                g_transpose_x2_flag = atoi(optarg);
                break;
            case 'n':
                g_bin_path = string(optarg);
                break;
            case 'o':
                g_loop_cnt = atoi(optarg);
                break;
            case 'p': {
                std::vector<std::string> alltoall_shape;
                split(optarg, alltoall_shape, ',');
                for (uint32_t i = 0; i < alltoall_shape.size(); ++i) {
                    g_alltoall_tensor_shape.push_back(atoi(alltoall_shape[i].c_str()));
                }
            }
                break;
            case 'q':
                g_alltoall_flag = atoi(optarg);
                break;
            case 'r':
                rankId = atoi(optarg);
                break;
            default:
                break;
        }
    }
}

int launchOneThread(Args &args) {
    printf("----launchOneThread aclnnAlltoAllMatmul\n");
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);

    std::vector<int64_t> x1Shape = g_x1_shape;
    std::vector<int64_t> x2Shape = g_x2_shape;
    std::vector<int64_t> biasShape = g_bias_shape;
    std::vector<int64_t> outShape = g_output_tensor_shape;
    std::vector<int64_t> alltoAllShape = g_alltoall_tensor_shape;
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *alltoAllDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *output = nullptr;
    aclTensor *alltoAllOut = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2");
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    auto outSize = GetShapeSize(outShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(outShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &output, g_output_tensor_dtype);
    aclDataType outputTensorDtype = ACL_DT_UNDEFINED;
    ret = aclGetDataType(output, &outputTensorDtype);
    CHECK_RET(ret == 0, printf("----[ERROR] matmulalltoall aclGetDataType failed. ret = %d \n", ret); return ret);
    printf("----outputTensorDtype = %d \n", outputTensorDtype);
    auto alltoAllSize = 0;
    if (g_alltoall_flag) {
        alltoAllSize = GetShapeSize(alltoAllShape) * dtype_to_sizeof(g_alltoalloutput_tensor_dtype);
        CreateTensorWithoutHostData(alltoAllShape, aclFormat::ACL_FORMAT_ND, &alltoAllDeviceAddr, &alltoAllOut, g_alltoalloutput_tensor_dtype);
    }
    for (int i = 0; i < g_loop_cnt; i++) {
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        printf("[INFO] aclnnAlltoAllMatmulGetWorkspaceSize start\n");
        ret = aclnnAlltoAllMatmulGetWorkspaceSize(x1, x2, bias, nullptr, hcomName, g_transpose_x1_flag, g_transpose_x2_flag, output, alltoAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnAlltoAllMatmulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        printf("[INFO] aclnnAlltoAllMatmul start\n");
        ret = aclnnAlltoAllMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnAlltoAllMatmul failed. ret = %d \n", ret); return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));
    }
    printf("[INFO] AlltoAllMatmul workspaceSize = %lu\n", workspaceSize);
    //保存结果
    saveOutPut(g_bin_path, outSize, args.rankId, outDeviceAddr, "output_npu.bin");
    if (g_alltoall_flag && (alltoAllSize != 0)) {
        saveOutPut(g_bin_path, alltoAllSize, args.rankId, alltoAllDeviceAddr, "alltoall_out_npu.bin");
    }
    // 释放资源
    if (x1 != nullptr) {aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (bias != nullptr) {aclDestroyTensor(bias);}
    if (output != nullptr) {aclDestroyTensor(output);}
    if (alltoAllOut != nullptr) {aclDestroyTensor(alltoAllOut);}
    if (x1DeviceAddr != nullptr) {aclrtFree(x1DeviceAddr);}
    if (x2DeviceAddr != nullptr) {aclrtFree(x2DeviceAddr);}
    if (biasDeviceAddr != nullptr) {aclrtFree(biasDeviceAddr);}
    if (outDeviceAddr != nullptr) {aclrtFree(outDeviceAddr);}
    if (alltoAllDeviceAddr != nullptr) {aclrtFree(alltoAllDeviceAddr);}
    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
    DestoryRes(args);
    return 0;
}

int main(int argc, char *argv[]) {
    printf("----Use aclnnAlltoAllMatmul main\n");
    GetOption(argc, argv);
    printf("----para transfer success\n");
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
    printf("----start aclrtSetDevice\n");
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    printf("----start aclrtCreateContext\n");
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    printf("----start aclrtCreateStream\n");
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = g_hcclBufferSize;
    printf("----start strncpy hcclCommName \n");
    strncpy(config.hcclCommName, "hccl_comm_test", COMM_NAME_MAX_LENGTH - 1);
    printf("----start getenv from RANK_TABLE_FILE \n");
    string rankTableFile = getenv("RANK_TABLE_FILE");
    std::cout << "rankTableFilePath is :" << rankTableFile << std::endl;
    // int rank_id = rankId + FIRST_RANK_ID;
    printf("----start HcclCommInitClusterInfoConfig \n");
    ret = HcclCommInitClusterInfoConfig(rankTableFile.c_str(), rankId, &config, &comms);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig ep world %d failed. ret = %d\n", rankId, ret); return ret
    );
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    LOG_PRINT("before launchOneThread");
    int re = launchOneThread(args);
    LOG_PRINT("end launchOneThread re = %d", re);
    ACL_CHECK(aclFinalize());
    return 0;
}

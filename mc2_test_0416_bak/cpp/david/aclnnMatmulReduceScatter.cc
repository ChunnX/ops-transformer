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
#include "aclnnop/aclnn_matmul_reduce_scatter.h"
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
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
// 转置开关
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 1;
int rankId = 0;

void GetOption(int argc, char **argv) {
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] =
                {
                    {"is_quant",               1, 0, 'a'},
                    {"x1_shape",               1, 0, 'b'},
                    {"x2_shape",               1, 0, 'c'},
                    {"bias_shape",             1, 0, 'd'},
                    {"input_tensor_dtype",     1, 0, 'e'},
                    {"input_tensor_format",    1, 0, 'f'},
                    {"output_shape",           1, 0, 'g'},
                    {"output_dtype",           1, 0, 'h'},
                    {"output_format",          1, 0, 'i'},
                    {"n_dev",                  1, 0, 'j'},
                    {"platform",               1, 0, 'k'},
                    {"bias_flag",              1, 0, 'l'},
                    {"transpose_b_flag",       1, 0, 'm'},
                    {"bin_path",               1, 0, 'n'},
                    {"loop_cnt",               1, 0, 'o'},
                    {"rankId",                 1, 0, 'p'},
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "q:", longOptions, &optionIndex);
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
            case 'e': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'f': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];
                }
            }
                break;
            case 'g': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(atoi(output_shape[i].c_str()));
                }
            }
                break;
            case 'h':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'i':
                g_output_tensor_format = string(optarg);
                break;
            case 'j':
                g_ndev = atoi(optarg);
                break;
            case 'k':
                g_platform = string(optarg);
                break;
            case 'l':
                g_bias_flag = atoi(optarg);
                break;
            case 'm':
                g_transpose_b_flag = atoi(optarg);
                break;
            case 'n':
                g_bin_path = string(optarg);
                break;
            case 'o':
                g_loop_cnt = atoi(optarg);
                break;
            case 'p':
                rankId = atoi(optarg);
                break;
            default:
                break;
        }
    }
}

int launchOneThread(Args &args) {
    printf("----Use aclnnMatmulReduceScatter\n");
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);

    std::vector<int64_t> x1Shape = g_x1_shape;
    std::vector<int64_t> x2Shape = g_x2_shape;
    std::vector<int64_t> biasShape = g_bias_shape;
    std::vector<int64_t> OutShape = g_output_tensor_shape;
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    // 转置
    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *scOut = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    int64_t commTurn = 0;
    int64_t streamMode = 1;

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2", g_transpose_b_flag);
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    auto OutSize = GetShapeSize(OutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &scOut, g_output_tensor_dtype);
    for (int i = 0; i < g_loop_cnt; i++) {
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        
        if (x1 == nullptr) {
            printf("[ERROR]x1 is nullptr \n");
        }
        ret = aclnnMatmulReduceScatterGetWorkspaceSize(x1, x2, bias, hcomName, "sum", commTurn, streamMode, scOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulReduceScatterGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));
        // 调用第二段接口
        ret = aclnnMatmulReduceScatter(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulReduceScatter failed. ret = %d \n", ret); return ret);
        //     }
        // }
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));
    }
    printf("[INFO] mm_reducescatter workspaceSize = %lu\n", workspaceSize);

    //保存结果
    saveOutPut(g_bin_path,OutSize,args.rankId,outDeviceAddr,"output_npu.bin");
    // 释放资源
    if (x1 != nullptr) { aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (bias != nullptr) {aclDestroyTensor(bias);}
    if (scOut != nullptr) {aclDestroyTensor(scOut);}
    if (x1DeviceAddr != nullptr) {aclrtFree(x1DeviceAddr);}
    if (x2DeviceAddr != nullptr) {aclrtFree(x2DeviceAddr);}
    if (biasDeviceAddr != nullptr) {aclrtFree(biasDeviceAddr);}
    if (outDeviceAddr != nullptr) {aclrtFree(outDeviceAddr);}
    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
    DestoryRes(args);
    return 0;
}

int main(int argc, char *argv[]) {
    int ret;
    Args args;
    GetOption(argc,argv);
    ret = init(args,g_ndev,rankId);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnMatmulReduceScatter init failed. ERROR: %d\n", ret); return -1);
    ret = launchOneThread(args);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnMatmulReduceScatter launchOneThread  failed. ERROR: %d\n", ret); return -1);
    LOG_PRINT("end launchOneThread ret = %d", ret);
    ACL_CHECK(aclFinalize());
    return 0;
}

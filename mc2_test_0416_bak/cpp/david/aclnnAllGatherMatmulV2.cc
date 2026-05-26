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
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"
#include "aclnnop/aclnn_all_gather_matmul_v2.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"
#include "common.h"
using namespace std;

// 定义全局变量
std::string g_platform = "1982";
std::string g_bin_path = "./";
std::string g_is_quant = "uquant";
// 输入
std::string g_input_tensor_dtype[128] = {};
std::string g_input_tensor_format[128] = {};
std::string g_amax_tensor_dtype = "fp32";
std::string g_gatherout_dtype;
std::vector<int64_t> g_x1_shape = {};
std::vector<int64_t> g_x2_shape = {};
std::vector<int64_t> g_bias_shape = {};

std::vector<int64_t> g_scale1_shape = {};
std::vector<int64_t> g_scale2_shape = {};

// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::vector<int64_t> g_gaoutput_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
// 转置开关
int g_transpose_a_flag = 0;
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int rankId = 0;
int g_amax_flag = 0;
// 保存卡id的结果
int save_rankid = -1;
// 是否执行小算子级联
int g_is_hccl_mm = 0;
int g_quant_flag = 0;

int64_t g_block_size = 128;
int64_t g_block_mode = 0;
bool g_per_block_flag = false;

/**
/mnt/s00654542/bin/allgather_v2_aarch --is_quant UQUANT --x1_shape 128,256 --x2_shape 256,64 --input_tensor_dtype fp8e5m2,fp8e5m2 --input_tensor_format ND,ND,ND --output_shape 256,64 --output_dtype fp16 --output_format ND --n_dev 2 --bin_path /mnt/s00654542/bin/256_256_64_false_false_fp8_2p --loop_cnt 1 --gatherout_shape 256,256 --rankId 0 --scale_value 1,1,1 --amax_dtype fp32 --gatherout_dtype fp8e5m2
**/

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
                    {"gatherout_shape",        1, 0, 'p'},
                    {"rankId",                 1, 0, 'q'},
                    {"transpose_a_flag",       1, 0, 'r'},
                    {"amax_dtype",             1, 0, 'x'},
                    {"gatherout_dtype",        1, 0, 'y'},
                    {"amax_flag",              1, 0, 'z'},
                    {"save_rankid",            1, 0, 'A'},
                    {"quant_flag",             1, 0, 'B'},
                    {"is_hccl_mm",             1, 0, 'C'},
                    {"block_size",             1, 0, 'D'},
                    {"block_mode",             1, 0, 'E'},
                    {"per_block_flag",         1, 0, 'F'},
                    {"scale1_shape",           1, 0, 'G'},
                    {"scale2_shape",           1, 0, 'H'},
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "r:", longOptions, &optionIndex);
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
            case 'p':{
                std::vector<std::string> gatherout_shape;
                split(optarg, gatherout_shape, ',');
                for (uint32_t i = 0; i < gatherout_shape.size(); ++i) {
                    g_gaoutput_tensor_shape.push_back(atoi(gatherout_shape[i].c_str()));
                }
            }
                break;
            case 'q':
                rankId = atoi(optarg);
                break;
            case 'r':
                g_transpose_a_flag = atoi(optarg);
                break;
            case 'x':
                g_amax_tensor_dtype = string(optarg);
                break;
            case 'y':
                g_gatherout_dtype = string(optarg);
                break;
            case 'z':
                g_amax_flag = atoi(optarg);
                break;
            case 'A':
                save_rankid = atoi(optarg);
                break;
            case 'B':
                g_quant_flag = atoi(optarg);
                break;
            case 'C':
                g_is_hccl_mm = atoi(optarg);
                break;
            case 'D':
                g_block_size = atoi(optarg);
                break;
            case 'E':
                g_block_mode = atoi(optarg);
                break;
            case 'F':
                g_per_block_flag = stringToBool(optarg);
                break;
            case 'G': {
                std::vector<std::string> shape;
                split(optarg, shape, ',');
                for (uint32_t i = 0; i < shape.size(); ++i) {
                    g_scale1_shape.push_back(atoi(shape[i].c_str()));
                }
            }
                break;
            case 'H': {
                std::vector<std::string> shape;
                split(optarg, shape, ',');
                for (uint32_t i = 0; i < shape.size(); ++i) {
                    g_scale2_shape.push_back(atoi(shape[i].c_str()));
                }
            }
                break;
            default:
                break;
        }
    }
}


int launchOneThread(Args &args) {
    printf("----Use aclnnAllGatherMatmulV2\n");
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);
    timespec tp1, tp2;
    long cost;

    std::vector<int64_t> x1Shape = g_x1_shape;
    std::vector<int64_t> x2Shape = g_x2_shape;
    // transx2Shape用于quantmatmulv5接口中创建x2 tensor时使用
    std::vector<int64_t> transx2Shape;
    transx2Shape.push_back(x2Shape[1]);
    transx2Shape.push_back(x2Shape[0]);


    std::vector<int64_t> biasShape = g_bias_shape;
    std::vector<int64_t> OutShape = g_output_tensor_shape;
    std::vector<int64_t> gaOutShape = g_gaoutput_tensor_shape;
    std::vector<int64_t> amaxShape;

    std::vector<int64_t> scale1_shape = g_scale1_shape;
    std::vector<int64_t> scale2_shape = g_scale2_shape;

    // input
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *transx2DeviceAddr = nullptr;
    void *transScale2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *scale1DeviceAddr = nullptr;
    void *scale2DeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    // mc2 out
    void *outDeviceAddr = nullptr;
    void *gaoutDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    // hccl_mm out
    void *golden_outDeviceAddr = nullptr;
    void *golden_gaoutDeviceAddr = nullptr;
    void *mm_workspaceAddr = nullptr;

    aclOpExecutor *executor;
    aclOpExecutor *mm_executor;
    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];
    std::string amax_dtype = g_amax_tensor_dtype;
    std::string gatherout_dtype = g_gatherout_dtype;

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *transx2 = nullptr;
    aclTensor *trans_scale2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *scale1 = nullptr;
    aclTensor *scale2 = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *amaxOut = nullptr;
    aclTensor *Out = nullptr;
    aclTensor *gaOut = nullptr;
    aclTensor *golden_Out = nullptr;
    aclTensor *golden_gaOut = nullptr;
    uint64_t workspaceSize = 0;
    uint64_t mm_workspaceSize = 0;
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t gatherIndex = 0;
    string scale_dtype = "fp32";

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2", g_transpose_b_flag);

    if (g_transpose_b_flag && g_is_hccl_mm)
    {
        // 转置场景，quantmatmulv5，x2 viewshape[n,k] storage shape:[n,k]，与mc2不同，需要单独创建
        CreateTensor(transx2Shape, aclFormat::ACL_FORMAT_ND, &transx2DeviceAddr, &transx2, x2Dtype, g_bin_path, args.rankId, "x2");
    }

    if (g_quant_flag)
    {  
        CreateTensor(scale1_shape, aclFormat::ACL_FORMAT_ND, &scale1DeviceAddr, &scale1, scale_dtype, g_bin_path, args.rankId, "scaleInv1");
        if(g_transpose_b_flag)
        {
            if(scale2_shape.size()==2)
            {    
            std::vector<int64_t>trans_scale2_shape;trans_scale2_shape.push_back(scale2_shape[1]);trans_scale2_shape.push_back(scale2_shape[0]);
            CreateTensor(trans_scale2_shape, aclFormat::ACL_FORMAT_ND, &scale2DeviceAddr, &scale2, scale_dtype, g_bin_path, args.rankId, "scaleInv2"); 
             }
        }
        else
        {
            CreateTensor(scale2_shape, aclFormat::ACL_FORMAT_ND, &scale2DeviceAddr, &scale2, scale_dtype, g_bin_path, args.rankId, "scaleInv2");   
        }
    }
// CreateTensor(scaleShape,  aclFormat::ACL_FORMAT_ND, &scaleDeviceAddr, &scale, scale_dtype, g_bin_path, args.rankId, "scale", 2);
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    auto OutSize = GetShapeSize(OutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &Out, g_output_tensor_dtype);
    auto gaOutSize = GetShapeSize(gaOutShape) * dtype_to_sizeof(x1Dtype);
    CreateTensorWithoutHostData(gaOutShape, aclFormat::ACL_FORMAT_ND, &gaoutDeviceAddr, &gaOut, x1Dtype);       // 与x1一致
    // 小算子级联输出
    if (g_is_hccl_mm && g_quant_flag){
        CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &golden_outDeviceAddr, &golden_Out, g_output_tensor_dtype);
        CreateTensorWithoutHostData(gaOutShape, aclFormat::ACL_FORMAT_ND, &golden_gaoutDeviceAddr, &golden_gaOut, x1Dtype);       // 与x1一致
        // 小算子级联：
        // 调用HCCL接口---AllGather
        auto x1Size = GetShapeSize(x1Shape);
        // allgather
        HcclAllGather((void *) x1DeviceAddr, golden_gaoutDeviceAddr, x1Size, dtype_to_HcclDataType[x1Dtype],args.hcclComm, args.stream);
        // matmul
        int64_t mm_groupt_size = 0;
        if(g_per_block_flag) {mm_groupt_size = 549764202624;}
        if (g_transpose_b_flag){
            ret = aclnnQuantMatmulV5GetWorkspaceSize(golden_gaOut, transx2, scale1, scale2, nullptr, nullptr, nullptr, nullptr, bias, 
                                                false, g_transpose_b_flag, mm_groupt_size, golden_Out, &mm_workspaceSize, &mm_executor);
        } else{
            ret = aclnnQuantMatmulV5GetWorkspaceSize(golden_gaOut, x2, scale1, scale2, nullptr, nullptr, nullptr, nullptr, bias, 
                                                false, false, mm_groupt_size, golden_Out, &mm_workspaceSize, &mm_executor);
        }
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);        
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (mm_workspaceSize > 0) {
            ret = aclrtMalloc(&mm_workspaceAddr, mm_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnQuantMatmulV5第二段接口
        ret = aclnnQuantMatmulV5(mm_workspaceAddr, mm_workspaceSize, mm_executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);   
    }

    for (int i = 0; i < g_loop_cnt; i++) {
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        printf("g_transpose_b_flag\n");
        //ret = aclnnAllGatherMatmulV2GetWorkspaceSize(
        //       x1, x2, bias, scale1, scale2, nullptr, hcomName, gatherIndex, commTurn, streamMode, Out, gaOut, amaxOut, &workspaceSize, &executor);
        ret = aclnnAllGatherMatmulV2GetWorkspaceSize(
            x1, x2, bias, scale1, scale2, nullptr,g_block_mode,g_block_size, hcomName, gatherIndex, commTurn, streamMode, Out, gaOut, amaxOut, &workspaceSize, &executor);

        printf("[INFO] aclnnAllGatherMatmulV2GetWorkspaceSize end\n");
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnAllGatherMatmulV2 failed. ret = %d \n", ret); return ret);
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        printf("[INFO] aclnnAllGatherMatmulV2 start\n");
        ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        printf("[INFO] aclnnAllGatherMatmulV2 end\n");
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnAllGatherMatmulV2 failed. ret = %d \n", ret); return ret);
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));
        clock_gettime(CLOCK_MONOTONIC, &tp2);
        if (tp2.tv_sec != tp1.tv_sec) {
            cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
        } else {
            cost = tp2.tv_nsec - tp1.tv_nsec;
        }
        printf("[INFO] device:%d allgatherV2 step:%d cost_time: %lu, \n", args.rankId, i, cost);
    }
    printf("[INFO] allgather_mm_v2 workspaceSize = %ld\n", workspaceSize);
    if (g_is_hccl_mm && g_quant_flag){
         saveOutPut(g_bin_path,OutSize,args.rankId,golden_outDeviceAddr,"output_hccl.bin");
    }

    if (golden_gaOut != nullptr) {
        aclDestroyTensor(golden_gaOut);
    }
    if (golden_Out != nullptr) {
        aclDestroyTensor(golden_Out);
    }
    if (golden_outDeviceAddr != nullptr) {
        aclrtFree(golden_outDeviceAddr);
    }
    if (golden_gaoutDeviceAddr != nullptr) {
        aclrtFree(golden_gaoutDeviceAddr);
    }
    if (save_rankid == -1 || save_rankid == args.rankId){
        //保存结果
         saveOutPut(g_bin_path,OutSize,args.rankId,outDeviceAddr,"output_npu.bin");
    }
    // 释放资源
    if (x1 != nullptr) {aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (scale1 != nullptr) {aclDestroyTensor(scale1);}
    if (scale2 != nullptr) {aclDestroyTensor(scale2);}
    if (bias != nullptr) {aclDestroyTensor(bias);}
    if (gaOut != nullptr) {aclDestroyTensor(gaOut);}
    if (Out != nullptr) {aclDestroyTensor(Out);}
    if (trans_scale2 != nullptr) {aclDestroyTensor(trans_scale2);}
    if (x1DeviceAddr != nullptr) {aclrtFree(x1DeviceAddr);}
    if (x2DeviceAddr != nullptr) {aclrtFree(x2DeviceAddr);}
    if (scale1DeviceAddr != nullptr) {aclrtFree(scale1DeviceAddr);}
    if (scale2DeviceAddr != nullptr) { aclrtFree(scale2DeviceAddr);}
    if (biasDeviceAddr != nullptr) {aclrtFree(biasDeviceAddr);}
    if (outDeviceAddr != nullptr) {aclrtFree(outDeviceAddr);}
    if (gaoutDeviceAddr != nullptr) {aclrtFree(gaoutDeviceAddr);}
    if (transScale2DeviceAddr != nullptr) {aclrtFree(transScale2DeviceAddr);}
    if (workspaceSize > 0) { aclrtFree(workspaceAddr);}
    DestoryRes(args);
    return 0;
}

int main(int argc, char *argv[]) {
    int ret;
    GetOption(argc,argv);
    Args args;
    ret = init(args,g_ndev,rankId);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnAllGatherMatmulV2 init failed. ERROR: %d\n", ret); return -1);
    ret = launchOneThread(args);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnAllGatherMatmulV2 launchOneThread  failed. ERROR: %d\n", ret); return -1);
    LOG_PRINT("end launchOneThread re = %d", ret);
    ACL_CHECK(aclFinalize());
    return 0;
}

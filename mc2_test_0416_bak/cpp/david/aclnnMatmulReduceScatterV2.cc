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
#include "aclnnop/aclnn_trans_quant_param_v2.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"
#include "aclnnop/aclnn_matmul_reduce_scatter_v2.h"
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
std::vector<int64_t> g_x1_shape = {};
std::vector<int64_t> g_x2_shape = {};
std::vector<int64_t> g_bias_shape = {};

std::vector<int64_t> g_scale1_shape = {};
std::vector<int64_t> g_scale2_shape = {};
int64_t g_block_size = 128;
int64_t g_block_mode = 0;
bool  g_per_block_flag=false; 

// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::vector<int64_t> g_mmoutput_tensor_shape = {};
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
                    {"mmout_shape",            1, 0, 'p'},
                    {"rankId",                 1, 0, 'q'},
                    {"transpose_a_flag",       1, 0, 'r'},
                    {"amax_dtype",             1, 0, 'x'},
                    {"amax_flag",              1, 0, 'y'},
                    {"save_rankid",            1, 0, 'z'},
                    {"is_hccl_mm",             1, 0, 'A'},
                    {"quant_flag",             1, 0, 'B'},
                    {"block_size",             1, 0, 'C'},
                    {"block_mode",             1, 0, 'D'},
                    {"per_block_flag",         1, 0, 'E'},
                    {"scale1_shape",           1, 0, 'F'},
                    {"scale2_shape",           1, 0, 'G'},
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "C:", longOptions, &optionIndex);
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
                std::vector<std::string> mmout_shape;
                split(optarg, mmout_shape, ',');
                for (uint32_t i = 0; i < mmout_shape.size(); ++i) {
                    g_mmoutput_tensor_shape.push_back(atoi(mmout_shape[i].c_str()));
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
                g_amax_flag = atoi(optarg);
                break;
            case 'z':
                save_rankid = atoi(optarg);
                break;
            case 'A':
                g_is_hccl_mm = atoi(optarg);
                break;
            case 'B':
                g_quant_flag = atoi(optarg);
                break;
            case 'C':
                g_block_size = atoi(optarg);
                break;
            case 'D':
                g_block_mode = atoi(optarg);
                break;
            case 'E':
                g_per_block_flag = stringToBool(optarg);
                break;
            case 'F': {
                std::vector<std::string> shape;
                split(optarg, shape, ',');
                for (uint32_t i = 0; i < shape.size(); ++i) {
                    g_scale1_shape.push_back(atoi(shape[i].c_str()));
                }
            }
                break;
            case 'G': {
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

int launchOneThread(Args &args, aclrtStream &stream) {
    int ret;
    printf("----Use aclnnMatmulReduceScatterV2\n");
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);

    std::vector<int64_t> x1Shape = g_x1_shape;
    std::vector<int64_t> x2Shape = g_x2_shape;
    // transx2Shape用于quantmatmulv5接口中创建x2 tensor时使用
    std::vector<int64_t> transx2Shape;
    transx2Shape.push_back(x2Shape[1]);
    transx2Shape.push_back(x2Shape[0]);
    std::vector<int64_t> biasShape = g_bias_shape;
    std::vector<int64_t> OutShape = g_output_tensor_shape;
    std::vector<int64_t> mmOutShape = g_mmoutput_tensor_shape;
    std::vector<int64_t> scale1_shape = g_scale1_shape;
    std::vector<int64_t> scale2_shape = g_scale2_shape;

    std::vector<int64_t> amaxShape;
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
    void *workspaceAddr = nullptr;
    // hccl_mm out
    void *golden_outDeviceAddr = nullptr;
    void *mmoutDeviceAddr = nullptr;
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
    aclTensor *mmOut = nullptr;
    aclTensor *golden_Out = nullptr;
    uint64_t workspaceSize = 0;
    uint64_t mm_workspaceSize = 0;
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    string scale_dtype = "fp32";

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2", g_transpose_b_flag);
    if (g_transpose_b_flag && g_is_hccl_mm){
        //  转置场景，quantmatmulv5，x2 viewshape[n,k] storage shape:[n,k]，与mc2不同，需要单独创建
        CreateTensor(transx2Shape, aclFormat::ACL_FORMAT_ND, &transx2DeviceAddr, &transx2, x2Dtype, g_bin_path, args.rankId, "x2");
        if(scale2_shape.size() == 2)
        {
            std::vector<int64_t>trans_scale2_shape;
            trans_scale2_shape.push_back(scale2_shape[1]);
            trans_scale2_shape.push_back(scale2_shape[0]);
            CreateTensor(trans_scale2_shape, aclFormat::ACL_FORMAT_ND, &transScale2DeviceAddr, &trans_scale2, scale_dtype, g_bin_path, args.rankId, "scaleInv2");
        }
    }

    if (g_quant_flag){
        CreateTensor(scale1_shape, aclFormat::ACL_FORMAT_ND, &scale1DeviceAddr, &scale1, scale_dtype, g_bin_path, args.rankId, "scaleInv1");
        if(g_transpose_b_flag)
        {
            std::vector<int64_t>trans_scale2_shape;
            trans_scale2_shape.push_back(scale2_shape[1]);
            trans_scale2_shape.push_back(scale2_shape[0]);
            CreateTensor(trans_scale2_shape, aclFormat::ACL_FORMAT_ND, &scale2DeviceAddr, &scale2, scale_dtype, g_bin_path, args.rankId, "scaleInv2");
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
    auto mmOutSize = GetShapeSize(mmOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    // 小算子级联输出
    if (g_is_hccl_mm && g_quant_flag){
        CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &golden_outDeviceAddr, &golden_Out, g_output_tensor_dtype);
        CreateTensorWithoutHostData(mmOutShape, aclFormat::ACL_FORMAT_ND, &mmoutDeviceAddr, &mmOut, g_output_tensor_dtype);       // 与x1一致
        // 小算子级联：
        // matmul
        int64_t mm_group_size = 0;
        if(g_per_block_flag) {mm_group_size = 549764202624;}
        if (g_transpose_b_flag){

            ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, transx2, scale1, trans_scale2, nullptr, nullptr, nullptr, nullptr, bias, 
                                                false, g_transpose_b_flag, mm_group_size, mmOut, &mm_workspaceSize, &mm_executor);
        } else{
            ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, scale1, scale2, nullptr, nullptr, nullptr, nullptr, bias, 
                                                false, false, mm_group_size, mmOut, &mm_workspaceSize, &mm_executor);
        }
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);        
        // 根据第一段接口计算出的workspaceSize申请device内存
        std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
        if (mm_workspaceSize > 0) {
            ret = aclrtMalloc(&mm_workspaceAddr, mm_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
            workspaceAddrPtr.reset(mm_workspaceAddr);
        }
        // 调用aclnnQuantMatmulV5第二段接口
        ret = aclnnQuantMatmulV5(mm_workspaceAddr, mm_workspaceSize, mm_executor, args.stream);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
        // reducescatter
        auto outSize = GetShapeSize(OutShape);
        HcclReduceScatter((void *) mmoutDeviceAddr, golden_outDeviceAddr, outSize, dtype_to_HcclDataType[g_output_tensor_dtype],
                                HcclReduceOp::HCCL_REDUCE_SUM, args.hcclComm, args.stream);
        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(args.stream);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);  
    }

    for (int i = 0; i < g_loop_cnt; i++) {
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        printf("[INFO] aclnnMatmulReduceScatterV2GetWorkspaceSize start\n");
        ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(
            x1, x2, bias, scale1, scale2, nullptr, hcomName, "sum", commTurn, streamMode, g_block_size, g_block_mode, Out, amaxOut, &workspaceSize, &executor);
        printf("[INFO] aclnnMatmulReduceScatterV2GetWorkspaceSize end\n");
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        printf("[INFO] aclnnMatmulReduceScatterV2 start\n");
        ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
        printf("[INFO] aclnnMatmulReduceScatterV2 end\n");
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));
    }
    printf("[INFO] reducescatterv2 workspaceSize = %ld\n", workspaceSize);
    if (g_is_hccl_mm && g_quant_flag){
        //保存结果
        saveOutPut(g_bin_path,OutSize,args.rankId,golden_outDeviceAddr,"output_hccl.bin");
    }

    if (save_rankid == -1 || save_rankid == args.rankId){
        //保存结果
         saveOutPut(g_bin_path,OutSize,args.rankId,outDeviceAddr,"output_npu.bin");
    }

    if (golden_Out != nullptr) {aclDestroyTensor(golden_Out);}
    if (golden_outDeviceAddr != nullptr) {aclrtFree(golden_outDeviceAddr);}
    if (x1 != nullptr) { aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (scale1 != nullptr) {aclDestroyTensor(scale1);}
    if (scale2 != nullptr) {aclDestroyTensor(scale2);}
    if (bias != nullptr) { aclDestroyTensor(bias);}
    if (mmOut != nullptr) {aclDestroyTensor(mmOut);}
    if (Out != nullptr) {aclDestroyTensor(Out);}
    if (trans_scale2 != nullptr) {aclDestroyTensor(trans_scale2);}
    if (x1DeviceAddr != nullptr) {aclrtFree(x1DeviceAddr);}
    if (x2DeviceAddr != nullptr) {aclrtFree(x2DeviceAddr);}
    if (scale1DeviceAddr != nullptr) {aclrtFree(scale1DeviceAddr);}
    if (scale2DeviceAddr != nullptr) {aclrtFree(scale2DeviceAddr);}
    if (biasDeviceAddr != nullptr) {aclrtFree(biasDeviceAddr);}
    if (outDeviceAddr != nullptr) {aclrtFree(outDeviceAddr);}
    if (mmoutDeviceAddr != nullptr) {aclrtFree(mmoutDeviceAddr);}
    if (transScale2DeviceAddr != nullptr) {aclrtFree(transScale2DeviceAddr);}
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
    HcclRootInfo *c = new HcclRootInfo;
    for (int i=0; i<10; i++) {
        c->internal[i] = 'a';
    }
    c->internal[10] = '\0';
    LOG_PRINT("before HcclCommInitRootInfo g_ndev: %d, rankId: %d\n", g_ndev, rankId);
    ret = HcclCommInitRootInfo(g_ndev, c, rankId, &comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitRootInfo failed. ERROR: %d\n", ret); return ret);
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    int re = launchOneThread(args, stream);
    LOG_PRINT("end launchOneThread re = %d", re);
    ACL_CHECK(aclFinalize());
    return 0;
}

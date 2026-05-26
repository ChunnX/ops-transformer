#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <cmath>
#include <map>
#include <unistd.h>
#include <sys/stat.h>
#include <getopt.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_matmul_all_reduce.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"
#include "aclnnop/aclnn_matmul_all_reduce.h"
#include "aclnnop/aclnn_matmul_all_reduce_v2.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce_v2.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce_v3.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce_v4.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"
#include "aclnnop/aclnn_add.h"
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
std::vector<int64_t> g_antiquant_scale_shape = {};
std::vector<int64_t> g_antiquant_offset_shape = {};
std::vector<int64_t> g_dequant_scale_shape = {};
std::vector<int64_t> g_x3_shape = {};
std::vector<int64_t> g_pertoken_shape = {};
std::vector<int64_t> g_commquant_1_shape = {};
std::vector<int64_t> g_commquant_2_shape = {};
std::vector<int64_t> g_mm_dequantscale_shape = {};
std::vector<int64_t> g_mm_pertoken_shape = {};
std::vector<int64_t> g_scale1_shape = {};
std::vector<int64_t> g_scale2_shape = {};
int64_t g_block_size = 0;
int64_t g_block_mode = 0;
bool  g_per_block_flag=false; 

// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
int g_antiquant_offset_flag = 0;
int g_x3_flag = 0;
int g_pertoken_flag = 0;
int g_commquant_flag = 0;
// 低精度场景开关
int g_low_pre_flag = 0;
// 低精度场景执行小算子级联开关
int g_is_hccl_mm = 0;
// 转置开关
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int g_antiquant_group_size = 0;

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
                    {"antiquant_scale_shape",  1, 0, 'e'},
                    {"antiquant_offset_shape", 1, 0, 'f'},
                    {"dequant_scale_shape",    1, 0, 'g'},
                    {"x3_shape",               1, 0, 'h'},
                    {"input_tensor_dtype",     1, 0, 'i'},
                    {"input_tensor_format",    1, 0, 'j'},
                    {"output_shape",           1, 0, 'k'},
                    {"output_dtype",           1, 0, 'l'},
                    {"output_format",          1, 0, 'm'},
                    {"n_dev",                  1, 0, 'n'},
                    {"platform",               1, 0, 'o'},
                    {"bias_flag",              1, 0, 'p'},
                    {"antiquant_offset_flag",  1, 0, 'q'},
                    {"x3_flag",                1, 0, 'r'},
                    {"transpose_b_flag",       1, 0, 's'},
                    {"bin_path",               1, 0, 't'},
                    {"loop_cnt",               1, 0, 'u'},
                    {"antiquant_group_size",   1, 0, 'v'},
                    {"pertoken_flag",          1, 0, 'w'},
                    {"pertoken_shape",         1, 0, 'x'},
                    {"commquant_1_shape",      1, 0, 'y'},
                    {"commquant_2_shape",      1, 0, 'z'},
                    {"commquant_flag",         1, 0, 'A'},
                    {"low_pre_flag",           1, 0, 'B'},
                    {"is_hccl_mm",             1, 0, 'C'},
                    {"rankId",                 1, 0, 'D'},
                    {"mm_dequant_scale_shape", 1, 0, 'E'},
                    {"block_size",             1, 0, 'F'},
                    {"block_mode",             1, 0, 'G'},
                    {"per_block_flag",         1, 0, 'H'},
                    {"mm_pertoken_scale_shape",1, 0, 'I'},
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "y:", longOptions, &optionIndex);
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
                    g_x1_shape.push_back(stoll(x1_shape[i].c_str()));
                }
            }
                break;
            case 'c': {
                std::vector<std::string> x2_shape;
                split(optarg, x2_shape, ',');
                for (uint32_t i = 0; i < x2_shape.size(); ++i) {
                    g_x2_shape.push_back(stoll(x2_shape[i].c_str()));
                }
            }
                break;
            case 'd': {
                std::vector<std::string> bias_shape;
                split(optarg, bias_shape, ',');
                for (uint32_t i = 0; i < bias_shape.size(); ++i) {
                    g_bias_shape.push_back(stoll(bias_shape[i].c_str()));
                }
            }
                break;
            case 'e': {
                std::vector<std::string> antiquant_scale_shape;
                split(optarg, antiquant_scale_shape, ',');
                for (uint32_t i = 0; i < antiquant_scale_shape.size(); ++i) {
                    g_antiquant_scale_shape.push_back(stoll(antiquant_scale_shape[i].c_str()));
                }
            }
                break;
            case 'f': {
                std::vector<std::string> antiquant_offset_shape;
                split(optarg, antiquant_offset_shape, ',');
                for (uint32_t i = 0; i < antiquant_offset_shape.size(); ++i) {
                    g_antiquant_offset_shape.push_back(stoll(antiquant_offset_shape[i].c_str()));
                }
            }
                break;
            case 'g': {
                std::vector<std::string> dequant_scale_shape;
                split(optarg, dequant_scale_shape, ',');
                for (uint32_t i = 0; i < dequant_scale_shape.size(); ++i) {
                    g_dequant_scale_shape.push_back(stoll(dequant_scale_shape[i].c_str()));
                }
            }
                break;
            case 'h': {
                std::vector<std::string> x3_shape;
                split(optarg, x3_shape, ',');
                for (uint32_t i = 0; i < x3_shape.size(); ++i) {
                    g_x3_shape.push_back(stoll(x3_shape[i].c_str()));
                }
            }
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
                    g_output_tensor_shape.push_back(stoll(output_shape[i].c_str()));
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
                g_antiquant_offset_flag = atoi(optarg);
                break;
            case 'r':
                g_x3_flag = atoi(optarg);
                break;
            case 's':
                g_transpose_b_flag = atoi(optarg);
                break;
            case 't':
                g_bin_path = string(optarg);
                break;
            case 'u':
                g_loop_cnt = atoi(optarg);
                break;
            case 'v':
                g_antiquant_group_size = atoi(optarg);
                break;
            case 'w':
                g_pertoken_flag = atoi(optarg);
                break;
            case 'x':{
                std::vector<std::string> pertoken_shape;
                split(optarg, pertoken_shape, ',');
                for (uint32_t i = 0; i < pertoken_shape.size(); ++i) {
                    g_pertoken_shape.push_back(stoll(pertoken_shape[i].c_str()));
                }
            }
                break;   
            case 'y':{
                std::vector<std::string> commquant_1_shape;
                split(optarg, commquant_1_shape, ',');
                for (uint32_t i = 0; i < commquant_1_shape.size(); ++i) {
                    g_commquant_1_shape.push_back(stoll(commquant_1_shape[i].c_str()));
                }
            }
                break;
            case 'z':{
                std::vector<std::string> commquant_2_shape;
                split(optarg, commquant_2_shape, ',');
                for (uint32_t i = 0; i < commquant_2_shape.size(); ++i) {
                    g_commquant_2_shape.push_back(stoll(commquant_2_shape[i].c_str()));
                }
            }
                break;
            case 'A':
                g_commquant_flag = atoi(optarg);
                break;
            case 'B':
                g_low_pre_flag = atoi(optarg);
                break; 
            case 'C':
                g_is_hccl_mm = atoi(optarg);
                break;
            case 'D':
                rankId = atoi(optarg);
                break;
            case 'E': {
                std::vector<std::string> golden_dequant_scale_shape;
                split(optarg, golden_dequant_scale_shape, ',');
                for (uint32_t i = 0; i < golden_dequant_scale_shape.size(); ++i) {
                    g_mm_dequantscale_shape.push_back(stoll(golden_dequant_scale_shape[i].c_str()));
                }
            }
                break;
            case 'F':
                g_block_size = atoi(optarg);
                break;
            case 'G':
                g_block_mode = atoi(optarg);
                break;
            case 'H':
                g_per_block_flag = stringToBool(optarg);
                break;
            case 'I': {
                std::vector<std::string> mm_pertoken_scale_shape;
                split(optarg, mm_pertoken_scale_shape, ',');
                for (uint32_t i = 0; i < mm_pertoken_scale_shape.size(); ++i) {
                    g_mm_pertoken_shape.push_back(stoll(mm_pertoken_scale_shape[i].c_str()));
                }
            }
                break;
            default:
                break;
        }
    }
}

//struct Args {
//    uint32_t rankId;
//    HcclComm hcclComm;
//    aclrtStream stream;
//    aclrtContext context;
//};

int launchOneThread(Args &args, aclrtStream &stream) {
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
    std::vector<int64_t> x3Shape;
    std::vector<int64_t> antiquantScaleShape;
    std::vector<int64_t> antiquantOffsetShape;
    std::vector<int64_t> dequantScaleShape;
    std::vector<int64_t>  pertokenShape;
    std::vector<int64_t>  commquant_1Shape;
    std::vector<int64_t>  commquant_2Shape;
    std::vector<int64_t> reduceOutShape = g_output_tensor_shape;
    std::vector<int64_t> scale1_shape = g_scale1_shape;
    std::vector<int64_t> scale2_shape = g_scale2_shape;
    std::vector<int64_t> mm_dequantScaleShape;
    std::vector<int64_t> mm_pertokenScaleShape = g_mm_pertoken_shape;

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *x3DeviceAddr = nullptr;
    void *pertokenDeviceAddr = nullptr;
    void *commquant_scale_1DeviceAddr = nullptr;
    void *commquant_scale_2DeviceAddr = nullptr;
    void *antiquantScaleDeviceAddr = nullptr;
    void *antiquantOffsetDeviceAddr = nullptr;
    void *dequantScaleDeviceAddr = nullptr;
    void *transDequantScaleDeviceAddr = nullptr;
    void * tanspose_dequantScaleDeviceAddr  = nullptr;
    void *mmdequantScaleDeviceAddr = nullptr;
    void *mmpertokenScaleDeviceAddr = nullptr;

    void *mmtransDequantScaleDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *add_out_DeviceAddr = nullptr;

    // nd->nz
    void *transWorkspaceAddr = nullptr;
    // fp32->uint64
    void *transParamWorkspaceAddr = nullptr;
    void *mmtransParamWorkspaceAddr = nullptr;
    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string x3Dtype = g_output_tensor_dtype;
    std::string pertokenDtype;
    std::string commquant_1_Dtype;
    std::string commquant_2_Dtype;
    std::string antiquantScaleDtype;
    std::string antiquantOffsetDtype;
    std::string dequantScaleDtype;
    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];
    std::string x3Format = g_output_tensor_format;
    std::string antiquantScaleFormat;
    std::string antiquantOffsetFormat;
    std::string dequantScaleFormat;
    std::string pertokenFormat;
    std::string commquant_1_Format;
    std::string commquant_2_Format;
    string per_block_scale_dtype = "fp32";
    bool mx_flag= false;
    

    int n = sizeof(g_input_tensor_dtype) / sizeof(g_input_tensor_dtype[0]);
    int count=0;
    for (int j = 0; j < n; j++) {
        if(g_input_tensor_dtype[j] != ""){
            count+=1;
        }
            // std::cout << g_input_tensor_dtype[j] << std::endl;}
    }
    
    int x3_num = 3;
    if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
        antiquantScaleShape = g_antiquant_scale_shape;
        antiquantOffsetShape = g_antiquant_offset_shape;
        antiquantScaleDtype = g_input_tensor_dtype[3];
        antiquantOffsetDtype = g_input_tensor_dtype[4];
        antiquantScaleFormat = g_input_tensor_format[3];
        antiquantOffsetFormat = g_input_tensor_format[4];
        x3_num = 5;
    }
    if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        dequantScaleShape = g_dequant_scale_shape;
        mm_dequantScaleShape = g_mm_dequantscale_shape;
        dequantScaleDtype = g_input_tensor_dtype[3];
        dequantScaleFormat = g_input_tensor_format[3];
        x3_num = 4;
        if (g_pertoken_flag){
            x3_num = 5;
            pertokenShape = g_pertoken_shape;
            pertokenDtype = g_input_tensor_dtype[x3_num-1];
            pertokenFormat = g_input_tensor_format[x3_num-1];
        }
        if (g_commquant_flag){
            commquant_1Shape = g_commquant_1_shape;
            commquant_2Shape = g_commquant_2_shape;
            commquant_1_Dtype = g_input_tensor_dtype[count-2];
            commquant_1_Format = g_input_tensor_format[count-2];
            commquant_2_Dtype = g_input_tensor_dtype[count-1];
            commquant_1_Format = g_input_tensor_format[count-1];
        } 
    }
    if (g_x3_flag){
        x3Shape = g_x3_shape;
        x3Dtype = g_input_tensor_dtype[x3_num];
        x3Format = g_input_tensor_format[x3_num];
    }

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *antiquantScale = nullptr;
    aclTensor *antiquantOffset = nullptr;
    aclTensor *dequantScale = nullptr;
    aclTensor *mmdequantScale = nullptr;
    aclTensor *mmpertokenScale = nullptr;
    aclTensor *x3 = nullptr;
    aclTensor *pertoken = nullptr;
    aclTensor *commquant_scale_1 = nullptr;
    aclTensor *commquant_scale_2 = nullptr;
    aclTensor *reOut = nullptr;
    aclTensor *transDequantScale = nullptr;
    aclTensor *transpose_dequantScale = nullptr;
    aclTensor *mmtransDequantScale = nullptr;
    aclTensor *add_out = nullptr;

    uint64_t workspaceSize = 0;
    // nd->nz
    uint64_t transWorkspaceSize = 0;
    // fp32->uint64
    uint64_t transParamWorkspaceSize = 0;
    uint64_t mmtransParamWorkspaceSize = 0;
    aclOpExecutor *executor;
    int64_t commTurn = 0;
    int64_t streamMode = 1;

    // transx2Shape用于quantmatmulv5接口中创建x2 tensor时使用
    aclTensor *transx2 = nullptr;
    std::vector<int64_t> transx2Shape;
    transx2Shape.push_back(x2Shape[1]);
    transx2Shape.push_back(x2Shape[0]);
    void *transx2DeviceAddr = nullptr;
    // add算子需要
    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    // golden 输出
    std::vector<int64_t> mmOutShape = g_output_tensor_shape;
    aclTensor *mmOut = nullptr;
    aclTensor *golden_Out = nullptr;
    void *golden_outDeviceAddr = nullptr;
    void *mmoutDeviceAddr = nullptr;
    void *mm_workspaceAddr = nullptr;
    uint64_t mm_workspaceSize = 0;
    aclOpExecutor *add_executor;
    aclOpExecutor *mm_executor;
    void *add_workspaceAddr = nullptr;
    uint64_t add_workspaceSize = 0;

     aclTensor *commbined_mm_x1  = nullptr;
     aclTensor *commbined_mm_out = nullptr;
     void *commbined_mm_x1DeviceAddr = nullptr;
     void *commbined_mm_outDeviceAddr = nullptr;

     if (pertokenDtype == "fp8_e8m0"){mx_flag = true;}
     int64_t mm_group_size = 0;
     if(mx_flag){mm_group_size = 32;}
      if(g_per_block_flag){mm_group_size= 549764202624;}


    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    if (x2Format == "NZ") {
    } else {
        CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2", g_transpose_b_flag);
    }
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    if (g_x3_flag) {
        CreateTensor(x3Shape, aclFormat::ACL_FORMAT_ND, &x3DeviceAddr, &x3, x3Dtype, g_bin_path, args.rankId, "x3");
        CreateTensorWithoutHostData(x3Shape, aclFormat::ACL_FORMAT_ND, &add_out_DeviceAddr, &add_out, g_output_tensor_dtype);
        
    }
    if (g_pertoken_flag){
        CreateTensor(pertokenShape, aclFormat::ACL_FORMAT_ND, &pertokenDeviceAddr, &pertoken, pertokenDtype, g_bin_path, args.rankId, "pertoken_scale");
        if(g_per_block_flag){
            CreateTensor(mm_pertokenScaleShape, aclFormat::ACL_FORMAT_ND, &mmpertokenScaleDeviceAddr,
                        &mmpertokenScale, pertokenDtype, g_bin_path, args.rankId, "pertoken_scale");
        }
    }
    if (g_commquant_flag){
        CreateTensor(commquant_1Shape, aclFormat::ACL_FORMAT_ND, &commquant_scale_1DeviceAddr, &commquant_scale_1, commquant_1_Dtype, g_bin_path, args.rankId, "comm_quant_scale_1");
        CreateTensor(commquant_2Shape, aclFormat::ACL_FORMAT_ND, &commquant_scale_2DeviceAddr, &commquant_scale_2, commquant_2_Dtype, g_bin_path, args.rankId, "comm_quant_scale_2");
    }

    if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
        CreateTensor(antiquantScaleShape, aclFormat::ACL_FORMAT_ND, &antiquantScaleDeviceAddr,
                     &antiquantScale, antiquantScaleDtype, g_bin_path, args.rankId, "antiquant_scale");
        if (g_antiquant_offset_flag) {
            CreateTensor(antiquantOffsetShape, aclFormat::ACL_FORMAT_ND, &antiquantOffsetDeviceAddr,
                         &antiquantOffset, antiquantOffsetDtype, g_bin_path, args.rankId, "antiquant_offset");
        }
        
        vector<int64_t> mm_x1_shape,mm_out_shape;
        mm_out_shape.push_back(g_x1_shape[0]);mm_out_shape.push_back(g_x2_shape[1]);
        if (g_x1_shape.size() == 3)
        {     
             mm_x1_shape.push_back(g_x1_shape[0]*g_x1_shape[1]);mm_x1_shape.push_back(g_x1_shape[2]);
             mm_out_shape.clear();mm_out_shape.shrink_to_fit();
             mm_out_shape.push_back(g_x1_shape[0]*g_x1_shape[1]);mm_out_shape.push_back(g_x2_shape[1]);
             CreateTensor(mm_x1_shape, aclFormat::ACL_FORMAT_ND, &commbined_mm_x1DeviceAddr,&commbined_mm_x1, x1Dtype, g_bin_path, args.rankId, "x1");
        }
        else 
        {
            commbined_mm_x1 = x1;
        }
         CreateTensorWithoutHostData(mm_out_shape, aclFormat::ACL_FORMAT_ND, &commbined_mm_outDeviceAddr, &commbined_mm_out, g_output_tensor_dtype); 
         alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
         CHECK_RET(alpha != nullptr, return ret);
         CreateTensorWithoutHostData(reduceOutShape, aclFormat::ACL_FORMAT_ND, &golden_outDeviceAddr, &golden_Out, g_output_tensor_dtype);

    }
   if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        CreateTensor(dequantScaleShape, aclFormat::ACL_FORMAT_ND, &dequantScaleDeviceAddr,
                     &dequantScale, dequantScaleDtype, g_bin_path, args.rankId, "dequant_scale");
        if ((g_per_block_flag || mx_flag) && g_transpose_b_flag && (dequantScaleShape.size()==2 || dequantScaleShape.size()==3))
        {
            if (mx_flag){
                vector<int64_t> transpos_deqaunt_scale_shape;
                transpos_deqaunt_scale_shape.push_back(dequantScaleShape[1]);
                transpos_deqaunt_scale_shape.push_back(dequantScaleShape[0]);
                transpos_deqaunt_scale_shape.push_back(dequantScaleShape[2]);
                CreateTensor(transpos_deqaunt_scale_shape, aclFormat::ACL_FORMAT_ND, &tanspose_dequantScaleDeviceAddr ,
                            &transpose_dequantScale, dequantScaleDtype, g_bin_path, args.rankId, "dequant_scale");
            }else{
                vector<int64_t> transpos_deqaunt_scale_shape; transpos_deqaunt_scale_shape.push_back(dequantScaleShape[1]);transpos_deqaunt_scale_shape.push_back(dequantScaleShape[0]);
                CreateTensor(transpos_deqaunt_scale_shape, aclFormat::ACL_FORMAT_ND, &tanspose_dequantScaleDeviceAddr ,
                            &transpose_dequantScale, dequantScaleDtype, g_bin_path, args.rankId, "dequant_scale");
            }
        }
        
        if(g_is_hccl_mm and !g_per_block_flag and !mx_flag) {
            // quantmatmulv5 perchannel场景，dequantscale shape仅支持[n]，单独创建tensor
            CreateTensor(mm_dequantScaleShape, aclFormat::ACL_FORMAT_ND, &mmdequantScaleDeviceAddr,
                        &mmdequantScale, dequantScaleDtype, g_bin_path, args.rankId, "dequant_scale");
            // 非pertoken场景，输入为fp8/hif8情况下，dequantscale转uint64
        }
        if (!g_per_block_flag && !mx_flag){
        // 非perblock且非mx场景，dequantscale要求为uint64时，做fp32-->uint64转换
            if (g_output_tensor_dtype == "fp16" || g_low_pre_flag) {
                // 输入为低精度，且非perblock，非mxfp场景；或者输入为int8，且输出是fp16场景；需要做fp32-->uint64转换
                std::string transDequantScaleDtype = "uint64";
                CreateTensorWithoutHostData(dequantScaleShape, aclFormat::ACL_FORMAT_ND, &transDequantScaleDeviceAddr,
                                            &transDequantScale, transDequantScaleDtype);
                ret = aclnnTransQuantParamV2GetWorkspaceSize(dequantScale, nullptr, transDequantScale, &transParamWorkspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                // 根据第一段接口计算出的workspaceSize申请device内存
                if (transParamWorkspaceSize > 0) {
                    ret = aclrtMalloc(&transParamWorkspaceAddr, transParamWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate transParamWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                }
                // 调用aclnnTransQuantParamV2第二段接口
                ret = aclnnTransQuantParamV2(transParamWorkspaceAddr, transParamWorkspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
                ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 200000000));
                if(g_is_hccl_mm){
                    // quantmatmulv5 用到的dequantscale，转uint64
                    CreateTensorWithoutHostData(mm_dequantScaleShape, aclFormat::ACL_FORMAT_ND, &mmtransDequantScaleDeviceAddr,
                                                &mmtransDequantScale, transDequantScaleDtype);
                    ret = aclnnTransQuantParamV2GetWorkspaceSize(mmdequantScale, nullptr, mmtransDequantScale, &mmtransParamWorkspaceSize, &executor);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                    // 根据第一段接口计算出的workspaceSize申请device内存
                    if (mmtransParamWorkspaceSize > 0) {
                        ret = aclrtMalloc(&mmtransParamWorkspaceAddr, mmtransParamWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate transParamWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                    }
                    // 调用aclnnTransQuantParamV2第二段接口
                    ret = aclnnTransQuantParamV2(mmtransParamWorkspaceAddr, mmtransParamWorkspaceSize, executor, args.stream);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
                    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 200000000));
                }
            }
        }

    }
    // 创建output
    auto reduceOutSize = GetShapeSize(reduceOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(reduceOutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &reOut, g_output_tensor_dtype);


    // a8w8 低精度场景，且执行小算子级联时，创建所需tensor
    if(g_low_pre_flag && g_is_hccl_mm && "A8W8" == g_is_quant){
        // 转置场景matmul输入
        if (g_transpose_b_flag){
            CreateTensor(transx2Shape, aclFormat::ACL_FORMAT_ND, &transx2DeviceAddr, &transx2, x2Dtype, g_bin_path, args.rankId, "x2");
        }
        // 小算子级联输出
        CreateTensorWithoutHostData(reduceOutShape, aclFormat::ACL_FORMAT_ND, &golden_outDeviceAddr, &golden_Out, g_output_tensor_dtype);
        auto mmOutSize = GetShapeSize(mmOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
        CreateTensorWithoutHostData(mmOutShape, aclFormat::ACL_FORMAT_ND, &mmoutDeviceAddr, &mmOut, g_output_tensor_dtype); 
        alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
        CHECK_RET(alpha != nullptr, return ret);
    }

    for (int i = 0; i < g_loop_cnt; i++) {

        if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
            // 调用第一段接口
            if(g_is_hccl_mm ) //小算子级联
            {
               ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(commbined_mm_x1,x2,antiquantScale,antiquantOffset,
                            NULL,NULL,bias,g_antiquant_group_size,commbined_mm_out,&workspaceSize, &executor);
               if (workspaceSize > 0 && i==0) {
                    printf("Only malloc workspacesize once.......\n");
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
               ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
               ret = aclrtSynchronizeStream(args.stream);
               CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
               if(g_x3_flag)
               {  
                    ret = aclnnAddGetWorkspaceSize(commbined_mm_out, x3, alpha, add_out,&add_workspaceSize, &add_executor);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                    // 根据第一段接口计算出的workspaceSize申请device内存
                    if (add_workspaceSize > 0 && i == 0) {
                        ret = aclrtMalloc(&add_workspaceAddr, add_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
                    }
                    // 调用aclnnInplaceAdd第二段接口
                    ret = aclnnAdd(add_workspaceAddr, add_workspaceSize,add_executor, args.stream);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdd failed. ERROR: %d\n", ret); return ret);     
               }
                void *tmp_addr = nullptr;
                g_x3_flag?tmp_addr=add_out_DeviceAddr:tmp_addr=commbined_mm_outDeviceAddr;
                auto outSize = GetShapeSize(reduceOutShape);
                HcclAllReduce((void *)tmp_addr , golden_outDeviceAddr, outSize,
                dtype_to_HcclDataType[g_output_tensor_dtype], HcclReduceOp::HCCL_REDUCE_SUM, args.hcclComm, args.stream);
                ret = aclrtSynchronizeStream(args.stream);
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);  
            }

            ret = aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(
                    x1, x2, bias, antiquantScale, antiquantOffset, x3, hcomName, "sum", commTurn,
                    streamMode, g_antiquant_group_size, reOut, &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnWeightQuantMatmulAllReduceGetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnWeightQuantMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduce failed. ret = %d \n", ret);return ret);
        } else if ("A8W8" == g_is_quant && !g_pertoken_flag && !g_commquant_flag && !g_low_pre_flag) {
            // 非低精度场景
            // 调用第一段接口
            if (g_output_tensor_dtype == "fp16") {
                ret = aclnnQuantMatmulAllReduceGetWorkspaceSize(
                        x1, x2, bias, x3, transDequantScale, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            }  else {
                ret = aclnnQuantMatmulAllReduceGetWorkspaceSize(
                        x1, x2, bias, x3, dequantScale, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            }
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceGetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnQuantMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduce failed. ret = %d \n", ret); return ret);
            
        }else if ("A8W8" == g_is_quant && g_pertoken_flag && !g_commquant_flag && !g_low_pre_flag){

            ret = aclnnQuantMatmulAllReduceV2GetWorkspaceSize(
                    x1, x2, bias, x3, dequantScale, pertoken, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnQuantMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV2 failed. ret = %d \n", ret); return ret);
            
        }else if ("A8W8" == g_is_quant && !g_pertoken_flag && !g_commquant_flag && g_low_pre_flag) {
            // 低精度场景
            // 小算子级联 quantmatmul

            if (g_is_hccl_mm){ //小算子级联执行
                if (g_transpose_b_flag){
                    ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, transx2, nullptr, mmtransDequantScale, nullptr, nullptr, nullptr, nullptr, bias, 
                                                        false, g_transpose_b_flag, mm_group_size, mmOut, &mm_workspaceSize, &mm_executor);
                } else{
                    ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, nullptr, mmtransDequantScale, nullptr, nullptr, nullptr, nullptr, bias, 
                                                        false, false, mm_group_size, mmOut, &mm_workspaceSize, &mm_executor);
                }
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);        
                // 根据第一段接口计算出的workspaceSize申请device内存
                std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
                if (mm_workspaceSize > 0 && i ==0) 
                {
                    ret = aclrtMalloc(&mm_workspaceAddr, mm_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
                    workspaceAddrPtr.reset(mm_workspaceAddr);
                }
                // 调用aclnnQuantMatmulV5第二段接口
                ret = aclnnQuantMatmulV5(mm_workspaceAddr, mm_workspaceSize, mm_executor, args.stream);
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
                // 小算子级联 add
                // 调用aclnnInplaceAdd第一段接口 --增加is_x3判断
                if (g_x3_flag){
                    ret = aclnnInplaceAddGetWorkspaceSize(mmOut, x3, alpha, &add_workspaceSize, &add_executor);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                    // 根据第一段接口计算出的workspaceSize申请device内存
                    if (add_workspaceSize > 0 && i ==0) {
                        ret = aclrtMalloc(&add_workspaceAddr, add_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
                    }
                    // 调用aclnnInplaceAdd第二段接口
                    ret = aclnnInplaceAdd(add_workspaceAddr, add_workspaceSize, add_executor, args.stream);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdd failed. ERROR: %d\n", ret); return ret);
                }
                // 小算子级联 allreduce
                auto outSize = GetShapeSize(reduceOutShape);
                HcclAllReduce((void *) mmoutDeviceAddr, golden_outDeviceAddr, outSize, dtype_to_HcclDataType[g_output_tensor_dtype],
                            HcclReduceOp::HCCL_REDUCE_SUM, args.hcclComm, args.stream);
                // 4. （固定写法）同步等待任务执行结束
                ret = aclrtSynchronizeStream(args.stream);
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);   
            }
            //quantmatmullallreduce 调用
            //x1, x2, biasOptional,  x3Optional,  x1Scale,   x2Scale, commQuantScale1Optional, commQuantScale2Optional,  group, reduceOp, icommTurn, streamMode,  blockMode, blockSize, output);
            ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken, transDequantScale,
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,mm_group_size,0, reOut, &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceGetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnQuantMatmulAllReduceV4(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduce failed. ret = %d \n", ret); return ret);
        } else if ("A8W8" == g_is_quant && g_pertoken_flag && !g_commquant_flag && g_low_pre_flag){
            // 低精度场景
            // 小算子级联 quantmatmul --增加入参控制是否要执行小算子级联
            if (g_is_hccl_mm)
            {   
                aclTensor *tmp_pertoken = nullptr;
                // perblock场景，matmul pertokenscale需要传3维，非perblock场景，可以与mc2复用同一个
                g_per_block_flag?tmp_pertoken=mmpertokenScale:tmp_pertoken=pertoken;
                aclTensor *tmp_x2 = nullptr;
                // 转置场景,matmul x2无法与mc2复用
                g_transpose_b_flag?tmp_x2=transx2:tmp_x2=x2;
                if(g_per_block_flag || mx_flag){
                    // 转置场景，x2scale需要转置
                    aclTensor *tmp_perblock_mx_dequant = nullptr;
                    g_transpose_b_flag?tmp_perblock_mx_dequant=transpose_dequantScale:tmp_perblock_mx_dequant=dequantScale;
                    ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, tmp_x2, tmp_pertoken, tmp_perblock_mx_dequant, nullptr, nullptr, nullptr, nullptr, bias, 
                                                        false, g_transpose_b_flag, mm_group_size, mmOut, &mm_workspaceSize, &mm_executor);
                }
                else{
                    ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, tmp_x2, tmp_pertoken, mmdequantScale, nullptr, nullptr, nullptr, nullptr, bias, 
                                                        false, g_transpose_b_flag, mm_group_size, mmOut, &mm_workspaceSize, &mm_executor);
                }
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);      
                // 根据第一段接口计算出的workspaceSize申请device内存
                std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
                if (mm_workspaceSize > 0 && i == 0) {
                    ret = aclrtMalloc(&mm_workspaceAddr, mm_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
                    workspaceAddrPtr.reset(mm_workspaceAddr);
                }
                // 调用aclnnQuantMatmulV5第二段接口
                ret = aclnnQuantMatmulV5(mm_workspaceAddr, mm_workspaceSize, mm_executor, args.stream);
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
                // 小算子级联 add
                // 调用aclnnInplaceAdd第一段接口 
                if (g_x3_flag) { // 带x3场景，调用add算子
                    ret = aclnnInplaceAddGetWorkspaceSize(mmOut, x3, alpha, &add_workspaceSize, &add_executor);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
                    // 根据第一段接口计算出的workspaceSize申请device内存
                    if (add_workspaceSize > 0 && i == 0) {
                        ret = aclrtMalloc(&add_workspaceAddr, add_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
                    }
                    // 调用aclnnInplaceAdd第二段接口
                    ret = aclnnInplaceAdd(add_workspaceAddr, add_workspaceSize, add_executor, args.stream);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdd failed. ERROR: %d\n", ret); return ret);
                }
                // 小算子级联 allreduce
                auto outSize = GetShapeSize(reduceOutShape);
                HcclAllReduce((void *) mmoutDeviceAddr, golden_outDeviceAddr, outSize, dtype_to_HcclDataType[g_output_tensor_dtype],
                            HcclReduceOp::HCCL_REDUCE_SUM, args.hcclComm, args.stream);
                // 4. （固定写法）同步等待任务执行结束
                ret = aclrtSynchronizeStream(args.stream);
                CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret); 
            }
            // quantmatmulallreduce调用
            if(g_per_block_flag || mx_flag)
            {
                if(g_transpose_b_flag)
                {
                   ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken,transpose_dequantScale ,
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,mm_group_size,0, reOut, &workspaceSize, &executor);
                        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);     
                }
                else
                {
                   ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken,dequantScale ,
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,mm_group_size,0, reOut, &workspaceSize, &executor);
                        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);  
                }
            }
            else
            {
                ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, pertoken, dequantScale,
                         nullptr,nullptr,hcomName,"sum",commTurn, streamMode,mm_group_size, 0,reOut, &workspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            }
           
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnQuantMatmulAllReduceV4(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV2 failed. ret = %d \n", ret); return ret);  

        } else if("A8W8" == g_is_quant && g_commquant_flag){ // a8w8 int8通信场景
            if (g_pertoken_flag){
                printf("*******pertoken = 1 *******\n");
                ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, x2, bias, x3, dequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            }
            else {
                if (g_output_tensor_dtype == "fp16") {
                    printf("*******pertoken = 0, output = fp16  *******\n");
                    ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, x2, bias, x3, transDequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                } else {
                    printf("*******pertoken = 0, output = bf16  *******\n");
                    ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, x2, bias, x3, dequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                }
            }
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnQuantMatmulAllReduceV3(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV3 failed. ret = %d \n", ret); return ret);
        }
        else {
            // 新增非量化_x3接口----2024/05/10
            if (g_x3_flag) {
                // 调用第一段接口--非量化_带x3
                ret = aclnnMatmulAllReduceV2GetWorkspaceSize(
                        x1, x2, bias, x3, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduceV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
                if (workspaceSize > 0 && i==0) {
                    printf("Only malloc workspacesize once.......\n");
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                ret = aclnnMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduceV2 failed. ret = %d \n", ret); return ret);
            } else {
                // 调用第一段接口--非量化_不带x3 
                // 推荐调用
                ret = aclnnMatmulAllReduceGetWorkspaceSize(x1, x2, bias, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduceGetWorkspaceSize failed. ret = %d \n", ret); return ret);
                if (workspaceSize > 0 && i==0) {
                    printf("Only malloc workspacesize once.......\n");
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                ret = aclnnMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduce failed. ret = %d \n", ret); return ret);
            }
        }
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 2000000000));
    }
    printf("[INFO] mm_allreduce workspaceSize = %lu\n", workspaceSize);
    //保存结果
    unsigned long long compareSize = reduceOutSize;
    if (compareSize == 0){
        compareSize = 1;
    } 
    void *hostAicpuBuff = nullptr;
    // ACL_CHECK(aclrtMallocHost((void **) &hostAicpuBuff, compareSize));
    hostAicpuBuff = malloc(compareSize);
    ACL_CHECK(aclrtMemcpy((void *) hostAicpuBuff, compareSize, (void *) outDeviceAddr, compareSize,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    writeBin(g_bin_path + "/output/" + to_string(args.rankId) + "/" + "output_npu.bin", hostAicpuBuff, compareSize);
    if (hostAicpuBuff != nullptr) {
        // ACL_CHECK(aclrtFreeHost(hostAicpuBuff));
        free(hostAicpuBuff);
    }
    // 低精度场景且执行小算子级联模式场景保存golden数据
    if ((g_low_pre_flag && g_is_hccl_mm)){
        unsigned long long golden_compareSize = reduceOutSize;
        void *golden_hostAicpuBuff = nullptr;
        golden_hostAicpuBuff = malloc(golden_compareSize);
        ACL_CHECK(aclrtMemcpy((void *) golden_hostAicpuBuff, golden_compareSize, (void *) golden_outDeviceAddr, golden_compareSize,
                            ACL_MEMCPY_DEVICE_TO_HOST));
        writeBin(g_bin_path + "/output/" + to_string(args.rankId) + "/" + "output_hccl.bin", golden_hostAicpuBuff, golden_compareSize);
        if (golden_hostAicpuBuff != nullptr) {
            free(golden_hostAicpuBuff);
        }
    }

     // 资源释放
    if (mmOut != nullptr) {aclDestroyTensor(mmOut);}
    if (golden_Out != nullptr) {aclDestroyTensor(golden_Out);}
    if (alpha != nullptr){aclDestroyScalar(alpha);}
    if (golden_outDeviceAddr != nullptr) {aclrtFree(golden_outDeviceAddr);}
    if (mmoutDeviceAddr != nullptr) {aclrtFree(mmoutDeviceAddr);}
    if (mm_workspaceSize > 0) {aclrtFree(mm_workspaceAddr);}
    if (add_workspaceSize > 0) {aclrtFree(add_workspaceAddr);}
    if (x1 != nullptr) {aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (bias != nullptr) {aclDestroyTensor(bias);}
    if (x3 != nullptr) {aclDestroyTensor(x3);}
    if(commbined_mm_x1 != nullptr){aclDestroyTensor(commbined_mm_x1);}
    if(commbined_mm_out != nullptr){aclDestroyTensor(commbined_mm_out);}
    if (antiquantScale != nullptr) {aclDestroyTensor(antiquantScale);}
    if (antiquantOffset != nullptr) { aclDestroyTensor(antiquantOffset);}
    if (dequantScale != nullptr) {aclDestroyTensor(dequantScale);}
    if (transDequantScale != nullptr) {aclDestroyTensor(transDequantScale);}
    if (transpose_dequantScale != nullptr) {aclDestroyTensor(transpose_dequantScale);}
    if (add_out != nullptr) {aclDestroyTensor(add_out);}

    if (reOut != nullptr) {aclDestroyTensor(reOut);}
    if (x1DeviceAddr != nullptr) {aclrtFree(x1DeviceAddr);}
    if (x2DeviceAddr != nullptr) {aclrtFree(x2DeviceAddr);}
    if (biasDeviceAddr != nullptr) {aclrtFree(biasDeviceAddr);}
    if (x3DeviceAddr != nullptr) {aclrtFree(x3DeviceAddr);}
    if (antiquantScaleDeviceAddr != nullptr) {aclrtFree(antiquantScaleDeviceAddr);}
    if (antiquantOffsetDeviceAddr != nullptr) {aclrtFree(antiquantOffsetDeviceAddr);}
    if (dequantScaleDeviceAddr != nullptr) {aclrtFree(dequantScaleDeviceAddr);}
    if (outDeviceAddr != nullptr) {aclrtFree(outDeviceAddr);}
    if (commbined_mm_x1DeviceAddr != nullptr) {aclrtFree(commbined_mm_x1DeviceAddr);}
    if (commbined_mm_outDeviceAddr != nullptr) {aclrtFree(commbined_mm_outDeviceAddr);};
    if (tanspose_dequantScaleDeviceAddr != nullptr) {aclrtFree(tanspose_dequantScaleDeviceAddr);}
    if (mmdequantScaleDeviceAddr != nullptr) {aclrtFree(mmdequantScaleDeviceAddr);}
    if (mmtransParamWorkspaceSize >0){aclrtFree(mmtransParamWorkspaceAddr);}
    if (add_out_DeviceAddr != nullptr){aclrtFree(add_out_DeviceAddr);}

    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
    // nd->nz
    if (transWorkspaceSize > 0) {aclrtFree(transWorkspaceAddr);}
    //fp32->uint64
    if (transParamWorkspaceSize > 0) {aclrtFree(transParamWorkspaceAddr);}
    ACL_CHECK(aclrtDestroyStreamForce(args.stream));
    ACL_CHECK(HcclCommDestroy(args.hcclComm));
    ACL_CHECK(aclrtDestroyContext(args.context));
    ACL_CHECK(aclrtResetDevice(args.rankId));
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
    // LOG_PRINT("before HcclRootInfo");
    // HcclRootInfo *c = new HcclRootInfo;
    // for (int i=0; i<10; i++) {
    //     c->internal[i] = 'a';
    // }
    // c->internal[10] = '\0';
    // LOG_PRINT("before HcclCommInitRootInfo g_ndev: %d, rankId: %d\n", g_ndev, rankId);
    // ret = HcclCommInitRootInfo(g_ndev, c, rankId, &comms);
    // LOG_PRINT("after HcclCommInitRootInfo\n");
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    
    config.hcclDeterministic = 1;
    int g_hcclBufferSize = 200;
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
    LOG_PRINT("before launchOneThread");
    int re = launchOneThread(args, stream);
    LOG_PRINT("end launchOneThread re = %d", re);
    ACL_CHECK(aclFinalize());
    return 0;
}




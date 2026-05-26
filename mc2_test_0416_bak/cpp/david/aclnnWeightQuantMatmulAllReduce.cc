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
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"
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
                    {"x3_shape",               1, 0, 'g'},
                    {"input_tensor_dtype",     1, 0, 'h'},
                    {"input_tensor_format",    1, 0, 'i'},
                    {"output_shape",           1, 0, 'j'},
                    {"output_dtype",           1, 0, 'k'},
                    {"output_format",          1, 0, 'l'},
                    {"n_dev",                  1, 0, 'm'},
                    {"platform",               1, 0, 'n'},
                    {"bias_flag",              1, 0, 'o'},
                    {"antiquant_offset_flag",  1, 0, 'p'},
                    {"x3_flag",                1, 0, 'q'},
                    {"transpose_b_flag",       1, 0, 'r'},
                    {"bin_path",               1, 0, 's'},
                    {"loop_cnt",               1, 0, 't'},
                    {"antiquant_group_size",   1, 0, 'u'},
                    {"low_pre_flag",           1, 0, 'v'},
                    {"is_hccl_mm",             1, 0, 'w'},
                    {"rankId",                 1, 0, 'x'},
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
                std::vector<std::string> x3_shape;
                split(optarg, x3_shape, ',');
                for (uint32_t i = 0; i < x3_shape.size(); ++i) {
                    g_x3_shape.push_back(stoll(x3_shape[i].c_str()));
                }
            }
                break;
            case 'h': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            }
                break;
            case 'i': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];

                }
            }
                break;
            case 'j': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(stoll(output_shape[i].c_str()));
                }
            }
                break;
            case 'k':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'l':
                g_output_tensor_format = string(optarg);
                break;
            case 'm':
                g_ndev = atoi(optarg);
                break;
            case 'n':
                g_platform = string(optarg);
                break;
            case 'o':
                g_bias_flag = atoi(optarg);
                break;
            case 'p':
                g_antiquant_offset_flag = atoi(optarg);
                break;
            case 'q':
                g_x3_flag = atoi(optarg);
                break;
            case 'r':
                g_transpose_b_flag = atoi(optarg);
                break;
            case 's':
                g_bin_path = string(optarg);
                break;
            case 't':
                g_loop_cnt = atoi(optarg);
                break;
            case 'u':
                g_antiquant_group_size = atoi(optarg);
                break;
            case 'v':
                g_low_pre_flag = atoi(optarg);
                break;
            case 'w':
                g_is_hccl_mm = atoi(optarg);
                break;
            case 'x':
                rankId = atoi(optarg);
                break;
            default:
                break;
        }
    }
}

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
    std::vector<int64_t> reduceOutShape = g_output_tensor_shape;

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *x3DeviceAddr = nullptr;
    void *antiquantScaleDeviceAddr = nullptr;
    void *antiquantOffsetDeviceAddr = nullptr;

    void *outDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *add_out_DeviceAddr = nullptr;
    void *mm_x3_DeviceAddr = nullptr;

    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string x3Dtype = g_output_tensor_dtype;
    std::string antiquantScaleDtype;
    std::string antiquantOffsetDtype;
    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];
    std::string x3Format = g_output_tensor_format;
    std::string antiquantScaleFormat;
    std::string antiquantOffsetFormat;

    int n = sizeof(g_input_tensor_dtype) / sizeof(g_input_tensor_dtype[0]);
    int count=0;
    for (int j = 0; j < n; j++) {
        if(g_input_tensor_dtype[j] != ""){
            count+=1;
        }
            // std::cout << g_input_tensor_dtype[j] << std::endl;}
    }

    int x3_num = 5;
    antiquantScaleShape = g_antiquant_scale_shape;
    antiquantOffsetShape = g_antiquant_offset_shape;
    antiquantScaleDtype = g_input_tensor_dtype[3];
    antiquantOffsetDtype = g_input_tensor_dtype[4];
    antiquantScaleFormat = g_input_tensor_format[3];
    antiquantOffsetFormat = g_input_tensor_format[4];

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
    aclTensor *x3 = nullptr;
    aclTensor *reOut = nullptr;
    aclTensor *add_out = nullptr;
    aclTensor *mm_x3 = nullptr;

    uint64_t workspaceSize = 0;
    // nd->nz
    aclOpExecutor *executor;
    int64_t commTurn = 0;
    int64_t streamMode = 1;

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
    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    if (x2Format == "NZ") {
    } else {
        CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2", g_transpose_b_flag);
    }
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }

    CreateTensor(antiquantScaleShape, aclFormat::ACL_FORMAT_ND, &antiquantScaleDeviceAddr,
                 &antiquantScale, antiquantScaleDtype, g_bin_path, args.rankId, "antiquant_scale", g_transpose_b_flag);
    if (g_antiquant_offset_flag) {
        CreateTensor(antiquantOffsetShape, aclFormat::ACL_FORMAT_ND, &antiquantOffsetDeviceAddr,
                     &antiquantOffset, antiquantOffsetDtype, g_bin_path, args.rankId, "antiquant_offset", g_transpose_b_flag);
    }
    vector<int64_t> mm_x1_shape,mm_out_shape;
    mm_out_shape.push_back(g_x1_shape[0]);mm_out_shape.push_back(g_x2_shape[1]);
    std::vector<int64_t> dimsData = {0, 1, 2};
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
    CreateTensor(mm_out_shape, aclFormat::ACL_FORMAT_ND, &mm_x3_DeviceAddr, &mm_x3, x3Dtype, g_bin_path, args.rankId, "x3");
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return ret);
    CreateTensorWithoutHostData(reduceOutShape, aclFormat::ACL_FORMAT_ND, &golden_outDeviceAddr, &golden_Out, g_output_tensor_dtype);
    if (g_x3_flag) {
        CreateTensor(x3Shape, aclFormat::ACL_FORMAT_ND, &x3DeviceAddr, &x3, x3Dtype, g_bin_path, args.rankId, "x3");
        if (g_is_hccl_mm){
             CreateTensorWithoutHostData(mm_out_shape, aclFormat::ACL_FORMAT_ND, &add_out_DeviceAddr, &add_out, g_output_tensor_dtype);
        }
    }

    // 创建output
    auto reduceOutSize = GetShapeSize(reduceOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(reduceOutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &reOut, g_output_tensor_dtype);

    for (int i = 0; i < g_loop_cnt; i++) {
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
                ret = aclnnAddGetWorkspaceSize(commbined_mm_out, mm_x3, alpha, add_out,&add_workspaceSize, &add_executor);
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

        // 调用第一段接口
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
    if ((g_is_hccl_mm)){
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
    if (x1 != nullptr) {aclDestroyTensor(x1);}
    if (x2 != nullptr) {aclDestroyTensor(x2);}
    if (bias != nullptr) {aclDestroyTensor(bias);}
    if (x3 != nullptr) {aclDestroyTensor(x3);}
    if (antiquantScale != nullptr) {aclDestroyTensor(antiquantScale);}
    if (antiquantOffset != nullptr) { aclDestroyTensor(antiquantOffset);}

    if (reOut != nullptr) {aclDestroyTensor(reOut);}
    if (x1DeviceAddr != nullptr) {aclrtFree(x1DeviceAddr);}
    if (x2DeviceAddr != nullptr) {aclrtFree(x2DeviceAddr);}
    if (biasDeviceAddr != nullptr) {aclrtFree(biasDeviceAddr);}
    if (x3DeviceAddr != nullptr) {aclrtFree(x3DeviceAddr);}
    if (antiquantScaleDeviceAddr != nullptr) {aclrtFree(antiquantScaleDeviceAddr);}
    if (antiquantOffsetDeviceAddr != nullptr) {aclrtFree(antiquantOffsetDeviceAddr);}
    if (outDeviceAddr != nullptr) {aclrtFree(outDeviceAddr);}

    if (workspaceSize > 0) {aclrtFree(workspaceAddr);}
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




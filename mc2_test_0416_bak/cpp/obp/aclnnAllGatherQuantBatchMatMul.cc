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
#include "aclnnop/aclnn_all_gather_quant_batch_matmul.h"
#include"common.h"

using namespace std;

// 定义全局变量
std::string g_bin_path = "./";
// 输入
std::string g_input_tensor_dtype[128] = {};
std::string g_input_tensor_format[128] = {};
std::vector<int64_t> g_x1_shape = {};
std::vector<int64_t> g_x2_shape = {};
std::vector<int64_t> g_bias_shape = {};
std::vector<int64_t> g_pertoken_shape = {};
std::vector<int64_t> g_dequant_scale_shape = {};
std::vector<int64_t> g_gaperput_tensor_shape = {};
// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::vector<int64_t> g_gaoutput_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
int g_pertoken_flag = 0;
// 转置开关
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int g_gather_axis = 0;
int g_gather_output = 0;

void GetOption(int argc, char **argv)
{
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] = {{"x1_shape", 1, 0, 'a'},
            {"x2_shape", 1, 0, 'b'},
            {"bias_shape", 1, 0, 'c'},
            {"pertoken_shape", 1, 0, 'd'},
            {"dequant_scale_shape", 1, 0, 'e'},
            {"input_tensor_dtype", 1, 0, 'f'},
            {"input_tensor_format", 1, 0, 'g'},
            {"output_shape", 1, 0, 'h'},
            {"output_dtype", 1, 0, 'i'},
            {"output_format", 1, 0, 'j'},
            {"n_dev", 1, 0, 'k'},
            {"bias_flag", 1, 0, 'l'},
            {"pertoken_flag", 1, 0, 'm'},
            {"transpose_b_flag", 1, 0, 'n'},
            {"bin_path", 1, 0, 'o'},
            {"loop_cnt", 1, 0, 'p'},
            {"gather_axis", 1, 0, 'q'},
            {"gatherout_shape", 1, 0, 'r'},
            {"gather_perout_shape", 1, 0, 's'},
            {"gather_out_flag", 1, 0, 't'},
            {0, 0, 0, 0}};

        int c = getopt_long(argc, argv, "r:", longOptions, &optionIndex);
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
            } break;
            case 'b': {
                std::vector<std::string> x2_shape;
                split(optarg, x2_shape, ',');
                for (uint32_t i = 0; i < x2_shape.size(); ++i) {
                    g_x2_shape.push_back(atoi(x2_shape[i].c_str()));
                }
            } break;
            case 'c': {
                std::vector<std::string> bias_shape;
                split(optarg, bias_shape, ',');
                for (uint32_t i = 0; i < bias_shape.size(); ++i) {
                    g_bias_shape.push_back(atoi(bias_shape[i].c_str()));
                }
            } break;
            case 'd': {
                std::vector<std::string> pertoken_shape;
                split(optarg, pertoken_shape, ',');
                for (uint32_t i = 0; i < pertoken_shape.size(); ++i) {
                    g_pertoken_shape.push_back(atoi(pertoken_shape[i].c_str()));
                }
            } break;
            case 'e': {
                std::vector<std::string> dequant_scale_shape;
                split(optarg, dequant_scale_shape, ',');
                for (uint32_t i = 0; i < dequant_scale_shape.size(); ++i) {
                    g_dequant_scale_shape.push_back(atoi(dequant_scale_shape[i].c_str()));
                }
            } break;
            case 'f': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            } break;
            case 'g': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];
                }
            } break;
            case 'h': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(atoi(output_shape[i].c_str()));
                }
            } break;
            case 'i':
                g_output_tensor_dtype = string(optarg);
                break;
            case 'j':
                g_output_tensor_format = string(optarg);
                break;
            case 'k':
                g_ndev = atoi(optarg);
                break;
            case 'l':
                g_bias_flag = atoi(optarg);
                break;
            case 'm':
                g_pertoken_flag = atoi(optarg);
                break;
            case 'n':
                g_transpose_b_flag = atoi(optarg);
                break;
            case 'o':
                g_bin_path = string(optarg);
                break;
            case 'p':
                g_loop_cnt = atoi(optarg);
                break;
            case 'q':
                g_gather_axis = atoi(optarg);
                break;
            case 'r': {
                std::vector<std::string> gatherout_shape;
                split(optarg, gatherout_shape, ',');
                for (uint32_t i = 0; i < gatherout_shape.size(); ++i) {
                    g_gaoutput_tensor_shape.push_back(atoi(gatherout_shape[i].c_str()));
                }
            } break;
            case 's': {
                std::vector<std::string> gather_per_shape;
                split(optarg, gather_per_shape, ',');
                for (uint32_t i = 0; i < gather_per_shape.size(); ++i) {
                    g_gaperput_tensor_shape.push_back(atoi(gather_per_shape[i].c_str()));
                }
            } break;
            case 't':
                g_gather_output = atoi(optarg);
                break;
            default:
                break;
        }
    }
}

int CreateTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
    std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name)
{
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    auto tensorSize = size * dtype_to_sizeof(dtype);
    if (dtype == "int4") {
        tensorSize = static_cast<uint64_t>(ceil(tensorSize / 2));
    }
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    void *tensorHostData = nullptr;
    ACL_CHECK(aclrtMallocHost((void **)&tensorHostData, tensorSize));
    if (dtype == "int4") {
        ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + "packed_" + name + ".bin",
            tensorSize,
            tensorHostData,
            tensorSize);
    } else {
        ReadFile(
            bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin", tensorSize, tensorHostData, tensorSize);
    }
    ret = aclrtMemcpy(*deviceAddr, tensorSize, tensorHostData, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] host -> device memory copy failed \n"); return ret);
    ACL_CHECK(aclrtFreeHost(tensorHostData));
    // 调用aclCreateTensor创建tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));
    int offset = 0;
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        g_data_type_map[dtype],
        strides.data(),
        offset,
        format,
        storageShape.data(),
        storageShape.size(),
        *deviceAddr);
    return 0;
}

int CreateWeightTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
    std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name)
{
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, g_data_type_map[dtype], &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    auto tensorSize = size * dtype_to_sizeof(dtype);
    if (dtype == "int4") {
        tensorSize = static_cast<uint64_t>(ceil(tensorSize / 2));
    }
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    void *tensorHostData = nullptr;
    ACL_CHECK(aclrtMallocHost((void **)&tensorHostData, tensorSize));
    if (dtype == "int4") {
        ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + "packed_" + name + ".bin",
            tensorSize,
            tensorHostData,
            tensorSize);
    } else {
        ReadFile(
            bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin", tensorSize, tensorHostData, tensorSize);
    }
    ret = aclrtMemcpy(*deviceAddr, tensorSize, tensorHostData, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] host -> device memory copy failed \n"); return ret);
    ACL_CHECK(aclrtFreeHost(tensorHostData));
    // 调用aclCreateTensor创建tensor
    std::vector<int64_t> strides(shape.size(), 1);
 
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    int offset = 0;
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        g_data_type_map[dtype],
        strides.data(),
        offset,
        format,
        storageShape.data(),
        storageShape.size(),
        *deviceAddr);
    return 0;
}

int launchOneThread(Args &args)
{
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
    std::vector<int64_t> biasShape = g_bias_shape;
    std::vector<int64_t> OutShape = g_output_tensor_shape;
    std::vector<int64_t> gaOutShape = g_gaoutput_tensor_shape;
    std::vector<int64_t> gaPerOutShape = g_gaperput_tensor_shape;
    std::vector<int64_t> ScaleShape = g_dequant_scale_shape;
    std::vector<int64_t> PertokenShape = g_pertoken_shape;
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *pertokenDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *transDequantScaleDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *gaoutDeviceAddr = nullptr;
    void *gPeroutDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    // nd->nz
    void *transWorkspaceAddr = nullptr;
    // fp32->uint64
    void *transParamWorkspaceAddr = nullptr;
    std::string x1Dtype = g_input_tensor_dtype[0];
    std::string x2Dtype = g_input_tensor_dtype[1];
    std::string biasDtype = g_input_tensor_dtype[2];
    std::string ScaleDtype = g_input_tensor_dtype[3];
    std::string PertokenDtype;

    std::string x1Format = g_input_tensor_format[0];
    std::string x2Format = g_input_tensor_format[1];
    std::string biasFormat = g_input_tensor_format[2];
    std::string ScaleFormat = g_input_tensor_format[3];
    std::string PertokenFormat;

    if (g_pertoken_flag == 1) {
        PertokenDtype = g_input_tensor_dtype[4];
        PertokenFormat = g_input_tensor_format[4];
    }

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *Out = nullptr;
    aclTensor *gaOut = nullptr;
    aclTensor *transDequantScale = nullptr;
    aclTensor *Scale = nullptr;
    aclTensor *Pertoken = nullptr;
    aclTensor *gperOut = nullptr;

    uint64_t workspaceSize = 0;
    // nd->nz
    uint64_t transWorkspaceSize = 0;
    // fp32->uint64
    uint64_t transParamWorkspaceSize = 0;
    aclOpExecutor *executor;
    int64_t commTurn = 0;
    int64_t streamMode = 1;

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    CreateWeightTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2");
    if (x2Format == "NZ") {
        ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &transWorkspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        if (transWorkspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        printf("[INFO] x2Format cast to NZ success\n");
        }

    // }
    if (g_bias_flag) {
        CreateTensor(
            biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    // 可选参数pertoken
    if (g_pertoken_flag == 1) {
        printf("----------use pertoken");
        CreateTensor(PertokenShape,
            aclFormat::ACL_FORMAT_ND,
            &pertokenDeviceAddr,
            &Pertoken,
            PertokenDtype,
            g_bin_path,
            args.rankId,
            "pertoken_scale");
    }
    // 1、创建scale---fp32/bf16 2、当ouput=fp16,无pertoken,scale=uint64/int64
    CreateTensor(ScaleShape,
        aclFormat::ACL_FORMAT_ND,
        &scaleDeviceAddr,
        &Scale,
        ScaleDtype,
        g_bin_path,
        args.rankId,
        "dequant_scale");
    if (g_output_tensor_dtype == "fp16" && g_pertoken_flag != 1) {
        std::string transDequantScaleDtype = "uint64";
        CreateTensorWithoutHostData(ScaleShape,
            aclFormat::ACL_FORMAT_ND,
            &transDequantScaleDeviceAddr,
            &transDequantScale,
            transDequantScaleDtype);
        ret = aclnnTransQuantParamV2GetWorkspaceSize(
            Scale, nullptr, transDequantScale, &transParamWorkspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (transParamWorkspaceSize > 0) {
            ret = aclrtMalloc(&transParamWorkspaceAddr, transParamWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate transParamWorkspaceSize failed. ERROR: %d\n", ret);
                      return ret);
        }
        // 调用aclnnTransQuantParamV2第二段接口
        ret = aclnnTransQuantParamV2(transParamWorkspaceAddr, transParamWorkspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
    }

    auto OutSize = GetShapeSize(OutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &Out, g_output_tensor_dtype);
    auto gaOutSize = GetShapeSize(gaOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    if (g_gather_output) {
        CreateTensorWithoutHostData(gaOutShape, aclFormat::ACL_FORMAT_ND, &gaoutDeviceAddr, &gaOut, x1Dtype);
        if (g_pertoken_flag) {
            auto gaPertokenOutSize = GetShapeSize(gaPerOutShape) * dtype_to_sizeof(PertokenDtype);
            CreateTensorWithoutHostData(
                gaPerOutShape, aclFormat::ACL_FORMAT_ND, &gPeroutDeviceAddr, &gperOut, PertokenDtype);
        }
    }
    // 内存校验
    aclrtMemAttr attr = ACL_HBM_MEM;
    size_t mem_free;
    size_t mem_total;
    auto ret_before = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_before = (mem_total - mem_free) / 1024 / 1024;
    printf(" before RunGraph : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n",
        mem_free / 1024 / 1024,
        mem_total / 1024 / 1024,
        mem_used_before,
        ret_before);

    for (int i = 0; i < g_loop_cnt; i++) {
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        // 释放上个step申请的内存
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }

        if (g_output_tensor_dtype == "fp16" && !g_pertoken_flag){
            printf("-----output=fp16---");
            ret = aclnnAllGatherQuantBatchMatMulGetWorkspaceSize(x1,
            x2,
            transDequantScale,
            bias,
            Pertoken,
            g_transpose_b_flag,
            hcomName,
            g_ndev,
            g_gather_axis,
            commTurn,
            Out,
            gaOut,
            gperOut,
            &workspaceSize,
            &executor);
        }
        else {
            printf("-----output=bf16---");
            ret = aclnnAllGatherQuantBatchMatMulGetWorkspaceSize(x1,
            x2,
            Scale,
            bias,
            Pertoken,
            g_transpose_b_flag,
            hcomName,
            g_ndev,
            g_gather_axis,
            commTurn,
            Out,
            gaOut,
            gperOut,
            &workspaceSize,
            &executor);
        }

        CHECK_RET(ret == ACL_SUCCESS,
                  printf("[ERROR] aclnnAllGatherQuantBatchMatMulGetWorkspaceSize failed. ret = %d \n", ret);
                  return ret);
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        ret = aclnnAllGatherQuantBatchMatMul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnAllGatherQuantBatchMatMul failed. ret = %d \n", ret); return ret);
        //     }
        // }
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
    printf("[INFO] allgather_mm workspaceSize = %ld\n", workspaceSize);
    // 内存校验
    auto ret_after = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;
    printf("last run : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n",
        mem_free / 1024 / 1024,
        mem_total / 1024 / 1024,
        mem_used_after,
        ret_after);
    printf("rank %d op mc2 cost memory :%ld M \n", args.rankId, mem_used_after - mem_used_before);
    // 保存结果
    unsigned long long compareSize = OutSize;
    void *hostAicpuBuff = nullptr;
    ACL_CHECK(aclrtMallocHost((void **)&hostAicpuBuff, compareSize));
    ACL_CHECK(
        aclrtMemcpy((void *)hostAicpuBuff, compareSize, (void *)outDeviceAddr, compareSize, ACL_MEMCPY_DEVICE_TO_HOST));
    writeBin(g_bin_path + "/output/" + to_string(args.rankId) + "/" + "output_npu.bin", hostAicpuBuff, compareSize);
    if (hostAicpuBuff != nullptr) {
        ACL_CHECK(aclrtFreeHost(hostAicpuBuff));
    }
    if (g_gather_output){
        unsigned long long compareSize2 = gaOutSize;
        void *hostAicpuBuff2 = nullptr;
        ACL_CHECK(aclrtMallocHost((void **)&hostAicpuBuff2, compareSize2));
        ACL_CHECK(aclrtMemcpy(
            (void *)hostAicpuBuff2, compareSize2, (void *)gaoutDeviceAddr, compareSize2, ACL_MEMCPY_DEVICE_TO_HOST));
        writeBin(
            g_bin_path + "/output/" + to_string(args.rankId) + "/" + "gatherout_npu.bin", hostAicpuBuff2, compareSize2);
        if (hostAicpuBuff2 != nullptr) {
            ACL_CHECK(aclrtFreeHost(hostAicpuBuff2));
        }
    }
    
    // 释放资源
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }

    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (transDequantScale != nullptr) {
        aclDestroyTensor(transDequantScale);
    }
    if (Out != nullptr) {
        aclDestroyTensor(Out);
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
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    // nd->nz
    if (transWorkspaceSize > 0) {
        aclrtFree(transWorkspaceAddr);
    }
    // fp32->uint64
    if (transParamWorkspaceSize > 0) {
        aclrtFree(transParamWorkspaceAddr);
    }
    ACL_CHECK(aclrtDestroyStreamForce(args.stream));
    ACL_CHECK(HcclCommDestroy(args.hcclComm));
    ACL_CHECK(aclrtDestroyContext(args.context));
    ACL_CHECK(aclrtResetDevice(args.rankId));
    return 0;
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);
    int ret;
    int32_t devices[g_ndev];
    for (int i = 0; i < g_ndev; i++) {
        devices[i] = i;
    }
    ACL_CHECK(aclInit(nullptr));
#ifdef DAVID_FAST_MODEL
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    LOG_PRINT("before launchOneThread");
    int re = launchOneThread(args);
    LOG_PRINT("end launchOneThread re = %d", re);

#else

    // 启动多线程
    Args args[g_ndev];
    aclrtStream stream[g_ndev];
    aclrtContext context[g_ndev];
    HcclComm comms[g_ndev];
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // 初始化集合通信域
    ret = HcclCommInitAll(g_ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    std::vector<std::unique_ptr<std::thread>> threads(g_ndev);
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThread, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        threads[rankId]->join();
    }

#endif 
  

    ACL_CHECK(aclFinalize());
    return 0;
}

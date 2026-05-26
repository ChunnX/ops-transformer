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
#include "aclnnop/aclnn_batch_matmul_reduce_scatter_all_to_all.h"

using namespace std;
#define ACL_CHECK(ret)                                                                                     \
    do {                                                                                                   \
        auto retcode = ret;                                                                                \
        if (retcode != ACL_SUCCESS) {                                                                      \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
            return retcode;                                                                                \
        }                                                                                                  \
    } while (0)
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

void split(const std::string &str, std::vector<std::string> &tokens,
           const char delim = '/') {
    // 用于分割字符串
    tokens.clear();
    std::istringstream iss(str);
    std::string tmp;
    while (std::getline(iss, tmp, delim)) {
        if (tmp != " ") {
            tokens.emplace_back(std::move(tmp));
        }
    }
}

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
                    {0,                        0, 0, 0}
                };

        int c = getopt_long(argc, argv, "t:", longOptions, &optionIndex);
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
            default:
                break;
        }
    }
}

struct Args {
    uint32_t rankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream stream;
    aclrtContext context;
};

std::map<std::string, aclDataType> g_data_type_map = {{"fp32",   ACL_FLOAT},
                                                      {"fp16",   ACL_FLOAT16},
                                                      {"bf16",   ACL_BF16},
                                                      {"int8",   ACL_INT8},
                                                      {"int4",   ACL_INT4},
                                                      {"int16",  ACL_INT16},
                                                      {"int64",  ACL_INT64},
                                                      {"uint64", ACL_UINT64},
                                                      {"int32",  ACL_INT32}};

size_t dtype_to_sizeof(std::string &dtype) {
    size_t ret;
    if (dtype == "fp32") {
        ret = sizeof(int32_t);
    } else if (dtype == "fp16") {
        ret = sizeof(int16_t);
    } else if (dtype == "bf16") {
        ret = sizeof(int16_t);
    } else if (dtype == "int8") {
        ret = sizeof(int8_t);
    } else if (dtype == "int4") {
        ret = sizeof(int8_t);
    } else if (dtype == "int32") {
        ret = sizeof(int32_t);
    } else if (dtype == "int16") {
        ret = sizeof(int16_t);
    } else if (dtype == "int64") {
        ret = sizeof(int64_t);
    } else if (dtype == "uint64") {
        ret = sizeof(int64_t);
    } else {
        printf("[ERROR] not invalid dtype\n");
        return -1;
    }
    return ret;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
        shape_size *= i;
    }
    return shape_size;
}

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize) {
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        printf("[ERROR]failed to get file = %s ", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        printf("[ERROR]%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("[ERROR]Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        printf("[ERROR]file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        printf("[ERROR]file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

// 保存bin文件
void writeBin(std::string path, void *buf, size_t size) {
    std::ofstream outfile(path, std::ifstream::binary);
    outfile.write((char *) buf, size);
    outfile.close();
}

int32_t as_uint(const float x) {
    return *(uint * ) & x;
}

float as_float(const int32_t x) {
    return *(float *) &x;
}


float float16_to_float(const int16_t x) {
    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const int32_t e = (x & 0x7C00) >> 10; // exponent
    const int32_t m = (x & 0x03FF) << 13; // mantissa
    const int32_t v = as_uint((float) m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) | ((e == 0) & (m != 0)) * ((v - 37) << 23 |
                                                                                                     ((m << (150 - v)) &
                                                                                                      0x007FE000))); // sign : normalized : denormalized
}

int CreateTensorWithoutHostData(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr,
                                aclTensor **tensor, std::string &dtype) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    auto tensorSize = size * dtype_to_sizeof(dtype);
    // 调用aclCreateTensor创建tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));
    int offset = 0;
    *tensor = aclCreateTensor(shape.data(), shape.size(), g_data_type_map[dtype], strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}


int CreateWeightTensorWithDeviceData(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr,
                                     aclTensor **tensor, std::string &dtype, void **deviceAddr_old) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
    // 计算nd2nz所需内存size
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, g_data_type_map[dtype], &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    auto tensorSize = size * dtype_to_sizeof(dtype);;
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将device侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, tensorSize, *deviceAddr_old, tensorSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("device -> device aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor创建tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));
    int offset = 0;
    if (*tensor != nullptr) {
        aclDestroyTensor(*tensor);
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), g_data_type_map[dtype], strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int CreateTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                 std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name) {
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
    ACL_CHECK(aclrtMallocHost((void **) &tensorHostData, tensorSize));
    ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin", tensorSize,
            tensorHostData, tensorSize);
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), g_data_type_map[dtype], strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int CreateWeightTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                       std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, g_data_type_map[dtype], &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    auto tensorSize = size * dtype_to_sizeof(dtype);;
    if (dtype == "int4") {
        tensorSize = static_cast<uint64_t>(ceil(tensorSize / 2));
    }
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    void *tensorHostData = nullptr;
    ACL_CHECK(aclrtMallocHost((void **) &tensorHostData, tensorSize));
    if (dtype == "int4") {
        ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + "packed_" + name + ".bin", tensorSize,
                 tensorHostData, tensorSize);
    } else {
        ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin", tensorSize,
                 tensorHostData, tensorSize);
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), g_data_type_map[dtype], strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
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
    std::vector<int64_t> transposeX2Shape;
    transposeX2Shape.push_back(x2Shape[0]);
    transposeX2Shape.push_back(x2Shape[2]);
    transposeX2Shape.push_back(x2Shape[1]);
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *transposeX2DeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
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
    aclTensor *Out = nullptr;
    aclTensor *transposeX2 = nullptr;
    uint64_t workspaceSize = 0;
    // 转置
    uint64_t transposeWorkspaceSize = 0;
    aclOpExecutor *executor;

    int64_t yShardType = shard_type;

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
    CreateTensorWithoutHostData(OutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &Out, g_output_tensor_dtype);
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
            ret = aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x1, transposeX2, bias, hcomEpName, hcomTpName, ep_size,
                tp_size, yShardType, Out, &workspaceSize, &executor);
        } else {
            ret = aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x1, x2, bias, hcomEpName, hcomTpName, ep_size,
                tp_size, yShardType, Out, &workspaceSize, &executor);
        }
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // 调用第二段接口
        ret = aclnnBatchMatMulReduceScatterAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAll failed. ret = %d \n", ret);
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
    unsigned long long compareSize = OutSize;
    void *hostAicpuBuff = nullptr;
    ACL_CHECK(aclrtMallocHost((void **) &hostAicpuBuff, compareSize));
    ACL_CHECK(aclrtMemcpy((void *) hostAicpuBuff, compareSize, (void *) outDeviceAddr, compareSize,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    writeBin(g_bin_path + "/output/" + to_string(args.rankId) + "/" + "output_npu.bin", hostAicpuBuff, compareSize);
    if (hostAicpuBuff != nullptr) {
        ACL_CHECK(aclrtFreeHost(hostAicpuBuff));
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




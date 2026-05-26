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
#include "aclnnop/aclnn_quant_matmul_all_reduce_v2.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce_v3.h"

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
std::vector<int64_t> g_antiquant_scale_shape = {};
std::vector<int64_t> g_antiquant_offset_shape = {};
std::vector<int64_t> g_dequant_scale_shape = {};
std::vector<int64_t> g_x3_shape = {};
std::vector<int64_t> g_pertoken_shape = {};
std::vector<int64_t> g_commquant_1_shape = {};
std::vector<int64_t> g_commquant_2_shape = {};
// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
int g_antiquant_offset_flag = 0;
int g_x3_flag = 0;
int g_pertoken_flag = 0;
int g_commquant_flag=0;
// 转置开关
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int g_antiquant_group_size = 0;

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
            default:
                break;
        }
    }
}


struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
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
        std::cout<<"Filepath:" << filePath <<std::endl;
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
    if (tensorSize == 0){
        tensorSize = 1;
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
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, g_data_type_map[dtype], &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    auto tensorSize = size * dtype_to_sizeof(dtype);;
    if (tensorSize == 0){
        tensorSize = 1;
    }
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
                 std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name,int isTransB=0) {
    
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    auto tensorSize = size * dtype_to_sizeof(dtype);
    if (dtype == "int4") {
        tensorSize = static_cast<uint64_t>(ceil(tensorSize / 2));
    }
    if (tensorSize == 0){
        tensorSize = 1;
    }
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    void *tensorHostData = nullptr;
    ACL_CHECK(aclrtMallocHost((void **) &tensorHostData, tensorSize));
    if (dtype == "int4") {
        ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + "packed_" + name + ".bin", tensorSize,
                 tensorHostData, tensorSize);
    } else {
        if (name == "comm_quant_scale_1" || name =="comm_quant_scale_2"){
            ReadFile(bin_path + "/" + name + ".bin", tensorSize, tensorHostData, tensorSize);
        }
        else {
        ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin", tensorSize,
                 tensorHostData, tensorSize);}
    }
    ret = aclrtMemcpy(*deviceAddr, tensorSize, tensorHostData, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] host -> device memory copy failed \n"); return ret);
    ACL_CHECK(aclrtFreeHost(tensorHostData));
    // 调用aclCreateTensor创建tensor
    std::vector<int64_t> strides(shape.size(), 1);
    if (isTransB) {
        // B矩阵转置，stride设置，调用aclCreateTensor创建tensor
        int shapeSize = shape.size();
        strides[shapeSize-1] =  shape[shapeSize-2];
        strides[shapeSize-2] = 1;
        if (shapeSize >= 3) {
            strides[shapeSize-3] = shape[shapeSize-2] * shape[shapeSize-1];
        }
        for (int64_t i = shape.size() - 4; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }     
    } else {
        // 不转置
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    std::vector<int64_t> storageShape;
    if (isTransB) {
        storageShape.push_back(shape[1]);
        storageShape.push_back(shape[0]);
    } else {
        //storageShape = shape;
        storageShape.push_back(GetShapeSize(shape));
    }
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
    if (tensorSize == 0){
        tensorSize = 1;
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
    char hcomName[128];
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcomName, args.stream, args.context);
    timespec tp1, tp2;
    long cost;

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
    std::vector<int64_t> transposeX2Shape;
    transposeX2Shape.push_back(x2Shape[1]);
    transposeX2Shape.push_back(x2Shape[0]);
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
    void *transposeX2DeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    // 转置
    void *transposeWorkspaceAddr = nullptr;
    // nd->nz
    void *transWorkspaceAddr = nullptr;
    // fp32->uint64
    void *transParamWorkspaceAddr = nullptr;
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
    aclTensor *x3 = nullptr;
    aclTensor *pertoken = nullptr;
    aclTensor *commquant_scale_1 = nullptr;
    aclTensor *commquant_scale_2 = nullptr;
    aclTensor *reOut = nullptr;
    aclTensor *transposeX2 = nullptr;
    aclTensor *transDequantScale = nullptr;
    uint64_t workspaceSize = 0;
    // 转置
    uint64_t transposeWorkspaceSize = 0;
    // nd->nz
    uint64_t transWorkspaceSize = 0;
    // fp32->uint64
    uint64_t transParamWorkspaceSize = 0;
    aclOpExecutor *executor;
    int64_t commTurn = 0;
    int64_t streamMode = 1;

    // 创建 tensor
    CreateTensor(x1Shape, aclFormat::ACL_FORMAT_ND, &x1DeviceAddr, &x1, x1Dtype, g_bin_path, args.rankId, "x1");
    if (x2Format == "NZ") {
        CreateWeightTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2");
        // weightNZ, trans_b = True
        if (g_transpose_b_flag) {
            // weightNZ, trans_b = True, 非量化
            if (g_is_quant == "UQUANT" || g_platform == "1971") {
                std::vector<int64_t> dimsData = {1, 0};
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
                // nd->nz
                ret = aclnnTransMatmulWeightGetWorkspaceSize(transposeX2, &transWorkspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
                if (transWorkspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
                ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
                printf("[INFO] x2Format cast to NZ success\n");
            // weightNZ, trans_b = True, 量化 / 伪量化
            } else {
                // 上aclnnWeightQuantBatchMatmulV2与aclnnQuantMatmulV3 kernel只支持[n,k]
                CreateWeightTensorWithDeviceData(x2Shape, aclFormat::ACL_FORMAT_ND, &transposeX2DeviceAddr, &transposeX2, x2Dtype, &x2DeviceAddr);
                ret = aclnnTransMatmulWeightGetWorkspaceSize(transposeX2, &transWorkspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
                if (transWorkspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret); return ret);
                ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
                printf("[INFO] x2Format cast to NZ success\n");
            }
        // weightNZ, trans_b = False
        } else {
            // weightNZ, trans_b = False, 非量化
            if (g_is_quant == "UQUANT" || g_platform == "1971") {
                ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &transWorkspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
                if (transWorkspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
                ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
                printf("[INFO] x2Format cast to NZ success\n");
            // weightNZ, trans_b = False, 量化 / 伪量化
            } else {
                std::vector<int64_t> dimsData = {1, 0};
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

                CreateWeightTensorWithDeviceData(transposeX2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, &transposeX2DeviceAddr);

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
        }
    } else {
        // weightND, trans_b = False
        CreateTensor(x2Shape, aclFormat::ACL_FORMAT_ND, &x2DeviceAddr, &x2, x2Dtype, g_bin_path, args.rankId, "x2",g_transpose_b_flag);
    }
    if (g_bias_flag) {
        CreateTensor(biasShape, aclFormat::ACL_FORMAT_ND, &biasDeviceAddr, &bias, biasDtype, g_bin_path, args.rankId, "bias");
    }
    if (g_x3_flag) {
        CreateTensor(x3Shape, aclFormat::ACL_FORMAT_ND, &x3DeviceAddr, &x3, x3Dtype, g_bin_path, args.rankId, "x3");
    }
    if (g_pertoken_flag){
        CreateTensor(pertokenShape, aclFormat::ACL_FORMAT_ND, &pertokenDeviceAddr, &pertoken, pertokenDtype, g_bin_path, args.rankId, "pertoken_scale");
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
    }
    if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        CreateTensor(dequantScaleShape, aclFormat::ACL_FORMAT_ND, &dequantScaleDeviceAddr,
                     &dequantScale, dequantScaleDtype, g_bin_path, args.rankId, "dequant_scale");
        if (g_output_tensor_dtype == "fp16") {
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
            ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        }
    }
    // 创建output
    auto reduceOutSize = GetShapeSize(reduceOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    CreateTensorWithoutHostData(reduceOutShape, aclFormat::ACL_FORMAT_ND, &outDeviceAddr, &reOut, g_output_tensor_dtype);
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

        if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
            // 调用第一段接口
            ret = aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(
                    x1, x2, bias, antiquantScale, antiquantOffset, x3, hcomName, "sum", commTurn,
                    streamMode, g_antiquant_group_size, reOut, &workspaceSize, &executor);
            // }
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnWeightQuantMatmulAllReduceGetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnWeightQuantMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduce failed. ret = %d \n", ret);return ret);
        } else if ("A8W8" == g_is_quant && !g_pertoken_flag && !g_commquant_flag) {
            // 调用第一段接口
            if (g_transpose_b_flag and g_output_tensor_dtype == "fp16") {
                ret = aclnnQuantMatmulAllReduceGetWorkspaceSize(
                        x1, transposeX2, bias, x3, transDequantScale, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            } else if (g_transpose_b_flag and g_output_tensor_dtype == "bf16") {
                ret = aclnnQuantMatmulAllReduceGetWorkspaceSize(
                        x1, transposeX2, bias, x3, dequantScale, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            } else if (!g_transpose_b_flag and g_output_tensor_dtype == "fp16") {
                ret = aclnnQuantMatmulAllReduceGetWorkspaceSize(
                        x1, x2, bias, x3, transDequantScale, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            } else {
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
        }else if ("A8W8" == g_is_quant && g_pertoken_flag && !g_commquant_flag){
            if (g_transpose_b_flag){
                ret = aclnnQuantMatmulAllReduceV2GetWorkspaceSize(
                        x1, transposeX2, bias, x3, dequantScale, pertoken, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            }else{
                ret = aclnnQuantMatmulAllReduceV2GetWorkspaceSize(
                        x1, x2, bias, x3, dequantScale, pertoken, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
            }
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            if (workspaceSize > 0 && i==0) {
                printf("Only malloc workspacesize once.......\n");
                ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            // 调用第二段接口
            ret = aclnnQuantMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
            CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnQuantMatmulAllReduceV2 failed. ret = %d \n", ret); return ret);
        }else if("A8W8" == g_is_quant && g_commquant_flag){
            if (g_pertoken_flag){
                if (g_transpose_b_flag){
                    printf("*******pertoken = 1, trans_b = 1 *******\n");
                ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, transposeX2, bias, x3, dequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                }else{
                    printf("*******pertoken = 1, trans_b = 0 *******\n");
                ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, x2, bias, x3, dequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                }
            }
            else {
                if (g_transpose_b_flag and g_output_tensor_dtype == "fp16") {
                    printf("*******pertoken = 0, trans_b = 1 ,output = fp16  *******\n");
                    ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, transposeX2, bias, x3, transDequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                } else if (g_transpose_b_flag and g_output_tensor_dtype == "bf16") {
                    printf("*******pertoken = 0, trans_b = 1 ,output = bf16  *******\n");
                    ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, transposeX2, bias, x3, dequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                } else if (!g_transpose_b_flag and g_output_tensor_dtype == "fp16") {
                    printf("*******pertoken = 0, trans_b = 0 ,output = fp16  *******\n");
                    ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
                        x1, x2, bias, x3, transDequantScale, pertoken,commquant_scale_1,commquant_scale_2, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                } else {
                    printf("*******pertoken = 0, trans_b = 0 ,output = bf16  *******\n");
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
                printf("----Use aclnnMatmulAllReduceV2");
                // 调用第一段接口--非量化_带x3
                if (g_transpose_b_flag) {
                    ret = aclnnMatmulAllReduceV2GetWorkspaceSize(
                            x1, transposeX2, bias, x3, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                } else {
                    ret = aclnnMatmulAllReduceV2GetWorkspaceSize(
                            x1, x2, bias, x3, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                }
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduceV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
                if (workspaceSize > 0 && i==0) {
                    printf("Only malloc workspacesize once.......\n");
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                ret = aclnnMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
                CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnMatmulAllReduceV2 failed. ret = %d \n", ret); return ret);
            } else {
                // 调用第一段接口--非量化_不带x3
                if (g_transpose_b_flag) {
                    ret = aclnnMatmulAllReduceGetWorkspaceSize(
                            x1, transposeX2, bias, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                } else {
                    ret = aclnnMatmulAllReduceGetWorkspaceSize(x1, x2, bias, hcomName, "sum", commTurn, streamMode, reOut, &workspaceSize, &executor);
                }
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
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));
        clock_gettime(CLOCK_MONOTONIC, &tp2);
        if (tp2.tv_sec != tp1.tv_sec) {
            cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
        } else {
            cost = tp2.tv_nsec - tp1.tv_nsec;
        }
        printf("[INFO] device:%d step:%d cost_time: %lu, \n", args.rankId, i, cost);
    }
    printf("[INFO] mm_allreduce workspaceSize = %lld\n", workspaceSize);
    // 内存校验
    auto ret_after = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;
    printf("last run : free :%lld M,  total:%lld M, used :%lld M, ret :%d \n",
           mem_free / 1024 / 1024, mem_total / 1024 / 1024, mem_used_after, ret_after);
    printf("rank %d op mc2 cost memory :%lld M \n", args.rankId, mem_used_after - mem_used_before);
    //保存结果
    unsigned long long compareSize = reduceOutSize;
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
    if (x3 != nullptr) {
        aclDestroyTensor(x3);
    }
    if (antiquantScale != nullptr) {
        aclDestroyTensor(antiquantScale);
    }
    if (antiquantOffset != nullptr) {
        aclDestroyTensor(antiquantOffset);
    }
    if (dequantScale != nullptr) {
        aclDestroyTensor(dequantScale);
    }
    if (transDequantScale != nullptr) {
        aclDestroyTensor(transDequantScale);
    }
    if (reOut != nullptr) {
        aclDestroyTensor(reOut);
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
    if (x3DeviceAddr != nullptr) {
        aclrtFree(x3DeviceAddr);
    }
    if (antiquantScaleDeviceAddr != nullptr) {
        aclrtFree(antiquantScaleDeviceAddr);
    }
    if (antiquantOffsetDeviceAddr != nullptr) {
        aclrtFree(antiquantOffsetDeviceAddr);
    }
    if (dequantScaleDeviceAddr != nullptr) {
        aclrtFree(dequantScaleDeviceAddr);
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
    // nd->nz
    if (transWorkspaceSize > 0) {
        aclrtFree(transWorkspaceAddr);
    }
    //fp32->uint64
    if (transParamWorkspaceSize > 0) {
        aclrtFree(transParamWorkspaceAddr);
    }
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
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        threads[rankId]->join();
    }
    ACL_CHECK(aclFinalize());
    return 0;
}



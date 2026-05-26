#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <math.h>
#include <map>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <getopt.h>
#include <sys/syscall.h>  
#include "aclnnop/aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "aclnnop/aclnn_matmul_all_reduce_add_rms_norm.h"
#include "aclnnop/aclnn_quant_matmul_all_reduce_add_rms_norm.h"
#include "aclnnop/aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "aclnnop/aclnn_inplace_matmul_all_reduce_add_rms_norm.h"
#include "aclnnop/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"
#define gettid() syscall(SYS_gettid)
using namespace std;
#define N 128

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

void split(const std::string &str, std::vector<std::string> &tokens, const char delim = '/')
{
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
std::string g_input_tensor_dtype[N] = {};
std::string g_input_tensor_format[N] = {};
std::vector<int64_t> g_x1_shape = {};
std::vector<int64_t> g_x2_shape = {};
std::vector<int64_t> g_bias_shape = {};
std::vector<int64_t> g_antiquant_scale_shape = {};
std::vector<int64_t> g_antiquant_offset_shape = {};
std::vector<int64_t> g_dequant_scale_shape = {};
std::vector<int64_t> g_x3_shape = {};
// 输出
std::string g_output_tensor_dtype = "fp16";
std::vector<int64_t> g_output_tensor_shape = {};
std::string g_output_tensor_format = "ND";
// 可选tensor开关
int g_bias_flag = 0;
int g_antiquant_offset_flag = 0;
int g_x3_flag = 0;
// 转置开关
int g_transpose_b_flag = 0;
// device数量
int g_ndev = 2;
int g_loop_cnt = 10;
int g_antiquant_group_size = 0;

// arn防除0参数--epsilon
float arn_epsilon = 0.000001;

// 是否使用inplace参数
int g_is_inplace = 0;

void GetOption(int argc, char **argv)
{
    while (1) {
        int optionIndex = 0;
        struct option longOptions[] = {{"is_quant", 1, 0, 'a'},
            {"x1_shape", 1, 0, 'b'},
            {"x2_shape", 1, 0, 'c'},
            {"bias_shape", 1, 0, 'd'},
            {"antiquant_scale_shape", 1, 0, 'e'},
            {"antiquant_offset_shape", 1, 0, 'f'},
            {"dequant_scale_shape", 1, 0, 'g'},
            {"x3_shape", 1, 0, 'h'},
            {"input_tensor_dtype", 1, 0, 'i'},
            {"input_tensor_format", 1, 0, 'j'},
            {"output_shape", 1, 0, 'k'},
            {"output_dtype", 1, 0, 'l'},
            {"output_format", 1, 0, 'm'},
            {"n_dev", 1, 0, 'n'},
            {"platform", 1, 0, 'o'},
            {"bias_flag", 1, 0, 'p'},
            {"antiquant_offset_flag", 1, 0, 'q'},
            {"x3_flag", 1, 0, 'r'},
            {"transpose_b_flag", 1, 0, 's'},
            {"bin_path", 1, 0, 't'},
            {"loop_cnt", 1, 0, 'u'},
            {"antiquant_group_size", 1, 0, 'v'},
            {"epsilon", 1, 0, 'w'},
            {"is_inplace", 1, 0, 'x'},
            {0, 0, 0, 0}};

        int c = getopt_long(argc, argv, "w:", longOptions, &optionIndex);
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
            } break;
            case 'c': {
                std::vector<std::string> x2_shape;
                split(optarg, x2_shape, ',');
                for (uint32_t i = 0; i < x2_shape.size(); ++i) {
                    g_x2_shape.push_back(atoi(x2_shape[i].c_str()));
                }
            } break;
            case 'd': {
                std::vector<std::string> bias_shape;
                split(optarg, bias_shape, ',');
                for (uint32_t i = 0; i < bias_shape.size(); ++i) {
                    g_bias_shape.push_back(atoi(bias_shape[i].c_str()));
                }
            } break;
            case 'e': {
                std::vector<std::string> antiquant_scale_shape;
                split(optarg, antiquant_scale_shape, ',');
                for (uint32_t i = 0; i < antiquant_scale_shape.size(); ++i) {
                    g_antiquant_scale_shape.push_back(atoi(antiquant_scale_shape[i].c_str()));
                }
            } break;
            case 'f': {
                std::vector<std::string> antiquant_offset_shape;
                split(optarg, antiquant_offset_shape, ',');
                for (uint32_t i = 0; i < antiquant_offset_shape.size(); ++i) {
                    g_antiquant_offset_shape.push_back(atoi(antiquant_offset_shape[i].c_str()));
                }
            } break;
            case 'g': {
                std::vector<std::string> dequant_scale_shape;
                split(optarg, dequant_scale_shape, ',');
                for (uint32_t i = 0; i < dequant_scale_shape.size(); ++i) {
                    g_dequant_scale_shape.push_back(atoi(dequant_scale_shape[i].c_str()));
                }
            } break;
            case 'h': {
                std::vector<std::string> x3_shape;
                split(optarg, x3_shape, ',');
                for (uint32_t i = 0; i < x3_shape.size(); ++i) {
                    g_x3_shape.push_back(atoi(x3_shape[i].c_str()));
                }
            } break;
            case 'i': {
                std::vector<std::string> dtypes;
                split(optarg, dtypes, ',');
                for (uint32_t i = 0; i < dtypes.size(); ++i) {
                    g_input_tensor_dtype[i] = dtypes[i];
                }
            } break;
            case 'j': {
                std::vector<std::string> formats;
                split(optarg, formats, ',');
                for (uint32_t i = 0; i < formats.size(); ++i) {
                    g_input_tensor_format[i] = formats[i];
                }
            } break;
            case 'k': {
                std::vector<std::string> output_shape;
                split(optarg, output_shape, ',');
                for (uint32_t i = 0; i < output_shape.size(); ++i) {
                    g_output_tensor_shape.push_back(atoi(output_shape[i].c_str()));
                }
            } break;
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
                arn_epsilon = atof(optarg);
                break;
            case 'x':
                g_is_inplace = atoi(optarg);
                break;
            default:
                break;
        }
    }
}
// workspacesize写入文件
void write_csv(string case_name, int space_size)
{
    fstream f;
    f.open("memory.txt", ios::out | ios::app);
    f << case_name << ", npu, " << space_size << endl;
    f.close();
}

int32_t as_uint(const float x)
{
    return *(uint *)&x;
}

float as_float(const int32_t x)
{
    return *(float *)&x;
}

float float16_to_float(const int16_t x)
{
    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
    // +-5.9604645E-8, 3.311 digits
    const int32_t e = (x & 0x7C00) >> 10;       // exponent
    const int32_t m = (x & 0x03FF) << 13;       // mantissa
    const int32_t v = as_uint((float)m) >> 23;  // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                    ((e == 0) & (m != 0)) *
                        ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));  // sign : normalized : denormalized
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream rtStream;
    aclrtContext context;
};

std::map<std::string, aclDataType> g_data_type_map = {{"fp32", ACL_FLOAT},
    {"fp16", ACL_FLOAT16},
    {"bf16", ACL_BF16},
    {"int8", ACL_INT8},
    {"int4", ACL_INT4},
    {"int16", ACL_INT16},
    {"int64", ACL_INT64},
    {"uint64", ACL_UINT64},
    {"int32", ACL_INT32}};

size_t dtype_to_sizeof(std::string &dtype)
{
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

aclTensor *CreateTensor(const int64_t *dims, uint64_t dimNum, aclDataType dataType, aclFormat format, void *dataAddr)
{
    std::vector<int64_t> strides(dimNum, 1);
    for (int64_t i = dimNum - 2; i >= 0; i--) {
        strides[i] = dims[i + 1] * strides[i + 1];
    }
    auto ret = aclCreateTensor(dims, dimNum, dataType, strides.data(), 0, format, dims, dimNum, dataAddr);
    return ret;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
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
void writeBin(std::string path, void *buf, size_t size)
{
    std::ofstream outfile(path, std::ifstream::binary);
    outfile.write((char *)buf, size);
    outfile.close();
}

int PrepareDeviceMemAndData(const std::vector<int64_t> &tensorShape, void **tensorDeviceAddr, std::string dtype,
    std::string &bin_path, int &rankId, std::string name)
{
    auto tensorSize = GetShapeSize(tensorShape) * dtype_to_sizeof(dtype);
    if (dtype == "int4") {
        tensorSize = static_cast<int64_t>(ceil(GetShapeSize(tensorShape) / 2));
    }
    auto ret = aclrtMalloc(tensorDeviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "[ERROR] aclrtMalloc failed" << ret << std::endl; return -1);
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
    ret = aclrtMemcpy(*tensorDeviceAddr, tensorSize, tensorHostData, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);

    CHECK_RET(ret == ACL_SUCCESS, std::cout << "[ERROR] host -> device memory copy failed" << ret << std::endl;
              return -1);
    ACL_CHECK(aclrtFreeHost(tensorHostData));
    // 获取输入数据
    //    auto size = GetShapeSize(tensorShape);
    //    std::vector<int16_t> resultData(size, 5);
    //    ret = aclrtMemcpy(resultData.data(), tensorSize, *tensorDeviceAddr,
    //                      tensorSize, ACL_MEMCPY_DEVICE_TO_HOST);
    //    for (int64_t i = 0; i < size; i++) {
    //        std::cout << "name:" << name << "device_id:" << rankId << std::endl;
    //        printf("result[%ld] is: %f\n", i, float16_to_float(resultData[i]));
    //    }
    return 0;
}

int callMmAllReduce(Args &args)  // 主函数
{
    thread_local void *reduceOutAddr = nullptr;
    thread_local void *normOutAddr = nullptr;
    timespec tp1, tp2;
    long cost;
    // 获取hcomm
    char hcom_name[N];
    auto hccl_ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(hccl_ret == ACL_SUCCESS, printf("[ERROR] HcclGetCommName failed. ret = %d \n", hccl_ret); return -1);
    printf("[INFO] rank %d hcom name is %s\n", args.rankId, hcom_name);

    std::vector<int64_t> selfShape = g_x1_shape;
    thread_local void *selfDeviceAddr = nullptr;
    std::vector<int64_t> otherShape = g_x2_shape;
    thread_local void *otherDeviceAddr = nullptr;
    std::vector<int64_t> residualShape = g_output_tensor_shape;
    thread_local void *residualDeviceAddr = nullptr;
    std::vector<int64_t> gammaShape = g_bias_shape;
    thread_local void *gammaDeviceAddr = nullptr;

    std::vector<int64_t> biasShape = g_bias_shape;
    thread_local void *biasDeviceAddr = nullptr;
    std::vector<int64_t> antiquantScaleShape;
    thread_local void *antiquantScaleDeviceAddr = nullptr;
    std::vector<int64_t> antiquantOffsetShape;
    thread_local void *antiquantOffsetDeviceAddr = nullptr;
    std::vector<int64_t> dequantScaleShape;
    thread_local void *dequantScaleDeviceAddr = nullptr;

    void *transDequantScaleDeviceAddr = nullptr;
    void *transParamWorkspaceAddr = nullptr;

    std::vector<int64_t> reduceOutShape = g_output_tensor_shape;
    thread_local void *workspaceAddr = nullptr;

    std::string x1_dtype = g_input_tensor_dtype[0];
    std::string x2_dtype = g_input_tensor_dtype[1];
    std::string bias_dtype = g_input_tensor_dtype[2];
    std::string residual_dtype = g_input_tensor_dtype[3];
    std::string gamma_dtype = g_input_tensor_dtype[4];
    std::string antiquant_scale_dtype;
    std::string antiquant_offset_dtype;
    std::string dequant_scale_dtype;

    std::string x1_format = g_input_tensor_format[0];
    std::string x2_format = g_input_tensor_format[1];
    std::string bias_format = g_input_tensor_format[2];
    std::string residual_format = g_input_tensor_format[3];
    std::string gamma_format = g_input_tensor_format[4];
    std::string antiquant_scale_format;
    std::string antiquant_offset_format;
    std::string dequant_scale_format;

    if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
        antiquantScaleShape = g_antiquant_scale_shape;
        antiquantOffsetShape = g_antiquant_offset_shape;
        antiquant_scale_dtype = g_input_tensor_dtype[5];
        antiquant_offset_dtype = g_input_tensor_dtype[6];
        antiquant_scale_format = g_input_tensor_format[5];
        antiquant_offset_format = g_input_tensor_format[6];
    }
    if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        dequantScaleShape = g_dequant_scale_shape;
        dequant_scale_dtype = g_input_tensor_dtype[5];
        dequant_scale_format = g_input_tensor_format[5];
    }

    int ret;
    ret = PrepareDeviceMemAndData(selfShape, &selfDeviceAddr, x1_dtype, g_bin_path, args.rankId, "x1");
    CHECK_RET(ret == 0, return ret);
    ret = PrepareDeviceMemAndData(otherShape, &otherDeviceAddr, x2_dtype, g_bin_path, args.rankId, "x2");
    CHECK_RET(ret == 0, return ret);
    ret = PrepareDeviceMemAndData(
        residualShape, &residualDeviceAddr, residual_dtype, g_bin_path, args.rankId, "residual");
    CHECK_RET(ret == 0, return ret);
    ret = PrepareDeviceMemAndData(gammaShape, &gammaDeviceAddr, gamma_dtype, g_bin_path, args.rankId, "gamma");
    CHECK_RET(ret == 0, return ret);

    if (g_bias_flag) {
        ret = PrepareDeviceMemAndData(biasShape, &biasDeviceAddr, bias_dtype, g_bin_path, args.rankId, "bias");
    }
    CHECK_RET(ret == 0, return ret);
    if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
        ret = PrepareDeviceMemAndData(antiquantScaleShape,
            &antiquantScaleDeviceAddr,
            antiquant_scale_dtype,
            g_bin_path,
            args.rankId,
            "antiquant_scale");
        CHECK_RET(ret == 0, return ret);
        ret = PrepareDeviceMemAndData(antiquantOffsetShape,
            &antiquantOffsetDeviceAddr,
            antiquant_offset_dtype,
            g_bin_path,
            args.rankId,
            "antiquant_offset");
        CHECK_RET(ret == 0, return ret);
    }
    if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        ret = PrepareDeviceMemAndData(
            dequantScaleShape, &dequantScaleDeviceAddr, dequant_scale_dtype, g_bin_path, args.rankId, "dequant_scale");
        CHECK_RET(ret == 0, return ret);
    }

    uint64_t workspaceSize = 0;
    uint64_t transParamWorkspaceSize = 0;
    aclOpExecutor *executor;
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    thread_local aclTensor *self = nullptr;
    thread_local aclTensor *other = nullptr;
    thread_local aclTensor *residual = nullptr;
    thread_local aclTensor *gamma = nullptr;

    thread_local aclTensor *gm_bias = nullptr;
    thread_local aclTensor *antiquant_scale = nullptr;
    thread_local aclTensor *antiquant_offset = nullptr;
    thread_local aclTensor *dequant_scale = nullptr;
    thread_local aclTensor *transDequantScale = nullptr;
    thread_local aclTensor *x3 = nullptr;
    thread_local aclTensor *reOut = nullptr;
    thread_local aclTensor *normOut = nullptr;

    // 创建input
    self = CreateTensor(
        selfShape.data(), selfShape.size(), g_data_type_map[x1_dtype], aclFormat::ACL_FORMAT_ND, selfDeviceAddr);
    residual = CreateTensor(residualShape.data(),
        residualShape.size(),
        g_data_type_map[g_output_tensor_dtype],
        aclFormat::ACL_FORMAT_ND,
        residualDeviceAddr);
    gamma = CreateTensor(gammaShape.data(),
        gammaShape.size(),
        g_data_type_map[g_output_tensor_dtype],
        aclFormat::ACL_FORMAT_ND,
        gammaDeviceAddr);

    if (x2_format == "NZ") {
        other = CreateTensor(otherShape.data(),
            otherShape.size(),
            g_data_type_map[x2_dtype],
            aclFormat::ACL_FORMAT_FRACTAL_NZ,
            otherDeviceAddr);
        printf("[INFO] mm_allreduce x2_format cast to NZ\n");
    } else {
        other = CreateTensor(
            otherShape.data(), otherShape.size(), g_data_type_map[x2_dtype], aclFormat::ACL_FORMAT_ND, otherDeviceAddr);
    }
    if (g_bias_flag) {
        gm_bias = CreateTensor(
            biasShape.data(), biasShape.size(), g_data_type_map[bias_dtype], aclFormat::ACL_FORMAT_ND, biasDeviceAddr);
        printf("[INFO] bias_flag=true, create tensor bias\n");
    } else {
        printf("[INFO] bias_flag=false, bias=nullptr\n");
    }
    if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
        antiquant_scale = CreateTensor(antiquantScaleShape.data(),
            antiquantScaleShape.size(),
            g_data_type_map[antiquant_scale_dtype],
            aclFormat::ACL_FORMAT_ND,
            antiquantScaleDeviceAddr);
        if (g_antiquant_offset_flag) {
            antiquant_offset = CreateTensor(antiquantOffsetShape.data(),
                antiquantOffsetShape.size(),
                g_data_type_map[antiquant_offset_dtype],
                aclFormat::ACL_FORMAT_ND,
                antiquantOffsetDeviceAddr);
            printf("[INFO] antiquant_offset_flag=true, create tensor antiquant_offset\n");
        } else {
            printf("[INFO] antiquant_offset_flag=false, antiquant_offset=nullptr\n");
        }
    }
    if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        dequant_scale = CreateTensor(dequantScaleShape.data(),
            dequantScaleShape.size(),
            g_data_type_map[dequant_scale_dtype],
            aclFormat::ACL_FORMAT_ND,
            dequantScaleDeviceAddr);
        if (dequant_scale_dtype == "fp32") {
            auto transDequantScaleSize = GetShapeSize(dequantScaleShape) * sizeof(int64_t);
            ret = aclrtMalloc((void **)&transDequantScaleDeviceAddr, transDequantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, std::cout << "[ERROR] transDequantScale aclrtMalloc failed" << ret << std::endl; return -1);
            transDequantScale = CreateTensor(dequantScaleShape.data(),
                dequantScaleShape.size(),
                g_data_type_map["uint64"],
                aclFormat::ACL_FORMAT_ND,
                transDequantScaleDeviceAddr);
            ret = aclnnTransQuantParamV2GetWorkspaceSize(
                dequant_scale, nullptr, transDequantScale, &transParamWorkspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, printf("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                      return ret);
            // 根据第一段接口计算出的workspaceSize申请device内存
            if (transParamWorkspaceSize > 0) {
                ret = aclrtMalloc(&transParamWorkspaceAddr, transParamWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, printf("allocate transParamWorkspaceSize failed. ERROR: %d\n", ret);
                          return ret);
            }
            // 调用aclnnTransQuantParamV2第二段接口
            ret = aclnnTransQuantParamV2(transParamWorkspaceAddr, transParamWorkspaceSize, executor, args.rtStream);
            CHECK_RET(ret == ACL_SUCCESS, printf("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
            ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.rtStream, 20000));
        }
    }

    // 创建output reOut+normOut
    auto reduceOutSize = GetShapeSize(reduceOutShape) * dtype_to_sizeof(g_output_tensor_dtype);
    ret = aclrtMalloc((void **)&reduceOutAddr, reduceOutSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "[ERROR] reduceOut aclrtMalloc failed" << ret << std::endl; return -1);
    reOut = CreateTensor(reduceOutShape.data(),
        reduceOutShape.size(),
        g_data_type_map[g_output_tensor_dtype],
        aclFormat::ACL_FORMAT_ND,
        reduceOutAddr);

    ret = aclrtMalloc((void **)&normOutAddr, reduceOutSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "[ERROR] normOut aclrtMalloc failed" << ret << std::endl; return -1);
    normOut = CreateTensor(reduceOutShape.data(),
        reduceOutShape.size(),
        g_data_type_map[g_output_tensor_dtype],
        aclFormat::ACL_FORMAT_ND,
        normOutAddr);
    // 内存校验
    aclrtMemAttr attr = ACL_HBM_MEM;
    size_t mem_free;
    size_t mem_total;
    size_t aa = 1024;
    auto ret_before = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_before = (mem_total - mem_free) / aa / aa;
    printf(" before RunGraph : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n",
        mem_free / aa / aa,
        mem_total / aa / aa,
        mem_used_before,
        ret_before);

    for (int i = 0; i < g_loop_cnt; i++) {
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
            if (g_is_inplace == 1) {
                // 调用第一段接口
                auto aclnnRet = aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                    other,
                    gm_bias,
                    antiquant_scale,
                    antiquant_offset,
                    residual,
                    gamma,
                    arn_epsilon,
                    hcom_name,
                    "sum",
                    commTurn,
                    streamMode,
                    g_antiquant_group_size,
                    normOut,
                    &workspaceSize,
                    &executor);
                CHECK_RET(
                    aclnnRet == ACL_SUCCESS,
                    printf(
                        "[ERROR] aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ret = %d \n",
                        aclnnRet);
                    return -1);
                if (workspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                aclnnRet = aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(
                    workspaceAddr, workspaceSize, executor, args.rtStream);
                CHECK_RET(
                    aclnnRet == ACL_SUCCESS,
                    printf("[ERROR] aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm failed. ret = %d \n", aclnnRet);
                    return -1);
            } else {
                // 调用第一段接口
                auto aclnnRet = aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                    other,
                    gm_bias,
                    antiquant_scale,
                    antiquant_offset,
                    residual,
                    gamma,
                    arn_epsilon,
                    hcom_name,
                    "sum",
                    commTurn,
                    streamMode,
                    g_antiquant_group_size,
                    reOut,
                    normOut,
                    &workspaceSize,
                    &executor);
                CHECK_RET(
                    aclnnRet == ACL_SUCCESS,
                    printf("[ERROR] aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ret = %d \n",
                        aclnnRet);
                    return -1);
                if (workspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                aclnnRet =
                    aclnnWeightQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.rtStream);
                CHECK_RET(aclnnRet == ACL_SUCCESS,
                          printf("[ERROR] aclnnWeightQuantMatmulAllReduceAddRmsNorm failed. ret = %d \n", aclnnRet);
                          return -1);
            }

        } else if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
            if (g_is_inplace == 1) {
                auto aclnnRet = 0;
                std::cout << "dequant_scale_dtype:" << dequant_scale_dtype << std::endl;
                if (dequant_scale_dtype == "fp32") {
                    // 调用第一段接口
                    aclnnRet = aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                        other,
                        gm_bias,
                        transDequantScale,
                        residual,
                        gamma,
                        arn_epsilon,
                        hcom_name,
                        "sum",
                        commTurn,
                        streamMode,
                        normOut,
                        &workspaceSize,
                        &executor);
                } else {
                    // 调用第一段接口
                    aclnnRet = aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                        other,
                        gm_bias,
                        dequant_scale,
                        residual,
                        gamma,
                        arn_epsilon,
                        hcom_name,
                        "sum",
                        commTurn,
                        streamMode,
                        normOut,
                        &workspaceSize,
                        &executor);
                }
                CHECK_RET(
                    aclnnRet == ACL_SUCCESS,
                    printf("[ERROR] aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ret = %d \n",
                        aclnnRet);
                    return -1);
                if (workspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                aclnnRet =
                    aclnnInplaceQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.rtStream);
                CHECK_RET(aclnnRet == ACL_SUCCESS,
                          printf("[ERROR] aclnnInplaceQuantMatmulAllReduceAddRmsNorm failed. ret = %d \n", aclnnRet);
                          return -1);
            } else {
                auto aclnnRet = 0;
                if (dequant_scale_dtype == "fp32") {
                    // 调用第一段接口
                    aclnnRet = aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                        other,
                        gm_bias,
                        transDequantScale,
                        residual,
                        gamma,
                        arn_epsilon,
                        hcom_name,
                        "sum",
                        commTurn,
                        streamMode,
                        reOut,
                        normOut,
                        &workspaceSize,
                        &executor);
                } else {
                    // 调用第一段接口
                    aclnnRet = aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                        other,
                        gm_bias,
                        dequant_scale,
                        residual,
                        gamma,
                        arn_epsilon,
                        hcom_name,
                        "sum",
                        commTurn,
                        streamMode,
                        reOut,
                        normOut,
                        &workspaceSize,
                        &executor);
                }
                CHECK_RET(
                    aclnnRet == ACL_SUCCESS,
                    printf("[ERROR] aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ret = %d \n", aclnnRet);
                    return -1);
                if (workspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                aclnnRet = aclnnQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.rtStream);
                CHECK_RET(aclnnRet == ACL_SUCCESS,
                          printf("[ERROR] aclnnQuantMatmulAllReduceAddRmsNorm failed. ret = %d \n", aclnnRet);
                          return -1);
            }
        } else {  // 非量化接口
            if (g_is_inplace == 1) {
                // 调用第一段接口
                auto aclnnRet = aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                    other,
                    gm_bias,
                    residual,
                    gamma,
                    arn_epsilon,
                    hcom_name,
                    "sum",
                    commTurn,
                    1,
                    normOut,
                    &workspaceSize,
                    &executor);
                CHECK_RET(aclnnRet == ACL_SUCCESS,
                          printf("[ERROR] aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ret = %d \n",
                              aclnnRet);
                          return -1);
                printf("[INFO] mm_allreduce workspaceSize = %lu\n", workspaceSize);
                if (workspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                aclnnRet = aclnnInplaceMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.rtStream);
                CHECK_RET(aclnnRet == ACL_SUCCESS,
                          printf("[ERROR] aclnnInplaceMatmulAllReduceAddRmsNorm failed. ret = %d \n", aclnnRet);
                          return -1);
            } else {
                // 调用第一段接口
                auto aclnnRet = aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(self,
                    other,
                    gm_bias,
                    residual,
                    gamma,
                    arn_epsilon,
                    hcom_name,
                    "sum",
                    commTurn,
                    1,
                    reOut,
                    normOut,
                    &workspaceSize,
                    &executor);
                CHECK_RET(
                    aclnnRet == ACL_SUCCESS,
                    printf("[ERROR] aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ret = %d \n", aclnnRet);
                    return -1);
                printf("[INFO] mm_allreduce workspaceSize = %lu\n", workspaceSize);
                if (workspaceSize > 0) {
                    ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
                }
                // 调用第二段接口
                aclnnRet = aclnnMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.rtStream);
                CHECK_RET(aclnnRet == ACL_SUCCESS,
                          printf("[ERROR] aclnnMatmulAllReduceAddRmsNorm failed. ret = %d \n", aclnnRet);
                          return -1);
            }
        }
        // 同步等待任务执行结束
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.rtStream, 10000));
        clock_gettime(CLOCK_MONOTONIC, &tp2);
        if (tp2.tv_sec != tp1.tv_sec) {
            cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
        } else {
            cost = tp2.tv_nsec - tp1.tv_nsec;
        }
        printf("[INFO] device:%d step:%d cost_time: %lu, \n", args.rankId, i, cost);
    }
    printf("[INFO] mm_allreduce workspaceSize = %lu\n", workspaceSize);

    // 内存校验
    write_csv(g_bin_path, workspaceSize);  // 内存写入文件
    auto ret_after = aclrtGetMemInfo(attr, &mem_free, &mem_total);
    size_t mem_used_after = (mem_total - mem_free) / aa / aa;
    printf("last run : free :%ld M,  total:%ld M, used :%ld M, ret :%d \n",
        mem_free / aa / aa,
        mem_total / aa / aa,
        mem_used_after,
        ret_after);
    printf("op cost memory :%ld M \n", mem_used_after - mem_used_before);

    // 保存结果--reout

    unsigned long long compareSize = reduceOutSize;
    void *hostAicpuBuff = nullptr;
    ACL_CHECK(aclrtMallocHost((void **)&hostAicpuBuff, compareSize));
    if (g_is_inplace == 1) {
        ACL_CHECK(aclrtMemcpy(
            (void *)hostAicpuBuff, compareSize, (void *)residualDeviceAddr, compareSize, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        ACL_CHECK(aclrtMemcpy(
            (void *)hostAicpuBuff, compareSize, (void *)reduceOutAddr, compareSize, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    writeBin(g_bin_path + "/output/" + to_string(args.rankId) + "/" + "output_y_npu.bin", hostAicpuBuff, compareSize);
    ACL_CHECK(aclrtFreeHost(hostAicpuBuff));

    // 保存结果--normOut
    void *hostnormOutBuff = nullptr;
    ACL_CHECK(aclrtMallocHost((void **)&hostnormOutBuff, compareSize));
    ACL_CHECK(
        aclrtMemcpy((void *)hostnormOutBuff, compareSize, (void *)normOutAddr, compareSize, ACL_MEMCPY_DEVICE_TO_HOST));
    writeBin(
        g_bin_path + "/output/" + to_string(args.rankId) + "/" + "output_norm_npu.bin", hostnormOutBuff, compareSize);
    ACL_CHECK(aclrtFreeHost(hostnormOutBuff));

    // 释放资源
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(residual);
    aclDestroyTensor(gamma);
    aclDestroyTensor(reOut);
    aclDestroyTensor(transDequantScale);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(residualDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    if (reduceOutAddr != nullptr) {
        aclrtFree(reduceOutAddr);
    }
    if (transParamWorkspaceAddr != nullptr) {
        aclrtFree(transParamWorkspaceAddr);
    }
    if (transDequantScaleDeviceAddr != nullptr) {
        aclrtFree(transDequantScaleDeviceAddr);
    }
    aclrtFree(normOutAddr);
    if (g_bias_flag) {
        aclDestroyTensor(gm_bias);
        aclrtFree(biasDeviceAddr);
    }
    if ("A16W8" == g_is_quant or "A16W4" == g_is_quant) {
        aclDestroyTensor(antiquant_scale);
        aclrtFree(antiquantScaleDeviceAddr);
        if (g_antiquant_offset_flag) {
            aclDestroyTensor(antiquant_offset);
            aclrtFree(antiquantOffsetDeviceAddr);
        }
    }
    if ("A8W8" == g_is_quant or "A8W4" == g_is_quant) {
        aclDestroyTensor(dequant_scale);
        aclrtFree(dequantScaleDeviceAddr);
    }
    aclrtFree(workspaceAddr);
    return 0;
}

int launchOneThread(Args &args)
{
    std::string thread_name = "test" + std::to_string(args.rankId);
    prctl(PR_SET_NAME, thread_name.c_str());
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    int cpu_start = 0;
    int cpu_num = 24;
    for (int i = cpu_start; i < cpu_start + cpu_num; i++) {
        CPU_SET(i, &cpu_set);
    }
    int ret = sched_setaffinity(gettid(), sizeof(cpu_set), &cpu_set);
    if (ret) {
        printf("failed to set cpu affinity, errno = %d\n", errno);
        return -1;
    }
    printf("[INFO] rankId: %d, stream: %p, context : %p\n", args.rankId, args.rtStream, args.context);

    ret = 0;
    ACL_CHECK(aclrtSetDevice(args.rankId));
    ACL_CHECK(aclrtSetCurrentContext(args.context));
    ACL_CHECK(aclrtCreateStream(&args.rtStream));

    ret = callMmAllReduce(args);
    if (ret != 0) {
        printf("[ERROR] execute mm_allreduce failed");
        return ret;
    }
    // device资源销毁
    ACL_CHECK(aclrtDestroyStreamForce(args.rtStream));
    ACL_CHECK(HcclCommDestroy(args.hcclComm));
    // 重置设备
    ACL_CHECK(aclrtResetDevice(args.rankId));
    return 0;
}

extern "C" int rtStreamCreateWithFlags(void **stm, int32_t priority, uint32_t flags);

int launchMultiThread(Args &input_args, int32_t *devices, HcclComm *comms)
{
    Args args[g_ndev];
    aclrtStream rtStream[g_ndev];
    aclrtContext context[g_ndev];
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        ACL_CHECK(aclrtSetDevice(rankId));
        ACL_CHECK(aclrtGetCurrentContext(&context[rankId]));
        ACL_CHECK(rtStreamCreateWithFlags(&rtStream[rankId], 0, 0x600));
    }

    std::vector<std::unique_ptr<std::thread>> threads(g_ndev);
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].rtStream = rtStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThread, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < g_ndev; rankId++) {
        threads[rankId]->join();
    }
    return 0;
}

//==================================================End Test function==========================================

int main(int argc, char *argv[])
{
    GetOption(argc, argv);
    int ret = 0;
    Args input_args;

    if (g_ndev != 1 && g_ndev != 2 && g_ndev != 4 && g_ndev != 8) {
        printf("[ERROR] device_num error, only support 8,4,2,1.\n");
        return -1;
    }
    int32_t devices[g_ndev];
    for (int i = 0; i < g_ndev; i++) {
        devices[i] = i;
    }

    HcclComm comms[N];
    ACL_CHECK(aclInit(NULL));
    // 初始化集合通信域
    for (int i = 0; i < g_ndev; i++) {
        ACL_CHECK(aclrtSetDevice(devices[i]));
    }
    ret = HcclCommInitAll(g_ndev, devices, comms);
    if (ret != 0) {
        printf("[ERROR] call HcclCommInitAll failed.\n");
        return -1;
    }

    // //启动测试
    ret = launchMultiThread(input_args, devices, comms);
    if (ret != 0) {
        printf("[ERROR] call launchMultiThread failed.\n");
        ACL_CHECK(aclFinalize());
        return -1;
    }
    ACL_CHECK(aclFinalize());
    return ret;
}

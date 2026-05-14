#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#ifndef __USE_LARGEFILE64
#define __USE_LARGEFILE64
#endif

#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "/data/huangjunzhe/Ascend.GDN/cann-9.0.0/aarch64-linux/include/aclnnop/aclnn_grouped_matmul_v5.h"
#include <cstring>
#include <cstdlib>

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << filePath << " fileSize " << fileSize << ", sizeof(T) " << sizeof(T)
                  << ", fileSize/sizeof(T) " << fileSize/sizeof(T) << std::endl;
        std::cerr << filePath << " cannot open file" << std::endl;
        return 1;
    }
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << filePath << " read file failed" << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    std::cout << "writing " << filePath << ", size " << size << ", sizeof(T) " << sizeof(T) << std::endl;
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_LARGEFILE | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        ERROR_LOG("Write file Failed.");
        std::cout << "Wrote file size : " << writeSize << ", not equal to expected size " << size * sizeof(T) << std::endl;
        return false;
    }

    return true;
}

template <typename T>
bool WriteLargeFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    const size_t BLOCK_SIZE = 1024 * 1024 * 1024;
    const size_t TOTAL_SIZE = size * sizeof(T);
    std::cout << "writing " << filePath << ", size " << size << ", sizeof(T) " << sizeof(T) << std::endl;
    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_LARGEFILE, 0666);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t written = 0;
    while (written < TOTAL_SIZE) {
        size_t to_write = BLOCK_SIZE;
        if (TOTAL_SIZE - written < BLOCK_SIZE) {
            to_write = TOTAL_SIZE - written;
        }
        ssize_t ret = write(fd, reinterpret_cast<char*>(hostData.data()) + written, to_write);
        if (ret < 0) {
            perror("write error");
            break;
        }
        written += ret;
    }

    (void)close(fd);
    if (written != size * sizeof(T)) {
        ERROR_LOG("Write file Failed.");
        std::cout << "Wrote file size : " << written << ", not equal to expected size " << size * sizeof(T) << std::endl;
        return false;
    }

    return true;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cout << "Device " << deviceId << " run failed. Set other devices.\n";
        int deviceid_inn = 0;
        while(ret != ACL_SUCCESS && deviceid_inn < 8) {
            std::cout << "Trying device " << deviceid_inn << "...";
            ret = aclrtSetDevice(deviceid_inn++);
            if (ret == ACL_SUCCESS) std::cout << " SUCCESS\n";
            else std::cout << "\n";
        }
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

// CreateAclTensorNz: used by single-weight mode (isSingleWeight=true)
// Computes NZ storage shape from logical shape {E, K, N} -> {E, N/64, K/16, 16, 64}
template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    int64_t E_local = shape[0];
    int64_t K_local = shape[1];
    int64_t N_local = shape[2];
    std::vector<int64_t> shapeNz = {E_local, N_local / 64, K_local / 16, 16, 64};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shapeNz.data(), shapeNz.size(), *deviceAddr);
    return 0;
}

// CreateAclTensorListNz (5-param): used by single-weight mode (isSingleWeight=true)
// Each tensor's NZ shape is computed internally via CreateAclTensorNz
template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensorNz<T>(hostData[i], shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

// CreateAclTensorListNz (6-param): used by multi-weight mode (isSingleWeight=false)
// Accepts explicit nzShapes parameter for per-group NZ storage shapes
template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                          const std::vector<std::vector<int64_t>> &nzShapes, void **deviceAddr,
                          aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        auto shapeSize = GetShapeSize(shapes[i]);
        auto memSize = shapeSize * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr + i, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*(deviceAddr + i), memSize, hostData[i].data(), memSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        std::vector<int64_t> strides(shapes[i].size(), 1);
        for (int64_t j = shapes[i].size() - 2; j >= 0; j--) {
            strides[j] = shapes[i][j + 1] * strides[j + 1];
        }
        tensors[i] = aclCreateTensor(shapes[i].data(), shapes[i].size(), dataType, strides.data(), 0,
                                     aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShapes[i].data(), nzShapes[i].size(),
                                     *(deviceAddr + i));
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

template <typename T>
void PrintVector(const std::vector<T> &arrays, const std::string &name) {
    std::cout << name << ": ";
    for(int i = 0; i < arrays.size(); ++i) {
        std::cout << arrays[i] << " ";
    }
    std::cout << " , sizeof(T) = " << sizeof(T);
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    int32_t deviceId = 5;
    int64_t E = 5;
    int64_t M = 64;
    int64_t K = 2048;
    int64_t N = 512;
    int DEBUG = 0;
    int isBF16 = 0;
    int isNZ = 0;
    int tuningNum = 0;
    int quantGroupSize = K;
    // isSingleWeight: true = single weight mode (test_gmms.cpp logic),
    //                 false = multi weight mode (test_gmms_multi.cpp logic)
    bool isSingleWeight = true;
    bool withOffset = false;
    const std::string bf16 = "bf16";
    const std::string nz = "nz";
    const std::string data_path = std::getenv("A8W4_DATA_PATH") ? std::getenv("A8W4_DATA_PATH") : "/data/huangjunzhe/gmm/a8w4-run/data";

    if(argc == 7) {
        E = atoi(argv[1]);
        M = atoi(argv[2]);
        K = atoi(argv[3]);
        N = atoi(argv[4]);
        isBF16 = bf16 == argv[5]? 1 : 0;
        isNZ = nz == argv[6]? 1 : 0;
    }
    if(argc == 10) {
        E = atoi(argv[1]);
        M = atoi(argv[2]);
        K = atoi(argv[3]);
        N = atoi(argv[4]);
        isBF16 = bf16 == argv[5]? 1 : 0;
        isNZ = nz == argv[6]? 1 : 0;
        quantGroupSize = atoi(argv[7]);
        tuningNum = atoi(argv[8]);
        deviceId = atoi(argv[9]);
    }
    // Parse isSingleWeight from argv[10], default is true
    if(argc >= 11) {
        E = atoi(argv[1]);
        M = atoi(argv[2]);
        K = atoi(argv[3]);
        N = atoi(argv[4]);
        isBF16 = bf16 == argv[5]? 1 : 0;
        isNZ = nz == argv[6]? 1 : 0;
        quantGroupSize = atoi(argv[7]);
        tuningNum = atoi(argv[8]);
        deviceId = atoi(argv[9]);
        withOffset = (strcmp(argv[10], "withOffset") == 0 || strcmp(argv[10], "true") == 0) ? true : false;
        isSingleWeight = (strcmp(argv[11], "multi") == 0 || strcmp(argv[11], "0") == 0) ? false : true;
    }

    int quantGroupNum = K / quantGroupSize;

    std::cout << "[test_gmms_all.cpp] isSingleWeight=" << isSingleWeight
              << ", tuningNum=" << tuningNum
              << ", deviceId=" << deviceId << ", E=" << E
              << ", quantGroupNum=" << quantGroupNum << ", quantGroupSize=" << quantGroupSize
              << ", M=" << M << ", K=" << K << ", N=" << N
              << ", isBF16=" << isBF16 << ", isNZ=" << isNZ << ", withOffset=" << withOffset << std::endl;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    std::cout << "[test_gmms_all.cpp] selected device: " << deviceId << std::endl;

    // ========================================================================
    // Conditional branch: isSingleWeight controls single-weight vs multi-weight mode
    // ========================================================================
    if (isSingleWeight) {
        // ====================================================================
        // SINGLE WEIGHT MODE (originally from test_gmms.cpp)
        // Uses a single weight tensor [E, K, N] stored as one TensorList entry
        // ====================================================================
        std::cout << "[test_gmms_all.cpp] Running SINGLE WEIGHT mode" << std::endl;

        std::vector<int64_t> xShape = {M, K};
        std::vector<int64_t> weightShapeFile = {E, K, N / 2};
        std::vector<int64_t> weightShape = {E, K, N};
        std::vector<int64_t> biasShape = {E, N};
        std::vector<int64_t> offsetShape = {E, 1, N};
        std::vector<int64_t> scaleShape = {E, quantGroupNum, N};
        std::vector<int64_t> perTokenScaleShape = {M, 1};
        std::vector<int64_t> groupListShape = {E};
        std::vector<int64_t> quantOutShape = {M, N};

        if (DEBUG == 1) {
            PrintVector(xShape, "xShape");
            PrintVector(weightShapeFile, "weightShapeFile");
            PrintVector(weightShape, "weightShape");
            PrintVector(biasShape, "biasShape");
            PrintVector(offsetShape, "offsetShape");
            PrintVector(scaleShape, "scaleShape");
            PrintVector(perTokenScaleShape, "perTokenScaleShape");
            PrintVector(groupListShape, "groupListShape");
            PrintVector(quantOutShape, "quantOutShape");
        }

        void* xDeviceAddr = nullptr;
        void* weightDeviceAddr = nullptr;
        void* biasDeviceAddr = nullptr;
        void* offsetDeviceAddr = nullptr;
        void* scaleDeviceAddr = nullptr;
        void* perTokenScaleDeviceAddr = nullptr;
        void* groupListDeviceAddr = nullptr;
        void* quantOutDeviceAddr = nullptr;
        void* quantOutScaleDeviceAddr = nullptr;

        aclTensorList* x = nullptr;
        aclTensorList* weight = nullptr;
        aclTensorList* bias = nullptr;
        aclTensorList* scale = nullptr;
        aclTensorList* offset = nullptr;
        aclTensorList* perTokenScale = nullptr;
        aclTensor* groupList = nullptr;
        aclTensorList* quantOut = nullptr;

        std::vector<int8_t> xHostData(M * K, 0);
        std::vector<int8_t> weightHostData(E * N * K, 0);
        std::vector<float> biasHostData(E * N, 0);
        std::vector<float> offsetHostData(E * N, 1.5f);
        std::vector<uint64_t> scaleHostData(E * N * quantGroupNum, 0);
        std::vector<float> perTokenScaleHostData(M, 0);
        std::vector<int64_t> groupListHostData(E, M / E);
        std::vector<short> quantOutHostData(M * N, 0);

        #ifndef NOFILE
                std::cout << "[test_gmms_all.cpp] (single) Reading files... " << std::endl;
                ReadFile(data_path + "/a8w4_nz_test_x.bin", xShape, xHostData);
                if (isNZ) ReadFile(data_path + "/a8w4_nz_test_weight_nz.bin", weightShapeFile, weightHostData);
                else ReadFile(data_path + "/a8w4_nz_test_weight_nd.bin", weightShapeFile, weightHostData);

                ReadFile(data_path + "/a8w4_nz_test_bias.bin", biasShape, biasHostData);
                if (withOffset) {
                    std::ifstream offsetFileCheck(data_path + "/a8w4_nz_test_offset.bin");
                    if (offsetFileCheck.is_open()) {
                        offsetFileCheck.close();
                        ReadFile(data_path + "/a8w4_nz_test_offset.bin", offsetShape, offsetHostData);
                        std::cout << "[test_gmms_all.cpp] (single) Read offset from file" << std::endl;
                    } else {
                        std::cout << "[test_gmms_all.cpp] (single) offset file not found, using default value 1.5" << std::endl;
                    }
                } else {
                    std::cout << "[test_gmms_all.cpp] (single) withOffset=false" << std::endl;
                }
                ReadFile(data_path + "/a8w4_nz_test_scale.bin", scaleShape, scaleHostData);
                ReadFile(data_path + "/a8w4_nz_test_group_list.bin", groupListShape, groupListHostData);
                ReadFile(data_path + "/a8w4_nz_test_per_token_scale.bin", perTokenScaleShape, perTokenScaleHostData);
        #else
                std::cout << "[test_gmms_all.cpp] (single) Will not read file. Default M per group: " << M / E << std::endl;
        #endif

        #ifdef GENERATED_CASES
                ReadFile(argv[10], xShape, xHostData);
                if (isNZ) ReadFile(argv[11], weightShapeFile, weightHostData);
                else ReadFile(argv[11], weightShapeFile, weightHostData);
                ReadFile(argv[12], biasShape, biasHostData);
                ReadFile(argv[13], scaleShape, scaleHostData);
                ReadFile(argv[14], groupListShape, groupListHostData);
                ReadFile(argv[15], perTokenScaleShape, perTokenScaleHostData);
        #endif

        ret = CreateAclTensorList<int8_t>({xHostData}, {xShape}, &xDeviceAddr, aclDataType::ACL_INT8, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        if (isNZ) {
            ret = CreateAclTensorListNz<int8_t>({weightHostData}, {weightShape}, &weightDeviceAddr, aclDataType::ACL_INT4, &weight);
        } else {
            ret = CreateAclTensorList<int8_t>({weightHostData}, {weightShape}, &weightDeviceAddr, aclDataType::ACL_INT4, &weight);
        }
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<float>({biasHostData}, {biasShape}, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<float>({offsetHostData}, {offsetShape}, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<uint64_t>({scaleHostData}, {scaleShape}, &scaleDeviceAddr, aclDataType::ACL_UINT64, &scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<float>({perTokenScaleHostData}, {perTokenScaleShape}, &perTokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &perTokenScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<short>({quantOutHostData}, {quantOutShape}, &quantOutDeviceAddr,
                                         isBF16 == 1 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16, &quantOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;

        std::vector<int64_t> tuningConfigData = {tuningNum};
        aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

        std::cout << "[test_gmms_all.cpp] (single) Entering aclnnGroupedMatmulV5GetWorkspaceSize\n";
        ret = aclnnGroupedMatmulV5GetWorkspaceSize(x, weight, bias, scale, withOffset ? offset : nullptr, nullptr, nullptr, perTokenScale, groupList,
                     nullptr, nullptr, nullptr, 3, 0, 1, 0, tuningConfig, quantOut, nullptr, nullptr, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        }

        std::cout << "[test_gmms_all.cpp] (single) Entering aclnnGroupedMatmulV5\n";
        ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

        ret = aclrtSynchronizeStream(stream);
        aclrtSynchronizeDevice();
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        auto size = GetShapeSize(quantOutShape);
        std::vector<short> out1Data(size, 0);
        std::cout << "[test_gmms_all.cpp] (single) GetShapeSize(quantOutShape) " << size
                  << ", out1Data.size() * sizeof(out1Data[0]): " << out1Data.size() * sizeof(out1Data[0])
                  << ", size * sizeof(out1Data[0]): " << size * sizeof(out1Data[0]) << std::endl;
        ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), quantOutDeviceAddr,
                          size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        std::cout << "[test_gmms_all.cpp] (single) aclrtMemcpy = " << ret << std::endl;

        #ifndef NOFILE
                if (size < 2147479552 / sizeof(short)) WriteFile(data_path + "/a8w4_nz_test_result.bin", size, out1Data);
                else WriteLargeFile(data_path + "/a8w4_nz_test_result.bin", size, out1Data);
        #else
                std::cout << "[test_gmms_all.cpp] (single) will not write result bin\n";
        #endif

        aclDestroyTensorList(x);
        aclDestroyTensorList(weight);
        aclDestroyTensorList(bias);
        aclDestroyTensorList(offset);
        aclDestroyTensorList(scale);
        aclDestroyTensorList(perTokenScale);
        aclDestroyTensor(groupList);
        aclDestroyTensorList(quantOut);
        aclDestroyIntArray(tuningConfig);

        aclrtFree(xDeviceAddr);
        aclrtFree(weightDeviceAddr);
        aclrtFree(biasDeviceAddr);
        aclrtFree(offsetDeviceAddr);
        aclrtFree(scaleDeviceAddr);
        aclrtFree(perTokenScaleDeviceAddr);
        aclrtFree(groupListDeviceAddr);
        aclrtFree(quantOutDeviceAddr);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }

    } else {
        // ====================================================================
        // MULTI WEIGHT MODE (originally from test_gmms_multi.cpp)
        // Uses per-group weight tensors, each [K, N] as separate TensorList entries
        // ====================================================================
        std::cout << "[test_gmms_all.cpp] Running MULTI WEIGHT mode" << std::endl;

        std::vector<int64_t> xShape = {M, K};
        std::vector<int64_t> weightShapeSingleND = {K, N};
        std::vector<int64_t> weightShapeSingleNz = {N / 64, K / 16, 16, 64};
        std::vector<int64_t> biasShapeSingle = {N};
        std::vector<int64_t> scaleShapeSingle = {quantGroupNum, N};
        std::vector<int64_t> offsetShapeSingle = {1, N};
        std::vector<int64_t> perTokenScaleShape = {M};
        std::vector<int64_t> groupListShape = {E};

        std::vector<std::vector<int64_t>> weightShapes(E);
        std::vector<std::vector<int64_t>> weightNzShapes(E);
        std::vector<std::vector<int64_t>> biasShapes(E);
        std::vector<std::vector<int64_t>> scaleShapes(E);
        std::vector<std::vector<int64_t>> offsetShapes(E);

        for (int64_t i = 0; i < E; i++) {
            if (isNZ) {
                weightShapes[i] = weightShapeSingleND;
                weightNzShapes[i] = weightShapeSingleNz;
            } else {
                weightShapes[i] = weightShapeSingleND;
                weightNzShapes[i] = weightShapeSingleND;
            }
            biasShapes[i] = biasShapeSingle;
            scaleShapes[i] = scaleShapeSingle;
            offsetShapes[i] = offsetShapeSingle;
        }

        void* xDeviceAddr = nullptr;
        void* weightDeviceAddrs[E];
        void* biasDeviceAddrs[E];
        void* scaleDeviceAddrs[E];
        void* offsetDeviceAddrs[E];
        void* perTokenScaleDeviceAddr = nullptr;
        void* groupListDeviceAddr = nullptr;
        void* quantOutDeviceAddr = nullptr;
        memset(weightDeviceAddrs, 0, sizeof(weightDeviceAddrs));
        memset(biasDeviceAddrs, 0, sizeof(biasDeviceAddrs));
        memset(scaleDeviceAddrs, 0, sizeof(scaleDeviceAddrs));
        memset(offsetDeviceAddrs, 0, sizeof(offsetDeviceAddrs));

        aclTensorList* x = nullptr;
        aclTensorList* weight = nullptr;
        aclTensorList* bias = nullptr;
        aclTensorList* scale = nullptr;
        aclTensorList* offset = nullptr;
        aclTensorList* perTokenScale = nullptr;
        aclTensor* groupList = nullptr;
        aclTensorList* quantOut = nullptr;

        std::vector<int8_t> xHostData(M * K, 0);
        std::vector<float> perTokenScaleHostData(M, 0);
        std::vector<int64_t> groupListHostData(E, M / E);
        std::vector<short> quantOutHostData(M * N, 0);

        std::vector<std::vector<int8_t>> weightHostDataPerGroup(E);
        std::vector<std::vector<float>> biasHostDataPerGroup(E);
        std::vector<std::vector<uint64_t>> scaleHostDataPerGroup(E);
        std::vector<std::vector<float>> offsetHostDataPerGroup(E);

        for (int64_t i = 0; i < E; i++) {
            weightHostDataPerGroup[i].resize(K * N, 0);
            biasHostDataPerGroup[i].resize(N, 0);
            scaleHostDataPerGroup[i].resize(quantGroupNum * N, 0);
            offsetHostDataPerGroup[i].resize(N, 0);
        }

#ifndef NOFILE
        std::cout << "[test_gmms_all.cpp] (multi) Reading files... " << std::endl;

        ReadFile(data_path + "/a8w4_nz_test_x.bin", xShape, xHostData);

        {
            std::vector<int64_t> weightShapeFile = {E, K, N / 2};
            std::vector<int8_t> weightHostDataAll(E * K * (N / 2), 0);
            if (isNZ) ReadFile(data_path + "/a8w4_nz_test_weight_nz.bin", weightShapeFile, weightHostDataAll);
            else ReadFile(data_path + "/a8w4_nz_test_weight_nd.bin", weightShapeFile, weightHostDataAll);
            int64_t singleWeightFileSize = K * (N / 2);
            for (int64_t i = 0; i < E; i++) {
                weightHostDataPerGroup[i].assign(K * N, 0);
                std::copy(weightHostDataAll.begin() + i * singleWeightFileSize,
                          weightHostDataAll.begin() + (i + 1) * singleWeightFileSize,
                          weightHostDataPerGroup[i].begin());
            }
        }

        {
            std::vector<int64_t> biasShapeAll = {E, N};
            std::vector<float> biasHostDataAll(E * N, 0);
            ReadFile(data_path + "/a8w4_nz_test_bias.bin", biasShapeAll, biasHostDataAll);
            for (int64_t i = 0; i < E; i++) {
                std::copy(biasHostDataAll.begin() + i * N,
                          biasHostDataAll.begin() + (i + 1) * N,
                          biasHostDataPerGroup[i].begin());
            }
        }

        {
            std::vector<int64_t> offsetShapeAll = {E, N};
            std::vector<float> offsetHostDataAll(E * N, 0);
            ReadFile(data_path + "/a8w4_nz_test_offset.bin", offsetShapeAll, offsetHostDataAll);
            for (int64_t i = 0; i < E; i++) {
                std::copy(offsetHostDataAll.begin() + i * N,
                          offsetHostDataAll.begin() + (i + 1) * N,
                          offsetHostDataPerGroup[i].begin());
            }
        }

        {
            std::vector<int64_t> scaleShapeAll = {E, quantGroupNum, N};
            std::vector<uint64_t> scaleHostDataAll(E * N * quantGroupNum, 0);
            ReadFile(data_path + "/a8w4_nz_test_scale.bin", scaleShapeAll, scaleHostDataAll);
            int64_t singleScaleSize = quantGroupNum * N;
            for (int64_t i = 0; i < E; i++) {
                std::copy(scaleHostDataAll.begin() + i * singleScaleSize,
                          scaleHostDataAll.begin() + (i + 1) * singleScaleSize,
                          scaleHostDataPerGroup[i].begin());
            }
        }

        ReadFile(data_path + "/a8w4_nz_test_group_list.bin", groupListShape, groupListHostData);
        ReadFile(data_path + "/a8w4_nz_test_per_token_scale.bin", perTokenScaleShape, perTokenScaleHostData);
#else
        std::cout << "[test_gmms_all.cpp] (multi) Will not read file. Default M per group: " << M / E << std::endl;
#endif

        ret = CreateAclTensorList<int8_t>({xHostData}, {xShape}, &xDeviceAddr, aclDataType::ACL_INT8, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        if (isNZ) {
            ret = CreateAclTensorListNz<int8_t>(weightHostDataPerGroup, weightShapes, weightNzShapes,
                                                weightDeviceAddrs, aclDataType::ACL_INT4, &weight);
        } else {
            ret = CreateAclTensorList<int8_t>(weightHostDataPerGroup, weightShapes, weightDeviceAddrs,
                                              aclDataType::ACL_INT4, &weight);
        }
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<float>(biasHostDataPerGroup, biasShapes, biasDeviceAddrs, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<uint64_t>(scaleHostDataPerGroup, scaleShapes, scaleDeviceAddrs, aclDataType::ACL_UINT64, &scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<float>(offsetHostDataPerGroup, offsetShapes, offsetDeviceAddrs, aclDataType::ACL_FLOAT, &offset);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensorList<float>({perTokenScaleHostData}, {perTokenScaleShape},
                                         &perTokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &perTokenScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr,
                              aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<int64_t> quantOutShape = {M, N};
        ret = CreateAclTensorList<short>({quantOutHostData}, {quantOutShape}, &quantOutDeviceAddr,
                                         isBF16 == 1 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16, &quantOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;

        std::vector<int64_t> tuningConfigData = {tuningNum};
        aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

        std::cout << "[test_gmms_all.cpp] (multi) Entering aclnnGroupedMatmulV5GetWorkspaceSize (multi-tensor)\n";
        ret = aclnnGroupedMatmulV5GetWorkspaceSize(
            x, weight, bias, scale, withOffset ? offset : nullptr, nullptr, nullptr,
            perTokenScale, groupList,
            nullptr, nullptr, nullptr,
            3, 0, 1, 0, tuningConfig,
            quantOut, nullptr, nullptr,
            &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        }

        std::cout << "[test_gmms_all.cpp] (multi) Entering aclnnGroupedMatmulV5\n";
        ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

        ret = aclrtSynchronizeStream(stream);
        aclrtSynchronizeDevice();
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        int64_t totalOutSize = GetShapeSize({M, N});
        std::vector<short> out1Data(totalOutSize, 0);
        ret = aclrtMemcpy(out1Data.data(), totalOutSize * sizeof(short),
                          quantOutDeviceAddr, totalOutSize * sizeof(short),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

#ifndef NOFILE
        if (totalOutSize < 2147479552 / sizeof(short)) WriteFile(data_path + "/a8w4_nz_test_result_multi.bin", totalOutSize, out1Data);
        else WriteLargeFile(data_path + "/a8w4_nz_test_result_multi.bin", totalOutSize, out1Data);
#else
        std::cout << "[test_gmms_all.cpp] (multi) will not write result bin\n";
#endif

        aclDestroyTensorList(x);
        aclDestroyTensorList(weight);
        aclDestroyTensorList(bias);
        aclDestroyTensorList(scale);
        aclDestroyTensorList(offset);
        aclDestroyTensorList(perTokenScale);
        aclDestroyTensor(groupList);
        aclDestroyTensorList(quantOut);
        aclDestroyIntArray(tuningConfig);

        aclrtFree(xDeviceAddr);
        for (int64_t i = 0; i < E; i++) {
            aclrtFree(weightDeviceAddrs[i]);
            aclrtFree(biasDeviceAddrs[i]);
            aclrtFree(scaleDeviceAddrs[i]);
            aclrtFree(offsetDeviceAddrs[i]);
        }
        aclrtFree(perTokenScaleDeviceAddr);
        aclrtFree(groupListDeviceAddr);
        aclrtFree(quantOutDeviceAddr);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

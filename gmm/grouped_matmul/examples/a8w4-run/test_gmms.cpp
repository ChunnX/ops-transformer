/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "/data/huangjunzhe/Ascend.GDN/cann-9.0.0/aarch64-linux/include/aclnnop/aclnn_grouped_matmul_v5.h"
#include <cstring>


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
// g++ -std=c++17 transpose_aclnnAdd.cpp -L/home/wnc/cann/ascend-toolkit/latest/lib64 -lascendcl -lopapi -lnnopbase -lcust_opapi -I/home/wnc/cann/ascend-toolkit/latest/include -o test_gmms
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
    	std::cout << filePath << " fileSize " << fileSize << ", sizeof(T) " << sizeof(T) << ", fileSize/sizeof(T) " << fileSize/sizeof(T) << std::endl;
        std::cerr << filePath << "无法打开文件" << std::endl;
        return 1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << filePath <<"读取文件失败" << std::endl;
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
    const size_t BLOCK_SIZE = 1024 * 1024 * 1024;  // 1G
    const size_t TOTAL_SIZE = size * sizeof(T);
    std::cout << "writing " << filePath << ", size " << size << ", sizeof(T) " << sizeof(T) << std::endl;
    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_LARGEFILE, 0666);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    // void *buf = malloc(BLOCK_SIZE);  // 1MB
    // if (!buf) {
    //     perror("malloc error in writing file");
    //     close(fd);
    //     return 1;
    // }
    //memset(buf, 0, BLOCK_SIZE); // 填充缓冲区

    //size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));

    size_t written = 0;
    while (written < TOTAL_SIZE) {
        size_t to_write = BLOCK_SIZE;
        if (TOTAL_SIZE - written < BLOCK_SIZE) {
            to_write = TOTAL_SIZE - written;
        }
        ssize_t ret = write(fd, reinterpret_cast<char*>(hostData.data())+written, to_write);
        if (ret < 0) {
            perror("write error");
            break;
        }
        written += ret;
    }

    //free(buf);
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
    // 固定写法，acl初始化
    //std::cout << "1\n";
    auto ret = aclInit(nullptr);
    //std::cout << "2\n";
    
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
    //std::cout << "3\n";
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    //std::cout << "4\n";
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    //std::cout << "CreateAclTensor" << hostData[0] << " " << hostData[1] << " " << hostData[2] << " " << hostData[3] << " " << hostData[4] << " \n";
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
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

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
int64_t E = shape[0];
int64_t K = shape[1];
int64_t N = shape[2];
std::vector<int64_t> shapeNz = {E, N/64 ,K/16,16,64};
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shapeNz.data(), shapeNz.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
//std::cout << "Nz shape " << shapes.size() << std::endl;
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensorNz<T>(hostData[i], shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
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
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // # // x: [8192, 7168], M, K
    // # // weight: [256, 8192, 512], E, K, N
    // # // out1: [8192, 256]
    // # // out2: [8192]
    int64_t E = 5;
    int64_t M = 64;
    int64_t K = 2048;
    int64_t N = 512;
    int DEBUG = 0;
    int isBF16 = 0;
    int isNZ = 0;
    int tuningNum = 0;
    int quantGroupSize = K;
    const std::string bf16 = "bf16";
    const std::string nz = "nz";
    const std::string data_path = "/data/huangjunzhe/gmm/a8w4-run/data";

    if(argc == 7) {
        E = atoi(argv[1]); // 1
        M = atoi(argv[2]); // 
        K = atoi(argv[3]);
        N = atoi(argv[4]);
        isBF16 = bf16 == argv[5]? 1 : 0;
        isNZ = nz == argv[6]? 1 : 0;
//        PERGROUP = atoi(argv[5]); // 512 1024 2048
    }
    if(argc == 10) {         //adding quantGroupSize, tuningNum, deviceId
        E = atoi(argv[1]); // 1
        M = atoi(argv[2]); // 
        K = atoi(argv[3]);
        N = atoi(argv[4]);
        isBF16 = bf16 == argv[5]? 1 : 0;
        isNZ = nz == argv[6]? 1 : 0;
        quantGroupSize = atoi(argv[7]);
        tuningNum = atoi(argv[8]);
        deviceId = atoi(argv[9]);
//        PERGROUP = atoi(argv[5]); // 512 1024 2048
    }
    // if(argc == 11) {    // adding DEBUG
    //     E = atoi(argv[1]); // 1
    //     M = atoi(argv[2]); // 
    //     K = atoi(argv[3]);
    //     N = atoi(argv[4]);
    //     isBF16 = bf16 == argv[5]? 1 : 0;
    //     isNZ = nz == argv[6]? 1 : 0;
    //     quantGroupSize = atoi(argv[7]);
    //     tuningNum = atoi(argv[8]);
    //     deviceId = atoi(argv[9]);
    //     DEBUG = atoi(argv[10]); // 512 1024 2048
    // }
    //quantGroupSize = 64;
    int quantGroupNum = K / quantGroupSize;
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightShapeFile = {E,K,N/2};
    std::vector<int64_t> weightShape = {E,K,N};
    //std::vector<int64_t> weightShapeFile = {E,N / 64 ,K / 16,16,64 / 2};
    //std::vector<int64_t> weightShape = {E,N / 64 ,K / 16,16,64};
    //std::vector<int64_t> weightShape = {E,K, N ,1, N};
    //std::vector<int64_t> weightShape = {E,K,N};
    std::vector<int64_t> biasShape = {E, N};
    std::vector<int64_t> scaleShape = {E, quantGroupNum, N};
    //std::cout << "quantGroupNum = " <<  quantGroupNum - 1 << ", not divisible to K(" << K << ")" << std::endl ;
    std::vector<int64_t> perTokenScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> quantOutShape = {M, N};
    //std::vector<int64_t> quantOutScaleShape = {M};
std::cout << "[test_gmms.cpp] tuningNum=" << tuningNum << ", deviceId=" << deviceId << ", E=" << E << ", quantGroupNum=" << quantGroupNum <<", quantGroupSize=" << quantGroupSize << ", " << "M=" << M << ", " << "K=" << K << ", " << "N=" << N << ", " << "isBF16=" << isBF16 << ", " << "isNZ=" << isNZ << ", DEBUG=" << DEBUG << std::endl;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    std::cout << "[test_gmms.cpp] selected device: " << deviceId << std::endl;
if (DEBUG==1) {
    PrintVector(xShape,"xShape");
    PrintVector(weightShapeFile, "weightShapeFile");
    PrintVector(weightShape, "weightShape");
    PrintVector(biasShape,"biasShape");
    PrintVector(scaleShape,"scaleShape");
    PrintVector(perTokenScaleShape,"perTokenScaleShape");
    PrintVector(groupListShape,"groupListShape");
    PrintVector(quantOutShape,"quantOutShape");
}



    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* perTokenScaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* quantOutDeviceAddr = nullptr;
    void* quantOutScaleDeviceAddr = nullptr;

    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* perTokenScale = nullptr;
    aclTensor* groupList = nullptr;
    aclTensorList* quantOut = nullptr;
    //aclTensor* quantOutScale = nullptr;

    
    std::vector<int8_t> xHostData(M * K, 0);
    std::vector<int8_t> weightHostData(E * N * K , 0);
    std::vector<float> biasHostData(E * N, 0);
    std::vector<uint64_t> scaleHostData(E * N * quantGroupNum, 0);
    std::vector<float> perTokenScaleHostData(M, 0);
    std::vector<int64_t> groupListHostData(E, M/E);
    std::vector<short> quantOutHostData(M * N, 0);
    //std::vector<float> quantOutScaleHostData(M, 0);

    // CHECK_RET(aclrtMallocHost((void **)(&xHostData), xFileSize) == ACL_SUCCESS, return ret);
    // CHECK_RET(aclrtMalloc((void **)&xDeviceAddr, xFileSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return ret);
    //ReadFile("/home/huangjunzhe/a8w4/aclnn/gen_data/x_ori.bin", xShape, xHostData);
    //ReadFile("/home/huangjunzhe/a8w4/aclnn/gen_data/weight.bin", weightShapeFile, weightHostData);
    //ReadFile("/home/huangjunzhe/a8w4/aclnn/gen_data/perTokenScale.bin", perTokenScaleShape, perTokenScaleHostData);
    //ReadFile("/home/huangjunzhe/a8w4/aclnn/gen_data/groupListI64.bin", groupListShape, groupListHostData);
    //ReadFile("/home/huangjunzhe/a8w4/aclnn/gen_data/scale.bin", scaleShape, scaleHostData);
    //ReadFile("/home/huangjunzhe/a8w4/aclnn/gen_data/bias.bin", biasShape, biasHostData);

#ifndef NOFILE
    std::cout << "[test_gmms.cpp] Reading files... " << std::endl;
    ReadFile(data_path+"/a8w4_nz_test_x.bin", xShape, xHostData);
    if (isNZ) ReadFile(data_path+"/a8w4_nz_test_weight_nz.bin", weightShapeFile, weightHostData);
    else ReadFile(data_path+"/a8w4_nz_test_weight_nd.bin", weightShapeFile, weightHostData);

    ReadFile(data_path+"/a8w4_nz_test_bias.bin", biasShape, biasHostData);
//    std::cout << "bias: " << biasHostData[0] << " " << biasHostData[1] << " " << biasHostData[2] << " " << biasHostData[3] << " " << biasHostData[4] << " " << biasHostData[5] << " \n";

    ReadFile(data_path+"/a8w4_nz_test_scale.bin", scaleShape, scaleHostData);

    ReadFile(data_path+"/a8w4_nz_test_group_list.bin", groupListShape, groupListHostData);
    ReadFile(data_path+"/a8w4_nz_test_per_token_scale.bin", perTokenScaleShape, perTokenScaleHostData);
#else
    std::cout << "[test_gmms.cpp] Will not read file. Default M per group: " << M/E << std::endl;
#endif

#ifdef GENERATED_CASES
    ReadFile(argv[10], xShape, xHostData);
    if (isNZ) ReadFile(argv[11], weightShapeFile, weightHostData);
    else ReadFile(argv[11], weightShapeFile, weightHostData);

    ReadFile(argv[12], biasShape, biasHostData);
//    std::cout << "bias: " << biasHostData[0] << " " << biasHostData[1] << " " << biasHostData[2] << " " << biasHostData[3] << " " << biasHostData[4] << " " << biasHostData[5] << " \n";

    ReadFile(argv[13], scaleShape, scaleHostData);

    ReadFile(argv[14], groupListShape, groupListHostData);
    ReadFile(argv[15], perTokenScaleShape, perTokenScaleHostData);
#endif
//std::cout <<  "ww " << (int)(weightHostData[0]) << " " << int(weightHostData[1]) << std::endl;
//std::cout <<  "xx " << (int)(xHostData[0]) << " " << int(xHostData[1]) << std::endl;
    // 创建x aclTensor
    //ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    ret = CreateAclTensorList<int8_t>({ xHostData }, { xShape } , &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    if (isNZ) {
        ret = CreateAclTensorListNz<int8_t>({weightHostData}, {weightShape}, &weightDeviceAddr, aclDataType::ACL_INT4, &weight);
    } else {
        ret = CreateAclTensorList<int8_t>({weightHostData}, {weightShape}, &weightDeviceAddr, aclDataType::ACL_INT4, &weight);
    }
    
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensorList<float>({biasHostData}, {biasShape}, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建Scale aclTensor
    ret = CreateAclTensorList<uint64_t>({scaleHostData}, {scaleShape}, &scaleDeviceAddr, aclDataType::ACL_UINT64, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建perTokenScale aclTensor
    ret = CreateAclTensorList<float>({perTokenScaleHostData}, {perTokenScaleShape}, &perTokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantOut aclTensor
    ret = CreateAclTensorList<short>({quantOutHostData}, {quantOutShape}, &quantOutDeviceAddr, isBF16 == 1 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16, &quantOut);
    //ret = CreateAclTensorList<short>({quantOutHostData}, {quantOutShape}, &quantOutDeviceAddr, aclDataType::ACL_FLOAT16, &quantOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantOutScale aclTensor
//    ret = CreateAclTensor(quantOutScaleHostData, quantOutScaleShape, &quantOutScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &quantOutScale);
//    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // -----------调用CANN算子库API----------------
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnAdd第一段接口
//    // 创建tuningconfig aclIntArray
    std::vector<int64_t> tuningConfigData = {tuningNum};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);
    std::cout << "[test_gmms.cpp] Entering aclnnGroupedMatmulV5GetWorkspaceSize\n";
    ret = aclnnGroupedMatmulV5GetWorkspaceSize(x, weight, bias, scale, nullptr, nullptr, nullptr, perTokenScale, groupList,
                 nullptr, nullptr, nullptr, 3, 0, 1, 0, tuningConfig, quantOut, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAdd第二段接口
//    ret = aclnnGroupedMatmulSwiglu(workspaceAddr, workspaceSize, executor, stream);
std::cout << "[test_gmms.cpp] Entering aclnnGroupedMatmulV5\n";
    ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(quantOutShape);
    std::vector<short> out1Data(size, 0);
std::cout << "[test_gmms.cpp] GetShapeSize(quantOutShape ) " << size << ", out1Data.size() * sizeof(out1Data[0]): " << out1Data.size() * sizeof(out1Data[0]) << ", size * sizeof(out1Data[0]): " << size * sizeof(out1Data[0]) << std::endl;
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), quantOutDeviceAddr,
                        size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
std::cout << "[test_gmms.cpp] aclrtMemcpy = " << ret << ", sleeping..." << std::endl;
//    sleep(10);
#ifndef NOFILE
    if(size < 2147479552 / sizeof(short)) WriteFile(data_path+"/a8w4_nz_test_result.bin", size, out1Data);
    else WriteLargeFile(data_path+"/a8w4_nz_test_result.bin", size, out1Data);
#else
    std::cout << "[test_gmms.cpp] will not write result bin\n";
#endif
    size = GetShapeSize(quantOutShape);

//    std::vector<float> out2Data(size, 0);
//    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), quantOutScaleDeviceAddr,
//                        size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//    WriteFile("/home/huangjunzhe/a8w4/aclnn/gen_data/out2.bin", size, out2Data);

    // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(scale);
    aclDestroyTensorList(perTokenScale);
    aclDestroyTensor(groupList);
    aclDestroyTensorList(quantOut);


    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(perTokenScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(quantOutDeviceAddr);
    //aclrtFree(quantOutScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

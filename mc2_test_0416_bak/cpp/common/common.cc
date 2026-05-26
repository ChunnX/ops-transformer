#include "common.h"
std::map <std::string, aclDataType> g_data_type_map = {
    {"fp16",   ACL_FLOAT16},
    {"bf16",   ACL_BF16},
    {"fp32",   ACL_FLOAT},
    {"fp64",   ACL_DOUBLE},
    {"int4",   ACL_INT4},
    {"int8",   ACL_INT8},
    {"uint8",   ACL_UINT8},
    {"int16",  ACL_INT16},
    {"uint16",  ACL_UINT16},
    {"uint32", ACL_UINT32},
    {"int32",  ACL_INT32},
    {"int64",  ACL_INT64},
#ifdef DAVID
    {"hif8",  ACL_HIFLOAT8},
    {"fp8_e5m2",  ACL_FLOAT8_E5M2},
    {"fp8_e4m3fn",  ACL_FLOAT8_E4M3FN},
    {"fp8_e8m0",  ACL_FLOAT8_E8M0},
    {"fp4_e2m1",  ACL_FLOAT4_E2M1},
    {"fp4_e1m2",  ACL_FLOAT4_E1M2},
#endif
    {"uint64", ACL_UINT64}
};

aclDataType checkAndGetDataType(const std::string& dtype) {
    aclDataType res = ACL_DT_UNDEFINED;
    auto it = g_data_type_map.find(dtype);
    if (it == g_data_type_map.end()) {
        // 找不到key
        std::cerr << "-----g_data_type_map [ERROR] 数据类型校验失败：找不到目标dtype = " << dtype << std::endl;
        std::cerr << "[INFO] 目前支持的dtype总数：" << g_data_type_map.size() << " 种" << std::endl;
        return res;
    }
    return it->second;
}

std::map<std::string, HcclDataType> dtype_to_HcclDataType = {
    {"fp32",  HCCL_DATA_TYPE_FP32},
    {"fp16",  HCCL_DATA_TYPE_FP16},
    {"bf16",  HCCL_DATA_TYPE_BFP16},
    {"fp64",  HCCL_DATA_TYPE_FP64},
    {"int8",  HCCL_DATA_TYPE_INT8},
    {"int16", HCCL_DATA_TYPE_INT16},
    {"int32", HCCL_DATA_TYPE_INT32},
    {"int64", HCCL_DATA_TYPE_INT8},
    {"uint8", HCCL_DATA_TYPE_UINT8},
    {"uint16", HCCL_DATA_TYPE_UINT16},
    {"uint32", HCCL_DATA_TYPE_UINT32},
#ifdef DAVID
    {"hif8",  HCCL_DATA_TYPE_HIF8},
    {"fp8_e5m2",  HCCL_DATA_TYPE_FP8E5M2},
    {"fp8_e4m3fn",  HCCL_DATA_TYPE_FP8E4M3},
#endif
    {"uint64", HCCL_DATA_TYPE_UINT64}
};

using namespace std;
void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

void split(const std::string &str, std::vector<std::string> &tokens,const char delim ) {
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

bool stringToBool(const std::string& str) {
    std::istringstream iss(str);
    bool result;
    iss >> std::boolalpha >> result;
    return result;
}

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
    } else if (dtype == "fp64") {
        ret = sizeof(int64_t);
    } else if (dtype == "uint32") {
        ret = sizeof(int32_t);
    } else if (dtype == "uint16") {
        ret = sizeof(int16_t);
    } else if (dtype == "uint8") {
        ret = sizeof(int8_t);
    } else if ((dtype == "hif8") || (dtype == "fp8_e5m2") || (dtype == "fp8_e4m3fn")||(dtype=="fp8_e8m0")) {
        ret = sizeof(int8_t);
    }else if((dtype=="fp4_e2m1")||(dtype =="fp4_e1m2")){
        ret = sizeof(int8_t); //使用的地方再除2
    }  
    else {
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
    std::vector<int64_t> storageShape = shape; // 是否能传二维
    int offset = 0;
    *tensor = aclCreateTensor(shape.data(), shape.size(), checkAndGetDataType(dtype), strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}


int CreateWeightTensorWithDeviceData(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr,
                                     aclTensor **tensor, std::string &dtype, void **deviceAddr_old) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, checkAndGetDataType(dtype), &size);
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), checkAndGetDataType(dtype), strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int CreateTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                 std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name, bool isTransB) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    auto tensorSize = size * dtype_to_sizeof(dtype);
    if (tensorSize == 0){
        tensorSize = 1;
    }
    if (dtype == "int4"||dtype == "fp4_e2m1"||dtype == "fp4_e1m2") {
        tensorSize = static_cast<uint64_t>(ceil(tensorSize / 2));
    }
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    void *tensorHostData = nullptr;
    // 用malloc规避aclmalloc失败问题
    #ifdef DAVID_FAST_MODEL
    tensorHostData = malloc(tensorSize);
    #else
    ACL_CHECK(aclrtMallocHost((void **) &tensorHostData, tensorSize));
    #endif

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
    #ifdef DAVID_FAST_MODEL
    free(tensorHostData);
    #else 
    ACL_CHECK(aclrtFreeHost(tensorHostData));
    #endif
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
    // shape [k,n]; storage shape[n,k]
    *tensor = aclCreateTensor(shape.data(), shape.size(), checkAndGetDataType(dtype), strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int CreateWeightTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                       std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, checkAndGetDataType(dtype), &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    auto tensorSize = size * dtype_to_sizeof(dtype);;
    if (dtype == "int4"||dtype == "fp4_e2m1"||dtype == "fp4_e1m2") {
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
        ReadFile(bin_path + "/input/" + std::to_string(rankId) + "/" + "packed_" + name + ".bin", tensorSize,
                 tensorHostData, tensorSize);
    } else {
        ReadFile(bin_path + "/input/" + std::to_string(rankId) + "/" + name + ".bin", tensorSize,
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), checkAndGetDataType(dtype), strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int init(Args & args,int n_dev,int rankId)
{
    int ret;
    int32_t devices[n_dev];
    for (int i = 0; i < n_dev; i++) {
        devices[i] = i;
    }
    ACL_CHECK(aclInit(nullptr));
    // 启动多线程
    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return -1);
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return -1);
    // 初始化集合通信域
    LOG_PRINT("before HcclRootInfo");
    HcclRootInfo *c = new HcclRootInfo;
    for (int i=0; i<10; i++) {
        c->internal[i] = 'a';
    }
    c->internal[10] = '\0';
    LOG_PRINT("before HcclCommInitRootInfo n_dev: %d, rankId: %d\n", n_dev, rankId);
    ret = HcclCommInitRootInfo(n_dev, c, rankId, &comms);
    LOG_PRINT("after HcclCommInitRootInfo\n");
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    return 0;
}

int saveOutPut(string &bin_path,uint64_t out_size,int rankId,void* out_device_Addr,string name)
{
    unsigned long long compare_size = out_size;
    if (compare_size == 0){compare_size = 1;} 
    void *host_buff = nullptr;
    #ifdef DAVID_FAST_MODEL
    host_buff = malloc(compare_size);
    #else
     ACL_CHECK(aclrtMallocHost((void **) &host_buff, compare_size));
    #endif
    ACL_CHECK(aclrtMemcpy((void *) host_buff, compare_size, out_device_Addr, compare_size,ACL_MEMCPY_DEVICE_TO_HOST));
    writeBin(bin_path + "/output/" + to_string(rankId) + "/" + name, host_buff, compare_size);
    if (host_buff != nullptr) {
    #ifdef DAVID_FAST_MODEL
        free(host_buff);
    #else
        ACL_CHECK(aclrtFreeHost(host_buff));
    #endif 
    }
    return 0;
}

int DestoryRes(Args & args)
{   
    ACL_CHECK(aclrtDestroyStreamForce(args.stream));
    ACL_CHECK(HcclCommDestroy(args.hcclComm));
    ACL_CHECK(aclrtDestroyContext(args.context));
    ACL_CHECK(aclrtResetDevice(args.rankId));
    return 0;
}

int create_u64_dequant_scale(vector<int64_t>shape,const aclTensor*dequant_scale,void** u64_dequant_scale_device_addr,aclTensor**u64_dequant_scale,aclrtStream&stream)
{
    //quantmatmulv5 用到的 dequantscale，转uint64
    int ret = 0;
    uint64_t u64_trans_workspaceSize = 0;
    void* u64_trans_WorkspaceAddr = nullptr;
    string trans_dequant_scale_dtype = "uint64";
    aclOpExecutor *executor = nullptr;
    
    CreateTensorWithoutHostData(shape, aclFormat::ACL_FORMAT_ND, u64_dequant_scale_device_addr,u64_dequant_scale, trans_dequant_scale_dtype);
    ret = aclnnTransQuantParamV2GetWorkspaceSize(dequant_scale, nullptr, *u64_dequant_scale, &u64_trans_workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (u64_trans_workspaceSize > 0) {
        ret = aclrtMalloc(&u64_trans_WorkspaceAddr, u64_trans_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate transParamWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnTransQuantParamV2第二段接口
    ret = aclnnTransQuantParamV2(u64_trans_WorkspaceAddr, u64_trans_workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, 200000000));
    if(u64_trans_workspaceSize > 0 ){aclrtFree(u64_trans_WorkspaceAddr);}

    return 0;
}

int CreateIntArrayFromFile( vector<int64_t>& host_data,aclIntArray** int_array,std::string bin_path,std::string name,uint32_t &rankId,std::string type) 
{
  std::string path = bin_path + "/input/" + to_string(rankId)+ "/" +name +".bin";
  FILE* fp = fopen(path.c_str(),"r");
  if(fp == NULL ){printf("[ERROR] fopen file %s failed \n",path.c_str());return -1;}
  int  ret;
  if(type == "int32")
  {   
      vector<int32_t>vec_int32 (host_data.size(),0);
	  ret = fread(vec_int32.data(),host_data.size()*sizeof(int32_t),1,fp);
	  for (int i = 0;i<host_data.size();i++ ) {
          host_data[i]= (static_cast<int64_t>(vec_int32[i]));
      }
  }
  else
  {
	  ret = fread(host_data.data(),host_data.size()*sizeof(int64_t),1,fp);
  }
  
  *int_array = aclCreateIntArray(host_data.data(), host_data.size());
  
  fclose(fp);
  return 0;
}
int CreateTensor_without_flag(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                 std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));
    auto tensorSize = size * dtype_to_sizeof(dtype);
    if (dtype == "int4") {
        tensorSize = static_cast<uint64_t>(ceil(tensorSize / 2));
    }
    // 调用aclrtMalloc申请device侧内存
    ACL_CHECK(aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    void *tensorHostData = nullptr;
    ACL_CHECK(aclrtMallocHost((void **) &tensorHostData, tensorSize));
    string path_anme = bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin";
    //ReadFile(bin_path + "/input/" + to_string(rankId) + "/" + name + ".bin", tensorSize,tensorHostData, tensorSize);
    FILE * fp = fopen(path_anme.c_str(),"rb");
    if(fp == NULL )
    {   std::cout << " open" <<path_anme << "failed" <<std::endl;
        return -1;
    }
    auto read_size = fread(tensorHostData,tensorSize,1,fp);
    fclose(fp);
    auto ret = aclrtMemcpy(*deviceAddr, tensorSize, tensorHostData, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), checkAndGetDataType(dtype), strides.data(), offset, format,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}


#ifndef __MC2_NEW_TEST_COMMON_H__
#define __MC2_NEW_TEST_COMMON_H__
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
#include "aclnn/acl_meta.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"

using namespace std;

typedef struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;

}Args;

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

#define CHECK_FREE_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      Finalize(args.rankId, stream);\
      return_expr;                   \
    }                                \
  } while (0)

extern std::map <std::string, aclDataType> g_data_type_map;
extern std::map<std::string, HcclDataType> dtype_to_HcclDataType ;

void Finalize(int32_t deviceId, aclrtStream stream) ;
void split(const std::string &str, std::vector<std::string> &tokens,const char delim = '/');
bool stringToBool(const std::string& str);
size_t dtype_to_sizeof(std::string &dtype);
int64_t GetShapeSize(const std::vector<int64_t> &shape);
bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize) ;
void writeBin(std::string path, void *buf, size_t size);
int32_t as_uint(const float x);
float as_float(const int32_t x);
float float16_to_float(const int16_t x) ;
int CreateTensorWithoutHostData(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr,aclTensor **tensor, std::string &dtype) ;
int CreateWeightTensorWithDeviceData(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr,
                                     aclTensor **tensor, std::string &dtype, void **deviceAddr_old);
int CreateTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                 std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name, bool isTransB = false) ;
int CreateWeightTensor(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                       std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name) ;
int init(Args & args,int n_dev,int rankId);
int saveOutPut(string &bin_path,uint64_t out_size,int rankId,void* out_device_Addr,string name);
int DestoryRes(Args & args);
int create_u64_dequant_scale(vector<int64_t>shape,const aclTensor*dequant_scale,void** u64_dequant_scale_device_addr,aclTensor**u64_dequant_scale,aclrtStream&stream);
int CreateIntArrayFromFile( vector<int64_t>& host_data,aclIntArray** int_array,std::string bin_path,std::string name,uint32_t &rankId,std::string type);
int CreateTensor_without_flag(const std::vector<int64_t> &shape, aclFormat format, void **deviceAddr, aclTensor **tensor,
                 std::string &dtype, std::string &bin_path, uint32_t &rankId, const std::string &name);
#endif

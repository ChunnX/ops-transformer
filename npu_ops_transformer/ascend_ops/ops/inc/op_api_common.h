/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
#define TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_

#include <fstream>
#include <sys/stat.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <acl/acl_base.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "op_log.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

typedef aclTensor *(*_aclCreateTensor)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t *stride, int64_t offset, aclFormat format,
                                       const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value, uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value, uint64_t size);
typedef aclScalarList *(*_aclCreateScalarList)(const aclScalar *const *value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);
typedef int (*_aclDestroyScalarList)(const aclScalarList *array);

using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

constexpr int OWNER_ROOT_UID = 0;

inline void warn_(const ::c10::Warning& warning)
{
}

static std::vector<std::string> split_str(std::string s, const std::string &del)
{
    size_t end = s.find(del);
    std::vector<std::string> path_list;
    while (end != std::string::npos) {
        path_list.push_back(s.substr(0, end));
        s.erase(s.begin(), s.begin() + end + 1);
        end = s.find(del);
    }
    path_list.push_back(s);
    return path_list;
}

static bool is_file_exist(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
}

inline std::string real_path(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
        return "";
    }
    return std::string(realPath);
}

inline std::vector<std::string> get_custom_lib_path()
{
    auto ascend_custom_opppath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    std::vector<std::string> custom_lib_path_list;

    if (ascend_custom_opppath == nullptr) {
        OP_LOGW("ASCEND_CUSTOM_OPP_PATH does not exist");
        return std::vector<std::string>();
    }

    std::string ascend_custom_opppath_str(ascend_custom_opppath);
    // split string with ":"
    custom_lib_path_list = split_str(ascend_custom_opppath_str, ":");
    if (custom_lib_path_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : custom_lib_path_list) {
        it = it + "/op_api/lib/";
    }

    return custom_lib_path_list;
}

inline std::vector<std::string> get_default_custom_lib_path()
{
    auto ascend_opp_path = std::getenv("ASCEND_OPP_PATH");
    std::vector<std::string> default_vendors_list;

    if (ascend_opp_path == nullptr) {
        OP_LOGW("ASCEND_OPP_PATH does not exist");
        return std::vector<std::string>();
    }

    std::string vendors_path(ascend_opp_path);
    vendors_path = vendors_path + "/vendors";
    std::string vendors_config_file = real_path(vendors_path + "/config.ini");
    if (vendors_config_file.empty()) {
        OP_LOGW("config.ini does not exist");
        return std::vector<std::string>();
    }

    if (!is_file_exist(vendors_config_file)) {
        OP_LOGW("config.ini does not exist or the path length is more than %d", PATH_MAX);
        return std::vector<std::string>();
    }

    std::ifstream ifs(vendors_config_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            break;
        }
    }
    std::string head = "load_priority=";
    line.erase(0, head.length());

    // split string with ","
    default_vendors_list = split_str(line, ",");
    if (default_vendors_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : default_vendors_list) {
        it = real_path(vendors_path + "/" + it + "/op_api/lib/");
    }

    return default_vendors_list;
}

inline std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
inline std::vector<std::string> g_default_custom_lib_path = get_default_custom_lib_path();

inline bool checkOwner(std::string cusLibPath)
{
    struct stat fileInfo;
    stat(cusLibPath.c_str(), &fileInfo);
    auto cusLibOwnerUid = fileInfo.st_uid;
    auto curOwnerUid = getuid();
    if (curOwnerUid != OWNER_ROOT_UID && cusLibOwnerUid == OWNER_ROOT_UID) {
        TORCH_NPU_WARN_ONCE("A common user is using the files of the root user.");
        return true;
    } else if ((curOwnerUid == OWNER_ROOT_UID && cusLibOwnerUid != OWNER_ROOT_UID) ||
            (curOwnerUid != OWNER_ROOT_UID && (curOwnerUid != cusLibOwnerUid))) {
        TORCH_NPU_WARN_ONCE("The ", cusLibPath,
            " owner does not match current owner or the root user is using the files of a common user, will skip this file.");
        return false;
    }
    return true;
}

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        OP_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        OP_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

#define GET_OP_API_FUNC_FROM_FEATURE_LIB(lib_handler, lib_name)                                                        \
    do {                                                                                                               \
        static auto lib_handler = GetOpApiLibHandler(lib_name);                                                        \
        if ((lib_handler) != nullptr) {                                                                                \
            auto funcAddr = GetOpApiFuncAddrInLib(lib_handler, lib_name, api_name);                                    \
            if (funcAddr != nullptr) {                                                                                 \
                return funcAddr;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    } while (false)

inline void *GetOpApiFuncAddrFromFeatureLib(const char *api_name)
{
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_infer_handler, "libaclnn_ops_infer.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_train_handler, "libaclnn_ops_train.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(math_handler, "libaclnn_math.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(sparse_handler, "libaclnn_sparse.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(fft_handler, "libaclnn_fft.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(rand_handler, "libaclnn_rand.so");
    return nullptr;
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    if (!g_custom_lib_path.empty()) {
        for (auto &it : g_custom_lib_path) {
            auto cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    if (!checkOwner(cust_opapi_lib)) {
                        continue;
                    }
                    OP_LOGI("%s is found in %s.", apiName, cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        OP_LOGI("%s is not in custom lib.", apiName);
    }

    if (!g_default_custom_lib_path.empty()) {
        for (auto &it : g_default_custom_lib_path) {
            auto default_cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (default_cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(default_cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    if (!checkOwner(default_cust_opapi_lib)) {
                        continue;
                    }
                    OP_LOGI("%s is found in %s.", apiName, default_cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        OP_LOGI("%s is not in default custom lib.", apiName);
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return GetOpApiFuncAddrFromFeatureLib(apiName);
}

#endif //  TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/torch.h>
#include <torch/all.h>
#include <torch/library.h>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <future>
#include <atomic>
#include <memory>
#include <chrono>
#include <sstream>
#include <ATen/Operators.h>
#include <ATen/ATen.h>
#include "acl/acl.h"
#include "op_log.h"
#include "op_api_common.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "stdio.h"

namespace ascend_ops {

namespace Reset {
const int64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
using aclnn_get_func = int32_t (*)(void *, const uint64_t, aclOpExecutor *, aclrtStream);
using aclnn_reset_get_workspace_func = int32_t (*)(const aclTensor *, const char *, const int, const int, uint64_t *, aclOpExecutor**);

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, message, ##__VA_ARGS__);  \
  } while (0)

class StreamCtxSingle {
private:
    aclrtContext collectCtx;
    aclrtStream collectStream;
    void* total_workspace_addr = nullptr;
    int device_id;

    StreamCtxSingle(int dev_id) : device_id(dev_id) {
        aclrtCreateContext(&collectCtx, device_id);
        aclrtSetCurrentContext(collectCtx);
        aclrtCreateStream(&collectStream);
        int64_t total_workspace_size = WORK_SPACE_SIZE;
        auto ret = aclrtMalloc(&total_workspace_addr, total_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR");
    }
    StreamCtxSingle(const StreamCtxSingle&) = delete;
    StreamCtxSingle& operator = (const StreamCtxSingle&) = delete;
public:
    ~StreamCtxSingle() {
        if (total_workspace_addr != nullptr) {
            aclrtFree(total_workspace_addr);
            total_workspace_addr = nullptr;
        }
    }
    static StreamCtxSingle& getInstance(int dev_id) {
        static StreamCtxSingle single_instance(dev_id);
        return single_instance;
    }
    aclrtContext GetCollectCtx() {
        return collectCtx;
    }
    aclrtStream GetCollectStream() {
        return collectStream;
    }

    void* GetWorkSpaceAddr() {
        return (void*)total_workspace_addr;
    }
};

static bool CleanAndReset(StreamCtxSingle& streamCtx, int64_t ep_world_size, c10::string_view group_detection, const at::Tensor &input_elastic_rank, 
                          at::Tensor &output_elastic_rank_npu, int64_t detection_world_size, int64_t needSync)
{
    aclrtSetCurrentContext(streamCtx.GetCollectCtx());
    aclrtStream collect_stream = streamCtx.GetCollectStream();
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    aclrtMemcpy(output_elastic_rank_npu.data_ptr(), detection_world_size * sizeof(int32_t),
        input_elastic_rank.data_ptr(), detection_world_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    static const auto resetGetWorkspaceSizeFuncAddr =
    reinterpret_cast<aclnn_reset_get_workspace_func>(GetOpApiFuncAddr("aclnnMoeDistributeBufferResetGetWorkspaceSize"));
    static const auto resetOpApiFuncAddr = reinterpret_cast<aclnn_get_func>(GetOpApiFuncAddr("aclnnMoeDistributeBufferReset"));
    TORCH_CHECK(resetGetWorkspaceSizeFuncAddr != nullptr && resetOpApiFuncAddr != nullptr,
            "aclnnMoeDistributeBufferResetGetWorkspaceSize or aclnnMoeDistributeBufferReset symbol not found.");
    const int64_t stride = 1;
    const int64_t storage_dims = ep_world_size;
    const int64_t view_dims = ep_world_size;
    aclTensor *elastic_rank_tensor = aclCreateTensor(&view_dims, 1, ACL_INT32, &stride, 0,
            ACL_FORMAT_ND, &storage_dims, 1, output_elastic_rank_npu.data_ptr());

    uint64_t detect_workspace_size = 0;
    void *detect_workspace_addr = nullptr;
    aclOpExecutor *detect_executor = nullptr;
    resetGetWorkspaceSizeFuncAddr(elastic_rank_tensor, const_cast<char*>(group_detection.data()),
        detection_world_size, needSync, &detect_workspace_size, &detect_executor);
    if (detect_workspace_size > 0) {
        detect_workspace_addr = streamCtx.GetWorkSpaceAddr();
    }
    resetOpApiFuncAddr(detect_workspace_addr, detect_workspace_size, detect_executor, collect_stream);
    aclError errorSync = aclrtSynchronizeStreamWithTimeout(collect_stream, 10000);
    if (errorSync == ACL_ERROR_NONE) {
        return true;
    }
    return false;
}

bool ResetThreadFun(StreamCtxSingle& streamCtx, const at::Tensor &input_elastic_rank, at::Tensor &output_elastic_rank_npu,
    int64_t ep_world_size, c10::string_view group_detection, int64_t detection_world_size, int64_t needSync) 
{
    auto start_time = std::chrono::high_resolution_clock::now();
    OP_DFX_LOGI("Start Reset.");
    bool execution_completed = CleanAndReset(streamCtx, ep_world_size, group_detection, input_elastic_rank, output_elastic_rank_npu, detection_world_size, needSync);
    auto clean_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(clean_time - start_time);
    OP_DFX_LOGI("Total time cost is %dms", total_duration);
    return execution_completed;
}

bool reset_npu(const at::Tensor &detection_result, int64_t ep_world_size, int64_t ep_rank_id,
               c10::string_view group_detection, int64_t detection_world_size, int64_t detection_rank_id, int64_t needSync) {
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
              "ep_rank_id should be in [0, ep_world_size), but got",
              " ep_world_size: ", ep_world_size, ", ep_rank_id: ", ep_rank_id);
    TORCH_CHECK(
        (detection_rank_id >= 0) && (detection_rank_id < detection_world_size),
        "detection_rank_id should be in [0, detection_world_size), but got",
        " detection_world_size: ", detection_world_size,
        ", detection_rank_id: ", detection_rank_id);

    at::Tensor output_elastic_rank_npu = at::empty({detection_world_size}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
        .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor elastic_rank_table = at::empty({detection_world_size * detection_world_size}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
        .memory_format(c10::MemoryFormat::Contiguous));

    StreamCtxSingle& streamCtx = StreamCtxSingle::getInstance(static_cast<int>(elastic_rank_table.device().index()));
    bool execution_flag = ResetThreadFun(streamCtx, detection_result, output_elastic_rank_npu, 
                                         ep_world_size, group_detection, detection_world_size, needSync);
    return execution_flag;
}

// Register Ascend implementations for isfinite
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("reset", reset_npu);
}

} // namespace BufferReset
} // namespace ascend_ops
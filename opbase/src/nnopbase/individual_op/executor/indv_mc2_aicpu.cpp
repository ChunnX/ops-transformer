/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_mc2_aicpu.h"
#include "utils/thread_var_container.h"
#include "runtime/runtime/rt_model.h"
#include "utils/indv_soc.h"

#ifdef __cplusplus
extern "C" {
#endif

const NnopbaseUChar NNOPBASE_MC2_AICPU_SO_NAME[NNOPBAE_AICPU_PARAM_LEN] = {"libccl_kernel.so"};
const NnopbaseUChar NNOPBASE_MC2_AICPU_KERNEL_NAME[NNOPBAE_AICPU_PARAM_LEN] = {"RunAicpuKfcSrvLaunch"};
const NnopbaseUChar NNOPBASE_MC2_SERVER_SO_NAME[NNOPBAE_AICPU_PARAM_LEN] = {"libmc2_server.so"};
const NnopbaseUChar NNOPBASE_MC2_SERVER_KERNEL_NAME[NNOPBAE_AICPU_PARAM_LEN] = {"Mc2ServerKernel"};
constexpr uint8_t NNOPBASE_MC2_NOTIFY_COUNT = 2;
constexpr uint16_t NNOPBASE_HCCL_DEFAULT_TIME = 1836;

namespace {
aclnnStatus DoHcclAllocComResourceByTiling(NnopbaseExecutor *executor, HcclComm comm, void *stream, void *tilingData,
    void **commCtx)
{
    const bool supportA5AiCpu = nnopbase::IndvSoc::GetInstance().NnopbaseSupportA5AiCpu(executor->mc2OpCfg.sType);
    OP_LOGI("Nnopbase MC2 alloc resource route, sType[%d], supportA5AiCpu[%d].",
        executor->mc2OpCfg.sType, supportA5AiCpu);
    if (supportA5AiCpu) {
        return nnopbase::IndvMc2ClientWrapper::GetInstance().HcclAllocComResourceByTiling(comm, stream, tilingData,
            commCtx);
    } else {
        return nnopbase::IndvHcclWrapper::GetInstance().HcclAllocComResourceByTiling(comm, stream, tilingData, commCtx);
    }
}
}

aclnnStatus NnopbaseGetHcomResource(NnopbaseExecutor *executor, rtStream_t const stream)
{
    executor->contextAddr.clear();
    executor->aicpuStream.clear();
    executor->aicpuNotify.clear();
    for (HcclComm commHandle : executor->mc2OpCfg.hcomHandle) {
        void *contextAddr = nullptr;
        if (commHandle == nullptr) {
            executor->contextAddr.push_back(contextAddr);
            executor->aicpuStream.push_back(nullptr);
            executor->aicpuNotify.push_back(std::make_pair(nullptr, nullptr));
            continue;
        }
        if (!executor->hasTiling) {
            // 静态mc2算子场景，用json读取的静态tilingData去激活Hccl通信
            NNOPBASE_ASSERT_OK_RETVAL(DoHcclAllocComResourceByTiling(
                executor,
                commHandle,
                stream,
                executor->args->tilingInfo.staticTilingData.data(),
                &contextAddr));
        } else {
            NNOPBASE_ASSERT_OK_RETVAL(DoHcclAllocComResourceByTiling(
                executor,
                commHandle,
                stream,
                (op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->tilingInfo.tilingData))->GetData(),
                &contextAddr));
        }
        executor->contextAddr.push_back(contextAddr);
        rtStream_t aicpuStream = nullptr;
        rtStream_t notify[NNOPBASE_MC2_NOTIFY_COUNT] = {};
        if (!nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2OpCfg.sType)) {
            NNOPBASE_ASSERT_OK_RETVAL(nnopbase::IndvHcclWrapper::GetInstance().HcclGetAicpuOpStreamAndNotify(
                commHandle, &aicpuStream, NNOPBASE_MC2_NOTIFY_COUNT, notify));
            executor->aicpuStream.push_back(aicpuStream);
            executor->aicpuNotify.push_back(std::make_pair(notify[0], notify[1]));
        }
        OP_LOGI("Executor is %p, stream is %p, commHandle is %p, get contextAddr is %p, get aicpuStream is %p, notify[0] is %p, notify[1] is %p.",
            executor, stream, commHandle, contextAddr, aicpuStream, notify[0], notify[1]);
    }
    return OK;
}

aclnnStatus NnopbaseExecutorGetMc2Num(
    NnopbaseExecutor *executor, rtStream_t const stream, NnopbaseExecutorArgsAddr *argsAddr, uint32_t *mc2Num)
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetHcomResource(executor, stream));
    argsAddr->hcclDesc->version = 1U;
    argsAddr->hcclDesc->groupNum = static_cast<uint64_t>(executor->contextAddr.size());
    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2OpCfg.sType)) {
        *mc2Num = argsAddr->hcclDesc->groupNum;
        executor->fusionArgs.isNoNeedH2DCopy = 0U;
        executor->fusionArgs.hostInputInfoPtr = nullptr;
        executor->fusionArgs.hostInputInfoNum = 0U;
        executor->fusionArgs.aicpuNum = 0U;
    } else {
        *mc2Num = argsAddr->hcclDesc->groupNum + 1U; // 1 is NnopbaseHcclCommParamDesc
        executor->aicpuArgs.kernelOffsetInfoPtr = nullptr;
        executor->aicpuArgs.kernelOffsetInfoNum = 0;
        executor->aicpuArgs.isNoNeedH2DCopy = false;
        executor->aicpuArgs.hostInputInfoPtr = nullptr;
        executor->aicpuArgs.hostInputInfoNum = 0U;
    }
    return OK;
}

aclnnStatus NnopbaseAddCapture(rtStream_t stream, std::vector<rtStream_t> aicpuStream)
{
    aclmdlRICaptureStatus status;
    aclmdlRI captureMdl;
    if ((aclmdlRICaptureGetInfo(stream, &status, &captureMdl) == ACL_SUCCESS) &&
        (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)) {
        for (auto &stm :aicpuStream) {
            if (stm != nullptr) {
                NNOPBASE_ASSERT_RTOK_RETVAL(rtStreamAddToModel(stm, captureMdl));
            }
        }
    }
    return OK;
}

aclnnStatus NnopbaseLaunchKFCTask(NnopbaseExecutor *const executor, rtStream_t stream)
{
    OP_LOGI("Launch kernel by KFC mode.");
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddCapture(stream, executor->aicpuStream));
    if (executor->aicpuStream[0] != nullptr) {
        CHECK_COND(aclrtWaitAndResetNotify(executor->aicpuNotify[0].first, executor->aicpuStream[0], UINT32_MAX) == ACL_SUCCESS,
            ACLNN_ERR_RUNTIME_ERROR,
            "Call aclrtWaitAndResetNotify failed, executor is %p, aicpuStream is %p, stream is %p, Notify is %p.",
            executor,
            executor->aicpuStream[0],
            stream,
            executor->aicpuNotify[0].first);
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtRecordNotify(executor->aicpuNotify[0].first, stream));
    }
    for (size_t i = 1U; i < executor->aicpuStream.size(); i++) {
        if ((executor->aicpuStream[0] != nullptr) && (executor->aicpuNotify[i].first != nullptr)) {
            NNOPBASE_ASSERT_RTOK_RETVAL(aclrtWaitAndResetNotify(executor->aicpuNotify[i].first, executor->aicpuStream[0], UINT32_MAX));
        }
        // aicpuStream和aicpuNotify是hccl同时创建的，aicpuStream不是null，aicpuNotify也不应该是null
        if (executor->aicpuStream[i] != nullptr) {
            NNOPBASE_ASSERT_RTOK_RETVAL(aclrtRecordNotify(executor->aicpuNotify[i].first, executor->aicpuStream[i]));
        }
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAicpuKernelLaunch(executor));
    for (size_t i = 1U; i < executor->aicpuStream.size(); i++) {
        if (executor->aicpuStream[i] != nullptr) {
            NNOPBASE_ASSERT_RTOK_RETVAL(aclrtWaitAndResetNotify(executor->aicpuNotify[i].second, executor->aicpuStream[i], UINT32_MAX));
        }
        if ((executor->aicpuStream[0] != nullptr) && (executor->aicpuNotify[i].second != nullptr)) {
            NNOPBASE_ASSERT_RTOK_RETVAL(aclrtRecordNotify(executor->aicpuNotify[i].second, executor->aicpuStream[0]));
        }
    }
    OP_LOGI("Launch kernel by KFC mode successfully.");
    return OK;
}

void NnopbaseCopyDavidMC2ParamDesc(NnopbaseExecutor *executor, NnopbaseExecutorArgsAddr *argsAddr)
{
    const NnopbaseUChar *const args = op::internal::PtrCastTo<NnopbaseUChar>(executor->fusionArgs.args);
    const uint16_t kfcArgsFmtOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    executor->fusionArgs.aicpuArgs[0].kfcArgsFmtOffset = kfcArgsFmtOffset / sizeof(void *);
    OP_LOGI("KfcArgsFmtOffset is %u.", kfcArgsFmtOffset);
    for (size_t i = 0U; i < sizeof(NnopbaseHcclCommParamDesc); i++) {
        argsAddr->hostInputData =
            NnopbaseAppend1Byte(argsAddr->hostInputData, (op::internal::PtrCastTo<NnopbaseUChar>(argsAddr->hcclDesc))[i]);
    }
    OP_LOGI("GroupNum is %u, hasffts is %u, tilingOff is %u, isDyn is %lu.",
        argsAddr->hcclDesc->groupNum,
        argsAddr->hcclDesc->hasFfts,
        argsAddr->hcclDesc->tilingOff,
        argsAddr->hcclDesc->isDyn);
    // 3 is soname/kernelname/opname
    executor->fusionArgs.argsSize = kfcArgsFmtOffset +
                                    sizeof(NnopbaseHcclCommParamDesc) + NNOPBASE_PARAM_EXT_LEN +
                                    static_cast<uint32_t>(sizeof(rtHostInputInfo_t));
}

void NnopbaseCopyMC2ParamDesc(NnopbaseExecutor *executor, NnopbaseExecutorArgsAddr *argsAddr)
{
    NnopbaseUChar *descAddr = op::internal::PtrCastTo<NnopbaseUChar>(executor->aicpuArgs.args);
    for (size_t i = 0U; i < sizeof(NnopbaseHcclCommParamDesc); i++) {
        descAddr = NnopbaseAppend1Byte(descAddr, (op::internal::PtrCastTo<NnopbaseUChar>(argsAddr->hcclDesc))[i]);
    }

    OP_LOGI("GroupNum is %u, hasffts is %u, tilingOff is %u, isDyn is %lu.",
        argsAddr->hcclDesc->groupNum,
        argsAddr->hcclDesc->hasFfts,
        argsAddr->hcclDesc->tilingOff,
        argsAddr->hcclDesc->isDyn);

    // 3 is soname/kernelname/opname
    executor->aicpuArgs.argsSize =
        executor->argsExt.argsSize + sizeof(NnopbaseHcclCommParamDesc) + NNOPBAE_AICPU_PARAM_LEN * 3U;
}

void NnopbasePrepareMC2Params(NnopbaseExecutor *executor, NnopbaseExecutorArgsAddr *argsAddr)
{
    NnopbaseUChar *args = nullptr;
    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2OpCfg.sType)) {
        args = (NnopbaseUChar *)executor->fusionArgs.args;
        executor->fusionArgs.aicpuArgs[0].soNameAddrOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    } else {
        args = (NnopbaseUChar *)executor->aicpuArgs.args;
        executor->aicpuArgs.soNameAddrOffset = static_cast<uint32_t>(argsAddr->hostInputData - args);
    }
    
    bool isA5AiCpu = nnopbase::IndvSoc::GetInstance().NnopbaseSupportA5AiCpu(executor->mc2OpCfg.sType);
    const NnopbaseUChar *pSoName = isA5AiCpu ? NNOPBASE_MC2_SERVER_SO_NAME :
                                                                      NNOPBASE_MC2_AICPU_SO_NAME;
    const NnopbaseUChar *pKernelName = isA5AiCpu ? NNOPBASE_MC2_SERVER_KERNEL_NAME :
                                                                          NNOPBASE_MC2_AICPU_KERNEL_NAME;

    for (size_t i = 0; i < NNOPBAE_AICPU_PARAM_LEN; i++) {
        argsAddr->hostInputData = NnopbaseAppend1Byte(argsAddr->hostInputData, pSoName[i]);
    }

    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2OpCfg.sType)) {
        executor->fusionArgs.aicpuArgs[0].kernelNameAddrOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    } else {
        executor->aicpuArgs.kernelNameAddrOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    }

    for (size_t i = 0; i < NNOPBAE_AICPU_PARAM_LEN; i++) {
        argsAddr->hostInputData = NnopbaseAppend1Byte(argsAddr->hostInputData, pKernelName[i]);
    }

    const std::string opName = std::string(executor->opType) + NNOPBAE_MC2_AICPU_SUFFIX;
    OP_CHECK(memcpy_s(argsAddr->hostInputData, opName.length(), &opName[0], opName.length()) == EOK,
        OP_LOGW("Memcpy aicpu opName failed, opName is %s, length %zu.", opName.c_str(), opName.length()),
        return); 

    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2OpCfg.sType)) {
        const size_t opLen = ((opName.length() + 7) / 8) * 8;
        argsAddr->hostInputData += opLen;
        return NnopbaseCopyDavidMC2ParamDesc(executor, argsAddr);
    } else {
        argsAddr->hostInputData += opName.length();
        return NnopbaseCopyMC2ParamDesc(executor, argsAddr);
    }
}

static aclnnStatus NnopbaseGetAicpuTimeout(uint32_t *time)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(time);
    CHECK_COND(aclrtGetOpExecuteTimeout(time) == ACL_RT_SUCCESS, ACLNN_ERR_RUNTIME_ERROR, "Get op timeout failed.");
    const NnopbaseChar *hcclTimeoutEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_EXEC_TIMEOUT, hcclTimeoutEnv);
    uint32_t hcclTimeout = NNOPBASE_HCCL_DEFAULT_TIME;
    if (hcclTimeoutEnv != nullptr) {
        hcclTimeout = std::atoi(hcclTimeoutEnv);
    }
    *time += hcclTimeout;
    OP_LOGI("HcclTimeout is %u, time is %u.", hcclTimeout, *time);
    return OK;
}

aclnnStatus NnopbaseAicpuKernelLaunch(NnopbaseExecutor *const executor)
{
    uint32_t time = 0;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetAicpuTimeout(&time));
    executor->aicpuArgs.timeout = time;
    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    const std::string opType = std::string(executor->opType) + NNOPBAE_MC2_AICPU_SUFFIX;
    const uint32_t numBlocks =
        executor->args->tilingInfo.aicpuNumBlocks == 0U ? 1U : executor->args->tilingInfo.aicpuNumBlocks;
    NNOPBASE_ASSERT_RTOK_RETVAL(rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC,
        &opType[0],
        numBlocks,
        &executor->aicpuArgs,
        nullptr,
        executor->aicpuStream[0],
        RT_KERNEL_USE_SPECIAL_TIMEOUT));
    OP_LOGI("%s launch successfully, numBlocks is %u.", opType.c_str(), numBlocks);

    NnopbaseInnerReportLaunchInfo(launchBeginTime, executor->aicpuItemId);
    NnopbaseReportAicpuAdditionInfo(launchBeginTime + 1, &opType[0]);
    NnopbaseReportCacheOpInfo(executor, 0U, MSPROF_GE_TASK_TYPE_AI_CPU, executor->aicpuStream[0]);
    return OK;
}

aclnnStatus NnopbaseFusionKernelLaunch(NnopbaseExecutor *const executor, rtStream_t const stream)
{
    OP_LOGI("Launch kernel by fusion mode.");
    uint32_t numBlocks = executor->args->tilingInfo.numBlocks;
    uint64_t tilingKey = executor->args->tilingInfo.tilingKey;
    OP_LOGI("numBlocks is %u, tilingKey is %lu.", numBlocks, tilingKey);

    rtLaunchAttribute_t launchAttr[1];
    launchAttr[0].id = RT_LAUNCH_ATTRIBUTE_BLOCKDIM;
    launchAttr[0].value.blockDim = numBlocks;
    rtLaunchConfig_t launchCfg = {launchAttr, 1U};
    rtAicoreFusionInfo_t aicoreInfo = {executor->args->binInfo->ccuBinHandle, tilingKey, &launchCfg, nullptr};
    rtFunsionTaskInfo_t fusionTaskInfo = {};
    rtCcuTaskGroup_t ccuTaskGroup = {};
    NNOPBASE_ASSERT_RTOK_RETVAL(nnopbase::IndvHcclWrapper::GetInstance().HcclGetCcuTaskInfo(
        executor->mc2OpCfg.hcomHandle[0], (op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->tilingInfo.tilingData))->GetData(), &ccuTaskGroup));
    fusionTaskInfo.subTask[0].type = RT_FUSION_CCU;
    fusionTaskInfo.subTask[0].task.ccuInfo = ccuTaskGroup;
    fusionTaskInfo.subTaskNum = 2U;
    fusionTaskInfo.subTask[1].type = RT_FUSION_AICORE;
    fusionTaskInfo.subTask[1].task.aicoreInfo = aicoreInfo;
    OP_LOGI("SubTaskNum is %u. subTask[0].type is %d, subTask[1].type is %d.",
        fusionTaskInfo.subTaskNum,
        fusionTaskInfo.subTask[0].type,
        fusionTaskInfo.subTask[1].type);

    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    NNOPBASE_ASSERT_RTOK_RETVAL(rtFusionLaunch(&fusionTaskInfo, stream, &executor->fusionArgs));
    NnopbaseExecutorReportProfiling(executor, numBlocks, MSPROF_GE_TASK_TYPE_FUSION, launchBeginTime, stream);
    OP_LOGI("Op %s fusion launch successfully.", executor->opType);
    return OK;
}

static aclnnStatus NnopbaseMC2KernelMTE(NnopbaseExecutor *executor, rtStream_t stream)
{
    return NnopbaseExecutorKernelLaunch(executor, stream);
}

static aclnnStatus NnopbaseMC2KernelCCU(NnopbaseExecutor *const executor, rtStream_t stream)
{
    return NnopbaseFusionKernelLaunch(executor, stream);
}

static aclnnStatus NnopbaseMC2KernelAicpu(NnopbaseExecutor *executor, rtStream_t stream)
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseLaunchKFCTask(executor, stream));
    return NnopbaseExecutorKernelLaunch(executor, stream);
}

aclnnStatus NnopbaseMC2KernelLaunch(NnopbaseExecutor *executor, rtStream_t stream)
{
    OP_LOGI("HcclServerType is %d.", executor->mc2OpCfg.sType);
    if (NnopbaseSkipKernelLaunch(executor)) {
        OP_LOGI("Output of MC2 operator is empty and ZeroEleOutputLaunch is not enabled, skipping launch.");
        return OK;
    }
    if (executor->mc2OpCfg.sType == NNOPBASE_HCCL_SERVER_TYPE_END) { // default
        if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2OpCfg.sType)) {
            return NnopbaseMC2KernelCCU(executor, stream);
        } else {
            return NnopbaseMC2KernelAicpu(executor, stream);
        }
    } else {
        switch (executor->mc2OpCfg.sType) {
            case NNOPBASE_HCCL_SERVER_TYPE_MTE:
                return NnopbaseMC2KernelMTE(executor, stream);
            case NNOPBASE_HCCL_SERVER_TYPE_CCU:
                return NnopbaseMC2KernelCCU(executor, stream);
            default:
                return NnopbaseMC2KernelAicpu(executor, stream);
        }
    }
}

#ifdef __cplusplus
}
#endif

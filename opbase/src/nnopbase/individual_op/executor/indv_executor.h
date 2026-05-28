/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_EXECUTOR_H_
#define INDV_EXECUTOR_H_

#include <vector>
#include <mutex>
#include "dump/adump_pub.h"
#include "individual_op_api.h"
#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "utils/indv_guard.h"
#include "utils/indv_lib_wrapper.h"
#include "indv_bininfo.h"
#include "indv_collecter.h"
#include "indv_args.h"
#include "aclnn/acl_meta.h"
#include "op_info_serialize.h"
#include "profiling/prof_common.h"
#include "indv_executor_utils.h"
#include "thread_local_context.h"
#include "opdev/op_cache.h"
#include "runtime/runtime/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif
extern NnopbaseSysGlobalParams g_nnopbaseSysCfgParams;

static constexpr uint64_t NNOPBASE_EXECUTOR_MAGIC_NUMBER = 0x19A312F588;
static constexpr uint32_t NNOPBASE_PARAM_EXT_LEN = 16U;
static constexpr size_t NNOPBASE_MAX_STATICKEY_LEN = 8192U;

static constexpr size_t NNOPBASE_HOST_DATA_LEN = 1024U * 4U;
static constexpr size_t NNOPBASE_TILING_DATA_OFFSET = 2048U;
static constexpr size_t NNOPBASE_MAX_ARGS_BUF_LEN = 10240U;
static constexpr size_t NNOPBASE_MAX_TILING_DATA_LEN = NNOPBASE_MAX_ARGS_BUF_LEN - NNOPBASE_TILING_DATA_OFFSET;
static constexpr size_t NNOPBASE_MIN_TILING_DATA_LEN = 1024U;
static constexpr size_t NNOPBASE_MAX_ARGS_KEY_LEN = 10240U;
static constexpr size_t NNOPBASE_SEVENS_BYTES = 7U;
static constexpr size_t NNOPBASE_EIGHT_BYTES = 8U;
static constexpr int32_t INFO_TYPE_CORE_NUM = 3;
static constexpr size_t NNOPBAE_AICPU_PARAM_LEN = 32U;
static const std::string NNOPBAE_MC2_AICPU_SUFFIX = "Mc2AicpuKernel";
constexpr size_t NNOPBASE_BLOCK_SIZE = 32U;
constexpr int32_t INDV_LAUNCH_KERNEL_RTS_DYN_UBUF_SIZE = 2;

constexpr uint64_t MAX_DIM_NUM = 25;
struct NnopbaseExecutorSpace;

struct NnopbaseExecutorArgs {
    NnopbaseExecutorArgs() : argsBuf(NNOPBASE_MAX_ARGS_BUF_LEN), inputKey(NNOPBASE_MAX_ARGS_KEY_LEN) {}

    std::vector<uint8_t> argsBuf;
    std::vector<uint8_t> inputKey;
    NnopbaseTilingInfo tilingInfo = {};
    NnopbaseBinInfo *binInfo = nullptr;
    size_t keyLen = 0U;
    size_t seed = 0U;
    size_t tilingDataOffset = NNOPBASE_TILING_DATA_OFFSET;
    bool isVist = false;
    bool enableCache = true; // 缓存是否启用的开关
    std::vector<uint8_t> memsetArgs;
    std::vector<uint64_t> dfxInfo;
    std::vector<uint8_t> attrsData;
    NnopbaseTensors inputs;
    NnopbaseTensors outputs;
    size_t inUncontWsSize = 0U;
    uint64_t workspaceLen = 0U;
    size_t workspaceNum = 0U;
    bool isOutEmpty = false;
    bool hasTiling = true;
    bool hasMemset = false;
    size_t remainKeyLen = NNOPBASE_MAX_ARGS_KEY_LEN;
};

typedef struct {
    bool isMc2 = false;
    NnopbaseHcclServerType sType;
    std::vector<HcclComm> hcomHandle;
} NnopbaseMc2OpCfg;

typedef struct {
    uint64_t magicNum = NNOPBASE_EXECUTOR_MAGIC_NUMBER; // 首8字节标识该executor类型属于nnopbase不可更改顺序
    bool repeateFlag;
    NnopbaseAttrs attrs;
    uint64_t *tilingKey;
    uint32_t *numBlocks;
    bool *needAtomic;
    uint32_t *scheMode;
    uint32_t *dynUbufSize;
    uint32_t *aicpuNumBlocks;
    NnopbaseExecutorArgs *args;
    NnopbaseExecutorArgs ownArgs;
    NnopbaseBinCollecter *collecter;
    NnopbaseRegInfo *regInfo;
    NnopbaseWorkSpaces workspaces;
    NnopbaseChar *opType;
    NnopbaseExecutorSpace *space;
    uint64_t opTypeHash;
    BinInfoKey binInfoKey;
    NnopbaseRTArgsExt argsExt;
    NnopbaseKernelRunContextExt contextExt;
    rtStream_t stream;
    bool hasTiling;
    bool isWork;
    NnopbaseDfxId *tilingId;
    OpSupportList *supportList;
    uint32_t *socSupportList;
    uint32_t socSupportListLen;
    uint32_t opTypeId;
    int32_t poolIndex;
    void *userHandle;
    bool isOutEmpty;
    bool isZeroEleOutputLaunch = false;
    std::vector<void *> contextAddr;
    rtAicpuArgsEx_t aicpuArgs;
    rtFusionArgsEx_t fusionArgs;
    std::vector<rtStream_t> aicpuStream;
    std::vector<std::pair<rtStream_t, rtStream_t>> aicpuNotify;
    aclOpExecutor *inUnContExe;
    aclOpExecutor *viewCopyExe;
    size_t inUncontWsSize;
    uint64_t itemId;
    uint64_t aicpuItemId;
    uint64_t memsetItemId;
    aclnnOpInfoRecord::OpKernelInfo opKernelInfo{"", 0};
    NnopbaseTimeStamp timeStamp{};
    std::vector<uint8_t> outputShapeData;
    bool hasMemset;
    NnopbaseFormatCheckOption formatCheckOption;
    NnopbaseMc2OpCfg mc2OpCfg;
    bool matchArgsV2 = false; // 兼容之前的缓存匹配流程，true表示走的新流程，匹配的时候不用再重新生成key
    bool isCachedArgs = false;
    NnopbaseCoreNum coreNum{0, 0};
    bool deterministic = false;
} NnopbaseExecutor;

typedef struct {
    uint64_t version:4; // 版本号，当前是1，预留参数
    uint64_t groupNum:4; // 一共有几个hcclgroup，每个group对应1个context参数
    uint64_t hasFfts:1; // 1971下是否是ffts融合算子
    uint64_t tilingOff:7; // tilingdata ptr指针所在的参数索引
    uint64_t isDyn:48; // bitmap，每个bit对应一个IR输入，如果是动态输入则为1，否则是0
} NnopbaseHcclCommParamDesc;

typedef struct {
    NnopbaseUChar *hostInputData;
    aclrtPlaceHolderInfo *hostInputInfo;
    NnopbaseUChar *ptr;
    aclrtPlaceHolderInfo *aicpuHostInputInfo;
    NnopbaseHcclCommParamDesc *hcclDesc;
} NnopbaseExecutorArgsAddr;

struct NnopbaseExecutorSpace {
    std::vector<NnopbaseExecutor*> executors;
    std::mutex spaceMtx;
};

typedef struct {
    std::vector<NnopbaseExecutorSpace*> spaces;
    bool isVist;
} NnopbaseExecutorSpaceSet;

typedef struct {
    NnopbaseChar *inputsDesc;
    uint32_t inputNum;
    NnopbaseChar *outputsDesc;
    uint32_t outputNum;
    NnopbaseChar *attrsDesc;
    uint32_t attrsNum;
} NnopbaseOpInfo;

typedef struct {
    rtStream_t stream;
    rtEvent_t eventA;
    rtEvent_t eventB;
} NnopbaseStreamForCombineExecution; // CUB+AIV or AICPU+AICORE组合执行场景

typedef struct {
    uint32_t aivNumBlocks;
    uint32_t aicNumBlocks;
    uint32_t aivNumBlocksOffset;
} NnopbaseBlockInfoForVectorCore;

aclnnStatus NnopbaseExecutorKernelLaunch(NnopbaseExecutor *executor, rtStream_t stream);
aclnnStatus NnopbaseExecutorGetStreamAndEvent(
    const rtStream_t stream, rtStream_t *subStream, rtEvent_t *evtA, rtEvent_t *evtB,
    std::shared_ptr<std::mutex> &streamLckPtr);
aclnnStatus NnopbaseCreateStreamResource(NnopbaseStreamForCombineExecution *nnopbaseStream);
void NnopbaseDestroyStreamCallBack(rtStream_t stream, const bool isCreate);

aclnnStatus NnopbaseExecutorInit(NnopbaseExecutor *executor, const NnopbaseOpInfo opInfo);
void NnopbaseExecutorDeInit(NnopbaseExecutor *executor);
void StreamMapClear(rtStream_t stream);
aclnnStatus NnopbaseExecutorCheckSocVersionAndParam(NnopbaseExecutor *executor);
aclnnStatus NnopbaseExecutorGenStaticKey(NnopbaseExecutor *executor, bool usingStride = false);
void NnopbaseExecutorGenDynamicKey(NnopbaseExecutor *executor);
size_t NnopbaseExecutorComputeGenKeySize(const NnopbaseExecutor *const executor, bool usingStride = false);
NnopbaseUChar *NnopbaseExecutorGenAttrsKey(NnopbaseAttrs *attrs, NnopbaseUChar *verKey);
NnopbaseUChar *NnopbaseExecutorGenTensorsKey(NnopbaseUChar *verKey, NnopbaseTensors *tensors,
    size_t tensorNum, bool usingStride = false);
bool NnopbaseExecutorGetStaticBinInfo(NnopbaseExecutor *executor);
NnopbaseUChar *NnopbaseExecutor8ByteCopy(size_t totalSize, NnopbaseUChar *verKey, const NnopbaseUChar *addr);
aclnnStatus SetTensorDataSizeToInitValue(NnopbaseExecutor *executor);

// prepare args for launch
void **NnopbaseExecutorPrepareNullTensors(
    const NnopbaseExecutor *const executor, void **addr, size_t *tensorIndex);
aclnnStatus NnopbaseExecutorPrepareParamsExt(NnopbaseExecutor *executor, rtStream_t const stream);
void **NnopbaseExecutorPrepareInputsParamsExt(NnopbaseExecutor *executor, void **addr,
                                              NnopbaseExecutorArgsAddr *argsAddr);
void **NnopbaseExecutorPrepareOutputsParamsExt(NnopbaseExecutor *executor, void **addr,
                                               NnopbaseExecutorArgsAddr *argsAddr);

aclnnStatus NnopbaseExecutorTilingAndUpdateBinInfo(NnopbaseExecutor *executor);
aclnnStatus NnopbaseExecutorGcSpace(void *data);
aclnnStatus NnopbaseExecutorClearSpace(NnopbaseExecutorSpace *space);
aclnnStatus NnopbaseExecutorSetRegInfo(NnopbaseExecutor *executor, const NnopbaseChar *opType);

void NnopbaseExecutorUpdateBinInfo(NnopbaseExecutor *executor);

// add io
struct NnopbaseTensorIoOptions {
    bool isInput;
    bool ignoreCont;
};

aclnnStatus NnopbaseExecutorAddTensor(NnopbaseExecutor *executor, const aclTensor *tensor,
                                      const uint32_t index, const bool isInput, const bool ignoreCont);
aclnnStatus NnopbaseExecutorUpdateTensorsIndex(NnopbaseTensors *tensors, const uint32_t index);
aclnnStatus NnopbaseExecutorAddDynamicTensors(NnopbaseExecutor *executor, const aclTensorList *tensorList,
                                              const uint32_t index, const bool isInput, const bool ignoreCont = false);
aclnnStatus NnopbaseExecutorSetRef(NnopbaseExecutor *executor, const size_t inputIrIdx, const size_t outputIrIdx);
aclnnStatus NnopbaseExecutorAddAttr(NnopbaseExecutor *executor, const void *const attrAddr, const size_t attrLen,
    const size_t index, const size_t elementSize, const NnopbaseAttrDtype dtype);

// IOcache
aclnnStatus NnopbaseExecutorInitIoCaches(NnopbaseTensors *tensors, const NnopbaseChar *paramDesc, const uint32_t num);
aclnnStatus NnopbaseExecutorExtendIoCaches(NnopbaseTensors *tensors);
void NnopbaseExecutorFixCache(NnopbaseExecutor *executor);

// execute
aclnnStatus NnopbaseExecutorRunWithWorkspace(NnopbaseExecutor *executor, rtStream_t stream, void *workspace,
                                             const uint64_t workspaceLen);
aclnnStatus NnopbaseExecutorRunForWorkspace(NnopbaseExecutor *executor, uint64_t *workspaceLen);
void NnopbaseExecutorClear(NnopbaseExecutor *executor);

size_t NnopbaseCalcArgsSize(NnopbaseExecutor *executor, const size_t tilingDataSize);
void NnopbaseExecutorPrepareDfxInfo(NnopbaseExecutor *executor);
aclnnStatus NnopbaseExecutorArgsGetDfxInfo(
    NnopbaseExecutor *executor, NnopbaseExecutorArgsAddr *argsAddr, const uint32_t workspaceNum);
bool NnopbaseMatchArgsCache(NnopbaseExecutor *executor, size_t &seed);
aclnnStatus NnopbaseCreateExecutorArgs(NnopbaseExecutor *executor, const size_t seed);

// for dumpdata
void NnopbaseDumpData(NnopbaseExecutor *executor, Adx::TensorType ioType, aclrtStream stream, NnopbaseChar *opType);
aclnnStatus NnopbaseDumpWorkspaceData(NnopbaseExecutor *executor, aclrtStream stream);
aclnnStatus NnopbasePrepareExceptionDumpInfo(NnopbaseExecutor *const executor, aclrtStream stream);
aclnnStatus NnopbaseArgsExceptionDumpAddr(NnopbaseExecutor *const executor);
aclnnStatus NnopbaseOverflowDump(NnopbaseExecutor *const executor, aclrtStream stream);

// for profiling
void NnopbaseReportMemsetAdditionInfo(const NnopbaseExecutor *const executor, uint32_t numBlocks,
                                      uint32_t taskType, const uint64_t timeStamp);
void NnopbaseReportAdditionInfo(void *const executor, uint32_t numBlocks, uint32_t taskType, const uint64_t timeStamp);
void NnopbaseInnerReportLaunchInfo(const uint64_t beginTime, const uint64_t itemId);
void NnopbaseReportTimeStampInfo(const std::vector<MsprofAicTimeStampInfo> &timeStampInfo);
void NnopbaseExecutorReportProfiling(NnopbaseExecutor *const executor, uint32_t numBlocks, const uint32_t taskType,
    const uint64_t launchBeginTime, aclrtStream stream);
void NnopbasePreportAttrAndHostInfo(const NnopbaseExecutor *const executor, const uint64_t timeStamp);
std::string NnopbaseGetHostInfoStr(const NnopbaseTensors &tensors);
std::string NnopbaseGetAttrVal(const NnopbaseAttrs &attrs);
void NnopbaseReportCacheOpInfo(NnopbaseExecutor *const executor, uint32_t numBlocks, uint32_t taskType,
    aclrtStream stream);

aclnnStatus NnopbaseExecutorAddScalarInput(NnopbaseTensors *tensors, const aclScalar *scalar, const uint32_t index,
    const int32_t srcIndex, const ge::DataType dtype);
aclnnStatus NnopbaseExecutorAddScalarListInput(NnopbaseTensors *tensors, const aclScalarList *scalarList,
                                               const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);

// global config
aclnnStatus NnopbaseExecutorSetGlobalConfig();
void NnopbaseReportContextIdInfoByRation(
    NnopbaseExecutor *const opExecutor, const uint64_t timeStamp, uint32_t &numBlocks, uint32_t &taskType);

void PrintNnopbaseAllTimeStampInfo(NnopbaseExecutor *const executor);
void PrintNnopbaseInitTimeStampInfo();

// for mesmet
aclnnStatus NnopbaseExecutorInsertMemsetOp(NnopbaseExecutor *executor);
aclnnStatus NnopbaseLaunchMemsetTask(NnopbaseExecutor *executor, rtStream_t stream);

aclnnStatus NnopbaseExecutorGetCoreTypeAndTaskRation(
    NnopbaseExecutor *executor, CoreType &coreType, NnopbaseTaskRation &taskRation);

// for mc2
aclnnStatus NnopbaseSetMc2Tiling(NnopbaseExecutor *executor, const char *const group);
static inline void NnopbaseExecutorSetHcclServerType(NnopbaseExecutor *executor, NnopbaseHcclServerType sType)
{
    executor->mc2OpCfg.sType = sType;
}


aclnnStatus NnopbaseSetRefTensorAddr(NnopbaseExecutor *executor, const size_t index, const void *const addr,
    const std::vector<NnopbaseParamInstance> &inInstances, const std::vector<NnopbaseParamInstance> &outInstances);

// for uncontiguous
aclnnStatus NnopbaseExecutorSetUnContExecutor(NnopbaseExecutor *executor, aclOpExecutor *inExe, const size_t inWsSize);
aclnnStatus NnopbaseExecutorSetViewCopyExecutor(NnopbaseExecutor *executor, aclOpExecutor *exe);

// clear and update tensor 
aclnnStatus NnopbaseUpdateDynamicTensors(NnopbaseTensors *dstTensors, NnopbaseTensors *tensors, uint32_t index);
aclnnStatus NnopbaseSaveCachedTensor(NnopbaseTensors *dstTensors, NnopbaseTensors *tensors, bool isInput);
void NnopbaseSetCachedInfo(NnopbaseExecutor *executor);
aclnnStatus NnopbaseUpdateInputAddr(NnopbaseExecutor *executor);
aclnnStatus NnopbaseUpdateOutputAddr(NnopbaseTensors *dstTensors, NnopbaseTensors *tensors);
aclnnStatus NnopbaseAddIoTensors(NnopbaseExecutor *executor);
void NnopbaseClearParamInstance(NnopbaseTensors *tensors);
aclnnStatus UpdateArgsIoAddr(NnopbaseTensors *dstTensors, NnopbaseTensors *tensors);
void UpdateArgsUncontiguousTensor(NnopbaseUnContTensors &dstUncontTensors, NnopbaseUnContTensors &srcUncontTensors);
bool NnopbasIsEnableNewCache(const NnopbaseExecutor *executor);
void NnopbaseCheckHasContiguous(NnopbaseExecutor *executor);
void NnopbaseSaveUnContiguousTensors(NnopbaseTensors *dstTensors, NnopbaseTensors *tensors);

aclnnStatus NnopbaseKernelRegister(NnopbaseExecutor *executor, NnopbaseBinInfo *binInfo);
inline bool NnopbaseIsExceptionDumpEnable()
{
    static const bool ret = Adx::AdumpGetDumpSwitch(Adx::DumpType::EXCEPTION);
    OP_LOGI("Get exception dump = %d", ret);
    return ret;
}

inline bool NnopbaseSkipKernelLaunch(const NnopbaseExecutor *executor)
{
    return executor->isOutEmpty && !executor->isZeroEleOutputLaunch;
}

inline void RecordNnopbaseTime(NnopbaseExecutor *const executor, const size_t index)
{
    if (g_nnopbaseSysCfgParams.enableTimeStamp) {
        clock_gettime(CLOCK_MONOTONIC, &(executor->timeStamp.tp[index]));
    }
}

inline void RecordNnopbaseInitTime(NnopbaseBinCollecter *const collecter, const size_t index)
{
    clock_gettime(CLOCK_MONOTONIC, &(collecter->collectorTp[index]));
}

inline uint32_t NnopbaseExecutorGetNumBlocks(NnopbaseExecutor *executor)
{
    if (executor->args->binInfo->isStaticShape) {
        return executor->args->binInfo->numBlocks;
    } else {
        return executor->args->tilingInfo.numBlocks;
    }
}

inline const NnopbaseWorkspaceSizes* NnopbaseGetWorkspacesSizesFromArgs(const NnopbaseExecutorArgs *args)
{
    if (args->binInfo->isStaticShape) {
        return op::internal::PtrCastTo<const NnopbaseWorkspaceSizes>(args->binInfo->staticWorkspaceSizes);
    } else {
        return op::internal::PtrCastTo<const NnopbaseWorkspaceSizes>(args->tilingInfo.workspacesSizes);
    }
}

static inline void NnopbaseExecutorAddTilingId(NnopbaseExecutor *executor, NnopbaseDfxId *tilingId)
{
    executor->tilingId = tilingId;
}

static inline void NnopbaseExecutorAddSupportList(NnopbaseExecutor *executor, OpSupportList *supportList,
                                                  uint32_t *socSupportList, size_t socSupportListLen)
{
    executor->supportList = supportList;
    executor->socSupportList = socSupportList;
    executor->socSupportListLen = socSupportListLen;
}

static inline void NnopbaseExecutorAddOpTypeId(NnopbaseExecutor *executor, const uint32_t opTypeId)
{
    executor->opTypeId = opTypeId;
}

static inline void NnopbaseExecutorGetRefUnContiguousTensors(NnopbaseExecutor *executor,
                                                             const aclTensorList **unContTensors,
                                                             const aclTensorList **contTensors)
{
    auto &inUnContTensors = executor->args->inputs.unContiguousTensors;
    auto &refUnContTensors = inUnContTensors.refUnContTensors;
    OP_LOGD("Op %s has %zu ref uncontiguous input.", executor->opType, refUnContTensors.size());
    if (!refUnContTensors.empty()) {
        inUnContTensors.refUnContTensorList.tensors = (&refUnContTensors[0U]);
        inUnContTensors.refUnContTensorList.size = refUnContTensors.size();
        *unContTensors = op::internal::PtrCastTo<const aclTensorList>(&inUnContTensors.refUnContTensorList);
        const aclTensor *const *value = &inUnContTensors.refContTensors[0U];
        inUnContTensors.refContTensorList = aclCreateTensorList(value, inUnContTensors.refContTensors.size());
        *contTensors = inUnContTensors.refContTensorList;
    } else {
        *unContTensors = nullptr;
        *contTensors = nullptr;
    }
}

void NnopbaseExecutorGetUnContiguousTensors(NnopbaseExecutor *executor, const aclTensorList **inTensors);

static inline void NnopbaseExecutorGetUnContExecutor(NnopbaseExecutor *executor, aclOpExecutor **inExe,
                                                     size_t *inWsSize)
{
    *inExe = executor->inUnContExe;
    *inWsSize = executor->inUncontWsSize;
    OP_LOGI("Op %s get contiguous workspace size %zu bytes, executor addr is %p.", executor->opType, *inWsSize, executor);
}

static inline void NnopbaseExecutorGetViewCopyExecutor(NnopbaseExecutor *executor, aclOpExecutor **exe)
{
    OP_LOGI("Op %s get viewCopyExe %p, executor addr is %p.", executor->opType, executor->viewCopyExe, executor);
    *exe = executor->viewCopyExe;
}

static inline void NnopbaseExecutorReleaseRefContiguousTensors(NnopbaseExecutor *executor,
                                                               const aclTensorList **tensors)
{
    *tensors = executor->args->inputs.unContiguousTensors.refContTensorList;
    executor->args->inputs.unContiguousTensors.refContTensorList = nullptr;
}

static inline void NnopbaseExecutorSpaceSetInit(NnopbaseExecutorSpaceSet *set)
{
    set->isVist = false;
}
static inline void NnopbaseExecutorAddSpaceToSet(NnopbaseExecutorSpaceSet *set, NnopbaseExecutorSpace *space)
{
    while (!__sync_bool_compare_and_swap(&set->isVist, false, true));
    set->spaces.push_back(space);
    set->isVist = false;
}

void NnopbaseExecutorClearSet(NnopbaseExecutorSpaceSet *set);

aclnnStatus NnopbaseExecutorGetAttr(NnopbaseExecutor *executor, const size_t index, NnopbaseAttrAddr **attr);

static inline void NnopbaseExecutorSetCollecter(NnopbaseExecutor *executor, NnopbaseBinCollecter *collecter)
{
    executor->collecter = collecter;
}

void NnopbaseReloadStaticBinJsonInfos(void);
aclnnStatus NnopbaseExecutorConvertScalarType(std::vector<uint8_t> &scalarValue, const aclScalar *scalar,
                                              ge::DataType dtype, const size_t offset);
aclnnStatus NnopbaseSetUnContiguousExecutorRepeatable(NnopbaseExecutor *executor);
aclnnStatus NnopbaseSetRepeatable(void *executor);
aclnnStatus NnopbaseReSetUnContiguousExecutorRepeatable(NnopbaseExecutor *executor);
aclnnStatus NnopbaseResetExecutor(void *executor);
bool NnopbaseIsInput(const void *executor, const size_t index, size_t *tensorIndex);
bool NnopbaseDynamicIsInput(const void *executor, const size_t irIndex, size_t *tensorIrIndex);
void NnopbaseGetRealIndex(const NnopbaseParamDesc &paramDescs, size_t *realIndex);
void NnopbaseExecutorCopyCacheAttr(NnopbaseExecutor *executor);
void NnopbaseExecutorClearUnContiguousTensors(NnopbaseTensors *tensors);

#ifdef __cplusplus
}
#endif

#endif

#include <string.h>
#include "graph/types.h"
#include "aclnnInner_all_gather_matmul_v3.h"

#ifdef ACLNN_WITH_BINARY
#include <vector>
#include <tuple>
#include <map>
#include "graph/ascend_string.h"
#include "AllGatherMatmulV3_op_resource.h"
using OP_HOST_FUNC_HANDLE = std::vector<void *>;
using OP_RES = std::tuple<const uint8_t *, const uint8_t *>;
using OP_BINARY_RES = std::vector<OP_RES>;
using OP_RUNTIME_KB_RES = std::vector<OP_RES>;
using OP_RESOURCES = std::map<ge::AscendString,
    std::tuple<OP_HOST_FUNC_HANDLE, OP_BINARY_RES, OP_RUNTIME_KB_RES>>;
using OP_SOC_RESOURCES = std::map<ge::AscendString, std::tuple<OP_HOST_FUNC_HANDLE,
    std::map<ge::AscendString, OP_BINARY_RES>, OP_RUNTIME_KB_RES>>;
namespace op {
extern uint32_t GenOpTypeId(const char *op_name, const OP_RESOURCES &op_resources);
extern uint32_t GenOpTypeId(const char *op_name, const OP_SOC_RESOURCES &op_resources);
}
#endif

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B = 2,
    SOC_VERSION_ASCEND910_93 = 3,
    SOC_VERSION_ASCEND950 = 4,
    SOC_VERSION_ASCEND310P = 5,
    SOC_VERSION_ASCEND310B = 6,
    SOC_VERSION_BS9SX1A = 7,
    SOC_VERSION_ASCEND610Lite = 8,
    SOC_VERSION_MC61AM21A = 10, // 9 is deprecated
    SOC_VERSION_MC62CM12A = 11,
    SOC_VERSION_BS9SX2A = 12,
    SOC_VERSION_ASCEND910_96 = 13,
    SOC_VERSION_KIRINX90 = 14,
    SOC_VERSION_KIRIN9030 = 15
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND950};
uint32_t socSupportListLen = 1;

TensorDesc inputDesc0_0[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_3[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_4[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_5[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_6[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_7[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_8[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc inputDesc0_9[6] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_3[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_4[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_5[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_6[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_7[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_8[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_9[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 6, outputDesc0_0, 3};
SupportInfo list0_1 = {inputDesc0_1, 6, outputDesc0_1, 3};
SupportInfo list0_2 = {inputDesc0_2, 6, outputDesc0_2, 3};
SupportInfo list0_3 = {inputDesc0_3, 6, outputDesc0_3, 3};
SupportInfo list0_4 = {inputDesc0_4, 6, outputDesc0_4, 3};
SupportInfo list0_5 = {inputDesc0_5, 6, outputDesc0_5, 3};
SupportInfo list0_6 = {inputDesc0_6, 6, outputDesc0_6, 3};
SupportInfo list0_7 = {inputDesc0_7, 6, outputDesc0_7, 3};
SupportInfo list0_8 = {inputDesc0_8, 6, outputDesc0_8, 3};
SupportInfo list0_9 = {inputDesc0_9, 6, outputDesc0_9, 3};
SupportInfo supportInfo0[10] = {list0_0, list0_1, list0_2, list0_3, list0_4, list0_5, list0_6, list0_7, list0_8, list0_9};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 10};

OpSocSupportInfo opSocSupportList[1] = {socSupportInfo0};
OpSupportList supportList = {opSocSupportList, 1};

[[maybe_unused]] uint32_t NNOPBASE_AllGatherMatmulV3 = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIgnoreContinuesInput(void *executor,
                                                   const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIntArrayInput(void *executor, const aclIntArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddBoolArrayInput(void *executor, const aclBoolArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddFloatArrayInput(void *executor, const aclFloatArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicInput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicOutput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index, const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray* array, const size_t index);
extern aclnnStatus NnopbaseAddFloatArrayAttr(void *executor, const aclFloatArray* array, const size_t index);
extern aclnnStatus NnopbaseAddBoolArrayAttr(void *executor, const aclBoolArray* array, const size_t index);
extern aclnnStatus NnopbaseAddArrayAttrWithDtype(void *executor, void *array, const size_t len, const size_t elementSize, const size_t index, const NnopbaseAttrDtype dtype);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace, uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);
extern aclnnStatus NnopbaseAddScalarInput(void *executor, const aclScalar *scalar, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern aclnnStatus NnopbaseAddScalarListInput(void *executor, const aclScalarList *scalarList, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern void NnopbaseAddOpTypeId(void *executor, const uint32_t opTypeId);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name, const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern aclnnStatus NnopbaseSetRef(void *executor, const size_t inputIrIdx, const size_t outputIrIdx);
extern void __attribute__((weak)) NnopbaseSetMatchArgsFlag(void *executor);
extern bool __attribute__((weak)) NnopbaseMatchArgs(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseGetUnContiguousTensors(void *executor, const aclTensorList **inTensors);
extern aclnnStatus NnopbaseSetUnContExecutor(void *executor, aclOpExecutor *inExe, const size_t inWsSize);
extern aclnnStatus NnopbaseGetUnContExecutor(void *executor, aclOpExecutor **inExe, size_t *inWsSize);
extern aclnnStatus NnopbaseGetRefUnContiguousTensors(void *executor, const aclTensorList **unContTensors, const aclTensorList **contTensors);
extern aclnnStatus NnopbaseSetViewCopyExecutor(void *executor, aclOpExecutor *exe);
extern aclnnStatus NnopbaseGetViewCopyExecutor(void *executor, aclOpExecutor **exe);
extern aclnnStatus NnopbaseReleaseRefContiguousTensors(void *executor, const aclTensorList **tensors);
extern void *NnopbaseGetApiFunc(const char *funcName);
using AclnnContiguousGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnViewCopyGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

#define ACLNN_SUCCESS  0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

aclnnStatus aclnnInnerAllGatherMatmulV3GetWorkspaceSize(
    const aclTensor *context,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    char *group,
    int64_t hcclBufferSize,
    bool isTransA,
    bool isTransB,
    int64_t rankSize,
    int64_t groupSize,
    int64_t yDtype,
    char *commModeOptional,
    const aclTensor *yOut,
    const aclTensor *gatherOutOut,
    const aclTensor *amaxOutOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
#ifdef ACLNN_WITH_BINARY
    static uint32_t AllGatherMatmulV3OpTypeId = op::GenOpTypeId("AllGatherMatmulV3", AllGatherMatmulV3_RESOURCES);
#endif
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnInnerAllGatherMatmulV3Tiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "AllGatherMatmulV3";
    char inputDesc[] = {1, 1, 1, 0, 0, 0};
    char outputDesc[] = {1, 1, 0};
    char attrDesc[] = {1, 1, 0, 0, 0, 0, 0, 0};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(context);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(x1);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(x2);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(yOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(gatherOutOut);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    if (NnopbaseSetMatchArgsFlag != NULL) {
        NnopbaseSetMatchArgsFlag(*executor);
    }
#ifdef ACLNN_WITH_BINARY
    NnopbaseAddOpTypeId(*executor, AllGatherMatmulV3OpTypeId);
#endif
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, context, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x1, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIgnoreContinuesInput(*executor, x2, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, biasOptional, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x1ScaleOptional, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x2ScaleOptional, 5));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(group);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(group), strlen(group) + 1, 0, kNnopbaseString));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&hcclBufferSize), sizeof(int64_t), 1, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&isTransA), sizeof(bool), 2, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&isTransB), sizeof(bool), 3, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&rankSize), sizeof(int64_t), 4, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&groupSize), sizeof(int64_t), 5, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&yDtype), sizeof(int64_t), 6, kNnopbaseInt));
    if (commModeOptional) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(commModeOptional), strlen(commModeOptional) + 1, 7, kNnopbaseString));
    } else {
        static char *commModeOptionalDef = "ai_cpu";
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(commModeOptionalDef), strlen(commModeOptionalDef) + 1, 7, kNnopbaseString));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, yOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, gatherOutOut, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, amaxOutOutOptional, 2));
    if (NnopbaseMatchArgs != NULL) {
        if (NnopbaseMatchArgs(*executor, workspaceSize)) {
            NnopbaseReportApiInfo(timeStamp, dfxId);
            return ACLNN_SUCCESS;
        }
    }
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "context", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "x1", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "x2", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "biasOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "x1ScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "x2ScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "yOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "gatherOutOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "amaxOutOutOptional", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));

    const aclTensorList *inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(*executor, &inUnContTensors);
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    if (inUnContTensors != nullptr) {
        static AclnnContiguousGetWorkspaceSizeFunc aclnnContiguousGetWorkspaceSize = (AclnnContiguousGetWorkspaceSizeFunc)NnopbaseGetApiFunc("aclnnContiguousGetWorkspaceSize");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguousGetWorkspaceSize);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor));
    }
    NnopbaseSetUnContExecutor(*executor, aclInExecutor, inContWorkspaceSize);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    *workspaceSize += inContWorkspaceSize;
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInnerAllGatherMatmulV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    NnopbaseGetUnContExecutor(executor, &aclInExecutor, &inContWorkspaceSize);
    if (workspaceSize < inContWorkspaceSize) {
        NnopbaseOpLogE(ACLNN_ERR_PARAM_INVALID, "input workspaceSize must be larger than contiguous size!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    workspaceSize -= inContWorkspaceSize;
    void *inWorkspace = (char *)workspace + workspaceSize;
    if (aclInExecutor != nullptr) {
        static AclnnFunc aclnnContiguous = (AclnnFunc)NnopbaseGetApiFunc("aclnnContiguous");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguous);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif

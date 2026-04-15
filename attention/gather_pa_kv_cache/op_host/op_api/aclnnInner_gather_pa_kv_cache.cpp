#include <string.h>
#include "graph/types.h"
#include "aclnnInner_gather_pa_kv_cache.h"

#ifdef ACLNN_WITH_BINARY
#include <vector>
#include <tuple>
#include <map>
#include "graph/ascend_string.h"
#include "GatherPaKvCache_op_resource.h"
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
    SOC_VERSION_ASCEND910B,
    SOC_VERSION_ASCEND910_93,
    SOC_VERSION_ASCEND910_95,
    SOC_VERSION_ASCEND310P,
    SOC_VERSION_ASCEND310B,
    SOC_VERSION_BS9SX1A,
    SOC_VERSION_ASCEND610Lite,
    SOC_VERSION_ASCEND910_55,
    SOC_VERSION_MC61AM21A,
    SOC_VERSION_MC62CM12A,
    SOC_VERSION_BS9SX2A,
    SOC_VERSION_ASCEND910_96,
    SOC_VERSION_KIRINX90,
    SOC_VERSION_KIRIN9030
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910_93,SOC_VERSION_ASCEND910B};
uint32_t socSupportListLen = 2;

TensorDesc inputDesc0_0[7] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[7] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[7] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[2] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 7, outputDesc0_0, 2};
SupportInfo list0_1 = {inputDesc0_1, 7, outputDesc0_1, 2};
SupportInfo list0_2 = {inputDesc0_2, 7, outputDesc0_2, 2};
SupportInfo supportInfo0[3] = {list0_0, list0_1, list0_2};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 3};

TensorDesc inputDesc1_0[7] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc inputDesc1_1[7] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc inputDesc1_2[7] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc1_0[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc1_1[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc1_2[2] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND}};
SupportInfo list1_0 = {inputDesc1_0, 7, outputDesc1_0, 2};
SupportInfo list1_1 = {inputDesc1_1, 7, outputDesc1_1, 2};
SupportInfo list1_2 = {inputDesc1_2, 7, outputDesc1_2, 2};
SupportInfo supportInfo1[3] = {list1_0, list1_1, list1_2};
OpSocSupportInfo socSupportInfo1= {supportInfo1, 3};

OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
OpSupportList supportList = {opSocSupportList, 2};

[[maybe_unused]] uint32_t NNOPBASE_GatherPaKvCache = 0U;
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

extern aclnnStatus aclnnInnerGatherPaKvCacheGetWorkspaceSize(
    const aclTensor *keyCache,
    const aclTensor *valueCache,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    aclTensor *keyRef,
    aclTensor *valueRef,
    const aclTensor *seqOffsetOptional,
    char *cacheModeOptional,
    bool isSeqLensCumsum,
    const aclIntArray *kvCacheStridesOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
)；

aclnnStatus aclnnInnerGatherPaKvCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)；

aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(
    const aclTensor *keyCache,
    const aclTensor *valueCache,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    aclTensor *keyRef,
    aclTensor *valueRef,
    const aclTensor *seqOffsetOptional,
    char *cacheModeOptional,
    bool isSeqLensCumsum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    
    int64_t *keyCacheStridesValue = nullptr;
 	uint64_t keyCacheStridesNum = 0;
 	 
    int64_t *valueCacheStridesValue = nullptr;
 	uint64_t valueCacheStridesNum = 0;
 	 
 	aclnnStatus ret = aclGetViewStrides(keyCacheRef, &keyCacheStridesValue, &keyCacheStridesNum);
 	ret = aclGetViewStrides(valueCacheRef, &valueCacheStridesValue, &valueCacheStridesNum);
 	 
 	std::vector<int64_t> sizeData = {0, 0};
 	 
 	if (!op::IsContiguous(keyCacheRef) && !op::IsContiguous(valueCacheRef)){
 	    sizeData = {keyCacheStridesValue[0], valueCacheStridesValue[0]};
 	}
 	 
 	aclIntArray *kvCacheStridesOptional = aclCreateIntArray(sizeData.data(), sizeData.size());
 	 
 	ret = aclnnInnerGatherPaKvCacheGetWorkspaceSize(
 	    key, keyCacheRef,slotMapping,value,valueCacheRef,compressLensOptional,compressSeqOffsetOptional,seqLensOptional,cacheModeOptional,scatterModeOptional,stridesOptional,offsetsOptional,kvCacheStridesOptional,workspaceSize,executor );
 	 
 	aclDestroyIntArray(kvCacheStridesOptional);
 	delete[] keyCacheStridesValue;
 	delete[] valueCacheStridesValue;
 	return ret;
}

aclnnStatus aclnnGatherPaKvCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerGatherPaKvCache(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

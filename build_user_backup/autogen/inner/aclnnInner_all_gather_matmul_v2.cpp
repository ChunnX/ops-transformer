#include <string.h>
#include "graph/types.h"
#include "aclnnInner_all_gather_matmul_v2.h"

#ifdef ACLNN_WITH_BINARY
#include <vector>
#include <tuple>
#include <map>
#include "graph/ascend_string.h"
#include "AllGatherMatmulV2_op_resource.h"
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
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910_93,SOC_VERSION_ASCEND910B,SOC_VERSION_ASCEND950};
uint32_t socSupportListLen = 3;

TensorDesc inputDesc0_0[6] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[6] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_3[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_4[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_5[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_6[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_7[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_8[6] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_9[6] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_10[6] =
    {{ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_11[6] =
    {{ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_3[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_4[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_5[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_6[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_7[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_8[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_9[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_10[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_11[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
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
SupportInfo list0_10 = {inputDesc0_10, 6, outputDesc0_10, 3};
SupportInfo list0_11 = {inputDesc0_11, 6, outputDesc0_11, 3};
SupportInfo supportInfo0[12] = {list0_0, list0_1, list0_2, list0_3, list0_4, list0_5, list0_6, list0_7, list0_8, list0_9, list0_10, list0_11};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 12};

TensorDesc inputDesc1_0[6] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_1[6] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_2[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_3[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_4[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_5[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_6[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_7[6] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_8[6] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_9[6] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_10[6] =
    {{ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_11[6] =
    {{ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_0[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_1[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_2[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_3[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_4[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_5[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_6[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_7[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_8[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_9[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_10[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_11[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT4, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list1_0 = {inputDesc1_0, 6, outputDesc1_0, 3};
SupportInfo list1_1 = {inputDesc1_1, 6, outputDesc1_1, 3};
SupportInfo list1_2 = {inputDesc1_2, 6, outputDesc1_2, 3};
SupportInfo list1_3 = {inputDesc1_3, 6, outputDesc1_3, 3};
SupportInfo list1_4 = {inputDesc1_4, 6, outputDesc1_4, 3};
SupportInfo list1_5 = {inputDesc1_5, 6, outputDesc1_5, 3};
SupportInfo list1_6 = {inputDesc1_6, 6, outputDesc1_6, 3};
SupportInfo list1_7 = {inputDesc1_7, 6, outputDesc1_7, 3};
SupportInfo list1_8 = {inputDesc1_8, 6, outputDesc1_8, 3};
SupportInfo list1_9 = {inputDesc1_9, 6, outputDesc1_9, 3};
SupportInfo list1_10 = {inputDesc1_10, 6, outputDesc1_10, 3};
SupportInfo list1_11 = {inputDesc1_11, 6, outputDesc1_11, 3};
SupportInfo supportInfo1[12] = {list1_0, list1_1, list1_2, list1_3, list1_4, list1_5, list1_6, list1_7, list1_8, list1_9, list1_10, list1_11};
OpSocSupportInfo socSupportInfo1= {supportInfo1, 12};

TensorDesc inputDesc2_0[6] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_1[6] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_2[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_3[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_4[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_5[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_6[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_7[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_8[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_9[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_10[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_11[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_12[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_13[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_14[6] =
    {{ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_15[6] =
    {{ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_16[6] =
    {{ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_17[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_18[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_19[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_20[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_21[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_22[6] =
    {{ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_23[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_24[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_25[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_26[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_27[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_28[6] =
    {{ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_29[6] =
    {{ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_30[6] =
    {{ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc2_31[6] =
    {{ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_0[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_1[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_2[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_3[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_4[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_5[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_6[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_7[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_8[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_9[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_10[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_11[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_12[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_13[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_14[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_15[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_16[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_HIFLOAT8, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_17[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_18[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_19[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_20[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_21[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_22[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_23[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_24[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_25[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_26[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_27[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_28[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_29[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_30[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc2_31[3] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list2_0 = {inputDesc2_0, 6, outputDesc2_0, 3};
SupportInfo list2_1 = {inputDesc2_1, 6, outputDesc2_1, 3};
SupportInfo list2_2 = {inputDesc2_2, 6, outputDesc2_2, 3};
SupportInfo list2_3 = {inputDesc2_3, 6, outputDesc2_3, 3};
SupportInfo list2_4 = {inputDesc2_4, 6, outputDesc2_4, 3};
SupportInfo list2_5 = {inputDesc2_5, 6, outputDesc2_5, 3};
SupportInfo list2_6 = {inputDesc2_6, 6, outputDesc2_6, 3};
SupportInfo list2_7 = {inputDesc2_7, 6, outputDesc2_7, 3};
SupportInfo list2_8 = {inputDesc2_8, 6, outputDesc2_8, 3};
SupportInfo list2_9 = {inputDesc2_9, 6, outputDesc2_9, 3};
SupportInfo list2_10 = {inputDesc2_10, 6, outputDesc2_10, 3};
SupportInfo list2_11 = {inputDesc2_11, 6, outputDesc2_11, 3};
SupportInfo list2_12 = {inputDesc2_12, 6, outputDesc2_12, 3};
SupportInfo list2_13 = {inputDesc2_13, 6, outputDesc2_13, 3};
SupportInfo list2_14 = {inputDesc2_14, 6, outputDesc2_14, 3};
SupportInfo list2_15 = {inputDesc2_15, 6, outputDesc2_15, 3};
SupportInfo list2_16 = {inputDesc2_16, 6, outputDesc2_16, 3};
SupportInfo list2_17 = {inputDesc2_17, 6, outputDesc2_17, 3};
SupportInfo list2_18 = {inputDesc2_18, 6, outputDesc2_18, 3};
SupportInfo list2_19 = {inputDesc2_19, 6, outputDesc2_19, 3};
SupportInfo list2_20 = {inputDesc2_20, 6, outputDesc2_20, 3};
SupportInfo list2_21 = {inputDesc2_21, 6, outputDesc2_21, 3};
SupportInfo list2_22 = {inputDesc2_22, 6, outputDesc2_22, 3};
SupportInfo list2_23 = {inputDesc2_23, 6, outputDesc2_23, 3};
SupportInfo list2_24 = {inputDesc2_24, 6, outputDesc2_24, 3};
SupportInfo list2_25 = {inputDesc2_25, 6, outputDesc2_25, 3};
SupportInfo list2_26 = {inputDesc2_26, 6, outputDesc2_26, 3};
SupportInfo list2_27 = {inputDesc2_27, 6, outputDesc2_27, 3};
SupportInfo list2_28 = {inputDesc2_28, 6, outputDesc2_28, 3};
SupportInfo list2_29 = {inputDesc2_29, 6, outputDesc2_29, 3};
SupportInfo list2_30 = {inputDesc2_30, 6, outputDesc2_30, 3};
SupportInfo list2_31 = {inputDesc2_31, 6, outputDesc2_31, 3};
SupportInfo supportInfo2[32] = {list2_0, list2_1, list2_2, list2_3, list2_4, list2_5, list2_6, list2_7, list2_8, list2_9, list2_10, list2_11, list2_12, list2_13, list2_14, list2_15, list2_16, list2_17, list2_18, list2_19, list2_20, list2_21, list2_22, list2_23, list2_24, list2_25, list2_26, list2_27, list2_28, list2_29, list2_30, list2_31};
OpSocSupportInfo socSupportInfo2= {supportInfo2, 32};

OpSocSupportInfo opSocSupportList[3] = {socSupportInfo0, socSupportInfo1, socSupportInfo2};
OpSupportList supportList = {opSocSupportList, 3};

[[maybe_unused]] uint32_t NNOPBASE_AllGatherMatmulV2 = 0U;
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
extern aclnnStatus NnopbaseSetMc2(void *const executor);
extern aclnnStatus NnopbaseSetHcomGroup(void *const executor, char *const group);

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

aclnnStatus aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    const aclTensor *quantScaleOptional,
    char *group,
    bool isTransA,
    bool isTransB,
    int64_t gatherIndex,
    int64_t commTurn,
    int64_t rankSize,
    int64_t blockSize,
    int64_t groupSize,
    bool isGatherOut,
    bool isAmaxOut,
    int64_t yDtype,
    char *commMode,
    const aclTensor *yOut,
    const aclTensor *gatherOutOut,
    const aclTensor *amaxOutOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
#ifdef ACLNN_WITH_BINARY
    static uint32_t AllGatherMatmulV2OpTypeId = op::GenOpTypeId("AllGatherMatmulV2", AllGatherMatmulV2_RESOURCES);
#endif
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnInnerAllGatherMatmulV2Tiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "AllGatherMatmulV2";
    char inputDesc[] = {1, 1, 0, 0, 0, 0};
    char outputDesc[] = {1, 1, 0};
    char attrDesc[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

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
    NnopbaseAddOpTypeId(*executor, AllGatherMatmulV2OpTypeId);
#endif
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x1, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIgnoreContinuesInput(*executor, x2, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, biasOptional, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x1ScaleOptional, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x2ScaleOptional, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantScaleOptional, 5));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(group);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(group), strlen(group) + 1, 0, kNnopbaseString));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&isTransA), sizeof(bool), 1, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&isTransB), sizeof(bool), 2, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&gatherIndex), sizeof(int64_t), 3, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&commTurn), sizeof(int64_t), 4, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&rankSize), sizeof(int64_t), 5, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&blockSize), sizeof(int64_t), 6, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&groupSize), sizeof(int64_t), 7, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&isGatherOut), sizeof(bool), 8, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&isAmaxOut), sizeof(bool), 9, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&yDtype), sizeof(int64_t), 10, kNnopbaseInt));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(commMode);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(commMode), strlen(commMode) + 1, 11, kNnopbaseString));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, yOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, gatherOutOut, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, amaxOutOutOptional, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetMc2(*executor));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetHcomGroup(*executor, group));
    if (NnopbaseMatchArgs != NULL) {
        if (NnopbaseMatchArgs(*executor, workspaceSize)) {
            NnopbaseReportApiInfo(timeStamp, dfxId);
            return ACLNN_SUCCESS;
        }
    }
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "x1", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "x2", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "biasOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "x1ScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "x2ScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "quantScaleOptional", true));
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

aclnnStatus aclnnInnerAllGatherMatmulV2(
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

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_EP_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_EP_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"

namespace optiling {

constexpr int64_t FLOAT_DATA_SIZE = 4;
constexpr int64_t MIN_BUFFER_NUM = 2;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t QUE_NUM = 2;
constexpr int64_t CAST_NUM = 2;
constexpr int64_t TILINGKEY_FLOAT = 1;
constexpr int64_t TILINGKEY_FLOAT16 = 2;
constexpr int64_t TILINGKEY_BF16 = 3;

struct CoreParam {
    int64_t maxCoreMemery = 0;
    int64_t maxCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t remainMemerySpace = 0;
    int64_t bufferNum = 0;
    int64_t tilingKey = 0;
};

struct InputParam {
    int64_t tokensNum = 0;
    int64_t topK = 0;
    int64_t hiddenSize = 0;
    int64_t PermuteProbGradSize = 0;
    int64_t totalLength = 0;
    int64_t start = 0;
    int64_t end = 0;
    int64_t numOutTokens = 0;
    int64_t tokensDtypeSize = 0;
    int64_t indicesDtypeSize = 0;
    int64_t probsDtypeSize = 0;
    bool haveProbs = false;
    bool isUnpermute = true;
};

struct TilingParam {
    int64_t length = 0;
    int64_t num = 0;
    int64_t remain = 0;
};

struct MoeTokenUnpermuteWithEpParam {
    InputParam input;
    TilingParam hiddenTiling;
    TilingParam tokenTiling;
    TilingParam tokenPerCore;
    CoreParam core;
};

ge::graphStatus UnpermuteWithEpTilingCompute(gert::TilingContext* context, const int64_t topK, const bool isUnpermute);

BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteWithEpTilingData)
TILING_DATA_FIELD_DEF(int64_t, hidden_size);
TILING_DATA_FIELD_DEF(int64_t, permuted_probs_grad_length);
TILING_DATA_FIELD_DEF(int64_t, top_k);
TILING_DATA_FIELD_DEF(int64_t, start);
TILING_DATA_FIELD_DEF(int64_t, end);
TILING_DATA_FIELD_DEF(int64_t, num_out_tokens);

TILING_DATA_FIELD_DEF(int64_t, hidden_splited_length);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_num);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, tokens_core_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_core_remain);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_num);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, buffer_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTokenUnpermuteWithEp, MoeTokenUnpermuteWithEpTilingData)

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_EP_H

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

const char *engramFetchInfo = "EngramFetchFallback";

static graphStatus EngramFetchExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD(engramFetchInfo, "start to fallback for EngramFetch");

    const gert::Tensor *bufferhandle = host_api_ctx->GetInputTensor(0);
    const gert::Tensor *indices = host_api_ctx->GetInputTensor(1);
    const gert::Tensor *fetched = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(bufferhandle == nullptr, OP_LOGE(engramFetchInfo, "bufferhandle is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(indices == nullptr, OP_LOGE(engramFetchInfo, "indices is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(fetched == nullptr, OP_LOGE(engramFetchInfo, "fetched is null"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    const int64_t *numRanks = attrs->GetInt(0);
    const int64_t *numEntriesPerRank = attrs->GetInt(1);
    const int64_t *hcclBufferSize = attrs->GetInt(2);
    const int64_t *outputDtype = attrs->GetInt(3);

    const auto apiRet = EXEC_OPAPI_CMD(aclnnEngramFetch, bufferhandle, indices, *numRanks, *numEntriesPerRank,
                                       *hcclBufferSize, *outputDtype, fetched);
    OP_CHECK_IF(apiRet != ge::GRAPH_SUCCESS, OP_LOGE(engramFetchInfo, "aclnn api error %u", apiRet),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(EngramFetch).OpExecuteFunc(EngramFetchExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"

namespace ops {
ge::Status EngramFetchCalcParamFunc(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Do general calc param");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status EngramFetchGenTaskFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    OPS_LOG_D(context->GetNodeName(), "Do A5 gen task");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

IMPL_OP(EngramFetch).CalcOpParam(EngramFetchCalcParamFunc).GenerateTask(EngramFetchGenTaskFunc);
} // namespace ops
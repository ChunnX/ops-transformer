#include <vector>
#include "log/log.h"
#include "../fa_tiling_info.h"
#include "quant_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using namespace ge;
using namespace arch35FA;

ge::graphStatus QuantChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    const std::vector<int64_t> supportedQuantModes = {1};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(static_cast<int64_t>(faInfo.faQuantMode), supportedQuantModes),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "quant_mode",
            std::to_string(static_cast<int64_t>(faInfo.faQuantMode)).c_str(), "quant_mode must be 1"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.kDescaleType != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "k_descale",
            DataTypeToSerialString(faInfo.kDescaleType).c_str(), "FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.vDescaleType != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "v_descale",
            DataTypeToSerialString(faInfo.vDescaleType).c_str(), "FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.inputKvType != ge::DT_FLOAT4_E2M1,
        OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "k/v",
            DataTypeToSerialString(faInfo.inputKvType).c_str(), "FLOAT4_E2M1"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.quantMode == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "quant_mode",
            "quant_mode must be provided"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.kDescale.desc == nullptr || faInfo.opParamInfo.kDescale.shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "k_descale"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.vDescale.desc == nullptr || faInfo.opParamInfo.vDescale.shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "v_descale"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.kDescaleType != faInfo.vDescaleType,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "v_descale",
            DataTypeToSerialString(faInfo.vDescaleType).c_str(),
            ("v_descale dtype must match k_descale dtype " + DataTypeToSerialString(faInfo.kDescaleType)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling

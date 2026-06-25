#ifndef MIXED_QUANT_FLASH_ATTN_QUANT_CHECKER_H
#define MIXED_QUANT_FLASH_ATTN_QUANT_CHECKER_H

#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {

class QuantChecker : public FABaseChecker {
public:
    QuantChecker() = default;
    ~QuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif

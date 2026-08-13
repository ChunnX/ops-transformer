#ifndef FLASH_ATTN_DROPMASK_CHECKER_H
#define FLASH_ATTN_DROPMASK_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace flash_attn {

class DropMaskChecker : public FABaseChecker {
public:
    DropMaskChecker() = default;
    ~DropMaskChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
};

} // namespace flash_attn
} // namespace optiling
#endif // FLASH_ATTN_DROPMASK_CHECKER_H

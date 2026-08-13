#include <map>
#include <numeric>
#include <cmath>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "dropmask_checker.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus DropMaskChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (!(faInfo.keepProb > 0.0f && faInfo.keepProb <= 1.0f)) {
        OP_LOGE(faInfo.opName, "keep_prob must be in (0,1], got %f", faInfo.keepProb);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling

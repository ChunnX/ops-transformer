#include "../../op_kernel/moe_ep_combine_tiling_key.h"

namespace optiling {
extern ge::graphStatus MoeEpCombineTilingFunc(gert::TilingContext *context);
REGISTER_OPS_TILING_TEMPLATE(MoeEpCombine, MoeEpCombineTilingFunc, 0);
}  // namespace optiling
#include "../../op_kernel/moe_ep_dispatch_tiling_key.h"

namespace optiling {
extern ge::graphStatus MoeEpDispatchTilingFunc(gert::TilingContext *context);
REGISTER_OPS_TILING_TEMPLATE(MoeEpDispatch, MoeEpDispatchTilingFunc, 0);
}  // namespace optiling
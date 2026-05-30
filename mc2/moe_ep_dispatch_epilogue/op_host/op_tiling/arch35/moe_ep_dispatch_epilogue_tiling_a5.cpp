#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling_key.h"

namespace optiling {
extern ge::graphStatus MoeEpDispatchEpilogueTilingFunc(gert::TilingContext *context);
REGISTER_OPS_TILING_TEMPLATE(MoeEpDispatchEpilogue, MoeEpDispatchEpilogueTilingFunc, 0);
}  // namespace optiling
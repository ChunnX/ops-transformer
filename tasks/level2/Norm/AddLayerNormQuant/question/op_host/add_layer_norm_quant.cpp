#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class AddLayerNormQuant : public OpDef {
public:
    explicit AddLayerNormQuant(const char *name) : OpDef(name) {}
};
OP_ADD(AddLayerNormQuant);
} // namespace ops

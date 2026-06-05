#ifndef ENGRAM_FETCH_TILING_ARCH35_H_
#define ENGRAM_FETCH_TILING_ARCH35_H_

#include "../engram_fetch_tiling_base.h"

namespace optiling {
class EngramFetchTilingA5 : public EngramFetchTilingBase {
public:
    explicit EngramFetchTilingA5(gert::TilingContext *context) : EngramFetchTilingBase(context)
    {
        socTilingName_ = "EngramFetchA5";
    }

private:
    ge::graphStatus DoOpTiling() final;
    bool IsCapable() final;
    void SetHcommCfg() final;
};
} // namespace optiling
#endif
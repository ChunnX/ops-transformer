#ifndef ENGRAM_FETCH_TILING_KEY_H
#define ENGRAM_FETCH_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ENGRAM_FETCH_DEFAULT_MODE 0
// 模板参数
ASCENDC_TPL_ARGS_DECL(EngramFetch,
    ASCENDC_TPL_UINT_DECL(engramFetchMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, ENGRAM_FETCH_DEFAULT_MODE),
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(engramFetchMode, ASCENDC_TPL_UI_LIST, ENGRAM_FETCH_DEFAULT_MODE),
    ),
);

#endif // ENGRAM_FETCH_TILING_KEY_H

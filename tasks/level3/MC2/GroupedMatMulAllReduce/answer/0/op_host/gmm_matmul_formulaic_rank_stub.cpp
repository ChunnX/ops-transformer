/**
 * NPUKernelBench: 为 liboptiling 提供 TBE 会解析的符号，避免链接 vendor 全量
 * mc2_tiling_utils.cpp / matmul_formulaic_tiling.cpp（依赖 arch35、ops_legacy 等）。
 */
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/mc2_tiling_utils.h"

#include <cstdlib>

namespace mc2tiling {

namespace {
constexpr char kWorldSizeEnv[] = "WORLD_SIZE";
constexpr char kDebugModeEnv[] = "ASCEND_MC2_DEBUG_MODE";
}  // namespace

uint32_t MatmulFormulaicTiling::GetRankSize(const char *group)
{
    (void)group;
    const char *ws = std::getenv(kWorldSizeEnv);
    if (ws != nullptr && ws[0] != '\0') {
        const int n = std::atoi(ws);
        if (n > 0) {
            return static_cast<uint32_t>(n);
        }
    }
    return 1U;
}

uint8_t Mc2TilingUtils::GetDebugMode()
{
    const char *v = std::getenv(kDebugModeEnv);
    if (v != nullptr && v[0] != '\0') {
        return static_cast<uint8_t>(std::atoi(v));
    }
    return 0;
}

}  // namespace mc2tiling

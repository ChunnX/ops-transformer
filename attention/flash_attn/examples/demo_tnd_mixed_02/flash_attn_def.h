#ifndef FLASH_ATTN_DEF_H_
#define FLASH_ATTN_DEF_H_

#include <cstdint>
#include <string>

// Extracted from: op_host/fa_tiling_info.h (FaLayout enum)
enum class FaLayout : uint32_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    PA_BBND = 3,
    PA_BNBD = 4,
    PA_Nz = 5,
};

// Extracted from: op_host/fa_tiling_info.h (MaskMode enum)
enum class MaskMode : int32_t {
    NO_MASK = 0,
    CAUSAL = 3
};

// Extracted from: op_kernel/utils/flash_attn_common_def.h (InOutLayoutType defines)
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
#define InOutLayoutType_BNSD_BSND 3

// Extracted from: op_kernel/utils/flash_attn_common_def.h (KvLayoutType defines)
#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

// Extracted from: op_host/fa_tiling_info.h (MASK_MODE_INT_MAX)
constexpr int64_t MASK_MODE_INT_MAX = 2147483647;

// Extracted from: op_host/fa_adjust_sinner_souter.h (tiling block size constants)
constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_256 = 256;

// Extracted from: op_host/arch35/flash_attn_tiling.h (FaTilingKeyInfo struct)
struct FaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t kvLayoutType = 0;
    bool hasAttenMask = false;
    uint64_t config = 0;
};

// Extracted from: op_host/fa_tiling_info.h (FaTilingInfo base params)
struct CaseParams {
    int64_t bSize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t qkHeadDim;
    int64_t vHeadDim;
    int64_t maskMode;
    int64_t winLeft;
    int64_t winRight;
    FaLayout qLayout;
    FaLayout kvLayout;
    FaLayout outLayout;
};

#endif // FLASH_ATTN_DEF_H_

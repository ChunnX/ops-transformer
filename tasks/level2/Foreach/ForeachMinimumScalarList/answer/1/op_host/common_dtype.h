#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_COMMON_DTYPE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_COMMON_DTYPE_H_

#include <string>
#include <sstream>

namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_BLOCK_FOR_BF16 = 64; 
constexpr uint32_t BYTE_REPEAT = 256;
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr uint32_t BYTE_LEN_8 = 8;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;
constexpr uint32_t BYTE_LEN_1 = 1;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_INT = 3;
constexpr uint64_t TILING_KEY_BF16 = 4;

inline uint8_t GetDataTypeSize(ge::DataType dataType) {
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

inline uint64_t GetTilingKeyByDtypeOnly(ge::DataType dataType) {
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_INT32:
            return TILING_KEY_INT;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        default:
            return 0;
    }
}

template <typename T>
std::string ConcatString(const T& arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(const T& arg, const Ts&... argLeft)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(argLeft...);
    return oss.str();
}

} // namespace optiling

#endif
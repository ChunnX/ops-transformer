#ifndef AICPU_STUB_CPU_TYPES_H
#define AICPU_STUB_CPU_TYPES_H

#include <map>

namespace aicpu {

#ifdef VISIBILITY
#define AICPU_VISIBILITY __attribute__((visibility("default")))
#else
#define AICPU_VISIBILITY
#endif

enum DataType {
    DT_FLOAT = 0,
    DT_FLOAT16 = 1,
    DT_INT8 = 2,
    DT_INT16 = 6,
    DT_UINT16 = 7,
    DT_UINT8 = 4,
    DT_INT32 = 3,
    DT_INT64 = 9,
    DT_UINT32 = 8,
    DT_UINT64 = 10,
    DT_BOOL = 12,
    DT_DOUBLE = 11,
    DT_STRING = 13,
    DT_DUAL = 25,
    DT_BFLOAT16 = 27,
    DT_UNDEFINED = 28,
    DT_MAX
};

enum Format {
    FORMAT_NCHW = 0,
    FORMAT_NHWC,
    FORMAT_ND,
    FORMAT_NC1HWC0,
    FORMAT_FRACTAL_Z,
    FORMAT_MAX = 0xff
};

enum DeviceType {
    HOST,
    DEVICE
};

} // namespace aicpu

#endif // AICPU_STUB_CPU_TYPES_H

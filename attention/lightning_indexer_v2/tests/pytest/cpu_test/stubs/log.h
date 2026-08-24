#ifndef AICPU_STUB_LOG_H
#define AICPU_STUB_LOG_H

#include <cstdio>
#include <cstdint>

#ifndef RUN_TEST
#define RUN_TEST
#endif

#define KERNEL_LOG_DEBUG(fmt, ...) ((void)0)
#define KERNEL_LOG_INFO(fmt, ...)  ((void)0)
#define KERNEL_LOG_WARN(fmt, ...) \
    printf("[WARN] [%s:%d]: " fmt "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...) \
    printf("[ERROR] [%s:%d]: " fmt "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...) ((void)0)

#define KERNEL_CHECK_NULLPTR_VOID(ptr) \
    do { if ((ptr) == nullptr) { return; } } while(0)
#define KERNEL_CHECK_FALSE(cond) \
    do { if (!(cond)) { return false; } } while(0)
#define KERNEL_CHECK_NULLPTR(ptr, ret) \
    do { if ((ptr) == nullptr) { return (ret); } } while(0)

#endif  // AICPU_STUB_LOG_H

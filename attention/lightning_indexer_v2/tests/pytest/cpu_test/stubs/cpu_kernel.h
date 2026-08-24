#ifndef AICPU_STUB_CPU_KERNEL_H
#define AICPU_STUB_CPU_KERNEL_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "cpu_context.h"

namespace aicpu {

class CpuKernel {
public:
    virtual uint32_t Compute(CpuKernelContext &ctx) = 0;
    virtual ~CpuKernel() {}
};

using KERNEL_CREATOR_FUN = std::function<std::shared_ptr<CpuKernel>(void)>;

inline bool RegistCpuKernel(const std::string &type, const KERNEL_CREATOR_FUN &fun)
{
    // Stub: registration is a no-op in CPU test mode
    (void)type;
    (void)fun;
    return true;
}

template <typename T, typename... Args>
static inline std::shared_ptr<T> MakeShared(Args &&... args)
{
    using T_NC = typename std::remove_const<T>::type;
    std::shared_ptr<T> ret(new (std::nothrow) T_NC(std::forward<Args>(args)...));
    return ret;
}

#define REGISTER_CPU_KERNEL(type, clazz) \
    std::shared_ptr<CpuKernel> Creator_##type##_Kernel() \
    { \
        std::shared_ptr<clazz> ptr = nullptr; \
        ptr = MakeShared<clazz>(); \
        return ptr; \
    } \
    bool g_##type##_Kernel_Creator __attribute__((unused)) = RegistCpuKernel(type, Creator_##type##_Kernel)

} // namespace aicpu

#endif // AICPU_STUB_CPU_KERNEL_H

// CPU-only replacement for the two integer arithmetic helpers used by layout.
// This intentionally has no CANN runtime or device emulation behavior.
#ifndef MEGA_MOE_CPU_RING_MATH_UTIL_STUB
#define MEGA_MOE_CPU_RING_MATH_UTIL_STUB
namespace Ops {
namespace Base {
template <typename T, typename U>
inline T CeilDiv(T value, U divisor)
{
    return static_cast<T>(value / divisor + (value % divisor != 0));
}
template <typename T, typename U>
inline T CeilAlign(T value, U alignment)
{
    return CeilDiv(value, alignment) * static_cast<T>(alignment);
}
} // namespace Base
} // namespace Ops
#endif

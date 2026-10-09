// Minimal freestanding test adapter, used only for the extracted layout UTs.
// These declarations are not a substitute for the normal gtest/CANN harness.
#ifndef MEGA_MOE_CPU_RING_TEST_SUPPORT_H
#define MEGA_MOE_CPU_RING_TEST_SUPPORT_H
#include <cstdint>
namespace std {
template <typename T>
class initializer_list {
    const T *values_;
    const T *last_;
public:
    constexpr initializer_list() : values_(nullptr), last_(nullptr) {}
    constexpr initializer_list(const T *values, const T *last) : values_(values), last_(last) {}
    constexpr const T *begin() const { return values_; }
    constexpr const T *end() const { return last_; }
    constexpr uint64_t size() const { return last_ - values_; }
};
template <typename T>
constexpr const T *begin(initializer_list<T> values) { return values.begin(); }
template <typename T>
constexpr const T *end(initializer_list<T> values) { return values.end(); }
template <typename T>
struct numeric_limits;
template <>
struct numeric_limits<uint32_t> {
    static constexpr uint32_t max() { return 0xFFFFFFFFU; }
};
template <typename T>
constexpr const T &min(const T &left, const T &right) { return left < right ? left : right; }
template <typename T>
constexpr const T &max(const T &left, const T &right) { return left > right ? left : right; }
}
static int CpuCheckFailureLine = 0;
#define TEST_F(fixture, name) static void name()
#define CPU_EXPECT(expr) do { if (!(expr) && CpuCheckFailureLine == 0) CpuCheckFailureLine = __LINE__; } while (0)
#define CPU_ASSERT(expr) do { if (!(expr)) { if (CpuCheckFailureLine == 0) CpuCheckFailureLine = __LINE__; return; } } while (0)
#define EXPECT_EQ(left, right) CPU_EXPECT((left) == (right))
#define EXPECT_LT(left, right) CPU_EXPECT((left) < (right))
#define EXPECT_LE(left, right) CPU_EXPECT((left) <= (right))
#define EXPECT_GT(left, right) CPU_EXPECT((left) > (right))
#define EXPECT_GE(left, right) CPU_EXPECT((left) >= (right))
#define EXPECT_TRUE(expr) CPU_EXPECT(expr)
#define EXPECT_FALSE(expr) CPU_EXPECT(!(expr))
#define ASSERT_TRUE(expr) CPU_ASSERT(expr)
#define ASSERT_GE(left, right) CPU_ASSERT((left) >= (right))
#endif

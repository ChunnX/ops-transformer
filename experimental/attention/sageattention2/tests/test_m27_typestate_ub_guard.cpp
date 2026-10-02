// test_m27_typestate_ub_guard.cpp — M2.7 架构特性（Typestate、Zero-Stall、UB Hard Guard、2D Tile 归约）单元验证。
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <type_traits>
#include <vector>

#include "../op_kernel/sa2_spec.h"
#include "../op_kernel/sa2_events.h"
#include "../op_kernel/sa2_ub_plan.h"
#include "../sim/sa2_sim_loop.h"
#include "../op_host/sa2_tiling.h"

template <typename StateTag, typename = void>
struct SlotDataDetected : std::false_type {};
template <typename StateTag>
struct SlotDataDetected<StateTag, std::void_t<decltype(sa2::SlotData<float>(std::declval<const sa2::SlotRef<StateTag> &>()))>>
    : std::true_type {};

// 验证 M2.7.4 Typestate 编译期约束
void TestTypestateSemantics() {
    std::cout << "[Test 1] Testing Typestate (DmaPending & ReadyTensor)..." << std::endl;

    // 1. DmaPending 必须是 Move-Only 类型（拷贝构造与拷贝赋值被 delete）
    static_assert(!std::is_copy_constructible<sa2::DmaPending<float>>::value, "DmaPending must not be copy constructible");
    static_assert(!std::is_copy_assignable<sa2::DmaPending<float>>::value, "DmaPending must not be copy assignable");
    static_assert(std::is_move_constructible<sa2::DmaPending<float>>::value, "DmaPending must be move constructible");
    static_assert(std::is_move_assignable<sa2::DmaPending<float>>::value, "DmaPending must be move assignable");

    // 2. CrossEnginePending 必须是 Move-Only 类型
    static_assert(!std::is_copy_constructible<sa2::CrossEnginePending<float, sa2::Sa2CrossFlag::kScoreOk>>::value,
                  "CrossEnginePending must not be copy constructible");
    static_assert(!std::is_copy_assignable<sa2::CrossEnginePending<float, sa2::Sa2CrossFlag::kScoreOk>>::value,
                  "CrossEnginePending must not be copy assignable");
    static_assert(std::is_move_constructible<sa2::CrossEnginePending<float, sa2::Sa2CrossFlag::kScoreOk>>::value,
                  "CrossEnginePending must be move constructible");
    static_assert(std::is_move_assignable<sa2::CrossEnginePending<float, sa2::Sa2CrossFlag::kScoreOk>>::value,
                  "CrossEnginePending must be move assignable");

    // 3. ReadyTensor 必须是 Move-Only 类型（严禁拷贝逃逸，确保就绪阶段独占所有权）
    static_assert(!std::is_copy_constructible<sa2::ReadyTensor<float>>::value, "ReadyTensor must not be copy constructible");
    static_assert(!std::is_copy_assignable<sa2::ReadyTensor<float>>::value, "ReadyTensor must not be copy assignable");
    static_assert(std::is_move_constructible<sa2::ReadyTensor<float>>::value, "ReadyTensor must be move constructible");
    static_assert(std::is_move_assignable<sa2::ReadyTensor<float>>::value, "ReadyTensor must be move assignable");

    // 3. 运行期状态跃迁测试：DmaPending -> Sync() && -> ReadyTensor
    float rawData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    sa2::DmaPending<float> pending(rawData, 4);

    // 必须通过 std::move(pending).Sync() 显式跃迁
    sa2::ReadyTensor<float> ready = std::move(pending).Sync();
    assert(ready.IsValid());
    assert(ready.Size() == 4);
    assert(ready[0] == 1.0f);
    assert(ready[3] == 4.0f);

    // 4. CrossEnginePending -> Await() && -> ReadyTensor
    sa2::CrossEnginePending<float, sa2::Sa2CrossFlag::kScoreOk> crossPending(rawData, 4, 0);
    sa2::ReadyTensor<float> crossReady = std::move(crossPending).Await();
    assert(crossReady.IsValid());
    assert(crossReady[2] == 3.0f);

    // 5. StateTypedSlot 线性类型状态机（Empty -> AcquireSlot -> Full -> ReleaseSlot -> Empty）
    // 编译期静态断言：Empty 态无 SlotData 数据访问（编译期物理堵死未同步先读写）
    static_assert(!SlotDataDetected<sa2::Empty>::value, "SlotData must NOT be accessible on Empty slot");
    static_assert(SlotDataDetected<sa2::Full>::value, "SlotData must be accessible on Full slot");
    static_assert(!std::is_default_constructible<sa2::SlotRef<sa2::Full>>::value, "SlotRef<Full> default ctor must be private");
    static_assert(!std::is_copy_constructible<sa2::SlotRef<sa2::Full>>::value, "SlotRef<Full> must not be copy constructible");
    static_assert(!std::is_copy_assignable<sa2::SlotRef<sa2::Full>>::value, "SlotRef<Full> must not be copy assignable");
    static_assert(std::is_move_constructible<sa2::SlotRef<sa2::Full>>::value, "SlotRef<Full> must be move constructible");
    static_assert(std::is_move_assignable<sa2::SlotRef<sa2::Full>>::value, "SlotRef<Full> must be move assignable");
    static_assert(!std::is_copy_constructible<sa2::SlotRef<sa2::Empty>>::value, "SlotRef<Empty> must not be copy constructible");
    static_assert(!std::is_copy_assignable<sa2::SlotRef<sa2::Empty>>::value, "SlotRef<Empty> must not be copy assignable");
    static_assert(std::is_move_constructible<sa2::SlotRef<sa2::Empty>>::value, "SlotRef<Empty> must be move constructible");

    // 运行期状态跃迁测试：AcquireSlot -> Full -> ReleaseSlot -> Empty
    alignas(64) float slotBuf[16] = {0.0f};
    auto fullSlot = sa2::AcquireSlot<sa2::Sa2CrossFlag::kPvOk>(slotBuf, sizeof(slotBuf), 0);
    assert(fullSlot.IsValid());
    assert(fullSlot.SizeBytes() == sizeof(slotBuf));
    float *fullData = sa2::SlotData<float>(fullSlot);
    assert(fullData == slotBuf);
    auto emptySlot = sa2::ReleaseSlot<sa2::Sa2CrossFlag::kPvOk>(std::move(fullSlot));
    assert(!fullSlot.IsValid()); // moved-from 句柄已被安全擦除
    assert(emptySlot.IsValid());

    std::cout << "  -> Typestate move semantics & state transition PASSED." << std::endl;
}

// 验证 M2.7.3（v27 重注记）：宿主 double+Kahan 仿真求和的收敛性（1e-9 界为宿主仿真口径；真机 VF 归约误差 = M5.1 实测项）
void TestTileFusedReductionsPrecision() {
    std::cout << "[Test 2] Testing host-sim fused reduction convergence (double domain, 1e-9 bound)..." << std::endl;

    constexpr int64_t cols = 128;
    float rowS[cols];
    // 构造具有较大动态范围的测试输入，模拟注意力得分分布
    for (int64_t c = 0; c < cols; ++c) {
        rowS[c] = -20.0f + static_cast<float>(c) * 0.15f;
    }

    // 1. TROWMAX 测试
    float mBlk = sa2::sim::TileFusedReductionsSim::TileRowMax<cols>(rowS);
    float expectedMax = rowS[cols - 1];
    assert(std::fabs(mBlk - expectedMax) < 1e-6f);

    // 2. TROWEXPANDSUB_EXP & 高精求和测试
    uint8_t pUb[cols];
    uint8_t pL1[cols];
    float sumP8 = 0.0f;
    float mNew = mBlk;

    sa2::sim::TileFusedReductionsSim::TileRowExpandSubExpFp8<cols>(rowS, mNew, pUb, pL1, sumP8);

    // 使用双精度真值对照高精求和累加器
    double trueSum = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
        float pVal = std::exp(rowS[c] - mNew);
        float pClamped = std::max(0.0f, std::min(448.0f, pVal));
        uint8_t code = sa2::FloatToFp8E4M3(pClamped);
        float pDeq = sa2::Fp8E4M3ToFloat(code);
        trueSum += static_cast<double>(pDeq);
    }

    // 计算硬件高精累加器与理论双精度真值之间的相对截断误差
    double absDiff = std::fabs(static_cast<double>(sumP8) - trueSum);
    double relDiff = absDiff / trueSum;

    std::printf("  -> sumP8: %.7f, trueSum: %.7f, absDiff: %.2e, relDiff: %.2e\n",
                sumP8, trueSum, absDiff, relDiff);

    // 宿主 double 域：Kahan 求和相对真值应在 1e-9 以内（真机 VF 单指令归约的实际误差 = M5.1 实测项）
    assert(relDiff < 1e-9);

    std::cout << "  -> 2D Tile Fused Reductions precision PASSED." << std::endl;
}

// 验证 M2.7.2 动态 UB 水位绝对硬防护（O(1) 常数性与 160KB/248KB 边界，S 从 128 扩展到 131072）
void TestUbCapacityHardGuard() {
    std::cout << "[Test 3] Testing UB Capacity Hard Guard (O(1) proof for S from 128 to 131072)..." << std::endl;

    // 验证静态断言常数
    static_assert(sa2::kUbEndPerWarp == 152448, "per_warp must be 152448 B");
    static_assert(sa2::kUbEndPerThread == 217984, "per_thread must be 217984 B");
    static_assert(sa2::kUbEndPerWarp <= sa2::kUbHardSafetyLimit, "per_warp must be <= 160KB");
    static_assert(sa2::kUbEndPerThread <= sa2::kUbHighWater, "per_thread must be <= 248KB");
    static_assert(sa2::kUbEndPerThread + 8192 <= sa2::kUbPhysicalLimit, "must preserve 8KB HW control registers");
    static_assert(sa2::kUbPhysicalLimit - sa2::kUbEndPerThread == 44160, "must preserve exactly 44160 B HW control registers");
    static_assert(sa2::kUbPhysicalLimit - sa2::kUbEndPerThread >= 8192, "must preserve >= 8KB HW control registers");

    std::cout << "  -> per_warp UB Slab: " << sa2::kUbEndPerWarp << " B ("
              << (sa2::kUbEndPerWarp / 1024.0) << " KB) <= 160 KB safety limit." << std::endl;
    std::cout << "  -> per_thread UB Slab: " << sa2::kUbEndPerThread << " B ("
              << (sa2::kUbEndPerThread / 1024.0) << " KB) <= 248 KB OS limit." << std::endl;
    std::cout << "  -> HW Control Register Protection Zone: >= "
              << (sa2::kUbPhysicalLimit - sa2::kUbEndPerThread) << " B (Safe > 8KB, exactly 44160 B)." << std::endl;

    // 形式化验证长序列 S 从 128 扩展到 131072 时，独立推导的 TileConfig UB 占用严格恒定（真 O(1) 证明）
    {
        const uint32_t kSeqLengths[] = {128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
        for (uint32_t gran = 0; gran <= 1; ++gran) {
            const uint64_t expectedFootprint = (gran == 1) ? sa2::kUbEndPerThread : sa2::kUbEndPerWarp;
            const uint64_t expectedPeak = (gran == 1)
                ? (sa2::kUbScoreThreadSlot + sa2::kUbPSlot + sa2::kUbPvSlot + sa2::kUbOAcc + sa2::kUbSegSlot + sa2::kUbStateBytes)
                : (sa2::kUbScoreWarpSlot + sa2::kUbPSlot + sa2::kUbPvSlot + sa2::kUbOAcc + sa2::kUbStateBytes);
            for (uint32_t s : kSeqLengths) {
                const optiling::Sa2Dims dims{1, 32, 32, s, s, 128};
                const auto cfg = optiling::Sa2Optiling::DeriveTileConfig(dims, gran, 128);
                assert(cfg.ubFootprintPerStep == expectedFootprint);
                assert(cfg.ubInstantaneousPeak == expectedPeak);
                assert(cfg.ubFootprintPerStep <= cfg.ubHighWater);
                assert(cfg.ubInstantaneousPeak <= cfg.ubHardSafetyLimit);
                assert(cfg.ubPhysicalLimit - cfg.ubFootprintPerStep >= 44160ULL);
                assert(cfg.ubFootprintPerStep + cfg.hardwareReservedHeadroom <= cfg.ubPhysicalLimit);
            }
        }
        std::cout << "  -> O(1) Constant UB residency across S in [128, 131072] verified (per_warp=152448B, per_thread=217984B, headroom>=44160B)."
                  << std::endl;
    }
}

int main() {
    std::cout << "=== Running M2.7 Architecture & Typestate Verification Suite ===" << std::endl;
    TestTypestateSemantics();
    TestTileFusedReductionsPrecision();
    TestUbCapacityHardGuard();
    std::cout << "=== All M2.7 Architectural Assertions & Tests PASSED! ===" << std::endl;
    return 0;
}

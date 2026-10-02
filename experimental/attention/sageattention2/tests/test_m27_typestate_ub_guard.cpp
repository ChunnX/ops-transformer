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

// 验证 M2.7.2 动态 UB 水位绝对硬防护（O(1) 常数性与 160KB/248KB 边界）
void TestUbCapacityHardGuard() {
    std::cout << "[Test 3] Testing UB Capacity Hard Guard (O(1) proof for S>=16384)..." << std::endl;

    // 验证静态断言常数
    static_assert(sa2::kUbEndPerWarp == 152448, "per_warp must be 152448 B");
    static_assert(sa2::kUbEndPerThread == 217984, "per_thread must be 217984 B");
    static_assert(sa2::kUbEndPerWarp <= sa2::kUbHardSafetyLimit, "per_warp must be <= 160KB");
    static_assert(sa2::kUbEndPerThread <= sa2::kUbHighWater, "per_thread must be <= 248KB");
    static_assert(sa2::kUbEndPerThread + 8192 <= sa2::kUbPhysicalLimit, "must preserve 8KB HW control registers");

    std::cout << "  -> per_warp UB Slab: " << sa2::kUbEndPerWarp << " B ("
              << (sa2::kUbEndPerWarp / 1024.0) << " KB) <= 160 KB safety limit." << std::endl;
    std::cout << "  -> per_thread UB Slab: " << sa2::kUbEndPerThread << " B ("
              << (sa2::kUbEndPerThread / 1024.0) << " KB) <= 248 KB OS limit." << std::endl;
    std::cout << "  -> HW Control Register Protection Zone: >= "
              << (sa2::kUbPhysicalLimit - sa2::kUbEndPerThread) << " B (Safe > 8KB)." << std::endl;

    // v27 P1-7：验证长序列 S 从 128 扩展到 16384 时，独立推导的 TileConfig UB 占用恒定（真 O(1) 检查，
    // 替换旧版同变量自比恒真式）
    {
        const optiling::Sa2Dims s128{1, 32, 32, 128, 128, 128};
        const optiling::Sa2Dims s4096{1, 32, 32, 4096, 4096, 128};
        const optiling::Sa2Dims s8192{1, 32, 32, 8192, 8192, 128};
        const optiling::Sa2Dims s16384{1, 32, 32, 16384, 16384, 128};
        const auto cfg128 = optiling::Sa2Optiling::DeriveTileConfig(s128, 1, 128);
        const auto cfg4096 = optiling::Sa2Optiling::DeriveTileConfig(s4096, 1, 128);
        const auto cfg8192 = optiling::Sa2Optiling::DeriveTileConfig(s8192, 1, 128);
        const auto cfg16384 = optiling::Sa2Optiling::DeriveTileConfig(s16384, 1, 128);
        assert(cfg128.ubFootprintPerStep == cfg16384.ubFootprintPerStep);
        assert(cfg4096.ubFootprintPerStep == cfg16384.ubFootprintPerStep);
        assert(cfg8192.ubFootprintPerStep == cfg16384.ubFootprintPerStep);
        assert(cfg128.ubInstantaneousPeak == cfg16384.ubInstantaneousPeak);
        std::cout << "  -> O(1) Constant UB residency across S in [128, 16384] verified (independent configs)."
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

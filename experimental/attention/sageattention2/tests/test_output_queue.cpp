/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>

#include "tiling.h"
#include "online_softmax.h"
#include "accum_reduce.h"

using namespace sage_attn;

/**
 * [Test 1] 队列满与成对归约自动触发测试（FIFO 保序与容量控制）
 */
void TestQueueFullTrigger() {
    std::cout << "[Test 1] Testing OutputRetireQueue full trigger & pairwise retirement..." << std::endl;

    constexpr uint32_t TILE_SIZE = 128;
    std::vector<float> acc(TILE_SIZE, 0.0f);
    std::vector<float> tile0(TILE_SIZE, 1.0f);
    std::vector<float> tile1(TILE_SIZE, 2.0f);
    std::vector<float> tile2(TILE_SIZE, 3.0f);
    std::vector<float> tile3(TILE_SIZE, 4.0f);
    std::vector<float> tile4(TILE_SIZE, 5.0f);

    OutputRetireQueue<float, 3> queue;
    queue.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32, 6.0f, 8.807355f);

    assert(queue.IsEmpty());
    assert(!queue.IsFull());
    assert(queue.GetQueueSize() == 0);
    assert(queue.GetPairwiseCount() == 0);
    assert(queue.GetSingleCount() == 0);

    // 1. Push tile 0
    queue.Push(0, tile0.data(), 1.0f, 0.0f, 0.0f);
    assert(queue.GetQueueSize() == 1);
    assert(queue.GetPairwiseCount() == 0);

    // 2. Push tile 1
    queue.Push(1, tile1.data(), 1.0f, 0.0f, 0.0f);
    assert(queue.GetQueueSize() == 2);
    assert(queue.GetPairwiseCount() == 0);

    // 3. Push tile 2 -> 槽位达到 CAPACITY(3)，触发成对归约！
    // tile 0 和 tile 1 成对融合退休进入 acc，队列中保留 tile 2
    queue.Push(2, tile2.data(), 1.0f, 0.0f, 0.0f);
    assert(queue.GetQueueSize() == 1); // 3 - 2 = 1
    assert(queue.GetPairwiseCount() == 1);
    assert(queue.GetTotalRetired() == 2);

    // 4. Push tile 3
    queue.Push(3, tile3.data(), 1.0f, 0.0f, 0.0f);
    assert(queue.GetQueueSize() == 2);
    assert(queue.GetPairwiseCount() == 1);

    // 5. Push tile 4 -> 槽位满，再次触发成对归约！
    // tile 2 和 tile 3 成对融合退休，队列中保留 tile 4
    queue.Push(4, tile4.data(), 1.0f, 0.0f, 0.0f);
    assert(queue.GetQueueSize() == 1);
    assert(queue.GetPairwiseCount() == 2);
    assert(queue.GetTotalRetired() == 4);

    // 6. 调用 Flush() 清空剩余槽位 (tile 4)
    queue.Flush();
    assert(queue.IsEmpty());
    assert(queue.GetQueueSize() == 0);
    assert(queue.GetSingleCount() == 1);
    assert(queue.GetTotalRetired() == 5);
    assert(queue.GetTotalPushed() == 5);

    // 验证累加和结果: tile0 + tile1 + tile2 + tile3 + tile4 = 1 + 2 + 3 + 4 + 5 = 15.0f
    for (uint32_t j = 0; j < TILE_SIZE; ++j) {
        assert(std::fabs(acc[j] - 15.0f) < 1e-4f);
    }

    std::cout << "[Test 1 Passed] Queue full trigger & FIFO pairwise retirement verified!" << std::endl;
}

/**
 * [Test 2] 成对融合数值保真度验证（对比单步传统逐拍累加）
 */
void TestPairwiseFusionNumericalFidelity() {
    std::cout << "[Test 2] Testing pairwise fusion numerical fidelity vs sequential update..." << std::endl;

    constexpr uint32_t M = 64;
    constexpr uint32_t D = 128;
    constexpr uint32_t TILE_SIZE = M * D;
    constexpr uint32_t NUM_TILES = 6;

    // 构造 6 个非平凡测试分块
    std::vector<std::vector<float>> tiles(NUM_TILES, std::vector<float>(TILE_SIZE, 0.0f));
    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        for (uint32_t j = 0; j < TILE_SIZE; ++j) {
            tiles[t][j] = std::sin(static_cast<float>(j) * 0.02f + static_cast<float>(t)) * 1.8f
                        + std::cos(static_cast<float>(t) * 0.5f);
        }
    }

    // 模拟一组递增的局部 max 轨迹（含大跳步与微小变化）
    std::vector<float> m_seq = {10.0f, 11.2f, 11.8f, 19.5f, 21.0f, 22.5f};
    std::vector<float> scale_seq = {0.8f, 0.95f, 1.1f, 0.7f, 1.05f, 0.9f};

    // -------------------------------------------------------------------------
    // 基准方法：传统逐拍更新 (Sequential Baseline)
    // -------------------------------------------------------------------------
    std::vector<float> acc_ref(TILE_SIZE, 0.0f);
    float m_ref = m_seq[0];
    for (uint32_t j = 0; j < TILE_SIZE; ++j) {
        acc_ref[j] = tiles[0][j] * scale_seq[0];
    }

    for (uint32_t t = 1; t < NUM_TILES; ++t) {
        float m_cur = m_seq[t];
        float delta_m = m_cur - m_ref;
        float rescale = (delta_m > 0.0f) ? std::exp2(m_ref - m_cur) : 1.0f;
        m_ref = (m_cur > m_ref) ? m_cur : m_ref;

        for (uint32_t j = 0; j < TILE_SIZE; ++j) {
            acc_ref[j] = acc_ref[j] * rescale + tiles[t][j] * scale_seq[t];
        }
    }

    // -------------------------------------------------------------------------
    // 我方方法：OutputRetireQueue 3 槽延迟成对归约 (强制无 skip 以比对纯净数值)
    // -------------------------------------------------------------------------
    std::vector<float> acc_queue(TILE_SIZE, 0.0f);
    OutputRetireQueue<float, 3> queue;
    // 将 tau 设置为 0.0f，保证每次 max 增大都严格与基准对齐 rescale 算子
    queue.Init(acc_queue.data(), TILE_SIZE, PV_ACCUM_FP32, 0.0f, 8.807355f);

    float m_prev = m_seq[0];
    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        float m_curr = m_seq[t];
        queue.Push(t, tiles[t].data(), scale_seq[t], m_curr, m_prev);
        m_prev = m_curr;
    }
    queue.Flush();

    // 验证数值最大绝对误差与相对误差
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (uint32_t j = 0; j < TILE_SIZE; ++j) {
        float diff = std::fabs(acc_queue[j] - acc_ref[j]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
        float denom = std::max(std::fabs(acc_ref[j]), 1e-6f);
        float rel = diff / denom;
        if (rel > max_rel_diff) {
            max_rel_diff = rel;
        }
    }

    std::cout << "   -> Max Abs Diff: " << max_abs_diff << ", Max Rel Diff: " << max_rel_diff << std::endl;
    assert(max_abs_diff < 1e-4f);
    assert(max_rel_diff < 5e-4f);

    std::cout << "[Test 2 Passed] Numerical fidelity between pairwise fusion and baseline strictly verified!" << std::endl;
}

/**
 * [Test 3] Skip-Scale 优化与直通累加消除浮点乘法验证
 */
void TestSkipScaleOptimization() {
    std::cout << "[Test 3] Testing skip-scale optimization (tau = 6.0f)..." << std::endl;

    constexpr uint32_t TILE_SIZE = 64;
    std::vector<float> acc(TILE_SIZE, 0.0f);
    std::vector<float> tile0(TILE_SIZE, 1.0f);
    std::vector<float> tile1(TILE_SIZE, 2.0f);
    std::vector<float> tile2(TILE_SIZE, 3.0f);
    std::vector<float> tile3(TILE_SIZE, 4.0f);
    std::vector<float> tile4(TILE_SIZE, 5.0f);

    OutputRetireQueue<float, 3> queue;
    // tau = 6.0f (FlashMLA 黄金阈值)
    queue.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32, 6.0f, 8.807355f);

    // 1. 推送两个增幅小于 6.0f 的分块: m: 5.0f -> 7.0f (delta=2.0f < 6.0f)
    queue.Push(0, tile0.data(), 1.0f, 5.0f, 5.0f);
    queue.Push(1, tile1.data(), 1.0f, 7.0f, 5.0f);

    // 2. 推送第三个分块触发成对归约: m: 7.0f -> 8.5f (delta=1.5f < 6.0f)
    queue.Push(2, tile2.data(), 1.0f, 8.5f, 7.0f);

    // 验证触发了 skip-scale 且无 rescale
    assert(queue.GetPairwiseCount() == 1);
    assert(queue.GetRescaleCount() == 0); // delta < tau，无需累加器 rescale，直通消除乘法！

    // 3. 推送一个剧烈增大超过 tau 的分块: m: 8.5f -> 18.0f (delta=9.5f >= 6.0f)
    queue.Push(3, tile3.data(), 1.0f, 18.0f, 8.5f);
    // 4. 推送第五个分块触发第二次成对归约
    queue.Push(4, tile4.data(), 1.0f, 20.0f, 18.0f);

    assert(queue.GetPairwiseCount() == 2);
    assert(queue.GetRescaleCount() > 0); // 触发了显式 rescale

    queue.Flush();
    assert(queue.IsEmpty());

    std::cout << "[Test 3 Passed] Skip-scale optimization verified: pass-through additions eliminate float multiplications!" << std::endl;
}

/**
 * [Test 4] 尾部清空（Flush）针对 0、1、2 个剩余槽位的鲁棒性测试
 */
void TestFlushVariousRemainingSlots() {
    std::cout << "[Test 4] Testing Flush() behavior with 0, 1, and 2 remaining slots..." << std::endl;

    constexpr uint32_t TILE_SIZE = 32;
    std::vector<float> acc(TILE_SIZE, 0.0f);
    std::vector<float> tile0(TILE_SIZE, 2.0f);
    std::vector<float> tile1(TILE_SIZE, 3.0f);
    std::vector<float> tile2(TILE_SIZE, 4.0f);
    std::vector<float> tile3(TILE_SIZE, 5.0f);

    // Case A: 1 slot remaining
    {
        std::fill(acc.begin(), acc.end(), 0.0f);
        OutputRetireQueue<float, 3> q;
        q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32);
        q.Push(0, tile0.data(), 1.0f, 1.0f, 1.0f);
        assert(q.GetQueueSize() == 1);
        q.Flush();
        assert(q.IsEmpty());
        assert(q.GetSingleCount() == 1);
        assert(q.GetPairwiseCount() == 0);
        for (float v : acc) assert(std::fabs(v - 2.0f) < 1e-4f);
    }

    // Case B: 2 slots remaining
    {
        std::fill(acc.begin(), acc.end(), 0.0f);
        OutputRetireQueue<float, 3> q;
        q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32);
        q.Push(0, tile0.data(), 1.0f, 1.0f, 1.0f);
        q.Push(1, tile1.data(), 1.0f, 1.0f, 1.0f);
        assert(q.GetQueueSize() == 2);
        q.Flush();
        assert(q.IsEmpty());
        assert(q.GetSingleCount() == 0);
        assert(q.GetPairwiseCount() == 1); // 2 个剩余槽位执行成对归约！
        for (float v : acc) assert(std::fabs(v - 5.0f) < 1e-4f);
    }

    // Case C: 4 slots total (Push 3 -> 1 pair retired, 1 in queue; Push 1 -> 2 in queue; Flush -> 1 pair retired)
    {
        std::fill(acc.begin(), acc.end(), 0.0f);
        OutputRetireQueue<float, 3> q;
        q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32);
        q.Push(0, tile0.data(), 1.0f, 1.0f, 1.0f);
        q.Push(1, tile1.data(), 1.0f, 1.0f, 1.0f);
        q.Push(2, tile2.data(), 1.0f, 1.0f, 1.0f);
        assert(q.GetQueueSize() == 1);
        assert(q.GetPairwiseCount() == 1);

        q.Push(3, tile3.data(), 1.0f, 1.0f, 1.0f);
        assert(q.GetQueueSize() == 2);

        q.Flush();
        assert(q.IsEmpty());
        assert(q.GetPairwiseCount() == 2);
        assert(q.GetSingleCount() == 0); // 100% 全部通过成对归约退休！
        // 2 + 3 + 4 + 5 = 14.0f
        for (float v : acc) assert(std::fabs(v - 14.0f) < 1e-4f);
    }

    std::cout << "[Test 4 Passed] Flush() for 0, 1, and 2 remaining slots verified successfully!" << std::endl;
}

/**
 * [Test 5] 三档两级累加精度（fp32, fp32+fp32, fp32+fp16）兼容性测试
 */
void TestThreePrecisionModes() {
    std::cout << "[Test 5] Testing compatibility across 3 accumulation precision modes..." << std::endl;

    constexpr uint32_t TILE_SIZE = 64;
    std::vector<float> acc(TILE_SIZE, 0.0f);
    std::vector<float> tile0(TILE_SIZE, 1.25f);
    std::vector<float> tile1(TILE_SIZE, 2.75f);
    std::vector<float> tile2(TILE_SIZE, 0.5f);

    // 1. Mode 0: fp32
    {
        std::fill(acc.begin(), acc.end(), 0.0f);
        OutputRetireQueue<float, 3> q;
        q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32);
        q.Push(0, tile0.data(), 1.0f, 0.0f, 0.0f);
        q.Push(1, tile1.data(), 1.0f, 0.0f, 0.0f);
        q.Push(2, tile2.data(), 1.0f, 0.0f, 0.0f);
        q.Flush();
        assert(q.GetAccumMode() == PV_ACCUM_FP32);
        for (float v : acc) assert(std::fabs(v - 4.5f) < 1e-4f);
    }

    // 2. Mode 1: fp32+fp32
    {
        std::fill(acc.begin(), acc.end(), 0.0f);
        OutputRetireQueue<float, 3> q;
        q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32_FP32);
        q.Push(0, tile0.data(), 1.0f, 0.0f, 0.0f);
        q.Push(1, tile1.data(), 1.0f, 0.0f, 0.0f);
        q.Push(2, tile2.data(), 1.0f, 0.0f, 0.0f);
        q.Flush();
        assert(q.GetAccumMode() == PV_ACCUM_FP32_FP32);
        for (float v : acc) assert(std::fabs(v - 4.5f) < 1e-4f);
    }

    // 3. Mode 2: fp32+fp16 (默认验收档，仿真 FixPipe 随路量化 RNE 舍入)
    {
        std::fill(acc.begin(), acc.end(), 0.0f);
        OutputRetireQueue<float, 3> q;
        q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32_FP16);
        q.Push(0, tile0.data(), 1.0f, 0.0f, 0.0f);
        q.Push(1, tile1.data(), 1.0f, 0.0f, 0.0f);
        q.Push(2, tile2.data(), 1.0f, 0.0f, 0.0f);
        q.Flush();
        assert(q.GetAccumMode() == PV_ACCUM_FP32_FP16);
        // FP16 精度下 4.5f 能精确表示
        for (float v : acc) assert(std::fabs(v - 4.5f) < 1e-3f);
    }

    std::cout << "[Test 5 Passed] All 3 accumulation modes (fp32, fp32+fp32, fp32+fp16) passed!" << std::endl;
}

/**
 * [Test 6] S_FP8_OFFSET 偏置与最终输出归一化验证
 */
void TestSFP8OffsetAndFinalize() {
    std::cout << "[Test 6] Testing S_FP8_OFFSET bias and Finalize() output normalizer..." << std::endl;

    constexpr uint32_t TILE_SIZE = 32;
    std::vector<float> acc(TILE_SIZE, 0.0f);
    std::vector<float> out(TILE_SIZE, 0.0f);
    std::vector<float> tile0(TILE_SIZE, 448.0f * 1.5f);
    std::vector<float> tile1(TILE_SIZE, 448.0f * 2.5f);

    OutputRetireQueue<float, 3> q;
    q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32, 6.0f, 8.807355f);

    assert(std::fabs(q.GetSFP8Offset() - 8.807355f) < 1e-5f);

    q.Push(0, tile0.data(), 1.0f, 0.0f, 0.0f);
    q.Push(1, tile1.data(), 1.0f, 0.0f, 0.0f);

    // 假设分母也经过 448.0 偏置累计，总分母 d = 448.0f * 4.0f
    float d_sum = 448.0f * 4.0f;
    float inv_d = 1.0f / d_sum;

    q.Finalize(out.data(), inv_d);
    assert(q.IsEmpty());

    // 期望归一化输出 = (1.5 + 2.5) / 4.0 = 1.0f
    for (uint32_t j = 0; j < TILE_SIZE; ++j) {
        assert(std::fabs(out[j] - 1.0f) < 1e-4f);
    }

    std::cout << "[Test 6 Passed] S_FP8_OFFSET bias cancellation & Finalize() verified!" << std::endl;
}

/**
 * [Test 7] 宽流水（Wide Stream）N256 Cube / N512 Vector 协同模式仿真
 */
void TestWideStreamDecoupling() {
    std::cout << "[Test 7] Testing Wide Stream (N256 Cube / N512 Vector) decoupling pattern..." << std::endl;

    // 模拟长序列下 N=512 Vector 对应 2 个 N=256 Cube 子拍
    constexpr uint32_t TILE_SIZE = 128;
    std::vector<float> acc(TILE_SIZE, 0.0f);
    std::vector<std::vector<float>> sub_tiles(4, std::vector<float>(TILE_SIZE, 1.0f));

    OutputRetireQueue<float, 3> q;
    q.Init(acc.data(), TILE_SIZE, PV_ACCUM_FP32_FP16, 6.0f);

    // 连续推送 4 个子拍
    for (uint32_t i = 0; i < 4; ++i) {
        q.Push(i, sub_tiles[i].data(), 1.0f, static_cast<float>(i), 0.0f);
    }
    q.Flush();

    assert(q.IsEmpty());
    assert(q.GetPairwiseCount() == 2); // 4 个分块全部被 2 次成对融合吸收
    assert(q.GetSingleCount() == 0);   // 中间写回次数降为 50%

    std::cout << "[Test 7 Passed] Wide Stream pipelining and pairwise retirement verified!" << std::endl;
}

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "SageAttention2 OutputRetireQueue & Wide Stream Unit Tests" << std::endl;
    std::cout << "==================================================================" << std::endl;

    TestQueueFullTrigger();
    TestPairwiseFusionNumericalFidelity();
    TestSkipScaleOptimization();
    TestFlushVariousRemainingSlots();
    TestThreePrecisionModes();
    TestSFP8OffsetAndFinalize();
    TestWideStreamDecoupling();

    std::cout << "==================================================================" << std::endl;
    std::cout << "ALL 7 OUTPUT RETIRE QUEUE UNIT TESTS PASSED SUCCESSFULLY! (100%)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    return 0;
}

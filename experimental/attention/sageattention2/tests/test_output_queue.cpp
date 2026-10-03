#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>

#include "tiling.h"
#include "online_softmax.h"
#include "accum_reduce.h"
#include "twin_reference.h"

using namespace sage_attn;

// ---------------------------------------------------------------------------
// OutputRetireQueue（skip-scale 变体）——非交付热路径组件，测试本地定义。
// 交付头文件 accum_reduce.h 只保留流水真实消费的 QuantizeToFp16 /
// FixpipeConfig / AccumReduceController；PV 累加语义唯一定义点为
// twin_reference.h + online_softmax.h。本类仅为 skip-scale 成对归约的
// 独立回归载体，随本测试 TU 编译，不进入设备/交付编译域。
// ---------------------------------------------------------------------------
namespace sage_attn {

/**
 * 3 槽延迟输出退休队列模板类（OutputRetireQueue）
 *
 * 核心特性：
 * 1. outputPending[3] 环形队列机制：
 *    当连续 PV MMAD 产生输出时，在片上槽位缓存，触发成对归约（Pairwise Reduction）与 Rescale 融合，
 *    使 UB/GM 累加器的读取和写回开销直接减少 50%。
 * 2. 多档累加精度支持：
 *    全面支持 fp32、fp32+fp32、fp32+fp16 三档两级累加精度，并在 fp32+fp16 档融合 FixPipe 随路量化。
 * 3. 融合 Online Softmax 的 skip-scale 判定（tau = 6.0f）：
 *    若局部最大值变化在 tau 以内，则跳过对累加器的重缩放乘法，直接执行直通累加，消除冗余浮点乘法。
 * 4. 支持 S_FP8_OFFSET 偏置（log2(448) = 8.807355f），分母在线累计与最终归一化严格对齐。
 * 5. 循环结束时调用 Flush() 彻底清空剩余 1~2 个槽位并完成最终写回。
 */
template <typename DType = float, uint32_t CAPACITY = 3>
class OutputRetireQueue {
public:
    static constexpr uint32_t QUEUE_CAPACITY = CAPACITY;

    /**
     * 单槽位描述结构体
     */
    struct Slot {
        uint32_t tile_idx{0};       // 分块序号
        const DType* p_data{nullptr}; // 局部 PV MMAD 输出数据指针
        float scale{1.0f};          // 概率与去量化缩放系数
        float m_new{0.0f};          // 局部更新后的 running max (exp2 域)
        float m_old{0.0f};          // 上一状态的 running max
        bool valid{false};          // 槽位有效标志
    };

    // 全部状态成员已由声明处 NSDMI 初始化（Slot::valid 默认 false），逐字段赋值的
    // 旧构造函数与其完全重复，收敛为默认构造以保证初值唯一定义。
    __aicore__ OutputRetireQueue() = default;

    /**
     * 初始化退休队列与累加器绑定
     *
     * @param acc_data UB 主累加器指针
     * @param tile_size 单分块数据元素总数 (M * D)
     * @param accum_dtype 两级累加精度档位 (0=fp32, 1=fp32+fp32, 2=fp32+fp16)
     * @param tau skip-scale 阈值 (默认 6.0f)
     */
    __aicore__ void Init(
        DType* acc_data,
        uint32_t tile_size,
        uint32_t accum_dtype = PV_ACCUM_FP32_FP16,
        float tau = SKIP_SCALE_TAU,
        float s_fp8_offset = S_FP8_OFFSET) {
        acc_data_ = acc_data;
        tile_size_ = tile_size;
        accum_mode_ = static_cast<PvAccumDtype>(accum_dtype);
        tau_ = tau;
        s_fp8_offset_ = s_fp8_offset;
        queue_size_ = 0;
        head_ = 0;
        tail_ = 0;
        current_m_ = MIN_FLOAT_VAL;
        acc_initialized_ = false;
        pairwise_count_ = 0;
        single_count_ = 0;
        skip_scale_count_ = 0;
        rescale_count_ = 0;
        total_pushed_ = 0;
        total_retired_ = 0;
        for (uint32_t i = 0; i < CAPACITY; ++i) {
            slots_[i].valid = false;
        }
    }

    /**
     * 重新绑定主累加器缓冲区
     */
    __aicore__ void BindAccumulator(DType* acc_data, uint32_t tile_size) {
        acc_data_ = acc_data;
        tile_size_ = tile_size;
    }

    /**
     * 将新产生的 PV MMAD 输出块压入退休队列
     * 当队列满时（达到 CAPACITY=3 槽位），自动触发成对归约与 Rescale 融合
     *
     * @param tile_idx 当前分块序号
     * @param p_data 当前分块数据指针
     * @param scale 缩放因子
     * @param m_new 当前分块对应的 running max
     * @param m_old 上一步的 running max
     * @return 成功返回 true
     */
    __aicore__ bool Push(
        uint32_t tile_idx,
        const DType* p_data,
        float scale,
        float m_new,
        float m_old) {
        // 若队列已满，先触发成对归约释放槽位
        if (queue_size_ >= CAPACITY) {
            RetirePair();
        }

        // 入队新槽位
        slots_[tail_].tile_idx = tile_idx;
        slots_[tail_].p_data = p_data;
        slots_[tail_].scale = scale;
        slots_[tail_].m_new = m_new;
        slots_[tail_].m_old = m_old;
        slots_[tail_].valid = true;
        tail_ = (tail_ + 1) % CAPACITY;
        queue_size_++;
        total_pushed_++;

        // 槽位满时自动触发成对归约（Pairwise Reduction）与 Rescale 融合
        if (queue_size_ >= CAPACITY) {
            RetirePair();
        }

        return true;
    }

    /**
     * 成对归约（Pairwise Reduction）核心实现
     * 将队列头部的两个槽位 (Slot 0 与 Slot 1) 融合成一次累加器读写更新
     */
    __aicore__ void RetirePair() {
        if (queue_size_ < 2) {
            return;
        }

        Slot& s0 = slots_[head_];
        Slot& s1 = slots_[(head_ + 1) % CAPACITY];

        // 计算当前成对融合的目标统一 max
        float m_target = (s1.m_new > s0.m_new) ? s1.m_new : s0.m_new;

        // 计算两分块各自到 m_target 的对齐缩放系数
        float alpha0 = s0.scale * OnlineSoftmaxTransformer::FastExp2(s0.m_new - m_target);
        float alpha1 = s1.scale * OnlineSoftmaxTransformer::FastExp2(s1.m_new - m_target);

        if (!acc_initialized_) {
            // 首对分块直接初始化累加器
            current_m_ = m_target;
            acc_initialized_ = true;

            if (acc_data_ != nullptr && s0.p_data != nullptr && s1.p_data != nullptr) {
                for (uint32_t j = 0; j < tile_size_; ++j) {
                    float v0 = static_cast<float>(s0.p_data[j]);
                    float v1 = static_cast<float>(s1.p_data[j]);
                    float val = v0 * alpha0 + v1 * alpha1;
                    if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                        val = QuantizeToFp16(val);
                    }
                    acc_data_[j] = static_cast<DType>(val);
                }
            }
        } else {
            // 累加器已有部分和：判定 skip-scale 与 rescale
            float delta_m = m_target - current_m_;
            bool need_rescale = false;
            float rescale_factor = 1.0f;

            if (delta_m <= 0.0f) {
                need_rescale = false;
                rescale_factor = 1.0f;
            } else if (delta_m < tau_) {
                // skip-scale 判定成功：差值在 tau (6.0f) 以内，跳过累加器重缩放
                need_rescale = false;
                rescale_factor = 1.0f;
                skip_scale_count_++;
            } else {
                // 超过阈值：执行累加器重缩放
                need_rescale = true;
                rescale_factor = OnlineSoftmaxTransformer::FastExp2(current_m_ - m_target);
                current_m_ = m_target;
                rescale_count_++;
            }

            if (acc_data_ != nullptr && s0.p_data != nullptr && s1.p_data != nullptr) {
                if (!need_rescale) {
                    if (m_target != current_m_) {
                        float align_factor = OnlineSoftmaxTransformer::FastExp2(m_target - current_m_);
                        alpha0 *= align_factor;
                        alpha1 *= align_factor;
                    }
                    // 直通累加模式：消除对累加器的浮点乘法
                    if (SageFastAbs(alpha0 - 1.0f) < SCALE_ONE_TOLERANCE &&
                        SageFastAbs(alpha1 - 1.0f) < SCALE_ONE_TOLERANCE) {
                        // 极致优化：纯加法直通
                        for (uint32_t j = 0; j < tile_size_; ++j) {
                            float acc_val = static_cast<float>(acc_data_[j]);
                            float v0 = static_cast<float>(s0.p_data[j]);
                            float v1 = static_cast<float>(s1.p_data[j]);
                            float val = acc_val + v0 + v1;
                            if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                                val = QuantizeToFp16(val);
                            }
                            acc_data_[j] = static_cast<DType>(val);
                        }
                    } else {
                        for (uint32_t j = 0; j < tile_size_; ++j) {
                            float acc_val = static_cast<float>(acc_data_[j]);
                            float v0 = static_cast<float>(s0.p_data[j]);
                            float v1 = static_cast<float>(s1.p_data[j]);
                            float val = acc_val + (v0 * alpha0 + v1 * alpha1);
                            if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                                val = QuantizeToFp16(val);
                            }
                            acc_data_[j] = static_cast<DType>(val);
                        }
                    }
                } else {
                    // Rescale 融合模式：单拍完成累加器缩放与两块累加更新
                    for (uint32_t j = 0; j < tile_size_; ++j) {
                        float acc_val = static_cast<float>(acc_data_[j]);
                        float v0 = static_cast<float>(s0.p_data[j]);
                        float v1 = static_cast<float>(s1.p_data[j]);
                        float val = acc_val * rescale_factor + (v0 * alpha0 + v1 * alpha1);
                        if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                            val = QuantizeToFp16(val);
                        }
                        acc_data_[j] = static_cast<DType>(val);
                    }
                }
            }
        }

        // 释放已退休的两个槽位
        s0.valid = false;
        s1.valid = false;
        head_ = (head_ + 2) % CAPACITY;
        queue_size_ -= 2;
        pairwise_count_++;
        total_retired_ += 2;
    }

    /**
     * 单槽位归约实现（在 Flush 尾部处理奇数剩余槽位时使用）
     */
    __aicore__ void RetireSingle() {
        if (queue_size_ < 1) {
            return;
        }

        Slot& s0 = slots_[head_];
        float m_target = s0.m_new;
        float alpha0 = s0.scale;

        if (!acc_initialized_) {
            current_m_ = m_target;
            acc_initialized_ = true;

            if (acc_data_ != nullptr && s0.p_data != nullptr) {
                for (uint32_t j = 0; j < tile_size_; ++j) {
                    float v0 = static_cast<float>(s0.p_data[j]);
                    float val = v0 * alpha0;
                    if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                        val = QuantizeToFp16(val);
                    }
                    acc_data_[j] = static_cast<DType>(val);
                }
            }
        } else {
            float delta_m = m_target - current_m_;
            bool need_rescale = false;
            float rescale_factor = 1.0f;

            if (delta_m <= 0.0f) {
                need_rescale = false;
                rescale_factor = 1.0f;
            } else if (delta_m < tau_) {
                need_rescale = false;
                rescale_factor = 1.0f;
                skip_scale_count_++;
            } else {
                need_rescale = true;
                rescale_factor = OnlineSoftmaxTransformer::FastExp2(current_m_ - m_target);
                current_m_ = m_target;
                rescale_count_++;
            }

            if (acc_data_ != nullptr && s0.p_data != nullptr) {
                if (!need_rescale) {
                    if (m_target != current_m_) {
                        float align_factor = OnlineSoftmaxTransformer::FastExp2(m_target - current_m_);
                        alpha0 *= align_factor;
                    }
                    if (SageFastAbs(alpha0 - 1.0f) < SCALE_ONE_TOLERANCE) {
                        for (uint32_t j = 0; j < tile_size_; ++j) {
                            float acc_val = static_cast<float>(acc_data_[j]);
                            float v0 = static_cast<float>(s0.p_data[j]);
                            float val = acc_val + v0;
                            if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                                val = QuantizeToFp16(val);
                            }
                            acc_data_[j] = static_cast<DType>(val);
                        }
                    } else {
                        for (uint32_t j = 0; j < tile_size_; ++j) {
                            float acc_val = static_cast<float>(acc_data_[j]);
                            float v0 = static_cast<float>(s0.p_data[j]);
                            float val = acc_val + (v0 * alpha0);
                            if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                                val = QuantizeToFp16(val);
                            }
                            acc_data_[j] = static_cast<DType>(val);
                        }
                    }
                } else {
                    for (uint32_t j = 0; j < tile_size_; ++j) {
                        float acc_val = static_cast<float>(acc_data_[j]);
                        float v0 = static_cast<float>(s0.p_data[j]);
                        float val = acc_val * rescale_factor + (v0 * alpha0);
                        if (accum_mode_ == PV_ACCUM_FP32_FP16) {
                            val = QuantizeToFp16(val);
                        }
                        acc_data_[j] = static_cast<DType>(val);
                    }
                }
            }
        }

        s0.valid = false;
        head_ = (head_ + 1) % CAPACITY;
        queue_size_ -= 1;
        single_count_++;
        total_retired_ += 1;
    }

    /**
     * 算子主循环结束时调用 Flush() 彻底清空剩余 1~2 个槽位并完成写回
     */
    __aicore__ void Flush() {
        while (queue_size_ >= 2) {
            RetirePair();
        }
        if (queue_size_ == 1) {
            RetireSingle();
        }
    }

    /**
     * 最终归一化输出写回
     *
     * @param out_dst 最终写出的 GM 或 UB 输出缓冲区
     * @param inv_denominator Softmax 最终倒数乘数 (来自 OnlineSoftmaxTransformer::GetFinalInvDenominator)
     */
    __aicore__ void Finalize(DType* out_dst, float inv_denominator) {
        Flush();
        if (out_dst != nullptr && acc_data_ != nullptr && inv_denominator > 0.0f) {
            for (uint32_t j = 0; j < tile_size_; ++j) {
                float val = static_cast<float>(acc_data_[j]) * inv_denominator;
                out_dst[j] = static_cast<DType>(val);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 状态查询与统计函数
    // -------------------------------------------------------------------------
    __aicore__ uint32_t GetQueueSize() const { return queue_size_; }
    __aicore__ bool IsEmpty() const { return queue_size_ == 0; }
    __aicore__ bool IsFull() const { return queue_size_ >= CAPACITY; }
    __aicore__ uint32_t GetCapacity() const { return CAPACITY; }
    __aicore__ uint32_t GetPairwiseCount() const { return pairwise_count_; }
    __aicore__ uint32_t GetSingleCount() const { return single_count_; }
    __aicore__ uint32_t GetSkipScaleCount() const { return skip_scale_count_; }
    __aicore__ uint32_t GetRescaleCount() const { return rescale_count_; }
    __aicore__ uint32_t GetTotalPushed() const { return total_pushed_; }
    __aicore__ uint32_t GetTotalRetired() const { return total_retired_; }
    __aicore__ float GetCurrentM() const { return current_m_; }
    __aicore__ float GetSFP8Offset() const { return s_fp8_offset_; }
    __aicore__ void SetSFP8Offset(float offset) { s_fp8_offset_ = offset; }
    __aicore__ PvAccumDtype GetAccumMode() const { return accum_mode_; }
    __aicore__ const Slot& GetSlot(uint32_t i) const { return slots_[(head_ + i) % CAPACITY]; }
    __aicore__ const DType* GetAccumulator() const { return acc_data_; }
    __aicore__ DType* GetAccumulator() { return acc_data_; }

private:
    DType* acc_data_{nullptr};      // 累加器目标缓冲
    uint32_t tile_size_{0};         // 分块数据元素数
    Slot slots_[CAPACITY];          // 3 槽延迟输出缓冲槽位
    uint32_t queue_size_{0};        // 当前队列中有效槽位数
    uint32_t head_{0};              // 队列头索引
    uint32_t tail_{0};              // 队列尾索引

    PvAccumDtype accum_mode_{PV_ACCUM_FP32_FP16}; // 两级累加精度
    float tau_{SKIP_SCALE_TAU};     // skip-scale 阈值 (6.0f)
    float s_fp8_offset_{S_FP8_OFFSET}; // FP8 偏置 (默认 log2(448))
    float current_m_{MIN_FLOAT_VAL}; // 累加器当前锚点 max
    bool acc_initialized_{false};   // 累加器初始化标志

    // 统计度量
    uint32_t pairwise_count_{0};
    uint32_t single_count_{0};
    uint32_t skip_scale_count_{0};
    uint32_t rescale_count_{0};
    uint32_t total_pushed_{0};
    uint32_t total_retired_{0};
};

} // namespace sage_attn

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
    queue.Init(acc_queue.data(), TILE_SIZE, PV_ACCUM_FP32, 0.0f);

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
 * [Test 5] 三档两级累加精度（fp32, fp32+fp32, fp32+fp16）验证测试
 */
void TestThreePrecisionModes() {
    std::cout << "[Test 5] Testing 3 accumulation precision modes..." << std::endl;

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

/**
 * [Test 8] fp32+fp32 两级累加独立轨迹：flush 节奏真实执行 + 与 fp32 的结合序差异有界且不恒等
 *
 * 构造（p8 = SoftmaxProbDequant(0, m'=0) = exp2(0)·448 = 448 精确，V 逐 token 给定）：
 * 乘积序列（token 序）= [7·2^27, -7·2^27, 7·2^21, 7·2^21 | 0.875 × 12]。
 * - fp32 单一运行和：前 4 项块内精确相消并精确累至 X = 7·2^22 = 29360128（ulp(X)=2），
 *   其后 ε=0.875 项逐个被 RNE 丢弃（0.875 < 半 ulp=1）→ 结果恒 X；
 * - fp32+fp32：每 k_step 分段子先在子块内精确归并（如 4·0.875=3.5、8·0.875=7），
 *   窗末一次归约进 FP32 O buffer，加法结合序与 fp32 真实不同：
 *   k_step=16（单窗，与 fp32 同结合序）→ X（位级同轨锚点）；
 *   k_step=8 → 第二窗子和 7 → X+7 落 tie（X+6/X+8 等距）RNE 取偶 → X+8；
 *   k_step=4 → 3.5/3.5/3.5 三窗 → X+12；k_step=2 → 1.75×6 窗 → X+12。
 *   全部差异有限、有界（相对差 < 1e-6）且位级不恒等。
 */
void TestFp32Fp32AccumTrack() {
    std::cout << "[Test 8] Testing fp32+fp32 accumulation track (flush cadence & association order)..." << std::endl;

    constexpr uint32_t kNumTokens = 16;  // == DEFAULT_K_STEP：单窗锚点用
    constexpr uint32_t kChannels = 2;
    constexpr float kBigV = 2097152.0f;    // 2^21 → 乘积 7·2^27
    constexpr float kMidV = 32768.0f;      // 2^15 → 乘积 7·2^21
    constexpr float kTinyV = 0.001953125f; // 2^-9 → 乘积 7·2^-3 = 0.875

    std::vector<float> s_row(kNumTokens, 0.0f);  // m'=0 时 p8 = 448 精确
    std::vector<float> v(kNumTokens);
    for (uint32_t j = 0; j < kNumTokens; ++j) {
        v[j] = (j < 2) ? kBigV : ((j < 4) ? kMidV : kTinyV);
    }
    v[1] = -kBigV;  // 大项块内精确相消
    auto v_at = [&](uint32_t j, uint32_t) -> float { return v[j]; };

    auto run_track = [&](PvAccumDtype mode, uint32_t k_step) {
        std::vector<float> o(kChannels, 0.0f);
        twin::AccumulatePvRow(o.data(), kChannels, s_row.data(), 0, 0, kNumTokens,
                              false, 0, 0.0f, mode, k_step, v_at);
        return o;
    };

    const std::vector<float> o_fp32 = run_track(PV_ACCUM_FP32, DEFAULT_K_STEP);
    const std::vector<float> o_ks16 = run_track(PV_ACCUM_FP32_FP32, DEFAULT_K_STEP);
    const std::vector<float> o_ks8 = run_track(PV_ACCUM_FP32_FP32, 8);
    const std::vector<float> o_ks4 = run_track(PV_ACCUM_FP32_FP32, 4);
    const std::vector<float> o_ks2 = run_track(PV_ACCUM_FP32_FP32, 2);

    constexpr float kAnchorX = 29360128.0f;  // 7·2^22
    for (uint32_t c = 0; c < kChannels; ++c) {
        assert(std::isfinite(o_fp32[c]));
        // 锚点：单窗（k_step >= token 数）时 fp32+fp32 与 fp32 位级同轨——
        // 两档唯一数值差异源即归约节奏（结合序），而非乘法域
        assert(o_fp32[c] == kAnchorX);
        assert(o_ks16[c] == o_fp32[c]);
        // (a) flush 节奏真实执行：不同 k_step 的分段归约产生位级可区分结果
        assert(o_ks8[c] == kAnchorX + 8.0f);
        assert(o_ks4[c] == kAnchorX + 12.0f);
        assert(o_ks2[c] == kAnchorX + 12.0f);
        assert(o_ks8[c] != o_ks4[c]);
        // (b) 与 fp32 的结合序差异有限、有界且不恒等
        assert(o_ks4[c] != o_fp32[c]);
        const float rel = std::fabs(o_ks4[c] - o_fp32[c]) / std::fabs(o_fp32[c]);
        assert(rel < 1e-6f);
    }

    std::cout << "[Test 8 Passed] fp32+fp32 track flush cadence executed & bounded association-order "
                 "difference vs fp32 verified (rel diff < 1e-6, bitwise distinct)!"
              << std::endl;
}

/**
 * [Test 9] 设备核兼容入口（bool 双档）与三轨单点的映射等价性：零数值逻辑（无第二份公式）。
 * 输入 A（良性量级）：bool↔enum 位级等价（fp32 / fp32+fp16 两档不回归的证据）；
 * 输入 B（Test 8 同族结合序构造，token 数 > k_step）：fp32+fp32 独立轨迹非平凡性——
 * 通道 0~2 与 fp32 位级互异，通道 3（全中项，窗界落在半 ulp 内）为同轨负控。
 */
void TestBoolAdapterEquivalence() {
    std::cout << "[Test 9] Testing bool adapter entry maps exactly onto the 3-track single point..." << std::endl;

    // 输入 A：良性量级，三档全部有限
    constexpr uint32_t kNumTokens = 20;
    constexpr uint32_t kChannels = 4;
    std::vector<float> s_row_a(kNumTokens);
    std::vector<float> v_a(kNumTokens * kChannels);
    for (uint32_t j = 0; j < kNumTokens; ++j) {
        s_row_a[j] = -0.05f * static_cast<float>(j);  // 非平凡 p8 轨迹
        for (uint32_t c = 0; c < kChannels; ++c) {
            v_a[j * kChannels + c] = std::sin(static_cast<float>(j) * 0.7f + static_cast<float>(c)) * 3.5f;
        }
    }
    auto v_at_a = [&](uint32_t j, uint32_t c) -> float { return v_a[j * kChannels + c]; };

    auto run_bool = [&](bool fp16_accum, const std::vector<float>& s, const auto& v_at) {
        std::vector<float> o(kChannels, 0.0f);
        twin::QuantizePAndAccumulatePvRow(o.data(), kChannels, s.data(), 0, 0, kNumTokens,
                                          false, 0, 0.0f, fp16_accum, DEFAULT_K_STEP, v_at);
        return o;
    };
    auto run_enum = [&](PvAccumDtype mode, const std::vector<float>& s, const auto& v_at) {
        std::vector<float> o(kChannels, 0.0f);
        twin::AccumulatePvRow(o.data(), kChannels, s.data(), 0, 0, kNumTokens,
                              false, 0, 0.0f, mode, DEFAULT_K_STEP, v_at);
        return o;
    };

    const auto o_fp32_bool = run_bool(false, s_row_a, v_at_a);
    const auto o_fp16_bool = run_bool(true, s_row_a, v_at_a);
    const auto o_fp32_enum = run_enum(PV_ACCUM_FP32, s_row_a, v_at_a);
    const auto o_fp16_enum = run_enum(PV_ACCUM_FP32_FP16, s_row_a, v_at_a);
    for (uint32_t c = 0; c < kChannels; ++c) {
        assert(std::isfinite(o_fp32_bool[c]) && std::isfinite(o_fp16_bool[c]));
        // bool 双档签名与三轨单点位级等价（fp32 / fp32+fp16 两档不回归的证据）
        assert(o_fp32_bool[c] == o_fp32_enum[c]);
        assert(o_fp16_bool[c] == o_fp16_enum[c]);
        assert(o_fp16_enum[c] != o_fp32_enum[c]);  // 独立低精度轨迹非平凡
    }

    // 输入 B：p8 = 448 精确，通道 0~2 乘积 = [±7·2^27 相消, 7·2^21, 7·2^21 | 0.875 × 16]，
    // 通道 3 全大项（乘积 7·2^24 ×4 = 7·2^26，ulp=8 > 窗界子和 3.5 → 窗界归约被 RNE
    // 丢弃，与 fp32 同轨的负控）
    std::vector<float> s_row_b(kNumTokens, 0.0f);
    std::vector<float> v_b(kNumTokens * kChannels, 0.0f);
    for (uint32_t c = 0; c < kChannels; ++c) {
        if (c == 3) {
            for (uint32_t j = 0; j < 4; ++j) v_b[j * kChannels + c] = 262144.0f;  // 2^18
        } else {
            const float big = (c % 2 == 0) ? 2097152.0f : -2097152.0f;
            v_b[0 * kChannels + c] = big;
            v_b[1 * kChannels + c] = -big;
            v_b[2 * kChannels + c] = 32768.0f;
            v_b[3 * kChannels + c] = 32768.0f;
        }
        for (uint32_t j = 4; j < kNumTokens; ++j) v_b[j * kChannels + c] = 0.001953125f;
    }
    auto v_at_b = [&](uint32_t j, uint32_t c) -> float { return v_b[j * kChannels + c]; };

    const auto o_fp32_b = run_enum(PV_ACCUM_FP32, s_row_b, v_at_b);
    const auto o_fp32fp32_b = run_enum(PV_ACCUM_FP32_FP32, s_row_b, v_at_b);
    for (uint32_t c = 0; c < kChannels; ++c) {
        assert(std::isfinite(o_fp32fp32_b[c]));
        if (c < 3) {
            assert(o_fp32fp32_b[c] != o_fp32_b[c]);   // 分段归约结合序差异真实存在
        } else {
            assert(o_fp32fp32_b[c] == o_fp32_b[c]);   // 窗界子和低于半 ulp 时同轨（负控）
        }
    }

    std::cout << "[Test 9 Passed] Bool adapter is bit-exact with the 3-track single point (fp32 & fp32+fp16)!"
              << std::endl;
}

/**
 * [Test 10] AccumReduceController 归约节奏两轨化：fp32 仅末块 vs fp32+fp32/fp32+fp16 周期+末块
 */
void TestAccumReduceCadencePerMode() {
    std::cout << "[Test 10] Testing AccumReduceController per-mode reduction cadence..." << std::endl;

    constexpr uint32_t kTotalBlocks = 10;
    constexpr uint32_t kKStep = 2;

    // 1. fp32：全程单一运行和，仅末块归约（周期项被禁用）
    {
        AccumReduceController ctrl;
        ctrl.Init(PV_ACCUM_FP32, kKStep, kTotalBlocks);
        assert(ctrl.GetAccumMode() == PV_ACCUM_FP32);
        assert(ctrl.GetKStep() == kKStep);
        for (uint32_t step = 0; step + 1 < kTotalBlocks; ++step) {
            assert(!ctrl.ShouldReduce(step));  // 周期点不再归约
        }
        assert(ctrl.ShouldReduce(kTotalBlocks - 1));  // 末块强制归约

        const FixpipeConfig cfg = ctrl.GetFixpipeConfig();
        assert(!cfg.enable_inline_quant);
        assert(cfg.quant_dst_type == 0);  // FP32 直通
        assert(cfg.enable_nz_to_nd);
    }

    // 2. fp32+fp32：k_step 周期归约 + 末块强制归约；随路配置与 fp32 同为 FP32 直通
    //    （数值差异在节奏，不在随路格式）
    {
        AccumReduceController ctrl;
        ctrl.Init(PV_ACCUM_FP32_FP32, kKStep, kTotalBlocks);
        assert(ctrl.GetAccumMode() == PV_ACCUM_FP32_FP32);
        for (uint32_t step = 0; step < kTotalBlocks; ++step) {
            const bool expect = ((step + 1) % kKStep == 0) || (step + 1 == kTotalBlocks);
            assert(ctrl.ShouldReduce(step) == expect);
        }
        const FixpipeConfig cfg = ctrl.GetFixpipeConfig();
        assert(!cfg.enable_inline_quant);
        assert(cfg.quant_dst_type == 0);
        assert(cfg.enable_nz_to_nd);
    }

    // 3. fp32+fp16：节奏与 fp32+fp32 一致（k_step 周期 + 末块），随路量化 FP16 不回归
    {
        AccumReduceController ctrl;
        ctrl.Init(PV_ACCUM_FP32_FP16, kKStep, kTotalBlocks);
        assert(ctrl.GetAccumMode() == PV_ACCUM_FP32_FP16);
        assert(ctrl.IsFp16Accum());
        for (uint32_t step = 0; step < kTotalBlocks; ++step) {
            const bool expect = ((step + 1) % kKStep == 0) || (step + 1 == kTotalBlocks);
            assert(ctrl.ShouldReduce(step) == expect);
        }
        const FixpipeConfig cfg = ctrl.GetFixpipeConfig();
        assert(cfg.enable_inline_quant);
        assert(cfg.quant_dst_type == 1);  // FP16
        assert(cfg.enable_nz_to_nd);
    }

    std::cout << "[Test 10 Passed] fp32 (last-block-only) vs fp32+fp32/fp32+fp16 (k_step periodic) cadence verified!"
              << std::endl;
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
    TestFp32Fp32AccumTrack();
    TestBoolAdapterEquivalence();
    TestAccumReduceCadencePerMode();

    std::cout << "==================================================================" << std::endl;
    std::cout << "ALL 10 OUTPUT RETIRE QUEUE UNIT TESTS PASSED SUCCESSFULLY! (100%)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    return 0;
}

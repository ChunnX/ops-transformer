#ifndef SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H
#define SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H

#include <cstdint>
#include <cstring>
#include "tiling.h"
#include "online_softmax.h"
// FP16 位级转换唯一定义点（STYLE-11 单一定义收敛）
#include "../../quant/quant_common.h"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace sage_attn {

/**
 * FixPipe 随路量化的 FP16 舍入仿真：返回"经 FP16 RNE 舍入后仍以 fp32 承载"的值。
 * 位级转换唯一来源为 quant::Fp32ToFp16/Fp16ToFp32（次正规 RNE 舍入语义）。
 */
__aicore__ static inline float QuantizeToFp16(float val) {
    return quant::Fp16ToFp32(quant::Fp32ToFp16(val));
}

/**
 * FixPipe 随路操作配置描述符（硬件数据通路）
 */
struct FixpipeConfig {
    bool enable_inline_quant;   // 是否开启 L0C 回写 UB 随路量化 (FP32 -> FP16)
    bool enable_nz_to_nd;       // 是否开启分形 NZ 到 ND 随路转置排布
    uint32_t quant_dst_type;    // 目标量化格式 (0=FP32, 1=FP16, 2=BF16)
};

/**
 * 两级累加归约控制器
 * 支持 fp32 / fp32+fp32 / fp32+fp16 三档两级累加：
 * - fp32：全程 FP32 单一运行和，仅末块归约（ShouldReduce 两轨节奏）；
 * - fp32+fp32：MMA 域 FP32，每 k_step 分块归约至 FP32 buffer；
 * - fp32+fp16：低精度 MMA 域 FP16，每 k_step 分块归约至 FP16 buffer（FixPipe 随路量化）。
 * 轨迹语义与 twin_reference.h AccumulatePvRow 三轨单点同源定标。
 */
class AccumReduceController {
public:
    __aicore__ AccumReduceController()
        : accum_mode_(PV_ACCUM_FP32_FP16), k_step_(DEFAULT_K_STEP), total_kv_blocks_(0) {}

    __aicore__ void Init(uint32_t accum_dtype, uint32_t k_step, uint32_t total_kv_blocks) {
        accum_mode_ = static_cast<PvAccumDtype>(accum_dtype);
        k_step_ = (k_step > 0) ? k_step : DEFAULT_K_STEP;
        total_kv_blocks_ = total_kv_blocks;
    }

    /**
     * 判断当前迭代是否需要触发 L0C 向 UB 主累加器的 FixPipe 搬出归约。
     * 末块判定由内部 total_kv_blocks_ 推导。节奏按档位两轨（与 twin_reference.h
     * AccumulatePvRow 三轨单点的结合序定标一致）：
     * - PV_ACCUM_FP32：全程 FP32 单一运行和（L0C 即主累加器），仅末块归约——
     *   周期归约会把"单一运行和"改写成"分段子和"，破坏与 fp32 golden 轨迹的
     *   位级一致性；
     * - PV_ACCUM_FP32_FP32 / PV_ACCUM_FP32_FP16：每 k_step 分块周期归约 + 末块
     *   强制归约（inst buffer 受 k_step 约束定期 flush，随路格式见 GetFixpipeConfig）。
     *
     * @param step_idx 当前处理的 KV 分块序号 (0-indexed)
     */
    __aicore__ bool ShouldReduce(uint32_t step_idx) const {
        const bool is_last_step = (total_kv_blocks_ > 0) && (step_idx + 1 >= total_kv_blocks_);
        if (accum_mode_ == PV_ACCUM_FP32) {
            return is_last_step;
        }
        return is_last_step || (((step_idx + 1) % k_step_) == 0);
    }

    /**
     * 获取当前档位对应的 FixPipe 硬件配置。
     * fp32+fp32 与 fp32 共用同一随路配置（FP32 直通、无随路量化）：两档的数值
     * 差异在归约节奏（ShouldReduce：k_step 周期 vs 仅末块），不在随路格式——
     * fp32+fp32 的周期 flush 与 fp32 的末块 flush 均以 FP32 位型搬出。
     */
    __aicore__ FixpipeConfig GetFixpipeConfig() const {
        FixpipeConfig cfg;
        cfg.enable_nz_to_nd = true; // 随路完成 NZ 到 ND 转换

        if (accum_mode_ == PV_ACCUM_FP32_FP16) {
            cfg.enable_inline_quant = true;
            cfg.quant_dst_type = 1; // FP16
        } else {
            // fp32 与 fp32+fp32 保持 FP32 精度
            cfg.enable_inline_quant = false;
            cfg.quant_dst_type = 0; // FP32
        }
        return cfg;
    }

    __aicore__ PvAccumDtype GetAccumMode() const {
        return accum_mode_;
    }

    __aicore__ uint32_t GetKStep() const {
        return k_step_;
    }

    __aicore__ bool IsFp16Accum() const {
        return accum_mode_ == PV_ACCUM_FP32_FP16;
    }

private:
    PvAccumDtype accum_mode_;
    uint32_t k_step_;
    uint32_t total_kv_blocks_;
};

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

#endif // SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H

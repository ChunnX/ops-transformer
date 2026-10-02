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

#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

#include <cstdint>
#include <cstring>
#include "../ascendc/tiling.h"
#include "../ascendc/accum_reduce.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"
#include "../../quant/quant_common.h"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace Catlass::SageAttn {

/**
 * FP8 PV 矩阵乘分块与硬件特征
 * 负责 P(FP8) x V(FP8) -> O(FP32) 的 MMAD 运算与 FixPipe 随路量化 Epilogue
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 128,
    uint32_t TileK = 64
>
struct SagePvMmadTraits {
    using ElementA = int8_t;  // Softmax 概率矩阵 S (FP8 e4m3)
    using ElementB = int8_t;  // Value 矩阵 V (FP8 e4m3)
    using ElementC = float;   // L0C 内部累加格式 (FP32)

    using LayoutA = Layout_zZ; // L0A 中 S 分形
    using LayoutB = Layout_nZ; // L0B 中 V 分形
    using LayoutC = Layout_zN; // L0C 中 O 分形

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;

    // 基本分形块尺寸 (FP8 下 16x16x32)
    static constexpr uint32_t M0 = FRACTAL_M0;
    static constexpr uint32_t N0 = FRACTAL_N0;
    static constexpr uint32_t K0 = FRACTAL_K0_FP8;

    static constexpr uint32_t A_BYTES = M * K * sizeof(ElementA);
    static constexpr uint32_t B_BYTES = K * N * sizeof(ElementB);
    static constexpr uint32_t C_BYTES = M * N * sizeof(ElementC);
};

/**
 * FP8 PV 矩阵乘执行器与 Epilogue 封装
 */
template <class Traits = SagePvMmadTraits<64, 128, 64>>
class SagePvMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;
    using RetireQueue = sage_attn::OutputRetireQueue<ElementC, 3>;

    // 成员初值统一由声明处 NSDMI 唯一定义，构造函数仅保留宿主仿真缓冲清零
    __aicore__ SagePvMmad() {
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        std::memset(l0c_buf_, 0, sizeof(l0c_buf_));
#endif
    }

    __aicore__ void BindBuffers(const void* ub_base, const void* l1_base, void* l0c_base) {
        ub_base_ = ub_base;
        l1_base_ = l1_base;
        l0c_base_ = l0c_base;
    }

    /**
     * 初始化 3 槽延迟输出退休队列（宽流水与成对融合延迟写回）
     */
    __aicore__ void InitRetireQueue(
        ElementC* acc_dst,
        uint32_t tile_size,
        uint32_t accum_dtype = sage_attn::PV_ACCUM_FP32_FP16,
        float tau = sage_attn::SKIP_SCALE_TAU,
        float s_fp8_offset = sage_attn::S_FP8_OFFSET) {
        retire_queue_.Init(acc_dst, tile_size, accum_dtype, tau, s_fp8_offset);
    }

    __aicore__ RetireQueue& GetRetireQueue() { return retire_queue_; }
    __aicore__ const RetireQueue& GetRetireQueue() const { return retire_queue_; }

    /**
     * 将局部 PV MMAD 产生的部分和压入延迟退休队列
     */
    __aicore__ bool PushOutputTile(
        uint32_t tile_idx,
        const ElementC* p_data,
        float scale,
        float m_new,
        float m_old) {
        const ElementC* src = (p_data != nullptr) ? p_data : GetL0C();
        return retire_queue_.Push(tile_idx, src, scale, m_new, m_old);
    }

    /**
     * 算子主循环收尾时彻底清空延迟退休队列中的剩余槽位
     */
    __aicore__ void FlushRetireQueue() {
        retire_queue_.Flush();
    }


    /**
     * 加载 Softmax 概率矩阵 S 分块至 L0A
     */
    __aicore__ void LoadL0A(uint32_t s_addr, [[maybe_unused]] uint32_t l0a_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    #if defined(__CCE__) && !defined(__CCE_KT_TEST__)
        // CANN dav-3510 范式（TPipe::GetAbsAddr 同款）：SetAddr 只接受 TBuffAddr 描述符，
        // logicPos 决定硬件单元（A2=L0A，VECIN=UB 收包槽），dataLen 为字节长度。
        AscendC::TBuffAddr l0a_desc{};
        l0a_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        l0a_desc.bufferHandle = nullptr;
        l0a_desc.bufferAddr = l0a_addr;
        l0a_desc.dataLen = static_cast<uint32_t>(m * k * sizeof(ElementA));
        AscendC::LocalTensor<ElementA> l0a_tensor;
        l0a_tensor.SetAddr(l0a_desc);
        AscendC::TBuffAddr ub_s_desc{};
        ub_s_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub_s_desc.bufferHandle = nullptr;
        ub_s_desc.bufferAddr = s_addr;
        ub_s_desc.dataLen = static_cast<uint32_t>(m * k * sizeof(ElementA));
        AscendC::LocalTensor<ElementA> ub_tensor;
        ub_tensor.SetAddr(ub_s_desc);
        AscendC::LoadData2DParams load_params;
        load_params.repeatTimes = static_cast<uint16_t>((m * k * sizeof(ElementA) + LOADDATA2D_REPEAT_BYTES - 1) / LOADDATA2D_REPEAT_BYTES);
        load_params.srcStride = 0;
        load_params.ifTranspose = false;
        AscendC::LoadData(l0a_tensor, ub_tensor, load_params);
    #else
        (void)s_addr;
        (void)l0a_addr;
    #endif
#else
        if (ub_base_ != nullptr) {
            s_ptr_ = reinterpret_cast<const ElementA*>(static_cast<const uint8_t*>(ub_base_) + s_addr);
        }
#endif
    }

    __aicore__ void LoadL0A(const ElementA* s_ptr, uint32_t m, uint32_t k) {
        s_ptr_ = s_ptr;
        m_ = m;
        k_ = k;
    }

    /**
     * 加载 V 分块至 L0B (从 L1 Bank 1 搬运)
     */
    __aicore__ void LoadL0B(uint32_t l1_v_addr, [[maybe_unused]] uint32_t l0b_addr, uint32_t k, uint32_t n) {
        k_ = k;
        n_ = n;
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    #if defined(__CCE__) && !defined(__CCE_KT_TEST__)
        // L0B=B2，L1 Bank1 (V)=B1
        AscendC::TBuffAddr l0b_desc{};
        l0b_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        l0b_desc.bufferHandle = nullptr;
        l0b_desc.bufferAddr = l0b_addr;
        l0b_desc.dataLen = static_cast<uint32_t>(k * n * sizeof(ElementB));
        AscendC::LocalTensor<ElementB> l0b_tensor;
        l0b_tensor.SetAddr(l0b_desc);
        AscendC::TBuffAddr l1_v_desc{};
        l1_v_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B1);
        l1_v_desc.bufferHandle = nullptr;
        l1_v_desc.bufferAddr = l1_v_addr;
        l1_v_desc.dataLen = static_cast<uint32_t>(k * n * sizeof(ElementB));
        AscendC::LocalTensor<ElementB> l1_tensor;
        l1_tensor.SetAddr(l1_v_desc);
        AscendC::LoadData2DParams load_params;
        load_params.repeatTimes = static_cast<uint16_t>((k * n * sizeof(ElementB) + LOADDATA2D_REPEAT_BYTES - 1) / LOADDATA2D_REPEAT_BYTES);
        load_params.srcStride = 0;
        load_params.ifTranspose = false;
        AscendC::LoadData(l0b_tensor, l1_tensor, load_params);
    #else
        (void)l1_v_addr;
        (void)l0b_addr;
    #endif
#else
        if (l1_base_ != nullptr) {
            v_ptr_ = reinterpret_cast<const ElementB*>(static_cast<const uint8_t*>(l1_base_) + l1_v_addr);
        }
#endif
    }

    __aicore__ void LoadL0B(const ElementB* v_ptr, uint32_t k, uint32_t n) {
        v_ptr_ = v_ptr;
        k_ = k;
        n_ = n;
    }

    /**
     * 执行 Cube FP8 MMAD 计算：O += S x V
     *
     * @param init 是否重置 L0C 累加器 (两级累加新周期起始为 true)
     */
    __aicore__ void Compute(bool init = false) {
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    #if defined(__CCE__) && !defined(__CCE_KT_TEST__)
        AscendC::MmadParams mmad_params;
        mmad_params.m = static_cast<uint16_t>(m_);
        mmad_params.n = static_cast<uint16_t>(n_);
        mmad_params.k = static_cast<uint16_t>(k_);
        mmad_params.cmatrixInitVal = init;
        // L0C=CO1，L0A=A2，L0B=B2；Mmad 直接从 L0 起始偏移 0 消费
        AscendC::TBuffAddr l0c_desc{};
        l0c_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
        l0c_desc.bufferHandle = nullptr;
        l0c_desc.bufferAddr = 0U;
        l0c_desc.dataLen = static_cast<uint32_t>(m_ * n_ * sizeof(ElementC));
        AscendC::LocalTensor<ElementC> l0c_tensor;
        l0c_tensor.SetAddr(l0c_desc);
        AscendC::TBuffAddr l0a_desc{};
        l0a_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        l0a_desc.bufferHandle = nullptr;
        l0a_desc.bufferAddr = 0U;
        l0a_desc.dataLen = static_cast<uint32_t>(m_ * k_ * sizeof(ElementA));
        AscendC::LocalTensor<ElementA> l0a_tensor;
        l0a_tensor.SetAddr(l0a_desc);
        AscendC::TBuffAddr l0b_desc{};
        l0b_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        l0b_desc.bufferHandle = nullptr;
        l0b_desc.bufferAddr = 0U;
        l0b_desc.dataLen = static_cast<uint32_t>(k_ * n_ * sizeof(ElementB));
        AscendC::LocalTensor<ElementB> l0b_tensor;
        l0b_tensor.SetAddr(l0b_desc);
        auto l0a_fp8 = l0a_tensor.template ReinterpretCast<AscendC::mx_fp8_e4m3_t>();
        auto l0b_fp8 = l0b_tensor.template ReinterpretCast<AscendC::mx_fp8_e4m3_t>();
        AscendC::Mmad(l0c_tensor, l0a_fp8, l0b_fp8, mmad_params);
    #else
        (void)init;
    #endif
#else
        ElementC* dst = (l0c_base_ != nullptr) ? static_cast<ElementC*>(l0c_base_) : l0c_buf_;
        if (s_ptr_ != nullptr && v_ptr_ != nullptr) {
            for (uint32_t i = 0; i < m_; ++i) {
                for (uint32_t c = 0; c < n_; ++c) {
                    float sum = 0.0f;
                    for (uint32_t j = 0; j < k_; ++j) {
                        float s_val = sage_attn::quant::E4m3fnToFloat(static_cast<uint8_t>(s_ptr_[i * k_ + j]));
                        float v_val = sage_attn::quant::E4m3fnToFloat(static_cast<uint8_t>(v_ptr_[j * n_ + c]));
                        sum += s_val * v_val;
                    }
                    if (init) {
                        dst[i * n_ + c] = sum;
                    } else {
                        dst[i * n_ + c] += sum;
                    }
                }
            }
        } else if (init) {
            std::memset(dst, 0, m_ * n_ * sizeof(ElementC));
        }
#endif
    }

    /**
     * FixPipe Epilogue 回写 UB 主累加器
     * 支持随路量化 (FP32 -> FP16) 与分形 NZ -> ND 转置
     * 支持通过 3 槽延迟输出退休队列（OutputRetireQueue）进行成对融合延迟写回
     */
    __aicore__ void ExecuteEpilogue(
        [[maybe_unused]] uint32_t l0c_src_offset,
        [[maybe_unused]] uint32_t ub_dst_offset,
        const sage_attn::FixpipeConfig& fixpipe_cfg,
        bool use_retire_queue = false,
        uint32_t tile_idx = 0,
        const ElementC* p_data = nullptr,
        float scale = 1.0f,
        float m_new = 0.0f,
        float m_old = 0.0f) {

        ElementC* src = (p_data != nullptr) ? const_cast<ElementC*>(p_data) : GetL0C();
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        if (src != nullptr && fixpipe_cfg.enable_inline_quant) {
            uint32_t total_elements = m_ * n_;
            for (uint32_t idx = 0; idx < total_elements; ++idx) {
                src[idx] = sage_attn::QuantizeToFp16(src[idx]);
            }
        }
#else
        (void)l0c_src_offset;
        (void)ub_dst_offset;
        (void)fixpipe_cfg;
#endif
        if (use_retire_queue && src != nullptr) {
            retire_queue_.Push(tile_idx, src, scale, m_new, m_old);
        }
    }

    __aicore__ const ElementC* GetL0C() const {
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        return (l0c_base_ != nullptr) ? static_cast<const ElementC*>(l0c_base_) : l0c_buf_;
#else
        return nullptr;
#endif
    }

    __aicore__ ElementC* GetL0C() {
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        return (l0c_base_ != nullptr) ? static_cast<ElementC*>(l0c_base_) : l0c_buf_;
#else
        return nullptr;
#endif
    }

private:
    RetireQueue retire_queue_;
    const void* ub_base_{nullptr};
    const void* l1_base_{nullptr};
    void* l0c_base_{nullptr};
    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
    const ElementA* s_ptr_{nullptr};
    const ElementB* v_ptr_{nullptr};
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
    ElementC l0c_buf_[Traits::M * Traits::N]{0.0f};
#endif
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

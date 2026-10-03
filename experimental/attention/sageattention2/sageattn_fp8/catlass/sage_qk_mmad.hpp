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

#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

#include <cstdint>
#include <cstring>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace Catlass::SageAttn {

/**
 * INT8 QK^T 矩阵乘分块与调度策略
 * 负责 L1 -> L0A/L0B -> MMAD -> L0C 的标准硬件通路封装
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 64,
    uint32_t TileK = 128
>
struct SageQkMmadTraits {
    using ElementA = int8_t;    // Query 元素类型 (INT8)
    using ElementB = int8_t;    // Key 元素类型 (INT8)
    using ElementC = int32_t;   // 原始 MMAD 累加类型 (INT32)
    using ElementDequant = float; // 反量化后用于 Softmax 的浮点类型

    using LayoutA = Layout_zZ;  // L0A 中 Q 分形
    using LayoutB = Layout_nZ;  // L0B 中 K 分形
    using LayoutC = Layout_zN;  // L0C 中 P 分形

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;

    // 基本分形块尺寸 (INT8 下 16x16x32)
    static constexpr uint32_t M0 = FRACTAL_M0;
    static constexpr uint32_t N0 = FRACTAL_N0;
    static constexpr uint32_t K0 = FRACTAL_K0_INT8;

    static constexpr uint32_t A_BYTES = M * K * sizeof(ElementA);
    static constexpr uint32_t B_BYTES = N * K * sizeof(ElementB);
    static constexpr uint32_t C_BYTES = M * N * sizeof(ElementC);
};

/**
 * INT8 QK^T 矩阵乘执行器
 */
template <class Traits = SageQkMmadTraits<64, 64, 128>>
class SageQkMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;
    using ElementDequant = typename Traits::ElementDequant;

    // 成员初值统一由声明处 NSDMI 唯一定义，构造函数仅保留宿主仿真缓冲清零
    __aicore__ SageQkMmad() {
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        std::memset(l0c_buf_, 0, sizeof(l0c_buf_));
#endif
    }

    __aicore__ void BindBuffers(const void* l1_base, void* l0c_base) {
        l1_base_ = l1_base;
        l0c_base_ = l0c_base;
    }

    /**
     * 加载 Q 分块：从 L1 Bank 0 搬入 L0A
     */
    __aicore__ void LoadL0A(uint32_t l1_q_addr, [[maybe_unused]] uint32_t l0a_addr, uint32_t m, uint32_t k) {
        m_ = m;
        k_ = k;
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    #if defined(__CCE__) && !defined(__CCE_KT_TEST__)
        // CANN dav-3510 范式（TPipe::GetAbsAddr 同款）：SetAddr 只接受 TBuffAddr 描述符，
        // logicPos 决定硬件单元（A2=L0A，A1=L1 Bank0），dataLen 为字节长度。
        AscendC::TBuffAddr l0a_desc{};
        l0a_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        l0a_desc.bufferHandle = nullptr;
        l0a_desc.bufferAddr = l0a_addr;
        l0a_desc.dataLen = static_cast<uint32_t>(m * k * sizeof(ElementA));
        AscendC::LocalTensor<ElementA> l0a_tensor;
        l0a_tensor.SetAddr(l0a_desc);
        AscendC::TBuffAddr l1_q_desc{};
        l1_q_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        l1_q_desc.bufferHandle = nullptr;
        l1_q_desc.bufferAddr = l1_q_addr;
        l1_q_desc.dataLen = static_cast<uint32_t>(m * k * sizeof(ElementA));
        AscendC::LocalTensor<ElementA> l1_tensor;
        l1_tensor.SetAddr(l1_q_desc);
        AscendC::LoadData2DParams load_params;
        load_params.repeatTimes = static_cast<uint16_t>((m * k * sizeof(ElementA) + LOADDATA2D_REPEAT_BYTES - 1) / LOADDATA2D_REPEAT_BYTES);
        load_params.srcStride = 0;
        load_params.ifTranspose = false;
        AscendC::LoadData(l0a_tensor, l1_tensor, load_params);
    #else
        (void)l1_q_addr;
        (void)l0a_addr;
    #endif
#else
        if (l1_base_ != nullptr) {
            q_ptr_ = reinterpret_cast<const ElementA*>(static_cast<const uint8_t*>(l1_base_) + l1_q_addr);
        }
#endif
    }

    __aicore__ void LoadL0A(const ElementA* q_ptr, uint32_t m, uint32_t k) {
        q_ptr_ = q_ptr;
        m_ = m;
        k_ = k;
    }

    /**
     * 加载 K 分块：从 L1 Bank 1 搬入 L0B
     */
    __aicore__ void LoadL0B(uint32_t l1_k_addr, [[maybe_unused]] uint32_t l0b_addr, uint32_t n, uint32_t k) {
        n_ = n;
        k_ = k;
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    #if defined(__CCE__) && !defined(__CCE_KT_TEST__)
        // L0B=B2，L1 Bank1 (K)=B1
        AscendC::TBuffAddr l0b_desc{};
        l0b_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        l0b_desc.bufferHandle = nullptr;
        l0b_desc.bufferAddr = l0b_addr;
        l0b_desc.dataLen = static_cast<uint32_t>(n * k * sizeof(ElementB));
        AscendC::LocalTensor<ElementB> l0b_tensor;
        l0b_tensor.SetAddr(l0b_desc);
        AscendC::TBuffAddr l1_k_desc{};
        l1_k_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::B1);
        l1_k_desc.bufferHandle = nullptr;
        l1_k_desc.bufferAddr = l1_k_addr;
        l1_k_desc.dataLen = static_cast<uint32_t>(n * k * sizeof(ElementB));
        AscendC::LocalTensor<ElementB> l1_tensor;
        l1_tensor.SetAddr(l1_k_desc);
        AscendC::LoadData2DParams load_params;
        load_params.repeatTimes = static_cast<uint16_t>((n * k * sizeof(ElementB) + LOADDATA2D_REPEAT_BYTES - 1) / LOADDATA2D_REPEAT_BYTES);
        load_params.srcStride = 0;
        load_params.ifTranspose = false;
        AscendC::LoadData(l0b_tensor, l1_tensor, load_params);
    #else
        (void)l1_k_addr;
        (void)l0b_addr;
    #endif
#else
        if (l1_base_ != nullptr) {
            k_ptr_ = reinterpret_cast<const ElementB*>(static_cast<const uint8_t*>(l1_base_) + l1_k_addr);
        }
#endif
    }

    __aicore__ void LoadL0B(const ElementB* k_ptr, uint32_t n, uint32_t k) {
        k_ptr_ = k_ptr;
        n_ = n;
        k_ = k;
    }

    /**
     * 执行 Cube MMAD 计算：P = Q x K^T，累加于 L0C
     *
     * @param init 是否清零 L0C 累加器 (新分块起始为 true，迭代累加为 false)
     */
    __aicore__ void Compute(bool init = true) {
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
        l0b_desc.dataLen = static_cast<uint32_t>(n_ * k_ * sizeof(ElementB));
        AscendC::LocalTensor<ElementB> l0b_tensor;
        l0b_tensor.SetAddr(l0b_desc);
        AscendC::Mmad(l0c_tensor, l0a_tensor, l0b_tensor, mmad_params);
    #else
        (void)init;
    #endif
#else
        ElementC* dst = (l0c_base_ != nullptr) ? static_cast<ElementC*>(l0c_base_) : l0c_buf_;
        if (q_ptr_ != nullptr && k_ptr_ != nullptr) {
            for (uint32_t i = 0; i < m_; ++i) {
                for (uint32_t j = 0; j < n_; ++j) {
                    int32_t dot = 0;
                    for (uint32_t c = 0; c < k_; ++c) {
                        dot += static_cast<int32_t>(q_ptr_[i * k_ + c]) * static_cast<int32_t>(k_ptr_[j * k_ + c]);
                    }
                    if (init) {
                        dst[i * n_ + j] = dot;
                    } else {
                        dst[i * n_ + j] += dot;
                    }
                }
            }
        } else if (init) {
            std::memset(dst, 0, m_ * n_ * sizeof(ElementC));
        }
#endif
    }

    /**
     * 反量化缩放融合：将 INT32 的 MMAD 结果乘以 Q Scale、K Scale 与 sm_scale_log2e
     * 遵循 task/01_design.md 2.2 节与策略四：
     * scale 不缩放 K 本身，而是在 dequant 阶段随 fragment 广播到 QK^T 结果
     */
    __aicore__ static inline float ComputeDequantScale(
        float q_scale,
        float k_scale,
        float sm_scale_log2e) {

        return q_scale * k_scale * sm_scale_log2e;
    }

    /**
     * A7（task/12_device_data_path.md D1 方案甲）：QK^T 的 L0C INT32 结果裸搬至 UB
     * logits 区（dav-3510 FixPipe，quantPre DISABLE = 位型透传，无精度转换）。
     * - 反量化不在本层：INT32→float 精确 cast 与 q_scale·k_scale·sm_scale_log2e
     *   左结合乘序由 V1 侧 DequantMaskLogitsRowVF 承担（对齐 twin::DequantQkLogit
     *   单点；|dot| ≤ 2^21 整数域内 cast 精确，01_design 2.9 节）；
     * - m > 64 时按 64 行两段发射，两段同分区（subBlockId=false）写入 UB 连续两块
     *   （ub_dst_half0/half1），V1 跨段读取；若 bisheng 门/真机否决该形态，按 D1
     *   退 CATLASS CopyL0CToUBTla（SPLIT_M）；
     * - 宿主/KT 分支无硬件 FIX 通路，设备态外为空（与 Mmad 同守卫范式）。
     */
    __aicore__ void EmitL0CToUbRaw(uint32_t ub_dst_half0, uint32_t ub_dst_half1,
                                   uint32_t m, uint32_t n) {
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
        constexpr uint32_t M_SPLIT = 64;
        static constexpr auto format = AscendC::CO2Layout::ROW_MAJOR;
        static constexpr AscendC::FixpipeConfig cfg{format, true};
        AscendC::FixpipeParamsArch3510<format> fp;
        fp.nSize = static_cast<uint16_t>(n);
        fp.srcStride = static_cast<uint16_t>(m);   // L0C zN 行距（整幅行数）
        fp.dstStride = static_cast<uint16_t>(n);   // UB ND 行距
        fp.params.ndNum = 1;
        fp.params.srcNdStride = 0;
        fp.params.dstNdStride = 0;
        // QuantMode_t 为全局作用域枚举（CANN 无 AscendC:: 限定版本），缺省成员
        // NoQuant 即"随路量化关闭"（kernel_utils_struct_dma_params.h 默认值）
        fp.quantPre = QuantMode_t::NoQuant;
        fp.dualDstCtl = 0;
        fp.subBlockId = false;
        AscendC::TBuffAddr l0c_desc{};
        l0c_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
        l0c_desc.bufferHandle = nullptr;
        l0c_desc.bufferAddr = 0U;
        l0c_desc.dataLen = static_cast<uint32_t>(m * n * sizeof(ElementC));
        AscendC::LocalTensor<ElementC> l0c_tensor;
        l0c_tensor.SetAddr(l0c_desc);
        AscendC::TBuffAddr ub0_desc{};
        ub0_desc.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ub0_desc.bufferHandle = nullptr;
        ub0_desc.bufferAddr = ub_dst_half0;
        ub0_desc.dataLen = static_cast<uint32_t>(m * n * sizeof(ElementC));
        AscendC::LocalTensor<ElementC> ub_dst0;
        ub_dst0.SetAddr(ub0_desc);
        if (m <= M_SPLIT) {
            fp.mSize = static_cast<uint16_t>(m);
            AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst0, l0c_tensor, fp);
        } else {
            fp.mSize = static_cast<uint16_t>(M_SPLIT);
            AscendC::TBuffAddr ub1_desc = ub0_desc;
            ub1_desc.bufferAddr = ub_dst_half1;
            ub1_desc.dataLen = static_cast<uint32_t>((m - M_SPLIT) * n * sizeof(ElementC));
            AscendC::LocalTensor<ElementC> ub_dst1;
            ub_dst1.SetAddr(ub1_desc);
            AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst0, l0c_tensor, fp);
            // 第二个 64 行段：L0C 侧按分形行偏移（int32 zN 每分形行 16 列）
            AscendC::Fixpipe<ElementC, ElementC, cfg>(ub_dst1, l0c_tensor[M_SPLIT * 16], fp);
        }
#else
        (void)ub_dst_half0;
        (void)ub_dst_half1;
        (void)m;
        (void)n;
#endif
    }

    __aicore__ const ElementC* GetL0C() const {
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        return (l0c_base_ != nullptr) ? static_cast<const ElementC*>(l0c_base_) : l0c_buf_;
#else
        return (l0c_base_ != nullptr) ? static_cast<const ElementC*>(l0c_base_) : nullptr;
#endif
    }

    __aicore__ ElementC* GetL0C() {
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
        return (l0c_base_ != nullptr) ? static_cast<ElementC*>(l0c_base_) : l0c_buf_;
#else
        return (l0c_base_ != nullptr) ? static_cast<ElementC*>(l0c_base_) : nullptr;
#endif
    }

private:
    const void* l1_base_{nullptr};
    void* l0c_base_{nullptr};
    uint32_t m_{Traits::M};
    uint32_t n_{Traits::N};
    uint32_t k_{Traits::K};
    const ElementA* q_ptr_{nullptr};
    const ElementB* k_ptr_{nullptr};
#if !defined(__CCE__) && !defined(__CCE_KT_TEST__) && !defined(__DAV_C220_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C310_CUBE__) && !defined(__DAV_C310_VEC__)
    ElementC l0c_buf_[Traits::M * Traits::N]{0};
#endif
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

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

#ifndef SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H
#define SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H

#include <cstdint>
#include "tiling.h"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace sage_attn {

/**
 * FixPipe 随路操作配置描述符（白皮书 4.1.1 节硬件数据通路）
 */
struct FixpipeConfig {
    bool enable_inline_quant;   // 是否开启 L0C 回写 UB 随路量化 (FP32 -> FP16)
    bool enable_nz_to_nd;       // 是否开启分形 NZ 到 ND 随路转置排布
    uint32_t quant_dst_type;    // 目标量化格式 (0=FP32, 1=FP16, 2=BF16)
    float quant_scale;          // 随路量化比例缩放因子
};

/**
 * 两级累加归约控制器
 * 支持 fp32, fp32+fp32, fp32+fp16 三档周期归约
 * 严格按照 task/01_design.md 2.4 节和 task.md 2.5 定理二实现
 */
class AccumReduceController {
public:
    __aicore__ AccumReduceController()
        : accum_mode_(PV_ACCUM_FP32_FP16), k_step_(16), total_kv_blocks_(0), current_step_(0) {}

    __aicore__ void Init(uint32_t accum_dtype, uint32_t k_step, uint32_t total_kv_blocks) {
        accum_mode_ = static_cast<PvAccumDtype>(accum_dtype);
        k_step_ = (k_step > 0) ? k_step : 16;
        total_kv_blocks_ = total_kv_blocks;
        current_step_ = 0;
    }

    /**
     * 判断当前迭代是否需要触发 L0C 向 UB 主累加器的 FixPipe 搬出归约
     * 
     * @param step_idx 当前处理的 KV 分块序号 (0-indexed)
     * @param is_last_step 是否为最后一个 KV 分块
     */
    __aicore__ bool ShouldReduce(uint32_t step_idx, bool is_last_step) const {
        if (accum_mode_ == PV_ACCUM_FP32) {
            // fp32 档：950 L0C 具备全 24-bit 尾数，全程在 L0C 累加，仅在最后一步搬出
            return is_last_step;
        }

        // fp32+fp32 和 fp32+fp16 档：每 k_step 周期归约一次，末尾块强制归约
        return is_last_step || (((step_idx + 1) % k_step_) == 0);
    }

    /**
     * 获取当前档位对应的 FixPipe 硬件配置
     */
    __aicore__ FixpipeConfig GetFixpipeConfig() const {
        FixpipeConfig cfg;
        cfg.enable_nz_to_nd = true; // 随路完成 NZ 到 ND 转换

        if (accum_mode_ == PV_ACCUM_FP32_FP16) {
            cfg.enable_inline_quant = true;
            cfg.quant_dst_type = 1; // FP16
            cfg.quant_scale = 1.0f;
        } else {
            // fp32 与 fp32+fp32 保持 FP32 精度
            cfg.enable_inline_quant = false;
            cfg.quant_dst_type = 0; // FP32
            cfg.quant_scale = 1.0f;
        }
        return cfg;
    }

    /**
     * 步进计数器
     */
    __aicore__ void Advance() {
        current_step_++;
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
    uint32_t current_step_;
};

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H

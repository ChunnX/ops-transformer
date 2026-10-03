#ifndef SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H
#define SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H

#include <cstdint>
#include <cstring>
#include "tiling.h"
#include "online_softmax.h"
// FP16 位级转换唯一定义点（STYLE-11 单一定义收敛）
#include "../../quant/quant_common.h"


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

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_ACCUM_REDUCE_H

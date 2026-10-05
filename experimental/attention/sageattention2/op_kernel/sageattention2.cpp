// ENABLE_CV_COMM_VIA_SSBUF 单点定义于 static_slot.h（本文件首个包含头，
// 先于 kernel_operator.h 生效，驱动 SSBUF 硬件通路选择）。
#include "static_slot.h"
#include "attention_sync.h"
#include "attention_cube.h"
#include "attention_vec0_quant.h"
#include "attention_vec1_softmax.h"
#include "attention_vec1_output.h"
#include "sage_attention2_template_tiling_key.h"

#if SAGE_ATTN_DEVICE_COMPILE

// =========================================================================
// 统一编译期诊断宏体系 SAGE_DIAG_STAGE
// 支持编译时通过 -DSAGE_DIAG_STAGE=N 一键注入探针，无需侵入源码：
// 0: 生产全量执行 (Production Run, Default)
// 1: Phase 0 空入口 (Empty Kernel Entry)
// 2: Phase 1 Tiling 加载 (Tiling & Workspace Load Probe)
// 3: Phase 2 Init 调度 (Kernel Init Probe)
// 4: Phase 3 V0 单核流水 (Vector 0 Standalone Pipeline)
// 5: Phase 4 V1 单核流水 (Vector 1 Standalone Pipeline)
// 6: Phase 5 Cube 单核流水 (Cube Standalone Pipeline)
// 7: Phase 6A 纯信号灯握手 (Pure Cross-Core Flag Handshake)
// 8: Phase 6B 循环信号灯握手 (Loop Cross-Core Handshake via Token Pre-setting)
// =========================================================================
#ifndef SAGE_DIAG_STAGE
#define SAGE_DIAG_STAGE 0
#endif

template <class ByteCountT>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingDataHelper(
    const __gm__ uint8_t *p_tilingdata, uint8_t *tilingdata, ByteCountT all_bytes)
{
    const uint64_t copy_bytes = static_cast<uint64_t>(all_bytes);
#if defined(ASCENDC_CPU_DEBUG)
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)p_tilingdata;
    uint32_t *dst = (uint32_t *)tilingdata;
    for (uint64_t i = 0; i < (copy_bytes + 3) / 4; i++) {
        *(dst + i) = *(src + i);
    }
#elif defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || \
      defined(__GET_CODE_CHANNEL__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    copy_data_align64(tilingdata, const_cast<__gm__ uint8_t *>(p_tilingdata), copy_bytes);
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    uint32_t len_burst = (copy_bytes + 31) / 32;
    copy_gm_to_ubuf_align_v2(tilingdata_in_ub, const_cast<__gm__ uint8_t *>(p_tilingdata),
        0, 1, len_burst * 32, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(tilingdata, tilingdata_in_ub, copy_bytes);
#endif
    pipe_barrier(PIPE_ALL);
}

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingDataHelper(tiling_arg, reinterpret_cast<uint8_t *>(&tiling_data), sizeof(tiling_struct))
#endif

extern "C" __global__ __aicore__ void sage_attention2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if SAGE_DIAG_STAGE == 1
    // Phase 0: 空入口测试，验证 PyTorch/ACLNN 调度与内核启动开销
    (void)query; (void)key; (void)value; (void)attentionOut; (void)softmaxLse; (void)workspace; (void)tiling;
    return;
#endif

    REGISTER_TILING_DEFAULT(sage_attn::SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(sage_attn::SageAttention2TilingData, t, tiling);
    AscendC::SetSysWorkspaceForce(workspace);
    __gm__ uint8_t *user_ws = workspace + AscendC::RESERVED_WORKSPACE;

#if SAGE_DIAG_STAGE == 2
    // Phase 1: 硬件核拓扑实测探测 (GetBlockIdx / GetSubBlockIdx 映射真机验证)
    __gm__ uint32_t *out_u32 = reinterpret_cast<__gm__ uint32_t *>(attentionOut);
    uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if ASCEND_IS_AIC {
        out_u32[blk * 8 + 0] = blk;
        out_u32[blk * 8 + 1] = 100; // AIC
    } else {
        uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        out_u32[blk * 8 + 2 + sub * 2] = blk;
        out_u32[blk * 8 + 3 + sub * 2] = 200 + sub; // AIV0 = 200, AIV1 = 201
    }
    return;
#elif SAGE_DIAG_STAGE == 3
    // Phase 2: 实例化内核并执行 Init 任务切分与 Die 拓扑绑定
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    (void)kernel;
    return;
#elif SAGE_DIAG_STAGE == 4
    // Phase 3: Vector 0 单核流水纯计算与访存测试（剥离跨核依赖）
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() == 0) {
            kernel.TestVec0Standalone();
        }
    }
    return;
#elif SAGE_DIAG_STAGE == 5
    // Phase 4: Vector 1 单核流水测试（剥离跨核依赖）
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() == 1) {
            kernel.TestVec1Standalone();
        }
    }
    return;
#elif SAGE_DIAG_STAGE == 6
    // Phase 5: Cube 单核流水 MMAD 计算与 Fixpipe 搬出测试
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    if ASCEND_IS_AIC {
        kernel.TestCubeStandalone();
    }
    return;
#elif SAGE_DIAG_STAGE == 7
    // Phase 6A: 纯跨核信号灯单步握手测试 (Cube + V0 + V1 全信道)
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    if ASCEND_IS_AIC {
        // 1. 与 V0 握手
        sage_attn::AttentionSync::CubeWaitQReady();
        sage_attn::AttentionSync::CubeWaitKvReady(true);
        sage_attn::AttentionSync::CubeReleaseKvFree(true);

        // 2. 与 V1 握手 (QK & Softmax)
        sage_attn::AttentionSync::CubeSignalQkReady();
        sage_attn::AttentionSync::CubeWaitDotFree();
        sage_attn::AttentionSync::CubeWaitSoftmaxReady(true);
        sage_attn::AttentionSync::CubeReleaseSoftmaxFree(true);

        // 3. 与 V1 握手 (PV Partial & Done)
        sage_attn::AttentionSync::CubeSignalPartialReady();
        sage_attn::AttentionSync::CubeWaitPartialFree();
        sage_attn::AttentionSync::CubeSignalPvDone();
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            sage_attn::AttentionSync::V0SignalQReady();
            sage_attn::AttentionSync::V0SignalKvReady(true);
            sage_attn::AttentionSync::V0WaitKvFree(true);
        } else {
            sage_attn::AttentionSync::V1WaitQkReady();
            sage_attn::AttentionSync::V1ReleaseDotFree();
            sage_attn::AttentionSync::V1SignalSoftmaxReady(true);
            sage_attn::AttentionSync::V1WaitSoftmaxFree(true);

            sage_attn::AttentionSync::V1WaitPartialReady();
            sage_attn::AttentionSync::V1ReleasePartialFree();
            sage_attn::AttentionSync::V1WaitPvDone();
        }
    }
    return;
#elif SAGE_DIAG_STAGE == 8
    // Phase 6B: 循环跨核信号灯握手测试 (1-Iteration Handshake with Token Pre-setting)
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    sage_attn::AttentionSync::InitSyncTokens();
    if ASCEND_IS_AIC {
            sage_attn::AttentionSync::CubeWaitQReady();
            sage_attn::AttentionSync::CubeWaitKvReady(true);
            sage_attn::AttentionSync::CubeWaitDotFree();
            sage_attn::AttentionSync::CubeSignalQkReady();
            sage_attn::AttentionSync::CubeWaitSoftmaxReady(true);
            sage_attn::AttentionSync::CubeReleaseSoftmaxFree(true);
            sage_attn::AttentionSync::CubeReleaseKvFree(true);
            sage_attn::AttentionSync::CubeWaitPartialFree();
            sage_attn::AttentionSync::CubeSignalPartialReady();
            sage_attn::AttentionSync::CubeSignalPvDone();
        } else {
            if (AscendC::GetSubBlockIdx() == 0) {
                sage_attn::AttentionSync::V0SignalQReady();
                sage_attn::AttentionSync::V0WaitKvFree(true);
                sage_attn::AttentionSync::V0SignalKvReady(true);
            } else {
                sage_attn::AttentionSync::V1WaitSoftmaxFree(true);
                sage_attn::AttentionSync::V1WaitQkReady();
                sage_attn::AttentionSync::V1ReleaseDotFree();
                sage_attn::AttentionSync::V1SignalSoftmaxReady(true);
                sage_attn::AttentionSync::V1WaitPartialReady();
                sage_attn::AttentionSync::V1ReleasePartialFree();
                sage_attn::AttentionSync::V1WaitPvDone();
            }
        }
#elif SAGE_DIAG_STAGE == 9
    // Phase 6C: V0 与 Cube 纯双拍握手隔离测试 (排除量化计算)
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    sage_attn::AttentionSync::InitSyncTokens();
    if ASCEND_IS_AIC {
        sage_attn::AttentionSync::CubeWaitQReady();
        for (uint32_t k = 0; k < kernel.GetContext().active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            sage_attn::AttentionSync::CubeWaitKvReady(is_ping);
            sage_attn::AttentionSync::CubeReleaseKvFree(is_ping);
        }
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            sage_attn::AttentionSync::V0SignalQReady();
            for (uint32_t k = 0; k < kernel.GetContext().active_kv_blocks; ++k) {
                bool is_ping = ((k % 2) == 0);
                sage_attn::AttentionSync::V0WaitKvFree(is_ping);
                sage_attn::AttentionSync::V0SignalKvReady(is_ping);
            }
        }
    }
    return;
#else
    // Phase 0: 生产全量流水执行
    sage_attn::SageAttnLaunch::RunByDtype(t.input_dtype, query, key, value, attentionOut, softmaxLse, user_ws, &t);
#endif
}
#else
extern "C" void sage_attention2(
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *attentionOut, uint8_t *softmaxLse,
    uint8_t *workspace, uint8_t *tiling)
{
    (void)workspace;
    const auto *t = reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling);
    sage_attn::SageAttnLaunch::RunByDtype(t->input_dtype, query, key, value, attentionOut, softmaxLse, nullptr, t);
}
#endif

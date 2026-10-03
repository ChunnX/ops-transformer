/**
 * \file twin_reference.cpp
 * \brief CPU 孪生参考实现的 C-ABI 出口（本地 ctypes 单测 / gtest 共用）。
 * 量化语义经 extern "C" 引用 quant 层唯一定义，本文件不复制任何量化逻辑。
 */

#include "twin_reference.h"

// quant 层 C-ABI 声明已由 quant_common.h 提供（唯一点）

extern "C" {

/**
 * CPU 孪生全流水入口。参数口径与 TwinIo 字段序一一对应；返回 0 成功。
 * lse 传 nullptr 表示 return_lse=False；key_mean 传 nullptr 且 smooth_k=1 时内部自算 km；
 * lse_correction 为 smooth_k=1 时调用方按开源公式预计算的行级校正（[b*hq*sq] fp32，
 * 与设备核同契约——外部传入，孪生不自算）；smooth_v 仅 pv_accum=fp32 档生效。
 */
int SageAttnTwinRunForward(
    const void* q, const void* k, const void* v,
    const float* key_mean, const float* lse_correction, void* out, float* lse,
    unsigned b, unsigned hq, unsigned hkv, unsigned sq, unsigned sk, unsigned d_og,
    unsigned tensor_layout, unsigned qk_quant_gran, unsigned pv_accum_dtype,
    unsigned is_causal, unsigned smooth_k, unsigned smooth_v, int dtype_mode, float sm_scale) {
    // C-ABI 形参序即 TwinIo 字段序（struct 契约），聚合初始化一处绑定
    const sage_attn::twin::TwinIo io{
        q, k, v, out, lse, key_mean, lse_correction,
        b, hq, hkv, sq, sk, d_og,
        tensor_layout, qk_quant_gran, pv_accum_dtype,
        is_causal, smooth_k, smooth_v, dtype_mode, sm_scale};
    return sage_attn::twin::RunSequential(io);
}

}  // extern "C"

extern "C" {

/**
 * 量化中间态导出（golden 位级比对单测专用）。
 * 必须先调用 SageAttnTwinRunForward；六个出参指针指向静态缓冲，下次 RunForward 前有效。
 * pad_sk_v 出参返回 V 序列维 pad 长度（d_pad 由 q/k/v 各自尺寸隐含）。
 */
void SageAttnTwinQuantDump(const signed char** q8, const signed char** k8,
                           const unsigned char** v8, const float** qs,
                           const float** ks, const float** vs,
                           unsigned* d_pad, unsigned* pad_sk_v) {
    // 直接复用头文件约定的静态缓冲导出接口：
    sage_attn::twin::ExportQuantDump(q8, k8, v8, qs, ks, vs, d_pad, pad_sk_v);
}

}  // extern "C"

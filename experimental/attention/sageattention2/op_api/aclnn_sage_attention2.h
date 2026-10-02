// aclnn 两段式接口声明：SageAttention2（INT8 QK^T + FP8 PV，统一 FP8 PV 入口）
#ifndef ACLNN_SAGE_ATTENTION2_H_
#define ACLNN_SAGE_ATTENTION2_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 第一段：校验参数、完成布局转换并计算 SageAttention2 所需的 workspace 大小。
 *
 * @param q [输入] Query，FP16/BF16。HND：[B, Hq, Sq, D]；NHD：[B, Sq, Hq, D]；末维 stride 须为 1。
 * @param k [输入] Key，FP16/BF16，与 v 同 shape。HND：[B, Hk, Sk, D]；NHD：[B, Sk, Hk, D]。
 * @param v [输入] Value，FP16/BF16，shape 与 k 相同。
 * @param tensorLayout [输入] 张量布局，"HND" 或 "NHD"。
 * @param isCausal [输入] 是否因果 mask；仅 Sq == Sk 时合法。
 * @param qkQuantGran [输入] Q/K INT8 量化粒度，"per_warp" 或 "per_thread"。
 * @param smScale [输入] softmax 缩放系数，须为有限值且绝对值不超过 FLT_MAX。
 * @param pvAccumDtype [输入] PV 两级累加精度，"fp32" / "fp32+fp32" / "fp32+fp16"。
 * @param smoothK [输入] 是否对 K 沿序列维减均值（outlier smoothing）。
 * @param smoothV [输入] 是否对 V 沿序列维减均值；非 fp32 累加档下被忽略。
 * @param returnLse [输入] 是否输出 LSE。
 * @param out [输出] 注意力输出，dtype 与 q 相同；HND：[B, Hq, Sq, D]；NHD：[B, Sq, Hq, D]。
 * @param lse [输出] returnLse 为 true 时须为 FP32 的 [B, Hq, Sq] 张量；为 false 时可传 nullptr。
 * @param workspaceSize [输出] device 侧 workspace 大小（字节）。
 * @param executor [输出] 第二段接口使用的执行器。
 * @return ACLNN_SUCCESS 或对应错误码。
 */
ACLNN_API aclnnStatus aclnnSa2Attention2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const char *tensorLayout, bool isCausal,
    const char *qkQuantGran, double smScale, const char *pvAccumDtype, bool smoothK, bool smoothV, bool returnLse,
    aclTensor *out, aclTensor *lse, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 第二段：在给定 stream 上执行 SageAttention2 计算。
 *
 * @param workspace [输入] device 侧 workspace 地址；workspaceSize 为 0 时可传 nullptr。
 * @param workspaceSize [输入] 第一段接口返回的 workspace 大小（字节）。
 * @param executor [输入] 第一段接口返回的执行器。
 * @param stream [输入] AscendCL stream。
 * @return ACLNN_SUCCESS 或对应错误码。
 */
ACLNN_API aclnnStatus aclnnSa2Attention2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SAGE_ATTENTION2_H_

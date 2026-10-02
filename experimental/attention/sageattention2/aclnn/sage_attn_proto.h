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

#ifndef SAGE_ATTN_PROTO_H
#define SAGE_ATTN_PROTO_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

// 接口数值编码（gran/accum/layout）与共享数学常量的唯一定义位于
// ../sageattn_fp8/ascendc/tiling.h（Host/Device 共享 ABI）；本头经 include
// 复用同一套枚举，并在下方提供唯一的 "字符串属性 -> 数值编码" 映射，
// 禁止在任何 TU 再引入第二套编码表或映射表。
#include "../sageattn_fp8/ascendc/tiling.h"

#if defined(__has_include)
#if __has_include("aclnn/aclnn_base.h")
#include "aclnn/aclnn_base.h"
#if __has_include("opdev/op_errno.h")
#include "opdev/op_errno.h"
#endif
#define SAGE_HAS_CANN_ACLNN_BASE 1
#endif
#endif

#ifndef SAGE_HAS_CANN_ACLNN_BASE
// 本地编译或无 CANN 依赖环境下的轻量类型定义与错误码桩
// （错误码取值对齐 CANN aclnn/opdev/op_errno.h）
typedef int32_t aclnnStatus;
constexpr aclnnStatus ACLNN_SUCCESS = 0;
constexpr aclnnStatus ACLNN_ERR_PARAM_NULLPTR = 161001;
constexpr aclnnStatus ACLNN_ERR_PARAM_INVALID = 161002;
constexpr aclnnStatus ACLNN_ERR_INNER_CREATE_EXECUTOR = 561101;
constexpr aclnnStatus ACLNN_ERR_INNER_NULLPTR = 561103;

struct aclTensor {
    std::vector<int64_t> shape;
    int32_t dtype{0};
    void *data{nullptr};

    int64_t GetShape(size_t dim) const {
        if (dim < shape.size()) return shape[dim];
        return 0;
    }
    size_t GetDimNum() const {
        return shape.size();
    }
};

struct aclOpExecutor {
    virtual ~aclOpExecutor() = default;
    virtual uint64_t GetWorkspaceSize() const { return 0; }
};
typedef void* aclrtStream;
#endif

namespace sage_attn {

/**
 * 接口字符串协议 -> 数值编码的唯一映射（C++ 侧唯一定义点）。
 * torch 桥接（csrc/attention.cpp）传入 HND/NHD、per_warp/per_thread、
 * fp32/fp32+fp32/fp32+fp16；GE 图属性（op_host）传入同义的 BNSD/BSND。
 * 非法串按冻结签名默认值兜底（per_thread / fp32+fp16 / HND），
 * 严格校验由各入口的前置检查负责。
 */
inline int64_t TensorLayoutCodeFromString(const char *layout)
{
    // NHD 与 BSND 同义（[B,S,H,D]）；HND 与 BNSD 同义（[B,H,S,D]）
    return (std::strcmp(layout, "NHD") == 0 || std::strcmp(layout, "BSND") == 0) ? static_cast<int64_t>(LAYOUT_NHD)
                                                 : static_cast<int64_t>(LAYOUT_HND);
}

inline int64_t QuantGranCodeFromString(const char *gran)
{
    return (std::strcmp(gran, "per_warp") == 0) ? static_cast<int64_t>(QUANT_GRAN_PER_WARP)
                                : static_cast<int64_t>(QUANT_GRAN_PER_THREAD);
}

inline int64_t PvAccumCodeFromString(const char *accum)
{
    if (std::strcmp(accum, "fp32") == 0) {
        return static_cast<int64_t>(PV_ACCUM_FP32);
    }
    if (std::strcmp(accum, "fp32+fp32") == 0) {
        return static_cast<int64_t>(PV_ACCUM_FP32_FP32);
    }
    return static_cast<int64_t>(PV_ACCUM_FP32_FP16);
}

// 闭式 Host Tiling 计算核心；头文件 inline 定义见 sage_attn_host_tiling.h，
// 供 aclnn 两段式 C 接口与 op_host TilingImpl 共用（弱符号，避免 op_host 库
// 反向依赖 op_api 层目标）。声明与定义均 inline：二者被多个 TU 交叉包含，
// 非一致 inline 声明属 ODR 隐患。
inline aclnnStatus ComputeHostTiling(
    uint32_t b,
    uint32_t sq,
    uint32_t sk,
    uint32_t hq,
    uint32_t hkv,
    uint32_t d_og,
    double smScale,
    int64_t qkQuantGran,
    int64_t pvAccumDtype,
    bool isCausal,
    int64_t tensorLayout,
    bool smoothK,
    bool smoothV,
    bool hasKeyMean,
    bool hasLse,
    SageAttention2TilingData &tiling_out,
    uint64_t &workspace_bytes_out);
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子第一段接口：aclnnSageAttention2GetWorkspaceSize
 * 功能：
 * 1. 严格校验输入参数合法性（形状匹配、数据类型、布局编码、GQA 头整除性）
 * 2. 闭式代数公式极速计算 Host Tiling 参数并对齐填充 SageAttention2TilingData
 * 3. 精准核算所需 Workspace 显存大小（包含 Split-KV 归并空间与 LSE 临时显存）
 * 4. 构造并返回 aclOpExecutor 上下文
 *
 * @param query [in] 输入 Query 张量，形状 [B, Sq, Hq, D] 或 [B, Hq, Sq, D]
 * @param key [in] 输入 Key 张量，形状 [B, Sk, Hkv, D] 或 [B, Hkv, Sk, D]
 * @param value [in] 输入 Value 张量，形状 [B, Sk, Hkv, D] 或 [B, Hkv, Sk, D]
 * @param keyMeanOptional [in, optional] Key 均值张量 (smooth_k 时传入)，形状 [B, Hkv, D]
 * @param lseCorrectionOptional [in, optional] LSE 修正项张量 (smooth_k 时传入)，形状 [B, Hq, Sq]
 * @param smScale [in] Softmax 缩放系数 (通常为 1.0 / sqrt(D))
 * @param qkQuantGran [in] 量化粒度编码 (2=per_warp, 3=per_thread)
 * @param pvAccumDtype [in] PV 累加精度编码 (0=fp32, 1=fp32+fp32, 2=fp32+fp16)
 * @param isCausal [in] 是否启用因果注意力掩码 (0=False, 1=True)
 * @param tensorLayout [in] 张量内存布局编码 (0=NHD, 1=HND)
 * @param smoothK [in] 是否启用 Key 均值减法平滑
 * @param smoothV [in] 是否启用 Value 均值平滑 (仅全精度 fp32 档有效)
 * @param out [out] 输出 Attention 结果张量，形状同 query
 * @param lse [out, optional] 输出 LSE 权重对数和张量，形状 [B, Hq, Sq]
 * @param workspaceSize [out] 计算得到的 Workspace 显存字节数
 * @param executor [out] 构造完成的算子执行器指针
 * @return 状态码：ACLNN_SUCCESS 表示成功，其余表示失败
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *keyMeanOptional,
    const aclTensor *lseCorrectionOptional,
    double smScale,
    int64_t qkQuantGran,
    int64_t pvAccumDtype,
    bool isCausal,
    int64_t tensorLayout,
    bool smoothK,
    bool smoothV,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * 算子第二段接口：aclnnSageAttention2
 * 功能：
 * 接收第一段生成的执行器与分配好的 Workspace 显存，发射下发至硬件 Stream 执行
 *
 * @param workspace [in] 工作区显存起始指针 (大小需 >= workspaceSize)
 * @param workspaceSize [in] 工作区显存大小
 * @param executor [in] 第一段生成的算子执行器
 * @param stream [in] ACL 执行计算流
 * @return 状态码：ACLNN_SUCCESS 表示成功，其余表示失败
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // SAGE_ATTN_PROTO_H

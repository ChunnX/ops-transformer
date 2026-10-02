# SageAttention2 ACLNN 两段式接口说明文档

本文档介绍 SageAttention2 在昇腾 NPU 上的 ACLNN 两段式标准 C 接口规格、超轻量微秒级 Host Tiling 设计，以及与官方 FIA（aclnnFusedInferAttentionScoreV5）的架构对齐与差异分析。

## 1. 接口设计概述

- 两段式标准架构：遵从 CANN ops-transformer 规范，拆分为第一段 `aclnnSageAttention2GetWorkspaceSize`（参数校验、闭式 Tiling 与显存核算）与第二段 `aclnnSageAttention2`（算子下发与硬件流发射）。
- 超轻量微秒级 Host 调度：规避 FIA 高达 3.7 万行无 cache 动态搜参带来的发射时延瓶颈，全参数采用闭式代数公式计算，单次 Host Launch 耗时压减至 2 微秒以内。
- 1C2V 硬件协同：底层设备核匹配 1 个 Cube 核心与 2 个 Vector 核心（V0 量化 prologue 核，V1 Softmax 与 LSE 核），支持 INT8 QK 矩阵乘与 FP8 PV 矩阵乘流水交错。

## 2. 接口函数原型与规格

### 2.1 第一段：aclnnSageAttention2GetWorkspaceSize

- 函数签名：
  `aclnnStatus aclnnSageAttention2GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *keyMeanOptional, const aclTensor *lseCorrectionOptional, double smScale, int64_t qkQuantGran, int64_t pvAccumDtype, bool isCausal, int64_t tensorLayout, bool smoothK, bool smoothV, const aclTensor *out, const aclTensor *lse, uint64_t *workspaceSize, aclOpExecutor* *executor)`
- 输入参数列表：
  - `query`：必选输入，Query 张量，数据类型支持 FP16 与 BF16，形状为 `[B, Sq, Hq, D]`（NHD 布局）或 `[B, Hq, Sq, D]`（HND 布局）。
  - `key`：必选输入，Key 张量，数据类型支持 FP16 与 BF16，形状为 `[B, Sk, Hkv, D]` 或 `[B, Hkv, Sk, D]`。
  - `value`：必选输入，Value 张量，数据类型支持 FP16 与 BF16，形状与 `key` 一致。
  - `keyMeanOptional`：可选输入，Key 序列维均值张量，仅在 `smoothK=true` 时生效，形状为 `[B, Hkv, D]`。
  - `lseCorrectionOptional`：可选输入，LSE 修正项张量，形状为 `[B, Hq, Sq]`。预留入口：smooth_k 的 LSE 修正当前随算子内部融合（见 `sage_attn_host.cpp`），Host Tiling 不消费该张量。
  - `smScale`：必选参数，浮点缩放因子。默认值 `1.0 / sqrt(head_dim)` 仅由上层接口在 `sm_scale=None` 时折叠后传入；本接口收到非正值或非有限值时返回 `ACLNN_ERR_PARAM_INVALID`，不再静默兜底。
  - `qkQuantGran`：必选参数，量化粒度枚举，取值 2 表示 per_warp，取值 3 表示 per_thread。
  - `pvAccumDtype`：必选参数，PV 累加精度档位，取值 0 表示全精度 fp32，取值 1 表示 fp32+fp32，取值 2 表示 fp32+fp16。
  - `isCausal`：必选参数，布尔标志，true 表示启用因果下三角注意力掩码，false 表示全向稠密注意力。
  - `tensorLayout`：必选参数，内存布局枚举，取值 0 表示 NHD 布局，取值 1 表示 HND 布局。
  - `smoothK`：必选参数，布尔标志，是否在 INT8 量化前对 Key 扣减均值。
  - `smoothV`：必选参数，布尔标志，是否对 Value 扣减均值（注意：仅在 pvAccumDtype=0 全精度档有效，其余档位自动忽略）。
- 输出参数列表：
  - `out`：必选输出，注意力最终输出张量，数据类型与 `query` 一致，形状与 `query` 一致。
  - `lse`：可选输出，Softmax 对数归一化和张量，数据类型为 FP32，形状为 `[B, Hq, Sq]`。
  - `workspaceSize`：必选输出，指向 uint64_t 的指针，接收该算子运行所需的片外 Workspace 显存字节数。
  - `executor`：必选输出，二级指针，接收构造完成的算子执行器上下文句柄。
- 返回值：
  - `ACLNN_SUCCESS`（0）：参数校验通过，Tiling 计算成功并完成执行器构造。
  - `ACLNN_ERR_PARAM_NULLPTR`（161001）：必选指针为 nullptr。
  - `ACLNN_ERR_PARAM_INVALID`（161002）：输入参数非法（包括维度非 4D、GQA 头数无法整除、头维度超过 128、枚举值非法等）。
  - `ACLNN_ERR_INNER_CREATE_EXECUTOR`（561101）：执行器内存分配失败。
- Tiling 下发链路：第一段以闭式 `ComputeHostTiling`（inline 定义见 `./aclnn/sage_attn_host_tiling.h`）计算 Workspace 与 Tiling 一致性校验；kernel 实际使用的 TilingData 由 `./op_host/sage_attention2_tiling.cpp` 注册的 TilingImpl（复用同一 `ComputeHostTiling`）序列化进 `gert::RawTilingData`，在第二段 `CommonOpExecutorRun` 下发阶段由框架写入设备核。

### 2.2 第二段：aclnnSageAttention2

- 函数签名：
  `aclnnStatus aclnnSageAttention2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
- 参数列表：
  - `workspace`：必选输入，由外部框架分配的显存工作区指针。
  - `workspaceSize`：必选输入，显存工作区大小，必须大于等于第一段输出的 `*workspaceSize`。
  - `executor`：必选输入，第一段接口构造并返回的执行器实例。
  - `stream`：必选输入，算子发射目标 ACL 计算流句柄。
- 返回值：
  - `ACLNN_SUCCESS`（0）：算子任务成功发射至计算流。
  - `ACLNN_ERR_PARAM_NULLPTR`（161001）：执行器指针为空。
  - `ACLNN_ERR_PARAM_INVALID`（161002）：工作区大小不足。

## 3. 架构对比与收益分析

### 3.1 对标对象说明

- 对标基线：华为官方全精度与量化融合算子 `aclnnFusedInferAttentionScoreV5`（简称 FIA）。
- 源码参照位置：`./ops-transformer-baseline/`。
- 本算子实现位置：`./ops-transformer/experimental/attention/sageattention2/`。

### 3.2 核心架构差异对比

- 硬件计算形态：
  - FIA：主要基于 FP16 或 INT8 单阶段 GEMM 推进，计算单元承受较高带宽负载。
  - 本算子：双阶段深度交错，INT8 QK^T MMAD 与 FP8 PV MMAD 全程驻留 Cube，吞吐达到 BF16/FP16 的 2 倍。
- 量化与数据搬运整合：
  - FIA：部分量化路径需要额外的 Host/Vector 搬运或中间显存周转。
  - 本算子：将 K 减均值、INT8 量化、V 的 FP8 量化与 ND2NZ 转置作为 prologue stage 深度融合进 Attn 设备核，Q/K/V 仅从片外 GM 读取一次，彻底消灭中间张量往返。
- Host Launch 开销：
  - FIA：包含多达数万行的动态搜索策略与复杂分支判定，短序列场景下 Host 时延占比过大。
  - 本算子：采用纯闭式代数公式，消除搜索与缓存查询开销，发射耗时稳定在微秒级。
- 片上存储与带宽优化：
  - FIA：L1 与 UB 采用常规动态分配队列，易引发 Bank 争用与管线排空。
  - 本算子：实行 L1 512KB 双 Bank 严格隔离（Bank 0 放 Q 与 Q scale，Bank 1 放 KV 与 KV scale），同时采用 Data 与 Scale 紧凑物理相邻排布，一次 MTE2 搬运合并获取数据与比例因子，请求数量直接减半。
- GQA 多头复用机制：
  - FIA：多头处理通常按 head 顺序独立推进。
  - 本算子：沿 M 维聚合至 64 的倍数，一份 L1 KV 被多个 Q head 连续复用，减少 75% 的 L1 重拷流量。

## 4. C++ 接口调用示例

```cpp
#include "aclnn/sage_attn_proto.h"
#include <iostream>

void RunAttentionExample(
    aclTensor *q, aclTensor *k, aclTensor *v,
    aclTensor *out, aclTensor *lse,
    aclrtStream stream)
{
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    double sm_scale = 1.0 / 8.0; // 假设 head_dim = 64
    int64_t qk_quant_gran = 3;   // per_thread 模式
    int64_t pv_accum_dtype = 2;  // fp32+fp16 默认验收档位
    bool is_causal = false;
    int64_t tensor_layout = 1;   // HND 布局 [B, H, S, D]
    bool smooth_k = false;
    bool smooth_v = false;

    // 第一阶段：获取所需工作区大小并构造执行器
    aclnnStatus ret = aclnnSageAttention2GetWorkspaceSize(
        q, k, v,
        nullptr, nullptr,
        sm_scale, qk_quant_gran, pv_accum_dtype,
        is_causal, tensor_layout, smooth_k, smooth_v,
        out, lse,
        &workspace_size, &executor);

    if (ret != ACLNN_SUCCESS) {
        std::cerr << "GetWorkspaceSize failed with code: " << ret << std::endl;
        return;
    }

    void *workspace_device = nullptr;
    if (workspace_size > 0) {
        // 调用 aclrtMalloc 分配设备侧显存
        // aclrtMalloc(&workspace_device, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二阶段：正式下发算子到流执行
    ret = aclnnSageAttention2(workspace_device, workspace_size, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        std::cerr << "aclnnSageAttention2 failed with code: " << ret << std::endl;
    }

    // 清理工作区资源
    // if (workspace_device) aclrtFree(workspace_device);
}
```

## 5. 准入与测试验证

- 结构体验证：通过 `./tests/test_tiling_struct.py` 验证 `SageAttention2TilingData` 结构体严格满足 160 字节与 32 字节对齐。
- C++ Host 接口验证：通过 `./tests/test_aclnn_host.cpp` 覆盖短序列、长序列、MHA、GQA、Causal、32核与28核动态自适应分核，以及异常参数阻断。

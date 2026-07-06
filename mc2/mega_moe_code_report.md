# mega_moe.h 逐行代码解读报告

> **文件路径**: `D:\op-transformer\ops-transformer\mc2\mega_moe\op_kernel\arch35\mega_moe.h`
> **总行数**: 1376
> **功能概述**: 华为 CANN (AscendC) 平台上 Mixture-of-Experts (MoE) 分布式推理的 kernel 实现。涵盖 token dispatch、MX 量化 (fp8/fp4)、分组矩阵乘 (GMM1+SwiGLU+GMM2+Combine)、反排列 (Unpermute) 等完整流程，支持 A8W4 / A4W4 / A8W8 等多种量化路径。

---

## 第1~9行: 版权声明

| 行号 | 代码 | 注释 |
|------|------|------|
| 1 | `/**` | 多行注释开始 |
| 2 | ` * Copyright (c) 2026 Huawei Technologies Co., Ltd.` | 华为技术有限公司版权声明，年份2026 |
| 3 | ` * This program is free software, ...` | 本程序是自由软件，可在 CANN 许可证条款下重新分发和修改 |
| 4 | ` * CANN Open Software License Agreement Version 2.0 (the "License").` | 许可证为 CANN 开源软件许可协议 v2.0 |
| 5 | ` * Please refer to the License for details. ...` | 详情参阅许可证，不合规不得使用本文件 |
| 6 | ` * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES ...` | 软件按"原样"提供，不附任何明示或暗示的担保 |
| 7 | ` * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, ...` | 包括但不限于不侵权、适销性或特定用途适用性担保 |
| 8 | ` * See LICENSE in the root of the software repository ...` | 完整许可证文本见软件仓库根目录 LICENSE |
| 9 | ` */` | 多行注释结束 |

---

## 第10~17行: 文件头注释 & 宏保护

| 行号 | 代码 | 注释 |
|------|------|------|
| 10 | (空行) | 分隔 |
| 11~14 | `/*! \file mega_moe.h \brief */` | Doxygen 文档注释：标注文件名，brief 描述为空(待补充) |
| 15 | (空行) | 分隔 |
| 16 | `#ifndef MEGA_MOE_H` | 防止头文件重复包含的宏定义开头 |
| 17 | `#define MEGA_MOE_H` | 定义 MEGA_MOE_H 宏，后续再包含时跳过整个文件 |

---

## 第19~35行: 头文件包含

| 行号 | 代码 | 注释 |
|------|------|------|
| 19 | `#include "kernel_operator.h"` | AscendC 核心运算符头文件，提供 DataCopy/GatherMask/CompareScalar 等基础 API |
| 20 | `#include "kernel_tiling/kernel_tiling.h"` | tiling 数据结构头文件，提供 MegaMoeTilingData 等切分参数 |
| 21 | `#if __has_include("../../common/mc2_kernel_utils.h")` | 条件编译：检查 mc2_kernel_utils.h 是否存在于相对路径 |
| 22 | `#include "../../common/mc2_kernel_utils.h"` | 若存在则从此路径包含 mc2 公共工具函数 |
| 23 | `#else` | 否则走回退路径 |
| 24 | `#include "../../../common/op_kernel/mc2_kernel_utils.h"` | 从替代路径包含同一文件(不同构建目录结构下的回退) |
| 25 | `#endif` | 条件编译结束 |
| 26 | `#include "kernel_operator_list_tensor_intf.h"` | ListTensor 接口头文件，支持动态分组矩阵乘的列表式权重传入 |
| 27 | `#include "mega_moe_base.h"` | MegaMoE 基类/基础定义头文件(基础常量、枚举等) |
| 28 | `#include "mega_moe_workspace_info.h"` | WorkspaceInfo 定义，管理 GM 上各段 buffer 地址与偏移 |
| 29 | `#include "block_epilogue_swiglu_mx_quant.h"` | SwiGLU + MX 量化 Epilogue 模块头文件 |
| 30 | `#include "mega_moe_impl.h"` | 各路径的具体实现函数头文件(GroupMatmulSwigluQuant 等) |
| 31 | `#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")` | 条件编译：检查量化函数头文件 |
| 32 | `#include "../../moe_distribute_dispatch_v2/quantize_functions.h"` | 若存在则从此路径包含 |
| 33 | `#else` | 否则 |
| 34 | `#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"` | 从替代路径包含量化函数 |
| 35 | `#endif` | 条件编译结束 |

---

## 第37~48行: 命名空间、类型别名、模板宏

| 行号 | 代码 | 注释 |
|------|------|------|
| 37 | `using namespace AscendC;` | 使用 AscendC 命名空间，避免每次调用 API 都加 AscendC:: 前缀 |
| 39 | `namespace MegaMoeImpl {` | 定义 MegaMoeImpl 命名空间，封装本文件所有实现 |
| 40 | `using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;` | 四元组 Shape 类型，描述 GMM 问题尺寸 (M, N, K, ?) |
| 41~43 | `using BlockOffset = Shape<int64_t, ...(14个int64_t)>;` | 十四元组 Shape 类型，描述各 buffer 的偏移量 |
| 45 | `// 预留：XType OutputType TopkWeightsType Weight1Type` | 中文注释说明宏参数含义 |
| 46 | `#define TemplateMegaMoeTypeClass ...` | 定义模板参数列表宏：4 个类型 + 2 个 int32_t 量化/Combine模式 |
| 48 | `#define TemplateMegaMoeTypeFunc ...` | 定义模板实参列表宏，用于特化函数时省去重复书写参数名 |

---

## 第50~71行: MegaMoe 类声明 (public)

| 行号 | 代码 | 注释 |
|------|------|------|
| 50~51 | `template <TemplateMegaMoeTypeClass> class MegaMoe {` | 使用宏展开模板参数列表，声明 MoE 核心算子模板类 |
| 53 | `template <int32_t QM> struct QuantTraits { using OutType = fp8_e4m3fn_t; };` | 量化特征结构体，默认输出类型 fp8_e4m3fn_t (8位浮点E4M3) |
| 54 | `template <> struct QuantTraits<E5M2_QUANT> { using OutType = fp8_e5m2_t; };` | 显式特化：E5M2 模式对应 fp8_e5m2_t (8位浮点E5M2) |
| 55 | `template <> struct QuantTraits<E2M1_QUANT> { using OutType = fp4x2_e2m1_t; };` | 显式特化：E2M1 模式对应 fp4x2_e2m1_t (4位浮点,每字节存2元素) |
| 56 | `using QuantOutType = typename QuantTraits<QuantMode>::OutType;` | 根据 QuantMode 自动推导量化输出数据类型 |
| 57~58 | `using ActivationType = std::conditional<IsSame<QuantOutType,fp4x2_e2m1_t>, uint8_t, QuantOutType>::type;` | 条件类型推导：fp4时取uint8_t(字节打包)，否则取QuantOutType |
| 59 | `using QuantScaleOutType = std::conditional<(QuantMode>=E5M2_QUANT), fp8_e8m0_t, float>::type;` | scale 输出类型：QuantMode>=E5M2 用 fp8_e8m0_t(纯指数)，否则用 float |
| 60~65 | `struct ExpertLoopState { ... };` | 专家循环状态结构体：problemShape(GMM尺寸)、baseOffset(buffer偏移)、expertBeforeCnt(前序累计行数) |
| 66 | `__aicore__ inline MegaMoe() {};` | __aicore__ 标注运行在 AI Core；默认空构造函数 |
| 67~70 | `void Init(GM_ADDR context, GM_ADDR x, ...);` | Init 函数声明：初始化 kernel，传入所有 GM 地址和 tiling 数据 |
| 71 | `__aicore__ inline void Process();` | Process 函数声明：kernel 主执行流程入口 |

---

## 第73~98行: MegaMoe 类声明 (private 函数)

| 行号 | 代码 | 注释 |
|------|------|------|
| 74 | `void DispatchBuffInit();` | 初始化 dispatch 阶段使用的 UB buffer |
| 75 | `void SendAndQuantBuffInit();` | 初始化 SendMaskCal/QuantProcessInRank 阶段使用的 UB buffer |
| 76 | `void UnpermuteBuffInit();` | 初始化 Unpermute 阶段使用的 UB buffer |
| 77 | `void ResetFlagList();` | 清零 workspace 上的同步 Flag 区域 |
| 78 | `void ResetGmm2CombineSyncCounters();` | 清零 GMM2→Combine 同步计数器区域 |
| 79 | `void SendMaskCal();` | 源卡按全局专家 ID 计算 mask 并推送到目标卡 |
| 80 | `void SendCntCal(int32_t localExpertId, uint64_t& sendCnt);` | 目标卡读 count 计数，得到当前专家收到的 token 总数 |
| 81 | `void TripleInfoCalAndDispatch(GMMAddrInfo&, int32_t);` | 计算 token 三元组信息并执行 dispatch，同时写 Flag |
| 82~84 | `template<AddrUpdateMode Mode> bool UpdateGroupParams(...);` | 更新当前专家的 problemShape 和 baseOffset，返回是否有有效 token |
| 85~86 | `template<AddrUpdateMode Mode> void UpdateGlobalBuffer(...);` | 根据偏移更新 GMM 各矩阵的 GM 地址视图 |
| 87 | `void Unpermute();` | 反排列：将各专家还回来的 token 按 topk 权重累加回原位置 |
| 88 | `void InitCombineBuffers();` | 初始化 Combine 所需的 UB buffer 大小 |
| 89~90 | `void ProcessCombine(...);` | Combine 后处理：等同步计数器后读取 triple+GMM2 输出执行 CombineRowGroup |
| 91 | `void CrossRankSyncInWorldSize();` | 全卡(worldSize范围内)软同步 |
| 92 | `void ExpertTokenNumCopyOut();` | 将本卡各专家收到的 token 数搬出到 GM 输出 |
| 93~94 | `void CopyGMToGMPerToken(...);` | 6-buffer 软流水搬运对端 rank 数据至本卡 |
| 95 | `void QuantProcessInRank();` | 对本卡 token 执行 MX 量化(fp8/fp4) |
| 96 | `void GroupMatmulWithSwigluQuant(...);` | GMM1 + SwiGLU + 量化路径入口 |
| 97~98 | `void GroupMatmulWithCombine(...);` | GMM2 + Combine 路径入口 |

---

## 第100~144行: 成员变量 (GM Tensors & 基础参数)

| 行号 | 代码 | 注释 |
|------|------|------|
| 100 | `__gm__ Mc2MoeContext* mc2Context_{nullptr};` | 全局内存指针：MoE 通信上下文(HCCL buffer地址、rankId等) |
| 101 | `Params params_{};` | Params 结构体实例：存储所有 GM 地址、workspaceInfo、peermemInfo、tilingData |
| 103 | `GlobalTensor<int32_t> swigluToGmm2FlagGm_;` | GM Tensor：SwiGLU→GMM2 同步 Flag 区域 |
| 104 | `GlobalTensor<int32_t> expertTokenNumsOut_;` | GM Tensor：各专家接收的 token 数输出 |
| 105 | `GlobalTensor<int32_t> tripleGlobalTensor_;` | GM Tensor：token 三元组信息 (rank, tokenIndex, topkIndex) |
| 106 | `GlobalTensor<int32_t> expertRevNumsGlobalTensor_;` | GM Tensor：各专家收到的 token 数(AIC/AIV间通信) |
| 107~109 | `// A8W4 路径下 cumsum 覆盖 V1 UB...cumsumInfoGlobalTensor_ 为 GM 持久备份` | 中文注释：A8W4下 SwigluQuant 覆盖 V1 UB，cumsumInfoGlobalTensor_ 为 cumsum 的 GM 持久化备份 |
| 110 | `GlobalTensor<int32_t> cumsumInfoGlobalTensor_;` | GM Tensor：cumsum 累加信息的 GM 持久化存储 |
| 112 | `uint32_t m_ = 0;` | 本卡 token 总数(batch size × seq) |
| 113 | `uint32_t k_ = 0;` | hidden 维度大小 |
| 114 | `uint32_t aicNum_ = 0;` | AI Core(Cube核心)数量 |
| 115 | `uint32_t topK_ = 0;` | TopK 选择的专家数 |
| 116 | `uint32_t rankId_ = 0;` | 当前卡(rank)编号 |
| 117 | `uint32_t worldSize_ = 0;` | 通信域中的总卡数 |
| 118 | `uint32_t expertPerRank_ = 0;` | 每张卡上的专家数量 |
| 119 | `int64_t hiddenDim_ = 0;` | hidden 维度(可能与 k_ 不同) |
| 120 | `uint64_t maxOutputSize_ = 0;` | 最大输出 token 数(dispatch flag槽位分配用) |
| 121 | `int32_t vecSetSyncCom_ = 0;` | Vector 与 Cube 间的同步通信计数器 |
| 122 | `uint32_t startBlockIdx_ = 0;` | GMM 起始 block 索引 |
| 123 | `uint32_t blockNumPerRank_ = 2;` | 每张卡上的 block 数量，默认为 2 |
| 124 | `int32_t dispatchFlagSlotsPerExpert_ = 0;` | 每个专家的 dispatch Flag 槽位数 |
| 125 | `int32_t maxWavesPerExpert_ = 0;` | 每个专家的最大 wave 数(L1 分片粒度) |
| 126 | `uint32_t blockNum_ = GetBlockNum();` | 获取当前芯片的总 block 数 |
| 127 | `uint32_t blockAivNum_ = GetBlockNum() * 2;` | AIV数量 = block数 × 2(每个block有2个subBlock) |
| 128 | `uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();` | 当前 block 的逻辑索引 |
| 129 | `uint32_t aivCoreIdx_ = GetBlockIdx();` | 当前 AIV 核心的物理索引 |
| 130 | `uint32_t subBlockIdx_ = GetSubBlockIdx();` | 当前 subBlock 索引(0或1，区分V0/V1) |
| 131 | `uint32_t mxQuantScaleNumAlignPerToken_ = 0;` | MX量化中每个token的scale数量(按32对齐) |
| 132 | `uint32_t mxQuantTokenAlignBytes_ = 0;` | MX量化中每个token数据的对齐字节数 |
| 133 | `uint32_t mxQuantScaleAlignBytes_ = 0;` | MX量化中每个token scale的对齐字节数 |
| 134 | `uint32_t mxQuantTokenScaleAlignBytes_ = 0;` | MX量化中每个token+scale总对齐字节数 |
| 135 | `uint32_t ubBufferUsedAddr_ = 0;` | UB buffer 已使用地址的追踪指针 |
| 136 | `uint16_t gmm2PingPongIdx_ = 0;` | GMM2 的 ping-pong buffer 索引 |
| 137 | `uint64_t sendTotalNum_ = 0;` | 发送 token 总数 = m_ × topK_ |
| 138 | `uint32_t maskAlignSize_ = 0;` | mask 位图的对齐大小(字节) |
| 139 | `uint32_t maskSlotSize_ = 0;` | 每个 win 槽位大小 = maskAlignSize + 32B(count) |
| 140 | `uint64_t maskWinOffset_ = 0;` | mask 接收区相对 win 基址偏移 |
| 141 | `uint64_t quantWinOffset_ = 0;` | 量化数据区相对 win 基址偏移 |
| 142 | `uint64_t cumsumRevCntInRank_ = 0;` | 本卡累计接收 token 数(跨专家累加) |
| 143 | `int32_t compareCount_ = 0;` | CompareScalar 所需元素数 |
| 144 | `int64_t combineUbTensorSize_ = 0;` | Combine UB Tensor 的元素数量 |

---

## 第146~178行: 成员变量 (编译期常量 & UB Tensors)

| 行号 | 代码 | 注释 |
|------|------|------|
| 146 | `static constexpr uint32_t A_ELEMS_PER_BYTE = ... ? 2U : 1U;` | 编译期常量：A矩阵每字节元素数，fp4时2，其余1 |
| 147 | `static constexpr uint32_t B_ELEMS_PER_BYTE = ... ? 2U : 1U;` | 编译期常量：B矩阵每字节元素数，fp4时2，其余1 |
| 148~150 | `static constexpr bool ENABLE_A8W4 = IsSame<W1,fp4> && IsSame<QOut,fp8_e4m3>;` | 启用 A8W4 路径(fp8激活+fp4权重1)的条件 |
| 151~154 | `static constexpr bool ENABLE_A4W4 = IsSame<W1,fp4> && IsSame<QOut,fp4>;` | 启用 A4W4 路径(fp4激活+fp4权重)的条件；GMM1走generic a4w4，GMM2走a8w4避免精度损失 |
| 155 | `static constexpr int32_t DISPATCH_BUFFER_NUM = 6;` | dispatch 软流水 buffer 数为 6(EVENT_ID0..5) |
| 156~178 | 各种 `LocalTensor<>` 成员 | UB 局部 Tensor：分别用于 dispatch、量化、mask计算、Unpermute 等各阶段 |

---

## 第180~193行: 成员变量 (SwigluQuantOutType & BlockEpilogue)

| 行号 | 代码 | 注释 |
|------|------|------|
| 180~182 | `// GMM2 走 A8W4 且 QuantMode==E2M1 时，SwigluQuant 输出需提升为 fp8_e4m3fn_t ...` | 中文注释：fp4量化下SwigluQuant输出需类型提升，避免A=fp4 vs B=fp8类型不匹配 |
| 183~185 | `using SwigluQuantOutType = conditional<(QuantMode==E2M1), fp8_e4m3fn_t, QuantOutType>::type;` | fp4量化时提升为fp8_e4m3fn_t，否则保持QuantOutType |
| 188 | `static constexpr uint32_t C_ELEMS_PER_BYTE = ... ? 2U : 1U;` | C矩阵每字节元素数，fp4时2，其余1 |
| 190~191 | `using BlockEpilogue = BlockEpilogueSwigluMxQuant<...>;` | Epilogue模块类型别名 |
| 192 | `BlockEpilogue epilogueOp_;` | Epilogue运算符实例 |
| 193 | `};` | MegaMoe 类声明结束 |
---

## 第198~266行: Init 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 195~197 | `// Init：初始化 & 偏移计算` | 函数功能说明 |
| 198 | `template <TemplateMegaMoeTypeClass>` | 模板参数宏展开 |
| 199 | `__aicore__ inline void MegaMoe::Init(...)` | Init 函数实现开始 |
| 204 | `m_ = tilingData->bs;` | 从 tilingData 获取 batch size(本卡 token 总数)赋给 m_ |
| 205 | `k_ = tilingData->h;` | 从 tilingData 获取 hidden 维度赋给 k_ |
| 206 | `aicNum_ = tilingData->aicNum;` | 获取 AIC(Cube核心)数量 |
| 207 | `topK_ = tilingData->topK;` | 获取 TopK 专家数 |
| 208 | `sendTotalNum_ = m_ * topK_;` | 发送总量 = token数 × TopK数 |
| 209 | `worldSize_ = tilingData->epWorldSize;` | 获取 EP 通信域总卡数 |
| 210 | `expertPerRank_ = tilingData->expertPerRank;` | 获取每卡专家数 |
| 211 | `blockNumPerRank_ = tilingData->blockNumPerEP;` | 获取每卡 block 数量 |
| 212 | `maxOutputSize_ = tilingData->maxOutputSize;` | 获取最大输出 token 数 |
| 214~217 | `maxWavesPerExpert_ = CeilDiv(maxOutputSize_, DISPATCH_WAVE_TILE_M); dispatchFlagSlotsPerExpert_ = CeilAlign(..., INT_CACHELINE);` | 每专家最大wave数和dispatch Flag位数计算，须与 WorkspaceInfo 分配公式一致 |
| 218 | `hiddenDim_ = tilingData->hiddenDim;` | 获取 hiddenDim |
| 219 | `mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);` | 将 context GM 地址 reinterpret 为 Mc2MoeContext 结构体指针 |
| 220 | `rankId_ = mc2Context_->epRankId;` | 从上下文获取当前卡的 EP rank 编号 |
| 221~223 | `for(int i=0; i<worldSize_; i++) winRankAddr_[i] = mc2Context_->epHcclBuffer[i];` | 将每卡的 HCCL win buffer 地址存入 winRankAddr_ 数组(用于跨卡读写) |
| 224~229 | `params_.aGmAddr=x; params_.expertIdxGmAddr=topkIds; params_.bGmAddr=GetTensorAddr(0,weight1); ...` | 将各输入 GM 地址存入 params_；GetTensorAddr 取列表式 Tensor 的第0组地址 |
| 231~233 | `params_.y2GmAddr=yOut; params_.expertTokenNumsOutGmAddr=expertTokenNumsOut; params_.probsGmAddr=topkWeights;` | 将输出地址存入 params_ |
| 234 | `params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);` | 构造 WorkspaceInfo 实例，计算各段 buffer 地址 |
| 235 | `params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);` | 构造 PeermemInfo 实例，计算 peermem 各段偏移 |
| 236 | `params_.tilingData = tilingData;` | 将 tilingData 指针存入 params_ |
| 237 | `expertTokenNumsOut_.SetGlobalBuffer(...)` | 设置 expertTokenNumsOut_ GlobalTensor 的 GM buffer 地址 |
| 238 | `expertRevNumsGlobalTensor_.SetGlobalBuffer(...)` | 设置 expertRevNumsGlobalTensor_ 的 GM buffer 地址(workspace中专家接收数区域) |
| 239 | `tripleGlobalTensor_.SetGlobalBuffer(...)` | 设置 tripleGlobalTensor_ 的 GM buffer 地址(三元组信息区域) |
| 240~245 | `cumsumStride = CeilAlign(worldSize*expertPerRank*sizeof(int32), ALIGN_32); cumsumInfoGlobalTensor_.SetGlobalBuffer(... + cumsumStride * blockIdx_)` | 每个block负责不同专家的cumsum区域；cumsumStride = 每专家间步长(字节) |
| 246~248 | `epilogueOp_.Init({swigluQuantDataPtr, swigluQuantScalePtr, flagSwiGluToGmm2Ptr, ..., tilingData->clamp_limit})` | 初始化 Epilogue 运算符，传入 SwiGLU 量化数据和 scale 的 GM 地址及 clamp 上限 |
| 249~253 | `maskWinOffset_ = maskRecvPtr - rankSyncInWorldPtr; quantWinOffset_ = quantTokenScalePtr - rankSyncInWorldPtr;` | 各 win 区相对于 win 基址的偏移；所有卡 win 布局一致，跨卡读写用同一偏移 |
| 255~256 | `compareCount_ = CeilAlign(sendTotalNum_*sizeof(int32), ALIGN_256) / sizeof(int32);` | CompareScalar 所需元素数(256B对齐后的int32数量) |
| 257 | `maskAlignSize_ = CeilAlign(compareCount_/8, ALIGN_32);` | mask 位图对齐大小 = 元素数/8(每元素1bit)上取整对齐到32B |
| 259 | `maskSlotSize_ = maskAlignSize_ + ALIGN_32;` | 每个win槽位 = mask对齐大小 + 32B(count字段) |
| 260 | `mxQuantScaleNumAlignPerToken_ = CeilDiv(k_, ALIGN_32);` | MX量化每token的scale数量 = k上取整除32(MX每32元素一组) |
| 261~262 | `mxQuantTokenAlignBytes_ = CeilAlign(k_/A_ELEMS_PER_BYTE, ALIGN_256) * sizeof(ActivationType);` | 量化后每token数据字节数(256B对齐) |
| 263 | `mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);` | 每token scale字节数(scale为e8m0格式,1字节) |
| 264~265 | `mxQuantTokenScaleAlignBytes_ = CeilAlign(mxQuantTokenAlignBytes_+mxQuantScaleAlignBytes_, ALIGN_32);` | 每token+scale总对齐字节数(32B对齐) |
| 266 | `}` | Init 函数体结束 |

---

## 第272~338行: DispatchBuffInit 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 268~270 | `// DispatchBuffInit：SendCntCal & TripleInfoCalAndDispatch & ExpertTokenNumCopyOut 中使用的buffer申请` | 函数功能说明 |
| 274~276 | `if constexpr(g_coreType == AIC) { return; }` | AIC 直接返回，以下 buffer 仅 AIV 使用 |
| 280~283 | `expertTokenCntTensor_ = LocalTensor<int32>(VECCALC, addr=0, size=ALIGN_32/sizeof(int32));` | 申请 32B 的专家token计数临时 Tensor(SendCntCal使用) |
| 286~290 | `cumsumInfoTensor_ = LocalTensor<int32>(VECCALC, addr=紧接, size=CeilAlign(...)/sizeof(int32));` | 申请 cumsum 累加信息 Tensor(worldSize × expertPerRank 个int32,32B对齐) |
| 293~298 | `gatherMaskTensor_ / gatherMaskInt32Tensor_ = LocalTensor(VECCALC, addr=紧接, size=maskSlotSize*worldSize);` | 申请 [mask|count] 槽位数据 Tensor(uint8和uint32双视角reinterpret) |
| 301~305 | `validTopkIndexTensor_ = LocalTensor<int32>(VECCALC, addr=紧接, ...);` | 申请 GatherMask dst Tensor(sendTotalNum × int32, 32B对齐) |
| 308~312 | `topkIndexTensor_ = LocalTensor<int32>(VECCALC, addr=紧接, ...);` | 申请 GatherMask src Tensor(同大小) |
| 315~324 | `copyTmpTensors_[0..5] = LocalTensor<Activation>(VECCALC, addr=紧接+index*COPY_TMP_BUFFER_SIZE, ...);` | 申请 6 个 dispatch 软流水 buffer(每块容纳 token+scale) |
| 327~332 | `expertTokenNumsOutTensor_ = LocalTensor<int32>(VECCALC, addr=紧接, ...);` | 申请专家token数输出 Tensor(expertPerRank × int32, 32B对齐) |
| 334 | `ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;` | 更新 UB 已使用地址追踪指针 |
| 335 | `CreateVecIndex(topkIndexTensor_, 0, N);` | 初始化 topkIndexTensor_ 为连续递增索引 [0,1,2,…,N-1] |
| 336 | `Duplicate<int32>(cumsumInfoTensor_, 0, N);` | 将 cumsumInfoTensor_ 全部填 0(初始化) |
| 337 | `PipeBarrier<PIPE_ALL>();` | 全管线屏障，确保初始化操作完成 |
| 338 | `}` | DispatchBuffInit 函数体结束 |

---

## 第344~413行: SendAndQuantBuffInit 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 346~348 | `if constexpr(g_coreType == AIC) { return; }` | AIC 直接返回 |
| 351~354 | `topkIdsTensor_ = LocalTensor<int32>(VECCALC, addr=0, ...);` | 申请 topkIds Tensor(256B对齐，CompareScalar要求) |
| 358~370 | `resetTensor_ = LocalTensor<int32>(VECCALC, ...);` | 申请清零 workspace Flag 用的零值源 Tensor；大小 = Flag区域总量均分到各AIV核，取较大值(含Combine同步计数器) |
| 371 | `Duplicate<int32>(resetTensor_, 0, N);` | 将 resetTensor_ 全部填 0 |
| 374~376 | `mxTempTensor_ = LocalTensor<uint16>(VECCALC, addr=紧接, size=2KB);` | 申请 MX量化临时 Tensor(2KB，存放maxExp/halfScale) |
| 379~389 | `xOutTensor1_/2_ = LocalTensor<Activation>(VECCALC, addr=紧接, ...);` | 申请量化输出双buffer(token对齐256B + scale数量) |
| 392~397 | `xInTensor1_/2_ = LocalTensor<bfloat16>(VECCALC, addr=紧接, ...);` | 申请量化输入双buffer(k上取整对齐128 × sizeof(bfloat16)) |
| 400~404 | `sendMaskTensor_[0..1] = LocalTensor<uint8>(VECCALC, addr=紧接+index*maskSlotSize, maskSlotSize);` | 申请 SendMaskCal 的 ping-pong 双buffer(mask槽位大小) |
| 408~412 | `sendGatherOutTensor_ = LocalTensor<int32>(VECCALC, addr=紧接, ...);` | 申请 GatherMask 废弃 dst scratch(256B对齐) |

---

## 第419~442行: ResetFlagList 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 421~423 | `if constexpr(g_coreType == AIC) { return; }` | AIC 直接返回 |
| 427 | `swigluToGmm2FlagGm_.SetGlobalBuffer(...);` | 设置 SwiGLU→GMM2 Flag 的 GM 地址(三段Flag连续排列，起始即SwiGLU段) |
| 428~430 | `flagNum = expertPerRank * (INT_CACHELINE + dispatchFlagSlots + INT_CACHELINE * aicNum);` | 三段 Flag 区域总 int32 数(SwiGLU段+Dispatch段+SendCntCal段) |
| 431~432 | `TilingByCore(flagNum, coreLen, coreOffset, 1);` | 将 flagNum 均分到各核，计算每核负责的长度和偏移 |
| 433 | `DataCopyExtParams rankSyncCopyParams{1, coreLen*sizeof(int32), 0,0,0};` | DataCopy 参数 |
| 434 | `SyncFuncStatic<V_MTE3, SYNC_EVENT_ID2>();` | 等待 Vector→MTE3 同步(Duplicate 填零已完成) |
| 435~436 | `if(coreLen!=0) DataCopyPad(swigluToGmm2FlagGm_[coreOffset], resetTensor_, ...);` | 将 resetTensor_(全0)搬到 GM 上本核负责的 Flag 区域 |
| 439~441 | `if constexpr(CombineQuantMode!=COMBINE_NO_QUANT) ResetGmm2CombineSyncCounters();` | Combine量化模式下还需清零 GMM2→Combine 同步计数器 |

---

## 第448~470行: ExpertTokenNumCopyOut 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 450~457 | `if constexpr(ENABLE_A8W4) { DataCopyPad(cumsumInfo from GM); SetFlag/WaitFlag<MTE2_S>; }` | A8W4路径下 cumsum 被 SwigluQuant 覆盖，从 GM 恢复 |
| 458 | `lastRankIdx = worldSize - 1;` | 最后一个 rank 的索引(cumsum中最后一个rank值即累计总数) |
| 459 | `expertTokenNumsOutTensor_.SetValue(0, cumsum[lastRank]);` | 第0个专家token数 = cumsum[0*worldSize+lastRank] |
| 460~463 | `for(expertIdx=1..) { cur=cumsum[exp*worldSize+last]; prev=cumsum[(exp-1)*worldSize+last]; SetValue(exp, cur-prev); }` | 第1到N-1个专家token数 = cumsum差值 |
| 465 | `SyncFuncStatic<S_MTE3, SYNC_EVENT_ID2>();` | 等待 Scalar→MTE3 同步 |
| 466~467 | `DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, ...);` | 将专家token数从UB搬到GM输出 |
| 468~469 | `SetFlag/WaitFlag<MTE3_MTE2>(0);` | 确保搬出完成 |

---

## 第479~535行: SendMaskCal 函数实现

**Phase 1 (485~493)**: 加载本卡 topkIds 到 UB

| 行号 | 代码 | 注释 |
|------|------|------|
| 486~487 | `srcGlobalTensor.SetGlobalBuffer(params_.expertIdxGmAddr);` | 设置 topkIds 全局 Tensor 地址 |
| 488 | `Duplicate(topkIdsTensor_, 0, compareCount_);` | 先填0(尾部padding清零) |
| 489 | `SyncFuncStatic<V_MTE2, SYNC_EVENT_ID1>();` | 等待 Duplicate 完成 |
| 490~492 | `DataCopyPad(topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);` | 从 GM 搬入 topkIds 到 UB |
| 493 | `SyncFuncStatic<MTE2_V, SYNC_EVENT_ID1>();` | 等待搬入完成 |

**Phase 2 (495~534)**: 逐全局专家算 [mask|count] 并 ping-pong 发送

| 行号 | 代码 | 注释 |
|------|------|------|
| 497 | `kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};` | 双buffer事件ID |
| 498~500 | `SetFlag<MTE3_V>(kBufEvents[0..1]);` | 预设标志(buffer初始空闲) |
| 501 | `totalExperts = worldSize * expertPerRank;` | 全局专家总数 |
| 502 | `countWordIdx = maskAlignSize / sizeof(int32);` | count在槽内的int32偏移 |
| 503 | `maskCopyParams{1, maskSlotSize, 0,0,0};` | DataCopy发送参数 |
| 506 | `for(curExpertId=aivCoreIdx_; curExpertId<totalExperts; curExpertId+=blockAivNum_, ++iter)` | 逐全局专家遍历，各核分摊 |
| 507 | `dstRank = curExpertId / expertPerRank;` | 目标卡编号 |
| 508 | `expertNumPer = curExpertId % expertPerRank;` | 目标卡内专家编号 |
| 509~512 | `eventId/maskBuf/maskBufU32/maskBufI32 = 按iter轮转选择ping-pong槽位` | 选择当前buffer |
| 514 | `WaitFlag<MTE3_V>(eventId);` | 等待上一轮发送完成(buffer空闲) |
| 515 | `CompareScalar(maskBuf, topkIdsTensor_, curExpertId, CMPMODE::EQ, compareCount_);` | 对 topkIds 中每个元素与 curExpertId 做 EQ 比较，结果写入 maskBuf(bit mask) |
| 517~519 | `GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true, sendTotalNum_, {1,1,0,0}, sendCnt);` | GatherMask：收集匹配元素，同时统计 count |
| 520 | `SyncFuncStatic<V_S, SYNC_EVENT_ID2>();` | 等待 sendCnt 标量就绪 |
| 521 | `maskBufI32.SetValue(countWordIdx, sendCnt);` | 将 count 写入槽内末尾 int32 |
| 522 | `SyncFuncStatic<S_MTE3, SYNC_EVENT_ID3>();` | 确保 count 对 MTE3 可见 |
| 524~528 | `dstOffset = maskWinOffset + (expertNumPer*worldSize+rankId)*maskSlotSize; dstGlobalTensor.SetGlobalBuffer(GetRankWinAddrWithOffset(dstRank, dstOffset));` | 计算目标卡 win 中的偏移并设置全局 Tensor 地址 |
| 529 | `DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);` | 将 [mask|count] 搬出到目标卡 win |
| 530 | `SetFlag<MTE3_V>(eventId);` | 标记发送完成，buffer可复用 |
| 532~534 | `WaitFlag<MTE3_V>(kBufEvents[0..1]);` | 收尾等待所有发送完成 |

---

## 第541~597行: QuantProcessInRank 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 543~545 | `if constexpr(g_coreType == AIC) { return; }` | AIC 直接返回 |
| 548~550 | `TilingByCore(m_, curentNum, curentOffset, 1);` | 按 batch size 与 AIV 核数均分 token |
| 551 | `H = k_;` | hidden 维度 |
| 554 | `xCopyInParams = {1, H*sizeof(bfloat16), 0, 0};` | 搬入参数 |
| 555 | `xCopyInPadParams{true, 0,0,0};` | Pad 参数(启用padding) |
| 556~557 | `xCopyOutParams = {1, mxQuantTokenAlignBytes+mxQuantScaleAlignBytes, 0,0,0};` | 搬出参数(token+scale总大小) |
| 558~559 | `SetFlag<MTE3_MTE2>(EVENT_ID0/1);` | 预设双buffer事件(初始空闲) |
| 560~594 | `for(index=0..curentNum) { ... }` | 逐 token 双 buffer 软流水量化处理 |
| 561~564 | `srcGlobalTensor/dstGlobalTensor.SetGlobalBuffer(...);` | 设置当前 token 的输入/输出 GM 地址 |
| 565~567 | `event/xInTensor/xOutTensor = ping-pong 选择` | 选择当前 buffer |
| 568 | `WaitFlag<MTE3_MTE2>(event);` | 等待上一轮搬出完成 |
| 569 | `DataCopyPad(xInTensor, srcGlobalTensor, ...);` | 从 GM 搬入 token 到 UB |
| 570~571 | `SetFlag/WaitFlag<MTE2_V>(event);` | 确保搬入完成 |
| 572~577 | `获取各区域 UB 物理地址: srcAddr, maxExpAddr, halfScaleAddr, outDataAddr, mxScaleAddr` | 获取量化计算所需的各区域地址 |
| 579 | `quant::ComputeMaxExp(srcAddr, maxExpAddr, H);` | 计算每组元素的最大指数值 |
| 580~581 | `quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, ...);` | 根据 maxExp 计算量化 scale |
| 582~588 | `if constexpr(E2M1) ComputeFp4Data; else ComputeFp8Data;` | fp4量化路径用 ComputeFp4Data，fp8用 ComputeFp8Data |
| 589~590 | `SetFlag/WaitFlag<V_MTE3>(event);` | 等待量化计算完成 |
| 591~592 | `DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);` | 将量化结果(token+scale)搬到 GM 的 peermem 区域 |
| 593 | `SetFlag<MTE3_MTE2>(event);` | 标记搬出完成 |
| 595~596 | `WaitFlag<MTE3_MTE2>(EVENT_ID0/1);` | 等待双buffer搬出完成 |

---

## 第607~656行: SendCntCal 函数实现

**Phase 1 (614~628)**: 从本卡 win 读入当前专家的全部 [mask|count] 槽位

| 行号 | 代码 | 注释 |
|------|------|------|
| 610 | `sendCnt = 0;` | 初始化 sendCnt |
| 611 | `slotWordStride = maskSlotSize / sizeof(int32);` | 每槽位占的 uint32 数量 |
| 612 | `countWordIdx = maskAlignSize / sizeof(int32);` | count 在槽内的 uint32 偏移 |
| 615~617 | `maskSrcGlobal.SetGlobalBuffer(maskRecvPtr + localExpertId*worldSize*maskSlotSize);` | 设置当前专家的 mask 源地址 |
| 618 | `DataCopy(gatherMaskTensor_, maskSrcGlobal, worldSize*maskSlotSize);` | 搬入 worldSize 个槽位数据 |
| 620~627 | `if constexpr(ENABLE_A8W4) { if(localExpertId!=0) DataCopyPad(cumsum from GM); }` | A8W4路径下恢复前序cumsum |
| 629 | `SyncFuncStatic<MTE2_S, SYNC_EVENT_ID2>();` | 等待count标量就绪 |
| 630 | `SyncFuncStatic<MTE2_V, SYNC_EVENT_ID1>();` | 等待mask供GatherMask使用 |

**Phase 2 (632~638)**: 逐源卡读取 count 并累加

| 行号 | 代码 | 注释 |
|------|------|------|
| 633 | `for(calRankId=0..worldSize)` | 逐源卡循环 |
| 634 | `perRankCnt = gatherMaskInt32.GetValue(calRankId*slotWordStride+countWordIdx);` | 从槽内读取该源卡的 count 值 |
| 635 | `sendCnt += perRankCnt;` | 累加到 sendCnt |
| 636 | `cumsumRevCntInRank_ += perRankCnt;` | 累加到跨专家累计数 |
| 637 | `cumsumInfoTensor_.SetValue(localExpertId*worldSize+calRankId, cumsumRevCntInRank_);` | 写入 cumsum 对应位置 |

**Phase 3 (640~656)**: 写 GM 并通知 AIC

| 行号 | 代码 | 注释 |
|------|------|------|
| 641 | `expertTokenCntTensor_.SetValue(0, sendCnt);` | 将 sendCnt 写入临时 Tensor |
| 642 | `SyncFuncStatic<S_MTE3, SYNC_EVENT_ID2>();` | 等待写入完成 |
| 643~644 | `DataCopy(expertRevNumsGlobalTensor_[偏移], expertTokenCntTensor_, INT32_PER_256B);` | 将 sendCnt 搬出到 GM 的 expertRevNums 区域 |
| 645~648 | `if constexpr(ENABLE_A8W4) DataCopyPad(cumsum to GM);` | A8W4路径下将cumsum写回GM |
| 650 | `PipeBarrier<PIPE_ALL>();` | 全管线屏障 |
| 652~655 | `AtomicAdd(sendCntFlag, 1);` | AtomicAdd 通知 AIC/AIV0 该专家的 count 已就绪 |
---

## 第668~749行: CopyGMToGMPerToken 函数实现 (6-buffer软流水)

**Phase 1 (696~703)**: 三元组 (rank, tokenIndex, topkIndex) 组装

| 行号 | 代码 | 注释 |
|------|------|------|
| 671~673 | `if(copyNum<=0) return;` | 无数据需搬运则返回 |
| 674 | `BufferNum = 6;` | buffer 数为 6 |
| 675 | `kBufEvents = {EVENT_ID0..5};` | 6 个事件 ID |
| 676 | `widthA = k / A_ELEMS_PER_BYTE;` | 量化后每 token 数据宽度(元素数) |
| 677~678 | `widthAScale = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;` | 每token scale宽度(元素数) |
| 679~680 | `copyInNum = CeilAlign(mxQuantTokenAlignBytes+mxQuantScaleAlignBytes, ALIGN_32);` | 输入token+scale拼接对齐元素数 |
| 684~689 | `设置 tokenRevGlobalTensor/scaleRevGlobalTensor/remoteRankGlobalTensor 的 GM 地址` | 设置接收端和远端rank的全局Tensor地址 |
| 692~694 | `SetFlag<MTE3_MTE2>(kBufEvents[0..5]);` | 预置6个标志(使prime/steady的WaitFlag可立即通过) |
| 695 | `PipeBarrier<PIPE_ALL>();` | 全管线屏障 |
| 697~703 | `for(i=0..copyNum) { topkIndex=validTopkIndex.GetValue(...); tokenIndex=topkIndex/topK; triple.SetValue(RANK_ID/TOKEN_ID/TOPK_INDEX); }` | 逐token组装三元组信息 |

**Phase 2 prime (706~715)**: 连发前 6 个 MTE2 读请求

| 行号 | 代码 | 注释 |
|------|------|------|
| 707 | `primeCount = min(copyNum, 6);` | prime数量 |
| 708~715 | `for(primeIdx=0..primeCount) { tokenIndex=triple.GetValue(TOKEN_ID); remoteCopyOffset=tokenIndex*copyInNum; WaitFlag<MTE3_MTE2>; DataCopy(copyTmp, remote[remoteCopyOffset], copyInNum); SetFlag<MTE2_MTE3>; }` | 连发6个跨卡MTE2读请求 |

**Phase 2 steady (716~741)**: MTE3出+MTE2入 软流水

| 行号 | 代码 | 注释 |
|------|------|------|
| 717~741 | `for(copyIdx=0..copyNum) { outIdx=copyIdx%6; WaitFlag<MTE2_MTE3>; 分离token/scale分别DataCopyPad到GM; SetFlag<MTE3_MTE2>; if(nextIdx<copyNum) { WaitFlag<MTE3_MTE2>; DataCopy下一个token; SetFlag<MTE2_MTE3>; } }` | 每轮搬出当前token并读入下一个token，6槽循环复用 |

**Phase 2 drain (742~745)**: 收尾等待所有 MTE3

| 行号 | 代码 | 注释 |
|------|------|------|
| 743~745 | `for(bufferIdx=0..6) WaitFlag<MTE3_MTE2>(kBufEvents[bufferIdx]);` | 等待所有buffer的MTE3完成 |

**Phase 3 (746~748)**: 三元组搬出到 GM

| 行号 | 代码 | 注释 |
|------|------|------|
| 747 | `SyncFuncStatic<S_MTE3, SYNC_EVENT_ID3>();` | 等待 triple SetValue 完成 |
| 748 | `DataCopy(tripleGlobalTensor_[rowDstOffset*INT32_PER_256B], tripleTensor_, copyNum*INT32_PER_256B);` | 将三元组信息搬到GM |

---

## 第758~825行: TripleInfoCalAndDispatch 函数实现

**Phase 1 (761~804)**: 分组分核三元组组装和 token dispatch

| 行号 | 代码 | 注释 |
|------|------|------|
| 762 | `L1_TILE_M_I32 = L1_TILE_M_256;` | L1分片M维度(256行) |
| 763~764 | `priorExpertCumsum = (localExpertId==0) ? 0 : cumsum[localExpertId*worldSize-1];` | 前序专家累计token数 |
| 765~766 | `topkIndexTensorSize = CeilAlign(sendTotalNum*sizeof(int32), ALIGN_32);` | topkIndexTensor大小 |
| 768~773 | `if constexpr(ENABLE_A8W4) { if(localExpertId!=0) CreateVecIndex(topkIndexTensor_); PipeBarrier<PIPE_V>; }` | A8W4路径下topkIndexTensor被覆盖需重新初始化 |
| 774 | `for(idx=blockIdx_; idx<worldSize*blockNumPerRank; idx+=blockNum)` | 逐卡分组循环 |
| 775 | `dstRankIdx = idx / blockNumPerRank;` | 目标rank索引 |
| 776 | `sendToCurExpTokenCnt = 0;` | 发送到当前专家来自该rank的token数 |
| 777~779 | `GatherMask(validTopkIndex, topkIndex, gatherMask[dstRankIdx*slot], true, sendTotalNum, {1,1,0,0}, sendToCurExpTokenCnt);` | 从topkIndex筛选属于该rank的有效索引 |
| 780 | `aivIdxInGroup = idx % blockNumPerRank;` | 组内AIV索引 |
| 781~782 | `rowStartIdxInDst = (dstRankIdx==0&&localExpertId==0) ? 0 : cumsum[localExpertId*worldSize+dstRankIdx-1];` | 该rank在当前专家中的行起始偏移 |
| 783~784 | `rowsThisCore=0; rowDstOffsetInCore=0;` | 初始化 |
| 785~804 | `if(rowStartIdxInDst<maxOutputSize) { 计算本核负责行数; if(rowsNumInCore>0) { 申请tripleTensor; CopyGMToGMPerToken(...); } }` | 分核搬运token数据 |

**Phase 2 (807~823)**: dispatch→GMM1 Flag AtomicAdd

| 行号 | 代码 | 注释 |
|------|------|------|
| 808 | `if(rowsThisCore>0)` | 本核有搬运任务 |
| 809 | `SyncFuncStatic<MTE3_S, SYNC_EVENT_ID5>();` | 等待搬出完成 |
| 811 | `rowStartLocal = rowDstOffsetInCore - priorExpertCumsum;` | 本专家内的局部偏移 |
| 812 | `rowEndLocal = rowStartLocal + rowsThisCore;` | 局部结束偏移 |
| 813~814 | `waveLo/waveHi = rowStartLocal/L1_TILE_M / (rowEndLocal-1)/L1_TILE_M;` | 行所在的wave编号范围 |
| 815 | `flagBase = gmmAddrInfo.dispatchToGmm1Flag;` | Flag基地址 |
| 816~822 | `for(w=waveLo..waveHi) { 计算交集范围; AtomicAdd(flagBase+w, 交集行数); }` | 逐wave AtomicAdd 通知AIC该wave有多少行已就绪 |

---

## 第835~892行: UpdateGroupParams 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 838 | `if(expertIdx!=0) { ... }` | 非0号专家需偏移前序专家行数 |
| 839~841 | `m/n/k = Get<M/N/K_VALUE>(state.problemShape);` | 获取当前problemShape的M/N/K |
| 842 | `state.expertBeforeCnt += m;` | 累加前序专家行数 |
| 843~859 | `各offset += m/n/k × 对应计算` | 偏移A/B/C/Flag/B2/Y2/M/GMM1/GMM2等所有buffer地址 |
| 862 | `// GMM1的count由V1写入GM，AIC/V0通过读Flag等待后读取` | 中文注释 |
| 863~880 | `if constexpr(Mode==kGmm1) { if(subBlockIdx_==0) { 自旋等Flag; 清缓存; 从GM读count; } else { 直接用sendCnt; } }` | GMM1路径：V0等待V1的SendCntCal完成；V1直接使用sendCnt |
| 881~886 | `else if constexpr(Mode==kGmm2) { 清缓存; 从GM读count; }` | GMM2路径：从GM读取count |
| 888~891 | `if(M==0) return false; else return true;` | M为0表示无token，跳过该专家 |

---

## 第900~944行: UpdateGlobalBuffer 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 904~924 | `if constexpr(Mode==kGmm1) { ... }` | GMM1路径：更新A/B/scale/epilogue的GM地址视图 |
| 906~909 | `if constexpr(ENABLE_A8W4) gmmAddrInfo.gmm1OutGlobal = gmm1MmadResPtr + offset*sizeof(bfloat16);` | A8W4路径下GMM1写MMAD结果buffer |
| 910~913 | `gmmAddrInfo.aGlobal/aScaleGlobal = dispatchRevData/Scale + offset;` | A矩阵=A量化数据 + scale |
| 915~917 | `gmmAddrInfo.bGlobal/bScaleGlobal = weight1/scale1 + offset;` | B矩阵=权重1 + scale1 |
| 919~924 | `if(AIV) epilogueOp_.UpdateGlobalAddr(vecBaseOffset);` | AIV路径下更新 Epilogue 的 GM 地址 |
| 925~938 | `else if constexpr(Mode==kGmm2) { ... }` | GMM2路径：更新A/B2/scale地址视图 |
| 927~930 | `if constexpr(ENABLE_A8W4/A4W4 || CombineQuant) gmm2OutGlobal = gmm2MmadResPtr + offset;` | A8W4/A4W4/Combine量化路径下GMM2写MMAD结果buffer |
| 931~937 | `aGlobal/aScaleGlobal = swigluQuantData/Scale + offset; bGlobal/bScaleGlobal = weight2/scale2 + offset;` | A矩阵=SwiGLU量化输出；B2=权重2 |
| 939~943 | `gmmAddrInfo.swigluToGmm2Flag/dispatchToGmm1Flag = flagPtr + offset * INT_CACHELINE/dispatchFlagSlots;` | 更新 Flag 地址 |

---

## 第950~965行: ResetGmm2CombineSyncCounters 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 952 | `if constexpr(g_coreType == AIV)` | 仅AIV执行 |
| 953~954 | `totalCounters = expertPerRank * blockAivNum * INT_CACHELINE;` | Combine同步计数器总int32数 |
| 955~956 | `TilingByCore(totalCounters, coreLen, coreOffset);` | 分核 |
| 957~958 | `gmm2CombineSyncCounterGm.SetGlobalBuffer(...);` | 设置GM地址 |
| 959~963 | `if(coreLen>0) { Duplicate(resetTensor_,0,coreLen); SyncFuncStatic<V_MTE3>; DataCopy(gm[coreOffset], resetTensor_, coreLen); }` | 清零计数器 |

---

## 第971~980行: InitCombineBuffers 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 973 | `if constexpr(CombineQuantMode!=COMBINE_NO_QUANT && g_coreType==AIV)` | 仅Combine量化路径的AIV执行 |
| 974~978 | `nAlign32/nScale/quantTokenSizeBytes/singleTokenBytes/combineUbTensorSize_ 计算` | 计算Combine所需UB buffer大小(双buffer × 每token bf16数据+量化数据) |

---

## 第988~1052行: ProcessCombine 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 991 | `nTilesPerGroup = CeilDiv(k_, L1_TILE_N);` | 每个token-group的GMM2 tile数 |
| 993~994 | `gmm2CombineSyncCounter.SetGlobalBuffer(...);` | 设置同步计数器GM地址 |
| 996~997 | `nScale/quantTokenSizeBytes 计算` | 量化参数 |
| 999 | `m_expert = Get<M_VALUE>(gmm2State.problemShape);` | 当前专家的token数 |
| 1000 | `tokenGroupsThisExpert = CeilDiv(m_expert, L1_TILE_M_256);` | token-group数 |
| 1002~1010 | `coreIdForGrouping/totalCoresForGrouping 计算; if(ENABLE_A8W4 && subBlockIdx!=1) return;` | 分核参数；A8W4下仅V1参与Combine |
| 1012~1018 | `ComputeCoreGrouping(coreId, tokenGroups, totalCores, myGroup, myIdxInGrp, myGrpSize); if(myGroup>=tokenGroups) return;` | 计算本核负责的token-group |
| 1020~1028 | `自旋等待同步计数器 == nTilesPerGroup (确保GMM2计算完成)` | 等待本token-group的GMM2计算完成 |
| 1029~1036 | `tokenStart/myTokenOffset/myTokenCount 计算` | 本核负责的token范围 |
| 1037~1051 | `if(myTokenCount>0) { 分配UB; 从GM搬入triple; PipeBarrier; CombineTokenGroup(...); }` | 执行Combine后处理 |

---

## 第1058~1120行: UnpermuteBuffInit 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 1060~1068 | `dataResBufAlign/dataResFp32BufAlign/topKWeightsBufAlign/tempBufAlign 计算` | 各buffer对齐大小 |
| 1072~1074 | `dataResTensor_ = LocalTensor<bfloat16>(VECCALC, addr=0, size=dataResBufAlign/sizeof(bf16));` | Unpermute输入/累加结果(3个token空间：2个mte2双buffer+1个累加输出) |
| 1077~1079 | `dataResFp32Tensor_ = LocalTensor<float>(VECCALC, addr=紧接, size=dataResFp32BufAlign/sizeof(float));` | fp32累加结果(dataRes × 2) |
| 1082~1084 | `topKWeightsTensor_ = LocalTensor<float>(VECCALC, addr=紧接, size=...);` | topK权重(fp32) |
| 1086~1101 | `if constexpr(CombineQuant!=NO_QUANT) { bf16ScaleTensor_/fp32ScaleTensor_ 分配; }` | Combine量化路径下的scale buffer |
| 1102~1108 | `if constexpr(IsSame<TopkWeightsType, float>) { DataCopyPad(topKWeights from GM); }` | topK权重为float时直接搬入 |
| 1109~1119 | `if constexpr(IsSame<TopkWeightsType, bfloat16>) { DataCopyPad(temp from GM); Cast(topKWeights, temp, CAST_NONE); }` | topK权重为bf16时先搬入再Cast为fp32 |

---

## 第1126~1186行: Unpermute 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 1128~1129 | `TilingByCore(m_, coreLen, coreOffset, 1);` | 分核：按token数均分 |
| 1130~1133 | `expandedX.SetGlobalBuffer(combineSendPtr); output.SetGlobalBuffer(y2GmAddr);` | 设置输入/输出GM地址 |
| 1134~1135 | `SetFlag<V_MTE2>(EVENT_ID0/1);` | 预设双buffer事件 |
| 1136 | `for(tokenIdx=coreOffset..coreLen+coreOffset)` | 逐token循环 |
| 1137 | `SyncFuncStatic<MTE3_MTE2, SYNC_EVENT_ID2>();` | 每轮开始前等待上一轮搬出完成 |
| 1138~1141 | `dataIn0Bf16/dataIn1Bf16/dataIn0Fp32/dataIn1Fp32 = dataRes偏移子区域` | 定义双buffer的输入区域 |
| 1142 | `for(expId=0..topK)` | 逐专家(topK个)循环 |
| 1143 | `expScale = topKWeights.GetValue(tokenIdx*topK+expId);` | 获取该token对第expId个专家的权重 |
| 1144~1146 | `ping-pong选择event/输入buffer` | 选择当前buffer |
| 1147~1154 | `if(CombineQuant==NO_QUANT) { WaitFlag<V_MTE2>; DataCopy(dataInBf16, expandedX[...], k_); SetFlag<MTE2_V>; WaitFlag<MTE2_V>; SetFlag/WaitFlag<S_V>; Cast(dataInFp32, dataInBf16, CAST_NONE, k_); }` | 无量化路径：直接搬入bf16数据并Cast为fp32 |
| 1155~1167 | `else { 拷入量化token+scale; DeQuantMxFp8反量化为fp32; }` | Combine量化路径：搬入量化数据后DeQuant |
| 1168 | `PipeBarrier<PIPE_V>();` | 等待Vector计算完成 |
| 1169~1176 | `if(expId==0) Muls(dataResFp32, dataInFp32, expScale, k); else { Muls(dataInFp32, dataInFp32, expScale); Add(dataResFp32, dataResFp32, dataInFp32); }` | 第一个专家直接乘scale，后续专家先乘scale再累加 |
| 1177 | `SetFlag<V_MTE2>(event);` | 标记buffer可复用 |
| 1180 | `Cast(dataRes, dataResFp32, CAST_RINT, k_);` | fp32累加结果Cast回bf16 |
| 1181 | `SyncFuncStatic<V_MTE3, SYNC_EVENT_ID3>();` | 等待Cast完成 |
| 1182 | `DataCopy(output[tokenIdx*k_], dataRes_, k_);` | 搬出到GM输出 |
| 1184~1185 | `WaitFlag<V_MTE2>(EVENT_ID0/1);` | 等待双buffer完成 |

---

## 第1192~1210行: CrossRankSyncInWorldSize 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 1194~1196 | `if constexpr(g_coreType == AIC) return;` | AIC直接返回 |
| 1197 | `syncRank = rankSyncInWorldPtr;` | 同步区域GM地址 |
| 1198~1199 | `syncCount = rankSyncInWorldPtr + 48KB + aivCoreIdx*64;` | 同步计数器地址(48K后，每核64B) |
| 1200 | `count = ReadGmByPassDCache(syncCount) + 1;` | 读取当前同步计数并加1 |
| 1201~1206 | `for(i=aivCoreIdx..worldSize step blockAivNum) { WriteGmByPassDCache(syncRemoteAddr, count); GmSignalWaitBarrier(syncCheck, count); }` | 写入所有远端卡的同步区域，然后等待所有卡写入本卡 |
| 1207 | `WriteGmByPassDCache(syncCount, count);` | 更新本卡同步计数 |
| 1208 | `PipeBarrier<PIPE_ALL>();` | 全管线屏障 |
| 1209 | `SyncAll<true>();` | 全核同步 |

---

## 第1217~1238行: GroupMatmulWithSwigluQuant 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 1220~1223 | `if constexpr(ENABLE_A8W4) GroupMatmulSwigluQuantA8W4<...>(epilogue, params, problemShape, gmmAddr, ...);` | A8W4路径：调用A8W4 prologue的GMM1+SwiGLU+量化 |
| 1225~1230 | `else if(NZ format) GroupMatmulSwigluQuant<..., true>(...); // isWeightNZ=true` | NZ格式(NZ weight layout)：调用generic GMM，isWeightNZ=true |
| 1231~1235 | `else GroupMatmulSwigluQuant<..., false>(...); // ND format` | ND格式：调用generic GMM，isWeightNZ=false |

---

## 第1246~1271行: GroupMatmulWithCombine 函数实现

| 行号 | 代码 | 注释 |
|------|------|------|
| 1249~1253 | `if constexpr(ENABLE_A8W4/A4W4) GroupMatmul2CombineA8W4<...>(...);` | A8W4/A4W4路径：调用A8W4 prologue的GMM2+Combine |
| 1256~1266 | `else { if(NZ) GroupMatmul2<..., true>(...); else GroupMatmul2<..., false>(...); }` | A8W8_NZ用ZN布局，其余用ND布局 |
| 1268~1270 | `if constexpr(CombineQuant!=NO_QUANT && AIV) ProcessCombine(...);` | Combine量化路径下执行Combine后处理 |

---

## 第1274~1373行: Process 函数实现 (主执行流程)

| 行号 | 代码 | 注释 |
|------|------|------|
| 1277~1278 | `oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE>(); SetCtrlSpr<OVERFLOW_MODE>(0);` | 保存并设置溢出模式为0(Saturation模式) |
| 1279 | `SendAndQuantBuffInit();` | 初始化SendMaskCal/QuantProcessInRank的UB buffer |
| 1280 | `SendMaskCal();` | 源卡按所有全局专家算mask并推送到目标专家卡 |
| 1281 | `ResetFlagList();` | 清理workspace上的Flag位 |
| 1283 | `QuantProcessInRank();` | 对本卡token的MX量化 |
| 1284~1286 | `if(AIV) PipeBarrier<PIPE_ALL>();` | AIV管线屏障 |
| 1287 | `SyncAll<false>();` | AIC等待Flag reset完成 |
| 1288 | `CrossRankSyncInWorldSize();` | 全卡同步(确保所有卡的mask和量化数据就绪) |
| 1290~1299 | `// Dispatch & GMM1 & SwiGLU阶段` | 第2阶段开始 |
| 1291 | `DispatchBuffInit();` | 初始化dispatch阶段UB buffer |
| 1292~1299 | `GMMAddrInfo/ExpertLoopState 初始化` | 初始化GMM地址信息和专家循环状态 |
| 1305~1306 | `curSendCnt/nextSendCnt 初始化` | dispatch预取计数转发机制 |
| 1309~1317 | `预调度expert 0: if(AIV && subBlock==1) SendCntCal(0,nextSendCnt); UpdateGroupParams<kGmm1>; UpdateGlobalBuffer<kGmm1>; TripleInfoCalAndDispatch;` | V1预调度第0个专家 |
| 1319 | `for(localExpertId=0..expertPerRank)` | 逐专家循环 |
| 1320 | `curSendCnt = nextSendCnt;` | 转发：dispatch(e) → GMM1(e) |
| 1323~1331 | `if(AIV && subBlock==1 && localExpertId+1<expertPerRank) { SendCntCal(e+1,nextSendCnt); UpdateGroupParams<kGmm1>(e+1); UpdateGlobalBuffer<kGmm1>(e+1); TripleInfoCalAndDispatch(e+1); }` | V1预调度下一个专家(与当前GMM1并发) |
| 1334~1338 | `if(!UpdateGroupParams<kGmm1>(gmm1State, localExpertId, curSendCnt)) continue; UpdateGlobalBuffer<kGmm1>; GroupMatmulWithSwigluQuant;` | GMM1 consumer消费当前专家 |
| 1340 | `EndSync(vecSetSyncCom_);` | GMM1同步结束 |
| 1341~1345 | `if(AIV && subBlock==1) ExpertTokenNumCopyOut();` | V1输出各专家token数 |
| 1347~1359 | `// GMM2 & Combine阶段` | 第3阶段开始 |
| 1348 | `vecSetSyncCom_ = 0;` | 重置同步计数器 |
| 1349~1351 | `GMMAddrInfo/ExpertLoopState 初始化; InitCombineBuffers();` | 初始化GMM2地址和Combine buffer |
| 1353 | `for(expertIdx=0..expertPerRank)` | 逐专家循环 |
| 1354~1358 | `if(!UpdateGroupParams<kGmm2>) continue; UpdateGlobalBuffer<kGmm2>; GroupMatmulWithCombine;` | GMM2+Combine处理 |
| 1360~1362 | `if constexpr(CombineQuant==NO_QUANT) EndGMM2Sync;` | 无量化Combine路径的GMM2同步结束 |
| 1363~1364 | `PipeBarrier<PIPE_ALL>(); SyncAll<true>();` | 全核同步 |
| 1366~1371 | `// Unpermute阶段` | 第4阶段开始 |
| 1367 | `if constexpr(AIV)` | 仅AIV执行Unpermute |
| 1368 | `UnpermuteBuffInit();` | 初始化Unpermute的UB buffer |
| 1369 | `CrossRankSyncInWorldSize();` | 全卡软同步(确认Combine send完成) |
| 1370 | `Unpermute();` | 反排列：将各专家还回的token按topk权重累加回原位置 |
| 1372 | `SetCtrlSpr<OVERFLOW_MODE>(oriOverflowMode);` | 恢复原始溢出模式 |

---

## 第1375~1376行: 命名空间结束 & 宏保护结束

| 行号 | 代码 | 注释 |
|------|------|------|
| 1375 | `}   // namespace MegaMoeImpl` | MegaMoeImpl 命名空间结束 |
| 1376 | `#endif` | MEGA_MOE_H 宏保护结束(对应第16行的 #ifndef) |

---

## 总结

本文件实现了一个完整的 MoE (Mixture-of-Experts) 分布式推理 kernel，核心流程为 4 个阶段：

1. **SendMaskCal + QuantProcessInRank**: 源卡计算 mask 并推送，同时对本卡 token 执行 MX 量化
2. **Dispatch + GMM1 + SwiGLU + SwigluQuant**: 接收远端 token (CopyGMToGMPerToken 6-buffer软流水)，执行分组矩阵乘1 + SwiGLU激活 + 量化
3. **GMM2 + Combine**: 执行分组矩阵乘2 + Combine(加权合并各专家输出)
4. **Unpermute**: 反排列，将 token 按原始位置和 topk 权重累加输出

关键技术特点：
- **A8W4/A4W4/A8W8 多量化路径**: 编译期条件分发，支持 fp8/fp4 激活和权重的多种组合
- **6-buffer 软流水**: CopyGMToGMPerToken 使用 6 个 buffer + EVENT_ID0..5 实现高效跨卡数据搬运
- **Ping-pong 双 buffer**: SendMaskCal/QuantProcessInRank/Unpermute 等均使用双 buffer 软流水
- **AIC/AIV 协作**: V1(subBlockIdx=1) 负责 SendCntCal/dispatch预取，V0/AIC 负责 GMM 计算
- **AtomicAdd Flag 同步**: dispatch→GMM1 使用 wave-grain AtomicAdd 通知 AIC 数据就绪
- **CrossRankSyncInWorldSize**: 全卡软同步，确保所有卡的数据交换完成
- **cumsum GM 持久化**: A8W4 路径下 SwigluQuant 覆盖 V1 UB，cumsum 数据在 GM 上持久备份

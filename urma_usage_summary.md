# URMA 使用总结文档

## 概述

在 commit 范围 `7a1240ee286b239d989ad4afbeab3a08dc798475` 到 `e12369e84106ff5ea5791545947486b87f27b1a6` 中，新增了对 URMA (Unified Remote Memory Access) 通信协议的支持，主要围绕 bandwidth_test 算子进行 URMA 通信带宽测试能力的扩展。以下从架构层面、Host 端、Kernel 端三个维度进行总结。

---

## 1. 架构层面的变更

### 1.1 通信协议体系扩展

原有系统仅支持 `COMM_PROTOCOL_UB_MEM`（基于 UB 内存的通信协议），此次变更新增了 `COMM_PROTOCOL_UBC_CTP` 协议，用于支持 URMA 和 ROCE 通信方式。

**关键文件**：
- `mc2/common/op_api/mc2_context.h`
- `mc2/common/op_api/mc2_context.cpp`

**主要变更**：

| 变更点 | 说明 |
|--------|------|
| `Mc2MoeContext` 新增 `hcommHandle_[HCCL_MAX_RANK_SIZE]` | 新增通信句柄数组，用于存储 URMA/ROCE 的 Hcomm 通道实体地址，注释标注"支持ROCE或者URMA" |
| `GetMc2ContextTensor` 增加 `CommProtocol Protocol` 参数 | 默认值为 `COMM_PROTOCOL_UB_MEM`，允许外部指定通信协议，当 `commAlg="urma"` 时传入 `COMM_PROTOCOL_UBC_CTP` |
| `GetHcclCommChannel` 签名变更 | 参数从 `uint32_t srcRankId` 变为 `Mc2MoeContext *mc2ContextStruct`，支持在 URMA 模式下将通道句柄写入 `hcommHandle_` |
| `COMM_PROTOCOL_UBC_CTP` 通道获取逻辑 | 当协议为 `COMM_PROTOCOL_UBC_CTP` 时，`HcclChannelAcquire` 将通道句柄写入 `hcommHandle_` 而非 `channels`，并获取 HCCL Buffer 地址 |
| `GetHcclCommResource` 分支处理 | UB_MEM 模式下继续使用原有 epHcclBuffer_ 逻辑；UBC_CTP 模式下走不同的通道获取流程 |
| `CheckLinks` / `CheckProtocolSupport` / `GetCommProtocol` | 均新增 `protocol` 参数，支持检测任意协议而非仅 UB_MEM |
| `Mc2Context::GetInstance()` | 改为单例模式，避免重复创建实例 |
| `HCCL_CHANNEL_SUPPORT_VERSION` | 版本号从 `89999700` 更新为 `90000000` |
| 新增 `#include "hcomm/hcomm_res.h"` | 引入 Hcomm 资源头文件，用于 URMA 通道操作 |

### 1.2 UB_MEM 模式下的修正

- 移除了 `EP_RANK_OFFSET_STEP = 8192` 常量，不再对 epHcclBuffer_ 做偏移计算
- `epHcclBuffer_[i]` 直接使用 `reinterpret_cast<uint64_t>(tempBuffer)` 而非加上 `i * EP_RANK_OFFSET_STEP`

---

## 2. Host 端变更

### 2.1 Tiling Key 体系

**文件**：`mc2/tools/bandwidth_test/op_kernel/bandwidth_test_tiling_key.h`

新增通信模式维度到模板 tiling key：

```cpp
#define TILINGKEY_TPL_MTE  0   // 传统 MTE 通信模式
#define TILINGKEY_TPL_URAMA 1  // URMA 通信模式
```

Tiling Key 计算公式变为：`GET_TPL_TILING_KEY(commModeByte, archTag)`，其中 `commModeByte` 根据属性 `comm_alg` 的值选择：
- `"mte"` → `TILINGKEY_TPL_MTE`
- `"urma"` → `TILINGKEY_TPL_URAMA`

### 2.2 Tiling 参数与校验

**文件**：`mc2/tools/bandwidth_test/op_host/op_tiling/bandwidth_test_tiling.cpp`

| 变更点 | 说明 |
|--------|------|
| 新增 `ATTR_HCCL_BUFFER_INDEX = 4U` | HCCL 缓冲区大小属性索引 |
| 新增 `ATTR_COMM_MODE_INDEX = 5U` | 通信模式属性索引 |
| `commMode` 校验 | 要求值为 `"mte"` 或 `"urma"`，否则报错 |
| `CalTilingKey` | 接收 `commMode` 参数，根据值计算不同的 tiling key |
| Workspace 大小计算 | 改为 `SYSTEM_NEED_WORKSPACE + dataSize + 1GB`，其中 `dataSize = worldSize * hcclBufferSize`，为 URMA 模式预留充足的全局内存空间 |

### 2.3 ACLNN API 层

**文件**：`mc2/tools/bandwidth_test/op_api/aclnn_bandwidth_test.cpp`

核心逻辑：在 `aclnnBandwidthTestGetWorkspaceSize` 中根据 `commAlg` 参数选择通信协议：

```cpp
if (commAlg != nullptr && std::strcmp(commAlg, "urma") == 0) {
    // URMA 模式：使用 COMM_PROTOCOL_UBC_CTP
    aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(
        group, opName, hcclBuffSize, mc2Context, CommProtocol::COMM_PROTOCOL_UBC_CTP);
} else {
    // 默认模式：使用 COMM_PROTOCOL_UB_MEM
    aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(
        group, opName, hcclBuffSize, mc2Context);
}
```

API 签名变更：`group` 和 `commAlg` 参数从 `const char*` 改为 `char*`（允许修改）。

---

## 3. Kernel 端变更

### 3.1 Kernel 入口分发

**文件**：`mc2/tools/bandwidth_test/op_kernel/bandwidth_test.cpp`

模板参数从 `template<uint8_t ArchTag>` 扩展为 `template<uint8_t CommMode, uint8_t ArchTag>`：

```cpp
if constexpr (CommMode == TILINGKEY_TPL_MTE) {
    BandwidthTest<DTYPE_X, DTYPE_Y> op;       // 传统 MTE 内核
    op.Init(...);
    op.Process();
} else if constexpr (CommMode == TILINGKEY_TPL_URAMA) {
    BandwidthUrmaTest<DTYPE_X, DTYPE_Y> op;   // URMA 内核
    op.Init(...);
    op.Process();
}
```

### 3.2 URMA Kernel 实现

新增三个 URMA 测试 kernel 文件，均实现 `BandwidthUrmaTest<XType, YType>` 模板类：

#### 3.2.1 `urma_all_data_send_test.h`（全量数据发送测试）

- **特点**：最基础的 URMA 实现，所有 AIV 核均参与计算和通信
- **流程**（mode=0）：
  1. `SendToMoeExpert` — 根据 expert_id 将 token 写入本地 workspace 窗口
  2. `SetStatus` — 统计发送到各 rank 的 token 数量，通过 `hcomm_.WriteNbi` 发送数据和状态到远端
  3. `WaitDispatch` — 轮询 URMA 状态区，等待所有 rank 完成
  4. `LocalWindowCopy` — 从本地窗口读取数据拷贝到输出 tensor

#### 3.2.2 `urma_mx_send.h` / `bandWidth_Urma_test.h`（MX 混合发送测试）

- **特点**：区分通信核和计算核，最多 8 个 AIV 核用于通信（`COMM_MAX_USE_AIV_NUM = 8`），其余用于计算
- **新增字段**：
  - `aivNumUsedComm_` — 通信使用的 AIV 核数
  - `aivNumUsedCompute_` — 计算使用的 AIV 核数
  - `sendHandleWorkspace_` — 发送状态句柄 workspace
  - `sendFlagHandleGMTensor_` — 发送状态标记
- **新增方法**：`CommProcess()` — 专门的通信处理流程
- **workspace 划分**：`localUrmaWorkspace_ = sendHandleWorkspace_ + axisBS_ * epWorldSize_ * sizeof(uint32_t) + ONE_MB`

#### 3.2.3 `urma_single_token_send_test.h`（单 token 发送测试）

- **特点**：与 `urma_all_data_send_test.h` 基本相同，但 `ProcessToken` 不带 `curExpertCnt` 参数，用于单 token 级别的细粒度测试

### 3.3 URMA 通信核心机制

所有 URMA kernel 共享以下核心通信机制：

#### 地址获取方法

```cpp
// 获取远端 URMA 数据区地址（跳过 STATE_SIZE）
GM_ADDR GetUrmaAddrByRankId(rankId) {
    if (rankId == epRankId_) {
        return urmaCommHandle_[0]->localBufferAddr->bufferInfo.rma.addr + STATE_SIZE;
    }
    return urmaCommHandle_[currRankId]->remoteBufferAddr->bufferInfo.rma.addr + STATE_SIZE;
}

// 获取远端 URMA 状态区地址
GM_ADDR GetUrmaStateAddrByRankId(rankId) {
    if (rankId == epRankId_) {
        return urmaCommHandle_[0]->localBufferAddr->bufferInfo.rma.addr;
    }
    return urmaCommHandle_[currRankId]->remoteBufferAddr->bufferInfo.rma.addr;
}

// 获取本地 workspace 窗口数据区地址
GM_ADDR GetWindAddrByRankId(rankId) {
    return localUrmaWorkspace_ + totalWinSizeEp_ * rankId + STATE_SIZE;
}

// 获取本地 workspace 窗口状态区地址
GM_ADDR GetWindStateAddrByRankId(rankId) {
    return localUrmaWorkspace_ + totalWinSizeEp_ * rankId;
}
```

#### Hcomm API 使用

```cpp
// 初始化 Hcomm 通信上下文
AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

// 非阻塞写入（发送数据到远端 URMA 地址）
hcomm_.WriteNbi(context_->hcommHandle_[currRankId], remoteAddr, srcRankGM, sendLength);

// 等待写入完成（Drain）
hcomm_.Drain(context_->hcommHandle_[currRankId]);
```

#### URMA 通道初始化

```cpp
// 从 Mc2MoeContext 中获取 URMA 通道实体
for (uint32_t i = 0; i < epWorldSize_ - 1; i++) {
    urmaCommHandle_[i] = reinterpret_cast<__gm__ ChannelEntity*>(context_->hcommHandle_[i]);
}
```

#### 状态同步机制

- 状态区结构：每个 rank 占 32 字节（8 个 int32），包含 flag（1.0f = 0x3F800000）和 token 计数
- 发送方：将状态写入本地 workspace，再通过 `hcomm_.WriteNbi` 发送到远端 URMA 状态区
- 接收方：轮询本地 URMA 状态区，通过 `ReduceSum` 计算所有 rank 的 flag 和，等待其等于目标值

---

## 4. 数据流向

### URMA 模式数据流

```
输入 x (GM) → UB (xTmpTensor_) → 本地 workspace 窗口 (GetWindAddrByRankId)
                                         ↓ hcomm_.WriteNbi
                                    远端 URMA 地址 (GetUrmaAddrByRankId)
                                         ↓ 状态同步完成
                                    本地 URMA 状态区 (轮询确认)
                                         ↓
本地 URMA 数据区 → UB → 输出 y (GM)
```

### MTE 模式数据流（对比）

```
输入 x (GM) → UB → HCCL 窗口 (epHcclBuffer_[rankId])
                    ↓ 状态同步
              HCCL 窗口 → UB → 输出 y (GM)
```

---

## 5. 关键差异对比

| 维度 | MTE 模式 (UB_MEM) | URMA 模式 (UBC_CTP) |
|------|-------------------|---------------------|
| 通信协议 | `COMM_PROTOCOL_UB_MEM` | `COMM_PROTOCOL_UBC_CTP` |
| 通信引擎 | AIV 直接窗口写入 | `Hcomm<COMM_PROTOCOL_UBC_CTP>` API |
| 数据传输方式 | 写入 HCCL 共享窗口，对端直接读取 | `hcomm_.WriteNbi` 主动推送到远端 URMA 地址 |
| 通道句柄存储 | `epHcclBuffer_[rankId]` | `hcommHandle_[rankId]`（ChannelEntity 指针） |
| 缓冲区获取 | `HcclChannelGetHcclBuffer` 获取每个通道的 buffer | `HcclGetHcclBuffer` 仅获取本地 buffer |
| Workspace 用途 | 仅系统 workspace | 额外用作本地 URMA 窗口（1GB+数据量） |
| 状态同步 | 写入窗口状态区，对端轮询窗口 | 写入 URMA 状态区 + `hcomm_.WriteNbi` 推送状态到远端 |
| 内核类 | `BandwidthTest` | `BandwidthUrmaTest` |
| Tiling Key | `TILINGKEY_TPL_MTE (0)` + `TILINGKEY_TPL_A5 (2)` | `TILINGKEY_TPL_URAMA (1)` + `TILINGKEY_TPL_A5 (2)` |

---

## 6. 文件清单

| 文件路径 | 变更类型 | 说明 |
|----------|---------|------|
| `mc2/common/op_kernel/mc2_moe_context.h` | 修改 | 新增 `hcommHandle_` 字段 |
| `mc2/common/op_api/mc2_context.h` | 修改 | 新增 `GetInstance()`、协议参数、`hcomm_res.h` 引用 |
| `mc2/common/op_api/mc2_context.cpp` | 修改 | URMA 通道获取、协议检测、单例模式等 |
| `mc2/tools/bandwidth_test/op_kernel/bandWidth_Urma_test.h` | 新增 | URMA MX 混合发送内核（727行） |
| `mc2/tools/bandwidth_test/op_kernel/urma_all_data_send_test.h` | 新增 | URMA 全量数据发送内核（660行） |
| `mc2/tools/bandwidth_test/op_kernel/urma_mx_send.h` | 新增 | URMA MX 发送内核（与 bandWidth_Urma_test.h 相同） |
| `mc2/tools/bandwidth_test/op_kernel/urma_single_token_send_test.h` | 新增 | URMA 单 token 发送内核（655行） |
| `mc2/tools/bandwidth_test/op_kernel/bandwidth_test.cpp` | 修改 | 增加 CommMode 模板参数和 URMA 分支 |
| `mc2/tools/bandwidth_test/op_kernel/bandwidth_test.h` | 修改 | 清理注释 |
| `mc2/tools/bandwidth_test/op_kernel/bandwidth_test_tiling_key.h` | 修改 | 新增 TILINGKEY_TPL_MTE/URAMA 和 CommMode 维度 |
| `mc2/tools/bandwidth_test/op_kernel/bandwidth_test_tiling.h` | 修改 | 清理注释 |
| `mc2/tools/bandwidth_test/op_api/aclnn_bandwidth_test.cpp` | 修改 | URMA 协议选择逻辑 |
| `mc2/tools/bandwidth_test/op_api/aclnn_bandwidth_test.h` | 修改 | API 签名变更 |
| `mc2/tools/bandwidth_test/op_host/op_tiling/bandwidth_test_tiling.cpp` | 修改 | Tiling 增加 commMode、workspace 计算 |
| `mc2/tools/bandwidth_test/op_host/CMakeLists.txt` | 修改 | 移除 ascend950 特殊编译选项 |
# PR #8120 检视报告

## 概述

本 PR 对 `moe_distribute_dispatch_v2` 算子的 cumsum（累加求和）阶段进行了两项核心改动：

1. **存储位置迁移**：将 cumsum 的 CAL 区和 FLAG 区从 HCCL 通信窗口状态空间（`selfRankWinInGMTensor_`）迁移到独立的 workspace 区域（`cumsumWsGMTensor_`），并使用 `GetUserWorkspace` 获取用户 workspace 基址。
2. **A5 核数分配优化**：将 A5（arch35）的 `CUMSUM_MAX_CORE_NUM` 从 8 提升到 32，并将 `aivUsedCumSum_` 的计算方式从固定公式改为基于 AllToAll 与 CumSum 两阶段工作量比例的动态分配。

涉及文件：
- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp`
- `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h`

---

## 问题列表

### [信息] arch35 host_kfc 实现未同步更新

**文件**: `op_kernel/arch35/moe_distribute_dispatch_v2_host_kfc.h:50-51, 1096, 1145, 1178, 1216, 1973`

`moe_distribute_dispatch_v2_host_kfc.h` 中仍使用旧常量 `CUMSUM_CAL_OFFSET`（868KB）和 `CUMSUM_FLAG_OFFSET`（876KB），且 cumsum 数据仍读写 `selfRankWinInGMTensor_`（通信窗口状态空间）。

```cpp
// host_kfc.h — 旧逻辑，未更新
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL;
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL;
DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], ...);
```

经确认，`host_kfc.h` 是完全独立的实现，定义了自己的 `MoeDistributeDispatchV2HostKfc` 类，未被任何 `.cpp` 文件 include 或实例化，与 full_mesh 无代码依赖关系。host_kfc 有自己的常量定义和 cumsum 逻辑，按自身路径独立运行，不受本 PR 改动影响。

**建议**：后续若启用 host_kfc 路径，需同步迁移到 workspace 方案。当前无需处理。

---

### [低] 非直连模式 `moe_distribute_dispatch_v2.h` 未使用 `GetUserWorkspace`

**文件**: `op_kernel/moe_distribute_dispatch_v2.h:402`

非 fullmesh 的 `MoeDistributeDispatchV2` 实现仍使用原始 workspace 指针：

```cpp
recvCntWorkspaceGM_ = workspaceGM;  // 非 fullmesh，未调用 GetUserWorkspace
```

而 fullmesh 实现已改为：

```cpp
recvCntWorkspaceGM_ = AscendC::GetUserWorkspace(workspaceGM);  // fullmesh
```

在 A5 上，非 fullmesh 模式（`CommMode == TILINGKEY_TPL_MTE && FullMesh != TILINGKEY_ENABLE_FULLMESH`）会被调用。经分析，A5 非	fullmesh 路径使用 `GetCumSumA5()` 从通信窗口读取 cumsum 数据（`windowInstatusFp32Tensor_`），不依赖 workspace 的 cumsum 区域，因此当前不会导致功能错误。但 `SyncCntOnCore` 函数（line 1040）仍向 `recvCntWorkspaceGM_` 原始偏移写入数据，如果 workspace 起始处为系统 workspace，存在潜在冲突风险。

**建议**：确认非 fullmesh 路径在 A5 上 `SyncCntOnCore` 是否被调用。如果是，应同步使用 `GetUserWorkspace`。

---

### [低] 关键常量在 tiling 与 kernel 间重复定义

**文件**: 
- `op_tiling/moe_distribute_dispatch_v2_tiling.cpp:83-86`
- `op_kernel/moe_distribute_dispatch_v2_full_mesh.h:64`

以下常量在 tiling 和 kernel 中各自独立定义，值需手工保持一致：

| 常量 | tiling 值 | kernel 值 |
|------|----------|-----------|
| `CUMSUM_WS_FLAG_OFFSET` | `32UL * 1024UL` | `32UL * 1024UL` |
| `CUMSUM_MAX_CORE_NUM` (A5) | `CUMSUM_MAX_CORE_NUM_A5 = 32U` | `CUMSUM_MAX_CORE_NUM = 32U` |
| `CUMSUM_MAX_CORE_NUM` (非A5) | `CUMSUM_MAX_CORE_NUM_NON_A5 = 16U` | `CUMSUM_MAX_CORE_NUM = 16U` |
| `UB_ALIGN_VAL` / `UB_ALIGN` | `32U` | `32`（来自外部头文件） |

tiling 使用 `GetSocVersion() == "Ascend950"` 判断 A5，kernel 使用 `__NPU_ARCH__ == 3510` 判断 A5。两者映射关系需保持一致。

若任一方修改值而另一方未同步，将导致 workspace 大小与实际访问范围不匹配，引发内存越界。

**建议**：考虑将共享常量提取到公共头文件（如 `moe_distribute_v2_constant.h`），消除重复定义。

---

### [信息] `CUMSUM_WS_FLAG_OFFSET` 命名与语义不完全匹配

**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h:63-64`

```cpp
// 按32核预算的 CAL 区大小（32*32*32=32KB）
constexpr uint64_t CUMSUM_WS_FLAG_OFFSET = 32UL * 1024UL;
```

该常量实际表示 cumsum workspace 中 CAL 区的大小（即 FLAG 区相对于 CAL 区起始的偏移），但名称中只包含 `FLAG_OFFSET`，容易误解为 FLAG 区本身的偏移。tiling 侧注释为"cumsum 区在 recvCnt workspace 后，由 CAL 区 + FLAG 区组成"，但常量名未体现这一语义。

**建议**：可考虑重命名为 `CUMSUM_CAL_AREA_SIZE` 或 `CUMSUM_FLAG_AREA_OFFSET`，使语义更清晰。非阻塞性问题。

---

### [信息] `aivUsedCumSum_` 新公式的整数除法截断行为

**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h:428-429`

```cpp
aivUsedCumSum_ = aivNum_ - ((axisBS_ * axisK_ * DATA_TO_CNT_TIME_RATIO * aivNum_) /
                    (axisBS_ * axisK_ * DATA_TO_CNT_TIME_RATIO  + totalExpertNum_));
```

该公式等价于 `aivNum_ * totalExpertNum_ / (axisBS_ * axisK_ * DATA_TO_CNT_TIME_RATIO + totalExpertNum_)`，使用整数除法会向下取整。后续的 clamp 逻辑（line 433-436）确保结果在 `[1, min(aivNum_/2, CUMSUM_MAX_CORE_NUM, rscvStatusNum_)]` 范围内，因此截断不会导致越界。`DATA_TO_CNT_TIME_RATIO = 4` 是经验常数，可能需要根据实际性能数据调优。

无溢出风险：最大乘积 `512 * 12 * 4 * 32 = 786432`，远在 `uint32_t` 范围内。

---

## Workspace 布局一致性验证

### Tiling 分配

```
workSpaces[0] = SYSTEM_NEED_WORKSPACE                           // 16MB 系统区
             + WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum       // recvCnt 区
             + CUMSUM_WS_FLAG_OFFSET                             // cumsum CAL 区 (32KB)
             + aivNum * cumsumMaxCoreNum * UB_ALIGN_VAL          // cumsum FLAG 区
```

### Kernel 访问

| 区域 | 最大访问偏移 | 分配大小 | 结论 |
|------|------------|---------|------|
| recvCnt | `Ceil(aivNum * rscvStatusNum * 4, 512) * 512` | `512 * aivNum * aivNum` | 安全（rscvStatusNum <= moeExpertNum <= 1024 < 128 * aivNum） |
| CAL 区 | `(aivUsedCumSum-1) * aivUsedCumSum * UB_ALIGN + aivUsedCumSum * UB_ALIGN` = `aivUsedCumSum^2 * 32` | `CUMSUM_WS_FLAG_OFFSET = 32768` | 安全（aivUsedCumSum <= 16 on A5-32core, 16^2*32=8192 < 32768） |
| FLAG 区 | `CUMSUM_WS_FLAG_OFFSET + (aivNum-1) * aivUsedCumSum * UB_ALIGN` | `aivNum * cumsumMaxCoreNum * UB_ALIGN_VAL` | 安全（aivUsedCumSum <= cumsumMaxCoreNum） |

**结论**：workspace 大小分配与 kernel 实际访问范围一致，无越界风险。

---

## 总结

本 PR 的核心改动（cumsum 存储迁移至 workspace + A5 核数动态分配）设计合理，workspace 大小计算与 kernel 访问范围一致，无内存越界风险。`GetUserWorkspace` 的引入修正了 fullmesh 路径的 workspace 基址问题。

主要关注点：
- **host_kfc.h 未同步**（信息）：独立实现，当前不受影响，后续启用时需同步
- **非 fullmesh 路径未使用 `GetUserWorkspace`**（低）：当前 A5 路径不受影响，但存在潜在风险
- **常量重复定义**（低）：建议提取到公共头文件

整体评估：**通过（有建议）**。

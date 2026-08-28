# AlltoAllMatmulV2 异常 Dump 设计方案

## 需求

1. dump tilingdata 数据
2. dump urma 路径"workspace"数据（peermem 窗口内容）
3. 记录 runtimeInfo（类似 v1 `allto_all_mx_quant_matmul_arch35.h` 的 `runtimeInfo_`）

## 调研结论

### urma 不使用 workspaceGM（数据通路）

- kernel 调用 `impl.Init(hcommCtx, x1, x1_scale, x2, x2_scale, y, bias, &tilingData)` 时未传 `workspaceGM`
- `AllToAllMxQuantMatmulUrmaImpl::Init` 签名无 workspace 参数
- impl 数据来源是 URMA peermem 窗口：`udmaCtx_->commBufferAddrs[baseParams_.rankId]`（从 context arg 的 CommContext 中取）
- 通信 scratch 用片上 UB（512B），不是 workspaceGM
- workspaceGM（arg 8）由框架分配，kernel 收到的指针已跳过 LIB_API 段

### workspace 的两种独立用途

| 用途 | 区域 | 使用者 |
|------|------|--------|
| 库 API workspace | LIB_API 段（如 16MB） | AscendC 库函数内部，kernel 不直接访问 |
| "黑匣子"记录器 | RUNTIME_INFO 段（64KB = 128 核 × 512B） | `Mc2RuntimeInfo` 通过 `DataCacheCleanAndInvalid` 写入执行进度 |

urma 数据通路（peermem + UB）与 workspace 完全独立。RUNTIME_INFO 段仅用于记录执行进度，不参与数据通信。

### workspace 指针的两面性（关键）

框架分配的 workspace 布局：
```
[==== LIB_API (如 16MB) ====][== RUNTIME_INFO (64KB) ==]
^                            ^
原始 workspaceGM              kernel 的 workspaceGM（已跳过 LIB_API）
（Mc2DumpTilingData 从 args 读到这个）  （kernel 函数参数收到这个）
```

- **kernel 侧**：`workspaceGM` = 原始 + libApiSize（已跳过 LIB_API），`Mc2RuntimeInfo::Init` 用此指针
- **host 侧**：`Mc2DumpTilingData` 从 args 读到的是**原始 workspaceGM**（含 LIB_API），比 kernel 的小 libApiSize

因此 workspaceLayout **必须包含 LIB_API 段**，让：
- `Mc2RuntimeInfo::Init`：`userOffset = RUNTIME_INFO.offset - LIB_API.size = libApiSize - libApiSize = 0` → `kernelWorkspaceGM + 0` ✓
- `Mc2DumpTilingData`：`原始workspaceGM + RUNTIME_INFO.offset = 原始workspaceGM + libApiSize` → 正确指向 RUNTIME_INFO ✓

### v2 tiling 结构（改动前）

```cpp
struct allToAllMatmulTilingData {
    CommTilingData commTilingData;
    CommTilingData scaleCommTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    uint32_t localMatmul{0};
    // 无 quantMode、无 workspaceLayout
};
```

### kernel arg 索引（0-based）

```
context=0  x1=1  x2=2  bias=3  x1_scale=4  x2_scale=5  y=6  all2all_out=7  workspaceGM=8  tilingGM=9
```

### Mc2RuntimeInfo 机制

- `Init(workspaceGM, wsLayout, coreNum)`：若 `wsLayout==nullptr` 则 no-op；否则扫描段表找 `WS_SEG_RUNTIME_INFO` 段，计算 `userOffset = seg.offset - libApiSize`，写入 workspaceGM
- kernel 的 workspaceGM 已跳过 LIB_API 段（框架管理），段表 offset 是绝对值（含 LIB_API），故需减去 libApiSize
- 通过 `DataCacheCleanAndInvalid` 强制刷盘，崩溃时仍可见
- 异常回调时 `Mc2DumpTilingData` 按段表从原始 workspaceGM dump 出来

### hcomm impl 的 runtimeInfo 模式（参照实现）

1. 成员：`Mc2RuntimeInfo runtimeInfo_` + `uint8_t execTurn_`
2. Init：`runtimeInfo_.Init(workspaceGM, &tilingData_->workspaceLayout, usedCoreNum)`
3. 通信前：`execTurn_++; WriteGmByPassDCache(execTurn_, POS_COMM_BEFORE, RT_PHASE_COMM_PREPARE)`
4. 每次 commit 后：`IncCommitCount()`
5. 每次 wait 后：`IncWaitCount()`
6. 计算前：`execTurn_++; WriteGmByPassDCache(execTurn_, POS_COMP_*_BEFORE, ...)`
7. 结束：`WritePhaseByPassDCache(RT_PHASE_COMM_FINALIZE)`

## 设计方案

### 关键设计决策

1. 给 `allToAllMatmulTilingData` 末尾新增 `workspaceLayout` 字段
2. workspaceLayout **必须包含 LIB_API 段 + RUNTIME_INFO 段**（2 段），offset 为绝对值（含 LIB_API）：
   - 段 0: `{offset=0, size=libApiSize, type=WS_SEG_LIB_API}`
   - 段 1: `{offset=libApiSize, size=64KB, type=WS_SEG_RUNTIME_INFO}`
   - 原因：`Mc2DumpTilingData` 从 args 读到原始 workspaceGM（含 LIB_API），需要 `原始workspaceGM + libApiSize` 才能定位 RUNTIME_INFO；`Mc2RuntimeInfo::Init` 用 kernel 的 workspaceGM（已跳过 LIB_API），通过 `userOffset = libApiSize - libApiSize = 0` 正确定位
3. workspaceSize = libApiSize + 64KB（RUNTIME_INFO_TOTAL_SIZE）
4. kernel 侧初始化 `Mc2RuntimeInfo` 并通过 `SetRuntimeInfo` 注入 urma impl
5. urma impl 在 RunAllToAll/RunMatmul/RunLocalMatmul 中加打点
6. 异常回调调用 `Mc2DumpTilingData`（dump tiling + workspace 含 RUNTIME_INFO 段）+ `Mc2DumpUrmaContext`（dump CommContext + peermem 窗口）
7. `Mc2DumpUrmaContext` 产出的 CommContext 文件命名为 `comm_context_*`（不叫 `runtime_info_*`，避免与 `Mc2RuntimeInfo` 混淆）

### 改动清单

#### 文件 1: all_to_all_matmul_tiling_data.h
- 给 `allToAllMatmulTilingData` 末尾新增 `Mc2Tiling::WorkspaceLayoutInfo workspaceLayout{}`

#### 文件 2: allto_all_matmul_v2_tiling_base.h
- 新增成员 `uint64_t libApiWorkSpaceSize_{0}`
- `GetWorkspaceSize()`: `libApiWorkSpaceSize_ = GetLibApiWorkSpaceSize(); workspaceSize_ = libApiWorkSpaceSize_ + RUNTIME_INFO_TOTAL_SIZE`
- `DoOpTiling()`: 填充 workspaceLayout（2 段：LIB_API + RUNTIME_INFO）

#### 文件 3: allto_all_matmul_v2_apt.cpp
- include mc2_runtime_info.h
- 初始化 Mc2RuntimeInfo（`Init(workspaceGM, &tilingData.workspaceLayout, usedCoreNum)`），通过 SetRuntimeInfo 注入 impl

#### 文件 4: all_to_all_mx_quant_matmul_urma_impl.h
- include mc2_runtime_info.h
- 新增 runtimeInfo_ 成员 + execTurn_ + position 常量 + SetRuntimeInfo
- RunAllToAll: 加 WriteGmByPassDCache/IncCommitCount/IncWaitCount/WritePhaseByPassDCache
- RunMatmul/RunLocalMatmul: 加 WriteGmByPassDCache

#### 文件 5: moe_distribute_comm_ctx.h
- 新增 UrmaCommContextForDump 结构体（镜像 CommUdmaContext 布局）

#### 文件 6: mc2_exception_dump.h (utils)
- 新增 Mc2DumpUrmaContext 函数（+ #else no-op stub）
- 逻辑：读 context 指针 → D2H CommContext → 落盘 `comm_context_*`；解析 commBufferAddrs[rankId] → D2H 1MB → 落盘 `peermem_window_*`

#### 文件 7: allto_all_matmul_v2_tiling.cpp
- include mc2_exception_dump.h + apace tiling data header
- 注册异常回调：Mc2DumpTilingData（tilingGmArgIdx=9, workspaceGmArgIdx=8, wsLayoutOffset=offsetof(workspaceLayout)）+ Mc2DumpUrmaContext（contextArgIdx=0）

### peermem 窗口 dump 方案

#### 背景

urma 的跨卡通信不走 workspaceGM，而是走 URMA peermem 窗口。每个 rank 在 `CommContext.udmaCtx.commBufferAddrs[rankId]` 处有一个接收窗口（`selfWinAddr`），AllToAll 通信后各 rank 的 A 矩阵分块和 ScaleA 数据按 rank 顺序排列在此窗口中。AIC 的 `RunMatmul` 直接从 `selfWinAddr` 读取远程数据做 matmul（`params.mmadParams.aGmAddr = baseParams_.selfWinAddr`）。

因此 peermem 窗口是 urma 路径的"workspace 等价物"——异常时 dump 它可以检查通信后的数据是否正确。

#### 数据定位链路

```
kernel args (arg 0: context)
  → D2H 读 uint64_t contextAddr (CommContext 的 device 地址)
    → D2H 读 sizeof(UrmaCommContextForDump) 从 contextAddr
      → 解析 udmaCtx.rankId
      → 取 commBufferAddrs[rankId] = selfWinAddr (peermem 窗口 device 地址)
        → D2H 读 WIN_SIZE (1MB) 从 selfWinAddr
          → 落盘 peermem_window_*
```

#### 实现位置

`mc2/common/utils/mc2_exception_dump.h` 中的 `Mc2DumpUrmaContext` 函数（`#if MC2_DFX_ENABLE` 块内）。

函数签名：
```cpp
inline void Mc2DumpUrmaContext(aclrtExceptionInfo *args, const char *op,
                               uint32_t contextArgIdx, uint32_t argsOffset = 0U)
```

#### dump 步骤

1. `GetAclrtGetArgsFromExceptionInfoFunc` → 获取 `devArgsPtr/devArgsLen`
2. `aclrtGetDeviceIdFromExceptionInfo` → `deviceId`
3. `GetAcldumpGetPathFunc` → `dumpPath`
4. 计算 `contextOffset = argsOffset + contextArgIdx * 8`，校验越界
5. D2H 读 `contextAddr`（arg 0 的值 = CommContext device 地址）
6. D2H 拷贝 `sizeof(UrmaCommContextForDump)` 从 `contextAddr` → 落盘 `comm_context_*`
   - 含 rankId、rankSize、channelHandles[64]、commBufferAddrs[64]（通信拓扑）
7. 解析 `rankId`，校验 `< URMA_MAX_RANK_NUM(64)`
8. 取 `selfWinAddr = commBufferAddrs[rankId]`，校验非 0
9. D2H 拷贝 `WIN_SIZE`(1MB) 从 `selfWinAddr` → 落盘 `peermem_window_*`
   - 含 AllToAll 后的 A 矩阵分块 + ScaleA 数据

#### UrmaCommContextForDump 结构体（镜像 CommUdmaContext 布局）

定义在 `mc2/common/op_kernel/moe_distribute_comm_ctx.h`：
```cpp
constexpr uint32_t URMA_MAX_RANK_NUM = 64;
struct UrmaCommContextForDump {
    uint32_t rankId = 0;                         // 当前卡 rank
    uint32_t rankSize = 0;                        // 总卡数
    uint64_t channelHandles[URMA_MAX_RANK_NUM];   // 每 rank 的 URMA channel 句柄
    uint64_t commBufferAddrs[URMA_MAX_RANK_NUM];  // 每 rank 的 peermem 窗口地址
};
```

#### 注册调用

在 `allto_all_matmul_v2_tiling.cpp` 的异常回调中调用：
```cpp
Mc2Exception::Mc2DumpUrmaContext(args, "AlltoAllMatmulV2",
    0U,   // contextArgIdx (context = arg 0)
    0U);  // argsOffset
```

### 异常时 dump 产出

| 文件 | 内容 | 大小 | 来源 |
|------|------|------|------|
| args_info_* | kernel args 全部 GM 指针 | 256B | Mc2DumpTilingData |
| tiling_data_* | tiling 结构体（含 workspaceLayout） | 2048B | Mc2DumpTilingData |
| workspace_seg0_type0_* | LIB_API 段原始数据（库 API workspace） | ~16MB | Mc2DumpTilingData |
| workspace_seg1_type12_* | **Mc2RuntimeInfo** — 每核执行进度（turn/position/phase/commit/wait） | 64KB | Mc2DumpTilingData |
| comm_context_* | **CommContext** — 通信拓扑（rankId/rankSize/channelHandles/commBufferAddrs） | ~1KB | Mc2DumpUrmaContext |
| peermem_window_* | peermem 窗口数据（AllToAll 后的 A+ScaleA） | 1MB | Mc2DumpUrmaContext |

### 崩溃定位信息对照

| 文件 | 回答的问题 |
|------|-----------|
| tiling_data_* | 算子参数对不对（M/N/K/tiling/coreNum） |
| workspace_seg1_type12_* | 卡在哪一步（execTurn/execPosition/commPhase），commit≠wait 说明卡在通信 Wait |
| comm_context_* | 通信环境对不对（rankId/rankSize/channel/bufferAddrs 拓扑） |
| peermem_window_* | 通信后的数据对不对（A 矩阵 + ScaleA 内容） |
| args_info_* | 所有 GM 地址对不对 |

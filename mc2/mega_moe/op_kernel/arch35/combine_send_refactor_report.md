# CombineSend 单核模块拆分报告

## 背景

`mega_moe` 的主流程可拆成 7 个阶段：

1. TopKGather
2. TokenDispatch
3. GMM1
4. SwigluQuant
5. GMM2
6. CombineSend
7. CombineReduce

本次修改目标是将第 6 步 `CombineSend` 从入口文件中拆出来，收敛到 `mega_moe_combine_send.h` 中作为单核逻辑模块。`mega_moe.h` 和 `mega_moe_layered.h` 保持多核入口职责，只负责计算/传递当前核上下文，不在模块内部直接使用 `blockIdx`。

## 修改前

### 1. `mega_moe_combine_send.h`

修改前已经包含部分 CombineSend 相关函数，例如：

- `CombineTokens`
- `CombineTokensLayered`
- `CombineSendTokenToRemote`
- `QuantMxFp8`
- `DeQuantMxFp8`
- `CombineQuantizedTokens`
- `CombineTokenGroup`

但是 `CombineSendTokenToRemote` 内部仍直接获取核上下文：

```cpp
uint32_t aivIdx = GetBlockIdx() / GetTaskRation();
uint32_t aivNum = GetBlockNum();
```

这意味着该函数并不是纯单核模块，它同时包含了多核入口的调度信息获取逻辑。

### 2. `mega_moe.h`

`ProcessCombine` 中包含完整的 combine-quant 单核处理流程，包括：

- 计算 `nTilesPerGroup`
- 设置 `gmm2CombineSyncCounter`
- 计算 `quantTokenSizeBytes`
- 根据当前 core 计算 token group 分配
- 等待 GMM2 tile 完成计数
- 从 `tripleInfoPtr` 搬入 triple 信息
- 调用 `MegaMoeCombineImpl::CombineTokenGroup`

这部分逻辑和 `mega_moe_layered.h` 中的实现高度重复。

### 3. `mega_moe_layered.h`

`ProcessCombine` 中也包含一份和 `mega_moe.h` 基本一致的 combine-quant 单核处理流程。

此外，在 `COMBINE_NO_QUANT` 路径中调用：

```cpp
MegaMoeCombineImpl::CombineSendTokenToRemote<bfloat16_t>(params_);
```

由于旧接口内部自己调用 `GetBlockIdx()` / `GetBlockNum()`，入口没有显式传递多核上下文。

## 修改后

### 1. `CombineSendTokenToRemote` 改为单核接口

位置：`mega_moe_combine_send.h`

修改后接口为：

```cpp
template <typename ElementMMadOut2>
__aicore__ inline void CombineSendTokenToRemote(uint32_t aivIdx, uint32_t aivNum, const Params& params)
```

变化点：

- 删除模块内部的 `GetBlockIdx()`。
- 删除模块内部的 `GetBlockNum()`。
- `aivIdx` 和 `aivNum` 改由入口文件计算后传入。
- 模块只根据传入的单核上下文处理当前核负责的 rank 范围。

修改后 `mega_moe_combine_send.h` 内不再出现：

- `blockIdx`
- `GetBlockIdx`
- `GetBlockNum`
- `GetSubBlockIdx`

### 2. 新增 `ComputeCombineCoreGrouping`

位置：`mega_moe_combine_send.h`

新增函数：

```cpp
__aicore__ inline void ComputeCombineCoreGrouping(
    uint32_t coreId,
    uint32_t numGroups,
    uint32_t totalCores,
    uint32_t& myGroup,
    uint32_t& myIdxInGrp,
    uint32_t& myGrpSize)
```

作用：

- 将 `totalCores` 均匀分配到 `numGroups` 个 token group。
- 计算当前 core 所属的 group、group 内索引以及 group 大小。
- 供 CombineSend 单核模块内部使用，避免依赖 `mega_moe_impl.h` 中的调度实现，降低头文件耦合。

### 3. 新增 `ProcessCombineSingleCore`

位置：`mega_moe_combine_send.h`

新增函数：

```cpp
template <int32_t CombineQuantMode>
__aicore__ inline void ProcessCombineSingleCore(
    uint32_t coreIdForGrouping,
    uint32_t totalCoresForGrouping,
    uint32_t aivCoreIdx,
    uint32_t blockAivNum,
    uint32_t expertIdx,
    uint32_t rankId,
    uint32_t k,
    uint32_t expertBeforeCnt,
    uint32_t mExpert,
    GM_ADDR gmm2OutAddr,
    int64_t combineUbTensorSize,
    const Params& params)
```

该函数承接原来分散在 `mega_moe.h` 和 `mega_moe_layered.h` 中的 combine-quant 单核逻辑。

内部职责包括：

- 计算当前 expert 的 token group 数量。
- 根据入口传入的 core 信息计算当前核负责的 token group。
- 等待 GMM2 到 Combine 的同步计数达到 `nTilesPerGroup`。
- 计算当前核负责的 token 范围。
- 搬入 triple 信息。
- 调用 `CombineTokenGroup` 完成量化后发送。

### 4. `mega_moe.h` 入口变薄

位置：`mega_moe.h`

`ProcessCombine` 修改后只保留入口侧多核逻辑：

- 默认使用 `aivCoreIdx_` / `blockAivNum_`。
- A8W4 路径下只允许 `subBlockIdx_ == 1` 参与。
- A8W4 路径下将物理 AIV 核号转换为参与 Combine 的逻辑核号。
- 调用 `MegaMoeCombineImpl::ProcessCombineSingleCore`。

修改后入口文件不再直接处理：

- token group 分配细节
- sync counter 等待
- triple 搬入
- token group 内的 CombineSend 执行

### 5. `mega_moe_layered.h` 入口变薄

位置：`mega_moe_layered.h`

`ProcessCombine` 做了与 `mega_moe.h` 相同的收敛：

- 保留 layered 入口自己的多核上下文计算。
- 将单核 CombineSend 执行委托给 `ProcessCombineSingleCore`。

`COMBINE_NO_QUANT` 路径也改为显式传入核上下文：

```cpp
MegaMoeCombineImpl::CombineSendTokenToRemote<bfloat16_t>(blockIdx_, blockNum_, params_);
```

## 文件级变化统计

基于提交 `213dfe5f6 single core`：

| 文件 | 新增行 | 删除行 | 说明 |
| --- | ---: | ---: | --- |
| `mega_moe.h` | 4 | 52 | 删除重复单核 Combine 逻辑，改为调用模块函数 |
| `mega_moe_combine_send.h` | 84 | 6 | 新增单核模块封装函数，接口改为显式接收核上下文 |
| `mega_moe_layered.h` | 5 | 53 | 删除重复单核 Combine 逻辑，改为调用模块函数，并更新 no-quant 调用 |
| 合计 | 93 | 111 | 净减少 18 行 |

PR 中显示 `93 insertions(+), 111 deletions(-)` 是 Git diff 的真实统计。

## 关于“231 行”和 PR “93 行”的差异

“231 行”可以理解为生成/搬迁时涉及的代码体量或可见代码块体量，不等同于 PR 的新增行数。

这次修改不是纯新增，而是：

- 从 `mega_moe.h` 删除一份重复实现。
- 从 `mega_moe_layered.h` 删除一份重复实现。
- 在 `mega_moe_combine_send.h` 新增统一的单核实现。
- 调整已有接口参数。

因此 Git 最终统计的是：

```text
93 insertions(+), 111 deletions(-)
```

也就是说，部分代码是“从入口文件移动/收敛到模块文件”，不是全部作为新增代码进入 PR。

## 修改后的职责边界

### `mega_moe.h` / `mega_moe_layered.h`

职责：

- 多核入口。
- 判断当前 core 是否参与 Combine。
- 处理不同模板路径的核号映射。
- 将必要上下文传入单核模块。

不再负责：

- CombineSend 内部 token group 处理。
- GMM2 sync counter 等待细节。
- triple 搬入和 token 范围拆分。

### `mega_moe_combine_send.h`

职责：

- CombineSend 单核逻辑。
- no-quant 发送逻辑。
- combine-quant token group 处理。
- 根据入口传入的 core 上下文完成当前核负责的工作。

约束：

- 模块内部不直接使用 `blockIdx`。
- 模块内部不调用 `GetBlockIdx()`。
- 模块内部不调用 `GetBlockNum()`。
- 模块内部不调用 `GetSubBlockIdx()`。

## 行为保持点

本次修改保持以下行为不变：

- A8W4 路径仍只允许 `subBlockIdx_ == 1` 参与 combine-quant。
- A8W4 路径仍使用 `aivCoreIdx_ / 2` 作为逻辑参与核号。
- sync counter 地址计算仍基于：

```cpp
expertIdx * blockAivNum * INT_CACHELINE + aivCoreIdx * INT_CACHELINE
```

- triple 信息仍从 `params.workspaceInfo.tripleInfoPtr` 按 `expertBeforeCnt + tokenStart + myTokenOffset` 读取。
- 最终仍调用 `CombineTokenGroup<CombineQuantMode, bfloat16_t>` 执行量化后发送。
- layered no-quant 路径仍由 `subBlockIdx == 1` 的 AIV 执行，只是核号改为入口显式传入。

## 验证情况

已完成：

- `git diff --check` 通过。
- `rg` 检查确认 `mega_moe_combine_send.h` 中不存在：
  - `blockIdx`
  - `GetBlockIdx`
  - `GetBlockNum`
  - `GetSubBlockIdx`

未完成：

- 未完成实际算子编译。
- 执行编译命令：

```bash
bash build.sh --pkg --soc=ascend950 --ops="mega_moe" --vendor_name="DC"
```

当前 Windows 环境中的 `bash` 指向 `C:\Windows\system32\bash.exe`，启动后提示 WSL/Linux 环境不可用或未安装发行版，因此编译未进入代码编译阶段。

## 风险点

1. `ProcessCombineSingleCore` 中重新放置了原先入口文件中的单核逻辑，理论行为应保持一致，但仍需要实际编译和算子用例验证。
2. `ComputeCombineCoreGrouping` 是从原有 grouping 逻辑等价迁移出来的本地 helper，需要重点关注 `numGroups` 和 `totalCoresForGrouping` 的边界。
3. no-quant layered 路径现在由入口传入 `blockIdx_` / `blockNum_`，与原先函数内部 `GetBlockIdx() / GetTaskRation()`、`GetBlockNum()` 的结果应一致。

## 总结

本次修改将第 6 步 `CombineSend` 的单核执行逻辑从两个多核入口文件中抽离，统一放到 `mega_moe_combine_send.h`。

修改后的结构更清晰：

- 入口文件负责多核调度和模板路径差异。
- `mega_moe_combine_send.h` 负责单核 CombineSend 执行。

同时满足“单核模块内部不能出现 `blockIdx`”的要求，并减少了 `mega_moe.h` 与 `mega_moe_layered.h` 中的重复实现。

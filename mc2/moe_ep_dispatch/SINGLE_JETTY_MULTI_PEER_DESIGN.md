# MoE EP Dispatch/Combine 单 Jetty 多对端改造说明

## 1. 改造目标

本次改造参考 `ops-transformer` PR 12570，将 `moe_ep_dispatch` 和 `moe_ep_combine` 的直连通信资源从“按对端 Rank 分配独立 Channel/Jetty”调整为“一个 Jetty 复用多个对端”。

这里的“单 Jetty 多对端”含义是：

- 一个 `MultiChannelHandle` 表示一个 Jetty；
- 该 Jetty 内包含当前 Rank 到其余 `epWorldSize - 1` 个 Rank 的子 Handle；
- 算子执行时通过 `GetHandleRef(batch, peerIndex, remoteWindow)` 选择本次通信的实际对端；
- 系统固定创建 32 个这样的共享 Jetty，32 个 AIV 与 32 个 Jetty 一一对应；
- 每个 AIV 独占自己的 Jetty，并在该 Jetty 上依次访问所有对端。

因此，本次改造不是把整个算子缩减到只有一个通信队列，而是把原先的“Jetty × 对端”资源组织方式改成“每个 Jetty 内部包含多个对端”。

## 2. 改造前后对比

| 项目 | 改造前 | 改造后 |
| --- | --- | --- |
| Handle 组织 | 按 Rank、Channel 索引保存普通 Channel Handle | 保存 32 个 `MultiChannelHandle` |
| 对端选择 | 先根据 Rank 计算独立 Handle 下标 | 从共享 Jetty 中通过 `peerIndex` 选择子 Handle |
| AIV 分工 | AIV/Channel 可能按 Rank 分组 | 一个 AIV 固定独占一个 Jetty，并遍历全部 Rank |
| Dispatch 数据切分 | 通信核与目标 Rank 绑定或分组 | 每个 Rank 的 payload 在 32 个 Jetty 间切分 |
| Combine 数据切分 | 每个 Rank 分配一组通信通道 | 每个 Rank 的 metadata 范围在 32 个 Jetty 间切分 |
| 完成标记 | 与旧 Channel 数或动态 notify 数关联 | 固定使用 `[rank][jetty]` 二维完成标记 |
| 拓扑 | 支持 direct/hybrid 分支 | 当前共享 Jetty 实现仅允许 direct |
| Kernel 启动核数 | 使用平台可用 AIV 数 | 平台至少需要 32 个 AIV，实际固定启动 32 个 AIV |

## 3. 总体结构

```text
                    MoeSharedJettyContext
                    ├── epRankId
                    ├── epRankSize
                    ├── epHcclBuffer[rank]
                    └── multiChannelHandles[32]
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
           Jetty 0             Jetty 1            Jetty 31
              │                   │                   │
      ┌───────┼───────┐   ┌───────┼───────┐   ┌───────┼───────┐
      │       │       │   │       │       │   │       │       │
    peer 0  peer 1  peer N ...  peer 0  peer 1 ...  peer 0  peer 1 ...

           AIV 0               AIV 1               AIV 31
              │                   │                   │
         独占 Jetty 0        独占 Jetty 1       独占 Jetty 31
```

本地 Rank 不占用远端子 Handle。远端 Rank 到 `peerIndex` 的映射为：

```cpp
peerIndex = dstRank < localRank ? dstRank : dstRank - 1;
```

例如 `worldSize = 8`、本地 `rankId = 3` 时，Jetty 内部的 7 个子 Handle 依次对应 Rank `0、1、2、4、5、6、7`。

## 4. 新的共享通信上下文

新增文件：

- `mc2/common/op_kernel/moe_shared_jetty_context.h`
- `torch_extension/cann_ops_transformer/common/moe_shared_jetty_adapter.h`
- `mc2/common/torch_extension/csrc/moe_shared_jetty_adapter.cpp`
- `mc2/common/torch_extension/csrc/moe_shared_jetty_context.cpp`

### 4.1 Context 布局

`MoeSharedJettyContext` 定义如下：

```cpp
struct MoeSharedJettyContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint32_t sharedJettyNum;
    uint32_t reserved;
    uint64_t epHcclBuffer[1024];
    uint64_t multiChannelHandles[32];
};
```

字段用途：

- `epRankId`：当前 EP Rank；
- `epRankSize`：EP 通信域总 Rank 数；
- `sharedJettyNum`：共享 Jetty 数，当前固定为 32；
- `epHcclBuffer`：每个 Rank 注册通信窗口的地址；
- `multiChannelHandles`：32 个多对端 Jetty Handle。

支持的 Rank 数范围为 `[2, 1024]`。Host Tiling 会按 `sizeof(MoeSharedJettyContext) / sizeof(int32_t)` 严格校验输入 context 的长度，防止把旧 `MoeCommContext` 误传给新 Kernel。

### 4.2 Context 创建流程

`MoeSharedJettyContextBuilder::Build()` 的处理顺序如下：

1. 根据通信组名获得 `HcclComm`；
2. 查询当前 Rank ID 和 Rank Size；
3. 遍历 HCCL Rank Graph 的网络层；
4. 查找同时覆盖全部 Rank 的 `COMM_PROTOCOL_UBC_CTP` 网络层；
5. 对每个远端收集 `CommLink`，并选择所有远端都能共同使用的本地 Endpoint；
6. 分配通信窗口，按 1 GiB 大页向上对齐，并清零实际请求范围；
7. 通过 `HcclCommMemReg` 注册通信窗口；
8. 创建 32 个 `MultiChannelHandle`，每个 Handle 都包含全部远端的 Channel 描述；
9. 查询远端注册内存并填充 `epHcclBuffer[rank]`；
10. 通过 HCCL Engine Context 保存 Context，并返回设备侧 `int32` Tensor。

Context tag 中包含 Jetty 数以及 SQ/CQ 深度版本信息：

```text
<group>_moe_dispatch_combine_shared_jetty_v2_j32_sq4096_cq64
```

这能避免不同资源布局错误复用同一个 HCCL Engine Context。

### 4.3 Endpoint 选择

多对端 Jetty 的关键要求是所有远端 Channel 能共享同一个本地 Endpoint。构建逻辑会：

- 先确认候选网络层覆盖整个通信域；
- 获取当前 Rank 到每个远端的候选 UBC Link；
- 以第一个远端的本地 Endpoint 为候选；
- 在其余远端的 Link 中寻找协议、通信地址和位置均一致的本地 Endpoint；
- 只有全部远端都匹配时才生成 `HcclChannelDesc` 数组。

如果没有任何 UBC 网络层满足要求，初始化会直接报错：

```text
No UBC layer supports one shared Jetty endpoint to all peers
```

### 4.4 Buffer 和 Context 复用

同一通信组初始化后，通信窗口和 Engine Context 会被缓存。再次创建相同 Context 时直接复用已有资源。

由于已注册的通信窗口不能在原 Context 上原地扩容，`ElasticBuffer::EnsureMoeContext()` 增加了限制：后续请求的 `cclBufferSize` 不得大于首次分配值。需要更大窗口时必须创建新的通信组或新的 `ElasticBuffer`。

## 5. HCCL Host 接口扩展

`torch_extension/cann_ops_transformer/common/hccl_common.h` 增加了共享 Jetty 初始化所需的动态接口：

- `HcclEngineCtxDestroy`；
- `HcclChannelConfigCreate`；
- `HcclChannelConfigDestroy`；
- `HcclChannelConfigSetInt`；
- `HcclChannelConfigSetStr`；
- `HcclChannelAcquireWithConfig`。

Channel Acquire 时设置共享队列属性以及队列 tag。`moe_shared_jetty_adapter.cpp` 对 `MakeMultiChannelHandle` 做了一层独立封装，避免将 Host 侧 HCOMM 细节暴露给 PyTorch Extension 的其他代码。

## 6. ElasticBuffer 改造

### 6.1 C++ 侧

`mc2/common/torch_extension/csrc/elastic_buffer.cpp` 不再使用原来的 `MoeContextBuilder` 创建旧 Context，改为：

```cpp
moeContextTensor_ = BuildMoeSharedJettyContext(groupName_, cclBufferSize);
```

同时：

- `ResolveRankNumPerServer()` 返回整个 `epWorldSize`；
- `ResolveTopoType()` 固定返回 `NETWORK_DIRECT`；
- 已初始化 Context 不允许扩大通信窗口。

### 6.2 Python 侧

`mc2/common/torch_extension/elastic_buffer.py` 做了以下同步修改：

- Extension 编译源加入共享 Jetty Context 和 Adapter；
- 增加 `_MOE_SHARED_JETTY_NUM = 32`；
- Dispatch notify 槽位数固定为 32；
- Combine 完成标记数改为 `world_size * 32`；
- 删除 hybrid 模式额外窗口的预留计算，只计算 direct 模式窗口。

`torch_extension/setup.py` 同时加入新头文件的打包规则。

## 7. Dispatch Kernel 改造

主要文件：`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h`。

### 7.1 Context 类型替换

Kernel 的 Context 从：

```cpp
MoeCommContext
```

替换为：

```cpp
MoeSharedJettyContext
```

窗口地址不再从旧 Context 的字段获得，而是统一通过 `epHcclBuffer[rank]` 获取目标 Rank 的注册窗口基址。

### 7.2 AIV 与 Jetty 一一绑定

`GetCoreAssignment()` 的分工改为：

```cpp
coreIndexInGroup = aivId;
groupSize = sharedJettyNum;
channelIndex = aivId;
copyRankStart = 0;
copyRankNum = epWorldSize;
```

即：

- AIV 0 固定使用 Jetty 0；
- AIV 1 固定使用 Jetty 1；
- ……
- AIV 31 固定使用 Jetty 31；
- 每个 AIV 都遍历所有目标 Rank；
- 每个 Rank 的 token 行在 32 个 AIV/Jetty 间切片。

切片逻辑为：

```cpp
cntPerChannel = ceil(sendTokenNum / 32);
tokenStart = aivId * cntPerChannel;
tokenNum = min(cntPerChannel, sendTokenNum - tokenStart);
```

即使某个 Jetty 没有分到 token，也必须向目标 Rank 写入该 Jetty 的完成 notify，确保 epilogue 不会一直等待。

### 7.3 对端子 Handle 选择

Payload 发送时先用本 AIV 对应的 Jetty 创建 Batch：

```cpp
auto batch = hcomm.MakeBatchHandle(
    static_cast<MultiChannelHandle>(multiChannelHandles[aivId]),
    batchBuffer,
    batchBufferBytes);
```

再根据目标 Rank 选择子 Handle：

```cpp
auto& handle = hcomm.GetHandleRef(batch, peerIndex, remoteWindow);
```

最后在该子 Handle 上提交 `WriteNbi` 或 `WriteWithNotifyNbi`。

Count/control 路径仍只由 `channelIndex == 0` 的 AIV 负责，避免同一份计数被重复发送；payload 路径则由全部 32 个 Jetty 并行处理。

### 7.4 Dispatch 完成标记

`dispatchNotifyCount` 固定为 32。完成标记按以下方式组织：

```text
slot_state[源 Rank][Jetty]
```

每个 Jetty 给每个目标 Rank 写一个完成标记。Dispatch epilogue 等待所有 `epWorldSize * 32` 个标记到齐后，再消费数据并清理状态。

## 8. Combine Kernel 改造

主要文件：`mc2/moe_ep_combine/op_kernel/moe_ep_combine.h`。

### 8.1 Handle 类型替换

原来的普通 Channel Batch：

```cpp
BatchHandle<ChannelHandle>
```

替换为：

```cpp
BatchHandle<MultiChannelHandle>
```

每个 AIV 通过 `multiChannelHandles[aivId]` 获取自己独占的 Jetty。

### 8.2 多对端循环

32 个 AIV 分别从不同目标 Rank 开始轮转：

```cpp
targetRank = (aivId + peerOffset) % epWorldSize;
```

这样可以错开不同 Jetty 的首个目标 Rank，减少所有 AIV 同时集中访问一个对端的情况。

对每个目标 Rank：

1. 从 `rankOffsets` 取出该 Rank 对应的 metadata 范围；
2. 使用 `SplitRange(..., 32, aivId, ...)` 将范围切给当前 Jetty；
3. 根据 metadata 生成 token/weight/metadata 的远端写请求；
4. 提交当前对端的 Batch；
5. 写入当前 `[源 Rank][Jetty]` 完成标记。

### 8.3 Batch 切换规则

一个 Batch 在提交期间只能绑定一个物理对端队列。因此从一个对端切换到另一个对端之前，必须先：

```cpp
FlushPreparedWrites(true);
```

然后更新：

- `activePeerIndex`；
- `activeRemoteWindow`；
- `GetHandleRef()` 选中的子 Handle。

这保证同一个多对端 Jetty 在遍历 Rank 时不会把待提交 WQE 发到错误的物理队列。

### 8.4 Combine 完成标记

完成标记地址使用：

```cpp
flagIndex = sourceRank * sharedJettyNum + jettyIndex;
```

因此完成标记总数为：

```text
epWorldSize * 32
```

远端完成标记使用通信窗口中的持久常量页作为异步写源；本地 Rank 则直接写本地窗口。Combine epilogue 汇总全部标记，期望和为 `epWorldSize * 32`，之后清零整个标记区。

## 9. Epilogue 和通信完成处理

主 Kernel 已切换到新的 Context 与状态布局，因此 Dispatch/Combine epilogue 必须同步修改，否则会用旧 Context 布局解析错误地址。

涉及文件：

- `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h`
- `mc2/moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h`
- 两个 epilogue 对应的 Tiling 文件。

`mc2/common/op_kernel/moe_ep_send_completion.h` 新增了针对 `MoeSharedJettyContext` 的完成队列处理：

1. 每个 AIV 领取一个或多个 Jetty；
2. 用 `MakeBatchHandle(MultiChannelHandle)` 创建 Batch；
3. 遍历所有远端；
4. 用 `GetHandleRef()` 选择子 Handle；
5. 对每个子 Handle执行 CQ/完成队列处理。

异常诊断写入器 `moe_ep_exception_dump_writer.h` 改为模板化 Context 类型，同时保留旧 Context 的重载，避免影响其他仍在使用旧通信 Context 的算子。

## 10. Window Layout 变化

公共窗口布局位于：

- `mc2/common/op_host/op_tiling/moe_ep_window_layout.h`
- `mc2/common/op_host/op_tiling/moe_ep_window_layout.cpp`

主要变化如下：

### 10.1 Dispatch notify 区

原先 notify 数会根据 Channel budget 和 Rank 数动态计算；现在固定为：

```text
dispatchNotifyCount = 32
```

### 10.2 Combine completion 区

原先完成区按 Rank 数或旧 Channel 数取较大值；现在明确按二维表分配：

```text
combineFlagCount = epWorldSize * 32
combineFlagBytes = combineFlagCount * 512
```

每个标记独占一个 512 字节对齐单元，避免不同发送者更新同一 cache line。

### 10.3 持久完成标记源

Combine 的异步完成标记写不能引用算子 workspace 中生命周期不足的临时数据，因此在通信窗口中保留一个 512 字节的常量源区域：

```text
combineFlagSourceWinOffset
```

## 11. Host Tiling 限制

Dispatch、Combine 及各自 epilogue 的 Tiling 均增加以下检查：

1. Context 长度必须与 `MoeSharedJettyContext` 完全一致；
2. `networkMode` 必须是 `NETWORK_DIRECT`；
3. 平台 AIV 数必须不少于 32；
4. 实际参与通信的 AIV 数固定为 32。

当前 API 层仍可能接受 direct/hybrid 枚举，但 hybrid 会在 Tiling 阶段被共享 Jetty 校验拒绝。后续如需恢复 hybrid，需要单独设计跨 Server 的共享 Endpoint、窗口布局以及 completion 语义，不能直接复用当前 direct 实现。

## 12. 修改文件清单

### 12.1 新增文件

- `mc2/common/op_kernel/moe_shared_jetty_context.h`
- `mc2/common/torch_extension/csrc/moe_shared_jetty_context.cpp`
- `mc2/common/torch_extension/csrc/moe_shared_jetty_adapter.cpp`
- `torch_extension/cann_ops_transformer/common/moe_shared_jetty_adapter.h`

### 12.2 公共模块

- `mc2/common/op_host/op_tiling/moe_ep_window_layout.h`
- `mc2/common/op_host/op_tiling/moe_ep_window_layout.cpp`
- `mc2/common/op_kernel/moe_ep_exception_dump_writer.h`
- `mc2/common/op_kernel/moe_ep_send_completion.h`
- `mc2/common/torch_extension/csrc/elastic_buffer.cpp`
- `mc2/common/torch_extension/elastic_buffer.py`
- `torch_extension/cann_ops_transformer/common/hccl_common.h`
- `torch_extension/setup.py`

### 12.3 Dispatch

- `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h`
- `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_base.h`
- `mc2/moe_ep_dispatch/op_host/op_tiling/moe_ep_dispatch_tiling.cpp`
- `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h`
- `mc2/moe_ep_dispatch_epilogue/op_host/op_tiling/moe_ep_dispatch_epilogue_tiling.cpp`

### 12.4 Combine

- `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h`
- `mc2/moe_ep_combine/op_host/op_tiling/moe_ep_combine_tiling.cpp`
- `mc2/moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h`
- `mc2/moe_ep_combine_epilogue/op_host/op_tiling/moe_ep_combine_epilogue_tiling.cpp`

## 13. 当前约束与注意事项

- 仅支持 direct 网络拓扑；
- EP Rank 数必须在 `[2, 1024]`；
- 硬件至少提供 32 个 AIV；
- 需要存在一个覆盖全部远端、并共享相同本地 Endpoint 的 UBC 网络层；
- 通信窗口大小必须为正数且 2 MiB 对齐；
- 实际物理内存使用 1 GiB 大页向上对齐；
- 同一通信组的共享窗口创建后不能扩容；
- `peerIndex` 排除了本地 Rank，不能直接使用 `dstRank` 作为子 Handle 下标；
- 对端切换前必须提交当前 Batch；
- 即使某个 Jetty 没有 payload，也必须发布对应完成标记。

## 14. 已完成的本地检查

本次修改已完成以下静态检查：

```text
pre-commit clang-format    PASS
pre-commit ruff-format     PASS
pre-commit ruff-check      PASS
pre-commit trailing-whitespace PASS
python -m py_compile       PASS
git diff --check           PASS
```

当前开发环境为 Windows，未检测到可用的 CANN/Linux 算子编译工具链，因此尚未完成真实算子编译与设备运行测试。

## 15. 建议的设备侧验证

在具备 CANN 和多卡环境后，建议至少覆盖以下场景：

1. `worldSize = 2`，验证最小通信域；
2. `worldSize = 8`，验证典型单机 EP；
3. token 数小于 32，验证部分 Jetty 无 payload 时仍能正确完成；
4. token 数不能被 32 整除，验证尾部切片；
5. 某些目标 Rank 的发送 token 数为 0；
6. cached 和 non-cached Dispatch；
7. Combine 包含本地 Rank 与远端 Rank 的混合写入；
8. 连续多轮调用，验证完成标记清零和 Context 复用；
9. 首次小窗口、后续请求大窗口，验证扩容保护；
10. 与改造前实现进行输出精度对比，并检查是否存在 CQ 超时或残留 WQE。

重点观察：

- 每个目标 Rank 是否收到 32 个 Dispatch notify；
- Combine epilogue 是否收到 `worldSize * 32` 个完成标记；
- 多对端切换时是否出现数据写入错误 Rank；
- 连续执行时是否出现上一轮状态残留；
- 异常退出或 Context 创建失败时，设备内存是否正确清理。

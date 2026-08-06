# mega_moe dispatch 分核移植说明（基于动态 wave 调度 20260803）

## 1. 背景

参考仓 `codex/mega-moe-wave-dynamic-scheduler-20260803`（`ababd4a92` → `150815168` → `d7e8e76cf`）在 GMM1/GMM2
侧实现了按专家内 256-row 分组（M Group）的动态 wave 调度：

- `DispatchWaveAndGetEndCursor`：AIV1 规划当前 Wave 的**逻辑终点**（`mGroupsPerWave_` 个 256-row 分组，
  Wave 边界可以落在热点专家内部），但不按 Wave 切分数据发送——热点专家首次进入时**一次性全量发送**，
  后续 Wave 只推进逻辑游标。
- `ProcessGmm1Wave` / `ProcessGmm2Wave`：AIC/AIV0 按同一 Wave 边界消费，同一专家落在当前 Wave 的连续
  分组合并成一行段、只建一次 GMM problem；跨专家时各专家独立行段。
- GMM1 通过 `WaitForUpstreamReady` 轮询 `dispatchToGmm1Flag`（专家内按 256 行一槽、按行数累加）等待
  dispatch 数据就绪。

工作仓（`ops-transformer_2614`）此前在 dispatch-only 分支上验证了两类 dispatch 侧优化：

- **token-group wave 合并**：小专家按 256 行预算合并成一个 dispatch wave，大专家独占 wave；实测
  scan max 234→183（-21.7%），dispatch max 1128→1117（-1%）。
- **wave 行均分分核**：所有 AIV1 逻辑核按 wave 总行数均分，每核只搬自己行区间相交的专家行段，
  按段增量累加 GMM1 wave flag。

本次移植以参考仓动态 wave 调度为基础，把 dispatch 侧的“分核 + 按 Wave 行段发送”带上来，
替换参考仓“专家全量发送”的实现，保留参考仓 AIC/AIV0 的动态 wave 计算调度与完整流水结构。

## 2. 移植基线

- 分支：`dispatch_token_group_full_port`
- 基线 commit：`d7e8e76cf`（`codex/mega-moe-wave-dynamic-scheduler-20260803` 最新）

## 3. 修改内容

### 3.1 新增 `mc2/common/utils/mc2_kernel_timing.h`

自工作仓 `dispatch_token_group_full` 原样拷贝的 per-stage 打点框架：

- `StartTiming/StopTiming`：单次阶段计时（slot0-6，覆盖 Process 各阶段）。
- `AccumStartTiming/AccumStopTiming`：累计计时（slot7-11，覆盖 dispatch 内部细分）。
- `WriteTimingToOutput`：结果写回 yOut 头部，AIV 核自动加 `aicNum_` 偏移，Python 侧按
  `core * MAX_SLOTS * 2 + slot * 2` 读取。

### 3.2 `mega_moe_wave.h` 改动

#### 3.2.1 `GetExpertDispatchRowCount(uint32_t localExpertId)`（新增）

返回专家受 `maxOutputSize_` 截断后的实际可写行数，wave 行预算与各核行区间使用同一份截断口径。

#### 3.2.2 `MetaInfoCalAndDispatch`（单专家版 → Wave 分核版）

签名从 `(GMMAddrInfo&, int32_t localExpertId, ...)` 改为：

```cpp
MetaInfoCalAndDispatch(uint32_t waveBeginExpert, uint32_t waveBeginRowOffset,
                       uint32_t waveEndExpert, uint32_t waveEndRowOffset,
                       uint32_t waveRowCount, const DispatchBufferConfig &bufferConfig);
```

- 按 `waveRowCount` 在所有 `blockNum_` 个 dispatch core 间均分，得到本 core 的 wave 行区间
  `[coreWaveRowBegin, coreWaveRowEnd)`。
- 沿 wave 内专家逐个求交集，支持起点/终点专家落在热点专家中部
  （`waveBeginRowOffset` / `waveEndRowOffset`，0 表示专家边界半开）。
- 只对覆盖的专家线性扫描 `cumsumInfoTensor_` 定位其覆盖的 source rank 区间，按段内命中序号区间
  做逐 batch 的 mask 交集搬运（沿用原单专家版的 batch 扫描与 `CopyGMToGMPerToken`）。
- GMM1 flag 按 rank 段增量更新：每搬完一个段即累加该段覆盖的专家内 wave 槽，槽位按
  `flagDispatchToGmm1Ptr + localExpertId * dispatchFlagSlotsPerExpert_ + waveIdx * INT_CACHELINE`
  直接定位，与参考仓 `UpdateGlobalBuffer<GMM1>` 的
  `... + IDX_FLAG_OFFSET * dispatchFlagSlotsPerExpert_ + expertMGroupOffset * INT_CACHELINE`
  完全对齐（`expertMGroupOffset` 即专家内全局 wave 序号）。
- 打点：slot7 dispatch prepare、slot8 mask scan + copy、slot9 GMM1 flag、slot10 CopyGMToGMPerToken、
  slot11 mask scan。

#### 3.2.3 `DispatchExpert`（去掉全量发送）

只保留状态推进：`SendCntCal`（写 cumsum + 通知 AIC/AIV0 count 就绪）→ `UpdateGroupParams` →
`UpdateGlobalBuffer`。数据发送全部移入 `DispatchWaveAndGetEndCursor` 的分核路径。

#### 3.2.4 `DispatchWaveAndGetEndCursor`（规划 + 分核发送）

- 规划逻辑与参考仓完全一致（`mGroupsPerWave_` 个 256-row 分组、`GetWaveSegmentRowCount` 切段、
  零行专家规范化、跨 Wave 复用完整专家状态）。
- 新增：记录本次调用的 `waveBeginExpert / waveBeginRowOffset / plannedRowCount`，规划完成后调用
  Wave 分核版 `MetaInfoCalAndDispatch` 下发该 Wave 的行段。
- prefetch 语义保留：首轮“当前 Wave + 下一 Wave”两次规划发送，稳态复用 `preparedWaveEndCursor`
  并只预取下一 Wave。

#### 3.2.5 Process / ProcessGmmPipeline 打点

slot0 QuantProcessInRank、slot1 SendMaskCal、slot2 ResetFlagList、slot3 input prep tail +
cross-rank sync、slot4 Dispatch/Combine buffer Init、slot5 路由专家 wave 全流程、
slot6 tail sync & finalize（含共享专家 GMM2、Unpermute）。

## 4. 协作机制（与 GMM1 动态 wave 的对齐）

- 参考仓 GMM1 侧 `WaitForUpstreamReady`：`waveIdx = mLoc / tileM`，等
  `dispatchToGmm1Flag + waveIdx * INT_CACHELINE` 达到 `min(tileM, m - mLoc)`。
  分段 GMM1 的 `gmmAddrInfo.dispatchToGmm1Flag` 已带 `expertMGroupOffset * INT_CACHELINE` 偏移，
  因此 flag 槽 = 专家内全局 wave 序号，与分核 dispatch 的累加槽完全一致。
- 小专家合批：wave 内各专家 flag 槽独立，GMM1 各专家按自身行数（不足 256 取实际行数）等待，
  互不阻塞。
- 热点专家跨 Wave：dispatch 每 Wave 发一段并累加 flag，GMM1 每满 256 行即可消费一段，流水保持。

## 5. 与工作仓 token-group 版的差异

| 项 | 工作仓 `dispatch_token_group_full` | 本分支（移植版） |
| --- | --- | --- |
| Wave 边界 | 完整专家合并，≤256 行；大专家独占不切分 | 参考仓 M-Group 边界，可切热点专家 |
| AIC/AIV0 | 原专家批循环（`GetWaveExpertCount`） | 参考仓动态 wave（`ProcessGmm1Wave/Gmm2Wave`） |
| 完整流水 | 恢复屏蔽后的完整流水 | 参考仓原生完整流水（`ProcessGmmPipeline`） |
| 共享专家 | 恢复共享专家流程 | 参考仓原生共享专家流程 |
| dispatch 发送 | Wave 分核（本次移植保留） | 同左 |
| 打点 | slot0-11 | 同左 |

## 6. 验证

本机无 CANN 编译工具链，需在 NPU 环境编译验证：

1. `git checkout dispatch_token_group_full_port`
2. 编译 mega_moe 算子（A8W8，含 TopkWeightsPrefetch=1 与 =0 两条路径）。
3. 对比 yOut 头部打点数据：重点看 slot5（wave 全流程）、slot8（mask scan + copy）、
   slot10（token copy）在各核的分布，与 `dispatch_token_group_full` 的 10slot 数据对比。
4. 正确性回归：与 master 分支输出对齐（yOut 数据区不变，仅头部打点）。
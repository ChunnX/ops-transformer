# 工作仓 pull 流程详解（ops-transformer_2614 @ 23be325e8，dispatch_token_group_full_port）

> 面向没接触过 `mega_moe_wave.h` 的读者。术语：**pull** = dispatch 从 peermem 接收区取数、搬进本卡 workspace。
> 所有行号基于 `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h`。与参考仓（ababd4a92）的净差异
> 集中在 `DispatchExpert` / `DispatchWaveAndGetEndCursor` / `MetaInfoCalAndDispatch` 三个函数。
> 名词对照表（A-D 组）与参考仓文档相同，这里只列**差异与新增项**，流程按工作仓实际代码描述。

---

## 1. 调用入口链

```
Host/Tiling → 每个物理核执行 MegaMoeWave::Process()
  ├─ 阶段1 输入准备（与参考仓一致）：QuantProcessInRank → SendMaskCal → ResetFlagList → SyncAll<false>
  ├─ 共享专家 GMM1（可选）
  ├─ CrossRankSyncInWorldSize（全卡 barrier，读 peermem 的前置条件）
  ├─ 阶段2 ProcessGmmPipeline（工作仓新增的封装，AIC 分支在此构造持久 BlockMmad）
  │       └─ ProcessRoutedExpertWaves()                    @2578
  │             ├─ AIV1 分支：DispatchWaveAndGetEndCursor（@1090，合批 + 一次全 pull）
  │             │        → DispatchExpert（@1063，只推进状态，不再 pull）
  │             │        → MetaInfoCalAndDispatch（@1331，wave 分核版）
  │             ├─ AIC/AIV0 分支：ProcessGmm1Wave（@2409，消费 wave + WaitForUpstreamReady）
  │             ├─ ProcessGmm2Wave（按 wave 终点重放）
  │             └─ ProcessCombineExperts（AIV1 按完整专家合并）
  ├─ SyncAll<true> → 共享专家 GMM2（可选）
  └─ 阶段3：CrossRankSyncInWorldSize + Unpermute
```

---

## 2. 名词对照（工作仓新增 / 改动的部分）

| 名词 | 代码位置 | 实际语义 |
| --- | --- | --- |
| `DISPATCH_WAVE_ROW_COUNT`（=256） | `DispatchWaveAndGetEndCursor` | **新增**：dispatch wave 的合批预算。小专家累计行数 ≤256 合成一个 wave；超过的专家独占一个 wave（不切专家） |
| `GetExpertDispatchRowCount` | @类声明 | **新增**：统一 `maxOutputSize_` 截断口径，wave 行预算与各核行区间共用同一计算 |
| wave 行区间 | `coreWaveRowBegin/End` | **新增**：本核在 wave 内的行区间，`rowsPerCore = ceil(waveRowCount / blockNum_)` 均分得到 |
| `waveBeginExpert/waveEndExpert` | `MetaInfoCalAndDispatch` 参数 | **新增**：dispatch wave 的专家半开区间（边界恒在专家边界） |
| `expertWaveLocalRowBegin` | `MetaInfoCalAndDispatch` | **新增**：专家首行在 wave 内的坐标，用于把核区间换算成专家内区间 |
| shard | 已删除 | 参考仓的 (source rank, 片) 分核单位；工作仓改为按 wave 总行数均分，不再按 rank 切 |
| `persistentBlockMmadContext_` | @类成员 | **新增**：AIC 侧跨 wave 复用同一个 BlockMmad 实例（同事提交 150815168 引入） |
| `allowWeightL2Bypass` | `GroupMatmulWithSwigluQuant` | **新增**：整专家一次消费时权重流式读取、绕过 L2（同事提交 d7e8e76cf 引入） |

其余名词（AIC/AIV0/AIV1、peermem、cumsumInfo、M Group、wave、tile、flag、expertMGroupOffset、
ordinal、batch、ExpertRowCursor、ExpertLoopState、GMMAddrInfo 等）语义与参考仓文档完全一致。

---

## 3. 工作仓 pull 流程

### 3.1 前置（与参考仓一致，未改）

`SendMaskCal` 把本卡 token 写进各目标卡 peermem → `ResetFlagList` 清零 flag → `CrossRankSyncInWorldSize`
全卡 barrier → 才能读远端数据。

### 3.2 AIV1 状态推进（DispatchExpert @1063）

对每个专家（localExpertId），**只做 3 步，不再 pull**：

```
① SendCntCal：读 count → 前缀和（cumsumInfoTensor_）→ 写 expertRevNums → AtomicAdd(sendCntFlag)
② UpdateGroupParams<GMM1>：推进专家状态（M、偏移、flag 基址）
③ UpdateGlobalBuffer<GMM1>：打包 GMMAddrInfo（含 dispatchToGmm1Flag + expertMGroupOffset 偏移）
```

与参考仓的差异：参考仓在这里还调用单专家版 `MetaInfoCalAndDispatch` 全量 pull 整个专家；
工作仓把 pull 全部移入 `DispatchWaveAndGetEndCursor` 的分核路径。

### 3.3 规划 + 一次全 pull（DispatchWaveAndGetEndCursor @1090）

```
while (waveBeginExpert < moeExpertPerRank_):
  ① 合批：从 waveBeginExpert 开始贪心并入小专家，
     累计行数 > DISPATCH_WAVE_ROW_COUNT(256) 断开；
     单专家行数超预算 → 独占一个 wave（不切专家）
  ② MetaInfoCalAndDispatch(waveBeginExpert, waveEndExpert, waveRowCount)  ← 28 个 V1 并行 pull 该 wave
  ③ waveBeginExpert = waveEndExpert
→ 一次调用完成全部 wave 的规划 + pull（每次 MetaInfoCalAndDispatch 处理一个 wave 内的全部专家）；预取路径的第二次调用直接返回末尾游标
```

### 3.4 wave 分核版 MetaInfoCalAndDispatch @1331

1. **空 wave 检查**：`waveRowCount == 0` 直接返回。
2. **行均分**：`rowsPerCore = ceil(waveRowCount / blockNum_)`；本核区间
   `[blockIdx_*rowsPerCore, min((blockIdx_+1)*rowsPerCore, waveRowCount))`；无行则返回（只有
   wave 总行数不足时才出现空转核）。
3. **逐专家求交集**：`expertWaveLocalRowBegin` 维护专家在 wave 内的起点；核区间 ∩ 专家行段
   → 专家内区间 `[coreExpertRowBegin, coreExpertRowEnd)`；零行专家跳过；核区间已耗尽则提前 break。
4. **定位 source rank 区间**：线性扫描 cumsumInfo，找首个前缀 > 核区间起点的 rank（firstRankIdx）
   和首个前缀 ≥ 核区间终点的 rank（lastRankIdx），一次扫描带出首段全局起点。
5. **段循环 pull**：对每个覆盖的 rank 段：
   - 段内 ordinal 区间交集 `[coreMatchOrdinalBegin, coreMatchOrdinalEnd)`；
   - 逐 batch：搬 mask 切片 → `GatherMask` 压缩 → 交集 → `CopyGMToGMPerToken` 从 peermem pull 到 workspace；
   - 段搬完 `MTE3_S` 同步 → 按 256 行拆专家内 wave 槽，`AtomicAdd(重叠行数)`。

### 3.5 GMM1 消费与 flag 握手（与参考仓一致）

`ProcessGmm1Wave` @2409：每轮消费 `mGroupsPerWave_` 个 M Group；每个 tile 计算前
`WaitForUpstreamReady` 轮询 `dispatchToGmm1Flag + waveIdx*64B == min(tileM, m-mLoc)`
（`waveIdx = mLoc / tileM`）；SwiGLU 完成累加 `swigluToGmm2Flag` → GMM2 → 专家 ready → Combine。

**关键**：dispatch 的 wave（专家边界合批）与 GMM1 的 wave（M Group 分组）是两套独立边界，
通过"专家内行数 flag"解耦——pull 多少行，GMM1 就能消费多少行，无需两侧边界一致。

### 3.6 AIV1 主循环节奏（ProcessRoutedExpertWaves @2578）

- 首轮：`DispatchWaveAndGetEndCursor` 一次完成全部 wave 的规划 + pull（逐 wave 处理，每次 MetaInfoCalAndDispatch 处理一个 wave 内的全部专家）；`preparedWaveEndCursor` = 末尾。
- 第二轮起：直接复用末尾游标，**AIV1 两轮内退出主循环**，之后阻塞在 Combine 的
  `WaitGmm2Ready` 上等 GMM1/GMM2 产出。
- AIC/AIV0 仍按 `mGroupsPerWave_` 分轮消费，直到 GMM1 全部完成。

---

## 4. flag 协议总表（两仓相同，写入方细节见上文）

| flag | 写入方 | 读取方 | 粒度 | 语义 |
| --- | --- | --- | --- | --- |
| sendCntFlag | AIV1 SendCntCal（AtomicAdd 1） | AIC/AIV0 UpdateGroupParams | 每 AIC 核 64B | 专家 count 已写入 GM |
| dispatchToGmm1Flag | AIV1 MetaInfoCalAndDispatch（AtomicAdd 行数） | AIC GMM1 WaitForUpstreamReady | 专家内 wave 槽 64B | 该 wave 已 pull 完成的行数 |
| swigluToGmm2Flag | AIV0 SwiGLU（AtomicAdd tile 数） | GMM2 WaitForUpstreamReady | 专家内 wave 槽 64B | 该 wave SwiGLU 完成 |
| gmm2 ready | GMM2 专家最后一段完成 | AIV1 Combine WaitGmm2Ready | 专家 slot | 专家可整体 Combine |

---

## 5. 工作仓 vs 参考仓（pull 侧差异要点）

| 项 | 参考仓 | 工作仓 |
| --- | --- | --- |
| pull 粒度 | 整个专家（一次全量，热点专家 3000 行也一次拉完） | wave 行段（≤256 合批 / 大专家独占） |
| 分核 | rank-shard 固定归属（负载随 rank 段分布） | wave 总行数均分（28 个 V1 满负载） |
| 规划节奏 | 多轮，每轮"当前 wave + 预取下一 wave" | 首调用规划全部 wave 并逐 wave pull 完 |
| flag 就位 | 专家首次进入才累加（后续专家等 AIV1 轮次） | 首轮全部就位，GMM1 的 flag 等待时间大幅缩短（等 flag 的协议不变，只是 flag 更早就满） |
| GMM1 消费 | mGroupsPerWave_ 分组分轮 | 不变（flag 槽对齐一致） |
| wave 边界 | M-Group 边界（可切热点专家） | 专家边界（合批 / 独占） |

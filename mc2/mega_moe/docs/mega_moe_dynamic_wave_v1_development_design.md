# MegaMoe 256-row Segment Wave V1 开发设计

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 适用路径 | MegaMoe Arch35 MTE Wave |
| 主线基线 | `27455755ec4fb4aac40cb6ff3d8ee962139ad5d3` |
| 前一版实现 | `2d15b43e5fb82be30f75d7aab39b1a849a63223d`：完整专家动态扩 Wave |
| 开发分支 | `codex/mega-moe-wave-dynamic-scheduler-20260803` |
| 当前版本 | V1：256-row segment 固定工作量 Wave |
| 外部 ABI | 不变 |
| 设备验证 | Ascend 950 Kernel 编译通过；NPU 正确性和性能待服务器验证 |

本文只描述当前 Segment V1。前一版“只合并冷专家、不切热点专家”的完整专家 V1 已被本设计
替换，不再代表当前 Kernel 行为。

## 2. 最终决策

Wave 不再以专家数作为结束条件，而是以固定数量的 expert-local 256-row chunk 作为结束条件：

```text
ordered chunks -> take Q chunks -> GMM1 -> GMM2 -> next Q chunks
```

其核心语义为：

1. 每个专家按本地行号拆成 `[0,256)`、`[256,512)` 等 chunk；
2. 一个热点专家可以跨多个 Wave；
3. 一个 Wave 可以同时包含前一热点专家的后半段和后续冷专家；
4. 同一专家落入同一 Wave 的连续 chunk 合并成一个 segment GMM，减少 Scalar 初始化；
5. Wave 大小 `Q` 只由 AIC 数、GMM1/GMM2 的 N-tile 数和 producer depth 决定；
6. Host `expertsPerBatch` 不再决定该 Kernel 路径的 Wave 边界；
7. GMM1、GMM2 继续使用现有全核朴素滚动分核。

这里的“与专家解耦”是指**阶段边界与专家解耦**，不是把不同专家拼成一次 GEMM。不同专家
使用不同权重，两个各 128 行的专家仍是两个独立 problem，只是可以进入同一个 Wave。

## 3. 原有实现及其限制

### 3.1 主线固定专家批次

Host 根据平均 token 数计算 `expertsPerBatch`，Kernel 执行：

```text
GMM1(expert batch) -> GMM2(expert batch) -> next expert batch
```

真实路由出现多个冷专家时，当前批次的 GMM1 owner 可能少于 AIC 数。未分到 GMM1 的核提前
进入 GMM2，在 SwiGLU-ready 上等待；后面虽然还有 GMM1 工作，却被专家批次边界挡住。

### 3.2 前一版完整专家 V1

前一版在 Kernel 中累计：

```text
sum(ceil(expertCount[e] / 256))
```

达到 GMM1 owner 目标后才结束 Wave，但仍有两个限制：

- 只能在完整专家之间结束；
- 单个热点专家即使远大于目标，也必须把全部 GMM1 做完后才切 GMM2。

因此它解决了多个冷专家没有合并的问题，但没有真正建立统一的 256-row 工作流。本版删除
Host 最小专家边界，并允许 Wave 终点位于专家内部。

### 3.3 原有 modulo 分核继续保留

每个 GMM problem 的 tile owner 仍是：

```text
localStart = (blockIdx - startBlockIdx + blockNum) % blockNum
loopIdx = localStart; loopIdx < tileNum; loopIdx += blockNum
startBlockIdx = (startBlockIdx + tileNum) % blockNum
```

本次不引入 owner table、固定核组或空间流水核池。Segment 只改变每次交给现有 scheduler 的 M
范围；`startBlockIdx_` 继续跨 segment、专家和 GMM1/GMM2 滚动。

## 4. 设计目标与非目标

### 4.1 目标

- 非尾 Wave 给每个 AIC 至少两个 GMM1 owner；
- 非尾 Wave 给每个 AIC 至少一个 GMM2 owner；
- Wave 边界可以切分热点专家；
- 不重复 Dispatch、SendCntCal、meta 生成或权重读取 problem；
- 前 256 行 Dispatch-ready 后即可启动 GMM1；
- 不增加外部 tensor、tiling 字段、workspace 或全核 barrier；
- GMM2 完成后继续安全地按完整专家 Combine；
- 支持 interleaved/non-interleaved 和 TopkWeightsPrefetch 变体。

### 4.2 非目标

- 不把不同专家的尾行拼成同一个矩阵乘；
- 不根据实际尾 M/N 成本生成加权 owner table；
- 不搜索每个 case 的最优 stride；
- 不建立 GMM1/GMM2 独立物理核池；
- 不做最后小 Wave 的全局预扫描或动态规划重分配；
- 不宣称静态模型等同于 NPU 性能结果。

## 5. Wave 大小公式

定义：

```text
C  = blockNum_                  // 物理 AIC 数
h  = k_                         // GMM2 输出 N
2n = hiddenDim_                 // GMM1 gate+up 总 N
TM = TN = 256
D1 = 2                          // 每核 GMM1 owner 深度
```

### 5.1 GMM1 owner 数

non-interleaved 下 gate/up 在同一个 scheduler owner 上依次执行：

```text
schedulerN1 = hiddenDim / 2
P1 = ceil(schedulerN1 / 256)
```

interleaved 下 scheduler 遍历完整 gate+up 宽度：

```text
schedulerN1 = hiddenDim
P1 = ceil(schedulerN1 / 256)
```

一个 256-row chunk 提供 `P1` 个 GMM1 owner。所需 chunk 数为：

```text
Q1 = ceil(D1 * C / P1)
```

### 5.2 GMM2 owner 数

```text
P2 = ceil(h / 256)
Q2 = ceil(C / P2)
```

### 5.3 固定 Wave 大小

```text
Q = max(Q1, Q2)
```

正常 Wave 固定消费 `Q` 个 chunk，因此满足：

```text
Q * P1 >= 2 * C
Q * P2 >= C
```

若 `Q * P1` 或 `Q * P2` 不能被 `C` 整除，朴素滚动下部分核会比其他核多一个 owner，但不会
再出现正常 Wave 中某些核 owner 为 0 的结构性空转。V1 解决“有没有工作”，不在本版追求
尾 N/M 成本完全相等。

最后剩余 chunk 少于 `Q` 时直接作为尾 Wave。V1 不为消除最后一次 partial Wave 增加全量 count
预扫描；这个尾部属于剩余总工作量限制。

## 6. Chunk、Segment 与 Cursor

### 6.1 Chunk

每个非零专家 `e` 的真实 count 为 `M[e]`，逻辑 chunk 为：

```text
chunk(e, r) = [r, min(r + 256, M[e]))
r = 0, 256, 512, ...
```

专家尾部不足 256 行仍占一个独立 chunk，因为它有自己的权重 problem。

### 6.2 Segment

同一专家、同一 Wave 内的相邻 chunk 合并：

```text
Segment {
    expertId;
    expertRowBegin;
    validRows;
    chunkCount;
}
```

例如 `Q=3`、热点专家有 2048 行时，前三个 chunk 合并为一次 `M=768` 的 GMM，而不是发起三
次 `M=256` GMM。GroupMatmul 内部仍以 256-row M-group 排 tile，owner 序列不变。

### 6.3 ExpertRowCursor

所有角色用二元游标描述阶段边界：

```cpp
ExpertRowCursor {
    uint32_t expertIdx;
    uint32_t rowOffsetInExpert;
};
```

`(e, 0)` 表示专家 `e` 开始，`(e, 512)` 表示专家 `e` 的前 512 行已消费。完成专家后游标规范
化为 `(e + 1, 0)`。

### 6.4 边界算法

```text
processedMGroupCount = 0
while expertIdx < expertCount and processedMGroupCount < Q:
    M = realCount[expertIdx]
    if M == 0:
        expertIdx += 1
        continue

    remainingMGroupCount = Q - processedMGroupCount
    segmentRowCount = min(M - rowOffsetInExpert, remainingMGroupCount * 256)
    process Segment(expertIdx, rowOffsetInExpert, segmentRowCount)
    processedMGroupCount += ceil(segmentRowCount / 256)
    rowOffsetInExpert += segmentRowCount

    if rowOffsetInExpert == M:
        expertIdx += 1
        rowOffsetInExpert = 0
```

Dispatch 与 GMM1 共用 `GetWaveSegmentRowCount` 和同一 `Q`，不需要新增 Wave 边界广播。

## 7. 基于原代码的改造

### 7.1 文件级变化

| 文件 | 变化 |
|---|---|
| `op_kernel/arch35/mega_moe_utils.h` | 增加 chunk target 和 segment 行数纯函数 |
| `op_kernel/arch35/mega_moe_wave.h` | 增加 ExpertRowCursor、地址切片和三阶段 segment 循环 |
| `op_kernel/arch35/mega_moe_base.h` | `GMMAddrInfo` 增加内部 `gmm1TileOffsetInExpert` |
| `op_kernel/arch35/mega_moe_impl.h` | TopkWeightsPrefetch status 使用 expert-local 全局 tile offset |
| `docs/mega_moe_dynamic_wave_v1_reference.py` | 改为 Segment V1 的精确 Scalar 模型 |
| Host tiling | 不修改；`expertsPerBatch` 保留 ABI，但 Wave 路径不再使用它收口 |

### 7.2 Init

Kernel 初始化时计算：

```cpp
gmm1TilesPerMGroup_ = ceil(gmm1SchedulerWidth / 256);
gmm2TilesPerMGroup = ceil(k_ / 256);
mGroupsPerWave_ = max(
    ceil(2 * blockNum_ / gmm1TilesPerMGroup_),
    ceil(blockNum_ / gmm2TilesPerMGroup));
```

热循环不再读取或比较 Host `expertsPerBatch`。

### 7.3 Dispatch

`DispatchExpert` 仍对每个专家只执行一次：

```text
SendCntCal -> UpdateGroupParams(full expert) -> MetaInfoCalAndDispatch(full expert)
```

`DispatchWaveAndGetEndCursor` 只用返回的真实 count 推进专家行游标。若一个热点专家跨多个 Wave，
后续 Wave 复用已经完成的 Dispatch，不重复扫描 mask 或写 meta。

AIV1 仍在 Combine 当前 Wave 前完整预取下一 Wave，保持原有 UB/事件复用约束。

### 7.4 完整专家状态与 segment 副本

每个阶段保留自己的完整 `ExpertLoopState`，只在进入新专家时调用一次 `UpdateGroupParams`。执行
segment 时复制状态并调用 `BuildExpertSegmentState`：

```text
fullExpertState --copy/shift--> expertSegmentState
```

完整状态中的 `M` 不被 segment 覆盖，因此进入下一专家时仍能按完整专家行数正确推进地址。

### 7.5 GMM1

`ProcessGmm1Wave` 从持久化 `gmm1Cursor` 开始消费最多 `Q` 个 chunk：

1. 当前专家 count ready 后立即处理；
2. 同一专家在本 Wave 的连续 chunk 合成一个 segment；
3. Wave 可以在专家内部返回；
4. AIC 执行 GMM1，AIV0 执行配对 SwiGLU/量化；
5. AIV1 不参与 GMM cursor。

没有全专家 count prepass，所以首个 256 行仍可尽早启动。

### 7.6 GMM2

`ProcessGmm2Wave` 从独立的逻辑 `gmm2Cursor` 重放 `[waveBegin, waveEnd)`，segment 范围与 GMM1
完全相同。物理 owner 仍使用共享 `startBlockIdx_`，因此保留用户要求的：

```text
GMM1 最后一个 owner 的下一核 -> GMM2 第一个 owner
```

AIV0 继续推进虚拟 GMM2 cursor，以保持下一 Wave 与 AIC 的核归属相位一致。

### 7.7 Combine

Combine 暂不切成 segment：

- GMM2 只在专家最后一个 segment 完成后调用 `PublishGmm2Ready(expertIdx)`；
- AIV1 只 Combine `waveEnd.expertIdx` 之前已经完整结束的专家；
- Wave 停在热点专家中部时，该专家延迟到后续 Wave 完成后整体 Combine。

这样复用现有每专家 GMM2-ready workspace，不增加容量，也不会让 Combine 读到热点专家尚未写完
的后半段。

## 8. 三角色时序

```text
AIV1:
  Dispatch current Q chunks (each expert only once)
  -> determine End(B)
  -> Dispatch next Q chunks
  -> Combine experts fully completed by B

AIC:
  stream real count
  -> GMM1 segments until End(B)
  -> GMM2 same segments until End(B)
  -> publish ready only for completed experts

AIV0:
  stream the same count
  -> SwiGLU/quant for the same GMM1 segments
  -> virtual GMM2 cursor to End(B)
```

三者使用相同 count、相同 `mGroupsPerWave_`、相同 `GetWaveSegmentRowCount` 和相同专家顺序，
因此独立得到同一个 `(expertIdx, rowOffsetInExpert)` 终点。

## 9. 地址平移证明

对完整专家状态构造行偏移为 `r`、行数为 `m` 的 segment 时：

| 数据 | 平移 |
|---|---:|
| Dispatch data A | `r * K` |
| Dispatch scale A | `r * scaleK` |
| GMM1 temporary output | `r * hiddenDim` |
| SwiGLU data | `r * (hiddenDim / 2)` |
| SwiGLU scale | `r * scaleN` |
| GMM2 output | `r * h` |
| metaInfo/global row | `expertBeforeCnt + r` |
| Dispatch-ready | `expertBase + (r / 256)` slot |
| SwiGLU-ready | `expertBase + (r / 256)` slot |

Weight1、Weight1 scale、Weight2 和 Weight2 scale 只由 `expertIdx` 选择，不随 `r` 平移。

完整专家状态不被修改，所以下一专家的 base offset 仍使用完整 `M`，不会因前一专家被拆成多个
segment 而重复推进或少推进。

## 10. 同步载体核验

### 10.1 Dispatch -> GMM1

现有 workspace 已按：

```text
[expert][ceil(maxOutputSize / 256)][64B cache line]
```

为 Dispatch-ready 预留槽位。Dispatch 本来就按 expert-local 256 行累计 ready 行数；segment
只把 GMM1 flag 基址平移到其首 chunk，不改变 dtype、容量或生命周期。

### 10.2 SwiGLU -> GMM2

MTE Wave 的 SwiGLU-ready 同样按每专家、每 256 行分槽。segment GMM2 的本地 `waveIdx=0`
映射到原专家的 `expertMGroupOffset`，多分组 segment 则继续使用后续连续槽。

### 10.3 TopkWeightsPrefetch tile status

原 status 地址为：

```text
expertIdx * maxTilesPerExpert + localLoopIdx
```

同一热点专家跨 segment 后 `localLoopIdx` 会重新从 0 开始，直接复用会读到旧 tag。本版改为：

```text
expertIdx * maxTilesPerExpert
  + expertMGroupOffset * gmm1TilesPerMGroup
  + localLoopIdx
```

Host 容量按 `ceil(maxOutputSize/256) * ceil(hiddenDim/256)` 上界预留；non-interleaved 的实际
N-tile 数只会更小，因此新偏移不会越过当前专家的 status 区域。

### 10.4 GMM2 -> Combine

GMM2-ready 保持 `[expert][AIC][64B]`，没有改成 per-segment。只有最后 segment 发布一次，每个
AIC 的 `FIX_S` 保证其此前所有 GMM2 GM 写回完成。Combine 仍等待全部 AIC 到达后读取完整专家。

## 11. ABI 与 workspace

本版没有修改：

- `MegaMoeTilingData` 字段或版本；
- 输入输出 tensor shape/dtype；
- `expertTokenNumsOut`；
- Dispatch/SwiGLU/count/GMM2-ready workspace 公式；
- Kernel tiling key；
- 通信和输出布局。

`GMMAddrInfo::gmm1TileOffsetInExpert` 是 Kernel 内部参数，不属于 Host/Kernel ABI。

Host 继续计算和下发 `expertsPerBatch`，以保持其他路径和数据结构兼容；当前 Wave Kernel 不再用
该值决定阶段边界，因此不需要升级 tiling version。

## 12. 示例

设：

```text
C = 24
h = 4096
hiddenDim = 8192
mode = non-interleaved
P1 = 16
P2 = 16
D1 = 2
Q1 = ceil(48 / 16) = 3
Q2 = ceil(24 / 16) = 2
Q = 3
```

真实 count：

```text
[1, 1, 1, 1, 1024, 0, 255]
```

得到：

| Wave | Segment | chunk 数 | GMM1 owner | GMM2 owner |
|---|---|---:|---:|---:|
| 0 | `e0[0,1) + e1[0,1) + e2[0,1)` | 3 | 48 | 48 |
| 1 | `e3[0,1) + e4[0,512)` | 3 | 48 | 48 |
| 2 | `e4[512,1024) + e6[0,255)` | 3 | 48 | 48 |

专家 4 明确被切在两个 Wave 中；它只 Dispatch 一次，第二个 Wave 继续使用同一权重和后半段行
地址。专家 4 到 Wave 2 完成后才发布 Combine-ready。

## 13. 边界场景

| 场景 | 行为 |
|---|---|
| `count == 0` | 不贡献 chunk，游标继续扫描；不发布 GMM2-ready |
| `count == 1/255/256` | 各贡献一个 chunk |
| `count == 257` | 贡献两个 chunk，可被 Wave 边界切为 256 + 1 |
| 单热点专家 | 每 `Q` 个 chunk 切换一次 GMM1/GMM2 |
| 多冷专家 | 连续纳入，直到凑满 `Q` 个独立 chunk |
| Wave 结束于专家尾 | 游标规范化到下一专家 `(e+1,0)` |
| Wave 结束于专家中部 | 保存 `(e,row)`，下一 Wave 从该行继续 |
| 全零尾部 | 最后一次循环扫描至专家末尾，无死循环 |
| 最后不足 Q | 执行 partial Wave，不伪造工作 |

## 14. 性能影响

### 14.1 预期收益

- 冷专家和热点专家使用统一 256-row 工作量标准；
- 正常 Wave 的两级 owner 都至少覆盖全部 AIC；
- GMM1 在切到 GMM2 前保持两层 producer depth；
- 热点专家不会形成无法调节的超大阶段；
- 相邻 chunk 合并为 segment，避免每 256 行都重复完整 problem 初始化。

### 14.2 新增开销

- 每个 segment 的 cursor、`min`、`ceilDiv` 和地址副本计算；
- 热点专家跨 Wave 时会多一次 GMM problem 初始化和权重视图设置；
- Combine 可能等待热点专家最后一个 segment，不能提前发送前半段；
- GMM1/GMM2 权重访问阶段顺序改变，可能影响 L2 行为。

这些 Scalar 开销通常小于 Cube 空转收益，但不能静态保证所有均衡/热点 case 都零回归，必须用
设备 trace 和端到端耗时验证。

## 15. 验证状态

### 15.1 已完成

1. Ascend 950 Kernel 编译：

   ```bash
   bash build.sh --opkernel --ops=mega_moe --soc=ascend950 -j8
   ```

   8 个 MegaMoe Kernel 变体全部生成成功。

2. 精确 Scalar 模型覆盖：

   - Wave cursor 严格前进并覆盖全部专家行；
   - Dispatch/GMM1 边界确定性一致；
   - GMM2 从相同 begin/end 重放出相同 segment；
   - 正常 Wave 同时满足 GMM1 depth 和 GMM2 覆盖；
   - 热点专家跨 Wave，地址区间不重叠、不遗漏；
   - 0/255/256/257、全零尾部和 interleaved 边界。

### 15.2 未完成

当前环境没有执行 NPU case，因此以下不能标记为通过：

- 多 rank 数值正确性；
- segment Dispatch/SwiGLU flag 的设备侧时序；
- TopkWeightsPrefetch status 无提前命中；
- GMM2 最后 segment 到 Combine 的可见性；
- 每核 owner trace 和空闲下降幅度；
- 热点、均衡和冷专家端到端性能。

编译和 Scalar 检查不能替代设备运行成功。

## 16. 服务器验证建议

正确性至少覆盖：

- `count=0/1/255/256/257/Q*256±1`；
- 单热点专家跨 2 个及多个 Wave；
- 热点尾部与后续冷专家落在同一 Wave；
- 中间和最后连续零专家；
- interleaved/non-interleaved；
- TopkWeightsPrefetch 开/关；
- ND/NZ 和当前支持的量化组合；
- 多 EP world size。

Trace 应按 AI block 对齐 AIC/AIV0/AIV1，并核验：

1. 实际 Wave 终点与参考模型 `(expert,row)` 一致；
2. 正常 Wave 的 GMM1/GMM2 owner 数满足公式；
3. 热点专家前后 segment 的 GM 地址连续且不重叠；
4. GMM2 不会在对应 256 行 SwiGLU-ready 前启动；
5. Combine 只在专家最后 segment 完成后启动；
6. 冷专家和热点专家 case 的空闲区间下降；
7. 均衡 case、单热点 case 和端到端最好稳定轮次无明显回归。

参考命令：

```bash
python3 mc2/mega_moe/docs/mega_moe_dynamic_wave_v1_reference.py \
  --counts 1,1,1,1,1024,0,255 \
  --cores 24 \
  --h 4096 \
  --hidden-dim 8192
```

## 17. 回滚与后续

本版没有运行时开关。若设备验证失败，可回滚 Segment V1 提交，恢复完整专家动态扩 Wave。
由于 ABI 和 workspace 未变，不需要迁移数据格式。

后续只有在 trace 证明朴素滚动仍存在明显尾 M/N 不均时，再考虑：

- GMM1/GMM2 独立 cursor；
- 满 chunk 的 stride 搜索；
- 尾 M/N 加权 owner table；
- 尾 Wave 重分配；
- segment 级早 Combine。

这些不属于当前 V1。

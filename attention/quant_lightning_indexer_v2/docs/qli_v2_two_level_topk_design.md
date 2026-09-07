# QuantLightningIndexerV2 两级 TopK（候选块选择）方案设计 — 仅 arch22 / 910b

> **Provenance**: 本方案基于 `ops-transformer` 仓库 `ds41` 分支 commit `32d64c27f`（提交MegaMoeWave模板）的代码调研。
> 参考语义来源: 用户提供的 `select_candidate_blocks` PyTorch 参考实现（口头描述，2026-09-06）。
> 适用范围: `attention/quant_lightning_indexer_v2`（主算子）+ `attention/quant_lightning_indexer_v2_metadata`（配套算子），**仅 910b (arch22) 路径**，arch35 (950) 不在本次范围。

---

## 1. 需求分析

### 1.1 模型侧语义（来自参考实现）

```python
# Level One: 每个query选 topk_blocks 个最高分的块（block_size 个位置为一块）
def select_candidate_blocks(logits, compress_lens, topk_blocks=2048, block_size=8):
    width = logits.size(-1)                                  # S2（压缩后KV长度）
    scores = F.pad(logits, (0, -width % block_size), -inf)   # 尾块补 -inf
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1) # 块内取max → [.., num_blocks]
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size                 # 最后一个未满块
    scores[last] = +inf                                      # pin：最新token所在块无条件保留
    top = scores.topk(min(topk_blocks, num_blocks))
    keep = scatter(top.indices, top.values > -inf)           # 丢弃全 -inf（不可达）块
    return keep.repeat_interleave(block_size, -1)[..., :width]  # 位置级 bool mask

# 调用侧（跨层共享 shared_attn.candidates）:
if self.is_candidate_source:      # source 层：产出候选块
    shared_attn.candidates = select_candidate_blocks(index_score, compress_lens, 2048, 8)
elif self.uses_candidates:        # consumer 层：候选块外 score 置 -inf 后再走原 topk
    index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)
```

关键点提炼：
1. **块定义**: 位置 p 属于块 `floor(p / 8)`，块数 `num_blocks = ceil(S2 / 8)`，S2 为**压缩后**的 key 有效长度。
2. **块得分**: 块内 8 个位置的 score 取 **amax**；不可达位置在 logits 中已是 -inf，不抬升块得分（与 pad 语义一致）。
3. **pin 规则（compress_lens 形态决定，两种）**：
   - **decode / mid-chunk prefill（模型 else 分支，op mask_mode=0）**：`compress_lens = end_pos // ratio` 为标量（= actS2Size），pin 块号 `(actS2Size - 1) // block_size`，batch 级（该 batch 所有行 pin 同一块）；
   - **prefill 首块（模型 start_pos==0 分支，op mask_mode=3）**：`compress_lens = (arange(1, seqlen+1) // ratio)` 为行级 `[S1, 1]`，行 i 的 pin 块号 `(rowValidLen(i) - 1) // block_size`，其中 `rowValidLen(i) = (actS2SizeOrig - actS1Size + i + 1) // cmpRatio`（与 op mask_mode=3 的行级有效长度公式逐行一致：首块时 actS2SizeOrig=actS1Size=seqlen → `(i+1)//ratio`，与模型代码吻合）→ **每行 pin 各自的最后一块**；
   - 保证最新 token 所在块必被选中的语义在两种形态下均成立。
4. **有效槽**: 可达块数不足 `topk_blocks` 时，未被选中的槽位无效（值 `-inf` 被丢弃）。
5. **Level-2 动态 topk（模型 gather 侧，无需 op 改动）**：
   ```python
   topk = min(self.index_topk, end_pos // ratio)                        # 动态 topk = min(sparse_count, actS2Size)
   idxs = index_score.topk(topk, sorted=False).indices.sort(dim=-1)     # 升序
   return torch.where(idxs < compress_lens, idxs + offset, -1)
   ```
   - `topk` 属性为每次调用传入的动态值（[1,2048]），模型传 min 值即可；
   - `where(idxs < compress_lens, ·, -1)` 与 kernel 现有语义**等价**：不可达槽位输出 -1（`InitSortOutBuf` 填 (-inf,-1)，有效长度外沉底）；"topk=sparse_count + 滤 -1" 与 "topk=min 直接输出" 有效索引集合相同；
   - `idxs + offset` 由模型侧完成（910b 不支持 output_idx_offset，无需支持）；
   - **结论：无 kernel/host 改动，仅测试覆盖**（§6.3/§6.5 增加 topk<2048 动态值与 min 等价性用例）。
6. **source 层自身仍做正常 topk**（if/elif 互斥，source 不做 mask 但照常输出 sparse_indices）。
7. **候选块跨层共享**，同一 forward 内 KV 状态不变 → 块划分一致；**跨 step 复用无效**（语义约束）。

### 1.2 现有算子流水线摘要（arch22, 910b）

| 阶段 | 执行单元 | 内容 | 数据位置 |
|---|---|---|---|
| S1 | AIC `ComputeMm1` | Q(256=s1×g, 128) @ K^T(128, s2≤2048) → L0C | L1/L0 |
| S2 | AIC `FixpSToL1` | DEQF16(scale 0.001) + **ReLU** → half [s1g, s2] | L1 `sL1_` |
| S3 | AIC `ComputeWs` | w(s1g) @ ReLU(QK)(s1g, s2) → g 维求和 | L0 |
| S4 | AIC `FixpResToGm` | nz2nd → **score[s1, s2]** float | GM `mm1ResGm`（loop%2 双buffer） |
| S5 | AIV `ProcessVec0` | (w × qScale) 广播 → 供 S3 使用 | GM `vec0OutGm` |
| S6 | AIV `ProcessVec1` | score × kScale → SortAll(降序) → MergeSort 进 `globalTopkUb_`(2048 value/idx pair/s1行) → 行末 ExtractIndex 直出 `sparse_indices` | UB/GM |

> **LD（跨核归约）在 910b 不激活**：`supportFd_` 默认 false 且仅 ASCEND950 分支置 true（metadata aicpu.cpp:329-338, aicpu.h:285）→ `AssignByBlock`（S2 块级跨核切分）被跳过 → 单个 (b, s1) 行的 S2 永不跨核 → kernel `isNeedLD` 恒 false，`ProcessLD` 的 QLD_V2 元数据全 0。**本方案不考虑 LD 路径**（kernel 中的 LD 代码保留不动，仅不激活）。

结论：**`index_score`（参考实现中的 logits）= S4 输出 × kScale（S6 第一步之后）**，按 (s1行, s2 tile ≤2048) 流式产生，且每行 topk 结果在本核内独立完成。两级 TopK 的插入点在 S6（ProcessVec1）内部。

### 1.3 输入输出定义（新增部分）

**主算子 QuantLightningIndexerV2 新增（命名与模式编号按用户 2026-09-06 决策）**：

| 项目 | 名称 | 方向 | 类型/Shape | 约束 |
|---|---|---|---|---|
| 输入 | `candidate_topk_index` | 可选输入 | INT32, `[B, S1, N2, 2048]`（BSND） | `candidate_mode=2` 时必传；值域 `[0, numBlocks)` 或 `-1`（无效槽） |
| 输出 | `candidate_topk_index` | 可选输出 | INT32, 同上 | `candidate_mode=1` 时输出；无序，槽位语义：块号或 -1 |
| 输出 | `sparse_indices`（即 topk_index） | 既有输出 | 不变 | mode=1: 全量 score 的 topk；mode=2: **从 candidate_topk_index 展开的 ≤16384 个候选位置中再挑 topk 个**；mode=3: 现网行为 |
| 属性 | `candidate_mode` | 可选属性 | INT, 默认 **3** | **1**=source（is_candidate_source，输出候选块索引 + 照常输出 topk_index）；**2**=consumer（use_candidate，输入候选块索引，在候选内选 topk）；**3**=关闭（现网行为，默认值保证既有调用零影响） |
| 属性 | `candidate_topk_blocks` | 可选属性 | INT, 默认 2048 | **当前仅支持 2048**（属性保留、host 校验限定 2048，未来扩展只放宽校验不改接口；受 BASE_TOPK=2048 的 sort/merge 结构约束，910b）|
| 属性 | `candidate_block_size` | 可选属性 | INT, 默认 8 | 范围 [1, 64] 且为 2 的幂（向量 reduce 实现约束）；**source 与 consumer 必须一致** |

**设计决策 — 输出索引而非 bool（用户决策"不再输出 bool，直接输出 topk index"）**：
- `candidate_topk_index` 为**块级索引**（2048 个 int32/行，值=块号），展开后等价于 2048×8=16384 个位置级索引；相比位置级索引（16384 个 int32=64KB/行）**GM 内存压缩 8×**，相比 bool mask `[B,S1,S2]`（prefill 大序列可达 512MB）更是量级差异；
- mode=2 消费时按 `position ∈ [8×blk, 8×blk+8)` 展开，天然还原块内连续 8 位置；
- 既有输出 `sparse_indices` 名称不变（aclnn 兼容），文档语境中的 topk_index 即指它。

**int 块索引 vs bool mask 的权衡记录（2026-09-06，结论：本场景无实质不方便）**：
| 维度 | bool mask | int 块索引（本方案） |
|---|---|---|
| 算子外消费 | 直接可用 | 需先展开（届时在 torch 封装层提供 expand 工具即可，不必改算子接口） |
| 接口约定 | 无 | -1=无效槽、槽位无序（topk 序）、块号×8=起始位置，需文档化（README 约束项） |
| mode=2 kernel 实现 | 读 tile mask 段 + Select，简单 | membership（compare-OR 分批）+ 展开，复杂一档（S4a 已设计） |
| 调试/比对直观性 | 高 | 低（dump 为块号列表；比对已集合化 §6.2） |
| GM 内存/带宽 | S2 字节/行（最坏 512MB） | 8KB/行，**省 8×**（决定性理由） |
语义完备性说明：块内有效性由原 score 兜底（mask_mode=3 下行内不可达位置 score 本为 -inf），mode=2 的 masked score 与 bool 语义严格等价。

**metadata 算子：不改接口**（论证见 §3.2）。

### 1.4 数学语义（新增部分的形式化）

对每个 batch b、query 行 s1（n2=1, 910b 上 kHeadNum=1）：

```
score(b, s1, :)  ∈ R^{S2}            # 现有 S4×kScale 结果，不可达位置语义为 -inf
numBlocks(b)     = ceil(actS2Size(b) / block_size)
blkScore(b, s1, j) = max_{8j ≤ p < min(8j+8, S2)} score(b, s1, p)     # 尾块不足8位置按 -inf pad

# pin 块号（行级，随 mask_mode 取行级/批次级有效长度）:
rowValidLen(b, i) = (actS2SizeOrig(b) - actS1Size(b) + i + 1) / cmpRatio   # mask_mode=3（行级）
rowValidLen(b, i) = actS2Size(b)                                          # mask_mode=0（batch 级，各行相同）
lastBlk(b, i)     = (rowValidLen(b, i) - 1) / block_size                 # 整除
blkScore(b, i, lastBlk(b, i)) = +inf                                     # pin：每行 pin 自己的最后一块

mode=1 (source): candidateTopkIndex(b, i, :) = topK_blocks 个 blkScore 最大的块号
        （并列值 tie-break 不保证与 PyTorch 一致，见 §6.2；blkScore=-inf 的槽输出 -1）
        sparse_indices(b, i, :) = topK(score)                            # 与现有逻辑完全一致（不 mask）
mode=2 (consumer): score'(b, i, p) = score(b, i, p)      若 floor(p/8) ∈ candidateTopkIndex(b, i, :)
                                          -inf          否则
        # 等价视角：sparse_indices 是"从 candidate_topk_index 展开的 ≤16384 个候选位置中再挑 topk 个"
        sparse_indices(b, i, :) = topK(score')                          # 复用现有 sort/merge 管线
mode=3 (off):     现网行为，candidate 功能整体关闭（默认值，向后兼容）
```

dtype/精度约束：score 全程 fp32（现有路径不变）；块化 amax 与 -inf/+inf 处理均在 fp32 域（`NEG_INF=0xFF800000` 已有，新增 `POS_INF=0x7F800000`）。

### 1.5 验收标准

**功能验收**（mode=3 回归为正确性门禁，非性能）：
- candidate_mode ∈ {1, 2, 3}；mode=3 输出与现网 bit 级一致（回归，默认值保证既有调用零影响）。
- **layout_q 仅 BSND**（用户指定本轮范围；layout_k = PA_BBND，910b 仅支持 PA）。TND 既有路径不动、本轮不覆盖不回归。
- mask_mode ∈ {0, 3}；**cmp_ratio ∈ {1, 2}**（用户指定重点场景，见 §6.6）。
- **q_head_num 仅测 32**（用户指定：所有 candidate 用例统一 g=32，不再覆盖 64；A1/A2 改造后 g=64 逻辑路径不变，由既有 910b 用例回归兜底）。
- **topk 动态值**：topk ∈ [1, 2048] 任意值，含 topk = min(index_topk, actS2Size) 语义（等价性验证，无代码改动）。
- S2 使 numBlocks < / = / > topk_blocks 三类；actS2Size 非 block_size 整除（尾块）。
- decode（S1=1）与 prefill（S1>1）。
- 910b 约束继承：quant_mode=2（INT8 + fp16 scale/weight）、无 return_value/output_idx_offset。

**精度验收**：
- `candidate_topk_index`：与参考实现的**选中块集合**一致（无序集合比较 + -1 槽位数一致）；允许 tie 场景按 §6.2 策略仲裁。
- mode=2 `sparse_indices`：与 "参考实现 mask 后走现有 golden topk" 一致（复用现有 compare 框架与 tie 容差策略）。
- mode=1 `sparse_indices`：与现网 golden 一致（不应受块级计算影响——同一 Vec1 内两套独立 sort buffer）。

> 本次交付不含性能验收（用户明确性能暂不关注），但设计保留向量化的块化/mask 实现路径（§3.3/§3.5），避免后续性能优化时返工。

**稳定性验收**：多 seed（≥3）、边界 shape（S2%8≠0、actS2Size=1、numBlocks=1、topk_blocks>numBlocks）、B>1 变长 batch。

---

## 2. 算法拆解（在现有 6 阶段上的增量）

| # | 阶段 | 现状 | 本次变更 | 执行单元 |
|---|---|---|---|---|
| 0 | 输入解析 | layout/actual seqlen/metadata 分核 | consumer(mode=2): 解析 `candidate_topk_index` GM 指针；新增 TilingData 字段下发 | Scalar |
| 1 | 预处理 | Vec0: w×qScale 广播 | 不变 | Vector |
| 2 | MatMul1 (QK) | AIC | 不变 | Cube |
| 3 | score 后处理 | Fixp DEQF16+ReLU → Ws g-sum → GM | 不变 | Cube |
| 4 | Vec1 score 生成 | mm1Res×kScale | **mode=2: 追加候选块 mask 生成 + Select 置 -inf**（插在 Mul(kScale) 之后、SortAll 之前） | Vector |
| 5 | 归约/块化 | —（无） | **mode=1: 块化 amax(8:1) + pin 尾块置 +inf**（与位置级 SortAll 并行，规模 1/8） | Vector |
| 6 | 选择/TopK + 输出 | SortAll+MergeSort → globalTopkUb_(2048) → 行末 ExtractIndex 直出 | **mode=1: 镜像维护 globalBlockTopkUb_(2048)，行末 ExtractIndex 直出 candidate_topk_index GM**；生命周期/重置点与位置级完全同步 | Vector |
| — | 无效清理 | CleanInvalidOutput 填 -1（sparse_indices） | **mode=1: 所有填 -1 的路径同步对 candidate_topk_index 填 -1**（DealActSeqLenIsZero / BSND 无效 S1——BSND 范围内共 2 处，见 §3.3 检查清单；TND padding 路径本轮不适用） | Vector |

阶段信息表（新增阶段细目）：

**S4a（mode=2, consumer mask）**：
- 输入：`candidate_topk_index` GM 行 `[topk_blocks]` int32（8KB）、当前 tile score UB `[tileLen]` fp32
- 输出：masked score UB（原地）
- 计算单元：Vector（compare-OR 分批，见 §3.5）
- 同步：无跨核（每行候选集独立，行内自洽）

**S5a（mode=1, 块化）**：
- 输入：score UB `[tileLen]`（kScale 相乘后）
- 输出：`blkScore UB [tileBlkNum=tileLen/8]` fp32 + `blkIdx UB` int32（基址 = tileS2Base/8）
- pin：当前行 pin 块 `lastBlk = (rowValidLen - 1) / block_size`（行级，`rowValidLen` 即 Vec1 已有的 `cuRealAcSeq`，mask_mode=0 时各行相同退化为 batch 级）；tile 包含该块时置 +inf；tile 内超出行有效长度（mask_mode=3）的块值为 -inf（pad 语义）
- 计算单元：Vector（8:1 strided ReduceMax + 尾块 -inf 对齐）

**S6a（mode=1, 块级 sort/merge + 直出）**：镜像 S6 结构，`SortAll(blkNumAligned)` + `MergeSort(globalBlockTopkUb_[row], topk_blocks, ...)` + 行末 `ExtractIndex` → `candidate_topk_index GM`；`AlignS2` 复用（块数对齐到 32/128/512 粒度）。

---

## 3. Host Tiling 设计（arch22）

### 3.1 维度建模

| 维度 | 含义 | 现值/来源 |
|---|---|---|
| B / N2 / G | batch / kv头(=1) / query头组 | **G = 32（本轮唯一测试规格）** |
| S1 | query 长度 | 不变 |
| S2 | 压缩后 key 长度（actS2Size） | 不变 |
| **numBlocks** | `ceil(actS2Size / block_size)` | **新增，随 batch 变长** |
| **tileBlkNum** | 单 tile 内块数 = `s2BaseSize / block_size = 256` | **新增，常量（block_size=8 时）** |
| topk | 位置级 topk | 动态 [1, 2048]，含 min(index_topk, actS2Size) 语义（无代码改动） |
| **topkBlocks** | 块级 topk | **固定 2048**（BASE_TOPK 结构复用；属性保留，host 校验限定，未来扩展只放宽校验） |

> **q_head_num=32 的支持方式（参照 v1 `quant_lightning_indexer` 实现，全量关键词搜索适配点）**：
> - **v1 与 v2 的关键差异**：v1 arch22 kernel 固定 `S1_BASE_SIZE=4`、推导 `mBaseSize = s1BaseSize × gSize`（g=32 → mBase=128）；v2 现状反之——固定 `M_BASE_SIZE=256`、推导 `s1BaseSize = 256/gSize`（g=32 → s1BaseSize=8，按行 UB 翻倍）。
> - **一致性论证**：v2 的 metadata aicpu 本就是 v1 风格（`s1BaseSize_=4` 默认、`mBaseSize_ = s1BaseSize_ × groupSize_`，aicpu.h:307 / aicpu.cpp:247）——g=64 时 4×64=256 与 v2 kernel 固定值恰好重合，**g=32 时若 kernel 不改，分核区间（128 基准）与 kernel 循环（256 基准）失配**。因此本修改同时是 g=32 正确性的必要条件。
> - **收益**：s1BaseSize 恒为 4 → sortOutBuf_/candidate globalBlockTopkUb_ 等 UB 与 workspace 全部不变，§3.3 的 g=32 UB 紧张场景（原 R3）不存在。
> - **验证范围（2026-09-07 用户确认）**：candidate 测试统一 **q_head_num=32**（g=64 由既有回归兜底，不在本轮范围）；R3 风险随之收窄为"推导变更正确性"而非 UB 容量。
>
> **适配点全量清单（v1↔v2 归一化 diff + 关键词扫描结论）**：
>
> | # | 位置 | v1 实现 | v2 现状 | 本次动作 |
> |---|---|---|---|---|
> | A1 | kernel `InitTilingData`（v2 kernel_arch22.h:191-193） | `s1BaseSize=4` 固定；`mBaseSize = s1BaseSize × gSize`（v1:171-175） | `mBaseSize=256` 固定；`s1BaseSize = 256/gSize` | **改**：对齐 v1 推导 |
> | A2 | host `GetGSize` 910b 分支（v2 tiling.cpp:857-860） | `gSize > 64` 才拒绝（v1:700-705，即 32 合法） | `gSize != 64` 拒绝 | **改**：放宽允许 32（复用 v2 tiling.h:87 既有常量 `G_SIZE_LIMIT_32_950`） |
> | A3 | cube L0/L1 切分（`ComputeMm1`） | 按 S1 维切：`s1L0LoopCnt = CeilDiv(actM/g, s1Base/2)`，L0 子块 `gSize×s1Base/2`（随 g 缩放，g=32 → 64 行/次） | 固定粒度：`CeilDiv(actM, S1G_BASIC_BLOCK_L0=128)`，L0 子块恒 128 行 | **不改**：g=32 时 actMBaseSize=128 → 单次 L0 循环，mExtension=CeilAlign(128,16)=128 ≤ L1 容量，v2 现有逻辑自动适配（两种切分等价可行，保持 v2 风格改动最小） |
> | A4 | cube `ComputeWs/LoadSToL0b/LoadWeightToL0a/FixpResToGm` | 全部 `gSize` 参数化（`k=gSize`、`s1gOffset += gSize`、`repeatTimes=CeilDiv(gSize,16)`） | 同样已 gSize 参数化 | **不改**（diff 确认一致） |
> | A5 | Vec0 `cuProcEleNum`（v2 service_vector:277） | `CeilAlign(cuS1ProcNum × gSize, 32)`（v1:289，UB 对齐防御） | 无对齐（g=64 时恒为 32 倍数而省略） | **建议同步 v1 的 CeilAlign**：g=32 时 4×32/3×32/2×32/1×32=128/96/64/32 恰好均为 32 倍数（数学上不改动也安全），但对齐写法防御未来 g 非 2 的幂 |
> | A6 | host workspace（v2 DoTiling arch22 分支） | `QliCalcWorkspaceSize` 用保守上界常量（mBase=512×s2Base=512，g 无关） | `M_BASE_SIZE(256) × S2_BASE_SIZE(2048)` 上界 | **不改**：256 是 g∈{32,64} 的 mBaseSize 上界（4×64），g=32 的 128 被覆盖 |
> | A7 | kernel Init workspace 布局（v2 kernel:463-478） | — | `mm1Res: 2×s1BaseSize×s2BaseSize`（s1BaseSize 恒 4 不变）；`weightMemSize: 16×mBaseSize×2`（mBaseSize=128 → 减半，变小安全） | **不改**：布局全部由 constInfo 推导，A1 改后自适应 |
> | A8 | metadata aicpu | — | 已是 v1 风格（`mBaseSize_ = 4×groupSize_`） | **不改**（g=32 与改后 kernel 基准一致） |
> | A9 | `GetS2BaseBlockNumOnMask/GetTotalBaseBlockNum/CalcGS1LoopParams` 等 | `s1BaseSize/gSize/mBaseSize` 参数化 | 同 | **不改**（A1 后自动正确） |
> | A10 | UT（tests/ut/op_host/arch22） | v1 有 arch22 tiling 单测 | v2 有同款 | **补**：g=32 的 tiling/infershape 单测 case |

### 3.2 多核切分 —— 不变性论证

- 910b 上 metadata 仅做 batch 级（`AssignByBatch`）与整行级（`AssignByRow`）切分，`AssignByBlock`（S2 块级）因 `supportFd_=false` 跳过 → 每个 (b, s1) 行整体落在一个核上，topk 天然单核完成，无需跨核归约。
- 块级 topk 与位置级 topk 在**同一 Vec1 调用**内、同一 (bN2, gS1, s2) 任务块上执行，共享现有 metadata 分核（QLI_V2_* AIC 段）。
- 负载增量：Vec1 每 tile 增加 `O(tileLen/8)` 的块化 + 块级 sort/merge（≈位置级 sort 开销的 1/8）→ 仅放大每块耗时，**不改变任务块数量与划分粒度** → metadata 的分核算法、输出布局、协议（1024 int32, AIC 36×8 + AIV 72×8）均不变，QLD_V2 段在 910b 本就全 0。

### 3.3 UB Buffer 规划（AIV，910b UB 256KB）

采用 §3.1 的 v1 对齐方案后，s1BaseSize 恒为 4（g∈{32,64} 通用），UB 不随 gSize 变化：

现有（`InitBuffers`）：inQueue 32KB + outQueue 8KB + indexBuf 8KB + tmpBuf 64KB + sortOutBuf 32KB = 144KB。

**mode=1 增量**：
| buffer | 元素数 | 字节 | 生命周期 | 复用 |
|---|---|---|---|---|
| globalBlockTopkUb_ | CeilDiv(s1BaseSize,2) × topkBlocks × 2 (fp32 pair) | 32KB | 整个 gS1 基本块 | 新增 TBuf，重置点与 globalTopkUb_ 同步 |
| 块化临时（blkScore+blkIdx+对齐 pad） | 2×s2BaseSize/8 | 2KB | 单 tile | 复用 tmpBuf 头部 |

合计 ~178KB < 256KB ✓（编译期从 PlatformInfo ubSize 校验，**禁止**硬编码 256KB 常量）。

**mode=2 增量**（均复用 tmpBuf）：
| buffer | 元素数 | 字节 | 复用 |
|---|---|---|---|
| candListUb | topkBlocks (int32) | 8KB | 复用 tmpBuf |
| blkMaskUb | tileBlkNum (fp32/bool) + 位置展开 tileLen | 10KB | 复用 tmpBuf |
| compare 批缓冲 | tileBlkNum × 32 | 32KB | 复用 tmpBuf |

**输出清理检查清单（mode=1 填 -1 的路径，BSND 范围内共 2 条）**：
1. `DealActSeqLenIsZero`（actS1Size=0 或 actS2Size=0，BSND 分支）
2. BSND 无效 S1 尾部（qSeqSize > actS1Size）

（causal 下 actS1Size > actS2SizeOrig 的行在 mask_mode=3 时按行有效长度自然处理为块 -inf，非独立清理路径；TND padding 路径本轮不适用，见 §5。）

### 3.4 Workspace 规划（GM）—— 无增量

现有布局（arch22，per AIC 核）不变：
```
[0]                    mm1ResGm      : 2 × s1BaseSize × s2BaseSize × 4B
[+off1]                vec0OutGm     : 16 × mBaseSize × 2 × 2B
[+off2]                vec1ResGm(LD) : s1BaseSize × 2 × 2 × BASE_TOPK × 4B   ← V1_DECODE 区（910b 不激活，保留）
```

**结论：candidate 功能不新增任何 workspace。** 理由：块级 topk 在单核 Vec1 内完成并直出（§3.2），无跨核中间结果落盘；mode=2 的候选集每 tile 从 GM 直接读入 UB。tiling.cpp 的 workspaceSize 计算不变（仍需随本方案回归确认无隐性依赖）。

### 3.5 分支策略（host 集中，kernel 单点判断）

| 分支 | 条件 | 行为 |
|---|---|---|
| mode=3 | candidate_mode=3 | 现有路径原样（默认值，向后兼容；模板内 if 包裹新增段） |
| mode=1 | candidate_mode=1 | S5a/S6a 全开（source） |
| mode=2 | candidate_mode=2 | S4a mask 生效（consumer） |

mode=2 的 membership 判定（候选列表 → tile 内 256 块 bool，向量化无标量循环）：
- `blkMask[i] = OR_j (candList[j] == tileBlkBase + i)`，compare-OR 分批：每批 32 候选 × 256 块（32KB fp32 view），共 topkBlocks/32 = 64 批/行；随后 `repeat_interleave(8)` 展开 + `Select` 置 -inf。

分支条件全部由 host tiling 写入 TilingData 字段（含 `candidateMode`、`candidateTopkBlocks`、`candidateBlockSize`），kernel 内不重复推导。

> 注：`numBlocks ≤ topkBlocks` 时 mask 恒为全选（等价于不 mask），实现上**不做专门 fast-path 分支**（性能暂不关注，减少分支与测试组合）；该等价性仍作为正确性用例覆盖（§6.3）。

---

## 4. 契约设计

### 4.1 TilingData（QLIV2TilingData 尾部追加，4B 对齐）

```cpp
TILING_DATA_FIELD_DEF(uint32_t, candidateMode)        // 1=source / 2=consumer / 3=off(默认)
TILING_DATA_FIELD_DEF(uint32_t, candidateTopkBlocks)  // ≤2048
TILING_DATA_FIELD_DEF(uint32_t, candidateBlockSize)   // 默认8，2的幂
```
host 写入 ↔ kernel 消费对照表随本方案落入 docs（维护接口式追踪）。字段分组注释：`// ---- candidate (two-level topk) ----`。

### 4.2 TilingKey —— 不变

理由（遵循"只编码影响模板实例的维度"）：candidate 路径是纯 Vector 数据通路，不改变 MatMul 形状/dtype/layout/模板实例，仅 kernel 内运行时分支；编码进 key 会造成 ×3 模板实例化与编译时间膨胀，无收益。

### 4.3 算子原型 / Infershape / aclnn / torch

- `quant_lightning_indexer_v2_def.cpp`：新增 optional input `candidate_topk_index`(INT32, ND)、optional output `candidate_topk_index`(INT32, ND)、三个可选属性；仅 `ascend910b` 配置声明（`aicore_config`），950 配置不动。
- `quant_lightning_indexer_v2_infershape.cpp`：`candidate_mode=1` 时输出 shape = sparse_indices 前缀 + 末维 `candidate_topk_blocks`（**仅 BSND 分支**，q 只考虑 BSND）；否则末维 0。InferDataType 补 SetOutputDataType(2, DT_INT32)。
- aclnn 两段式接口（`aclnnQuantLightningIndexerV2`）：GetWorkspaceSize/执行签名追加 optional `candidateBlocks` 输入/输出指针与 3 个属性，向后兼容（指针可空）。
- torch schema（**已定方案 b——新增 overload，旧接口不动**）：
  ```python
  # 新增变体（source/consumer 统一入口，返回固定三元组）
  quant_lightning_indexer.candidate(
      query, key, weights, q_descale, k_descale, topk, quant_mode, *,
      Tensor? candidate_topk_index=None,        # mode=2 时必传
      int candidate_mode=3,                 # 1=source / 2=consumer / 3=off(默认)
      int candidate_topk_blocks=2048,
      int candidate_block_size=8,
      ...其余参数同旧接口...) -> (Tensor sparse_indices, Tensor sparse_values, Tensor candidate_topk_index)
  # 非 source 模式第三元返回 shape (0,) 占位，元信息（block_size 等）随返回对象附带的 sidecar 属性传递
  ```
  - 旧 `quant_lightning_indexer` schema 二元组完全不变，现网调用零影响；
  - 模型侧 source/consumer 层统一走 `.candidate` 入口，按 candidate_mode 区分行为；
  - block_size 一致性断言（R4）在 python 封装层实现：shared 对象携带 `candidate_block_size` 元信息，consumer 调用时校验。

### 4.4 metadata 算子（`quant_lightning_indexer_v2_metadata`）

**结论：不改**。分核算法/输出协议/属性均不变（论证见 §3.2；910b 无 FD/LD 路径，QLD_V2 段维持全 0）。配套约束：
- 主算子 mode≠0 时 metadata 照常传入；
- 文档（README）补充：candidate 相关属性不参与分核，因负载模型未变。

---

## 5. 与既有机制的交互确认

### 5.0 mask 计算结论：不修改、不新增 mask mode

现有 mask_mode 与模型代码分支**逐行数学等价**，无需任何 mask 侧改动。

**关键澄清**：模型 if/else 是**推理时间步**的分支，不是一次调用内的分支——每次算子调用仍只有一种 mask_mode：

```
prefill 首块 (start_pos==0):  调用 QLI 1 次 (S1=seqlen)  → mask_mode=3, 行级 pin
decode step  (start_pos>0):   每步调用 QLI 1 次 (S1=1)   → mask_mode=0, batch 级 pin
两级 TopK 时间线:
  prefill 首块: QLI(mode=1, mask_mode=3) → 产出 candidates 存 shared_attn
  decode step:  QLI(mode=1/2, mask_mode=0) → source 更新候选 / consumer 消费候选
```

| 模型分支 | 模型行为 | op 对应 | 等价性论证 |
|---|---|---|---|
| `start_pos == 0`（prefill 首块） | `compress_lens[i] = (i+1)//ratio`（`[S1,1]` 行级），`p ≥ compress_lens[i]` 置 -inf（模型侧 masked_fill_；score 下沉算子后由 mask_mode=3 承担） | mask_mode=3 | op 行级有效长度 `(actS2SizeOrig - actS1Size + i + 1)/cmpRatio` 在首块时 `actS2SizeOrig = actS1Size = seqlen` → 退化为 `(i+1)//ratio`，与模型逐行恒等（golden `create_mask`、kernel `cuRealAcSeq`、模型三方公式一致） |
| `else`（decode） | 标量 `compress_lens = end_pos//ratio`，不 mask | mask_mode=0 | op 不 mask，天然匹配；pin 用 batch 级 `(actS2Size-1)/block_size` |

新增的行级 pin（S5a）**复用同一个 `cuRealAcSeq`**，不构成新 mask——pin 与 mask 同源，不会出现 pin 块落在被 mask 区域的矛盾。

**开放问题 O3**：`else` 分支为标量 compress_lens，意味着 mid-chunk prefill（start_pos≠0 且 S1>1）时模型**不做行内因果 mask**。若该调用形态实际不存在，测试矩阵中 "mid-chunk prefill×mode0" 用例删除；若存在，op 照实不 mask（复现模型行为），同样无需改动。待用户确认。

> ⚠ **O3 状态（标红保留）**：**此问题未关闭，不允许随本需求悄悄消化。** 当前决策为"暂不测试"——mid-chunk prefill×mode0 用例从本轮测试范围剔除（用例定义保留在矩阵，注释标明暂不执行），但设计上模型语义为"prefill 不做行内 mask"，**与常规认知相反**。在模型侧确认该调用形态（chunked prefill / 混布场景是否走 else 分支）之前：
> 1. 禁止有人"顺手"在算子里给 mode0+S1>1 补行级 mask（那是语义变更，不是 bug fix）；
> 2. 精度测试若碰到 mode0+S1>1 的数据，比对结论一律以"模型不 mask"的 golden 为准；
> 3. 本问题最终去向二选一：确认形态不存在 → 删除用例；确认存在 → 启用用例并补 golden 说明。

| 机制 | 交互 | 结论 |
|---|---|---|
| LD / ProcessDecode | 910b 不激活（`supportFd_` 仅 950 置 true，aicpu.cpp:329-338） | **不适用，方案不含 LD 路径**；kernel 现有 LD 代码与 V1_DECODE workspace 保留不动 |
| mask_mode=0（decode / mid-chunk prefill） | 无行 mask，全行有效长度 = actS2Size；pin 为 batch 级 `(actS2Size-1)/block_size`（对应模型 else 分支标量 `compress_lens = end_pos//ratio`） | 兼容 |
| mask_mode=3（prefill 首块） | 行级有效长度 `rowValidLen(i) = (actS2SizeOrig-actS1Size+i+1)/cmpRatio`（块化时行尾块按该行有效长度 pad -inf，等价模型 `masked_fill_`）；**pin 为行级** `(rowValidLen(i)-1)/block_size`（对应模型 `[S1,1]` 行级 compress_lens） | 兼容，S5a 内处理；行级 pin 与行级 mask 公式同源（Vec1 `cuRealAcSeq`），天然一致 |
| cmp_ratio 压缩 | 块划分基于压缩后 actS2Size（与 Python 的 S2=logits.size(-1) 一致） | 兼容 |
| Vec1 双缓冲预取（`LI_QUANT_PRELOAD_TASK_CACHE_SIZE=2`） | 块级 sort 结果随 runInfo 双缓冲流转，重置点与位置级 `globalTopkUb_` 完全同步（`info.s2Idx==0` 时重置、行末直出后重置） | 设计强约束：两套 buffer 的重置/输出点成对出现，UT 加不变量断言 |
| TND padding（sequsedQ < cuSeqlensQ） | padding 行的 candidate_topk_index 需填 -1（同 sparse_indices 无效值路径） | **不适用**（仅 TND 布局存在该路径，本轮 q 只考虑 BSND）；若未来放开 TND 需同步补上 |
| Sort32/MrgSort 降序假设 | 假设：降序（top-k 语义自洽，-inf 沉底/初始填充行为验证） | 实现期以 API 文档+单测确认（验证点 V1） |

---

## 6. 验证计划

### 6.1 Golden 与 provenance（强制）

- **参考实现独立重建**：golden 不得只搬 Python 片段——用 numpy 按 §1.4 数学定义逐行重建 `select_candidate_blocks`（pad→amax→pin→topk→scatter→expand），与用户 PyTorch 实现交叉验证后再作为 ground truth。
- **sidecar 强制字段**（`xxx.ref.json`，遵循全局 AGENTS 规则）：分支 `ds41` + commit `32d64c27f`、candidate 参数（mode/topk_blocks/block_size/mask_mode/cmp_ratio/layout）、随机种子、生成脚本路径、机器（910b 节点）、时间。
- **使用前核验**：比对结果与预期矛盾时，第一嫌疑人是 golden 本身（先独立复算，再怀疑 kernel）。

### 6.2 精度比对策略（tie 容差）

- 块级 topk 并列值（blkScore 相等）时 NPU 与 PyTorch 的 tie-break 不保证一致 → `candidate_topk_index` 采用**集合比较**：`set(有效块号)` 相等 + `-1` 槽数一致即 PASS；
- 构造 tie 密集用例（常值 score）单独验证 tie 不影响**集合**正确性；
- mode=2 的 sparse_indices 沿用现有 `result_compare_method.py` 的比对与容差框架。

### 6.3 测试矩阵（每格至少 1 用例，mode=3 抽样回归）

| 维度 | 取值 |
|---|---|
| candidate_mode | 0 / 1 / 2 |
| cmp_ratio | **1 / 2**（用户指定两类场景） |
| layout_q × layout_k | BSND×PA（q 仅 BSND，用户指定本轮范围） |
| q_head_num | **仅 32**（用户指定；全部 candidate 用例统一 g=32，A1 推导下 mBase=128，重点验分核对齐） |
| topk（动态） | 2048 / **min(index_topk, actS2Size) 场景**（topk<2048，验证 -1 padding 与 min 等价性） |
| 阶段 × mask_mode（按模型分支联动） | decode(S1=1)×mode0；prefill 首块(S1>1)×mode3（行级 pin）；~~mid-chunk prefill×mode0~~（**暂不测试，问题保留见 R5/§5.0 O3，问题解决后补测**） |
| numBlocks vs topkBlocks | <, =, > |
| S2 对齐 | actS2Size%8=0 / ≠0（含 actS2Size=1） |
| B | 1 / 4（变长 batch） |
| topk_blocks | 固定 2048（O2 已定，非默认值用例删除） |

重点组合：mode1+尾块不对齐+causal、**mode1+prefill 首块（验证行级 pin：不同行 pin 不同块）**、mode2+causal、`numBlocks ≤ topkBlocks` 时 mode=2 与 mode=3 的等价性（mask 全选）、**全部用例统一 g=32（mBase=128 分核对齐，A1 推导的本命规格）**、**topk=min(2048, actS2Size) 与 topk=2048+滤-1 的等价性**、mode1+多核（B>1 触发行间切分，验证各核独立直出正确）、cmp_ratio=2 下块划分与 pin（actS2Size 为压缩后长度，cmp_residual_k 参与原始长度还原）。

### 6.4 分阶段验证（中间结果可 dump）

开发期在 `candidate_topk_index` 输出之外保留 debug 开关：dump 块化得分（S5a 后）与块级 merge 中间结果（S6a 后），用于阶段边界定位（kernel 错 vs golden 错）。合入前移除或进 debug 分支。

### 6.5 测试脚本设计（基于 test_quant_lightning_indexer_v2_single.py 修改）

新增 `tests/pytest/test_quant_lightning_indexer_v2_candidate.py`，结构复制自 `test_quant_lightning_indexer_v2_single.py`（保留 SAVE_PT_DIR/RESULT_PATH 环境变量、run_mode eager/graph 分支、QliV2ResultWriter 落盘机制），做以下修改：

**(1) 参数扩展**：`param_names` 尾部追加
```python
"candidate_mode",        # 1=source / 2=consumer / 3=off(默认)
"candidate_topk_blocks", # 默认 2048
"candidate_block_size",  # 默认 8
```
`test_data` 元组同步扩展；`QliV2ResultWriter.case_name/row` 的列随之扩展（sidecar 记录 candidate 参数，满足 §6.1 provenance 要求）。

**(2) paramset 用例**（新文件内定义或扩展 `test_quant_lightning_indexer_v2_paramset.py`）：
- 基线模板沿用 910b int8 形态（参照 `quant_li_default_a3`：quant_mode=2、qk_dtype=int8、dequant=float16、**q_head_num=32**、layout_key=PA_BBND），去掉 910b 不支持的 return_value/output_idx_offset。
- **设备分支新增 `Ascend910B`**（当前服务器为 910B3；现有 paramset 仅有 `Ascend910_93`/`Ascend950` 分支，910B 会 NameError）。
- 用例集（cmp_ratio × mode 全组合 + 边界）：

| 用例名 | cmp_ratio | mode | 覆盖点 |
|---|---|---|---|
| cand_r1_mode3_decode（回归） | 1 | 3 | decode 基线（mask_mode=0，标量 compress_lens；现网行为 bit 级一致） |
| cand_r1_mode1_decode | 1 | 1 | decode source 基础（batch 级 pin） |
| cand_r1_mode1_prefill_m3 | 1 | 1 | **prefill 首块 + mask_mode=3：行级 pin（不同行 pin 不同块）**+ 行级 mask |
| cand_r1_mode1_prefill_m0 | 1 | 1 | ~~mid-chunk prefill + mask_mode=0~~ **暂不执行**（问题 O3 未决保留，见 §5.0/R5；用例定义保留在矩阵中，待确认后启用） |
| cand_r1_mode1_tail | 1 | 1 | actS2Size%8≠0 + causal + 变长 B |
| cand_r1_mode2_self | 1 | 2 | 自洽候选（见 (3)a） |
| cand_r1_mode2_rand | 1 | 2 | 随机候选子集（见 (3)b）+ numBlocks<topkBlocks |
| cand_r1_mode2_few | 1 | 2 | numBlocks>topkBlocks（S2>16K） |
| cand_r2_mode1_decode | 2 | 1 | cmp_ratio=2 + cmp_residual_k 传入 |
| cand_r2_mode1_prefill_m3 | 2 | 1 | cmp_ratio=2 prefill 首块（行级 pin 含 residual 还原） |
| cand_r2_mode2_self | 2 | 2 | cmp_ratio=2 消费 |
| cand_r1_mode1_g32 | 1 | 1 | **q_head_num=32（本轮统一规格；mBase=128，验证 v1 式推导下分核/循环/输出全链路）**。其余用例同规格执行，不再单列 64 回归 |
| cand_r1_mode1_topk_min | 1 | 1/2/3 | **topk=min(2048, actS2Size)<2048：验证 -1 padding 与 min 等价性（与 topk=2048+滤-1 对比）** |

**(3) mode=2 的 `candidate_topk_index` 输入生成（golden 侧）**：
- (a) **自洽候选**：对同一 score 调用参考 `select_candidate_blocks` 的输出作为输入（等价于"source 层与 consumer 层权重相同的退化情形"，可校验 mode=2 结果 ⊆ mode=0 结果且含 pin 块）；
- (b) **随机子集**：从 `[0, numBlocks)` 随机采样 `min(topk_blocks, numBlocks)` 个块（覆盖任意候选集，含 numBlocks<topkBlocks 时的全选等价性）；两者都以 int32 tensor（`-1` padding 到 topk_blocks）随测试数据下发。

**(4) golden 扩展**（`quant_lightning_indexer_v2_golden.py`）：
- 新增 `select_candidate_blocks_ref(score, compress_lens, topk_blocks, block_size)`：按 §1.4 数学定义实现（pad→amax→pin→topk→-1 槽），**同时提供 numpy 逐行实现与 torch 实现交叉验证**（§6.1 要求的独立重建）。`compress_lens` 支持标量（decode/mask_mode=0 → batch 级 pin）与 `[S1,1]` 行级（prefill 首块/mask_mode=3 → 行级 pin）两种形态，行级值按模型公式 `(actS2SizeOrig - actS1Size + i + 1) // cmpRatio` 生成。
- `GeneralizedQLIV2` 扩展：mode=1 时在 `cal_atten_per_batch_int8` 的 `reduce_sum`（kScale 相乘 + causal mask 之后、sort 之前，fp32 域）上计算候选块 golden 并按 out 布局（BSND）返回；mode=2 时在同一位置对 `reduce_sum` 做 `masked_fill` 后走原 sort/topk。
- int8 路径 score 保持 fp32（无 bf16 cast），与 kernel 块化计算域一致，保证块 golden 确定性。

**(5) compare 扩展**（`result_compare_method.py`）：
- 新增 `check_result_candidate`：`candidate_topk_index` 按行集合比较——`set(有效块号)` 相等 + `-1` 槽数一致即 PASS（tie 容差见 §6.2）；
- `sparse_indices` 比对复用现有 `check_result`；mode=2 的 sparse golden 由 (4) 的 masked 路径产生。

**(6) 执行**：沿用现有 `test_run.sh`/pytest.ini 机制，单独一条 pytest 选择器（`test_quant_lightning_indexer_v2_candidate.py`），不混入现网 CI 集。

---

## 7. 风险与开放问题

| # | 风险/问题 | 影响 | 缓解 |
|---|---|---|---|
| R1 | tie-break 不一致 | 误报精度问题 | §6.2 集合比较 |
| R2 | Sort32/MrgSort 降序假设不成立 | 整体语义反转 | 验证点 V1：实现前单测确认 API 排序方向 |
| R3 | g=32 改动触碰 `mBaseSize/s1BaseSize` 推导，影响分核与循环边界（kernel 与 metadata 基准必须一致） | g=32 分核错乱/越界 | 对齐 v1 推导（§3.1）+ `cand_r1_mode1_g32` 用例 + g=64 全量回归（推导变更影响既有路径，mode=0 也需回归） |
| R4 | source/consumer 的 block_size 不一致（跨算子约定） | 语义错乱 | 文档强约束 + torch 层断言（`quant_lightning_indexer.candidate` python 封装内校验，§4.3） |
| **R5（标红，O3 未决）** | **mid-chunk prefill（start_pos≠0 且 S1>1）形态下模型不做行内因果 mask，与常规认知相反**；该调用形态是否存在未确认 | 若误当作 bug "修复"（擅自加行级 mask）将引入语义变更；测试碰 mode0+S1>1 数据可能误报精度问题 | **暂不测试该用例，问题保留**（§5.0 O3 标红段）；三不准：不准顺手加 mask / 不准以"常规认知"为 golden / 处理前必须先与模型侧确认调用形态 |

已决策记录：
- ~~O1~~ **已定**：torch 接口采用方案 b——新增 overload `quant_lightning_indexer.candidate(...) -> (Tensor, Tensor, Tensor)`，旧 schema 二元组不动（§4.3）。
- ~~O2~~ **已定**：`candidate_topk_blocks` 当前仅支持 2048；属性保留、host 校验限定 2048，未来扩展只放宽校验不改接口；非默认值测试用例已删除。

## 8. 实施拆解（文件级）

| 文件 | 变更 |
|---|---|
| `op_host/quant_lightning_indexer_v2_tiling.h` | TilingData +3 字段；ParaInfo/常量（CANDIDATE_* 索引） |
| `op_host/quant_lightning_indexer_v2_tiling.cpp` | 属性解析与校验（mode 互斥、范围、consumer 必传输入）；**`GetGSize` 910b 分支放宽允许 gSize=32（适配点 A2）**；TilingData 写入（workspace 计算不变，A6） |
| `op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h` | **`InitTilingData` 改 v1 式推导（A1）：`s1BaseSize=4` 固定、`mBaseSize = s1BaseSize×gSize`**；删除/降级 `M_BASE_SIZE=256` 常量；Init 传参/新 GM 张量（candidateTopkIndexInGm/candidateTopkIndexOutGm）、Vec1 调用点注入 |
| `op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h` | **Vec0 `cuProcEleNum` 补 `CeilAlign(·, 32)`（A5，同步 v1:289）**；S4a/S5a/S6a 实现、globalBlockTopkUb_、CleanInvalidCandidateOutput（不动 ProcessLD） |
| `tests/ut/op_host/arch22/` | **补 g=32 tiling 单测（A10）** |
| `op_host/quant_lightning_indexer_v2_def.cpp` | 新 input/output/attrs（仅 ascend910b 配置） |
| `op_host/quant_lightning_indexer_v2_infershape.cpp` | candidate_topk_index 输出 shape/dtype |
| `op_kernel/quant_lightning_indexer_v2.cpp` | arch22 入参透传（arch35 分支不动） |
| `examples/` + `docs/aclnnQuantLightningIndexerV2.md` | aclnn 用例与接口文档 |
| `torch_extension/` | schema overload + python 封装 |
| `tests/pytest/test_quant_lightning_indexer_v2_candidate.py` | **新增**，基于 `test_quant_lightning_indexer_v2_single.py` 修改（§6.5） |
| `tests/pytest/quant_lightning_indexer_v2_golden.py` | `select_candidate_blocks_ref`（numpy+torch 双实现）、`GeneralizedQLIV2` 的 mode=1/2 golden 路径 |
| `tests/pytest/result_compare_method.py` | `check_result_candidate`（集合比较 + -1 槽核对） |
| `tests/pytest/qliv2_test_utils.py` | case_name/row 列扩展（candidate 参数入 sidecar） |
| `README.md`（两算子） | 参数表、约束（block_size 一致性、同 forward 有效期、910b 无 LD 说明） |

**明确不做**：arch35 (950) 全部路径；LD/ProcessDecode 相关改动；metadata 算子代码；workspace 布局变更；TilingKey 变更。

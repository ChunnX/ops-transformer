# block_sparse_attention_grad 算子 arch22 确定性支持方案

> 参考 PR: https://gitcode.com/cann/ops-transformer/pull/1544 (`dense_lightning_indexer_grad_kl_loss` 确定性方案)
> 目标芯片: arch22 (DAV_2201 / Ascend 910B, `--soc=ascend910b`)
> 代码仓: /home/lsy/ops-transformer_7560

---

## 1. 非确定性根因分析

### 1.1 现状定位

arch22 kernel (`op_kernel/arc22/block_sparse_attention_grad_kernel.h`) 中：

```cpp
BlockMmadBSAG2 blockMmad2(resource, L1_SIZE_OFFSET, PINGPONG_OFFSET_2, true);  // isAtomicAdd=true, 写 gDq
BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 2, PINGPONG_OFFSET_4, true); // isAtomicAdd=true, 写 gDv/gDk
```

`block_mmad_bsag2.hpp:98-162` 显示：`isAtomicAdd=true` 时写回 GM 调用 `AscendC::SetAtomicAdd<ElementC>()`。

### 1.2 非确定性来源

| 输出 | 写回路径 | 跨核碰撞? | 确定性 |
|------|----------|-----------|--------|
| dQ   | blockMmad2 → gDq[qOffset] | 否（qOffset 按 Q 头划分，每核独立） | ✅ 确定 |
| dK   | blockMmad3 → gDk[kvOffset] | **是** | ❌ 非确定 |
| dV   | blockMmad3 → gDv[kvOffset] | **是** | ❌ 非确定 |

关键证据（`initTaskInfo`，kernel:230/235）：

```cpp
taskInfo.kvOffset = preKVSeqLengths * kvHeads * headDim + taskInfo.curHeadIdx / groupSize * headDim;
```

GQA 场景下 `groupSize = numHeads / kvHeads > 1`，**同组内 g 个 Q 头共享同一 `kvOffset`**。这 g 个 Q 头被 `AssignCoreTasks` 分配到不同 AIC 核（BNSD: `curHeadNum = remain / qBlocksPerHead`），多核并发对同一 `gDk/gDv` GM 地址执行 `SetAtomicAdd`。

浮点加法不满足结合律 → 原子加的跨核交错顺序不确定 → **dK/dV 结果非确定性**。

> dQ 因按 Q 头划分、跨核无碰撞，单核内顺序累加本身确定，**无需改造**。

post 阶段 (`VecPost → EpilogueFAGPost`) 仅把 workspace `gDk/gDv` 拷贝/转换到最终输出 `params.dk/dv`，不引入新的非确定性，但其依赖的 workspace 内容已非确定。

---

## 2. 确定性方案总体思路

参考 PR #1544 的核心范式：**用「每核私有 GM 缓冲 + 固定顺序聚合」替代「多核原子加到共享输出 GM」**，并以 `maxLoopSize` 锁步保证 `SyncAll` 不死锁。

### 与 PR 的差异

| 维度 | PR (dense_lightning_indexer_grad_kl_loss) | 本算子 (block_sparse_attention_grad) |
|------|------------------------------------------|--------------------------------------|
| 非确定输出 | dKeyIndex (跨核原子加) | dK / dV (GQA 同组跨核原子加) |
| 聚合粒度 | 单一 S2 tile | (task, kvBlock, basicKVSubBlock) tile |
| AIC:AIV | 1:2 | 1:2 (已复用 post 阶段 vector 核) |
| 锁步机制 | maxLoopSize + 每轮 SyncAll | 同 |
| 聚合执行者 | AIV (DeterAddKIndexGrad) | AIV (新增 VecDeterReduce, 复用 post 前 vector 核) |
| loss 类 reductions | 有 (per-AIV buffer + DeterSumLoss) | 无 (本算子无 loss) |

---

## 3. 详细设计

### 3.1 Tiling Host 侧改造

#### 涉及文件
- `op_host/block_sparse_attention_grad_tiling.h`
- `op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp`

#### (a) 读取 deterministic 开关

`BSAGradTiling` 类新增成员：
```cpp
bool deterministic_ = false;
```

在 `ProcessAttrs` 中：
```cpp
deterministic_ = (context->GetDeterministic() == 1);
OP_LOGD(context->GetNodeName(), "deterministic[%d].", deterministic_);
```

#### (b) TilingData 字段扩展

`BlockSparseAttentionGradTilingData` 新增字段：
```cpp
TILING_DATA_FIELD_DEF(uint8_t,  deterministic);       // 是否确定性模式
TILING_DATA_FIELD_DEF(uint32_t, maxLoopSize);         // 锁步最大轮次（tile 总数）
TILING_DATA_FIELD_DEF(uint32_t, deterBatchSize);      // 一批缓冲的轮次，固定 40
TILING_DATA_FIELD_DEF(uint32_t, numBatches);          // 批数 = ceil(maxLoopSize / 40)
TILING_DATA_FIELD_DEF(uint32_t, aicNum);
TILING_DATA_FIELD_DEF(uint32_t, aivNum);
TILING_DATA_FIELD_DEF(uint64_t, perCoreDkTileSize);   // 单轮单核 dk tile 字节数 = basicKVBlockSize*headDim*4
TILING_DATA_FIELD_DEF(uint64_t, dkDeterGmSize);       // 每核私有 dK 缓冲 = 40 * perCoreDkTileSize
TILING_DATA_FIELD_DEF(uint64_t, dvDeterGmSize);
TILING_DATA_FIELD_DEF(uint64_t, deterCoreInfoGmSize); // 每核 run info = 40 * DETER_CORE_INFO_NUM * 8B
TILING_DATA_FIELD_DEF(uint64_t, dkDeterGmOffset);     // workspace 内偏移
TILING_DATA_FIELD_DEF(uint64_t, dvDeterGmOffset);
TILING_DATA_FIELD_DEF(uint64_t, deterCoreInfoGmOffset);
```

> **保留** 原 `sOutSize/dPOutSize/dQOutSize/dKOutSize/dVOutSize` 共享 workspace（含共享 float `gDk/gDv`），deter 模式下共享 `gDk/gDv` 仍作为「跨批 float 累加器」+ VecPost 的 `float→fp16` cast 源，**不删除**。私有 40 轮缓冲为**新增**。

#### (c) CalcMaxLoop (锁步轮次 + 批数计算)

新增 `BSAGradTiling::CalcMaxLoop()`，参考 PR 的 `CalcMaxLoop`：

遍历每个 AIC 核 (coreIdx 0..aicNum-1) 的 task 列表，统计该核实际写 dK/dV 的 tile 次数：

```
perCoreLoop[core] = Σ_{task ∈ core} Σ_{kvBlock, mask==true} ceil(kvBlockSize / basicKVBlockSize)
```

- 复用 `AssignCoreTasks` 已算出的 per-core 起始 task 信息（`beginBatch/beginHead/beginQSeqOffset`）与 `taskNumPerCore_/tailTaskNum_`。
- 按 TND / BNSD 两种 layout 分别计算每核 qBlocks 与每 task 的 kvBlock 数（参考 kernel `operator()<AIC>` 的遍历逻辑）。
- `maxLoopSize_ = max(perCoreLoop[*])`。
- 批参数：
  ```
  deterBatchSize_ = 40;                                   // 固定 40 轮一批
  numBatches_     = CeilDiv(maxLoopSize_, deterBatchSize_); // 锁步到批粒度
  ```
- 不足 `maxLoopSize_` 的核在 kernel 侧用 invalid run info 填充剩余轮次；不足一批的末批用 invalid 填满 40 槽。

#### (d) Workspace 重算 (CalculateWorkSpace)

`deterministic == true` 时，**保留**原共享区（`sOutSize/dPOutSize/dQOutSize/dKOutSize/dVOutSize/gradSize` 原样），在其后**追加**私有 40 轮缓冲：

```cpp
constexpr uint64_t DETER_CORE_INFO_NUM = 8; // int64_t: bIdx/kvStart/curKvSize/qHead/group/kvLocalOffset/sOffset/valid
constexpr uint64_t DETER_BATCH = 40;
constexpr uint64_t WORKSPACE_ALIGN = 256;

if (deterministic_) {
    // 单轮单核 dk tile = basicKVBlockSize × headDim × float
    uint64_t perTile = CeilDiv(basicKVBlockSize_ * headDim_ * sizeof(float), WORKSPACE_ALIGN) * WORKSPACE_ALIGN;
    perCoreDkTileSize_ = perTile;
    // 每核 40 轮，dk + dv 各一份
    dkDeterGmSize_ = aicNum_ * DETER_BATCH * perTile;   // gDkDeter: [aicNum][40][perTile]
    dvDeterGmSize_ = aicNum_ * DETER_BATCH * perTile;   // gDvDeter: 同构
    deterCoreInfoGmSize_ = aicNum_ * DETER_BATCH * DETER_CORE_INFO_NUM * sizeof(int64_t); // [aicNum][40][8]
    workSpaceSize_ += dkDeterGmSize_ + dvDeterGmSize_ + deterCoreInfoGmSize_;
}
```

`FillTilingData` 中下发 `dkDeterGmOffset/dvDeterGmOffset/deterCoreInfoGmOffset`（依次紧跟 `gradSize_` 之后，256B 对齐），以及 `deterBatchSize=40 / numBatches / perCoreDkTileSize`。

> 体积估算（910B, basicKVBlockSize=128, headDim=128, aicNum≈24）：
> 单轮单核 tile = 128×128×4B = 64KB；dk+dv = 128KB/轮/核；
> 40 轮 × 24 核 × 128KB ≈ **120 MB**。runInfo ≈ 24×40×64B ≈ 60KB（可忽略）。

#### (e) GenerateTilingKey 不扩展（保持原 4 个 key）

> **本方案不新增 tilingKey**，确定性分支由 TilingData 的 `deterministic` 字段在 kernel 运行期判断，避免污染 tilingKey 命名空间与编译产物矩阵。

`GenerateTilingKey()` 维持原逻辑（0/1/10/11，按 dataType × layout），不引入 deter 偏移。kernel 侧通过 `tilingData->deterministic` 运行期分支选择共享写回 / 私有缓冲聚合两条路径（见 3.2）。

| dataType | layout | tilingKey (deter 与非 deter 共用) |
|----------|--------|----------------------------------|
| FP16     | TND    | 0 |
| FP16     | BNSD   | 1 |
| BF16     | TND    | 10 |
| BF16     | BNSD   | 11 |

---

### 3.2 Kernel 侧改造

#### 涉及文件
- `op_kernel/arc22/block_sparse_attention_grad_kernel.h`
- `op_kernel/arc22/attn_infra/gemm/block/block_mmad_bsag2.hpp`

> 不改动 `op_kernel/block_sparse_attention_grad.cpp` 的 tilingKey 模板分发（保持原 4 个 key），确定性走运行期分支。

#### (a) deterministic 运行期分支

`BlockSparseAttentionGradKernel` 不新增模板参数，改为在 `operator()<AIC>/<AIV>` 入口从 tilingData 读取运行期开关：

```cpp
const uint8_t deter = tilingData->deterministic;
```

后续所有 deter 差异点用 `if (deter)` 分支包裹（kernel 内分支开销可忽略，因均在 task 粒度外层）。不使用 `if constexpr`，避免依赖 tilingKey 的编译期分发。

#### (b) blockMmad3 写回目标参数化（私有 40 轮缓冲）

`block_mmad_bsag2.hpp` 中 `BlockMmadBSAG3`：
- 构造增加 `GlobalTensor<ElementC> &deterDkGm, &deterDvGm` 及 `bool deter`。
- `isAtomicAdd` 含义调整：
  - 非 deter：`isAtomicAdd=true`，写共享 `gDk/gDv`（原逻辑）。
  - deter：写**每核私有 40 轮缓冲** `deterDkGm/deterDvGm`；核内仍可保留 `SetAtomicAdd`（单核顺序原子加是确定的），**关键是跨核不再共享地址**。

私有缓冲按 `[aicIdx][roundInBatch][kvLocalOffset]` 三维布局，`roundInBatch = globalRound % 40`。kernel `operator()<AIC>` 初始化：

```cpp
GlobalTensor<float> gDkDeter, gDvDeter;
const uint8_t deter = tilingData->deterministic;
const uint64_t perTile = tilingData->perCoreDkTileSize;
if (deter) {
    gDkDeter.SetGlobalBuffer((__gm__ float*)(params.workspace + tilingData->dkDeterGmOffset));
    gDvDeter.SetGlobalBuffer((__gm__ float*)(params.workspace + tilingData->dvDeterGmOffset));
}
```

blockMmad3 调用目标改为（运行期分支，`ridx = globalRound % 40`）：
```cpp
uint64_t deterOff = coreIdx * (40 * perTile) + ridx * perTile + kvLocalOffset;
auto &dkDst = deter ? gDkDeter[deterOff] : gDk[preTaskInfo.kvOffset];
auto &dvDst = deter ? gDvDeter[deterOff] : gDv[preTaskInfo.kvOffset];
blockMmad3(gDs[...], gQ[...], dkDst, ...);  // dK
blockMmad3(gP[...], gDout[...], dvDst, ...); // dV
```

#### (c) 每轮 run info 发布 + 批级锁步（40 轮一次 SyncAll）

参考 PR 的 `SaveDeterRunInfo` / `SaveDeterRunInfoInvalid`。在 AIC `operator()` 的 (task, kvBlock, kvSubBlock) 循环内，每写完一个 dK/dV tile，把 run info 写入**本批对应槽位** `runInfo[coreIdx][ridx]`（`ridx = globalRound % 40`）：

```cpp
if (deter) {
    deterCoreInfoUb_.SetValue(0, curInfo.curBatchIdx);
    deterCoreInfoUb_.SetValue(1, kvStartIdx);       // 全局 kv 行起点
    deterCoreInfoUb_.SetValue(2, curInfo.curCalKVSize);
    deterCoreInfoUb_.SetValue(3, curInfo.curHeadIdx);
    deterCoreInfoUb_.SetValue(4, curInfo.curHeadIdx / groupSize); // groupIdx
    deterCoreInfoUb_.SetValue(5, ridx);             // 私有缓冲内轮次槽位
    deterCoreInfoUb_.SetValue(6, kvLocalOffset);    // tile 内偏移
    deterCoreInfoUb_.SetValue(7, 1);                // valid
    uint64_t infoOff = coreIdx * (40 * DETER_CORE_INFO_NUM) + ridx * DETER_CORE_INFO_NUM;
    DataCopy(deterCoreInfoGm[infoOff], deterCoreInfoUb_, DETER_CORE_INFO_NUM);
}
// ... 原 mmad 计算 ...
globalRound++;
if (deter && (globalRound % 40 == 0)) {
    SyncAll<false>();   // 攒满 40 轮才同步一次
}
```

AIC 主循环按**批**外层包裹：`for (batch = 0; batch < numBatches; batch++) { 填 40 轮私有 tile; SyncAll; 等 AIV 聚合完; SyncAll }`（交替停顿，见 (d)）。不足一轮/一批的用 invalid（valid=0）填充但**仍参与每批的两次 SyncAll**。

#### (d) 新增聚合阶段 VecDeterReduce (AIV 执行，按批 + kv 行划分)

AIV 主流程改为按批交替：
```cpp
template <>
void operator()<AscendC::AIV>(Params const &params) {
    const uint8_t deter = tilingData->deterministic;
    VecPre(params);
    AscendC::SyncAll<false>();
    VecOp(params);
    AscendC::WaitEvent(CUBE2POST);
    AscendC::SyncAll<true>();
    if (deter) {
        const uint32_t numBatches = tilingData->numBatches;
        for (uint32_t b = 0; b < numBatches; b++) {
            AscendC::SyncAll<false>();      // 等 cube 填满本批 40 轮
            VecDeterReduceBatch(params, b); // 聚合本批 40 轮 → 共享 float gDk/gDv
            AscendC::SyncAll<false>();      // 通知 cube 可写下一批
        }
    }
    VecPost(params);
}
```

`VecDeterReduceBatch` 逻辑（参考 PR `DeterAddKIndexGrad`，**按 kv 行划分避免 AIV0/AIV1 碰撞**）：
```cpp
__aicore__ inline void VecDeterReduceBatch(Params const &params, uint32_t batchIdx) {
    uint32_t subBlockIdx = AscendC::GetBlockIdx() % 2;  // AIV0 / AIV1
    for (uint32_t r = 0; r < 40; r++) {                 // 批内 40 轮
        for (uint32_t aicIdx = 0; aicIdx < aicNum; aicIdx++) {  // 固定核序
            // 1. 读 runInfo[aicIdx][r]
            DataCopy(deterCoreInfoUbConsumer_,
                     deterCoreInfoGm[aicIdx*(40*INFO_NUM) + r*INFO_NUM], INFO_NUM);
            if (valid == 0) continue;
            // 2. kv 行划分：AIV0 处理 curKvSize 前半行，AIV1 后半行 → 行不重叠，无跨 AIV 碰撞
            uint32_t rowStart = subBlockIdx * (curKvSize / 2);
            uint32_t rowNum   = (subBlockIdx == 0) ? (curKvSize - curKvSize/2) : (curKvSize/2);
            // 3. 把 gDkDeter[aicIdx][r][rowStart..] 累加到共享 float gDk[全局kvOffset+rowStart..]
            //    ping-pong UB 读私有 → 单核 SetAtomicAdd(顺序确定) 或 读改写 写共享 gDk
            //    dV 同理写共享 gDv
        }
    }
}
```

聚合写回**共享 float `gDk/gDv`**（保留的原 workspace），此时：
- 跨核原子加被消除（AIV 按固定 aicIdx 顺序累加）；
- 跨 AIV 碰撞被消除（AIV0/AIV1 行不重叠）；
- 单 AIV 内顺序确定 → dK/dV 确定性。

聚合完成后进入 `VecPost`，照原逻辑 `float→fp16` 写最终输出 `params.dk/dv`，**post 阶段不动**。

#### (e) dQ 不动

dQ 跨核无碰撞，`blockMmad2` 保持 `isAtomicAdd=true` 写共享 `gDq`，deter 模式不改造。

---

## 4. 锁步与死锁规避（批粒度，交替停顿）

- `maxLoopSize` 由 host `CalcMaxLoop` 计算，`numBatches = ceil(maxLoopSize / 40)`。
- AIC 与 AIV 均以**批**为外层锁步单位，每批固定 40 轮（末批不足补 invalid）：
  ```
  for (b = 0; b < numBatches; b++) {
      AIC:  填本批 40 轮私有 tile + runInfo     // 不足补 invalid
      SyncAll<false>();                          // ① cube 完成 → 唤醒 AIV
      AIV:  VecDeterReduceBatch(b)               // 聚合本批 → 共享 float gDk/gDv
      SyncAll<false>();                          // ② AIV 完成 → 唤醒 cube 写下一批
  }
  ```
- 「交替停顿」：cube 在 AIV 聚合期间空转，AIV 在 cube 填充期间空转。每批 2 次 `SyncAll`，相比「每轮一次 SyncAll」减少 40 倍同步开销。
- 所有核无论实际 tile 数多少，都执行 `numBatches` 批并参与每批两次 `SyncAll` → 无挂死风险。
- 参考 PR 中 `for(int i = taskId; i < constInfo.maxLoopSize; i++)` 的填充思想，本方案按批填充。

---

## 5. 改动文件清单

| 文件 | 改动要点 |
|------|----------|
| `op_host/block_sparse_attention_grad_tiling.h` | TilingData 增加 deter 相关字段；`BSAGradTiling` 增加 `deterministic_` 成员、`CalcMaxLoop()` 声明 |
| `op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp` | `ProcessAttrs` 读 `GetDeterministic`；新增 `CalcMaxLoop`；`CalculateWorkSpace` 追加私有缓冲；`GenerateTilingKey` **不改**；`FillTilingData` 下发 `deterministic` 字段与新偏移 |
| `op_kernel/arc22/block_sparse_attention_grad_kernel.h` | kernel 入口读 `tilingData->deterministic` 运行期分支；AIC 写回改私有缓冲 + 每轮 run info 发布 + `SyncAll`；AIV 新增 `VecDeterReduce` 聚合阶段 |
| `op_kernel/arc22/attn_infra/gemm/block/block_mmad_bsag2.hpp` | `BlockMmadBSAG3` 支持目标缓冲参数化（共享 vs 私有），`isAtomicAdd` 由运行期 `deter` 标志控制 |

> 注：不新增 tilingKey，`op_kernel/block_sparse_attention_grad.cpp` 的 tilingKey→模板分发表无需改动，deter 与非 deter 共用同一组 kernel 实例。

---

## 6. 验证计划

1. **确定性验证**：设置环境变量 `ASCEND_LAUNCH_KERNEL_DETERMINISTIC=1`（或框架层 `GetDeterministic()==1`），多次执行 `examples/test_aclnn_block_sparse_attention_grad.cpp`，逐 bit 比对 dK/dV 输出一致。
2. **精度验证**：deter 与非 deter 模式结果对比，dK/dV 误差在浮点容差内（fp16 atol/rtol 1e-3，bf16 1e-2）。
3. **GQA 重点用例**：`numHeads != kvHeads`（如 numHeads=8, kvHeads=2, groupSize=4）场景，复现跨核原子加非确定性问题，验证 deter 模式消除。
4. **性能基线**：对比 deter / 非 deter 的端到端耗时，评估聚合阶段 + 私有缓冲带来的开销（预期 dK/dV 聚合占比小）。
5. **UT/ST**：复用 `tests/` 下既有用例，新增 deter 模式 run。

---

## 7. 风险与备注

- **GM 占用增加**：私有 40 轮 dK/dV 缓冲 ≈ `aicNum × 40 × basicKVBlockSize × headDim × 4B × 2`。910B aicNum≈24, basicKVBlockSize=128, headDim=128 时 ≈ 24×40×128KB ≈ **120 MB**（共享 gDk/gDv 保留不删，但其体积通常远小于私有缓冲）。需确认 CANN/框架 workspace 上限。
- **性能**：交替停顿下 cube 在 AIV 聚合期间空转；40 轮一批使 SyncAll 频率降 40 倍。聚合阶段 GM→UB→GM 往返量 ≈ 私有缓冲总量，长序列/MQA 场景需重点评估；后续可升级为 ping-pong 双缓冲（私有区再 ×2 ≈ 240MB）隐藏聚合延迟。
- **AIV 行划分边界**：`curKvSize` 为奇数时 AIV0/AIV1 行数差 1，需保证两核行区间不重叠不遗漏（参考 kernel `VecOp` 已有的 `(curCalQSize + 1 - subBlockIdx) / 2` 写法）。
- **MQA (kvHeads=1)**：groupSize 最大，跨核碰撞最严重，deter 收益最显著，但聚合压力也最大，需重点测试。
- **dQ 暂不改造**：若未来 dQ 也出现跨核碰撞（如布局变化），可同法扩展。
- **GetDeterministic API**：需确认本仓 CANN 版本（cann-9.0.0）`gert::TilingContext::GetDeterministic()` 可用；若不可用，回退为通过 attr 传入。
- **deterBatchSize=40 可调**：40 为权衡值，后续可按实测调整为 tiling 参数（如 16/32/64），权衡 SyncAll 频率与私有缓冲体积。

---

## 8. 落地实现状态（本轮）

### 已完成并验证
| 模块 | 状态 | 验证 |
|------|------|------|
| 设计文档 | ✅ | 本文件 |
| Tiling host (`tiling.h` + `tiling_arch22.cpp`) | ✅ 编译通过 | `GetDeterministic` 读取、`CalcMaxLoop`、workspace 追加、`FillTilingData` 下发；**非 deter 路径无回归**（examples 通过） |
| `block_mmad_bsag2.hpp` `SetAtomicAdd` 开关 | ✅ 编译通过 | — |
| Kernel AIC deter 分支（私有缓冲写回 + runInfo + 逐轮 SyncAll + padding） | ✅ 编译通过 | 非 deter 路径无回归 |
| Kernel AIV `VecDeterReduceOneRound` 聚合（kv 行划分） | ✅ 编译通过 | 非 deter 路径无回归 |

### 当前实现采用的同步粒度
本轮落地的是**逐轮 SyncAll**（每核 1 个 slot，每轮 cube 写完→SyncAll#1→AIV 聚合→SyncAll#2→下一轮），作为 40 轮批缓冲的**正确性基线**。私有缓冲极小（~3MB）。升级到 40 轮批缓冲为局部改动（slot 数 1→40、同步频率 1→1/40）。

### 已知运行问题（deter 路径，待调试）
强制 `deterministic_=true` 验证时，deter 路径可启动（workspace 涨至 ~117MB，符合私有缓冲预期），但 `aclrtSynchronizeStream` 报 **507015**（AIC 侧异常/超时）。疑似原因：

1. **UB/TPipe 冲突（首要怀疑）**：AIV 的 softmax epilogue 自带 TPipe 管理 UB；deter 聚合新增的局部 `deterPipe`/`deterQue` 在同核上二次分配 UB，可能与 epilogue 的 UB 区间冲突 → AIV 异常。
   - 解决方向：复用 epilogue 的 TPipe/UB 分配一块聚合 buffer，或改用 GM 侧累加（如 `WholeReduceSum`/`Adds` 等 GM 原语）避免新增 TPipe。
2. **逐轮 SyncAll 时序**：cube mmad3 与 AIV softmax 的流水线偏移（preTaskInfo 延迟 1 轮）可能导致 SyncAll 配对错位 → 死锁超时。
   - 解决方向：在 AIV 侧用与 cube 一致的 round 计数（含 pipeline 偏移），或把 deter SyncAll 移到 task 粒度外层。

### 默认状态
- `deterministic_` 由 `context->GetDeterministic()==1` 决定（框架默认返回 `int32_max`，即关闭）。
- **当前仓库：deter 默认关闭，非 deter 路径完整可用，examples 通过。**

### 下一步
1. 排查 507015：先用 `ASCEND_GLOBAL_LOG_LEVEL=1` 抓 AIV 异常地址，定位是 UB 冲突还是 SyncAll 死锁。
2. 修复后强制 `deterministic_=true` 跑 examples，比对 dK/dV 多次执行逐 bit 一致。
3. 验证通过后升级为 40 轮批缓冲（性能优化）。

---

## 9. 调试进展（二分定位记录）

通过设备上迭代二分，已精确定位 deter 路径运行故障的根因链：

### 已修复的问题
| # | 问题 | 现象 | 修复 |
|---|------|------|------|
| 1 | **AIC 标量 GM 写非法** | runInfo 由 AIC 用 `GlobalTensor::SetValue` 发布 → AIC `_mix_aic` 异常 (0x26) | runInfo 发布移到 AIV（cube 核不支持标量 GM 写） |
| 2 | **AIV kvOffset 未按 kvBlock 更新** | AIV 的 `curInfo.kvOffset` 在循环里只取初值，发布的 runInfo 偏移 stale → 聚合写共享 gDk 越界 | AIV 循环补 `kvOffset = beginKVOffset + (kvBlockOffset+kvBlockBasicOffset)*stride`，镜像 AIC 逻辑 |
| 3 | **聚合循环用 aicNum 而非 blockDim** | `deterAicNum=23` 但实际启动 AIC=2，聚合读未启动核的未初始化 runInfo → OOB | 改用 `deterLaunchedAic = (usedVecCoreNum+1)/2` |

### 当前卡点（BLOCKER）
经二分确认（禁用 runInfo 发布块 → 通过；启用 → 507015），故障在 **runInfo 发布**：
- **`GlobalTensor::SetValue` 在 AIV 运行时不被支持**（编译通过但运行触发 aicore 异常）。
- 改用 UB `LocalTensor.SetValue + DataCopy` 发布 **同样故障**：因 `deterPipe`（独立 TPipe）与 softmax epilogue 自带 TPipe 在同核 UB 上冲突——A2 单核 UB 由 epilogue 的 TPipe 独占管理，第二个 TPipe 的 UB 读写引发异常。
- `PipeBarrier<PIPE_ALL>` 前置/后置均无效。

### 解决方向（下一步）
1. **复用 epilogue 的 TPipe/UB**：通过 `GetTPipePtr()` 全局 pipe 或改造 epilogue 暴露其 TPipe，从中分配 runInfo/aggregation 的 TBuf，避免第二 TPipe。
2. **GM 侧原语聚合**：用 `AscendC::DataCopy` GM→GM（若支持）或 cube 侧 FixOut 原子加做聚合，完全避开 AIV UB。
3. **runInfo 发布改用 AIV 已有 UB buffer**（softmax epilogue 的某块空闲 UB）临时复用。

### 验证状态
- ✅ 非 deter 路径：examples 通过，无回归（多次确认）。
- ✅ deter 路径编译通过、可启动、workspace 正确（~117MB）。
- ⏳ deter 路径运行：卡在 runInfo 发布（上述 BLOCKER），待复用 epilogue TPipe 方案验证。

---

## 10. 突破：deter 路径跑通 ✅

经持续调试，找到并修复了 deter 路径的全部运行时故障，**deter 路径现已跑通**（强制 `deterministic_=true` 时 `Execution Success`，run_exit=0）。

### 修复链（按定位顺序）
| # | 根因 | 修复 |
|---|------|------|
| 1 | AIC `GlobalTensor::SetValue` 不支持 | runInfo 发布移到 AIV |
| 2 | AIV `curInfo.kvOffset` 循环里未按 kvBlock 更新 → runInfo 偏移 stale | AIV 补 `kvOffset = beginKVOffset + (kvBlockOffset+kvBlockBasicOffset)*stride` |
| 3 | 聚合循环用 `deterAicNum=23` 而非实际启动 `blockDim=2` → 读未启动核未初始化 runInfo OOB | 改用 `deterLaunchedAic=(usedVecCoreNum+1)/2` |
| 4 | 独立 `deterPipe` 与 softmax epilogue 全局 TPipe 的 UB 冲突 | 移除独立 TPipe，复用 kernel `resource.ubBuf`（全局 TPipe）高地址空闲区 |
| 5 | `GlobalTensor::SetValue` 在 AIV 也不支持 | runInfo 改用 `LocalTensor.SetValue + DataCopy`（UB→GM）|
| 6 | runInfo 用 int64 DataCopy | 改 int32（值域足够：kvOffset<2^31）|
| 7 | **workspace 布局错误（最终根因）**：deter 缓冲放在 16MB libapi 保留区**之后**（~18MB，超出 `params.workspace` 可达区）→ GM 越界异常 | deter 缓冲移到 data 区之后、libapi **之前**（~1.25MB）；`workSpaceSize = data + deter + libapi` |

### 关键认知
- **libapi 是 16MB 保留区**，位于 workspace 末尾；`params.workspace` 可达区是 `[0, data+deter)`，libapi 在其后。deter 必须放在 data 与 libapi 之间。
- softmaxgrad epilogue UB 占 ~184KB/192KB，deter 复用 `resource.ubBuf` 的 [184K,192K) 高地址空闲区（runInfo 64B + 聚合分块）。
- AIC 不支持标量 GM 写；AIV 用 UB SetValue+DataCopy 发布 runInfo。

### 当前状态
- ✅ deter 路径：强制开启时 `Execution Success`（run_exit=0）。
- ✅ 非 deter 路径：3× 运行稳定，无回归。
- ✅ 默认关闭（`GetDeterministic()==1`，框架默认 `int32_max`≠1）。
- ⏳ **确定性验证待补**：当前 examples 用例 `numHeads=1, kvHeads=1`（groupSize=1，无 GQA 跨核碰撞），deter 与非 deter 结果一致但未触发非确定性场景。需补 GQA 用例（`numHeads>kvHeads`）验证 dK/dV 多次逐 bit 一致。
- ⏳ **40 轮批缓冲**：当前为 per-round（1 slot 复用），后续升级为 40 轮批缓冲降 SyncAll 频率。

---

## 11. GQA 正确性验证与根因（bsag_gqa_bnsd.py）

用 `bsag_gqa_bnsd.py`（B=10,N1=12,kvHeads=4→groupSize=3，npu:7，`torch.use_deterministic_algorithms(True)` 激活 `GetDet=1`）验证：

### 结果
- ✅ **逐 bit 确定性达成**：3 次运行 dK/dV 误差模式完全一致（bit-identical）。
- ✅ **dQ 正确**（True，未走聚合）。
- ✅ **cube 正确写私有缓冲**（no-op 聚合时 dK/dV 全 0 → cube 未碰共享 gDk）。
- ❌ **dK/dV 正确性 bug**：聚合开启时 dK/dV 错误（abs diff 183 vs golden 0.04，混合正确小值与错误大值如 11.3）。

### 根因（确认）
**`SyncAll<false>` 是按核心类型（per-type）同步，不跨 AIC↔AIV**：
- 证明：非 deter 路径 AIV 的 VecPre `SyncAll<false>`（line 620）若是全局的，AIC 无匹配 SyncAll 会死锁；但非 deter 正常运行 → `SyncAll<false>` 是 per-type。
- 后果：deter 的 AIV 聚合在 `SyncAll#1` 后读私有缓冲，但 `SyncAll#1` 只同步 AIV 核，不等 AIC 的 mmad3（私有写）完成 → AIV 读到 stale/未初始化私有数据 → dK/dV 错误。
- 旁证：清零私有缓冲后仍出垃圾（183），说明 AIV 读到的不是 cube 写的正确数据（时序未对齐）。

### 修复方向（待实现）
用**跨核 flag**（`CrossCoreSetFlag`/`CrossCoreWaitFlag`，参考现有 CUBE2VEC/VEC2CUBE 机制）做 AIC↔AIV 同步：
1. AIC mmad3 写私有后 → `CrossCoreSetFlag(DETER_C2V)`（私有就绪）。
2. AIV 聚合前 → `CrossCoreWaitFlag(DETER_C2V)`（等私有就绪）→ 读私有 → 聚合 → `CrossCoreSetFlag(DETER_V2C)`（聚合完成）。
3. AIC 下一轮前 → `CrossCoreWaitFlag(DETER_V2C)`（等聚合完成，避免覆盖 slot）。

**挑战**：聚合需读**所有 AIC 核**的私有（GQA 同组跨核求和），但跨核 flag 是 per-block（AIC↔其 AIV）。需：
- (a) 每核 AIV 只聚合本 block 私有 + 跨 block 用共享 gDk 原子加（但跨 AIV 顺序不确定 → 破坏确定性）；或
- (b) 用全局跨类型同步原语（若 CANN 提供）；或
- (c) 改用「同组 Q 头绑核」方案（(B)）从根本上避免跨核聚合。

### 已完成的调试修复（保留）
- runInfo 改用 DataCopy(DMA) 读 UB 再 GetValue（避免标量 GM 读不一致）。
- kv 行划分修正（AIV0:[0,half) AIV1:[half,curKvSize) 无重叠）。
- 聚合循环用 `deterLaunchedAic`（实际启动核数）。

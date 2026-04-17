# 阶段 2 产出：FA 设计决策记录

## 算子信息（从 phase1 继承）
- 算子名称：kv_quant_sparse_flash_attention_pioneer (QSFA-Pioneer)
- 开发场景：S1 特性注入（新增 Flash Decode）
- 基座算子：QSFA-Pioneer 自身
- 目标平台：A5（arch35, Ascend 950）

---

## FA 通用检查（F1-F8 + F9）

### F1 流水阶段划分

**继承判定：Phase 1 完全继承，Phase 2 新增**

**Phase 1（5 阶段三级流水，与 base 完全一致）：**

| 阶段 | 名称 | 执行核 | 计算内容 | 输入来源 | 输出去向 |
|------|------|--------|---------|---------|---------|
| Vec0 | KV 搬运+反量化 | AIV | sparse indices 索引 → GM KV 搬运 → FP8 dequant VF → L1 | GM(KV_FP8, sparseIndices, blockTable) | L1(KV_BF16) |
| C1(BMM1) | Q×K^T | AIC | MatmulK: Q_tile × K_tile^T → S_tile | L0A(Q from L1), L0B(K from L1) | UB(bmm1Buffers, FP32) |
| Vec1(Softmax) | Scale+Softmax+Cast | AIV | scale → RowMax → Sub → Exp → RowSum → Div → Cast → L1 | UB(bmm1Buffers, FP32) | L1(P_BF16, BMM2 左矩阵) |
| C2(BMM2) | P×V | AIC | MatmulN: P_tile × V_tile → O_tile | L0A(P from L1), L0B(V from L1) | UB(bmm2Buffers, FP32) |
| Vec2(Update) | FlashUpdate+输出 | AIV | Online rescale 累积更新 | UB(bmm2Buffers, FP32) | **base**: GM(attentionOut) / **FD**: GM(workspace: accumOut) |

**Phase 1 FD 变更点（仅 Vec2 输出去向）：**
- base 模式：Vec2 最后一轮执行 `Bmm2DataCopyOut` → Cast FP32→OUTPUT_T → DataCopyPad → attentionOutGm
- FD 模式：Vec2 最后一轮执行 `Bmm2FDOut` → DataCopy FP32 → accumOutGm（不做 Cast，保持 FP32）
- FD 模式：Phase 1 结束后额外调用 `ComputeLogSumExpAndCopyToGm` → 将 softmaxMax/softmaxSum 写入 workspace GM

**Phase 2（FD 新增，仅 AIV 执行）：**

| 阶段 | 名称 | 执行核 | 计算内容 | 输入来源 | 输出去向 |
|------|------|--------|---------|---------|---------|
| FD-Barrier | SyncAll | AIC+AIV | 全核屏障同步 | - | - |
| FD-Init | 缓冲区重置 | AIV | tPipe->Reset() + 分配 5×32KB UB | - | UB |
| FD-CopyLse | LSE 加载 | AIV | 从 GM 加载所有分片的 softmaxMax/softmaxSum | softmaxFDMaxGm, softmaxFDSumGm | UB(softmaxMaxInputQue, softmaxSumInputQue) |
| FD-Scale | 归一化权重 | AIV | ComputeScaleValue_VF: globalMax → exp(max_k - globalMax) × sum_k → totalSum → scale[k] | UB(LSE data) | UB(lseTmpBuff) |
| FD-Reduce | 加权求和 | AIV | ReduceFinalRes_VF: dst = Σ scale[k] × accumOut[k] | accumOutGm → UB(accumOutInputQue) | UB(FDResOutputQue) |
| FD-CopyOut | 结果输出 | AIV | Cast(FP32→OUTPUT_T) + DataCopy → attentionOutGm | UB(FDResOutputQue) | attentionOutGm |

**CV 比例：不变，AIC:AIV = 1:2（CV_RATIO = 2）**

### F2 核间同步设计

**继承判定：Phase 1 完全继承，Phase 2 新增 SyncAll**

**Phase 1 同步事件（与 base 完全一致，通过 BuffersPolicy 自动管理）：**

| 同步点 | Buffer 类型 | 发送方 | 接收方 | 语义 |
|--------|------------|--------|--------|------|
| l1RightBuffers | L1, CROSS_CORE_SYNC_FORWARD, 3buff | AIV(Vec0 写) | AIC(BMM1/BMM2 读) | KV 搬运完成 → Cube 可使用 |
| bmm1Buffers | UB, CROSS_CORE_SYNC_BOTH, DB | AIC(BMM1 写) → AIV(Vec1 读) → AIV(释放) → AIC(下轮写) | 双向 | BMM1 结果传递 + 释放 |
| bmm2Buffers | UB, CROSS_CORE_SYNC_BOTH, SingleBuf | AIC(BMM2 写) → AIV(Vec2 读) → AIV(释放) → AIC(下轮写) | 双向 | BMM2 结果传递 + 释放 |

**Phase 2 新增同步：**

| 同步点 | 类型 | 发送方 | 接收方 | 语义 |
|--------|------|--------|--------|------|
| SyncAll() | 全核屏障 | 所有核 | 所有核 | Phase 1 → Phase 2 过渡，确保所有分片写入 workspace 完成 |

**SyncAll 调用位置分析（关键设计决策）：**

参考 attention_pioneer 的实现，Process() 中的 SyncAll 仅在 AIV 侧调用：
```cpp
if constexpr (isFd) {
    if ASCEND_IS_AIV {
        SyncAll();
        vecBlock.InitFDBuffers(constInfo);
        vecBlock.FlashDecodeCompute(constInfo, keyGm, actualSeqKvlenAddr);
    }
}
```

但 QSFA-Pioneer 的架构不同：AIC 和 AIV 通过编译期模板参数实例化不同的服务类（QSFAMatmulService vs QSFAMatmulServiceDummy），两者共享同一个 Process() 函数。因此 SyncAll 需要在 `if constexpr (isFd)` 分支内、`ASCEND_IS_AIV` 判断之前调用，确保 AIC 也参与屏障：

```cpp
if constexpr (isFd) {
    SyncAll();  // AIC+AIV 均参与
    if ASCEND_IS_AIV {
        vecBlock.InitFDBuffers(constInfo);
        vecBlock.FlashDecodeCompute(constInfo, keyGm, actualSeqKvlenAddr);
    }
}
```

**空闲核处理**：`aicIdx >= actualCoreNums` 的核在 ProcessMainLoop 中直接 return，但仍需参与 SyncAll。因此 SyncAll 必须放在 ProcessMainLoop 之外（Process 函数末尾），所有核（包括空闲核）均执行。

### F3 分核策略

**继承判定：需修改 — FD 模式新增分核分支**

**base 分核（继承不变）：**
- 切分轴：B × N2 × S1 三维展开
- 策略：Kernel 侧动态均衡分核（遍历所有 batch 的 actualSeqQ 累加 totalBaseNum，均分到各核）
- 每核处理量：`avgBaseNum = ceil(totalBaseNum / coreNum)` 个 S1 基本块

**FD 分核（新增 `if constexpr (isFd)` 分支）：**
- 切分轴：(B × S1) × splitKVNum
- 策略：每个 (B, S1) 块固定分配 splitKVNum 个核，S2 按实际长度非均匀切分
- 核索引解码：
  - `s1BlockIdx = aicIdx / splitKVNum`
  - `flashDecodeS2Idx = aicIdx % splitKVNum`
  - 从 s1BlockIdx 反解 `(boIdx, s1Idx)`
- S2 分片范围：
  - `sInnerLoopSize = ceil(effectiveS2Size / splitKVNum)`
  - `fdS2Start = sInnerLoopSize * flashDecodeS2Idx`
  - `fdS2End = min(fdS2Start + sInnerLoopSize, effectiveS2Size)`
- 空闲核保护：`aicIdx >= totalS1 * splitKVNum` 时直接 return（但仍参与 SyncAll）

**splitKVNum 计算（Tiling 侧 IsFlashDecode + SplitBalanced）：**
- 触发条件：`bs1 < 0.4 * coreNum` 且 `maxActualseqKV >= 256`
- GQA 场景额外条件：`gSize <= 128` 且 `maxActualseqKV >= 2048`
- 计算：`splitNum = aicNum / bs1`，然后 while 循环确保每片 S2 >= 256

### F4 Buffer 4 层设计

**关键维度参数：**

| 参数 | 值 | 来源 | 说明 |
|------|---|------|------|
| s1BaseSize | 64 | tiling | Query tile 行数 |
| s2BaseSize | 128 | tiling | KV tile 列数 |
| dSize | 576 | common | Q head dim (nope512 + rope64) |
| dSizeV / dSizeNope | 512 | common | V/输出 head dim |
| dSizeVInput | 672 | common | KV 输入 dim (656→672 对齐) |
| CV_RATIO | 2 | common | AIC:AIV = 1:2 |
| Q_T sizeof | 2B (BF16/FP16) | OpDef | Query 数据类型 |
| KV_T sizeof | 1B (FP8) | OpDef | KV 数据类型 |
| T sizeof | 4B (float) | common | 中间计算类型 |

**Phase 1 L1（与 base 一致，无变化）：**

| Buffer | 内容 | 大小公式 | 大小 | BUF_NUM | 总大小 |
|--------|------|---------|------|---------|-------|
| l1RightBuffers | KV tile (NZ 格式) | max(576, 512) × 128 × 2B = 147,456 B | 144 KB | 3 (triple) | 432 KB |

L1 初始化：524,288 B = 512 KB（`l1BufferManager.Init(pipe, 524288)`）
Q 左矩阵在 S2 循环首轮通过 `CopyToL1Nd2Nz` 加载到 L1，后续复用 `l1QBuffers.GetPre()`。

**Phase 1 L0A/L0B/L0C（Matmul 库管理，与 base 一致）：**

| Buffer | 内容 | 大小 | BUF_NUM | 总大小 |
|--------|------|------|---------|-------|
| L0A (mmL0ABuffers) | BMM 左操作数 | 16 KB | DB(×2) | 32 KB |
| L0B (mmL0BBuffers) | BMM 右操作数 | 32 KB | DB(×2) | 64 KB |
| L0C (mmL0CBuffers) | BMM 结果 | 128 KB | DB(×2) | 256 KB |

**Phase 1 UB（与 base 一致，无变化）：**

| Buffer | Position | 大小公式 | 大小 | BUF_NUM | 总大小 | 备注 |
|--------|----------|---------|------|---------|-------|------|
| bmm1Buffers | CROSS_CORE_SYNC_BOTH | 32×128×4B | 16 KB | DB(×2) | 32 KB | AIC写/AIV读 |
| bmm2Buffers | CROSS_CORE_SYNC_BOTH | 32×512×4B | 64 KB | 1(Single) | 64 KB | AIC写/AIV读 |
| stage0InQue | VECIN | 672×16×1B | 10.5 KB | 2 | 21 KB | Vec0 KV 输入 |
| stage0OutQue | VECOUT | 576×17×2B | 19.1 KB | 2 | 38.3 KB | Vec0 dequant 输出 |
| stage1OutQue[0] | VECOUT | 33×128×2B | 8.25 KB | 1 | 8.25 KB | Vec1 softmax→L1 |
| stage1OutQue[1] | VECOUT | 33×128×2B | 8.25 KB | 1 | 8.25 KB | Vec1 pingpong |
| stage2OutBuf | TBuf | 32×512×4B | 64 KB | 1 | 64 KB | Vec2 累积结果 |
| softmaxMaxBuf[0,1] | TBuf | 256B | 2 | 512 B | softmax max |
| softmaxSumBuf[0,1] | TBuf | 256B | 2 | 512 B | softmax sum |
| softmaxExpBuf[0,1] | TBuf | 256B | 2 | 512 B | softmax exp |
| commonTBuf | TBuf | 512B | 1 | 512 B | 临时空间 |

Phase 1 UB 汇总：
- CrossCore 部分：16K×2 + 64K = 96 KB
- AIV 独占部分：1.5KB + 0.5KB + 21KB + 38.3KB + 16.5KB + 64KB ≈ 141.8 KB
- **Phase 1 UB 总计 ≈ 237.8 KB**

**Phase 2 UB（[变更] FD 新增，tPipe->Reset() 后重新分配）：**

| Buffer | Position | 大小 | BUF_NUM | 总大小 | 备注 |
|--------|----------|------|---------|-------|------|
| lseTmpBuff | TBuf | 32 KB | 1 | 32 KB | 归约临时空间+scale 结果 |
| softmaxMaxInputQue | VECIN | 32 KB | 1 | 32 KB | 加载各分片 softmaxMax |
| softmaxSumInputQue | VECIN | 32 KB | 1 | 32 KB | 加载各分片 softmaxSum |
| accumOutInputQue | VECIN | 32 KB | 1 | 32 KB | 加载各分片 accumOut |
| FDResOutputQue | VECOUT | 32 KB | 1 | 32 KB | 最终归约结果输出 |

**Phase 2 UB 总计：5 × 32 KB = 160 KB**

> Phase 1 和 Phase 2 的 UB 分时复用（tPipe->Reset() 释放后重新分配），不叠加。

**Workspace（[变更] FD 新增三块 GM 区域）：**

| Workspace | 大小公式 | 数据类型 | 大小 | 备注 |
|-----------|---------|---------|------|------|
| accumOut | aicNum × 2 × n2Size × mBaseSize × dSizeV × 4B = aicNum × 2 × 1 × 64 × 512 × 4 | FP32 | aicNum × 256 KB | 各分片 attention 中间结果 |
| softmaxFDMax | 2 × aicNum × 2 × n2Size × mBaseSize × 8 × 4B | FP32 | aicNum × 8 KB | 各分片 softmax max |
| softmaxFDSum | 同 softmaxFDMax | FP32 | aicNum × 8 KB | 各分片 softmax sum |

FD workspace 总计：aicNum × (256 + 8 + 8) KB = aicNum × 272 KB

**硬件容量约束校验汇总：**

| 存储层级 | 硬件容量 | 代码初始化 | 实际使用 | 使用率 | 校验结果 |
|---------|---------|-----------|---------|-------|---------|
| UB | 248 KB | Phase1: 各Queue分配; Phase2: Reset后重分配 | Phase1: 237.8 KB; Phase2: 160 KB | Phase1: 95.9%; Phase2: 64.5% | ✅通过 ⚠️ Phase1 接近上限 |
| L1 | 1 MB | 512 KB | 432 KB (l1Right) + Q tile | 84.4% | ✅通过 |
| L0A | 64 KB | 64 KB | ≤64 KB | ≤100% | ✅通过 |
| L0B | 64 KB | 64 KB | ≤64 KB | ≤100% | ✅通过 |
| L0C | 256 KB | 256 KB | ≤256 KB | ≤100% | ✅通过 |
| Workspace | GM | 动态计算 | aicNum × 272 KB | — | ✅通过 |

> ⚠️ Phase 1 UB 使用率 95.9%，接近 248KB 上限。FD 特性不增加 Phase 1 UB 消耗（Bmm2FDOut 复用 stage2OutBuf），风险可控。

### F5 Online Softmax 精度链

**继承判定：Phase 1 完全继承，Phase 2 新增全 FP32 归约链**

**Phase 1 精度链（与 base 一致）：**

| 计算步骤 | 输入 dtype | 计算 dtype | 是否需 Cast | 精度说明 |
|----------|------------|------------|-------------|----------|
| KV 搬运 | FP8_E4M3(1B) | — | — | GM → UB |
| KV Dequant (VF) | FP8 | FP32 | FP8→FP32→×scale→FP32→BF16 | AntiquantVFImplFp8D448 |
| KV UB→L1 | BF16 | — | — | NZ 格式搬运 |
| BMM1 (Q×K^T) | Q:BF16, K:BF16 | FP32 | Cube 自动升精 | Cube 输出 FP32 |
| Softmax Scale | FP32 | FP32 | 否 | × scaleValue |
| Softmax RowMax | FP32 | FP32 | 否 | VF 实现 |
| Softmax Sub+Exp | FP32 | FP32 | 否 | 先减 max 再 exp |
| Softmax RowSum | FP32 | FP32 | 否 | VF 实现 |
| Softmax Normalize | FP32 | FP32 | 否 | ×(1/sum) |
| Cast P | FP32 → BF16 | — | 显式降精度 | BMM2 输入要求 |
| BMM2 (P×V) | P:BF16, V:BF16 | FP32 | Cube 自动升精 | Cube 输出 FP32 |
| FlashUpdate | FP32 | FP32 | 否 | old × exp_factor + new |
| Last Div | FP32 | FP32 | 否 | 最终除以 sum |
| **[base] Output Cast** | FP32 → BF16 | — | 显式降精度 | 直接输出 |
| **[FD] Bmm2FDOut** | FP32 | FP32 | 否 | 写入 workspace，不做 Cast |

**Phase 2 归约精度链（[变更] FD 新增）：**

| 计算步骤 | 输入 dtype | 计算 dtype | 是否需 Cast | 精度说明 |
|----------|------------|------------|-------------|----------|
| 加载 accumOut | FP32(GM) | FP32 | 否 | 各分片 attention 中间结果 |
| 加载 softmaxMax/Sum | FP32(GM) | FP32 | 否 | 各分片统计量 |
| globalMax 计算 | FP32 | FP32 | 否 | max(各分片 max) |
| rescale 计算 | FP32 | FP32 | 否 | exp(max_k - globalMax) × sum_k |
| 归一化 scale | FP32 | FP32 | 否 | rescale_k / totalSum |
| 加权求和 | FP32 | FP32 | 否 | Σ scale_k × accumOut_k |
| Output Cast | FP32 → OUTPUT_T | — | 显式降精度 | 最终降精度 |

**精度验收标准：**

| 输入 dtype | atol | rtol | 说明 |
|------------|------|------|------|
| BF16 | 2e-2 | 1e-2 | FD 结果对比非 FD 结果 |
| FP16 | 1e-3 | 1e-3 | FD 结果对比非 FD 结果 |

### F6 Mask 处理

**继承判定：完全继承，FD 不影响 Mask 处理**

| Mask 类型 | 数据来源 | 应用方式 | FD 影响 |
|-----------|---------|---------|---------|
| sparse_mode=0 (全计算) | 无 | 全量计算 | 无影响，各分片独立处理 |
| sparse_mode=3 (rightDownCausal) | 计算生成 | s2LineStart/End 裁剪 | 无影响，各分片按自身 S2 范围裁剪 |
| Sparse Indices | sparseIndices tensor | 离散索引取 KV | 无影响，各分片按自身 S2 偏移索引 |

FD 模式下，每个分片的 S2 循环范围由 `fdS2Start ~ fdS2End` 限定，sparse_mode 的 s2LineStart/End 裁剪在分片范围内独立计算，不影响 Mask 语义。

### F7 确定性计算

**继承判定：继承，FD 满足确定性要求**

- QSFA-Pioneer 是前向推理算子，不涉及梯度累加
- Phase 1：各核处理不同 S2 分片，无竞争写入（accumOut 按核索引偏移写入独立区域）
- Phase 2：每个 (b, n2) 只有一个 AIV 核执行归约（`aivIdx = bIdx × N2 + n2Idx`），不存在多核写同一地址
- Online Softmax 的分片归约通过 LogSumExp 数学等价保证数值一致性

**结论：不需要额外确定性保护，FD 模式天然满足确定性。**

### F8 异常场景

| 场景 | 处理方式 | FD 特殊处理 |
|------|----------|------------|
| actualSeqKV = 0 | tiling 侧不启用 FD | N/A |
| actualSeqKV < 256 | IsFlashDecode 返回 false | N/A |
| FD 尾片 S2 < s2BaseSize | actualSInnerLoopSize 动态裁剪 | s2RealSize 按实际值处理 |
| 分片起始超出 actualSeqKV | actualSInnerLoopSize = 0，核直接 return | Phase 2 中 actualCombineLoopSize 自动排除空分片 |
| sparseIndices 中有 -1 | 现有 sparse 搬运逻辑 token < 0 跳过 | FD 不影响 |
| B × splitKVNum > aicNum | tiling 侧调整 splitKVNum ≤ aicNum/bs1 | N/A |
| 空闲核未参与 SyncAll | Process 中 SyncAll 放在 ProcessMainLoop 之外 | **关键保护点** |
| gSize > 128 | IsFlashDecode 返回 false | 避免 Phase 2 workspace 过大 |
| hasSink + FD | effectiveS2Size += s2BaseSize(128) | sink 块计入 S2 总长度参与切分 |

### F9 同步正确性检查

| 检查项 | 检查内容 | 检查结果 | 风险等级 | 备注/修复方案 |
|--------|---------|---------|---------|--------------|
| 配对完整性 | Set/Wait 1:1 配对 | ✅通过 | 高 | Phase 1 继承 BuffersPolicy 自动管理；Phase 2 SyncAll 无需手动配对 |
| 循环依赖 | 依赖图无环 | ✅通过 | 高 | Phase 1 单向流水 Vec0→BMM1→Vec1→BMM2→Vec2；Phase 2 SyncAll→单核归约，无反向依赖 |
| CrossCore 访问序 | 读写同步完整 | ✅通过 | 高 | Phase 1 继承 base；Phase 2 无 CrossCore Buffer（归约在单 AIV 核内完成，数据通过 GM workspace 传递） |
| 事件 ID 冲突 | ID 无冲突 | ✅通过 | 中 | 继承 base 事件 ID；FD 不新增手动 SetFlag/WaitFlag（SyncAll 是独立原语） |
| Ping-Pong 隔离 | 奇偶事件分离 | ✅通过 | 中 | 继承 base DB/3buff 机制 |
| **空闲核 SyncAll** | 所有核参与屏障 | ⚠️需确保 | 高 | SyncAll 必须放在 ProcessMainLoop 之外，空闲核（aicIdx >= actualCoreNums）跳过 ProcessMainLoop 后仍执行 SyncAll |
| **综合判定** | 全部通过方可编码 | ✅通过 | — | 关键点：SyncAll 位置正确性 |

---

## Regbase 专项（R1-R3，A5 平台）

### R1 VF 函数结构设计

**Phase 1 VF 函数（继承 base，不修改）：**

| VF 函数 | 文件 | 用途 | 修饰 |
|---------|------|------|------|
| AntiquantVFImplFp8D448 | vf/vf_basic_block_utils.h 等 | Vec0: KV FP8 反量化 | `__simd_vf__` |
| ProcessVec1Vf (6 条路径) | vf/vf_mul_sel_softmaxflashv2_cast_nz_qsfa.h | Vec1: Softmax 全流程 | `__simd_vf__` |
| FlashUpdateNew / FlashUpdateLastNew / LastDivNew | vf/vf_flashupdate_new_qsfa.h | Vec2: Flash Update | `__simd_vf__` |
| QSFAUpdateExpSumAndExpMax | vf/vf_mul_sel_softmaxflashv2_cast_nz_qsfa.h | Vec1: Online Softmax 状态更新 | `__simd_vf__` |

**Phase 2 FD 归约 VF 函数（[变更] 引用 attention_pioneer 已有实现）：**

| VF 函数 | 来源文件 | 用途 | MicroAPI 序列 | RegTensor 需求 |
|---------|---------|------|--------------|---------------|
| ComputeScaleValue_VF | vf/vf_flash_decode.h | 计算归约权重 scale[k] | Load(max)→Max→Sub→Exp→Load(sum)→Mul→Sum→Rcp→Mul→Store | 5 个 |
| ReduceFinalRes_0_VF | vf/vf_flash_decode.h | 首片：dst = scale[0] × accumOut[0] | Load(accumOut)→Load(scale)→Mul→Store | 3 个 |
| ReduceFinalRes_Rest_VF | vf/vf_flash_decode.h | 后续片：dst += scale[k] × accumOut[k] | Load(dst)→Load(accumOut)→Load(scale)→Mul→Add→Store | 4 个 |

**VF 函数参数适配（QSFA-Pioneer 特定）：**
- `dLoops = dSizeV / 64 = 512 / 64 = 8`（每次 VF 处理 64 个 FP32 元素）
- `dealRowCount` = 当前 gSplitSize 内的有效行数
- ComputeScaleValue_VF 按 g≤8 / 8<g≤16 分档（ComputeScaleValue_8_VF / ComputeScaleValue_16_VF）

### R2 Scalar 前移审查

**Phase 1（继承 base，不涉及修改）：**
所有 Scalar 计算已在 VF 外部完成，通过参数传入。

**Phase 2 FD 归约 VF 函数：**

| Scalar 计算内容 | 当前位置 | 处理方案 |
|----------------|---------|---------|
| splitKVNum | VF 外 | ✅ constInfo 传入 |
| dealRowCount | VF 外 | ✅ CombineSplitKVRes 计算后传入 |
| dLoops = dSizeV/64 | VF 外 | ✅ 编译期常量 |
| gSplitSize | VF 外 | ✅ CombineSplitKVRes 计算后传入 |
| VF 内 for 循环计数器 | VF 内 | ✅ 必须保留（i++, j++） |
| scale 偏移地址 | VF 外 | ✅ 通过 lseTmpBuff 偏移传入 |

**结论：所有可移出的 Scalar 计算已在 VF 外部完成，无需修改。**

### R3 寄存器分配策略

**Phase 2 FD 归约 VF 函数寄存器评估：**

| VF 函数 | RegTensor 数量 | 2×VL 模式 | 溢出风险 | 说明 |
|---------|---------------|-----------|---------|------|
| ComputeScaleValue_VF | 5 | 否 | 无 | 远小于 256 上限 |
| ReduceFinalRes_0_VF | 3 | 否 | 无 | 极轻量 |
| ReduceFinalRes_Rest_VF | 4 | 否 | 无 | 极轻量 |

**否定性论证（Phase 1 VF 不涉及修改）：**
1. 本次 FD 特性注入不修改任何 Phase 1 VF 函数（AntiquantVF、SoftmaxVF、FlashUpdateVF 均继承 base）
2. Phase 2 新增的 3 个 VF 函数来自 attention_pioneer 已验证实现，寄存器需求极低（≤5 个）
3. 不涉及 regbase 文件的修改（util_regbase.h 仅新增 ConstInfo/RunParamStr 字段，不涉及 VF 寄存器）

---

## 流水效率专项（E1-E3）

### E1 Bubble 分析

**Phase 1 Bubble（与 base 一致，FD 不影响）：**

三任务槽流水时序：
```
taskId:    0          1          2          3          4       ...
AIC:    BMM1[0]    BMM1[1]    BMM1[2]    BMM1[3]    (排空)
                   BMM2[0]    BMM2[1]    BMM2[2]    BMM2[3]
AIV:    Vec0[0]    Vec0[1]    Vec0[2]    Vec0[3]    (排空)
                   Vec1[0]    Vec1[1]    Vec1[2]    Vec1[3]
                              Vec2[0]    Vec2[1]    Vec2[2]
```

瓶颈分析：Vec0 阶段（KV FP8 反量化 + 离散稀疏搬运）耗时较长，可能成为流水瓶颈。此为 base 已知特性，FD 不影响。

**FD 对 Phase 1 Bubble 的影响：**
- 每个分片的 S2 循环次数减少（`sInnerLoopSize / s2BaseSize` 轮），流水排空（PRELOAD_NUM=2）的相对开销增大
- 当 splitKVNum 过大导致每片 S2 < 256 时，tiling 侧会减小 splitKVNum 保护

**Phase 2 Bubble（无流水，顺序执行）：**
```
CopyLseIn → ComputeScaleValue_VF → [for each split: CopyAccumOutIn → ReduceFinalRes_VF] → CopyFinalResOut
```
Phase 2 耗时主要取决于 splitKVNum × dSizeV 的搬运和计算量。典型场景（G=1, splitKVNum=4~8）Phase 2 总耗时约占 kernel 时间的 5%~10%。

### E2 Double Buffer / Preload 策略

**Phase 1（继承 base，无变化）：**

| 优化措施 | 状态 | 说明 |
|---------|------|------|
| L1 3-Buffer Preload | ✅启用 | l1RightBuffers 3buff, Vec0 预加载下一块 KV |
| UB bmm1 Double Buffer | ✅启用 | bmm1Buffers DB |
| UB bmm2 Single Buffer | 单 Buffer | dSizeV=512 导致 buffer 较大，无法 DB |
| Vec0 Double Buffer | ✅启用 | stage0InQue/stage0OutQue 各 2 buffer |
| stage1OutQue Ping-Pong | ✅启用 | 2 份 stage1OutQue |

**Phase 2（无 Double Buffer）：**
归约计算量较小，且 Reset UB 后只有 32K 级别缓冲区，不需要 Double Buffer。

### E3 带宽瓶颈评估

**瓶颈类型：Memory Bound**

QSFA-Pioneer 已有 KV FP8 反量化开销（FP8→BF16 数据量翻倍），Vec0 的 GM→UB 搬运和 UB→L1 搬运是主要带宽消耗。BMM1/BMM2 的 Cube 计算相对较快（s1BaseSize=64, s2BaseSize=128, dSizeV=512）。

**FD 对带宽的影响：**
- Phase 1：每个分片的 GM 搬运量与 base 相同（只是分片范围不同），总带宽不变
- Phase 2：新增 accumOut 搬运（splitKVNum × 64 × 512 × 4B per (b,n2)），但仅执行一次，开销较小
- Workspace 写入：Phase 1 末尾新增 accumOut + softmaxMax/Sum 的 GM 写入，增加约 (256K + 16K) per core 的写带宽

**结论：FD 不改变瓶颈类型，Phase 2 的额外带宽开销在可接受范围内。**

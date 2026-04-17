# 阶段 1 产出：FA 算子识别与基座匹配

## 算子基本信息
- 算子名称：kv_quant_sparse_flash_attention_pioneer (QSFA-Pioneer)
- 概要设计文档路径：`attention/kv_quant_sparse_flash_attention_pioneer/docs/kv_quant_sparse_flash_attention_pioneer_需求概设.md`
- 代码梳理文档路径：`attention/kv_quant_sparse_flash_attention_pioneer/docs/kv_quant_sparse_flash_attention_pioneer_代码梳理.md`
- 目标代码仓路径：`attention/kv_quant_sparse_flash_attention_pioneer/`
- 参考实现路径：`attention/attention_pioneer/op_kernel/common/arch35/` (isfd 分支)

## 开发场景
- 场景：S1 特性注入
- 判定依据：概设明确描述"在已有 QSFA-Pioneer 算子上新增 Flash Decode (FD) 多核 S2 轴并行加速特性"，算子目录已存在，核心流水/Buffer/同步架构不变，仅新增 FD 分支
- 基座算子：kv_quant_sparse_flash_attention_pioneer 自身（Phase 1 流水完全继承 base）
- 注入特性：Flash Decode — 多核切分 S2 轴并行计算 + Phase 2 全局归约

## 基座特征继承

### 流水设计：继承 + 新增 Phase 2
- Phase 1（5 阶段三级流水）：完全继承，Vec0→BMM1→Vec1→BMM2→Vec2 不变
- Phase 2（FD 归约）：**新增**，SyncAll → tPipe->Reset() → InitFDBuffers → FlashDecodeCompute
- CV 比例：不变，AIC:AIV = 1:2

### Buffer 设计：继承 + 新增
- Phase 1 UB/L1/L0：完全继承，无修改
- Phase 2 UB：**新增** 5×32KB 缓冲区（tPipe->Reset() 后重新分配，与 Phase 1 分时复用）
- Workspace：**新增** accumOut + softmaxFDMax + softmaxFDSum 三块 GM 区域

### 分核逻辑：需修改
- base：B×N2×S1 三维均衡分核（Kernel 侧动态）
- FD：每个 (B, S1) 块固定分配 splitKVNum 个核，S2 按实际长度非均匀切分
- InitCalcParamsEach 新增 `if constexpr (isFd)` 分支

### 精度链：继承
- Phase 1 精度链完全继承（FP8→FP32→BF16→Cube FP32→Softmax FP32→Cast BF16→Cube FP32→FlashUpdate FP32）
- Phase 2 归约全程 FP32，最终 Cast 到 OUTPUT_T

### 同步事件：继承 + 新增
- Phase 1 CrossCore 同步：完全继承（bmm1Buffers DB / bmm2Buffers Single / l1RightBuffers 3buff）
- Phase 2：**新增** SyncAll() 全核屏障（AIC+AIV 均参与）

## 目标硬件平台
- 平台：A5（arch35, Ascend 950）
- 唯一支持平台，AIC/AIV 异构

## 从概要设计提取的关键信息

### 算法描述
在 base QSFA-Pioneer 的 Sparse Attention + MLA-Absorb + FP8 KV Cache 基础上，新增 Flash Decode 特性：
- 当核利用率不足（B×S1 < 0.4×coreNum）且 S2 足够长（≥256）时，将 S2 轴切分为 splitKVNum 份
- 每份由独立核并行执行 Phase 1（与 base 完全相同的 5 阶段流水）
- Phase 1 结束后，通过 SyncAll 全核屏障同步
- Phase 2 由前 B×N2 个 AIV 核执行全局归约：加载各分片的 accumOut/softmaxMax/softmaxSum → 计算归约权重 → 加权求和 → 输出

### 输入张量（与 base 一致，无新增）
| 名称 | 数据类型 | Shape | 说明 |
|------|---------|-------|------|
| query | BF16/FP16 | [B,S1,Q_N,576] / [Q_T,Q_N,576] | Q 输入 |
| key | FP8_E4M3/HIFLOAT8 | [block_num,block_size,1,656] / [B,S2,1,656] / [KV_T,1,656] | K 输入 |
| value | FP8_E4M3/HIFLOAT8 | 同 key | V 输入（MLA 共享） |
| sparse_indices | INT32 | [B,Q_S,KV_N,sparse_size] / [Q_T,KV_N,sparse_size] | 稀疏索引 |
| key_dequant_scale | FP32 | - | 预留参数 |
| value_dequant_scale | FP32 | - | 预留参数 |
| block_table | INT32 | [B, S2/block_size] | PA block 映射表 |
| actual_seq_lengths_query | INT32 | (B,) | Q 有效 token 数 |
| actual_seq_lengths_kv | INT32 | (B,) | KV 有效 token 数 |
| key_sink | BF16/FP16 | [128, KV_N, D] | Sink Key |
| value_sink | BF16/FP16 | [128, KV_N, D] | Sink Value |

### 输出张量（与 base 一致）
| 名称 | 数据类型 | Shape | 说明 |
|------|---------|-------|------|
| attention_out | BF16/FP16 | [B,S1,Q_N,512] / [Q_T,Q_N,512] | 输出 |

### 属性参数（无新增，isFd 通过 TilingKey 模板参数控制）
与 base 一致，15 个属性不变。

### 计算流图摘要
```
Phase 1 (splitKVNum 个核并行):
  Vec0(KV搬运+FP8反量化) → BMM1(Q×K^T) → Vec1(Softmax) → BMM2(P×V) → Vec2(FlashUpdate)
  → [FD] Bmm2FDOut → accumOutGm
  → [FD] ComputeLogSumExpAndCopyToGm → softmaxMaxGm/sumGm

Phase 2 (前 B×N2 个 AIV 核):
  SyncAll → Reset UB → CopyLseIn → ComputeScaleValue_VF → ReduceFinalRes_VF → CopyFinalResOut
```

### TilingKey 变更
- 新增 `FLASH_DECODE` 布尔位（已预留声明）
- isFd 作为编译期模板参数，通过 `if constexpr (isFd)` 隔离 FD 分支

### 参考实现关键发现（attention_pioneer isfd 分支）
1. **Process() 中 SyncAll 位置**：仅 AIV 侧调用 `SyncAll()` + `InitFDBuffers()` + `FlashDecodeCompute()`，AIC 在 ProcessMainLoop 返回后自然退出
2. **FD 分核**：`aicIdx / (n2Size * splitKVNum)` 解码 boIdx，`(aicIdx / splitKVNum) % n2Size` 解码 n2oIdx
3. **sInnerLoopSize**：`CeilDiv(s2Size, splitKVNum)` — 每个分片的 S2 循环次数
4. **Phase 2 归约**：CombineSplitKVRes 按 gSplitSize 分片处理 G 轴，每片内 CopyLseIn → ComputeScaleValue_VF → 逐分片 ReduceFinalRes_VF → CopyFinalResOut
5. **VF 函数**：ComputeScaleValue_VF 按 g≤8 / 8<g≤16 分档；ReduceFinalRes_VF 按 splitKVIndex==0 / >0 分档

## 概设文档补充记录

概设文档信息较为完整，覆盖了 TilingKey、Tiling 策略、Buffer 设计、Kernel 伪代码、同步设计、异常场景、精度分析和性能分析。

需确认的细节点：
1. **SyncAll 调用方式**：概设中描述"AIC+AIV 均参与 SyncAll"，但参考实现（attention_pioneer）中 SyncAll 仅在 AIV 侧调用（Process 中 `if ASCEND_IS_AIV { SyncAll(); ... }`）。QSFA-Pioneer 的 Process 结构与 attention_pioneer 不同（QSFA-Pioneer 的 AIC/AIV 通过编译期分支实例化不同服务类），需确认 SyncAll 的调用位置。
2. **FD 模式下 hasSink 的交互**：概设提到 `effectiveS2Size += s2BaseSize` 处理 sink，但参考实现中 sink 处理方式可能不同（sinkBlockCnt 机制），需在详设中明确。
3. **gSize 对 Phase 2 的影响**：QSFA-Pioneer 的 gSize 可能较大（Q_N 支持 1~64），Phase 2 归约的 gSplitSize 计算需适配。

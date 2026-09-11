# MQSMLA `IS_BATCH_CONSISTENCY` 性能劣化静态分析报告（N1=64）

分析对象：`attention/mixed_quant_sparse_flash_mla/op_kernel/arch35`
开关语义：`IS_BATCH_CONSISTENCY` 由 tiling key 决定，运行时由 `context_->GetDeterministicLevel() == 3` 触发
（`op_host/mixed_quant_sparse_flash_mla_tiling.cpp:128`，`BATCH_CONSISTENCY_LEVEL = 3`）。

> 本文按 **N1=64（每核 M=64，`s1BaseSize=64`；每 AIV 子核 `halfMRealSize=32`）** 推导。
> N1=128 见同目录 `batch_consistency_perf_analysis_N128.md`。

## 1. 结论

**完整建模 CV 同步 + buffer 释放 + FIFO 阻塞，三档 HBM 带宽下的结果（详见第 6~8 节）**：

| 场景 | 不开启 | 开启 | 耗时比 | 性能比 |
|---|---|---|---|---|
| HBM 4T（题给） | 122,133 cyc | 214,426 cyc | **1.76×** | 0.57× |
| HBM 1.6T（V100） | 300,094 cyc | 502,567 cyc | **1.68×** | 0.60× |
| HBM 1.4T（V100） | 342,466 cyc | 574,308 cyc | **1.68×** | 0.60× |

**结论：开启后耗时约为不开启的 1.68~1.76 倍，性能约为不开启的 0.57~0.60 倍。**
- 4T：不开启瓶颈 AIV-MTE3（V0 写回，3556 cyc/块）；开启临界环 `m→a→b→f→h→m = 6469 cyc/块`。
- 1.6T / 1.4T：HBM 变慢后 V0 MTE3 绝对值增大，BC 额外开销被摊薄，瓶颈统一为 AIV-MTE3，劣化略降。
- 1.6T ↔ 1.4T 比值几乎不变（HBM 阶段近似等比缩放），仅绝对耗时 ×1.14。

## 2. 两条路径的差异（代码依据）

### 2.1 规约粒度

- 关闭：一个核负责的整段 S2 用**在线 softmax 在 UB 内一次累加**，只在末尾 stage 一次 partial-O。
- 开启：S2 被切成 `reductionBlockSize = ceil(S2/32)` 向上取整到 128 的**规约块**，每 token 最多
  `BATCH_CONSISTENCY_MAX_REDUCTION_PARTS = 32` 块
  （`mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/..._aicpu.cpp:972-985`）。

| S2 (topK) | 基本块 N | reductionBlockSize R | 规约块数 K |
|---|---|---|---|
| 1024 | 8 | 128 | 8 |
| 2048 | 16 | 128 | 16 |
| 4096 | 32 | 128 | 32 |
| 8192 | 64 | 256 | 32 |
| 16384 | 128 | 512 | 32 |

### 2.2 开启后新增的搬运与规约

1. **每个规约块末尾 stage max/sum**：`StageVec1Lse`（`..._csa_block_vector.h:1354-1396`）。
2. **每个规约块末尾 stage partial-O**：N1=64 时每核 `m=64`（`64×512×fp32 = 128KB`），
   每 AIV 子核 `halfMRealSize=32`（`64KB`）；`StageVec2PartialOAndWait`（`flash_decode.h:214-224`）。
3. **核内规约合并**：`ReduceIntraBlockAndStage` → `MergeStagedAndCurrentChunk`
   （`..._csa_block_vector.h:1434-1527`，`flash_decode.h:462-510`），每个规约块边界一次。
4. **Vec2 在每个规约块重启在线 softmax**：`isFirstBase`/`isLastBase` 分支（`:1554-1590`）。
5. **FD 末尾规约改为确定性左折叠**：`ReducePairwiseWithLse`（`flash_decode.h:352-453`）。

## 3. 成本模型

设 `m = gSize`，`D = 512`，`S = topK`，KV 为 fp8、dK≈576：

```
额外 Vector 计算   E_vec ≈ (K-1) × 2 × m × D
额外 partial-O 流量 E_gm ≈ (2K ~ 4K) × m × D × 4 Bytes
基线 Matmul MAC   ≈ m × S × (dK + D)
基线 KV 流量      ≈ S × (dK + 1) Bytes
```

## 4. 重要的反向因素

开启 BC 的规约块切分本身是负载均衡手段：关闭时每核 S2 切分上限 2，长序列/小 batch 下核利用率可能不足；
开启后可把单 token 的 S2 切到最多 32 个核上并行。因此在**核数富余、S2 很长**的 shape 下，开启后可能持平甚至更快。

## 5. 建议

- 若需确定倍数，建议在 Ascend950 上按 `set_deterministic_level(0)` vs `(3)` 跑同一批 shape，
  抓 `Task Duration(us)` 求比值（`tests/pytest/perf_run.sh` + `process_perf_data.py`）。
- 优化方向：merge 用更大 `chunkRows` 减少 GM 往返、partial-O 用 fp16/bf16 stage 降低带宽。

## 6. Pipeline 差异与阻塞点分析（CV 同步 + buffer 释放 + FIFO）

### 6.1 建模前提

规格：`b = 核数`、`s1 = 1`、`s2 = 4096`、`d = 512`，每核一个 batch 的整段 S2。
32 个规约块、每块 1 个基本块、核内两两规约 31 次。
Cube 与 Vector 独立指令队列；同一队列严格 FIFO，前一条阻塞则后续无法发射。

单基本块各阶段 cycle（题给）：

| 阶段 | 队列 | cycles | 依赖 |
|---|---|---|---|
| V0 MTE2 `GM→UB` | AIV-MTE2 | 2997 | — |
| V0 Vector（dequant） | AIV-V | 420 | V0 MTE2 |
| V0 MTE3 `UB→GM` | AIV-MTE3 | **3556** | V0 Vector |
| KV MTE2 `GM→L1` | AIC-MTE2 | 1852 | **V0 MTE3（CV 同步）** |
| Matmul1 (QK) | AIC-Cube | 1024 | KV 搬入 |
| V1 Vector（softmax） | AIV-V | 600 | **Matmul1（CV 同步）** |
| Matmul2 (PV) | AIC-Cube | 1024 | **V1（CV 同步）** |
| V2 Vector（O 累加） | AIV-V | 50 | **Matmul2（CV 同步）** |
| BC：V2 merge 读回 `GM→UB` | AIV-MTE2 | 2402 | V2 Vector / 上一块 merge 结果 |
| BC：V2 merge 写回 `UB→GM` | AIV-MTE3 | 2402 | merge 读回 |

队列发射顺序：

| 队列 | 每 loop 发射顺序 | 单拍 |
|---|---|---|
| AIV-MTE2 | `a_t`(V0 load) → `m_{t-3}`(BC merge read) | 2997 / 2402 |
| AIV-MTE3 | `c_t`(V0 write) → `n_{t-3}`(BC merge write) | 3556 / 2402 |
| AIV-V | `b_t`(V0 vec) → `f_{t-2}`(V1) → `h_{t-3}`(V2) | 420 / 600 / 50 |
| AIC-MTE2 | `Q`（一次） → `d_{t-1}`(KV GM→L1) | 1201 / 1852 |
| AIC-Cube | `e_{t-2}`(Matmul1) → `g_{t-3}`(Matmul2) | 1024 / 1024 |

Buffer 深度：stage0In=2, stage0Out=2, v0ResGm=3, **L1K=3**, bmm1=2, l1P=2, bmm2=1。

### 6.2 不开启：AIV-MTE3（V0 写回）独占瓶颈

| 队列 | busy |
|---|---|
| AIV-MTE2 | 95,904 |
| AIV-MTE3 | **113,792** |
| AIV-V | 32,640 |
| AIC-MTE2 | 60,465 |
| AIC-Cube | 65,536 |

- 临界环：**AIV-MTE3 资源环 = 3556 cyc/块**（AIV-MTE2 2997、AIC-Cube 2048 均被掩盖）。
- **makespan = 122,133 cyc**。

### 6.3 开启：merge 读回与 AIV-V 组成 6469 临界环

| 队列 | busy |
|---|---|
| AIV-MTE2 | 170,366 |
| AIV-MTE3 | 190,656 |
| AIV-V | 34,240 |
| AIC-MTE2 | 60,465 |
| AIC-Cube | 65,536 |

- 临界环：
  ```
  m_i (2402) → a_{i+4} (2997) → b_{i+4} (420) → f_{i+2} (600) → h_{i+1} (50) → m_{i+1} (2402)
  周期 = 2402 + 2997 + 420 + 600 + 50 = 6469 cyc / 块
  ```
  （AIV-MTE3 的 `c+n = 3556+2402 = 5958 < 6469`，未成为瓶颈）
- **makespan = 214,426 cyc**。

### 6.4 开启后具体被阻塞的位置

1. **AIV-MTE2 被 `m` 抢占（最核心）**：`m_i`（2402）与 `a_{i+4}`（2997）同队列 FIFO，V0 搬入被推迟。
2. **`m_i` 被 AIV-V 队列尾部阻塞**：`m_i` 依赖 `h_i`，`h_i` 排在 `b_{i+3}`、`f_{i+1}` 之后，额外气泡 `b+f+h = 1070`。
3. **merge 链串行**：`m_i→n_i→m_{i+1}` = 4804/块。
4. **CV 同步传导**：`c_i→d_i→…→h_i→m_i` 闭环，AIV 侧推迟无法被 AIC（65,536）掩盖。
5. 开启后 AIV-MTE3 busy 涨到 190,656，但仍小于临界环，不构成瓶颈。

### 6.5 结论

- **开启后耗时 214,426 cyc，不开启 122,133 cyc，比值 1.76×；性能约为不开启的 0.57×。**
- 若只做"资源取 max"粗算，比值为 5958/3556 ≈ 1.68×；**CV 同步 + buffer + FIFO 额外贡献约 0.08×**。

## 7. V100（HBM 1.6T）带宽折算

题给数据在 HBM **4T** 环境测得；V100 环境 HBM **1.6T**，比值 **4/1.6 = 2.5×**。
**所有 HBM 读写阶段耗时 ×2.5**，片上阶段不变：

| 类别 | 阶段 | 折算 |
|---|---|---|
| HBM 读 | `Q MTE2`、`KV MTE2`(C1)、`V0 MTE2`、`BC merge MTE2` | ×2.5 |
| HBM 写 | `V0 MTE3`、`BC merge MTE3` | ×2.5 |
| 片上 | `Matmul1/2`、`V0 Vector`、`V1`、`V2` | 不变 |

折算后各阶段 cycle：

| 阶段 | 4T | 1.6T |
|---|---|---|
| Q MTE2 | 1201 | 3002 |
| KV MTE2 | 1852 | 4630 |
| V0 MTE2 | 2997 | 7492 |
| V0 Vector | 420 | 420 |
| V0 MTE3 | 3556 | 8890 |
| Matmul1 / Matmul2 | 1024 / 1024 | 1024 / 1024 |
| V1 / V2 | 600 / 50 | 600 / 50 |
| merge MTE2 / MTE3 | 2402 | 6005 |

结果：

| 队列 | 不开启 busy | 开启 busy |
|---|---|---|
| AIV-MTE2 | 239,744 | 425,899 |
| AIV-MTE3 | 284,480 | 476,640 |
| AIV-V | 32,640 | 34,240 |
| AIC-MTE2 | 151,162 | 151,162 |
| AIC-Cube | 65,536 | 65,536 |
| **makespan** | **300,094** | **502,567** |

- 不开启瓶颈：**AIV-MTE3 = 32×8890 = 284,480**（V0 写回）。
- 开启瓶颈：**AIV-MTE3 资源环 `c+n = 8890+6005 = 14,895 cyc/块`**（32×14,895 = 476,640；大于 `m→a→b→f→h→m = 14,567`）。
- **开启后耗时约为不开启的 1.68 倍，性能约为不开启的 0.60 倍。**

| 指标 | 4T（原） | 1.6T（折算） |
|---|---|---|
| 不开启 | 122,133 | 300,094 |
| 开启 | 214,426 | 502,567 |
| 耗时比 | 1.76× | **1.68×** |
| 性能比 | 0.57× | **0.60×** |

> HBM 变慢后 V0 MTE3 占比更大，BC 额外开销被摊薄，劣化比反而略降。

## 8. V100-1.4T 带宽折算

V100 1.4T 环境，HBM 4T→1.4T，比值 **4/1.4 ≈ 2.857×**。
HBM 读写阶段（`Q MTE2`、`KV MTE2`、`V0 MTE2`、`V0 MTE3`、`BC merge MTE2/MTE3`）×2.857；片上阶段不变。

折算后各阶段 cycle：

| 阶段 | 4T | 1.4T |
|---|---|---|
| Q MTE2 | 1201 | 3431 |
| KV MTE2 | 1852 | 5291 |
| V0 MTE2 | 2997 | 8563 |
| V0 Vector | 420 | 420 |
| V0 MTE3 | 3556 | 10160 |
| Matmul1 / Matmul2 | 1024 / 1024 | 1024 / 1024 |
| V1 / V2 | 600 / 50 | 600 / 50 |
| merge MTE2 / MTE3 | 2402 | 6863 |

结果：

| 队列 | 不开启 busy | 开启 busy |
|---|---|---|
| AIV-MTE2 | 274,016 | 486,769 |
| AIV-MTE3 | 325,120 | 544,736 |
| AIV-V | 32,640 | 34,240 |
| AIC-MTE2 | 172,743 | 172,743 |
| AIC-Cube | 65,536 | 65,536 |
| **makespan** | **342,466** | **574,308** |

- 不开启瓶颈：**AIV-MTE3 = 32×10160 = 325,120**。
- 开启瓶颈：**AIV-MTE3 资源环 `c+n = 10160+6863 = 17,023 cyc/块`**（32×17,023 = 544,736）。
- **开启后耗时约为不开启的 1.68 倍，性能约为不开启的 0.60 倍。**

三档带宽对比：

| 指标 | 4T | 1.6T | 1.4T |
|---|---|---|---|
| 不开启 | 122,133 | 300,094 | 342,466 |
| 开启 | 214,426 | 502,567 | 574,308 |
| 耗时比 | 1.76× | 1.68× | **1.68×** |
| 性能比 | 0.57× | 0.60× | **0.60×** |

> 1.4T 相对 1.6T 绝对耗时约 ×1.14（=2.857/2.5），比值基本不变。

> 说明：以上为基于代码结构与题给 cycle 规格的静态/仿真估算，未在硬件上实测。

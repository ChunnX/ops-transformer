# 设计文档 vs 代码实现 一致性审查报告

> **审查日期**: 2026-08-17
> **设计文档**: `design(1).md` (quantMatmulReduceScatter 设计文档初稿)
> **代码版本**: Round 4 (per-tile 流水 + 子问题 tiling + float32 累加)
> **审查范围**: kernel/quant_matmul_reduce_scatter_impl.h, kernel/quant_matmul_reduce_scatter_tiling_data.h, kernel/reduce_sum.h, src/kernel_launcher.h, src/main.cpp

---

## 一、功能定义（§2.1）

**设计文档**：
```
output = reduceSum(AlltoAll(Dequant(QuantA @ QuantB) + bias))
```
每卡完整 K，M 按 rankSize 切分，无 L0C 原子累加。

**代码实现**：
- `impl.h:279` — AIC LOCAL 模式 `MatmulMode::LOCAL, splitKNum=1`，无 `SetAtomicAdd`
- `impl.h:137` — `CollectiveComm<AllToAll, PUT, CType, TeamBarrier>` 通信 CType(bfloat16)
- `reduce_sum.h` — 逐 rank Add 累加
- **bias 未实现**：`kernel_launcher.h:22-28` 入参无 `biasGM`

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| MatMul + AlltoAll + reduceSum | ✅ | ✅ | 一致 |
| bias 支持 | 有 | **无** | ❌ 差异 1 |
| 无 L0C 原子累加 | ✅ | ✅ | 一致 |
| 每卡完整 K | ✅ | ✅ `main.cpp:130-132` | 一致 |

---

## 二、核角色分工（§3.2）

**设计文档**：AIC 做量化 MatMul，AIV 做 AlltoAll + reduceSum。

**代码实现**：
- `impl.h:225-237` — `if ASCEND_IS_AIC { RunMatmul() }` / `if ASCEND_IS_AIV { RunAllToAllAndReduceSum() }`
- `impl.h:35` — **仅 block 0 的 AIV 参与通信**（`totalJobs=1`）

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| AIC = MatMul | ✅ | ✅ | 一致 |
| AIV = AlltoAll + reduceSum | ✅ | ✅ | 一致 |
| 多少核参与通信 | 未明确 | 仅 block 0 | ❌ 差异 2（设计未指定） |

---

## 三、通信模式（§6.2）

**设计文档**：
```cpp
Apace::AivComm::CommMode::PUT,   // 或 GET，按硬件/场景选择
```

**代码实现**：
- `impl.h:137` — `CommMode::PUT`（固定）
- 用户补充信息要求 "PUT 模式改成 GET 模式"，但 GET 在 4+ rank 不稳定，回退到 PUT

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 通信模式 | PUT 或 GET | PUT | ❌ 差异 3（与用户要求 GET 不一致） |

---

## 四、同步机制（§6.4 + §8.3 + §8.4）

**设计文档 §8.3**：
```cpp
CrossCoreSetFlag<0x2, PIPE_MTE3>(0);  // AIC 通知 AIV
```

**代码实现**：
- `impl.h:317` — `CrossCoreSetFlag<0x2, PIPE_FIX>(0)`
- `impl.h:547` — `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)`

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| AIC SetFlag 管道 | `PIPE_MTE3` | `PIPE_FIX` | ❌ 差异 4（**设计文档有误**，AIC 侧无 PIPE_MTE3） |
| AIV WaitFlag 管道 | `PIPE_MTE2` | `PIPE_MTE2` | 一致 |
| SyncAll 每轮通信后 | ✅ | ✅ `impl.h:548,557` | 一致 |
| 不使用 AtomicAdd | ✅ | ✅ | 一致 |

---

## 五、流水线架构（§6.3 + §9.3）

**设计文档 §9.3**：
> "天然的两阶段流水：AIC 计算 C_i → AIV 通信+reduceSum"
> "若需要进一步重叠，可将 AIC 对 C 的 FixPipe 按 M 分片进行，让 AIV 边收边算（**未来优化方向**）"

**设计文档 §6.3 期望的流水**：
```
AIC: [完整计算 C_i]
AIV:                    [AlltoAll 分块发送] [收齐 rank_0 → reduceSum] [收齐 rank_1 → 累加] ...
```

**代码实现**（方案 D per-tile 三级流水）：
- `impl.h:318-391` — T×R 次子调用，每 tile 后 `SetFlag` 通知 AIV
- `impl.h:544-563` — per-tile `WaitFlag` → `AllToAll` → `reduceSum` 循环

```
AIC: [tile0 R段mm] SetFlag [tile1 R段mm] SetFlag ...
AIV:                WaitFlag [comm0+rs0]  WaitFlag [comm1+rs1] ...
```

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 流水架构 | 两阶段串行 | per-tile 三级流水 | ❌ 差异 5（**正向差异**，代码已实现设计中的"未来优化方向"） |
| 逐块求和 | "每收齐一个 rank → Add" | 每 tile 完成后对全部 R 个 rank 一次性 Add | ❌ 差异 6（求和粒度不同） |

---

## 六、reduceSum 精度策略（§6.3 + §7.3）

**设计文档**：未提及累加精度策略，隐含 BF16 直接 Add。

**代码实现**：
- `reduce_sum.h:23-26` — BF16 → Cast float32 → Add float32 → Cast 回 BF16

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 累加精度 | 未提及 | float32 累加 | ❌ 差异 7（**正向差异**，修复了 K=8192 精度问题） |

---

## 七、Kernel 入口签名（§8.1）

**设计文档**：
```cpp
void QuantMatmulReduceScatterKernel(
    __gm__ CommContext* hcommCtx,
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM,
    GM_ADDR biasGM, GM_ADDR cGM,
    QuantMatmulReduceScatterTilingData tilingData)
```

**代码实现**（`kernel_launcher.h:22-28`）：
```cpp
void QuantMatmulReduceScatterKernelE4M3E4M3(
    __gm__ CommContext *hcommCtx,
    GM_ADDR aGM, GM_ADDR scaleAGM,
    GM_ADDR bGM, GM_ADDR scaleBGM,
    GM_ADDR workspaceGM,    // 替代 biasGM
    GM_ADDR cGM,
    QuantMatmulReduceScatterTilingData tilingData)
```

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| biasGM 参数 | 有 | **无** | ❌ 差异 1（重复） |
| workspaceGM 参数 | 无 | **有** | ❌ 差异 8（用户要求新增） |
| 方法名 | `Init/Process` | `Init/Run` | ❌ 差异 9（命名差异） |
| 模板参数 | `<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t>` | `<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true>` | ❌ 差异 10（增加 TransA/TransB） |

---

## 八、AIC 输出目标（§4.3 + §8.3）

**设计文档 §8.3**：
> "FixPipe 将完整 C_i [m, n] 写出到 GM"

**代码实现**：
- `impl.h:292` — `mmadParams.cGmAddr = baseParams_.workspaceGm`（写入 workspaceGM，不是 cGM）

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| AIC 输出目标 | GM（隐含 output GM） | workspaceGM | ❌ 差异 11（用户要求 workspace） |

---

## 九、TilingData 结构体（§5.1）

**设计文档**：
```cpp
struct QuantMatmulReduceScatterTilingData {
    QuantMatmulTilingData + CommTilingData
};
```

**代码实现**（`tiling_data.h:38-42`）：
```cpp
struct QuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;
    QuantMatmulTilingData mmTilingData;
    QuantMatmulTilingData subMmTilingData;  // 新增：子问题 tiling
};
```

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 基础字段 | ✅ | ✅ | 一致 |
| subMmTilingData | 无 | 有 | ❌ 差异 12（Round 4 优化新增） |

---

## 十、通信窗口布局（§7.3）

**设计文档**：
> 数据窗口：`rank_size × (m / rank_size) × n × sizeof(CType)` 字节
> 累加缓冲区：`(m / rank_size) × n × sizeof(CType)` 字节（UB 侧）

**代码实现**：
- `main.cpp:256` — `sizeWorkspace = m * n * sizeof(uint16_t)`（= rank_size × chunkM × n）
- `reduce_sum.h` — UB 侧 accumF32 + tempCType + tempF32 三缓冲区

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 数据窗口大小 | ✅ | ✅ | 一致 |
| 累加缓冲区位置 | UB | UB | 一致 |
| 累加缓冲区结构 | 单一 accumBuf | 3 个（accumF32 + tempCType + tempF32） | ❌ 差异 13（float32 累加需要更多 UB） |

---

## 十一、Process 流程伪代码（§8.3）

**设计文档 §8.3**：
```cpp
if ASCEND_IS_AIC {
    RunMatmul();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
}
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    for (turn = 0..commTurn_) {
        allToAll_.Commit();
        allToAll_.Wait<...>();
        ReduceSumOneRank();    // 逐 rank 求和
    }
    DataCopy(outputGM_, accumBuf_, totalBytes);
    allToAll_.Finalize();
}
```

**代码实现**（`impl.h:222-238` + `impl.h:530-570`）：
```cpp
if ASCEND_IS_AIC {
    RunMatmul();  // 内含 per-tile SetFlag
}
if ASCEND_IS_AIV {
    RunAllToAllAndReduceSum();  // 内含 per-tile WaitFlag + Commit + Wait + reduceSum
}
```

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| AIC SetFlag 位置 | RunMatmul 外 | RunMatmul 内 | ❌ 差异 14（per-tile 需要在循环内） |
| AIV WaitFlag 位置 | RunAllToAll 前 | RunAllToAllAndReduceSum 内 | 同上 |
| 逐 rank 求和 | `ReduceSumOneRank()` 每 rank 一次 | `RunReduceSumTile(t)` 每 tile 一次 | ❌ 差异 6（重复） |
| 最终 DataCopy | 单独写回 | reduceSum 内直接写 cGm | ❌ 差异 15（合并了写回步骤） |

---

## 十二、差异汇总表

| # | 差异项 | 类型 | 严重度 | 根因 |
|---|--------|------|--------|------|
| 1 | **无 bias 参数** | 功能缺失 | 中 | 首版暂不实现，PLAN.md 已记录 |
| 2 | 仅 block 0 通信 | 性能瓶颈 | 中 | 设计未指定通信核分配 |
| 3 | PUT 而非 GET | 与用户要求不一致 | 中 | GET 4+rank 不稳定 |
| 4 | **PIPE_FIX 而非 PIPE_MTE3** | **设计文档有误** | 高 | AIC 侧无 PIPE_MTE3，代码正确 |
| 5 | **per-tile 流水已实现** | **正向差异** | 低 | 设计列为"未来方向"，代码已实现 |
| 6 | 逐 tile 求和 vs 逐 rank 求和 | 粒度差异 | 低 | 数学等价，流水粒度不同 |
| 7 | **float32 累加** | **正向差异** | 低 | 修复 K=8192 精度问题 |
| 8 | **新增 workspaceGM** | 与用户要求一致 | 低 | 用户补充信息要求 |
| 9 | Run() vs Process() | 命名差异 | 低 | 无功能影响 |
| 10 | 增加 TransA/TransB 模板参数 | 接口差异 | 低 | 支持转置配置 |
| 11 | AIC 写 workspaceGM | 与用户要求一致 | 低 | 用户补充信息要求 |
| 12 | **新增 subMmTilingData** | **正向差异** | 低 | Round 4 cube_util 优化 |
| 13 | 3 个 UB 缓冲区 | float32 精度需要 | 低 | 差异 7 的连带 |
| 14 | SetFlag 在 RunMatmul 内 | per-tile 需要 | 低 | 差异 5 的连带 |
| 15 | reduceSum 内直接写 cGm | 步骤合并 | 低 | 简化流程 |

---

## 十三、结论

**15 项差异分为 4 类**：

### A. 设计文档有误（1 项）
- 差异 4：`PIPE_MTE3` 应为 `PIPE_FIX` — **代码正确，设计文档需更新**

### B. 正向差异 — 代码超出设计要求（4 项）
- 差异 5：per-tile 流水已实现（设计列为"未来方向"）
- 差异 7：float32 累加（设计未提及）
- 差异 12：subMmTilingData（Round 4 优化）
- 这些是性能/精度改进，**设计文档应更新以反映已实现状态**

### C. 与用户补充要求一致（3 项）
- 差异 8：workspaceGM（用户要求）
- 差异 11：AIC 写 workspace（用户要求）
- 差异 1：无 bias（用户说"暂不实现"）

### D. 待优化项（3 项）
- 差异 3：PUT 而非 GET（GET 4+rank 不稳定，待后续解决）
- 差异 2：仅 block 0 通信（cube_util 低，待后续优化）
- 差异 6：逐 tile 求和（可优化为逐 rank）

**核心架构完全一致**：AIC LOCAL matmul → AIV AllToAll + reduceSum，无原子累加，复用 Apace + Blaze 框架。差异主要集中在性能优化迭代带来的实现细节上。

---

## 十四、建议后续动作

| 优先级 | 动作 | 关联差异 |
|--------|------|---------|
| 高 | 更新设计文档：PIPE_MTE3 → PIPE_FIX | 差异 4 |
| 高 | 更新设计文档：per-tile 流水从"未来方向"改为"已实现" | 差异 5 |
| 中 | 更新设计文档：补充 float32 累加策略说明 | 差异 7 |
| 中 | 更新设计文档：入口签名增加 workspaceGM、移除 biasGM | 差异 1/8 |
| 中 | 更新设计文档：TilingData 增加 subMmTilingData 字段 | 差异 12 |
| 低 | 后续优化：解决 GET 模式 4+rank 稳定性 | 差异 3 |
| 低 | 后续优化：多核并行通信提升 cube_util | 差异 2 |
| 低 | 后续实现：bias 支持 | 差异 1 |

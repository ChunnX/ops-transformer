# 设计文档 vs 代码实现 一致性审查报告

> **审查日期**: 2026-08-18 (更新至 BF16 直接累加 + pingpong + V核流水掩盖 + 自适应 headMSize)
> **设计文档**: `docs/DESIGN.md` (quantMatmulReduceScatter 设计文档)
> **代码版本**: BF16 直接累加 + pingpong 双 buffer + Te::MakeTensor + V核流水掩盖 + 自适应 headMSize
> **审查范围**: kernel/apace_quant_matmul_reduce_scatter_udma_impl.h, kernel/qmm_mx_kernel_rs_frag.h, kernel/apace_quant_matmul_reduce_scatter_tiling_data.h, kernel/reduce_sum.h, src/kernel_launcher.h, src/main.cpp

---

## 一、功能定义（§2.1）

**设计文档**：
```
输入:
  A: [m, k], fp8_e4m3fn, 每卡完整持有（K 不切分）
  ScaleA: [m, ceil(k/64)*2], fp8_e8m0, 每卡完整持有
  B: [k, n], fp8_e4m3fn (TransB=true), 所有卡共享
  ScaleB: [ceil(k/64)*2, n], fp8_e8m0, 所有卡共享
  workspace: [m, n, sizeof(bfloat16)], Host 侧申请传入

数学公式:
  C_partial[rank] = A_rank × B      // [m, n], bf16
  C_out[rank] = reduceSum(AlltoAll(C_partial, dim=m), op=SUM)
  // C_out shape: [m/rankSize, n], bf16
```

**代码实现**：
- `impl.h` RunMatmul — FragmentTensor 量化 MatMul，A 数据连续 `[m,k]`，C 输出到 workspaceGM
- `impl.h` — `CollectiveComm<AllToAll, PUT, CType, TeamBarrier>` 通信 CType(bfloat16)
- `reduce_sum.h` — 逐 rank float32 Add 累加
- `kernel_launcher.h` — 入参有 `workspaceGM`，无 `biasGM`（bias 暂未实现）

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 量化MatMul + AlltoAll + reduceSum | ✅ | ✅ | 一致 |
| workspace 参数 | 有 | ✅ `workspaceGM` | 一致 |
| bias 支持 | 标注"暂未实现" | 无 | 一致（设计已声明） |
| 无 L0C 原子累加 | ✅ | ✅ 无 SetAtomicAdd | 一致 |
| 每卡完整 K | ✅ | ✅ | 一致 |
| 输出 shape [m/rankSize, n] | ✅ | ✅ | 一致 |

---

## 二、核角色分工（§3.2）

**设计文档**：AIC 做 FragmentTensor 量化 MatMul，AIV 做 AlltoAll PUT + reduceSum。

**代码实现**：
- `impl.h` Run() — `if ASCEND_IS_AIC { RunMatmul() }` / `if ASCEND_IS_AIV { RunAllToAllAndReduceSum() }`
- `impl.h` Init() — `totalJobs=1`，仅 block 0 的 AIV 参与通信

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| AIC = FragmentTensor MatMul | ✅ | ✅ `fragKernelImpl_` | 一致 |
| AIV = AllToAll + reduceSum | ✅ | ✅ | 一致 |
| 多少核参与通信 | 未明确 | 仅 block 0 (totalJobs=1) | 设计未指定，代码实现合理 |

---

## 三、通信模式（§6.2）

**设计文档**：`CommMode::PUT`（GET 在 4+rank 不稳定，回退到 PUT）

**代码实现**：
- `impl.h` — `CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, CType, TeamBarrier>`

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 通信模式 | PUT | PUT | 一致 |
| 通信数据类型 | CType (bfloat16) | CType (bfloat16) | 一致 |
| GET 模式 | 标注"未来扩展" | 未实现 | 一致（设计已声明） |

---

## 四、同步机制（§6.4 + §8.4）

**设计文档**：
- AIC: `CrossCoreSetFlag<0x2, PIPE_FIX>(0)` (PIPE_FIX = AIC FixPipe)
- AIV: `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)` (PIPE_MTE2 = AIV MTE2)
- per-tile: T 次 Set ↔ T 次 Wait (计数式 flag)
- SyncAll<true>() isAIVOnly=true, AIC 为 no-op

**代码实现**：
- `impl.h` RunMatmul — `CrossCoreSetFlag<0x2, PIPE_FIX>(0)` per-tile 循环内
- `impl.h` RunAllToAllAndReduceSum — `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)` per-tile 循环内
- `impl.h` — `SyncAll<true>()` 在 WaitFlag 后、Commit 前

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| AIC SetFlag 管道 | PIPE_FIX | PIPE_FIX | 一致 |
| AIV WaitFlag 管道 | PIPE_MTE2 | PIPE_MTE2 | 一致 |
| per-tile 计数式 flag | ✅ T 次 Set ↔ T 次 Wait | ✅ | 一致 |
| SyncAll<true>() | ✅ AIV only | ✅ | 一致 |
| 不使用 AtomicAdd | ✅ | ✅ | 一致 |

---

## 五、FragmentTensor 架构（§3.2 + §9.2）

**设计文档**：
- 参考 allgather `QmmMxKernelAgUdma` 的 FragmentTensor 设计
- A 数据连续，各 rank 段打包为 FragmentTensor
- `QmmMxBlockMmadFragment` per-fragment L1 隔离
- 一次 matmul 调用覆盖所有 rank 段，无 R 循环

**代码实现**：
- `qmm_mx_kernel_rs_frag.h` — `QmmMxKernelRsFrag` 类，528 行
- 使用 `BlockMmadFragC = QmmMxBlockMmadFragment<0, false, ...>`
- 使用 `FragmentTensor<2, MAX_FRAGMENT_COUNT, MakeLayoutA, AType>`
- `BuildFragmentTensors` — 打包 rankSize 个 rank 段地址
- `ResolveTileCtx` — HEAD/MAIN/TAIL region 解析（当前只用 HEAD）
- `impl.h` RunMatmul — per-tile 调用 `fragKernelImpl_(params)`，无 R 循环

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| FragmentTensor 打包 rank 段 | ✅ | ✅ `BuildFragmentTensors` | 一致 |
| QmmMxBlockMmadFragment | ✅ | ✅ | 一致 |
| 消除 R 循环 | ✅ | ✅ 无 `for r` 循环 | 一致 |
| per-fragment L1 隔离 | ✅ | ✅ `FragmentSliceCopy` | 一致 |
| HEAD/MAIN/TAIL region | ✅ | ✅ 当前只用 HEAD（per-tile 调用） | 一致 |

---

## 六、流水线架构（§4.3 + §9.5）

**设计文档**：
- per-tile 流水: AIC tile N 计算 || AIV tile N-1 通信
- 统一路径: 无 if/else, T=1 自然退化

**代码实现**：
- `impl.h` RunMatmul — `for t = 0..commTurn` 单一循环，无 if/else
- `impl.h` RunAllToAllAndReduceSum — `for t = 0..commTurn` per-tile WaitFlag + Commit + reduceSum

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| per-tile 流水 | ✅ AIC tile N || AIV tile N-1 | ✅ | 一致 |
| if/else 归一化 | ✅ 无 if/else | ✅ 统一 for t 循环 | 一致 |
| T=1 自然退化 | ✅ | ✅ 循环执行 1 次 | 一致 |

---

## 七、reduceSum 精度策略（§6.3 + §9.4）

**设计文档**：BF16 直接累加（950 硬件级精度, 无 Cast）

**代码实现**：
- `reduce_sum.h` — `Te::Transform<Inst::MulScalar>` 清零 sum (float 视图), 逐 rank `Te::Transform<Inst::Add>` BF16 直接累加, `Te::Copy` 输出 BF16
- pingpong 双 buffer: `pingPongCount_ & 1` 交替使用 input A/B
- Te::MakeTensor + Te::Copy + Te::Transform (不用 LocalTensor/VECIN)

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| BF16 直接累加 | ✅ | ✅ | 一致 |
| pingpong 双 buffer | ✅ | ✅ | 一致 |
| Te::MakeTensor | ✅ | ✅ | 一致 |
| V核流水掩盖 | ✅ | ✅ | 一致 |
| 自适应 headMSize | ✅ | ✅ | 一致 |

---

## 八、Kernel 入口签名（§8.1）

**设计文档**：
```cpp
void ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma(
    __gm__ CommContext *hcommCtx,
    GM_ADDR aGM, GM_ADDR scaleAGM,
    GM_ADDR bGM, GM_ADDR scaleBGM,
    GM_ADDR workspaceGM, GM_ADDR cGM,
    ApaceQuantMatmulReduceScatterTilingData tilingData)
```

**代码实现**：
- `kernel_launcher.h` — 签名完全匹配
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`
- `impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM, &tilingData)`
- `impl.Run()` (不是 Process())

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| 入口签名 | ✅ workspaceGM | ✅ | 一致 |
| 方法名 | Run() | Run() | 一致 |
| MIX_AIC_1_1 | ✅ | ✅ | 一致 |
| 模板参数 | `<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true>` | ✅ | 一致 |

---

## 九、TilingData 结构体（§5.1）

**设计文档**：
```cpp
struct ApaceQuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;
    QuantMatmulTilingData mmTilingData;
    QuantMatmulTilingData subMmTilingData;
};
```

**代码实现**：
- `apace_quant_matmul_reduce_scatter_tiling_data.h` — 结构体完全匹配

| 项 | 设计 | 代码 | 差异 |
|----|------|------|------|
| commTilingData | ✅ | ✅ | 一致 |
| mmTilingData | ✅ | ✅ | 一致 |
| subMmTilingData | ✅ | ✅ | 一致 |

---

## 十、差异汇总

| # | 差异项 | 类型 | 严重度 | 说明 |
|---|--------|------|--------|------|
| 1 | bias 未实现 | 已知限制 | 低 | 设计已声明"暂未实现" |
| 2 | 仅 block 0 通信 | 设计未指定 | 低 | totalJobs=1, 设计可补充 |
| 3 | GET 模式未实现 | 已知限制 | 低 | 设计已声明"未来扩展" |
| 4 | reduceSum repeatTime 分批 | 设计未详细描述 | 低 | 代码已实现, 设计可补充 |
| 5 | HEAD/MAIN/TAIL 只用 HEAD | 架构简化 | 低 | per-tile 调用模式, 无需 MAIN/TAIL |

**结论**：设计文档与代码实现**高度一致**，无阻塞差异。5 项差异均为设计未详细描述或已知限制，不影响正确性和功能。

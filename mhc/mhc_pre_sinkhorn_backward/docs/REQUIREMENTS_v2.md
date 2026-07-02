# MhcPreSinkhornBackward 重构需求文档 v2

> 创建时间: 2026-07
> 迭代: v2（n≤8 约束 + TilingData 标准化 + 空Tensor支持 + c切分下沉Kernel）
> 关联: `7月迭代` 标记文件

---

## 一、背景与目标

### 1.1 上月状态
- 功能已实现（`mhc_pre_sinkhorn_backward_deterministic.h` 确定性路径）
- 非确定性路径（`mhc_pre_sinkhorn_backward_one_stage.h`）已有独立实现

### 1.2 本月目标
1. **代码重构**：文件改名、循环合并、逻辑清晰化
2. **性能提升**：搬运对齐优化、UB 依赖精简
3. **工程规范**：TilingData 标准化、空 Tensor 支持
4. **精度泛化**：泛化用例精度（后续单独验证，非本次重点）

### 1.3 改动范围
**仅涉及 4 个文件/类**：

| # | 文件 | 类/结构体 |
|---|------|----------|
| ① | `op_kernel/arch35/mhc_pre_sinkhorn_backward_deterministic.h` | `MhcPreSinkhornBackwardDeterministic` |
| ② | `op_kernel/arch35/mhc_pre_sinkhorn_backward_deterministic_simd_vf.h` | 辅助函数（SigmoidGrad, ComputeGradBias/Alpha 等） |
| ③ | `op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h` | `MhcPreSinkhornBackwardArch35DeterminiticTilingData` |
| ④ | `op_host/op_tiling/arch35/mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.{h,cpp}` | `MhcPreSinkhornBackwardArch35DeterminiticTiling` |

---

## 二、核心约束

### 2.1 n ≤ 8
- `n_` 最大为 8，意味着 `nn_ = n*n ≤ 64`，`k_ = n*n + 2n ≤ 80`
- UB 压力极小（n=8, bs=64 时全部分 buffer 约 200KB），不再需要 DoUbTiling 倒退循环
- n×n 矩阵（≤ 64 floats = 256B）在 32B 对齐下为 8 个 block，天然适合对齐访问

### 2.2 c 切分下沉 Kernel
- Tiling 层不再切 C 维度（`cLoopDataLen` 固定 = `c_`）
- C 切分逻辑完全放在 Kernel 内部处理
- DoUbTiling 中 `cLoopDataLen_ /= 2` 逻辑删除

### 2.3 计算逻辑不变
- 所有数学语义保持，仅优化数据搬运和代码组织

---

## 三、当前实现分析

### 3.1 k = n*n + 2n 维度结构

```
k_ 的内存布局（每 B*S 行）：
┌──────────────┬──────────────┬──────────────────────────┐
│  hPre (n_)   │  hPost (n_)  │  Sinkhorn 矩阵 (n_*n_)   │
│  offset=0    │  offset=n_   │  offset=2n_              │
└──────────────┴──────────────┴──────────────────────────┘
```

### 3.2 当前搬运模式 — 问题诊断

**hcBeforeNorm 搬运**：当前使用 stride-copy，每次只搬部分：

```cpp
// 当前：搬 bsLen*colLen，跳 (k_-colLen)，重复 bsLen 次
CopyIn(dst, src, bsLen, colLen, 0, (k_ - colLen) * sizeof(U));
```

| 分支 | colLen | 跳过 | 问题 |
|------|--------|------|------|
| hPre/hPost | n_ | n*n + n_ | 跨大量无效数据，带宽浪费 |
| Sinkhorn | n*n | 2n_ | 仅跳 64B，但仍非对齐 |

**改造方向**：一次性搬入完整 `kAlignFp32_` 元素，计算时通过 UB 内偏移访问子区域。

### 3.3 三阶段循环

当前 Process() 用 `firstDone`/`secondDone` flag 分为三段：
1. **阶段 1**：hPre 分支（n_ 维）— `ComputeFirst()`
2. **阶段 2**：hPost 分支（n_ 维）— `ComputeSecond()`
3. **阶段 3**：Sinkhorn 分支（n*n 维）— `ComputeThird()`

阶段 1 和 2 逻辑高度重复，仅有 bias offset 和 sigmoid mode 不同。

**改造方向**：合并阶段 1+2 → `ComputeSigmoidBranch()`，阶段 3 → `ComputeSinkhornBranch()`。

---

## 四、TilingData 标准化

### 4.1 参考标准

| 参考算子 | 命名模式 |
|---------|---------|
| `mhc_post` | `usedCoreNum`, `bsInner`/`bsOuter`/`bsTail`, `dInner`/`dOuter`/`dTail` |
| `rotary_position_embedding_grad` | `ubFactorS`/`ubFactorB`/`ubFactorN`, `blockFactorN`/`blockNumN`, `usedCoreNum` |

### 4.2 字段映射

| 当前字段（个性化） | 标准化字段 | 语义 | 模式 |
|---|---|---|---|
| `batchSize` | `b` | 批次大小 | 短名 |
| `seqLength` | `s` | 序列长度 | 短名 |
| `c` | `c` | 特征维度 | 保持 |
| `n` | `n` | 头维度 | 保持 |
| `usedAivNum` | `usedCoreNum` | 实际使用核数 | usedCoreNum |
| `aicNum` | `aicCoreNum` | AIC 核数 | 区分 |
| `cLoopDataLen` | `ubFactorC` | UB 一次处理 C 数 | ubFactor |
| `bsLoopDataLen` | `ubFactorBS` | UB 一次处理 BS 数 | ubFactor |
| `skIterCount` | `skIterCount` | Sinkhorn 迭代次数 | 业务保持 |
| `bsTaskCount` | `blockFactorBS` | 每核处理 BS 数 | blockFactor |
| `tailBsTaskCount` | `blockTailBS` | 尾核 BS 处理数 | blockTail |
| `cBlockLoops` | `blockNumC` | C 循环次数 | blockNum |
| `cBlockTailLoops` | `blockTailNumC` | 尾核 C 循环次数 | blockTailNum |
| `cBlockCount` | `blockFactorC` | 每块 C 处理数 | blockFactor |
| `cBlockTailCount` | `blockTailFactorC` | 尾块 C 处理数 | blockTailFactor |
| `cBlockTailTailCount` | `blockTailTailFactorC` | 尾核尾块 C 处理数 | blockTailTailFactor |
| `finalUsedAivNum` | **合并到 usedCoreNum** | Tiling reset 后直接更新 | 简化 |
| `eps` | `eps` | 数值稳定项 | 保持 |
| `mm1TilingData` | `mm1TilingData` | MatMul tiling | 保持 |
| `mm2TilingData` | `mm2TilingData` | MatMul tiling | 保持 |

### 4.3 目标 TilingData 结构

```cpp
struct MhcPreSinkhornBackwardArch35DeterminiticTilingData {
    // 基础形状
    int64_t b = 0;
    int64_t s = 0;
    int64_t c = 0;
    int64_t n = 0;

    // 核分配
    int64_t usedCoreNum = 0;       // AIV 实际使用核数（含 reset 后更新）
    int64_t aicCoreNum = 0;        // AIC 核数

    // UB 因子：单次 UB 处理多少元素
    int64_t ubFactorBS = 0;        // UB 一次处理 BS 数
    int64_t ubFactorC = 0;         // UB 一次处理 C 数（固定=c_，Kernel内处理）

    // Block 因子：每核处理多少元素
    int64_t blockFactorBS = 0;     // 每核处理 BS 数
    int64_t blockTailBS = 0;       // 尾核 BS 处理数
    int64_t blockFactorC = 0;      // 每块 C 处理数
    int64_t blockTailFactorC = 0;  // 尾块 C 处理数
    int64_t blockTailTailFactorC = 0; // 尾核尾块 C 处理数

    // Block 循环次数
    int64_t blockNumC = 0;         // C 循环次数
    int64_t blockTailNumC = 0;     // 尾核 C 循环次数

    // 业务参数
    int64_t skIterCount = 0;       // Sinkhorn 迭代次数
    float eps = 0;                 // 数值稳定项

    // MatMul Tiling
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};
```

---

## 五、空 Tensor 支持

### 5.1 参考实现

参考前向算子 `mhc_pre_sinkhorn` 的 `aclnn_mhc_pre_sinkhorn.cpp:308-314`：

```cpp
if (x->IsEmpty() || hin->IsEmpty() || hPost->IsEmpty() || hRes->IsEmpty()) {
    OP_LOGW("[aclnnMhcPreSinkhorn] Input tensor is empty, skip computation and return success.");
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

### 5.2 改动点

| 层级 | 文件 | 改动 |
|------|------|------|
| op_api | `mhc_pre_sinkhorn_backward/op_host/op_api/aclnn_*.cpp` | 添加 `IsEmpty()` 检查，空输入直接返回 workspaceSize=0 |
| Tiling | `arch35_tiling_deterministic.cpp` | 空张量时跳过 tiling 计算，返回 0 workspace |
| Kernel | `deterministic_base.h` | 不触发（op_api 层已拦截） |

---

## 六、修改清单

### P0 — 架构级改动

| # | 文件 | 改动描述 |
|---|------|---------|
| **1** | ① | **文件改名**：`deterministic.h` → `deterministic_base.h` |
| **2** | ① | **k_ 统一对齐搬入**：`CopyInHcBeforeNorm` stride-copy → `CopyIn(..., 1, bsLen * kAlignFp32_, 0, 0)`；计算时 VF 内偏移访问 |
| **3** | ① | **三阶段循环合并**：`ComputeFirst`+`ComputeSecond` → `ComputeSigmoidBranch`（hPre+hPost 合并）；`ComputeThird` → `ComputeSinkhornBranch` |

### P1 — 性能提升

| # | 文件 | 改动描述 |
|---|------|---------|
| **4** | ①② | **搬运对齐优化**：n≤8 时 n×n 矩阵行 ≤ 32B，`LoadUnAlign` → `DataCopy`/`LoadAlign` |
| **5** | ① | **UB Buffer 精简**：多个 tmp buffer 因 n≤8 尺寸极小，合并减少 InitBuffer 调用 |

### P2 — 工程规范

| # | 文件 | 改动描述 |
|---|------|---------|
| **6** | ③④① | **TilingData 标准化**：字段按 §4.3 重命名，`finalUsedAivNum` 合并到 `usedCoreNum` |
| **7** | ④ | **c 切分下沉**：DoUbTiling 中删除 `cLoopDataLen_ /= 2`，`ubFactorC` 固定 = c_ |
| **8** | ①④ | **UB Buffer 扩容**：`hcBeforeNormQue_` 从 `bsLen*nnAlignFp32_` → `bsLen*kAlignFp32_`（多 2n 元素） |
| **9** | key + include | **include 路径更新**：`deterministic.h` → `base.h` 后更新 `key_arch35.h` 等引用 |
| **10** | op_api | **空 Tensor 支持**：按 §5.2 添加 IsEmpty 检查 |

---

## 七、文件改动索引

```
mhc/mhc_pre_sinkhorn_backward/
├── op_kernel/arch35/
│   ├── mhc_pre_sinkhorn_backward_deterministic.h        → ① 改名 + 循环合并 + k_对齐搬运
│   ├── mhc_pre_sinkhorn_backward_deterministic_simd_vf.h → ② 对齐优化
│   ├── mhc_pre_sinkhorn_backward_data_arch35.h          → ③ TilingData 标准化
│   └── mhc_pre_sinkhorn_backward_key_arch35.h            → include 路径更新
├── op_host/
│   ├── op_tiling/arch35/
│   │   ├── mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.h  → ④
│   │   └── mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.cpp → ④ c切分下沉 + TilingData标准化
│   └── op_api/
│       └── aclnn_mhc_pre_sinkhorn_backward.cpp           → 空Tensor支持
└── docs/
    └── REQUIREMENTS_v2.md                                 ← 本文档
```

---

## 八、待确认项

| # | 问题 | 状态 |
|---|------|------|
| 1 | Pipe::Reset() 后重新 InitBuffer 是否支持？ | 待验证 |
| 2 | n=3,5,7 时 kAlignFp32_ padding 到 32B 后的精度影响？ | 需测试 |
| 3 | 空 Tensor 时 workspaceSize=0 是否影响上层调度？ | 参考前向已验证 |
| 4 | 文件改名后，APT 编译路径是否受影响？ | 待确认 CMakeLists |

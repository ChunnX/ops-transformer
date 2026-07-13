# MhcPreSinkhornBackward 确定性路径重构方案

> 迭代跟踪文档 | 创建: 2026-07 | 状态: 需求分析完成

---

## 一、背景与目标

### 1.1 上月已完成
- 确定性路径功能实现（deterministic 分支）
- 基本精度验证通过

### 1.2 本月目标
| 目标 | 优先级 |
|------|--------|
| 代码重构（文件改名、循环合并、逻辑解耦） | P0 |
| 性能提升（UB 依赖优化、搬运对齐） | P0 |
| 精度泛化验证 | 后续 |
| TilingData 字段规范化 | P2 |
| 空 Tensor 支持 | P2 |

### 1.3 约束
| # | 约束 | 影响 |
|---|------|------|
| 1 | **n ≤ 8（主路径 n=4,8）** | k_ = n²+2n ≤ 80 floats (320B)，UB 压力极小；其余 n 值 kAlignFp32_ padding 兜底 |
| 2 | **C 切分在 Kernel 内部处理** | Tiling 层 `cLoopDataLen_` 固定 = c_，不降 |
| 3 | **计算逻辑不改变** | 数学语义不变，仅重构搬运/循环结构 |
| 4 | **改动范围限定** | 仅 4 个文件/类（见 3.1） |

---

## 二、当前实现深度分析

### 2.1 核心维度 `k_ = n*n + 2n`

```
k_ 构成:
  [0, n_)    — hPre 分支 (sigmoid)
  [n_, 2n_)  — hPost 分支 (sigmoid × 2)
  [2n_, k_)  — Sinkhorn 分支 (n×n 矩阵)
```

`hcBeforeNorm` 形状 `[B*S, k_]`，是连接三个计算分支的核心张量。

### 2.2 当前 Process() 流程（三阶段式）

```
Process():
  for bsIdx:
    ┌─ ComputeFirst  — hPre 分支 (n_)
    │   CopyIn hcBeforeNorm (stride-copy, 仅搬 n_)
    │   ... compute sigmoid grad + gradAlphaAndBias ...
    │   ComputeGradZ(n_, offset=0)
    │
    ├─ ComputeSecond — hPost 分支 (n_)
    │   CopyIn hcBeforeNorm (stride-copy, 仅搬 n_)
    │   ... compute sigmoid grad + gradAlphaAndBias ...
    │   ComputeGradZ(n_, offset=n_)
    │
    └─ ComputeThird  — Sinkhorn 分支 (n*n)
        CopyIn gradHRes (n*n)
        for skIterCount_:
          SinkhornGradSimdVf → ExpGradSimdVf
          ComputeGradAlphaAndBias (n*n)
          ComputeGradZ(n*n, offset=2n_)
  
  SyncAll → CrossCore 同步 AIC matmul
  
  ProcessGradXFromRms:
    CopyIn gradNormOutWs_[k_] + hcBeforeNormGm_[k_]  ← 唯一完整搬入 k_ 的地方
    ComputeGradInvRms → 逐 C 块 ComputeGradXFromRms
  
  if blockIdx==0: AddAlphaAndBias (跨核归约)
  
  ComputeFinal:
    Add(xFromRms + xFromHin + xFromMatmul) → Cast → CopyOut gradX
```

### 2.3 四个核心问题

#### 问题 A：hcBeforeNorm 搬运碎片化

当前 `CopyInHcBeforeNorm(bsLen, colLen, gmOft)` 用 stride-copy：

```cpp
CopyIn(hcBeforeNormLocal, hcBeforeNormGm_[gmOft], bsLen, colLen,
       0, (k_ - colLen) * uTypeSize_);  // ← stride 跳过无效数据
```

| 场景 | 搬入量 | 跳过量 | 效率 |
|------|--------|--------|------|
| hPre/hPost | bsLen × n_ | bsLen × (nn_+n_) | 低 — 非对齐碎片读 |
| Sinkhorn | bsLen × nn_ | bsLen × 2n_ | 中 |

**方案**：改为 `CopyIn(..., 1, bsLen * kAlignFp32_, 0, 0)` 一次性搬入全部 k_，计算时 VF 内偏移访问。

#### 问题 B：三阶段循环冗余

`ComputeFirst`、`ComputeSecond` 逻辑几乎相同（仅 bias offset 和 sigmoid 模式不同），应合并为 `ComputeSigmoidBranch`。Sinkhorn 分支独立为 `ComputeSinkhornBranch`。

#### 问题 C：UB Buffer 生命周期过长

`gradHPreBuf_` (bs×n×4B)、`normOutForwardBuf_` (bs×nn×4B)、`zBuf_` (bs×nn×4B)、`gradZBuf_` (bs×nn×4B) 仅阶段 1/2 使用，但全程占用。结合 n≤8，总量虽然不大，但可进一步精简。

#### 问题 D：SIMD VF 非对齐访问

`SinkhornGradStage2` 和 `ExpGradSimdVf` 大量使用 `LoadUnAlign`/`StoreUnAlign`。n≤8 时 n×n 矩阵每行 ≤ 32B，改为全量对齐搬入 k_ 后天然对齐。

---

## 三、CopyIn/CopyOut 封装评估与重构

> 评估日期：2026-07-01 | 触发：变量名规范化 + 封装易用性提升

### 3.1 当前实现（`deterministic_simd_vf.h:49-69`）

```cpp
// CopyIn — 6 参数
template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T> &dst, const GlobalTensor<PARAM_T> &src,
                              int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride), 0};
    DataCopyPad(dst, src, copyParams, {false, 0, 0, 0});
}

// CopyOut — 参数签名同 CopyIn，但内部构造时 ubStride/gmStride 顺序不同！
template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T> &dst, const LocalTensor<PARAM_T> &src,
                               int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(ubStride), static_cast<uint32_t>(gmStride), 0};
    //                                 ↑↑ CopyIn 这里放 gmStride，CopyOut 放 ubStride——同签名不同语义！
    DataCopyPad(dst, src, copyParams);
}
```

### 3.2 六大致命问题

| # | 问题 | 严重度 | 详情 |
|---|------|--------|------|
| 1 | **CopyIn/CopyOut stride 语义暗换** | 🔴 致命 | 两者参数签名同为 `(ubStride, gmStride)`，但 CopyIn 内部传给 DataCopyExtParams 的是 `{gmStride, ubStride}`，CopyOut 内部是 `{ubStride, gmStride}`——**同签名不同语义，极难发现** |
| 2 | **参数名不匹配 Ascend C API** | 🟡 严重 | API 用 `blockCount`/`blockLen`/`srcStride`/`dstStride`，此处用 `repTime`/`dataLen`/`ubStride`/`gmStride`——每次调用需脑内翻译 |
| 3 | **连续搬运也要传 6 个参数** | 🟡 严重 | 25 个调用点中 **17 个是连续拷贝（stride 全 0）**，仍需传 `repTime, dataLen, 0, 0` |
| 4 | `PARAM_T` 命名不规范 | 🟢 建议 | 代码库约定用 `T`（参考 `mhc_pre` 的 `DataCopyGamma<T>`） |
| 5 | `dataLen` 是元素数非字节数 | 🟡 严重 | 内部 `*sizeof` 转字节，但参数名暗示是元素数，与 `blockLen`（字节）不一致 |
| 6 | 无 padding 支持 | 🟢 建议 | 固定 `{false, 0, 0, 0}`，无法做右对齐补齐——n≤8 时 n×n 对齐需要补齐到 32B |

### 3.3 调用点分类统计（deterministic.h 25 处）

```
Contiguous: CopyIn/Out(..., 1, len, 0, 0)      9 处
Contiguous: CopyIn/Out(..., n, len, 0, 0)      2 处
Contiguous 单元素: CopyOut(..., 1, 1, 0, 0)     6 处
                                                 ───
                                                 17 处连续 —— 可简化为 2~3 参数

Strided:    CopyIn/Out(..., n, len, 0, stride)  8 处 —— 保留 stride 但命名清晰化
```

### 3.4 参考实现：`mhc_pre` 风格（直接命名字段）

```cpp
DataCopyExtParams copyParams;
copyParams.blockCount = static_cast<uint16_t>(curMLen);    // 块数 → 一字不译
copyParams.blockLen   = uint32_t(curNdLen * sizeof(T));    // 每块字节
copyParams.srcStride  = uint32_t(0);                       // 源步长(字节)
copyParams.dstStride  = uint32_t(0);                       // 目标步长(字节)
DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};
DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
```

**核心差异**：`mhc_pre` 直接暴露 `DataCopyExtParams` 命名字段，不封装，不翻译。当前确定性路径多包了一层，引入了参数名翻译错误。

### 3.5 改进方案：拆为语义明确的 6 个函数

```cpp
// ============================================================
// 文件② deterministic_simd_vf.h — 新 Copy 函数
// ============================================================

// ── CopyIn 三件套 ──

// 单块连续搬入（覆盖 14 个调用点）
template <typename T>
__aicore__ inline void CopyInContiguous(
    const LocalTensor<T> &dstUb, const GlobalTensor<T> &srcGm, int64_t elemCount)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(elemCount * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    DataCopyPad(dstUb, srcGm, copyParams);
}

// 多块连续搬入（块间无跨步）
template <typename T>
__aicore__ inline void CopyInBlocks(
    const LocalTensor<T> &dstUb, const GlobalTensor<T> &srcGm,
    int64_t blockCount, int64_t elemPerBlock)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(blockCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerBlock * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    DataCopyPad(dstUb, srcGm, copyParams);
}

// 多块跨步搬入（GM 中块间非连续，srcStrideBytes 为源端字节跨步）
template <typename T>
__aicore__ inline void CopyInStrided(
    const LocalTensor<T> &dstUb, const GlobalTensor<T> &srcGm,
    int64_t blockCount, int64_t elemPerBlock, int64_t srcStrideBytes)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(blockCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerBlock * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(srcStrideBytes);
    copyParams.dstStride  = 0;
    DataCopyPad(dstUb, srcGm, copyParams);
}

// ── CopyOut 三件套 ──

template <typename T>
__aicore__ inline void CopyOutContiguous(
    const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t elemCount)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(elemCount * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    DataCopyPad(dstGm, srcUb, copyParams);
}

template <typename T>
__aicore__ inline void CopyOutBlocks(
    const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb,
    int64_t blockCount, int64_t elemPerBlock)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(blockCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerBlock * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    DataCopyPad(dstGm, srcUb, copyParams);
}

template <typename T>
__aicore__ inline void CopyOutStrided(
    const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb,
    int64_t blockCount, int64_t elemPerBlock, int64_t dstStrideBytes)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(blockCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerBlock * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = static_cast<uint32_t>(dstStrideBytes);
    DataCopyPad(dstGm, srcUb, copyParams);
}
```

### 3.6 调用点对照

| 当前调用（旧） | → 新调用 | 模式 |
|---|---|---|
| `CopyIn(dst, src, 1, len, 0, 0)` | `CopyInContiguous(dst, src, len)` | 连续（14 处） |
| `CopyIn(dst, src, usedAivNum_, len, 0, 0)` | `CopyInBlocks(dst, src, usedCoreNum_, len)` | 多块连续（2 处） |
| `CopyIn(dst, src, rows, elems, 0, stride)` | `CopyInStrided(dst, src, rows, elems, stride)` | 跨步读（6 处） |
| `CopyOut(dst, src, 1, len, 0, 0)` | `CopyOutContiguous(dst, src, len)` | 连续搬出（6 处） |
| `CopyOut(dst, src, rows, elems, 0, stride)` | `CopyOutStrided(dst, src, rows, elems, stride)` | 跨步写（2 处） |

### 3.7 辅助结构体改名

```cpp
// 旧：含义不明
struct Process1Info {
    int64_t bsLen = 0;
    int64_t cLen = 0;
    int64_t cLenXTAlign = 0;       // 混用 X_T 类型大小 + 对齐
    int64_t cLenUAlign = 0;
    int64_t cLenGRADHINTAlign = 0; // 全大写 + 类型前缀
};

// 新：语义明确
struct ComputeBlockInfo {
    int64_t bsCount = 0;           // 当前 BS 批次数
    int64_t cCount = 0;            // 当前 C 元素数
    int64_t cAlignedX = 0;         // C 对齐后元素数 (X_T 类型)
    int64_t cAlignedU = 0;         // C 对齐后元素数 (U/fp32 类型)
    int64_t cAlignedGradHin = 0;   // C 对齐后元素数 (GRADHIN_T 类型)
};
```

---

## 四、修改范围与 TilingData 规范化

### 4.1 涉及文件/类（仅 4 个）

| # | 文件 | 当前类名 | 改动 |
|---|------|---------|------|
| ① | `op_kernel/arch35/mhc_pre_sinkhorn_backward_deterministic.h` | `MhcPreSinkhornBackwardDeterministic` | 改名为 `_base.h`，循环合并 |
| ② | `op_kernel/arch35/mhc_pre_sinkhorn_backward_deterministic_simd_vf.h` | — (函数集) | CopyIn/CopyOut 封装重构 + SIMD VF 对齐优化 |
| ③ | `op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h` | `MhcPreSinkhornBackwardArch35DeterminiticTilingData` | 字段规范化 |
| ④ | `op_host/op_tiling/arch35/mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.{h,cpp}` | `MhcPreSinkhornBackwardArch35DeterminiticTiling` | 字段适配 + cLoopDataLen 固定 |

### 4.2 TilingData 字段规范化

参考 `mhc_post` 和 `rotary_position_embedding_grad` 的标准命名：

| 当前字段（个人化） | 标准化字段 | 语义 | 参考来源 |
|---|---|---|---|
| `batchSize` | `b` | Batch 大小 | mhc_post 风格 |
| `seqLength` | `s` | Seq 长度 | mhc_post 风格 |
| `c` | `c` | C 维度 | 保持 |
| `n` | `n` | N 维度 | 保持 |
| `usedAivNum` | `usedCoreNum` | 使用的 AIV 核数 | mhc_post `usedCoreNum` |
| `aicNum` | `aicCoreNum` | AIC 核数 | 区分 AIV/AIC |
| `cLoopDataLen` | `ubFactorC` | UB 单次处理 C 元素数 | rope `ubFactorN` |
| `bsLoopDataLen` | `ubFactorBS` | UB 单次处理 BS 元素数 | rope `ubFactorBS` |
| `skIterCount` | `skIterCount` | Sinkhorn 迭代次数 | 业务保持 |
| `bsTaskCount` | `blockFactorBS` | 普通核处理 BS 数 | rope `blockFactorBS` |
| `tailBsTaskCount` | `blockTailBS` | 尾核处理 BS 数 | rope `blockTailBS` |
| `cBlockLoops` | `blockNumC` | C 维度循环次数 | rope `blockNumN` |
| `cBlockTailLoops` | `blockTailNumC` | 尾核 C 循环次数 | rope |
| `cBlockCount` | `blockFactorC` | 普通核单次 C 处理数 | rope `blockFactorN` |
| `cBlockTailCount` | `blockTailFactorC` | 尾核单次 C 处理数 | rope |
| `cBlockTailTailCount` | `blockTailTailFactorC` | 尾核尾块 C 处理数 | rope |
| `finalUsedAivNum` | **合并到 `usedCoreNum`** | 最终使用核数 | 单一字段 |
| `eps` | `eps` | Epsilon | 保持 |
| `mm1TilingData` | `mm1TilingData` | MatMul1 Tiling | 保持 |
| `mm2TilingData` | `mm2TilingData` | MatMul2 Tiling | 保持 |

#### 标准化后完整结构体：

```cpp
struct MhcPreSinkhornBackwardArch35DeterminiticTilingData {
    // 形状参数
    int64_t b = 0;
    int64_t s = 0;
    int64_t c = 0;
    int64_t n = 0;
    
    // 核数
    int64_t usedCoreNum = 0;   // AIV 核数（含 reset 后修正）
    int64_t aicCoreNum = 0;    // AIC 核数
    
    // UB 因子
    int64_t ubFactorC = 0;     // UB 单次处理 C 元素数
    int64_t ubFactorBS = 0;    // UB 单次处理 BS 元素数
    
    // Block 因子（每核处理量）
    int64_t blockFactorBS = 0;       // 普通核 BS 处理数
    int64_t blockTailBS = 0;         // 尾核 BS 处理数
    int64_t blockFactorC = 0;        // 普通核单次 C 处理数
    int64_t blockTailFactorC = 0;    // 尾核单次 C 处理数
    int64_t blockTailTailFactorC = 0;// 尾核尾块 C 处理数
    
    // Block 循环次数
    int64_t blockNumC = 0;           // C 维度循环次数
    int64_t blockTailNumC = 0;       // 尾核 C 循环次数
    
    // 业务参数
    int64_t skIterCount = 0;         // Sinkhorn 迭代次数
    float eps = 0;                   // 数值稳定常数
    
    // MatMul Tiling
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};
```

### 4.3 空 Tensor 支持

参考 `aclnn_mhc_pre_sinkhorn.cpp` 的实现模式：

```cpp
// 在 op_api 层（aclnn_mhc_pre_sinkhorn_backward.cpp）：
if (gradHin->IsEmpty() || gradHPost->IsEmpty() || gradHRes->IsEmpty() || x->IsEmpty()) {
    OP_LOGW("Input tensor is empty, skip computation and return success.");
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

**改动位置**：`op_host/op_api/aclnn_mhc_pre_sinkhorn_backward.cpp`（如存在）或对应的 aclnn 入口文件。

---

## 五、修改清单

### P0 — 架构级改动

| # | 文件 | 改动 | 描述 |
|---|------|------|------|
| **P0-0** | ② | **CopyIn/CopyOut 封装重构** | 旧 2 函数（参数歧义）→ 新 6 函数（Contiguous/Blocks/Strided × In/Out），用命名字段构造 `DataCopyExtParams`；`Process1Info` → `ComputeBlockInfo` |
| **P0-0a** | ① | **调用点统一更新 + 变量名规范化** | 所有旧 `CopyIn`/`CopyOut` → 新三件套；Kernel 内变量按 §5.1 标准表改名 |
| **P0-1** | ① | 文件改名 | `mhc_pre_sinkhorn_backward_deterministic.h` → `mhc_pre_sinkhorn_backward_deterministic_base.h` |
| **P0-2** | ① | k_ 统一对齐搬入 | `CopyInHcBeforeNorm` stride-copy → `CopyInContiguous(..., bsLen * kAlignFp32_)`；`hcBeforeNormQue_` 扩容为 kAlignFp32_ 尺寸；计算时 VF 内通过 `[i * kAlignFp32_ + offset]` 偏移访问 |
| **P0-3** | ① | 三阶段循环合并 | `ComputeFirst` + `ComputeSecond` → `ComputeSigmoidBranch`（hPre/hPost 合并）；`ComputeThird` → `ComputeSinkhornBranch` |

### P1 — 性能提升

| # | 文件 | 改动 | 描述 |
|---|------|------|------|
| **P1-1** | ② | SIMD VF 对齐 | n≤8 时 n×n 矩阵每行 ≤ 32B：`LoadUnAlign` → `LoadAlign`/`DataCopy`；`StoreUnAlign` → `StoreAlign` |
| **P1-2** | ① | UB Buffer 精简 | 阶段 1/2 专属 buffer 在阶段 3 通过 pipe Reset 复用 |

### P2 — 规范配套

| # | 文件 | 改动 | 描述 |
|---|------|------|------|
| **P2-1** | ③ | TilingData 字段重命名 | 按 §4.2 标准映射表改名 |
| **P2-2** | ④ | Tiling 适配 | DoUbTiling: 删除 `cLoopDataLen_ /= 2`，固定 = c_；字段名同步更新 |
| **P2-3** | ①④ | `usedCoreNum` 合并 | `usedAivNum` + `finalUsedAivNum` → 单一 `usedCoreNum` |
| **P2-4** | ① + key_arch35 | include 路径 | `deterministic.h` → `deterministic_base.h` 后所有引用路径更新 |
| **P2-5** | op_api | 空 Tensor | 添加 `IsEmpty()` 检查，空输入直接返回成功 |

### 5.1 Kernel 内部变量命名规范化

与 TilingData 标准化同步，Kernel 类内变量统一匹配 TilingData 字段：

| 当前变量名 | 标准化后 | 说明 |
|---|---|---|
| `k_` | `kDim_` | K 维度 = n²+2n |
| `nn_` | `nSquared_` | n² |
| `bsCount_` | `totalBs_` | B×S 总数 |
| `curCoreProcessbsTask_` | `bsPerCore_` | 当前核处理的 BS 数 |
| `bsLoopDataLen_` | `ubFactorBS_` | 匹配 TilingData |
| `cLoopDataLen_` | `ubFactorC_` | 匹配 TilingData |
| `cLoopsPerBS_` | `ubNumCPerBS_` | C 循环次数（per BS） |
| `cCountPerBS_` | `ubFactorCPerBS_` | C 步长元素数（per BS） |
| `cTailPerBS_` | `ubTailCPerBS_` | C 尾块元素数（per BS） |
| `cBlockLoops_` | `blockNumC_` | 匹配 TilingData |
| `cBlockTailLoops_` | `blockTailNumC_` | 匹配 TilingData |
| `cBlockCount_` | `blockFactorC_` | 匹配 TilingData |
| `cBlockTailCount_` | `blockTailFactorC_` | 匹配 TilingData |
| `blockToTalNum_` | `totalCBlocks_` | C 块总数 |
| `finalUsedAivNum_` | → 合并到 `usedCoreNum_` | 与 TilingData 合并同步 |
| `nnAlignFp32_` | `nSquaredAlignFp32_` | 匹配 nSquared_ |
| `cLenAlignFp32_` | `ubFactorCAlignFp32_` | 匹配 ubFactorC_ |
| `bsAlignFp32_` | `ubFactorBSAlignFp32_` | 匹配 ubFactorBS_ |

---

## 六、风险与确认（已裁决）

| # | 风险 | 裁决 | 状态 |
|---|------|------|------|
| **R1** | k_ 全量搬入增加 UB | ✅ 忽略 — n=8 时每 BS 仅多 64B | 已确认 |
| **R2** | VF 内偏移访问策略 | ⚠️ 两级搬运：GM→UB 对齐全量搬 kAlignFp32_；UB→VF 跳步取 n_/n_/nn_ 三块，VF 内仍对齐；计算完对非对齐场景单独搬出 | 已确认（见下方 R2 详案） |
| **R3** | Sinkhorn 循环方向 | ✅ `for reversed(iters)` 语义不变 | 已确认 |
| **R4** | Pipe Reset 重新 InitBuffer | ✅ 支持，ComputeFinal 已有先例 | 已确认 |
| **R5** | TilingData 字段改名同步 | ✅ 编译器类型检查保证 | 自动通过 |
| **R6** | n 取值与对齐 | ✅ n 主路径 4/8，其余数 kAlignFp32_ padding 兜底；后续补充用例验证 | 已确认 |

### R2 详案：两级搬运策略

```
Level 1 — GM→UB（对齐全量，一次 CopyInContiguous）:
  CopyInContiguous(hcBeforeNormUb, hcBeforeNormGm_[offset], bsLen * kAlignFp32_);
  // UB 布局: [bs0: kAlignFp32_ | bs1: kAlignFp32_ | ...]

Level 2 — UB→VF（VF 内跳步取三块）:
  // 取 hPre (n_): 从 hcBeforeNormUb + i*kAlignFp32_ + 0    取 n_ 个
  // 取 hPost (n_): 从 hcBeforeNormUb + i*kAlignFp32_ + n_   取 n_ 个
  // 取 sinkhorn (nn_): 从 hcBeforeNormUb + i*kAlignFp32_ + 2n_ 取 nn_ 个
  // VF 内数据仍按 kAlignFp32_ 对齐边界

Level 3 — VF→GM（搬出）:
  // 对齐场景: CopyOutContiguous / CopyOutBlocks
  // 非对齐尾块: CopyOutStrided（仅尾块场景，数量极少）
```

**关键优势**：GM 带宽从碎片 stride 读 → 对齐连续读；VF 内跳步成本远低于 GM stride。

---

## 七、迭代记录

| 日期 | 更新内容 |
|------|---------|
| 2026-07-01 | 初版：深度分析 + 修改清单 + TilingData 规范化 + 空Tensor支持 + n≤8 约束分析 |
| 2026-07-01 | 新增：CopyIn/CopyOut 封装评估（6 问题诊断 + 6 函数改进方案 + Process1Info→ComputeBlockInfo） |
| 2026-07-01 | 上帝裁决：R1~R6 全部确认；R2 明确两级搬运策略（GM→UB 对齐全量 → VF 跳步取三块）；n 主路径 4/8 |
| — | 待补充 |

---

## 附录：UB Buffer 完整布局

### 当前布局（n=8, bsLen=64 参考值）

```
Queues (Double Buffer):
  xQue_              bsLen * n_ * cLenAlignFp16_ * 2B     ≈ ?
  hPreAndGradHPostQue_ bsLen * nAlignFp32_ * 4B            ≈ 2KB
  gradHinQue_        bsLen * cLenAlignFp16_ * 2B           ≈ ?
  hcBeforeNormQue_   bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  invRmsQue_         bsAlignFp32_ * 4B                     ≈ 256B
  gradHResQue_       bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  xFp32Que_          bsLen * n_ * cLenAlignFp32_ * 4B      ≈ ?
  gradXFromHinQue_   bsLen * n_ * cLenAlignFp32_ * 4B      ≈ ?
  gradAlphaAndBiasQue_ nnAlignFp32_*4B + 32B               ≈ 288B

Bufs (Single Buffer):
  gradHPreBuf_       bsLen * nAlignFp32_ * 4B              ≈ 2KB
  gradZBuf_          bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  biasBuf_           kAlignFp32_ * 4B                      ≈ 320B
  normOutForwardBuf_ bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  zBuf_              bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  gradCalcBuf1_      kAlignFp32_ * 4B                      ≈ 320B
  gradCalcBuf2_      kAlignFp32_ * 4B                      ≈ 320B
  tmpDivBuf_         bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  tmpDiv2Buf_        bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  tmpDiv3Buf_        bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  tmpDiv4Buf_        bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  tmpMulBuf_         bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  sumOutRowBuf_      skIter * bsLen * nAlignFp32_ * 4B     ≈ 2KB × skIter
  sumOutColBuf_      skIter * bsLen * nAlignFp32_ * 4B     ≈ 2KB × skIter
  normOutBuf_        skIter * bsLen * nnAlignFp32_ * 4B    ≈ 16KB × skIter
  zResMaxBuf_        bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  zResMaxIdxBuf_     bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  tmpBuf_            bsLen * nnAlignFp32_ * 4B             ≈ 16KB
  isMaxBuf_          bsLen * nnAlignFp32_ * 4B             ≈ 16KB
```

### 改造后布局（hcBeforeNormQue_ 扩容 + Buffer 复用）

```
hcBeforeNormQue_ → bsLen * kAlignFp32_ * 4B               ≈ 16KB + 512B

复用关系 (pipe Reset 后):
  gradHPreBuf_ (2KB)         → tmpDivBuf_
  normOutForwardBuf_ (16KB)  → tmpDiv3Buf_
  zBuf_ (16KB)               → tmpDiv4Buf_
  gradZBuf_ (16KB)           → tmpDiv2Buf_
```

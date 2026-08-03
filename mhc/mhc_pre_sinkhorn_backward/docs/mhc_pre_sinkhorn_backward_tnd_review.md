# MhcPreSinkhornBackward 算子支持 TND(3D) 格式输入 — 需求反串讲

## 1. 需求背景

### 1.1 现状

`MhcPreSinkhornBackward` 是 mHC（Manifold-Constrained Hyper-Connections）结构中 Sinkhorn 变换的反向算子。PR 之前，该算子仅支持 **4D BSND 格式** `(B, S, N, C)` 输入，其中 B=batchSize，S=seqLength。

上游框架（PyTorch / MindSpore）在组网时经常将 B 和 S 维度提前合并为 T（T = B × S），以 3D TND 格式 `(T, N, C)` 传递张量。算子不支持 3D 输入时，框架侧需额外做 `reshape(4D) → 调算子 → reshape(3D)` 的胶水操作，带来额外开销和内存拷贝。

### 1.2 目标

算子 tiling 层和 kernel 层同时支持 4D `(B,S,N,C)` 和 3D `(T,N,C)` 两种输入格式，**框架侧无需额外 reshape**。

### 1.3 设计原则

> **在 tiling 层完成维度归一化，kernel 层无感。**

tiling 层将 4D 的 `B×S` 或 3D 的 `T` 统一为一个 `bs` 字段传入 tiling data，kernel 侧仅读取 `bs` 做计算，不感知输入是 3D 还是 4D。

---

## 2. 算子概述

### 2.1 输入输出

| 类别 | 名称 | 类型 | 4D Shape | 3D Shape | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 输入 | gradHin | BF16/FP16 | (B,S,C) | (T,C) | h_in 的梯度 |
| 输入 | gradHPost | FP32 | (B,S,N) | (T,N) | h_post 的梯度 |
| 输入 | gradHRes | FP32 | (B,S,N,N) 或 (B,S,N²) | (T,N,N) 或 (T,N²) | h_res 的梯度 |
| 输入 | x | BF16/FP16 | (B,S,N,C) | (T,N,C) | 前向输入 |
| 输入 | phi | FP32 | (2N+N², N·C) | 同左 | 前向参数 |
| 输入 | alpha | FP32 | (3,) | 同左 | 缩放系数 |
| 输入 | bias | FP32 | (2N+N²,) | 同左 | 偏置 |
| 输入 | hPre | FP32 | (B,S,N) | (T,N) | 前向缓存 |
| 输入 | hcBeforeNorm | FP32 | (B,S,N²+2N) | (T,N²+2N) | 前向缓存 |
| 输入 | invRms | FP32 | (B,S,1) | (T,1) | RMSNorm 逆方差 |
| 输入 | sumOut | FP32 | (2·iter, B,S,N) | (2·iter, T,N) | Sinkhorn 行/列和 |
| 输入 | normOut | FP32 | (2·iter, B,S,N,N) | (2·iter, T,N,N) | Sinkhorn 归一化矩阵 |
| 输出 | gradX | BF16/FP16 | (B,S,N,C) | (T,N,C) | x 的梯度 |
| 输出 | gradPhi | FP32 | (2N+N², N·C) | 同左 | phi 的梯度 |
| 输出 | gradAlpha | FP32 | (3,) | 同左 | alpha 的梯度 |
| 输出 | gradBias | FP32 | (2N+N²,) | 同左 | bias 的梯度 |

**属性：** `hcEps`（FLOAT，默认 1e-6，数值稳定性参数）

### 2.2 规格约束

| 约束项 | 值 | 说明 |
| :--- | :--- | :--- |
| sk_iter_count | = 20 | Sinkhorn 迭代次数，固定 |
| N | > 0, ≤ 8 | 头数上限（本 PR 将原"N=4"扩展为"N>0且≤8"） |
| C | > 0, < 100000, % 128 == 0 | 通道对齐约束 |
| 维度一致性 | — | 所有带 B/S 维的输入须同为 4D 或同为 3D |

### 2.3 计算逻辑（简述）

1. **Sinkhorn 反向迭代**（20 次）：用 `sumOut`/`normOut` 做行/列归一化反向
2. **门控激活反向**：三段 Sigmoid 门控（h_pre / h_post / h_res），计算 alphaGrad、biasGrad
3. **RMSNorm Fusion 反向**：`hMixGrad = hMixTmpGrad × invRms`
4. **线性投影反向**：`xRsGrad = hMixGrad @ phi`，`phiGrad = G^T @ X`
5. **最终 gradX**：`xGrad = xGradVec3 + xGradVec1`（hin 路径 + rms 路径 + matmul 路径）

---

## 3. 改动范围

### 3.1 文件清单

| 文件 | 改动类型 | 要点 |
| :--- | :--- | :--- |
| `op_tiling/arch35/...tiling_base.h` | 接口变更 | `CheckShapeBase` 加 `bool is3D = false` 参数 |
| `op_tiling/arch35/...tiling_base.cpp` | **重构** | 新增 `VerifyShape` 通用校验函数，`CheckShapeBase` 参数化支持 3D/4D |
| `op_tiling/arch35/...tiling_deterministic.h` | 接口变更 | 新增 `bs_` 成员，删除 `batchSize_`/`seqLength_` |
| `op_tiling/arch35/...tiling_deterministic.cpp` | **重构** | `GetShapeAttrsInfo` 分支 3D/4D，`bs_` 合并，魔鬼数字消除 |
| `op_kernel/arch35/...data_arch35.h` | **结构变更** | tiling data：`batchSize`+`seqLength` → `bs` |
| `op_kernel/arch35/...deterministic.h` | **kernel 变更** | 删除 `batchSize_`/`seqLength_`，`bsCount_` 直接读 `tilingData->bs` |
| `tests/ut/op_host/test_..._tiling.cpp` | **测试** | 新增 3 个 3D UT 用例 |
| `tests/assets/golden.py` | **测试** | golden 计算 3D unsqueeze/reshape 支持 |
| `tests/assets/input.py` | **新增** | aclnn 测试输入生成器（3D/4D 双格式） |
| `README.md` / `docs/aclnnMhcPreSinkhornBackward.md` | 文档 | 补充 3D shape 说明、维度约束 |

### 3.2 改动范围限定

- **仅 arch35 deterministic 路径** 支持了 3D 输入
- arch35 非 deterministic 路径 **未改动**（仍仅支持 4D）
- arch22 路径 **未改动**

---

## 4. Tiling 层改动

### 4.1 维度解析 — GetShapeAttrsInfo

`GetShapeAttrsInfo` 根据 x 的维度数（3 或 4）分支解析，统一写入 `bs_` 字段：

```
          4D 输入                        3D 输入
 x: (B, S, N, C)                  x: (T, N, C)
     ↓                                 ↓
 bs_ = B × S                      bs_ = T
 n_ = dim[2]                      n_ = dim[1]
 c_ = dim[3]                      c_ = dim[2]
     ↓                                 ↓
 CheckShapeBase(B, S, N, C)       CheckShapeBase(T, 1, N, C, is3D=true)
```

关键点：
- 3D 时传 `seqLength=1`，使 `bs = T × 1 = T`，复用同一套校验逻辑
- `is3D=true` 告知 `CheckShapeBase` 按降维规则校验其他 tensor

### 4.2 形状校验 — CheckShapeBase 重构

**核心思路**：每个 tensor 的 shape 结构是 `[leading] + batchPrefix + suffixDims`

- `batchPrefix`：3D 时为 `{T}`（1维），4D 时为 `{B, S}`（2维）
- `leading`：sumOut/normOut 前面的 iter 维，由 `startDim` 跳过
- `suffixDims`：每个 tensor 独有的尾部维度

**VerifyShape 函数**：
```
expectedDimNum = startDim + batchDimCount + suffixDims.size()
逐维比对：batchPrefix 部分 + suffix 部分
```

**调用示例**：
```
gradHin:   VerifyShape("gradHin", shape, batchPrefix, batchDimCount, {c})
sumOut:    VerifyShape("sumOut", shape, batchPrefix, batchDimCount, {n}, startDim=1)
phi:       VerifyShape("phi", shape, nullptr, 0, {hcMix, n*c})  // 无 batch 维
```

**gradHRes 兼容两种尾维**：
```
(T, N²) 或 (B,S,N²)   → suffix = {n*n}
(T, N, N) 或 (B,S,N,N) → suffix = {n, n}
```

### 4.3 Tiling Data 结构变更

```
改动前:                              改动后:
┌───────────────────┐               ┌───────────────────┐
│ batchSize = 0     │  ← 删除        │ bs = 0            │  ← 新增（合并）
│ seqLength = 0     │  ← 删除        │ c = 0             │
│ c = 0             │               │ n = 0             │
│ n = 0             │               │ ...               │
│ ...               │               └───────────────────┘
└───────────────────┘
```

**字段映射**：

| 输入格式 | tiling 写入 | kernel 读取 |
| :--- | :--- | :--- |
| 4D (B,S,N,C) | `bs_ = B × S` → `tilingData->bs` | `bsCount_ = tilingData->bs` |
| 3D (T,N,C) | `bs_ = T` → `tilingData->bs` | `bsCount_ = tilingData->bs` |

### 4.4 魔鬼数字消除

| 原始 | 常量 | 含义 |
| :--- | :--- | :--- |
| `8` | `MAX_N` | N 最大值 |
| `20` | `EXPECTED_SK_ITER_COUNT` | Sinkhorn 迭代次数 |
| `100000` | `MAX_C_VALUE` | C 上限 |
| `128` | `C_ALIGNMENT` | C 对齐 |
| `n*n+2*n` (重复15处) | `hcMix` | hcMix 局部变量 |
| `3 *` | `GRAD_X_FROM_BUF_COUNT` / `GRAD_ALPHA_BUF_COUNT` | buffer 数量 |
| `5 *` / `4 *` | `N_SQUARE_GRAD_BUF_COUNT` / `N_SQUARE_FINAL_BUF_COUNT` | buffer 数量 |

---

## 5. Kernel 层改动

### 5.1 核心变更

```
改动前:                              改动后:
┌─────────────────────────┐         ┌─────────────────────────┐
│ batchSize_ = tiling->bS │ 删除     │                         │
│ seqLength_ = tiling->sL │ 删除     │  (无 batchSize_/seqLength_)│
│ bsCount_ = bS * sL      │         │ bsCount_ = tiling->bs   │
└─────────────────────────┘         └─────────────────────────┘
```

### 5.2 影响范围

所有原来用 `batchSize_ * seqLength_` 的地方统一替换为 `bsCount_`，共约 8 处：

| 场景 | 改动前 | 改动后 |
| :--- | :--- | :--- |
| workspace 偏移 | `batchSize_ * seqLength_ * n_ * c_` | `bsCount_ * n_ * c_` |
| matmul M 维 | `mm1M_ = batchSize_ * seqLength_` | `mm1M_ = bsCount_` |
| matmul K 维 | `mm2K_ = batchSize_ * seqLength_` | `mm2K_ = bsCount_` |
| AIC 任务切分 | `totalSize = batchSize_ * seqLength_` | `totalSize = bsCount_` |

### 5.3 设计要点

kernel 层完全不感知输入是 3D 还是 4D。因为 tiling 层已把维度归一化为 `bs` 传入，kernel 内部所有按 BS 维度的循环、偏移、matmul 维度统一用 `bsCount_`。

---

## 6. 测试覆盖

### 6.1 新增 3D UT 用例

| 测试名 | T | N | C | 期望结果 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| test_tiling_T256_N4_C256_3d | 256 | 4 | 256 | SUCCESS | 正常 3D 场景 |
| test_tiling_T64_N8_C128_3d | 64 | 8 | 128 | SUCCESS | 3D + N=8 边界 |
| test_tiling_T64_N9_C128_3d_fail | 64 | 9 | 128 | FAILED | 3D + N=9 越界校验 |

### 6.2 原有 4D UT 用例（保持不变，7 个）

| 测试名 | B | S | N | C | 期望结果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| test_tiling_B2_S128_N4_C256 | 2 | 128 | 4 | 256 | SUCCESS |
| test_tiling_B1_S35_N1_C512 | 1 | 35 | 1 | 512 | SUCCESS |
| test_tiling_B2_S64_N2_C256 | 2 | 64 | 2 | 256 | SUCCESS |
| test_tiling_B1_S64_N8_C128 | 1 | 64 | 8 | 128 | SUCCESS |
| test_tiling_B1_S64_N9_C128_fail | 1 | 64 | 9 | 128 | FAILED |
| test_tiling_B1_S512_N4_C128 | 1 | 512 | 4 | 128 | SUCCESS |
| test_tiling_B4_S64_N4_C512 | 4 | 64 | 4 | 512 | SUCCESS |

### 6.3 Golden 测试

- `golden.py` 新增 3D 分支：3D 输入时 `unsqueeze(0)` 还原为 4D 走原有计算逻辑，最后 `view(-1, n, d)` 还原回 3D
- `input.py` 新增：通过 torch 前向 + autograd 自动生成参考梯度，同时支持 3D/4D 格式输入

---

## 7. 风险与注意事项

### 7.1 Tiling Data 二进制兼容性

`batchSize` + `seqLength` 合并为 `bs` 是 **二进制不兼容** 的变更。`MhcPreSinkhornBackwardArch35DeterminiticTilingData` 结构体布局变化，需要：
- tiling host 和 kernel **同时升级**
- 旧版 binary 不兼容新版 tiling data

### 7.2 范围限定

- 3D 支持 **仅限 arch35 deterministic 路径**
- arch35 非 deterministic 路径和 arch22 路径未做适配，仍仅支持 4D
- 如果这些路径也需要 3D 支持，后续需同步改造

### 7.3 N 约束扩展

本 PR 同时将 N 的约束从 `N == 4` 扩展为 `N > 0 且 ≤ 8`（deterministic 路径）。非 deterministic 路径仍保持 `N == 4` 约束。

---

## 8. 问答预设

**Q: 为什么 3D 路径传 `seqLength=1`？**
A: 复用 `CheckShapeBase` 的逻辑。3D 时 T = B×S 已经合并，传 `seqLength=1` 使 `bs = T × 1 = T`，加上 `is3D=true` 标志让校验函数按降维规则检查其他 tensor，避免写两套校验代码。

**Q: 为什么不直接在 kernel 里判断维度数？**
A: tiling 层做归一化更高效。kernel 侧只做计算，不应承担维度解析职责。统一 `bs` 字段后 kernel 代码零改动量、零分支，性能无损失。

**Q: gradHRes 为什么有两种尾维？**
A: 框架侧有时用 `(T, N²)` 展平格式，有时用 `(T, N, N)` 保留格式。两种都是合法输入，校验时根据实际维度数自适应。

**Q: 非 deterministic 路径后续会支持 3D 吗？**
A: 本 PR 不含。如果后续需要，可复用 `CheckShapeBase(is3D=true)` 的设计，kernel 侧同样只需读取 `bs` 字段。

---

## 9. 空 Tensor 支持

### 9.1 背景

上游框架在组网时可能出现 B=0 或 S=0 的场景（空 batch / 空 seqLength），此时所有含 BS 维度的输入变为空 tensor。算子需正确处理，不能崩溃或报错。

### 9.2 空 tensor 触发条件

唯一合法的空 tensor 场景：**B=0 或 S=0（即 T=0 / bs=0）**。

此时各输入的空与非空情况：

| 输入 | 含 BS 维度 | bs=0 时是否 empty | 说明 |
| :--- | :--- | :--- | :--- |
| gradHin | 是 | 是 | (B,S,C) → (0,0,C) |
| gradHPost | 是 | 是 | (B,S,N) → (0,0,N) |
| gradHRes | 是 | 是 | (B,S,N,N) → (0,0,N,N) |
| x | 是 | 是 | (B,S,N,C) → (0,0,N,C) |
| phi | 否 | 否 | (2N+N², N·C) 固定，不依赖 B/S |
| alpha | 否 | 否 | (3,) 固定 |
| bias | 否 | 否 | (2N+N²,) 固定 |
| hPre | 是 | 是 | (B,S,N) |
| hcBeforeNorm | 是 | 是 | (B,S,N²+2N) |
| invRms | 是 | 是 | (B,S,1) |
| sumOut | 是 | 是 | (2·iter, B,S,N) |
| normOut | 是 | 是 | (2·iter, B,S,N,N) |

> **注意**：N、C、skIterCount 均不允许为 0，规格约束已要求 `N > 0`、`C > 0`、`sk_iter_count = 20`。空 tensor 的合法场景仅限 B=0 或 S=0。phi/alpha/bias 的 shape 依赖 N 和 C（不依赖 B/S），故 N>0 且 C>0 时它们不会为空，aclnn 文档标注为"不支持空Tensor"。

### 9.3 实现方案

**在 aclnn 层 `GetWorkspaceSize` 入口处早退**，与同家族的 `aclnnMhcPreSinkhorn`（前向）和 `aclnnMhcSinkhornBackward` 保持一致。

实现位置：`op_host/op_api/aclnn_mhc_pre_sinkhorn_backward.cpp` 的 `aclnnMhcPreSinkhornBackwardGetWorkspaceSize` 函数。

处理流程（与前向 `aclnnMhcPreSinkhorn` 顺序完全对齐）：
1. `CheckParams` — 完整参数校验（not null + dim + shape + dtype），确保参数合法
2. 逐个检查含 BS 维度的输入是否 `IsEmpty()` — 任意一个为空即触发早退
3. 触发时：`*workspaceSize = 0`，释放 executor，返回 `ACLNN_SUCCESS`
4. 不调 `Contiguous`、不调 tiling、不启动 kernel

```cpp
auto ret = CheckParams(params);
CHECK_RET(ret == ACLNN_SUCCESS, ret);

// Check if input tensors are empty
if (gradHin->IsEmpty() || gradHPost->IsEmpty() || gradHRes->IsEmpty() || x->IsEmpty() ||
    hPre->IsEmpty() || hcBeforeNorm->IsEmpty() || invRms->IsEmpty() || sumOut->IsEmpty() ||
    normOut->IsEmpty()) {
    OP_LOGW("[aclnnMhcPreSinkhornBackward] Input tensor is empty, skip computation and return success.");
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 正常流程：Contiguous → l0op::MhcPreSinkhornBackward → CopyOutput
ret = MhcSinkhornGradCovertDataContiguous(params, uniqueExecutor.get());
...
```

### 9.4 为什么只检查含 BS 维度的输入

与前向 `aclnnMhcPreSinkhorn` 完全一致，只检查含 BS 维度的 9 个输入：

| 检查的输入（9 个） | 不检查的输入（3 个） |
| :--- | :--- |
| gradHin、gradHPost、gradHRes、x、hPre、hcBeforeNorm、invRms、sumOut、normOut | phi、alpha、bias |

- 含 BS 维度的输入：B=0 或 S=0 时变空，是空 tensor 的直接触发者，**必须检查**
- phi/alpha/bias：shape 固定（依赖 N 和 C，不依赖 B/S），正常场景不会空。若异常为空，`CheckParams` 的 shape 校验会先拦截

### 9.5 为什么不需要改 tiling 和 kernel

- aclnn 层拦截后，`workspaceSize=0`，tiling 和 kernel **根本不会执行**
- 这是最小改动、最低风险的方案
- 与同家族算子 `aclnnMhcPreSinkhorn`、`aclnnMhcSinkhornBackward` 实现完全一致

### 9.6 输出处理

bs=0 时输出情况：

| 输出 | 含 BS 维度 | 空 tensor？ | 处理 |
| :--- | :--- | :--- | :--- |
| gradX | 是 | 是 | 框架侧保证不读 |
| gradPhi | 否 | 否 | 理论上应为全 0（reducesum 无样本贡献），aclnn 早退后不写入，框架侧需保证已初始化为 0 |
| gradAlpha | 否 | 否 | 同上 |
| gradBias | 否 | 否 | 同上 |

> 与前向算子 `aclnnMhcPreSinkhorn` 行为一致：空输入时直接返回，不为非空输出写零值。框架侧负责初始化。

### 9.7 测试建议

参照 `mhc_sinkhorn_backward` 的空 tensor UT（`test_aclnn_mhc_sinkhorn_backward.cpp` 的 `aclnn_mhc_sinkhorn_backward_empty_tensor` 用例），新增：
- `test_aclnn_mhc_pre_sinkhorn_backward_empty_tensor`：B=0 或 S=0，断言 `aclRet == ACL_SUCCESS` 且 `workspaceSize == 0`

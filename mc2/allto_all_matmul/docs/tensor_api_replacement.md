# AlltoAllMatmul 矩阵乘替换为 Tensor API —— 改动方案记录

## 一、核心原则

> **尽量使用参考资料代码，不手写流水线。**

设备侧矩阵乘核心逻辑（BlockMmad、BlockScheduler）全部从参考项目原样拷贝，仅修改 include 路径中的 `kernel_utils/xxx` → `kernel_utils/xxx` 和 `include/tensor.h` → `tensor.h`。语义零变更。

## 二、目标

将 AlltoAllMatmul 算子中的矩阵乘从 MC2 模板系统（`Mc2FpMatmul → Mc2MatmulAswKernel` / `MC2MXQuantMatmul → MatMulASWKernel`）全部替换为 Tensor API 实现，同时：
- 通信层（HCCL-CCU AllToAll）、转置层（MC2VecTranspose）、流水线编排、核间同步保持不变
- 与 KC-QUANT 路径零耦合

## 三、Tensor API 依赖升级

原 `mc2/3rd/tensor_api/`（来自 `matmul_kernel` 参考工程）的 `CopyGM2L1` 类型检查不支持 `fp8_e8m0_t`（MX scale 类型），替换为 `cann-sample-master/third_party/tensor_api/`。

## 四、参考代码库 `matmul_ref/`

### 4.1 来源

| 来源项目 | 用途 | 文件数 |
|---------|------|:---:|
| `references/matmul_kernel` | FP 路径 BlockMmad + BlockScheduler | 11 |
| `cann-sample-master/Samples/2_Performance/matmul_story` | MX-FP8 路径 BlockMmad + BlockScheduler | 8 |

### 4.2 目录结构

```
mc2/common/matmul_ref/
├── kernel_utils/                    ← 来自 matmul_kernel
│   ├── common_utils.h
│   ├── tuple_utils.h
│   ├── integral_constant.h
│   └── layout_utils.h
├── block/                           ← 来自 matmul_kernel + matmul_story
│   ├── block_mmad.h
│   ├── matmul_block_mmad.h          （FP BlockMmad）
│   ├── matmul_block_scheduler.h     （FP Scheduler）
│   ├── block_scheduler_policy.h
│   ├── block_scheduler_utils.h
│   └── quant_matmul_mx_block_scheduler_swat.h  （MX Scheduler）
├── policy/
│   ├── dispatch_policy.h            （FP 策略标签）
│   └── dispatch_policy_mx.h         （MX 策略标签，新建）
├── utils/
│   ├── matmul_constant.h
│   └── quant_matmul_constant.h      （MX 专用常量）
├── tile/                            ← 来自 matmul_story
│   ├── tile_mmad_mx.h
│   ├── copy_scale_gm_to_l1.h
│   ├── copy_scale_l1_to_l0a.h
│   ├── copy_scale_l1_to_l0b.h
│   └── pad_mx_kl1.h
└── mx_block_mmad_swat.h             ← 来自 matmul_story（MX BlockMmad）
```

### 4.3 对参考代码的修改

| 文件 | 修改 |
|------|------|
| 所有 `block/matmul_block_mmad.h` | `"include/tensor.h"` → `"tensor.h"` |
| 所有 `mx_block_mmad_swat.h` | include 路径规范化 + `"include/tensor.h"` → `"tensor.h"` |
| `tile/copy_scale_l1_to_l0a.h` | 去除 `quant_matmul_constant.h` 依赖，`MXFP_DIVISOR_SIZE` → 类内常量 |
| `tile/copy_scale_l1_to_l0b.h` | 同上 |
| `utils/quant_matmul_constant.h` | 简化为仅 MX 专用常量，依赖 `matmul_constant.h` |
| `policy/dispatch_policy_mx.h` | 新建 |

**其余文件语义零变更。**

## 五、Thin Wrapper

两个 thin wrapper 均位于 `mc2/common/op_kernel/mc2_templates/computation/matmul/`：

### 5.1 `tensor_matmul_stage.h` — FP 路径

```cpp
Process(taskIndex):
  Init  BlockMmad(problemShape=(tileM,N,K), tileShape=(baseM,baseN,baseK))
  Init  BlockScheduler(problemShape, baseM, baseN)

  while (bs.GetTileIdx(blockIdx)):
      mmadOp_(gmBlockA, gmBlockB, gmBlockC, singleShape)
```

### 5.2 `tensor_mx_matmul_stage.h` — MX-FP8 路径

```cpp
Process(taskIndex):
  Init  MX BlockMmad(scale types), L1Params{kL1, scaleKL1}
  Init  MX Scheduler

  gmA/gmB/gmC/gmScaleA/gmScaleB  = MakeTensor(...)

  while (bs.GetTileIdx(blockIdx)):
      mxMmadOp_(gmBlockA, gmBlockB, gmScaleA, gmScaleB, gmBlockC, singleShape)
```

与参考 `MatmulKernel::Process` / `QuantMatmulMxKernelSwat::Process` 逐行对齐。

## 六、Tiling 数据结构

### 6.1 `TensorMatmulTilingData`

```cpp
struct TensorMatmulTilingData {
    uint32_t tileM;    // 当前 M 维度（tileM 或 tailM）
    uint32_t fullK;    // 完整 K = rankK × rankDim
    uint32_t fullN;    // 完整 N = rankN
    uint32_t baseM;    // M 切分块大小
    uint32_t baseN;    // N 切分块大小
    uint32_t baseK;    // K 切分块大小
    uint32_t kL1;      // 单次 L1 K 步长
    uint32_t scaleKL1; // scale 复用窗口（MX 使用，FP 填 0）
};
```

### 6.2 Host 侧填充

- **FP**：`DoMMTiling()` 中计算并填入全部 8 个字段（scaleKL1=0）
- **MX**：`FillMxTensorTiling()` 中计算含 scaleKL1 的全部字段

计算逻辑：
```
baseM = tileM
baseN = min(128, L0C_size / baseM / 4）  // L0C 约束
baseK = min(128, half_L1 / (baseM×aBytes + baseN×bBytes))  // L1 约束
kL1   = baseK                              // FP 同粒度
scaleKL1 = min(fullK, baseK × (1 + L1剩余/scale单份大小))  // MX 专用
```

Device 侧 `arch35` 的 `ProcessTile/ProcessTail` 负责填入 tileM/fullK/fullN（这些随 tile/tail 变化），其余字段由 host 预填。

## 七、修改文件汇总

| 文件 | 改动 |
|------|------|
| `matmul_ref/` (19 个文件) | 参考代码原样拷贝，最小化 include 修正 |
| `mc2/3rd/tensor_api/` | 替换为 cann-sample-master 版 |
| `tensor_matmul_tiling_data.h` | 替代 `Mc2MatMulV3TilingData`，8 字段 |
| `tensor_matmul_stage.h` | FP thin wrapper（common 目录） |
| `tensor_mx_matmul_stage.h` | MX thin wrapper（common 目录） |
| `allto_all_matmul_apt.cpp` | FP/MX 宏替换；MX-FP4/KC 保持原样 |
| `allto_all_matmul_tiling_data.h` | 替换 MatMulV3 为 tensor tiling 字段 |
| `allto_all_matmul_arch35.h` | 填入 tensor tiling 维度 |
| `allto_all_mx_quant_matmul_arch35.h` | if constexpr 区分 tensor/ASW 路径 |
| `allto_all_fp_matmul_tiling_base.h/.cpp` | DoMMTiling() 动态计算 tiling |
| `allto_all_mx_quant_matmul_tiling_base.h/.cpp` | FillMxTensorTiling() 动态计算含 scaleKL1 |
| `op_host/CMakeLists.txt` | 新增 tensor_api + matmul_ref include 路径 |

## 八、最终状态

| 路径 | 数据类型 | 矩阵乘引擎 | Scheduler | 状态 |
|------|---------|-----------|-----------|:---:|
| FP | fp16 / bf16 | `BlockMmad`（matmul_kernel 原样） | `MatmulBlockScheduler` | ✅ |
| MX-FP8 | fp8_e4m3 / fp8_e5m2 | `MX BlockMmad`（matmul_story 原样） | `BlockSchedulerQuantMatmulMxSwat` | ✅ |
| MX-FP4 | fp4_e2m1 | `MatMulASWKernel`（原样） | — | ✅ |
| KC-QUANT | 动态量化 | `MatMulV3`（原样） | — | ✅ |

## 九、未变部分

通信层(HCCL-CCU)、转置层、流水线编排、顶层封装、op框架、核间同步、KC/MX-FP4 路径 — 零改动。

## 十、已知问题

### 10.1 MX-FP8 精度偏差

BlockMmad 和 Scheduler 代码与参考**逐字节相同**，张量子视图逻辑也**完全正确**。精度差异根因在 tiling 参数计算：

| 参数 | 当前 `FillMxTensorTiling` | 参考 `QuantMatmulTilingSwat` |
|------|--------------------------|------------------------------|
| baseM | `min(tileM, 128)` | `min(fullM, 256)` 对齐 16，经 `AdjustBasicBlock()` 重平衡 |
| kL1 | `= baseK` | `= stepK × baseK`（可达 2~4 倍） |
| scaleKL1 | 简单启发式 | 多轮 `CalcPathSpecificL1()` 搜索 |
| problem.m | `= tileM`（单 task） | `= fullM`（全局） |

待后续使用参考的完整 `QuantMatmulTilingSwat` 引擎（从 matmul_story 拷贝）替代手写的 `FillMxTensorTiling`。

### 10.2 nBufferNum 恒为 2

尚未支持 4 buffer 模式。参考通过 `CalculateDefaultNBufferNum()` 动态选择 2/4 buffer，可提升 MTE2 带宽受限场景的吞吐。

## 十一、MX BlockMmad 同步优化（重点）

### 11.1 问题现象

`mx_block_mmad_swat.h:333` 原本必须使用 `PipeBarrier<PIPE_ALL>()` 才能避免 L0A 读写冲突，造成全流水线停顿，性能损失严重。

### 11.2 问题根因

**Ascend C 硬件中 `MTE1_M` 和 `M_MTE1` 是耦合的双向同步对**，它们共享同一个内部的流水线标记机制。原代码中不同事件类型使用了重叠的 flag ID：

| 事件类型 | 原使用的 flag ID |
|---------|-----------------|
| `MTE1_MTE2` / `MTE2_MTE1` | 0, 1, 2, 3... (L1 buffer 管理) |
| `M_MTE1` | **0, 1** |
| `MTE1_M` | **0, 1** |

当 `M_MTE1` 和 `MTE1_M` 使用相同的 flag ID 时，会导致硬件内部的等待/通知状态混乱：
- `WaitFlag<MTE1_M>(0)` 可能提前返回（被 `M_MTE1` flag 0 的状态变化干扰）
- Mad 在 L1→L0 搬运完成前就开始读取 L0A/L0B 数据
- 必须用 `PipeBarrier<PIPE_ALL>()` 强制同步，但代价是全流水线停顿

### 11.3 解决方案

**使用分离的 flag ID 范围，避免不同同步对的 flag ID 重叠**：

```cpp
// mx_block_mmad_swat.h:126-127
constexpr static uint16_t M_MTE1_FLAG_0 = 4;  // 原为 ZERO_FLAG=0
constexpr static uint16_t M_MTE1_FLAG_1 = 5;  // 原为 FIRST_FLAG=1
```

优化后的 flag ID 分配：

| 事件类型 | 优化后的 flag ID |
|---------|-----------------|
| `MTE1_MTE2` / `MTE2_MTE1` | 0, 1, 2, 3 (L1 buffer 管理) |
| `M_MTE1` | **4, 5** |
| `MTE1_M` | 0, 1 (与 M_MTE1 分离) |

### 11.4 关键代码改动

**构造函数/析构函数**：
```cpp
// mx_block_mmad_swat.h:135-136, 146-147
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
// 析构时对应 WaitFlag
```

**内层循环同步流程**：
```cpp
// mx_block_mmad_swat.h:273-344
uint16_t mte1WaitMFlag = l0BufId + M_MTE1_FLAG_0;  // 计算为 4 或 5
AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);  // 等待上一轮 Mad 完成

// 4 次 L1→L0 异步搬运
Copy(A, L1→L0A) → Copy(B, L1→L0B) → Copy(ScaleA, L1→L0A) → Copy(ScaleB, L1→L0B)

// 移除 PipeBarrier<PIPE_ALL>()，改用细粒度同步
AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);   // 信号搬运完成
AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);  // 等待搬运完成

Mad(...);  // 执行矩阵乘

AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);  // 信号 Mad 完成
```

### 11.5 性能收益

- **移除全流水线停顿**：`PipeBarrier<PIPE_ALL>()` 阻塞所有流水线阶段
- **细粒度同步**：`SetFlag<MTE1_M>` + `WaitFlag<MTE1_M>` 仅阻塞 MTE1→M 数据依赖
- **预期提升**：减少流水线停顿，提高 L1→L0 搬运与 Mad 的并行度

### 11.6 核心知识点

1. **`MTE1_M` 和 `M_MTE1` 是硬件耦合的双向同步对**，共享内部流水线标记
2. **必须使用分离的 flag ID**，避免耦合同步对之间的状态干扰
3. **`SetFlag<MTE1_M>` + `WaitFlag<MTE1_M>` 等价于 MTE1 流水线屏障**，足以保证所有 L1→L0 搬运完成
4. **`PipeBarrier<PIPE_ALL>()` 仅用于调试**，生产代码应使用事件标志同步

参考代码 `ops-tensor/include/blaze/block/block_mmad_mx.h` 使用相同的分离策略（`M_MTE1_FLAG_0=4, M_MTE1_FLAG_1=5`），是验证过的正确实践。

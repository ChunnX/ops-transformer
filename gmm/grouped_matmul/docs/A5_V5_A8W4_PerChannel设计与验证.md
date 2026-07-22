# A5 V5 A8W4 PerChannel Tiling、Kernel 与验证说明

## 1. 文档范围

本文描述 Ascend A5 上 GroupedMatmul V5 A8W4 非对称 per-channel 场景的当前实现，包括：

- Host 输入约束与 tiling key；
- `baseM/baseN/baseK`、K 方向 L1 window 和双缓冲的选择依据；
- 完整 W4→W8、前置 RowSum、AIC/AIV tile 流水的计算逻辑；
- L1、L0C、UB 和 GM workspace 的空间计算；
- 已完成的 UT、构建和产物验证；
- 当前没有经过 A5 实卡验证的部分。

当前真实数值 Kernel 只覆盖非对称 per-channel。对称 per-group 已完成 Host tiling 和 key 接入，但数值 Kernel 仍为占位实现，不属于本文的已打通范围。

## 2. 场景定义

### 2.1 输入与输出

当前实现针对：

```text
x                INT8      [M, K]
weight           INT4      [E, K, N]
weight packed    INT32     物理上每个 INT32 保存 8 个 INT4
bias             FLOAT32   [E, N]，V5 ABI 必选
scale            UINT64    [E, 1, N]
offset           FLOAT32   [E, 1, N]，非空表示非对称 per-channel
perTokenScale    FLOAT32   [M]
groupList        INT64
out              FLOAT16   [M, N]
```

其中 packed INT32 只是 INT4 的物理载体。Host/API 会保留逻辑 INT4 shape，并通过 storage shape 表达 8:1 packing；Kernel 始终把 weight 当作 signed INT4 nibble 流处理。

支持的 weight 布局包括：

- ND，逻辑 shape `[E,K,N]`；
- 普通 NZ，逻辑 shape `[E,K,N]`，INT4 输入 `C0=64`；
- 特殊 ENK NZ，逻辑 shape `[E,N,K]`，预处理后统一输出标准 `[E,K,N]` INT8 NZ，`C0=32`。

### 2.2 数学计算

对 expert `e` 中第 `m` 行、输出列 `n`，当前实现等价于：

```text
rowSum[m] = sum(x[m, k], k=0...K-1)

cube[m, n] = FixpipeScale(
    sum(x[m, k] * SignExtendInt4(weight[e, k, n]), k=0...K-1),
    scale[e, 0, n])

out[m, n] = (cube[m, n] + rowSum[m] * offset[e, 0, n])
              * perTokenScale[m]
```

W4 在进入 Cube 前直接做 signed W4→signed W8，因此 V5 的辅助 bias 不送入 Cube。bias 仍按 V5 输入合同要求存在，用于保持 API/ABI 一致性。

## 3. Tiling key 与路由

四个 key 保持不变：

| weight format | quant mode | key |
|---|---|---:|
| ND | per-channel | `0x20600200` |
| NZ | per-channel | `0x20600201` |
| ND | per-group | `0x20600300` |
| NZ | per-group | `0x20600301` |

APT 使用完整 tuple 区分 V5 A8W4，不能只依赖 `MULTI_SCALE_DEQUANT`。非对称 per-channel 的 FP16 签名进入真实 Tensor API facade；per-group 暂时只保证 key 和二进制分支可生成，不进入 per-channel Kernel。

生产 Host 路由已经能够把 V5 A8W4 送入 `GroupedA8W4BasicApiTiling`，并输出专用的 `GMMA8W4BasicApiTilingData`。

## 4. Base 块的计算

### 4.1 配置值

非对称 per-channel 的最大基础块配置为：

```text
baseM = 256
baseN = 256
baseK = 128
kAL1  = 256
kBL1  = 256
stepKa = kAL1 / baseK = 2
stepKb = kBL1 / baseK = 2
L1 A/B buffer num = 2
dbL0C = 1
```

Host 实际写入的 M/N tile 会考虑小 shape：

```text
baseM = AlignUp(min(M, 256), 16)
baseN = AlignUp(min(N, 256), 16)
baseK = 128
```

因此 `256×256×128` 是最大基础块，不表示小 M/N 也必须计算完整 256×256。M/N tail 由 scheduler 生成实际 `curM/curN`。

对单个 expert，scheduler 还会对 M tile 做一次均衡：

```text
mBlockCount = ceil(groupM / baseM)
balancedM   = ceil(groupM / mBlockCount)
curBaseM    = AlignUp(balancedM, 16)
```

这样可以避免最后一个 M tile 过小。N tail、跨 expert 轮转和尾块补核由 `BlockSchedulerGmmSwatWithTailSplit` 处理。

### 4.2 为什么 baseM/baseN 取 256

`baseM=256` 和 `baseN=256` 的目标是增加每次完整 K 扫描对应的输出面积，从而提高：

- weight 和 activation L1 window 的复用；
- Cube 的持续计算时间；
- 每次 AIC/AIV 握手对应的有效输出量。

`256×256` 不会导致 epilogue UB 超限，因为一个 AIC 配两个 AIV subblock，每个 AIV 只处理约 128 行。之前出现 UB 超限，是按每个 AIV 都保存完整 256 行规划导致的。

### 4.3 为什么 baseK 取 128，K window 取 256

`baseK=128` 是 BlockMmad 的内层 Cube K tile。一次 L1 window 保存 256 个 K 元素，因此：

```text
一个 L1 K window = 两个 baseK tile
stepKa = stepKb = 2
```

这样可以同时满足：

- INT8 A/B 的 Cube K 对齐要求；
- L0A/L0B 单次 tile 大小可控；
- L1 中 A/B 可以采用双缓冲；
- K 不要求整除 128 或 256，最后一个 K window/tile 由 BlockMmad 处理 tail。

per-channel 没有 `K % 256 == 0` 的逻辑约束。Host 对任意正 K 都允许进入该路径；`kAL1/kBL1` 对小 K 仍按对齐后的 window 处理。

## 5. Kernel 计算流水

### 5.1 整体阶段

当前流水为：

```text
全部 AIV：完整 W4→W8
全部 AIV：完整计算 rowSum(x)
                    |
               唯一一次 SyncAll
                    |
每个 AIC/AIV 组：
  AIC WaitForVector（slot 被使用过时）
  AIC 计算一个 M×N tile 到 per-core FP16 workspace
  AIC NotifyVector
  两个 AIV WaitForCube
  两个 AIV 分别处理约一半 M 行
  AIV 写最终 out
  两个 AIV NotifyCube
                    |
AIC 在退出前 WaitForVector，回收最后一个 tile
```

weight cast 和 RowSum 顺序执行，共用一个前处理阶段，两者之间没有全核同步。唯一一次 `SyncAll` 位于二者都完成之后，保证：

- AIC 读取 expanded W8 前，所有 weight tile 已经转换完成；
- AIV epilogue 读取 RowSum 前，所有 `[M]` RowSum 已经写完。

### 5.2 W4→W8 预处理

所有 AIV 按以下总任务数分配工作：

```text
taskCount = GetBlockNum() * GetTaskRation()
```

ND 按连续元素 tile 解包；NZ 按物理 block 重排：

```text
INT4 NZ C0=64 → INT8 NZ C0=32
```

特殊 ENK NZ 会在预处理阶段完成 ENK→EKN 转换。expanded workspace 中始终是标准计算布局。特殊 ENK 转换会主动将 K/N padding 写零；普通 NZ 转换当前会保留输入物理块中的 padding，因此依赖上游 ND2NZ 已经将输入 padding 清零。

### 5.3 RowSum 预处理

RowSum 只在 AIV 上执行，每个 AIV 以行交错方式处理：

```text
for row = taskIdx; row < M; row += taskCount
```

一行 K 方向按 4096 元素分块：

```text
INT8 → FP16 → FP32 → ReduceSum
```

每个 K chunk 的结果累加到 FP32 标量，最后写入连续的 FP32 workspace `[M]`。K tail 不需要 padding，也没有整除约束。

### 5.4 AIC/AIV tile 流水

每个 AIC 对应一个独立 FP16 workspace slot。首个 tile 可以直接写；从第二个 tile 开始，AIC 必须等待对应的两个 AIV 完成上一 tile，才能复用该 slot。

AIC 使用 A8W8 Fixpipe BlockMmad 完成完整 K 循环和 INT32 L0C 累加。Fixpipe 加载 `[E,1,N]` UINT64 scale，输出 FP16 中间 tile。

两个 AIV 按如下规则拆分 M：

```text
firstHalfM = ceil(curM / 2)
AIV0 rows  = [0, firstHalfM)
AIV1 rows  = [firstHalfM, curM)
```

两部分无重叠、无遗漏。AIV 将中间 tile、RowSum、offset、perTokenScale 搬入 UB，计算最终结果后直接写 out。

## 6. 片上空间计算

以下预算按最大 tile `256×256×128` 计算。

### 6.1 L1

A/B 都是 INT8，K window 为 256，buffer 数为 2：

```text
A L1 = 2 × 256 × 256 × 1 B = 128 KiB
B L1 = 2 × 256 × 256 × 1 B = 128 KiB
```

Host 当前采用保守检查，还计入：

```text
scale           = 2 × 1 × 256 × 8 B = 4 KiB
perTokenScale   = 2 × 256 × 4 B     = 2 KiB
offset          = 2 × 256 × 4 B     = 2 KiB
bias            = 1 × 256 × 4 B     = 1 KiB
```

总检查值约为：

```text
128 + 128 + 4 + 2 + 2 + 1 = 265 KiB
```

A5 UT 使用的 L1 容量为 512 KiB，因此该配置可以通过。这里对 perTokenScale、offset、bias 的计入偏保守；它们主要由 AIV epilogue 使用，但保守预算不会影响正确性。

### 6.2 L0C

L0C 保存 INT32 累加结果：

```text
256 × 256 × sizeof(int32_t)
= 256 × 256 × 4 B
= 256 KiB
```

A5 L0C 容量为 256 KiB，因此：

```text
dbL0C = 1
```

如果开启 L0C 双缓冲，需要 512 KiB，无法装入当前 L0C。

### 6.3 Epilogue UB

每个 AIV 最多处理 128 行：

```text
FP16 input tile   = 128 × 256 × 2 B = 64 KiB
FP16 output tile  = 128 × 256 × 2 B = 64 KiB
perTokenScale     = 128 × 4 B       = 512 B
rowSum            = 128 × 4 B       = 512 B
offset            = 256 × 4 B       = 1 KiB
------------------------------------------------
合计                                     = 130 KiB
```

各 buffer 还按 32B 对齐。最大地址低于 A5 UT 使用的 256 KiB UB，也低于约 248 KiB 的实际可用预算假设。

### 6.4 前处理 UB

ND W4→W8 的最大本地 tile 为 24K 个逻辑元素：

```text
packed W4 input   = 24K / 2          = 12 KiB
INT8 output       = 24K × 1 B        = 24 KiB
FP16 cast buffer  = 24K × 2 B        = 48 KiB
------------------------------------------------
W4 cast 合计                            = 84 KiB
```

RowSum 的 4096 K chunk：

```text
INT8 input        = 4096 × 1 B       = 4 KiB
FP16 buffer       = 4096 × 2 B       = 8 KiB
FP32 buffer       = 4096 × 4 B       = 16 KiB
Reduce work       = 4096 × 4 B       = 16 KiB
Reduce output                           = 32 B
------------------------------------------------
RowSum 合计                          ≈ 44.03 KiB
```

两个 preprocess 共用同一个 `TPipe`，当前最坏 ND 预留合计约 128.03 KiB，低于 UB 容量。

NZ cast tile 更小：cast 部分约 7.5 KiB，加 RowSum 后约 51.5 KiB。

## 7. GM Workspace 计算

API workspace 的前 16 MiB 为系统保留区。Kernel 使用 `GetUserWorkspace(workspace)` 获得用户区起始地址，以下 offset 都相对用户区计算。

所有 segment 均按 512B 对齐。

### 7.1 Expanded weight

ND：

```text
expertStride = K × N
```

NZ：

```text
expertStride = AlignUp(K, 16) × AlignUp(N, 32)
```

INT4 与 packed INT32 解包后都是 INT8，因此 expanded workspace 完全相同：

```text
expandedWeightOffset = 0
expandedWeightSize   = AlignUp(E × expertStride, 512)
```

### 7.2 Per-core tile workspace

每个 AIC 一个 FP16 slot：

```text
tileWorkspaceOffset = expandedWeightOffset + expandedWeightSize

tileWorkspaceStride = AlignUp(baseM × baseN × sizeof(half), 512)

tileWorkspaceSize   = aicNum × tileWorkspaceStride
```

最大块下：

```text
tileWorkspaceStride
= AlignUp(256 × 256 × 2, 512)
= 131072 B
= 128 KiB
```

### 7.3 RowSum workspace

```text
rowSumOffset = tileWorkspaceOffset + tileWorkspaceSize
rowSumSize   = AlignUp(M × sizeof(float), 512)
```

### 7.4 总 workspace

```text
userWorkspaceSize = rowSumOffset + rowSumSize

API workspaceSize = 16 MiB + userWorkspaceSize
```

Host 对以下操作全部做显式溢出检查：

- K/N 对齐加法；
- `K×N`、`E×expertStride`；
- `baseM×baseN×sizeof(half)`；
- `aicNum×tileStride`；
- `M×sizeof(float)`；
- segment offset 累加；
- `uint64_t` 到 `size_t` 的转换。

### 7.5 示例

以 `E=2, M=512, K=1024, N=256, aicNum=32, ND` 为例：

```text
expertStride       = 1024 × 256 = 262144 B
expandedWeightSize = 2 × 262144 = 524288 B

tileStride         = 256 × 256 × 2 = 131072 B
tileWorkspaceSize  = 32 × 131072 = 4194304 B

rowSumSize         = AlignUp(512 × 4, 512) = 2048 B

userWorkspaceSize  = 524288 + 4194304 + 2048
                   = 4720640 B

API workspaceSize  = 16777216 + 4720640
                   = 21497856 B
```

## 8. TilingData 地址合同

专用 `GMMA8W4BasicApiParams` 显式保存：

```text
expandedWeightOffsetBytes
expandedWeightSizeBytes
expandedWeightStrideBytes

tileWorkspaceOffsetBytes
tileWorkspaceSizeBytes
tileWorkspaceStrideBytes

rowSumOffsetBytes
rowSumSizeBytes

userWorkspaceSizeBytes
```

ops-transformer adapter 只按这些 offset 填充 Tensor API facade：

```text
weightPreprocessParams.workspaceGmAddr = userWorkspace + expandedWeightOffset
mmadParams.cGmAddr                     = userWorkspace + tileWorkspaceOffset
epilogueParams.xRowSumGmAddr           = userWorkspace + rowSumOffset
```

Tensor API 的 BufferManager 只负责片上 L1/L0 buffer，不管理上述 GM workspace，因此这里没有直接复用通用 GM workspace manager。

## 9. 已完成验证

### 9.1 Host Tiling UT

GroupedMatmul Host 全量 UT 结果：

```text
432 / 432 passed
```

A8W4 专项覆盖：

- 四个 tiling key 精确值；
- per-channel/per-group 生产路由；
- `256/256/128`、`kAL1=kBL1=256`、`dbL0C=1`；
- ND/NZ expanded stride；
- 三段 workspace offset、size、stride 和 512B 对齐；
- workspace 随 AIC 核数变化；
- per-channel 任意 K 和 K tail；
- per-group 拒绝 `K % 256 != 0`；
- INT4 与 packed INT32 的 key、逻辑 tiling、workspace 等价；
- packed storage 8:1 shape 校验；
- 特殊 ENK NZ 输入与标准 W8 NZ 输出 stride 分离；
- bias 缺失、antiquantScale/antiquantOffset 非空等非法输入拒绝。

### 9.2 Kernel CPU UT

GroupedMatmul A8W4 Kernel UT 结果：

```text
6 / 6 passed
```

覆盖：

- ND signed INT4 的 `-8/-1/0/7` 边界值；
- 多 expert W4→W8；
- 普通 NZ `C0=64→C0=32`，并在输入 padding 已为零时验证零 padding 被保留；
- 特殊 ENK NZ→标准 EKN NZ；
- RowSum 跨 4096 chunk 及 K tail；
- `baseM=256` 时 epilogue UB 约 130 KiB；
- 两个 AIV 的 128/127 等 M tail 拆分；
- per-channel ND、非零 offset、不同 perTokenScale、K tail 的 CPU 数值参考。

注意：CPU simulator 无法完整模拟真实 AIC/AIV 混合核的局部握手，因此完整 facade 的 CPU 数值用 scalar reference 校验；W4 cast、NZ 重排和 RowSum 调用的是实际 Tensor API preprocess 实现。

### 9.3 API UT

GroupedMatmul OpAPI 全量 UT：

```text
216 / 216 passed
```

新增覆盖原生 INT4、packed INT32 和非对称 per-channel offset 场景。

### 9.4 ops-tensor 回归

PR 169 的 Kernel UT CMake 原本缺少 Tensor API 子模块 include path，并把 CANN 系统目录固定为 `x86_64-linux`。补齐 include path、按宿主架构选择 `aarch64-linux/x86_64-linux` 后，quant_batch_matmul 回归结果为：

```text
23 / 23 passed
```

覆盖已有 A8W8、MX 和 MIX 路径，未发现本次公共 BlockMmad 兼容修改造成回归。

### 9.5 A5 全量 Kernel 构建与产物

执行 A5 grouped_matmul 全量 Kernel 构建，结果：

```text
ops_transformer_kernel build passed
```

产物检查结果：

- FP16 ND/NZ A8W4 签名生成非空 `.o`，大小约 69 KiB/74.6 KiB；
- metadata 同时包含 `0x20600200`、`0x20600201`、`0x20600300`、`0x20600301`；
- Kernel 源码中只有一次全核 `SyncAll`；
- 两个仓库 `git diff --check` 通过。

## 10. 尚未验证和后续工作

以下内容不能由当前 CPU simulator 或编译检查替代：

- A5 实卡上的真实 AIC/AIV `Wait/Notify` 时序；
- 多轮滑窗下单 slot 是否发生死锁或提前复用；
- AIC 等待最后一个 Vector tile 的真实行为；
- ND/NZ、INT4/packed INT32 的实卡最终数值等价；
- 零 token expert、多个 expert 和复杂 M/N/K tail 的实卡组合；
- 真实带宽、Cube 利用率、AIV epilogue 是否成为瓶颈；
- 端到端性能对比，以及后续是否需要双 slot ping-pong。
- 普通 NZ 输入 padding 为非零时的主动清零；当前普通 NZ 路径依赖 ND2NZ 输入 padding 为零，只有特殊 ENK 转换路径会显式清零输出 padding。

因此当前结论是：Host、API、Tensor API 代码链、CPU 可验证的预处理逻辑和 A5 编译产物已经打通；真实设备上的混合核同步正确性、最终数值和性能仍需 A5 实卡验证。

## 11. 主要代码位置

- Host tiling：`op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp`
- 专用 tilingData：`op_kernel/arch35/grouped_matmul_tiling_data_apt.h`
- ops-transformer adapter：`op_kernel/arch35/a8w4_v5/gmm_a8w4_perchannel_tensor_api.h`
- APT 分流：`op_kernel/grouped_matmul_apt.cpp`
- Tensor API facade：`ops-tensor/include/blaze/gemm/kernel/kernel_gmm_a8w4.h`
- Weight preprocess：`ops-tensor/include/blaze/gemm/kernel/kernel_gmm_a8w4_weight_preprocess.h`
- RowSum preprocess：`ops-tensor/include/blaze/gemm/kernel/kernel_gmm_a8w4_rowsum_preprocess.h`
- AIV epilogue：`ops-tensor/include/blaze/epilogue/block/block_epilogue_gmm_a8w4.h`
- Host UT：`tests/ut/op_host/test_grouped_a8w4_basic_api_arch35_tiling.cpp`
- Kernel UT：`tests/ut/op_kernel/arch35/test_grouped_matmul_a8w4_v5.cpp`

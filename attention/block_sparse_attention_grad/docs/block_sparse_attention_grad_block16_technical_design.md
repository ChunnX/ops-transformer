# BlockSparseAttentionGrad 稀疏 Block 16 对齐支持技术方案

## 1. 文档目的

本文说明 Ascend 950（DAV_3510/Arch35）`BlockSparseAttentionGrad` 支持稀疏 block 粒度按 16 对齐的设计与实现，重点覆盖：

- 反向计算公式与稀疏语义；
- Host API、Tiling、Kernel 的完整调用链；
- 逻辑稀疏 block 与硬件计算 tile 的关系；
- AIC/AIV 协同计算流、任务切分和尾块处理；
- 本次代码修改的原因、正确性依据和资源约束；
- 已完成测试、风险边界和后续扩展建议。

本文描述的实现位于：

```text
attention/block_sparse_attention_grad/
```

## 2. 背景与目标

原 Arch35 实现要求 `blockShapeX`、`blockShapeY` 为 64 的整数倍，无法表达 16、32、48 等更细粒度的稀疏 pattern。更细的 block 可以减少无效 token 对的计算，但会带来以下变化：

1. 一个逻辑稀疏 block 可能小于原物理计算 tile；
2. AIC 矩阵乘和 NZ/ZN 搬运仍要求 16 对齐；
3. 两个 AIV 子核拆分 M 方向后，单个子核可能只处理 8 行；
4. 非方形 block 会导致 `baseM != baseN`，原 L1 偏移计算不能再假设二者相等；
5. 尾块的实际长度可能小于 16，但硬件计算长度必须向上对齐到 16。

本方案目标如下：

- `blockShapeX`、`blockShapeY` 支持正的 16 整数倍；
- `BlockSparseMask` 的逻辑粒度保持不变，不因物理 tile 改变而扩大或合并；
- 支持方形和非方形 block；
- 支持序列尾块不足一个完整 block 的场景；
- 保持 FP16/BF16、BNSD/BSND/TND 和 GQA 的既有计算路径；
- `dQ`、`dK`、`dV` 与 CPU Golden 结果满足既有精度标准。

## 3. 当前适用范围

Arch35 当前实现约束如下：

| 项目 | 当前约束 |
| --- | --- |
| NPU 架构 | DAV_3510，Ascend 950PR/950DT |
| 输入类型 | FP16、BF16 |
| `headDim` | 128 |
| Layout | BNSD、BSND、TND |
| 稀疏 mask | BOOL/UINT8，当前为必选输入 |
| `blockShapeX/Y` | 正数且按 16 对齐 |
| Deterministic | 不支持 |
| `attentionMask` | 当前只支持空指针 |
| `maskType` | 仅支持 0 |
| 滑窗 | `preTokens`、`nextTokens` 均须为 2147483647 |

当前已完成 NPU 验证的 block 范围为 16、32、48、64、128，以及 `[16, 32]`、`[32, 16]` 非方形组合。对于大于 128 的 block，见第 12 节的边界风险说明。

## 4. 数学计算逻辑

记：

- `Q`、`K`、`V` 为 Query、Key、Value；
- `dO` 为 attention 输出梯度；
- `O` 为正向 attention 输出；
- `scale` 通常为 `1 / sqrt(D)`；
- `M` 为从 block mask 展开得到的 token 级稀疏选择矩阵；
- `LSE` 为正向 Softmax 的 log-sum-exp 结果。

正向中间量为：

```text
S = scale * Q * K^T
P = exp(S - LSE)
O = (P .* M) * V
```

反向计算为：

```text
softmaxGradFront = reduce_sum(dO .* O, axis=D)
dP               = dO * V^T
dS               = P .* (dP - softmaxGradFront)
dV               = (P .* M)^T * dO
dQ               = scale * ((dS .* M) * K)
dK               = scale * ((dS .* M)^T * Q)
```

Kernel 不生成完整的稠密 `P`、`dP`、`dS`。它只遍历 `BlockSparseMask` 中值为 1 的逻辑 block，并在每个有效 block 内重算 `QK^T` 和 `dO V^T`，从而避免对无效 block 执行五次矩阵乘。

## 5. 稀疏 Mask 语义

### 5.1 逻辑形状

用户侧 mask 形状为：

```text
[B, Nq, ceilDiv(Sq, blockShapeX), ceilDiv(Skv, blockShapeY)]
```

其中：

- `mask[b, n, qb, kb] = 1`：逻辑 Q block `qb` 与 KV block `kb` 参与计算；
- `mask[b, n, qb, kb] = 0`：整个 token 子矩阵跳过；
- 最后一个 block 可小于声明的 block shape，按真实序列长度裁剪。

### 5.2 Arch35 内部排布

Arch35 ACLNN 入口将 mask 执行 `[0, 1, 3, 2]` 转置，内部连续排布变为 `[B, Nq, KVBlock, QBlock]`。Kernel 使用以下 offset 读取：

```text
offset = b * Nq * QBlockNum * KVBlockNum
       + n * QBlockNum * KVBlockNum
       + kvBlockIdx * QBlockNum
       + qBlockIdx
```

因此，修改物理 tile 大小不能改变 `qBlockIdx = s1 / blockShapeX` 和 `kvBlockIdx = s2 / blockShapeY` 的逻辑含义。

## 6. 整体软件与计算流

```text
ACLNN API
  |
  |-- 参数、dtype、layout、blockShape 校验
  |-- mask 转置并将输入转为连续内存
  v
Host Tiling
  |
  |-- 解析 B/N/S/D、GQA 分组、block shape
  |-- 生成 baseM/baseN/singleM
  |-- 分配 FP32 workspace
  |-- 生成 SoftmaxGradFront tiling
  v
Mixed Kernel (1 AIC : 2 AIV)
  |
  |-- AIV Pre: workspace 清零，计算 reduce_sum(dO * O)
  |-- Address: 遍历 B/N/S2/S1，读取 block mask，分发有效 tile
  |-- AIC MM1: Q * K^T
  |-- AIC MM2: dO * V^T
  |-- AIV: P = exp(scale * QK - LSE)
  |-- AIV: dS = P * (dP - softmaxGradFront)
  |-- AIC MM3: dV += P^T * dO
  |-- AIC MM4: dQ += dS * K
  |-- AIC MM5: dK += dS^T * Q
  |-- AIV Post: dQ/dK 乘 scale，FP32 转回输入 dtype
  v
dQ / dK / dV
```

## 7. Tiling 与切分策略

### 7.1 两级 block 概念

本方案区分两种 block：

1. **逻辑稀疏 block**：由 `blockShapeX × blockShapeY` 定义，决定 mask 的一个元素覆盖哪些 token；
2. **物理计算 tile**：由 `baseM × baseN` 定义，决定一次 Cube 矩阵乘和 Vector 处理的数据规模。

逻辑 block 决定正确性，物理 tile 决定硬件可执行性和性能。物理 tile 可以小于逻辑 block，但不能跨越两个具有不同 mask 值的逻辑 block。

### 7.2 当前 Host 参数选择

Arch35 当前使用：

```cpp
baseM = blockShapeX <= 32
            ? blockShapeX * 2
            : (blockShapeX <= 128 ? blockShapeX : 128);
baseN = blockShapeY <= 128 ? blockShapeY : 128;
singleM = (blockShapeX < 128 || blockShapeY < 128)
              ? blockShapeX
              : AlignTo(2048, baseM);
```

典型配置如下：

| `blockShape` | `baseM` | `baseN` | `singleM` | 说明 |
| --- | ---: | ---: | ---: | --- |
| `[16, 16]` | 32 | 16 | 16 | 物理 M 缓冲按 32 预留，任务不跨 16 行逻辑 block |
| `[32, 32]` | 64 | 32 | 32 | 物理 M 缓冲按 64 预留，任务不跨 32 行逻辑 block |
| `[48, 48]` | 48 | 48 | 48 | 直接按逻辑 block 计算 |
| `[64, 64]` | 64 | 64 | 64 | 直接按逻辑 block 计算 |
| `[128, 128]` | 128 | 128 | 2048 | 一个 singleM 聚合多个 Q block，由 mask 跳过无效块 |
| `[16, 32]` | 32 | 32 | 16 | 非方形 block |
| `[32, 16]` | 64 | 16 | 32 | 非方形 block |

`blockShapeX <= 32` 时扩大 `baseM` 的作用是保留满足 Cube/AIV 双子核执行所需的物理缓冲和基础 tile 能力；实际任务长度仍由 `s1Len` 决定，并使用 `s1LenAlign = AlignUp(s1Len, 16)` 执行。

### 7.3 遍历顺序与多核分发

地址计算模块的逻辑遍历顺序为：

```text
S1 -> S2 -> Q Head -> Batch
```

每个候选任务先检查逻辑 block mask。有效任务使用轮转方式分发给 Cube 核：

```text
ownerAic = validTaskIndex % cubeCoreNum
```

Mixed Kernel 使用 `KERNEL_TYPE_MIX_AIC_1_2`：

- 一个 AIC 对应两个 AIV 子核；
- AIC 的逻辑核号为 `GetBlockIdx()`；
- AIV 通过 `GetBlockIdx() / 2` 映射回对应 AIC；
- 两个 AIV 通过 `GetSubBlockIdx()` 将 M 方向分成上下两半。

### 7.4 SingleBlock 处理

`singleM` 是固定 `B/N/S2` 下沿 S1 聚合的任务窗口。`SingleBlock` 在窗口内：

1. 以 `baseM` 为上限生成 S1 tile；
2. 根据 `s1 / blockShapeX` 查找下一个有效逻辑 Q block；
3. 对同一个 S2 tile 连续累加 `dK`、`dV`；
4. 最后一段通过 `is_singlekv_last` 将 L0C 结果写回 FP32 workspace。

对于小 block，将 `singleM` 收缩为 `blockShapeX`，可以防止 `baseM` 大于逻辑 block 时一次计算覆盖两个 mask 值不同的 Q block。

### 7.5 尾块对齐

真实序列长度不要求为 16 的整数倍。每个任务记录：

```text
s1LenAlign = AlignUp(s1Len, 16)
s2LenAlign = AlignUp(s2Len, 16)
```

GM 搬运使用真实长度，L1/L0、NZ/ZN 和 MMAD 使用对齐长度；Fixpipe 写回使用真实 `mSize/nSize`，避免把 padding 写入下一个逻辑区域。

## 8. AIC/AIV 计算流

### 8.1 前处理

所有 AIV 均分执行：

1. 将 `dQ/dK/dV` FP32 workspace 清零；
2. 对每个 Q token 计算：

```text
softmaxGradFront = reduce_sum(dO * O, axis=D)
```

结果以逻辑 `[B, N, S, 8]` 排布存入 workspace。每行只使用广播所需值，其余空间用于满足 DataCopy 和 Vector API 的块对齐。

### 8.2 MM1：重算注意力分数

AIC 执行：

```text
QK = Q_tile * K_tile^T
```

Q/K 从 GM 搬到 L1，再转换为 Cube 需要的 NZ/ZN 格式。`K` 在同一 S2 tile 下可通过 ping-pong L1 缓存复用。

本次将 MMAD 的 M 参数由特殊处理的真实长度改为 `mProcessAlign`：

```cpp
madParams.m = mProcessAlign;
```

原因是 16 粒度下 M 可能为 16、32、48 或不足 16 的尾块，Cube 的矩阵布局和后续 Fixpipe 均以 16 对齐形状组织。只对 `m == 1` 特判为 2 无法保证 L0C 形状与 `s1LenAlign` 一致。

### 8.3 MM2：计算 dP

AIC 执行：

```text
dP = dO_tile * V_tile^T
```

QK 和 dP 分别写入两组 ping-pong UB，随后通过跨核事件通知两个 AIV 子核。

### 8.4 Vector：恢复 P 并计算 dS

每个 AIV 子核处理 `s1LenAlign / 2` 行：

```text
P  = exp(scale * QK - LSE)
dS = P * (dP - softmaxGradFront)
```

Vector 结果从 FP32 转为输入类型并执行 ND -> NZ，写入共享 L1 的 `P`、`dS` ping-pong 区，供后续 MM3/MM4/MM5 使用。

当尾块的实际 M 小于半块时，只有对应子核执行有效数据搬运；物理计算仍按对齐长度进行，最终写回按真实长度裁剪。

### 8.5 MM3/MM4/MM5：梯度累加

AIC 执行：

```text
dV += P^T * dO
dQ += dS * K
dK += dS^T * Q
```

由于不同稀疏 block、不同 Q head 或 GQA group 可能写入同一梯度区域，结果先以 FP32 通过原子加写入 workspace：

- `dQ`：同一 Q tile 对多个有效 KV block 的贡献累加；
- `dK/dV`：多个 Q tile、多个 GQA Q head 对同一 KV head 的贡献累加。

GQA 映射关系为：

```text
qGroup = qHeadNum / kvHeadNum
n2Idx  = n1Idx / qGroup
```

### 8.6 后处理

全部 AIC/AIV 完成后，AIV 均分 workspace：

```text
dQ = Cast(scale * dQ_workspace)
dK = Cast(scale * dK_workspace)
dV = Cast(dV_workspace)
```

最终输出类型与输入 Q/K/V 保持一致。

## 9. 流水与同步

Kernel 使用两级 ping-pong：

- `taskId % 2` 切换当前 QK/dP UB、P/dS L1；
- `kv_ping_pong_idx` 切换 K/V L1，支持固定 S2 数据复用。

关键跨核事件如下：

| Flag | 方向 | 含义 |
| --- | --- | --- |
| `FLAG_C1_V1` | AIC -> AIV | QK 已写入 UB，可恢复 P |
| `FLAG_C2_V2` | AIC -> AIV | dP 已写入 UB，可计算 dS |
| `FLAG_V1_C3` | AIV -> AIC | P 已写入 L1，可计算 dV |
| `FLAG_V2_C45` | AIV -> AIC | dS 已写入 L1，可计算 dQ/dK |
| `FLAG_CUBE_POST` | AIC/AIV | 前处理和主循环结束后的全局阶段同步 |

主循环形成软件流水：当前任务执行 MM1/MM2 时，上一个任务执行 MM3/MM4/MM5，从而覆盖部分搬运和 Vector 计算延迟。

## 10. 本次代码修改说明

### 10.1 ACLNN API 增加 16 对齐校验

文件：`op_host/op_api/aclnn_block_sparse_attention_grad.cpp`

修改点：

```text
blockShapeOptional 非空
-> 至少包含两个元素
-> X/Y 均为正数
-> X/Y 均满足 value % 16 == 0
```

作用：在执行器创建前返回清晰的参数错误，避免非法 shape 进入 Tiling/Kernel。

### 10.2 Arch35 能力校验从 64 放宽到 16

文件：`op_host/arch35/block_sparse_attention_grad_tiling_arch35.cpp`

修改：

```text
blockShapeX % 64 == 0 && blockShapeY % 64 == 0
```

调整为：

```text
blockShapeX % 16 == 0 && blockShapeY % 16 == 0
```

作用：Tiling 模板正式接受 16 粒度 block，与 ACLNN API 约束保持一致。

### 10.3 自适应 baseM/baseN/singleM

文件：`op_host/arch35/block_sparse_attention_grad_tiling_arch35.cpp`

原实现固定以最大 128 生成物理 tile，并将 `singleM` 对齐到 2048。对于 16/32 block，`baseM` 可能跨越多个逻辑 mask block，导致把未选中的 token 区域一起计算。

新策略使小 block 的任务边界与逻辑 Q block 对齐，同时保留硬件所需物理缓冲空间。

### 10.4 修正 Value L1 ping-pong 偏移

文件：`op_kernel/arc35/cube_op.h`

Value L1 tensor 的容量按 `baseN * headDimAlign` 申请，偏移也必须按同一容量推进。原偏移错误使用 `baseM`：

```text
错误：offset += baseM * headDimAlign
正确：offset += baseN * headDimAlign
```

当 `baseM != baseN` 时，原实现可能产生：

- `baseM < baseN`：后续 buffer 与 Value buffer 重叠，引发数据覆盖；
- `baseM > baseN`：产生空洞并增加 L1 越界风险。

该修复是支持 `[16, 32]`、`[32, 16]` 等非方形 block 的必要条件。

### 10.5 MM1/MM2 使用对齐 M

文件：`op_kernel/arc35/cube_op.h`

将：

```cpp
madParams.m = mProcess == 1 ? 2 : mProcess;
```

调整为：

```cpp
madParams.m = mProcessAlign;
```

作用：保证 MMAD、L0C 和 Fixpipe 对 M 维的形状理解一致，统一处理 16 粒度和不足 16 的序列尾块。

### 10.6 新增泛化测试

文件：`tests/test_bsa_generalization.py`

测试脚本复用 `/home/dhn/test_bsa.py` 的 CPU Golden、mask 展开和精度比较逻辑，每个 case 在独立 Python 进程中运行，隔离 NPU runtime 状态。

### 10.7 Host Tiling UT 覆盖

文件：`tests/ut/op_host/test_block_sparse_attention_grad_tiling.cpp`

当前工作区还包含以下 Arch35 Tiling 用例：

- `[64, 64]` 原能力回归；
- `[16, 16]` 且序列长度为 70 的尾块；
- `[16, 32]` 非方形 block；
- `[24, 16]` 非 16 对齐输入拒绝。

这些用例验证 Host 侧模板选择、TilingKey `1003` 和参数校验，不替代 NPU Kernel 精度测试。

## 11. 片上内存与 Workspace 设计

### 11.1 UB

主 QK/dP ping-pong FP32 缓冲约为：

```text
4 * (baseM / 2) * baseN * sizeof(float)
```

最大 `baseM = baseN = 128` 时约 128 KiB。加上 P/dS、LSE、SoftmaxGradFront 等 Vector 缓冲后仍受 247 KiB UB 上限约束。

### 11.2 L1

共享 P/dS ping-pong：

```text
4 * baseM * baseN * sizeof(inputType)
```

Cube Q/K/V/dO ping-pong：

```text
2 * baseM * DAlign * sizeof(inputType)
+ 6 * baseN * DAlign * sizeof(inputType)
```

在 `baseM/baseN <= 128`、`DAlign = 128` 时总量小于 512 KiB L1 上限。

### 11.3 GM Workspace

Workspace 使用 FP32 保存：

```text
softmaxGradFront: B * Nq * Sq * 8
dQ workspace:    B * Nq  * Sq  * D
dK workspace:    B * Nkv * Skv * D
dV workspace:    B * Nkv * Skv * D
```

TND 使用 total token 数替代固定 `B * S`。FP32 workspace 用于规避 FP16/BF16 多次原子累加带来的精度损失。

## 12. 正确性边界与风险

### 12.1 已解决问题

- 16/32 小 block 不再被 64 对齐校验拒绝；
- 小 Q block 不会因 `singleM = 2048` 被物理 tile 跨 mask 计算；
- 非方形 block 的 Value L1 不再发生偏移重叠或空洞；
- MMAD 对尾块统一使用 16 对齐 M；
- 序列尾块通过真实长度搬运和对齐长度计算兼容。

### 12.2 大于 128 的非整倍数 block 风险

Host 当前只校验“正数且为 16 的整数倍”，但当 `blockShapeX` 或 `blockShapeY` 大于 128 且不是 128 的整数倍时，固定 `baseM/baseN = 128` 可能跨越逻辑 sparse block 边界。

例如 `blockShapeY = 144`：

```text
第一个 S2 tile: [0, 128)   -> 逻辑 KV block 0
第二个 S2 tile: [128, 256) -> 跨越逻辑 KV block 0 和 1
```

如果两个逻辑 KV block 的 mask 值不同，按第二个 tile 起点读取一次 mask 会产生错误计算。

生产化必须选择以下策略之一：

1. **接口限界**：明确约束 `16 <= blockShapeX/Y <= 128`；
2. **边界感知切分**：tile 长度取 `min(base, seqTail, logicalBlockTail)`，确保每个 tile 不跨逻辑 block；
3. **整除约束**：允许大于 128，但要求 `blockShapeX/Y % 128 == 0`。

推荐策略 2，兼容性最好。实现时 S1 和 S2 的推进函数都必须使用逻辑 block 剩余长度，不能只修改 MMAD 参数。

### 12.3 验证覆盖缺口

当前快速回归主要覆盖 BNSD/FP16。完整交付建议补充：

- BF16：16、48、128；
- BSND：16、32、非方形；
- TND：多 Batch 变长、空序列、尾块；
- GQA：`Nq/Nkv = 2、4、8`；
- 稀疏率：0%、10%、50%、90%、100%；
- 负向参数：0、负数、非 16 对齐、mask shape 不匹配；
- 大 block：144、160、256，用于验证第 12.2 节策略。

## 13. 已完成验证

验证环境：

```text
Conda:     /root/miniforge3/envs/dhn
CANN:      /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
Python:    3.13.13
Torch:     2.10.0+cpu
Torch NPU: 2.10.0
Device:    ASCEND_RT_VISIBLE_DEVICES=3, logical device 0
```

构建命令：

```bash
bash build.sh --pkg --soc=ascend950 --ops=block_sparse_attention_grad -j16
```

泛化测试结果：

| Case | SeqLen | Block | 结果 |
| --- | ---: | --- | --- |
| `block16_multiblock` | 64 | `[16, 16]` | PASS |
| `block32_multiblock` | 128 | `[32, 32]` | PASS |
| `block48_multiblock` | 144 | `[48, 48]` | PASS |
| `block64_multiblock` | 128 | `[64, 64]` | PASS |
| `block16_tail` | 70 | `[16, 16]` | PASS |
| `block16x32_tail` | 70 | `[16, 32]` | PASS |
| `block32x16_tail` | 70 | `[32, 16]` | PASS |
| `block128_regression` | 404 | `[128, 128]` | PASS |

汇总：`8/8 cases passed`。每个 case 的 `dQ/dK/dV` 均通过 CPU Golden 精度比较，且未检测到 NaN。

## 14. 验收标准

本功能满足以下条件时可验收：

1. ACLNN API 对非正数和非 16 对齐 block 返回参数错误；
2. 16、32、48、64、128 方形 block 正确；
3. 至少覆盖 `[16, 32]`、`[32, 16]` 非方形 block；
4. Q/KV 序列尾块不足 16 时结果正确且无越界；
5. BNSD、BSND、TND 与 FP16、BF16 的目标组合通过；
6. MHA/GQA 的 `dK/dV` 累加结果正确；
7. 稀疏率边界用例无 NaN、无死锁；
8. 构建通过，`git diff --check` 通过；
9. 若接口继续声明支持任意 16 整数倍，必须关闭第 12.2 节的大 block 跨界风险。

## 15. 结论

本次改造不是简单地将校验条件从 64 改为 16，而是同时处理了小 block 的任务边界、物理 tile、非方形 L1 布局以及 MMAD 对齐形状。核心原则是：

> 稀疏语义始终由逻辑 block mask 决定，物理计算 tile 必须在逻辑 block 边界内按 16 对齐执行。

在当前已验证的 `blockShapeX/Y <= 128` 范围内，方案已通过 8 组 NPU 泛化测试。若要把接口能力扩展为任意大于 128 的 16 对齐 block，应进一步引入边界感知的 S1/S2 tile 切分。

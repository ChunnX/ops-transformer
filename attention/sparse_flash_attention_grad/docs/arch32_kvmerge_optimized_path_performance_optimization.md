# SFAG Arch32 KVmerge Optimized 路径性能优化方案

## 1. 背景

当前 Sparse Flash Attention Grad（SFAG）Arch32 KVmerge optimized 路径已经完成以下优化：

- 去除 `valueGm -> UB -> selectedVWorkspace` 的重复搬运，KVmerge 场景只 Gather Key。
- 将 `selectedKWorkspace` 从 `[selectedS2, 576]` 调整为两个连续平面：
  - Key：`[selectedS2, 512]`
  - Rope：`[selectedS2, 64]`
- 利用新的平面布局保证 Key 每行 stride 为 512 元素，从而改善 L2/cacheline 对齐。

上述改动已获得约 10% 性能收益。当前性能瓶颈已经转移到 Fixpipe，其他流水大部分被 Fixpipe 掩盖。

本文总结在**不改变当前计算公式和 scale 位置**的前提下，可继续尝试的优化方案。当前 KVmerge 计算保持为：

```text
dK_out = dK_raw * scale + dV
```

## 2. 当前关键流水

Cube4 和 Cube5 当前分别计算并输出结果：

```text
Cube4: dK_raw = dS^T * Q
       576 维 = 128 + 128 + 128 + 128 + 64
       共 5 次 Fixpipe，写入 mm4ResWorkspace

Cube5: dV = P^T * dY
       512 维 = 128 + 128 + 128 + 128
       共 4 次 Fixpipe，写入 mm5ResWorkspace
```

随后 Vector optimized scatter 执行：

```text
mm4 NZ -> ND
mm5 NZ -> ND
dK_raw *= scale
dK_raw[:, 0:512] += dV
scatter 到最终 dK workspace
```

相关代码主要位于：

- `basic_modules/cube_modules/cube4.h`
- `basic_modules/cube_modules/cube5.h`
- `basic_modules/sparse_flash_attention_grad_cube_op.h`
- `basic_modules/sparse_flash_attention_grad_matmul.h`
- `basic_modules/sparse_flash_attention_grad_vec_op.h`

## 3. 优化方案概览

| 优先级 | 方案 | 是否改变公式 | 预期端到端收益 | 改动风险 |
| --- | --- | --- | --- | --- |
| P0 | 检查并优化 mm4/mm5 workspace 的 512B 对齐 | 否 | 0%～8% | 低至中 |
| P0 | 开启并正确适配 unitFlag | 否 | 0%～8% | 中 |
| P1 | 小 `singleM` 场景使用 `N=256` | 否 | 0%～10% | 中至高 |
| P1 | Cube4/Cube5 按 D tile 交织调度 | 否 | 0%～6% | 中至高 |
| P2 | 增大 optimized scatter 的 `UB_ROW_SIZE` | 否 | 0%～3% | 低至中 |
| P2 | Fixpipe 直接输出 ND | 否 | 0%～5% | 中 |
| P3 | Rope 64 维尾块合批 | 否 | 0%～4% | 高 |

收益范围高度依赖实际 shape 和流水 timeline。若 Fixpipe 已经连续满载且没有明显空泡，不减少 Fixpipe 数据量的方案通常只能获得几个百分点。

## 4. 方案一：mm4/mm5 workspace 512B 对齐

### 4.1 优化目标

除了 workspace 总体起始地址，还需要检查以下各级地址是否保持 512B 对齐：

- `mm4ResWorkspaceGm` 起始地址
- `mm5ResWorkspaceGm` 起始地址
- 每个 `scatterTaskId` 对应的 buffer 起始地址
- 每个 AIC core 对应的 buffer 起始地址
- 每个 `mIdx` tile 的起始地址
- 不同 `dIdx` Fixpipe 输出 tile 的起始地址

optimized 路径输出紧密 NZ 数据，tile 字节数近似为：

```text
tileBytes = singleM * singleN * sizeof(float)
```

主块 `singleN=128` 时：

```text
tileBytes = singleM * 512B
```

因此整数 `singleM` 下主块天然为 512B 整数倍。但是以下情况仍需单独检查：

- Rope 的 `singleN=64` 尾块
- last block 的小 `singleM`
- mm4 和 mm5 workspace 之间的切换
- scatter 三缓冲区之间的地址跨度
- 对齐后的物理基址与逻辑 offset 是否一致

### 4.2 可选实现

为每个输出 tile 设置独立 pitch：

```cpp
tilePitchBytes = AlignUp(singleM * singleN * sizeof(float), 512);
```

写入和读取端统一使用该 pitch。需要同步修改：

- Cube4/Cube5 的输出 offset
- optimized scatter 的 `currentDkSrcOffset/currentDvSrcOffset`
- workspace 大小计算
- scatter buffer 的 core/task stride

### 4.3 预期收益

- 如果当前已经对齐：接近 0%。
- 如果存在 Rope、小 M 或 buffer base 拆包：约 2%～8%。

建议先通过 profiler 检查 Fixpipe 写 GM 的 transaction、带宽利用率和拆包情况，再决定是否修改布局。

## 5. 方案二：开启 unitFlag

### 5.1 当前状态

SFAG 的 `MmadInnerWithSyncFixpNzOut()` 当前通过显式事件完成 M/FIX 同步：

```cpp
WaitFlag<FIX_M>(l0cEvent);
Mmad(...);

SetFlag<M_FIX>(l0cEvent);
WaitFlag<M_FIX>(l0cEvent);
Fixpipe(...);

SetFlag<FIX_M>(l0cEvent);
```

当前没有设置：

```cpp
mmadParams.unitFlag
fixpipeParams.unitFlag
```

前向 MLA Arch22 已存在 unitFlag 用法，可作为同步语义参考。

### 5.2 优化方向

在 optimized NZ 输出接口中实验：

```cpp
mmadParams.unitFlag = 0b11;
fixpipeParams.unitFlag = 0b11;
```

不能只增加参数，还需要重新核对：

- `M_FIX` 和 `FIX_M` 是否仍全部需要
- 两个 L0C ping-pong buffer 的复用时机
- 中间 MMAD和最后一次 MMAD的 unitFlag差异
- 函数退出前是否需要额外完成同步
- Cube切换和 kernel结束前是否存在未完成的 Fixpipe

### 5.3 预期收益

- Fixpipe 已经连续满载：约 0%～3%。
- 每个 tile 之间存在明显 M/FIX 同步空泡：约 3%～8%。
- 小 M 或64维尾块比例较高时，空泡占比可能更明显。

unitFlag 不减少 Fixpipe 输出量，只能减少流水握手和发射空泡。

## 6. 方案三：小 singleM 场景使用 N=256

### 6.1 优化目标

当前固定：

```cpp
N_SPLIT_SIZE = 128;
```

因此 Cube4/Cube5 分别需要5次和4次 Fixpipe。对于小 `singleM`，可以增大 N tile：

```text
singleM <= 64：N_SPLIT_SIZE = 256
singleM > 64： N_SPLIT_SIZE = 128
```

对应 Fixpipe 次数可以降低为：

```text
Cube5: 512 = 256 + 256             4次 -> 2次
Cube4: 576 = 256 + 256 + 64        5次 -> 3次
```

公式和 scale 顺序保持不变。

### 6.2 容量约束

FP32 L0C 输出大小为：

```text
l0cBytes = singleM * singleN * sizeof(float)
```

典型情况：

```text
64  * 256 * 4B = 64KB
128 * 128 * 4B = 64KB
128 * 256 * 4B = 128KB
```

因此 `N=256` 应只对小 M 开启。还需检查：

- L0B是否能容纳相应数据
- `LoadDataMm1B()` 是否支持该 N
- L0C ping-pong 是否仍有足够空间
- NZ workspace offset和 tile pitch
- `NzToNd()` 是否支持256宽源 tile
- 512维 Key 平面与64维 Rope平面的地址计算

### 6.3 预期收益

- 小 M tile占比较高：约 4%～10%。
- 主路径几乎都是 `singleM=128`：接近 0%。

该方案是“不改公式”条件下最有机会获得明显收益的方向之一。

## 7. 方案四：Cube4/Cube5 按 D tile 交织

### 7.1 当前调度

当前 `cube345Process()` 顺序为：

```cpp
cube5Process();
cube4Process();
cube3Process();
```

Cube5连续完成4个 D tile，随后 Cube4连续完成5个 D tile。

### 7.2 优化方向

保持 mm4/mm5 独立输出和现有公式，只改变调度顺序：

```text
Cube5 D0 -> Fixpipe L0C0
Cube4 D0 -> Fixpipe L0C1
Cube5 D1 -> Fixpipe L0C0
Cube4 D1 -> Fixpipe L0C1
...
Cube4 Rope -> Fixpipe
```

潜在收益来源：

- 改善两个 L0C buffer 的交替使用
- 利用 Cube4和Cube5不同的 L1 输入，增加 MTE1/M/FIX重叠机会
- 避免某一路径内部连续等待同一资源
- 让一组输入搬运尽量遮盖另一组结果的 Fixpipe

### 7.3 实现影响

需要将 Cube4和Cube5拆为 tile级接口，并重新处理：

- L1 DS、P、Query和DY的生命周期
- L0A/L0B ping-pong flag
- L0C ping-pong flag
- `MM_L1_DS_EVENT`
- `MM_L1_P_EVENT`
- reload Query/DY的 last block分支

### 7.4 预期收益

- 当前存在 L1 或 L0C切换空泡：约 2%～6%。
- Fixpipe本来就连续发射：接近 0%。

## 8. 方案五：增大 optimized scatter 的 UB_ROW_SIZE

### 8.1 当前状态

optimized scatter 当前使用：

```cpp
UB_ROW_SIZE = 4;
```

每轮只处理4行，需要重复执行：

```text
mm4 NZ -> ND
mm5 NZ -> ND
Muls(dK)
Add(dK, dV)
scatter
```

### 8.2 优化方向

根据 UB容量尝试：

```cpp
UB_ROW_SIZE = 8;
```

若空间仍充足，可进一步测试16。需要重新计算并检查：

- `scatterAddTensorK/V`
- `tempTensorK/V`
- ping-pong offset
- `rowsumUbOffset`
- 整体 UB峰值
- bank conflict和对齐

### 8.3 预期收益

可以减少：

- NZ2ND循环次数
- event同步次数
- DataCopyPad发射次数
- indices读取和循环分支

如果 Vector已经被 Fixpipe完全掩盖，收益约0%～3%；如果 Cube结束后仍暴露明显 scatter尾巴，收益可能更高。

## 9. 方案六：Fixpipe 直接输出 ND

### 9.1 当前路径

```text
L0C -> Fixpipe NZ workspace
NZ workspace -> UB
NZ2ND
Muls + Add
scatter
```

### 9.2 实验路径

```text
L0C -> Fixpipe ND workspace
ND workspace -> UB
Muls + Add
scatter
```

该方案可以删除 Vector侧的两次 NZ2ND，但需要重新设计：

- Fixpipe `dstStride`
- ND workspace 行 stride
- mm4/mm5 tile offset
- scatter连续读取方式
- last block和小 M布局

### 9.3 风险与收益

风险包括：

- NZ是Cube天然输出格式，ND Fixpipe本身可能效率更低
- ND行 stride可能产生新的拆包
- 小 M下输出事务可能更碎
- NZ2ND当前可能已经被 Fixpipe掩盖

建议通过编译期开关做 A/B。预期收益约0%～5%。

## 10. 方案七：Rope 64维尾块合批

Cube4最后一个 `M x 64` tile 的 Fixpipe宽度利用率低于主块。可以尝试把多个独立 Rope任务合批，例如：

```text
任务 A：M x 64
任务 B：M x 64
```

组合成一次更大的 Fixpipe任务，或者利用：

```cpp
fixpipeParams.ndNum
fixpipeParams.srcNdStride
fixpipeParams.dstNdStride
```

一次输出多个独立 ND/NZ区域。

主要难点：

- 两个任务的 L0C布局和生命周期不同
- GM目标地址未必连续
- 合批可能增加同步复杂度
- 需要保证不会延迟主流水

预期端到端收益约0%～4%，建议放在其他方案之后评估。

## 11. 推荐验证顺序

### 阶段一：低成本诊断和实验

1. 检查 mm4/mm5 所有层级地址的512B对齐。
2. 从 profiler确认 Fixpipe是否存在明显 M/FIX发射空泡。
3. 开启 unitFlag并进行单变量 A/B。
4. 将 `UB_ROW_SIZE` 从4调整到8进行单变量 A/B。

### 阶段二：根据 shape 优化 tile

1. 统计运行时 `singleM` 分布。
2. 如果 `singleM <= 64` 占比较高，增加 `N=256` 分支。
3. 分别比较主块、last block和 Rope尾块的耗时。

### 阶段三：流水重排

1. 将 Cube4/Cube5 拆成 tile级接口。
2. 实验交织调度。
3. 评估 ND直出。
4. 最后评估 Rope尾块合批。

## 12. 每项实验建议采集的指标

每次只修改一个变量，并至少记录：

- 端到端 kernel耗时
- Cube侧耗时和 Vector侧尾部耗时
- Fixpipe busy ratio
- Fixpipe调用次数
- 相邻 Fixpipe调用间的空泡
- M/FIX同步等待时间
- GM写带宽和 transaction数量
- L2命中率及拆包情况
- 主块与 last block分别耗时
- FP16/BF16精度结果

建议对同一 shape进行多轮预热和重复测量，报告中同时给出均值、P50和波动范围，避免将调度噪声误判为收益。

## 13. 总结

在不改变 `dK * scale + dV` 公式、不改变浮点计算顺序的条件下，优化空间主要来自：

1. 消除 Fixpipe地址拆包。
2. 利用 unitFlag减少 M/FIX同步空泡。
3. 在小 M场景扩大 N tile，减少 Fixpipe调用次数。
4. 改善 Cube4/Cube5与双 L0C之间的调度重叠。
5. 缩短 Vector scatter尾部。

若 profiler确认 Fixpipe已经无空泡、满带宽持续运行，同时 `singleM=128` 占绝大多数，那么上述方案的总收益通常只有几个百分点。此时若要继续获得两位数收益，需要进一步减少 Fixpipe输出次数或输出字节，这将超出本文“不改变当前公式与 scale 位置”的范围。

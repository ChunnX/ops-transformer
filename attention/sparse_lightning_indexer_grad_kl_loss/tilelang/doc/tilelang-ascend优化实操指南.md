# TileLang-Ascend 优化实操指南

## 1. 适用范围

本文面向在 TileLang 中手写 Ascend AIC/AIV 混合 kernel 的开发者，重点覆盖：

- attention、indexer、softmax、梯度和 scatter 类算子；
- 手工管理 GM、UB、L1、L0A、L0B、L0C；
- 手工管理 MTE2、MTE1、Vector、MMA、FIX、MTE3 event；
- 从 TileLang 生成 AscendC 后进行精度、异常和性能定位；
- 将官方 AscendC service 迁移为 TileLang 时的能力边界判断。

本文经验来自 `sparse_lightning_indexer_grad_kl_loss/tilelang/0721.py` 的完整重写和
优化过程，但方法可复用于其他 TileLang-Ascend kernel。

## 2. 第一原则：生成 C++ 才是可执行事实

遇到源码意图、文档描述和实际行为不一致时，按以下优先级判断：

1. 当前安装版本的 TileLang-Ascend intrinsic 和 compiler pass；
2. 当前 shape 生成的 AscendC C++；
3. 官方 AscendC 源码和编程指南；
4. 已验证的历史 kernel；
5. 其他 TileLang 版本或未运行的实验代码。

不要只看 Python/TIR 就判断：

- 两个逻辑 buffer 是否物理独立；
- slice 是否保持期望的 L0 物理 shape；
- event 是否被 compiler 改写或补充；
- scalar 是否进入寄存器还是变成 UB `GetValue`；
- Vector repeat/stride immediate 是否溢出；
- copy 是否变成逐行、多 block 或错误 tail。

每个结构性改动后至少搜索生成 C++：

```text
GetWithOffset
copy_gm_to_ub / copy_gm_to_l1
copy_l1_to_l0a / copy_l1_to_l0b
mma / Mmad
copy_l0c_to_gm / Fixpipe
CompareScalar / Select / Exp
SetFlag / WaitFlag / CrossCoreSetFlag / CrossCoreWaitFlag
PipeBarrier / SyncAll
```

## 3. 建立可信基线

### 3.1 基线必须同时包含

- 精度正确的 TileLang 源码；
- 对应 shape 的生成 C++；
- CANN、TileLang-Ascend、芯片型号和 core 数；
- 测试输入 shape 和 dtype；
- warmup、bench 次数和计时方式；
- 关键 workspace 的 host 对照公式；
- 重复中位数，而不是一次最小值。

不能把“结构看起来更像官方”的实验版本当基线。0721 开发中，
`0715_bn128.py` 本身也会产生 NaN；真正可用的参考是动态 0715 的 `32x64`
物理 backward tile。

### 3.2 一次只改一个架构变量

将改动分类为：

```text
数学公式
tile 形状
buffer 布局
event ownership
阶段顺序
core 分区
workspace ABI
```

一次只改变其中一类。否则精度恢复或性能变化后无法归因。

## 4. 先验证 ABI，再调 kernel

### 4.1 输入、workspace、输出顺序

建议 JIT ABI 固定为：

```text
输入
caller-managed workspace
输出
```

对需要 host 检查的 workspace，先确认当前 TileLang 版本中 `workspace_idx` 的含义。
在 0721 使用的版本里，把 caller 已传入的 workspace 列入 `workspace_idx` 会让 JIT
重新分配私有 tensor，host 看到的原 buffer 将保持全零。

验收方法：

1. 给 workspace 填充 sentinel；
2. 执行 kernel；
3. 统计 sentinel 剩余数量；
4. 检查生成 `call(...)` 参数顺序；
5. 再比较 host 公式。

### 4.2 不要把未写入误判为合法零

全零 workspace 可能表示：

- kernel 正确输出了零；
- 分支没有执行；
- 参数顺序错误；
- JIT 使用了私有 workspace；
- MTE3 没有写回；
- launch core/task ownership 错误。

sentinel 比“是否为零”更可靠。

## 5. 精度定位的最短路径

按照依赖顺序检查：

```text
输入
 -> GatherKey/GatherKeyIndex/GatherKeyRope
 -> BMM1/BMM2
 -> PReduced
 -> SoftmaxI
 -> reluGrad
 -> dQuery/dKey
```

每个中间 tensor 至少输出：

```text
shape
NaN 数
Inf 数
finite_abs_max
host max_diff
host mean_diff
```

利用兄弟输出缩小范围：

- gather 正确：排除 sparse index ownership 和 gather DMA；
- BMM 正确：排除 Cube producer；
- PReduced/SoftmaxI 正确：排除 softmax 和归约；
- dWeight 正确：证明 `bmm2` 和 `diff=SoftmaxI-PReduced` 有限；
- Loss 正确：独立证明 P/SY 和 Loss reduction；
- 只有 reluGrad/dQ/dK 错：优先检查 weight expansion、mask、Cast、MTE3 和 UB
  ownership。

0721 的 `32x128` 实验中，dWeight 和 Loss 正确，而 dQ/dK 大误差，最终定位到
row-expand 权重映射，而不是 C2。

## 6. 手工 event 的 ownership 方法

### 6.1 event 不是“同步一下”，而是资源所有权

常见资源状态：

| 资源状态 | event 方向 |
| --- | --- |
| L1 slot 可被 GM refill | `MTE1_MTE2` |
| L1 数据可被 L0 load | `MTE2_MTE1` |
| L0A/L0B slot 可被 MTE1 覆盖 | `M_MTE1` |
| L0A/L0B 数据可被 MMA 使用 | `MTE1_M` |
| L0C 可被下一次 MMA 写入 | `FIX_M` |
| L0C 结果可被 FIX 使用 | `M_FIX` |
| UB 可被 MTE2 覆盖 | `V_MTE2` 或 `MTE3_MTE2`，取决于最后消费者 |
| UB Vector 结果可被 MTE3 写回 | `V_MTE3` |

对每个复用槽必须回答：

1. 谁是下一次 writer；
2. 谁是上一轮最后 reader；
3. 第一轮 free token 在哪里 seed；
4. 每轮 set/wait 是否一一匹配；
5. 退出 scope 前是否 drain；
6. event ID 是否在其他 phase 复用前已回收。

### 6.2 RAW 和 WAR 必须分别建模

0721 曾出现：

```cpp
bmm2_fp32  offset = 20992
rg_half    offset = 20992
```

这类地址复用可以合法，但下一轮 MTE2 writer 必须等待上一轮 MTE3 reader 完成：

```text
MTE3_MTE2
```

仅有 `MTE3_V` 和 `V_MTE2` 不足以证明 MTE2 晚于 MTE3。event 必须到达真正的
下一 writer。

### 6.3 单槽模板

```python
T.set_flag("v", "mte2", v_done)
T.set_flag("mte3", "mte2", store_done)

for tile in T.serial(tile_count):
    T.wait_flag("v", "mte2", v_done)
    T.wait_flag("mte3", "mte2", store_done)
    T.copy(gm_src[tile], fp32_ub)
    T.set_flag("mte2", "v", load_ready)
    T.wait_flag("mte2", "v", load_ready)

    # Vector work and Cast.
    T.set_flag("v", "mte3", output_ready)
    T.wait_flag("v", "mte3", output_ready)
    T.copy(fp16_ub, gm_dst[tile])
    T.set_flag("mte3", "mte2", store_done)

    # 所有 V 消费完成后再允许下一轮 refill。
    T.set_flag("v", "mte2", v_done)

T.wait_flag("v", "mte2", v_done)
T.wait_flag("mte3", "mte2", store_done)
```

### 6.4 死锁的典型标志

```text
507014
AICore execution times out
长时间卡住后 synchronize 报错
Cube 和 Vector 同时等待
```

第一检查项不是数学，而是生成 C++ 中是否存在：

```text
WaitFlag(X)
...没有任何可达 SetFlag(X)...
WaitFlag(X)
```

0721 的 P/SY 外层缓存实验正是因为外层消费了初始 `V_MTE2`，进入内层前没有
重新发布 token，首个 tile 永久等待。

## 7. 内存与 intrinsic 审计

### 7.1 审计物理 offset，而不是只看逻辑 shape

对每个 buffer 记录：

```text
dtype
元素数
字节数
物理 offset
对齐
生命周期
可能重叠的其他 buffer
最后 reader
下一 writer
```

逻辑 buffer 不重叠不代表物理 offset 不重叠；逻辑 ping/pong 也不保证生成两个物理
槽。

### 7.2 L0 必须使用物理精确 shape

不要分配一个大 L0A/L0B/L0C，再用切片模拟多个不同矩阵。当前 codegen 可能生成：

```cpp
copy_l1_to_l0a<srcM, srcN>(dst, src, parentM, parentN)
```

即模板参数看起来正确，目标 descriptor 仍携带父 shape。后果包括：

- MTE1 layout 错误；
- MMA 读取错误；
- FIX template 与 L0C 不匹配；
- illegal instruction；
- L0C read/write conflict。

每种 MMA 物理 shape 应使用独立的精确 allocation，再由生命周期规划复用地址。

### 7.3 immediate 宽度必须检查

若编译器警告：

```text
implicit conversion from int to uint8_t changes value from 256 to 0
```

这不是无害 warning，而是 Vector repeat/stride 已经失效。必须重新切 tile，不能
继续运行。

### 7.4 row-expand 的真实边界

在当前后端中，FP32 每 64 元素是一个 Vector repeat。row-expand helper 若每个
repeat 都推进 scalar source，则只对每行 64 元素正确。

扩大到 128 时，必须在生成 C++ 中确认同一 scalar 是否保持两个 repeat。不能只
因为 Python shape 是 `(rows, 128)` 就认为广播正确。

### 7.5 动态 UB scalar

`weight_ub[index]` 可能 lowering 为：

```cpp
weight_ub.GetValue(index)
```

如果该标量由 Vector 指令读取，地址通常需要 32 字节对齐。FP32 动态索引 1、2、3
只偏移 4、8、12 字节，会触发未对齐异常。需要后端寄存器 scalar lowering，或重构
数据布局；加 barrier 不能修复地址对齐。

## 8. 正确使用 Compare/Select、Exp 和归约

### 8.1 Compare/Select

官方 ReLUGrad：

```text
CompareScalar(relu_output > 0)
Select(value, mask, value, 0)
```

TileLang 中建议使用压缩 `uint8` mask：

```python
mask = T.alloc_ub(element_count // 8, "uint8")
T.tile.compare(mask, input, 0.0, "GT")
T.pipe_barrier("v")
T.tile.select(out, mask, value, 0.0, "VSEL_TENSOR_SCALAR_MODE")
```

但性能取决于 tile 粒度。官方 `4x2048` 只发射少量 Compare/Select；0721 的
`32x64` 会发射 32 组，bitmask 启动和 barrier 成本难以摊薄。

### 8.2 指数近似

```text
mask ~= 1 - exp(-10000 * relu_output)
```

它不完全等价于 `relu_output > 0`，但连续 FP32 Vector 指令可能比频繁的小块
Compare/Select 更快。是否采用必须在相同 core 数、相同 tile、相同同步下 A/B，
并明确精度需求。

### 8.3 reduction

- 输出 shape 必须符合当前后端 fast path，而不只是数学语义；
- 所有可能被物理 intrinsic 读取的 padding lane 必须初始化；
- 不随意改变已经验证的 reduction 顺序；
- scalar partial 写 GM 时使用 32 字节对齐槽；
- 避免为了 reshape 添加 `Muls(..., 1.0)`，可直接在二维 reduce 输入中计算。

## 9. 性能优化手段

### 9.1 先画 profile 时间线

将命名 kernel 的 profile 分为：

```text
V0
C1-P
C1-SY
V1-P
V1-SY
V1 backward
C2
V2
```

同时记录 AIC/AIV/MTE2/MTE1/M/FIX/MTE3 的 WAIT_FLAG。优化最长关键路径，而不是
指令最多的函数。

### 9.2 常驻不变输入

适合常驻 L1/L0 的数据：

- 跨 K tile 不变；
- 体积可控；
- 不阻塞其他 phase 的物理容量；
- preload 可与其他 pipe 重叠。

0721 保留了 Query/QueryRope L1 cache 和 C1-SY QueryIndex L0A cache。单纯减少
copy 次数不保证加速；若 preload 形成关键路径前导，可能持平或变慢。

### 9.3 原地复用

高价值模式：

- 生命周期结束后复用同一 UB；
- 将下一阶段结果直接写入 reduce 输入；
- 三操作数 Mul 直接写最终目标；
- 原地保存 Exp，避免下一遍重复计算。

任何地址复用都要补齐最后 reader 到下一 writer 的 WAR ownership。

### 9.4 软件预取

只有满足以下条件才值得：

1. 下一 tile copy 能在当前 MMA/Vector/FIX 期间真实发射；
2. 没有新增更长的 event 前导；
3. buffer 容量和 ownership 独立；
4. profile 显示搬运被覆盖，而不是只是源码顺序提前。

0721 的 C1 Key 单槽预取有效；C1-SY/C2 单槽预取因控制成本和单 L0 限制没有稳定
收益。

### 9.5 core 数调优

不要固定使用硬件全部 core。区分：

- 主任务数；
- V2 clear/scatter/finalize 批次数；
- 每个物理 core 的两个 AIV；
- SyncAll 参与者；
- atomic 竞争；
- 尾部分支。

0721 当前模型：

```text
v2_batches = max(ceil(K/16), ceil(B*S2/16))
v2_core_need = ceil(v2_batches / (2*16))
launch_core_num = min(hw, max(B*S1, v2_core_need, 1))
```

当前形状 24 核约 `0.483 ms`，8 核约 `0.461 ms`，4 核约 `0.455 ms`。缩核的收益
来自更少的全核同步、atomic 竞争和尾部条件，而不是主 Cube 并行度。

### 9.6 接受标准

一个性能补丁只有同时满足以下条件才保留：

- 多轮精度稳定，无 NaN/Inf 和非确定性漂移；
- 生成 C++ 与预期物理布局和 event 协议一致；
- 重复中位数改善超过噪声；
- profile 关键路径确实缩短；
- 没有增加 kernel 数或额外大规模 GM 往返；
- 没有删除已证明必要的 ownership。

## 10. 已证伪或高风险的优化手段

### 10.1 只扩大 tile

仅将 `32x64` 改成 `32x128` 或 `4x2048` 不代表 intrinsic 能正确处理：

- row-expand 可能映射错误；
- repeat/stride 可能溢出；
- reduction fast path 可能不支持；
- UB 峰值和别名关系会变化；
- MTE3/MTE2 WAR 可能重新暴露。

### 10.2 逻辑 ping-pong 等于物理 ping-pong

错误。必须从生成 C++ 证明：

- 两个 slot offset 不同；
- 每个 slot 物理 extent 足够；
- 每槽 event 独立 seed、循环、drain；
- 不与其他活跃 buffer 重叠。

即使 offset 正确，也可能缺少硬件内部 L0C ownership。C1-P 双缓冲实验在 offset、
shape 和 PIPE_M barrier 均正确时仍报告 `L0C read/write conflict`。

### 10.3 为减少 SyncAll 提前 scatter

clear、scatter、finalize 之间存在真实全局依赖。没有完整 ownership 模型时收敛
SyncAll 会产生 NaN、大误差或未初始化 accumulator。减少 barrier 之前先证明没有
跨 core RAW/WAR。

### 10.4 增加 kernel

对亚毫秒算子，额外 launch 和 GM round-trip 往往高于微优化收益。除非能删除大段
关键路径或复用已有 framework kernel，否则保持单 kernel。

### 10.5 用一次最小值判断优化

NPU 频率、队列状态和系统负载会造成明显波动。至少记录：

```text
多轮 avg
多轮 median
多轮 min
同轮官方时间
TileLang 绝对时间
```

不要只看加速比。官方本身变慢时，加速比可能变好，而 TileLang 没有任何改善。

## 11. 故障现象速查

| 现象 | 优先检查 |
| --- | --- |
| workspace 全零 | ABI、workspace_idx、call 参数、launch 分支、sentinel |
| gather 正确，P/SY 错 | BMM layout、softmax、reduce shape |
| dWeight/Loss 正确，dQ/dK 错 | reluGrad weight/mask/Cast/MTE3/ownership |
| NaN 数每次不同 | race、未初始化、tail、UB overlap |
| 507014 timeout | event token 缺失、cross-core flag 未发布、循环 drain |
| 507015 AICore exception | 生成指令地址/shape/descriptor/event ownership |
| L0C read/write conflict | L0C offset、MMA/FIX shape、FIX_M/M_FIX、硬件槽状态 |
| `256 -> uint8_t 0` warning | Vector repeat/stride immediate 溢出 |
| 性能下降但 copy 变少 | preload 前导、event 数、关键路径、GM/atomic 竞争 |
| 加速比变好但绝对时间不变 | 官方计时波动 |

## 12. 推荐开发流程

### 修改前

1. 保存精度正确源码和生成 C++。
2. 记录 shape、dtype、core 数、CANN 和 TileLang 版本。
3. 画 buffer ownership 表。
4. 标出当前 profile 最长关键路径。
5. 明确本次只改变哪个架构变量。

### 本地静态检查

1. Python AST 或等价语法检查；
2. 未定义变量和重复 allocation 名称检查；
3. `git diff --check`；
4. buffer 字节容量和对齐计算；
5. event seed/set/wait/drain 计数；
6. kernel launch core 和 task ownership 计算。

### 远端编译后

1. 保存精确生成 C++；
2. 检查 `GetWithOffset`；
3. 检查 intrinsic extent、repeat、stride、transpose；
4. 检查 event 链和 SyncAll；
5. 检查 `<<<core_num>>>`；
6. 确认没有 compiler warning 被忽略。

### 硬件验证

1. 先跑 stage debug 和 sentinel；
2. 重复精度测试，排除非确定性；
3. 再跑 warmup/bench；
4. 最后 profile 命名 kernel；
5. 只有关键路径缩短且中位数改善才保留。

## 13. 判断“源码优化是否到头”

当剩余差距主要来自以下能力时，应停止继续堆叠源码 event：

- 物理精确 L0 双缓冲；
- phase-scoped local-memory reuse；
- 安全寄存器 scalar；
- 大 tile row-expand lowering；
- 经过硬件验证的 MMA/FIX service；
- 动态批量 DMA descriptor。

这些属于 TileLang-Ascend 后端能力，不是再增加一个 `PipeBarrier` 或逻辑 buffer 就能
解决。此时正确路线是制作最小 reproducer、修改 intrinsic/codegen、验证生成 C++，
再回到业务 kernel。

## 14. 最终经验

1. 精度问题优先按 workspace 依赖链定位，不要从最终 dQ/dK 反推整个 kernel。
2. 事件是所有权协议，不是装饰性的同步语句。
3. 物理 offset、shape 和 descriptor 比 Python slice 更可信。
4. 减少 copy 次数不等于缩短关键路径。
5. 大 tile、ping-pong 和多核都必须先证明后端能正确表达。
6. 任何 compiler warning、超时或非确定性 NaN 都应视为结构错误，而不是精度容差。
7. 官方优化往往依赖整套 service；只移植其中一个局部顺序通常没有同等收益。
8. 亚毫秒 kernel 应用重复中位数和 profile 做决策，不能凭单次结果。
9. 在当前后端能力边界内，简洁、单槽、可审计的实现通常优于复杂但不可证明的
   ping-pong。
10. 若目标性能需要约 100 us 的结构性改善，应修改后端，而不是继续寻找 1 us 的
    源码微优化。

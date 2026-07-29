# TileLang-Ascend 后端能力需求

## 1. 目的与判定口径

本文从手工异步 Attention 类 kernel 的共性需求出发，定义 TileLang-Ascend 当前缺失或不完整的能力。0721 是验证载体，但需求不绑定固定 batch、序列长度、head 数、tile shape、局部变量名或单一问题用例。

一项能力只有同时满足以下条件，才能标记为“已具备”：

1. **源码语义明确**：可以表达 ownership、shape、stride、尾块和同步关系。
2. **lowering 正确**：生成 AscendC 保留相同的区域与依赖语义。
3. **硬件验证稳定**：最小设备测试重复运行无越界、死锁、非有限值和随机漂移。
4. **结果可审计**：编译报告或生成代码能解释物理地址、指令参数和 event 生命周期。

仅有 API，或只在少量特定 shape 上可用，均记为“部分具备”。

## 2. 优先级与能力总表

- **P0**：影响正确性或阻止稳定异步流水。
- **P1**：可由保守路径保证正确性，但造成显著额外搬运、计算或同步。
- **P2**：主要降低维护、诊断和性能调优成本。

| 编号 | 优先级 | 能力 | 当前状态 | 核心验收标准 |
| --- | --- | --- | --- | --- |
| R1 | P0 | 异步 pipe-aware 本地内存生命周期与 no-alias slot | 缺失 | producer 到最后 consumer 完成前，物理地址不得被 planner 复用 |
| R2 | P0 | generation-safe 跨核 event 与 task gate | 部分具备 | event ID 跨 task/slot 重用时不会消费旧 generation |
| R3 | P0 | 多维 strided BufferRegion lowering | 部分具备 | 子区域保留父 stride，不被错误线性化 |
| R4 | P0 | 通用 Compare/Select/Reduction shape 与 stride | 部分具备 | 合法 tile 自动拆分硬件 repeat，参数不截断 |
| R5 | P1 | 全路径动态尾块 | 部分具备 | DMA、MMA、Vector、reduce、scatter 使用一致的有效区间 |
| R6 | P1 | Indexed gather/scatter 与批量 DMA descriptor | 缺失 | 非连续索引可批量发射，重复索引语义明确 |
| R7 | P0 | 短 GM region、标量 DMA 与对齐边界语义 | 部分具备 | 小区域不越界、不误读，padding 行为有正式契约 |
| R8 | P1 | 跨 repeat 行广播与动态 UB scalar 寄存器提升 | 缺失 | 一个运行时 scalar 可覆盖完整逻辑行 |
| R9 | P0 | phase-scoped L0 slot 与 MMA/FIX ownership | 部分具备 | L0 ping/pong 无读写冲突，phase 能正确 drain |
| R10 | P1 | specialization-aware 宏收集和静态分支裁剪 | 缺失 | 不校验已删除分支的 BufferRegion |
| R11 | P2 | 选择性软件流水多版本化 | 缺失 | 仅环化指定 scratch，不复制全部本地 allocation |
| R12 | P2 | 物理布局、指令和事件审计信息 | 缺失 | 可追踪逻辑 buffer 到物理地址和 event generation |
| R13 | P1 | Frontend 受限语法的明确支持或诊断 | 部分具备 | 不支持语法在 frontend 阶段给出准确错误 |

## 3. 详细能力要求

### R1. 异步 pipe-aware 本地内存生命周期与 no-alias slot

**抽象场景**

本地 buffer 的计算已经结束，但异步 DMA 仍在读取它；下一轮同时申请其他逻辑 buffer。源码词法生命周期结束不等于物理数据已经归还。

**所需语义**

- buffer 生命周期延伸到指定 pipe/event 的完成点。
- 用户可声明两个 buffer 不得 alias，或显式绑定到固定 ping/pong slot。
- planner 复用地址前必须证明所有异步 consumer 已完成。

**最低实现要求**

提供等价于 `async_live_until(buffer, event)`、`no_alias(a, b)`、`alloc_slot(group, slot)` 的表达能力；API 名称可不同。容量不足或 ownership 冲突必须编译失败，不得静默 coalesce。

**验收**

- 最小 kernel 连续发起 `UB -> GM`，等待前写入下一 buffer；启用约束后重复运行稳定。
- 覆盖 UB、L1、L0A、L0B 和 L0C。
- 生成报告列出每个 slot 的 base offset、size、alignment 和最后异步 consumer。

### R2. Generation-safe 跨核 event 与 task gate

**抽象场景**

producer 和 consumer 以环形 task slot 运行，同一 event ID 跨轮次复用。consumer 必须等待当前 task generation，不能消费上一轮遗留 flag。

**所需语义**

- event 至少由 `(channel, slot, generation)` 唯一标识。
- 一对一、一对多和多对一的 set/wait 消费规则明确。
- 初始化、稳态、核数不足和尾部 drain 使用同一协议。
- 高层 intrinsic 不得隐式占用用户 event；若占用必须公开 ownership。

**最低实现要求**

编译器管理 generation，或提供不会混淆旧 flag 的 token/barrier 对象。重复 set、缺失 wait、未 drain 和 slot 覆盖应有静态诊断。

**验收**

producer/consumer 使用不同延迟，覆盖不足一轮、整轮和多轮 wrap-around；改变 task 数与 core 数后重复运行稳定。生成代码中每个 generation 的 set/wait 可配对。

### R3. 多维 strided BufferRegion lowering

**抽象场景**

对二维或更高维 parent buffer 的列块、行块或带 stride view 执行 DMA、Vector 或 reduction。逻辑矩形区域在物理内存中不一定连续。

**所需语义**

- slice 保留 parent shape、element stride、byte stride 和 base offset。
- intrinsic 不支持该 stride 时，自动 pack/unpack 或编译期拒绝。
- 不得把多行列切片降级为单个线性连续区间。

**最低实现要求**

BufferRegion 显式携带 strides，copy、Vector、Compare/Select 和 reduction lowering 统一消费；编译期可查询区域是否连续以及是否需要 pack。

**验收**

对 `[R, 2C]` parent 的左右半列执行不同操作，覆盖 `R > 1`、非单位行 stride 和非零 offset。结果与逐元素参考一致，相邻区域不互相污染，生成指令 stride 与 parent layout 一致。

### R4. 通用 Compare/Select/Reduction shape 与 stride

**抽象场景**

对不同高宽、strided view 和跨多个 vector repeat 的 tile 执行条件选择或按轴归约。底层 repeat、mask 或 stride 字段可能不足以直接编码逻辑 shape。

**所需语义**

- Compare mask layout 与 Select 消费 layout 有正式契约。
- Select 支持 tensor-tensor 和 tensor-scalar，保留 NaN、正负零和无穷语义。
- reduction 明确 axis、累加 dtype、输出 shape 和 tail predicate。
- 底层字段超限时自动拆分，不能发生 `uint8_t` 等窄字段静默截断。

**最低实现要求**

lowering 按硬件上限切分 repeat，并正确推进 data/mask offset。fast path 不支持时选择通用正确路径，或报告明确 shape/stride 错误。

**验收**

在底层 repeat 边界前后测试连续和 strided tile，覆盖 Compare/Select、axis-0/axis-1 reduction、有限值、NaN、正负无穷和临界零值。生成代码中的窄整数参数不得溢出。

### R5. 全路径动态尾块

**抽象场景**

编译 shape 按 tile 对齐，但每个 task 的真实有效长度不同；最后一个 tile 可能部分有效，也可能完整无效。

**所需语义**

- 同一有效长度或 predicate 贯穿 GM/UB/L1/L0 DMA、MMA、Vector、softmax、reduction 和 scatter。
- 无效输入不越界读取，无效输出不写回。
- softmax max 的无效值等价于负无穷，sum 的无效值等价于零。
- 完整块无逐元素动态分支；完整无效块可跳过昂贵计算。

**最低实现要求**

- `copy` 支持实际行列 extent 与 padding 值。
- `mma` 支持实际 M/N/K，并明确中间累加和最后 FIX 的 phase 语义。
- Vector/reduction/scatter 接受同源 predicate 或有效 extent。
- 运行时 extent 超过静态上界时必须报错。

**验收**

每种 tile 覆盖有效长度 0、1、边界前一项、恰好边界和边界后一项。用非法哨兵填充 padding，证明无效值不会污染输出。生成代码应显示完整块走无 mask 快路径，尾块只处理实际 extent。

### R6. Indexed gather/scatter 与批量 DMA descriptor

**抽象场景**

按运行时索引从不连续 GM 地址读取固定宽度行，或把本地行写到不连续目标。索引可能无序、重复或无效。

**所需语义**

- 批量描述多个源/目标地址和每行 extent，减少逐项 DMA 发射。
- 明确定义无效索引、重复索引、atomic sum 和确定性模式。
- 索引输入与数据输入可具有不同 stride。

**最低实现要求**

提供 indexed copy 或运行时 DMA descriptor 数组；scatter-reduce 至少支持 add。descriptor 的存储 scope、对齐和最大批量可查询。

**验收**

覆盖连续、随机、重复、逆序和无效索引。结果与逐项参考一致；确定性模式下重复索引结果稳定。timeline 必须证明接口没有被重新展开为等量标量 DMA。

### R7. 短 GM region、标量 DMA 与对齐边界语义

**抽象场景**

GM 与本地内存之间搬运短于硬件 block 的标量或小区域，且区域可能位于 allocation 尾部，不能合法多读相邻数据。

**所需语义**

- 精确定义 byte extent、最小对齐、padding 和越界规则。
- padding 不得导致 GM allocation 外多读，也不得污染目标声明范围外数据。
- 不满足硬件要求时自动使用安全 lowering 或编译期拒绝。

**最低实现要求**

`copy`/`DataCopyPad` 分别表达 source extent、destination extent 和 pad value；标量 GM load/store 有独立可靠 lowering，不伪装成不安全 block DMA。

**验收**

在 allocation 末端测试从 1 个元素到一个 block 的全部长度，前后放置哨兵，覆盖 GM->UB、UB->GM 和 reduction partial。

### R8. 跨 repeat 行广播与动态 UB scalar 寄存器提升

**抽象场景**

每行有一个运行时 scalar，需要广播到长行并参与多个 vector repeat；scalar 来源可能是 UB 或 GM。

**所需语义**

- 一个逻辑行在所有 repeat 中使用同一 scalar。
- 动态本地 scalar 安全提升到寄存器，或通过对齐 broadcast 展开。
- 行 stride、repeat stride 和 scalar stride 可分别表达。

**最低实现要求**

提供 row-wise broadcast/expand intrinsic，自动按硬件 repeat 上限分段；禁止生成非对齐 `GetValue` 或等价 UB 标量访问。

**验收**

行宽跨越一个和多个 repeat，每行使用不同 scalar，覆盖连续 parent、strided view 和动态行数。生成代码中每段 repeat 的 scalar source 必须属于同一逻辑行。

### R9. Phase-scoped L0 slot 与 MMA/FIX ownership

**抽象场景**

矩阵乘包含多个 K 累加块和多个输出块，需要 MTE1、MMA 与 FIX 分别处理相邻 phase，并在 ping/pong L0 slot 上重叠。

**所需语义**

- L0A/L0B/L0C slot 的 owner、可写、可读和归还时刻明确。
- 中间 K block 保留 L0C 累加，最后 K block 才交给 FIX。
- phase 切换和尾部 drain 不依赖未公开的内部 event。

**最低实现要求**

提供 phase/token 化的 MMA/FIX service，或允许用户绑定 L0 slot 和 event。`copy(real_m, real_n, real_k)`、`mma(actual_m, actual_n, actual_k)` 与 unit flag 组合必须有正式契约，编译器静态检查 L0C 冲突。

**验收**

覆盖单/多 K block 和单/多输出，验证首块清零、中间累加、末块 FIX。ping/pong 与串行单槽结果一致，无 L0C 冲突，timeline 可观察 MTE1、MMA、FIX 重叠。

### R10. Specialization-aware 宏收集和静态分支裁剪

**抽象场景**

host 根据静态属性编译多个 variant。某个 variant 已删除动态尾块或可选输入，但宏收集仍检查不可达分支中的 BufferRegion。

**所需语义**

在 BufferRegion 收集和 rank 校验前完成常量传播与死分支删除。未发射分支不得要求占位输入，也不得影响函数签名、本地内存规划或 cache key；动态分支仍完整校验。

**验收**

同一宏的静态分支引用不同 rank 的可选 buffer，各 specialization 只要求实际分支参数，生成函数不包含已删除分支资源。

### R11. 选择性软件流水多版本化

**抽象场景**

循环体同时包含可环化输入 tile、不可复制的大 scratch、跨迭代 accumulator 和异步输出。全部 allocation 统一乘 `num_stages` 会导致内存超限或错误 ownership。

**所需语义**

用户可区分 `stage_private`、`loop_carried`、`single_instance` 和 `async_output`。自动流水只复制 stage-private buffer，并生成正确 prologue、steady state 和 epilogue。

**验收**

编译报告列出每个 buffer 的版本数和理由；改变 stage 数不复制 single-instance accumulator；结果与串行版本一致，本地内存用量符合报告。

### R12. 物理布局、指令和事件审计信息

**抽象场景**

手工流水出现偶发精度错误或硬件异常，需要从逻辑 IR 追踪到物理地址、实际 intrinsic 参数和事件配对。

**所需语义与最低输出**

- 每个逻辑 buffer：scope、base offset、extent、alignment、alias group 和 lifetime。
- 每条 DMA/Vector/MMA/FIX：实际 shape、repeat、stride、mask 和 unit flag。
- 每个 event：producer、consumer、slot、generation 和 drain 状态。
- 提供机器可读报告，并在生成 C++ 中保留稳定 source map。
- 地址越界、窄字段截断、未配对 event 和异步 lifetime overlap 必须给出 warning/error。

**验收**

故意构造 alias、stride 和 event 错误，报告应同时定位 TileLang 源码和生成 C++；正常 kernel 可自动证明所有异步读取完成前无地址复用。

### R13. Frontend 受限语法的正式支持或明确诊断

**抽象场景**

kernel 作者使用普通 Python 条件表达式、动态索引或循环形式，但 frontend 只支持 TIR 子集。当前错误可能延迟到 BufferRegion 或 lowering，位置与根因不一致。

**所需语义**

- 正式公布支持的表达式、循环、条件和索引子集。
- 对 Python 三元表达式等不支持语法，在 frontend 阶段给出稳定、可操作的错误。
- 错误不得伪装成无关的 loop、shape 或 BufferRegion InternalError。

**验收**

每类不支持语法都有负向编译测试，检查错误类型、源码位置和替代写法。若未来支持三元表达式，只校验实际可达分支的 BufferRegion。

## 4. 已有能力的边界

`T.copy` 已暴露部分实际 extent 参数，`T.mma` 已暴露 `k_actual`、`n_actual` 和 `unit_flag`。这些接口可以描述局部尾块和 L0C 累加阶段，但不能单独构成稳定的异步 Cube service。

完整能力仍依赖：

- R1 的物理 slot 与异步 lifetime；
- R2 的 event generation；
- R5 的跨阶段统一尾块 predicate；
- R9 的 MMA/FIX ownership 和 phase drain。

因此，不能仅凭 `real_k/real_n` 参数存在就认定动态尾块和双缓冲流水已经完成。

## 5. 推荐实施顺序

1. **先保正确性**：R1、R2、R3、R4、R7。它们直接对应偶发污染、错误区域和硬件异常。
2. **再统一尾块**：R5，并与现有 copy/MMA extent 能力做端到端契约测试。
3. **解锁主要性能路径**：R9、R8、R6，分别解决 Cube 重叠、Vector 大 tile 和稀疏搬运。
4. **降低维护成本**：R10、R11、R12、R13，减少占位参数、手工流水和人工审计。

每项能力应先通过独立最小 kernel，再接入复杂算子；接入时一次只替换一层，并保留可对照的串行正确版本。

## 6. 0721 接入验收

1. full-valid 和 tail-protected 均通过完整泛化精度集，重复运行无 NaN 和随机漂移。
2. 完整块、部分尾块和完整无效块分别有独立诊断结果。
3. 生成 C++ 中 region stride、物理 offset、repeat 参数和 event generation 与源码意图一致。
4. 诊断开关关闭后 warmup，再比较多次中位数；输入准备、JIT 和官方算子时间单独记录。
5. 新能力若不能同时证明正确性和收益，应保留现有保守路径，不直接替换精度基线。

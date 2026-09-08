# B=1 / T=3500 / Nk=16 / Nv=32 / Dk=Dv=128 / BF16 state / gate：独立源码分析

本报告由只读源码分析子代理完成；未运行 NPU、未切换 custom 包、未修改算子源码。当前源码提交为 `6f599eb3a`，算术实现来自 `121832027`。CANN 原版对照使用本机 CANN 9.2.0 内置包实际附带的 arch35 源码，不把此前小维度 Streaming 实验当作本用例的性能证据。设备测量以父任务整理的 timing/msopprof 报告为准。

## 1. 本例准确命中的路径

- `UseStreamingExact()`：streamScore=`3500*128*128*7*256=102760448000`；cubeScore=`55*2*5832704=641597440`，前者为后者约160.16倍，因此走融合 Cube，不走 Streaming。
- chunk 数为55：54个完整64-token块，最后44个有效token；固定tile共3520个token，padding为20/3520=0.56818%。Dk/Dv均为128，无特征维padding。
- `B*Nv=32`，32个AIC核组各持有一个value head，一波完成；两个AIV分别处理32-token预处理切片和64个Dv状态行，两个AIV都承担有效计算。这与小Dv的ASC-024只有一半AIV参与的情形不同。
- `kvHead=nvId/2`：相邻两个value head共享一个Q/K head，但当前分别执行输入搬运及KKT/QKT计算。
- 命中 `PrepareStage1FullDimSliceVF` 特化、`InverseAIVPanel8DirectBrcVFImpl` 求逆，以及BF16 compact-zN状态路径。

源码证据：

- [静态分流](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:336)
- [AIC workId分配](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:837)
- [AIV分工及初态读入](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:1476)
- [完整128维特化入口](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:1300)

## 2. 性能收益的机制

### 2.1 三阶段完整串行调度改为按head持续递推

内置版先执行整个chunk group的Stage1，再执行Stage2，最后Stage3，每阶段结束做全核屏障和TPipe重置；阶段间结果经GM传递。当前实现将同一head的全部chunk留在一个mixed核组，下一chunk无状态依赖的前缀工作与当前chunk状态递推重叠。

本例长度3500小于公共tiling公式的`maxGroupLength=2*32*64=4096`，只有一个chunk group。不能把原版的全核屏障数量误写成每chunk三次或165次：源码为入口一次加三个阶段末尾；新版为入口和结束汇合。收益不仅是屏障指令本身，还包括三大阶段之间无法重叠和大量GM发布/消费依赖。

- [内置版三阶段及SyncAll](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_arch35.h:175)
- [新版AIC提前计算下一个前缀](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:773)
- [新版AIV准备下一前缀后更新当前状态](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:1434)

### 2.2 状态和中间量留在片上，并增加AIV有效并行度

内置Stage2每chunk让vector0从GM读入整个state、衰减后写回；AIC的两次状态矩阵乘再从GM读state，随后对GM状态累加更新。内置Stage3也仅vector0构造maskedQKT。新版两个AIV各拥有64个Dv行，state一直留在UB，供AIC的BF16状态位于L1；初态只在序列开始读入、末态只在结束写回。

新版L1有两个160KiB的chunk槽和32KiB状态，共352KiB。UB物理布局末尾已到248KiB，不能直接追加一个完整预处理槽。L0A/B双缓冲；L0C分别保留临时结果、Vnew和output，使矩阵乘结果可直接Fixpipe到UB/L1，output两项在FP32 L0C累加后只写一次GM。

- [原版BF16状态GM递推](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_stage2_arch35.h:182)
- [原版Stage3仅vector0工作](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_stage3_arch35.h:113)
- [新版片上布局](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:137)
- [新版BF16状态更新及发布](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:1449)

### 2.3 Cube算术基本保留，减少重复装载和调度

原版每完整chunk包括10次高层Matmul调用；新版9次MMAD调用，主要把KKT/QKT合成一次`[128,128]*[128,64]`矩阵乘，复用同一份L0B的K。两者对完整tile的主要MAC量相同，不能把约6倍性能收益解释成约6倍乘加数量减少。

新版每head每chunk固定5,832,704 MAC，整个case为10,265,559,040 MAC，按乘加两次操作计20.53111808 GFLOPs，未计Vector计算。尾chunk仍按固定tile算。`RunStatePair`的`kcd*S^T`、`qPrime*S^T`复用一份L0B状态；将S从L1搬到L0B后便可释放L1状态，但必须持有L0B锁直到两次矩阵乘结束。

- [KKT/QKT合并](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:680)
- [两次状态乘法复用L0B](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:644)
- [完整递推算术](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:778)

### 2.4 含gate情况下，消除重复Gamma矩阵计算

内置Stage1的两个AIV分别构造完整64×64的`exp(G_i-G_j)`矩阵，然后各自只使用对应的32行求逆。内置Stage3 vector0再次构造一份完整矩阵，用于maskedQKT。

新版两个AIV分别计算32×64，单次exp结果同时服务严格下三角KKT和含对角线QKT；完整chunk的pairwise gate矩阵Exp有效计算覆盖从约3×4096元素降为4096元素，并消除GM mask装载和独立Gamma中间数组。该3倍只描述这一部分的元素计算次数，不能当作整算子加速比。

- [内置GammaCompute](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_stage1_arch35.h:465)
- [内置Stage3再算gate矩阵](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_stage3_arch35.h:130)
- [新版共用exp](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/vf/chunk_gated_delta_rule_stage1_vf.h:348)

### 2.5 对角块求逆：相同FMA数量，减少加载、Gather、串行可见性屏障

下表是每个32×32对角逆块的**源码级操作调用数**。它们可能被编译器变换，不能直接当成PMU指令数或硬件字节数。

| 项目 | CANN内置原版 | 当前DirectBrc Panel8 |
|---|---:|---:|
| 前代FMA | 496 | 496 |
| 读取已求得inverse行 | 496 | 48 |
| Gather系数广播 | 496 | 0 |
| Gather索引Duplicate | 496 | 0 |
| 系数读取 | 31次整行Load | 496次B32直接广播 |
| inverse结果行Store | 32 | 32 |
| store→load LocalMemBar | 32 | 4 |

新版每8行共用48=0+8+16+24次旧inverse行读取，panel内部新求出的行常驻寄存器，旧inverse行加载减少90.32%；各个目标行的`j=0..i-1`累加顺序仍保留。系数读取模式同时发生变化，因此不能把90.32%写成总UB读带宽减少90.32%。

整个case共有3520=55×32×2个对角块，旧inverse行加载1,745,920→168,960次，LocalMemBar112,640→14,080次。原版每chunk构造identity，当前每head只初始化一次；identity标量SetValue由112,640→2,048次，减少55倍。新代码中的Pair、AoSoA、Gather实现均不在当前入口中。

- [内置逐行Gather前代](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/vf/chunk_gated_delta_rule_stage1_vf.h:30)
- [内置每chunk初始化identity](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_stage1_arch35.h:526)
- [当前DirectBrc前代](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/vf/chunk_gated_delta_rule_stage1_vf.h:949)
- [当前每head初始化identity](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:1502)

## 3. 工作区分配与GM逻辑流量分开计算

### 3.1 分配量

按公共tiling的P=2、stageOneParaNum=4、aiCoreNum=32、maxGroupLength=4096计算，BF16状态版原分阶段路径的interWorkspace为178.5MiB，stageWorkspace为10MiB，加16MiB系统区共204.5MiB。当前ASCEND950只申请16MiB系统区。该数值是由源码公式推导，原版实际运行的申请量应以其tiling/运行记录核对。

178.5MiB中包含按最大group长度预留的空间以及arch22兼容attnInter缓冲；内置arch35实际将跨chunk输出写入out，不能把这块预留attnInter当作本例必然发生的读写。工作区容量不是HBM流量，也不能直接用于带宽除法。

- [公共工作区公式](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:117)
- [当前仅申请系统区](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:255)
- [内置GM中间张量视图](/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/arch35/chunk_gated_delta_rule_arch35.h:130)

### 3.2 主要跨阶段中间量的等价逻辑读写量

下表单位为十进制MB（1MB=1,000,000B），忽略cacheline放大、cache命中和Matmul内部重读；Matmul的accum更新按一次读加一次写计算。它描述软件表达的等价逻辑读写，**不是实测HBM或GM总流量**。

令`P=32*3520*128*2=28.83584MB`为Stage1输出满tile的矩阵，`V=32*3500*128*2=28.672MB`为有效token矩阵；QKT按54个64×64与一个44×44尾块计算。

| 跨阶段张量 | 原版读写组成 | 原版逻辑MB | 当前跨阶段GM逻辑MB |
|---|---|---:|---:|
| kcd | Stage1写P，Stage2读V | 57.50784 | 0 |
| qPrime | Stage1写P，Stage2读V | 57.50784 | 0 |
| kg | Stage1写P，Stage2读V | 57.50784 | 0 |
| vInner/Vnew | Stage1写P，Stage2累加读写2V，state更新与Stage3各读V | 143.52384 | 0 |
| qkt | Stage1写、Stage3读 | 28.55936 | 0 |
| gCum | Stage1写、Stage3读、Stage2每chunk读取最后标量 | 0.90304 | 0 |
| 合计 | 上述六项 | 345.50976 | 0 |

这未包含Stage1内部Q/K连续化、KKT、attn、rawKcd、vBeta临时GM读写及mask，也未包含原始输入、输出和state，不能称为“原版总GM流量”。

另外，在相同等价逻辑口径下：

- out由Stage2写一次、Stage3累加读写一次：86.016MB→28.672MB，减少57.344MB。
- state按每chunk计AIV读、AIC两次读、AIV写、AIC累加读写，共6次state大小；55个chunk合计346.03008MB。新版仅初态读、终态写，共2.097152MB，逻辑减少343.932928MB。原版部分操作可能由原子累加及cache实现，不能把这些逻辑量直接解释为HBM事务。

新版本例各输入输出唯一tensor元素量合计约88.785152MB；考虑每value head的AIC/AIV分别读取Q/K、相邻value head共享Q/K但分别读取，以及两AIV重复读取g，源码访问范围合计约175.249152MB（未计seqlens微量访问、cacheline放大）。这也不是HBM流量；它说明新版并未做到所有输入只读一次。

## 4. 值得实测的后续优化

### 优先候选：批量预计算kg所需的gate decay

`PrepareStage1FullDimSliceVF`每个token将`G_last-G_i`广播到64个FP32 lane，再执行一次向量Exp。每AIV每完整chunk32次Exp实际上每次只求同一个标量的64份副本。

可在`CumSumExpExactVF`已经持有完整64-lane G的时刻，一次Sub+Exp计算全部`exp(G_last-G_i)`，存入现有gExp槽的空余256B；之后各token只广播相应预计算值。当前gExp槽从138KiB到139KiB，共1KiB，仅已有256B exp(G)，容得下额外256B decay；不必增加一个完整UB缓冲槽。

完整chunk每AIV的kg decay Exp调用可由32→1。本例实际总计112,000个有效token-head，对应该处112,000次向量Exp；预计算变为3520次，相差约31.82倍。这只针对该处Exp调用，不是总Vector工作量或总kernel时延的预测。其好处是保留相同FP32差值和exp、相同state递推及归约顺序，精度风险低于重新拆chunk或改近似算法，但仍必须实测attention与final state误差。

- [目前逐行广播后Exp的位置](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/vf/chunk_gated_delta_rule_stage1_vf.h:147)
- [可以预计算的寄存器前缀和位置](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/vf/chunk_gated_delta_rule_stage1_vf.h:83)
- [gExp槽布局](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h:170)

### 其次：检查求逆panel和VF生成代码

若msopprof确认Vector繁忙而Cube不饱和，可针对DirectBrc求逆测4/8/16行panel，或分析寄存器溢出、广播load依赖及指令发射。16行panel能把旧inverse行读取48→16、panel可见性屏障4→2，但增加常驻累加器/结果行数量及代码体积，未必更快。已有Pair/AoSoA版本也需要独立测量，不能因更宽lane布局而预设收益。

可同时核对`PrepareStage1FullDimSliceVF`内循环的`-beta*exp(G)`是否已由编译器提升到每token一次；源码当前对128维的两个64-lane repeat重复计算相同系数，人工提升是否有效以汇编与测量为准。

### 有条件再做：流水关键依赖重排

当前AIV完整准备下一chunk前缀（含求逆）后才消费当前chunk state增量。需由PipeTimeline判断AIC等待`PREFIX_CRITICAL_READY`还是`STATE_READY`更久，再研究前缀VF拆分或state更新插入点。不能盲目移动Wait/SetFlag：两个chunk共享state和部分UB，生命周期错误会破坏数据或死锁；单纯提前一处信号也不代表整条关键路径缩短。

### 对本例优先级较低的方向

- 尾块只有0.568%token padding，Dk/Dv无padding；尾块/小维tile专用化难以解释大的剩余空间。
- 32个head已填满32个AIC核组，两个AIV均有效；增加head分工或沿Dv拆分无法凭空再增加一倍有效核数。
- 两个value head共享Q/K，确有重复加载与KKT/QKT计算，但前缀QK仅占固定Cube MAC的17.98%。完美复用两head也只减少总Cube MAC约8.99%；若合并到一个核组导致有效AIC从32降至16，可能得不偿失。跨核共享还会引入发布/同步开销。
- 修改Streaming分界不会影响此例；把chunk64改为更小的递推chunk会改变BF16状态量化位置，不能当作无精度影响的简单调度调整。

## 5. 对profiling归因的约束

源码可以确认移除了哪些搬运、广播和串行依赖，但不能给这些优化点分配可相加的加速百分比。PIPE时间可重叠，Vector忙碌占比不等于有效FLOPs峰值利用率；单个wait_ratio不应直接解释为墙钟等待比例。原版和新版必须取同一用例、seed、g开关、状态dtype、执行方式的独立正式耗时；msopprof重放耗时不与msprof常规样本混合计算加速比。

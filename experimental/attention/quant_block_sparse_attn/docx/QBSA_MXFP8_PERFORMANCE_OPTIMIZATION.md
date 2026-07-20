# QBSA MXFP8 性能优化候选清单

## 1. 文档目的

本文依据 `/Users/kane/Documents/Codex/zyh/cann-skills-zyh` 中的 Ascend C 性能优化方法，对 QBSA MXFP8 kernel 做静态代码审计，并给出可以小步实施、逐项验证的优化建议。

审计范围：

- `op_kernel/arch35/quant_block_sparse_attn_mx_kernel.h`
- `op_kernel/arch35/quant_block_sparse_attn_mx_block_cube.h`
- `op_kernel/arch35/quant_block_sparse_attn_mx_block_vec.h`
- `op_kernel/arch35/common/util_regbase_mx.h`
- `op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h`
- MXFP8 tiling key、tiling data 和 host 分核逻辑

本文是静态审计结果，没有使用 `msProf` 或流水图，因此不承诺具体收益，也不直接判断当前一定属于 ScalarBound、MTE2Bound、CubeBound 或 VectorBound。每个候选点都应独立做 A/B 测试，确认有收益后再保留。

## 2. 当前实现基线

MXFP8 当前固定计算模板为 `M=128`、`S2Base=512`、`D=128`、`DV=128`。一个 logical S2 tile 被拆成两个 256-token 子块：

1. C1 分别计算两个 `K[256,128] * Q^T[128,128]`，输出两个 FP32 小 P。
2. V1 对两个小 P 做 scale、online softmax、update，并生成完整 `P[128,512]` 及 PScale。
3. C2 按两个 256-token K 循环计算 `P[128,512] * V[512,128]`。
4. V2 对跨 S2 tile 的结果做 update、归一化、cast 和写回。
5. 主流水使用三槽任务缓存，当前任务执行 C1/V1，前两个槽位的任务执行 C2/V2。

当前已经存在且应优先保留的优化：

- Q/QScale 只在当前 Q block 的首个有效 S2 tile 搬入，后续 S2 tile 复用 L1 数据。
- Q、K、V、L0A、L0B、L0C 已配置多 buffer；P 使用三 buffer 配合两级 preload。
- softmax 的 max、sum、exp 状态保留在三槽 UB buffer 中，没有落回 GM。
- mask 使用 `HAS_ATTEN` 编译期分支；无 mask kernel 不执行 mask 搬运。
- V1 先发起 mask 搬运，再等待 C1 结果，已有 MTE2 与 C1 的重叠意图。
- P 和 PScale 保留在 L1，直接供 C2 的 MX 随路反量化使用。
- VF 已使用 `FusedExpSub`，P 的 E4M3 cast 和 PScale 的 E8M0 cast 属于算法及 MX MMAD 输入要求，不能按通用“减少 cast”原则直接删除。

## 3. 优化依据

本次主要采用以下 CANN skills 原则：

- Scalar：减少 Load/Store、缩短变量生命周期、避免大结构体和结构体内动态索引阻断常量传播、把可确定参数转为 `constexpr` 或模板参数。
- MTE2：优先连续大粒度搬运，减少重复地址计算和小包搬运；只有在物理地址满足条件时才合并 PA 搬运。
- Pipeline：buffer 数量不等于已经产生重叠；必须在当前计算期间提前发出下一块搬运，双 buffer 才真正发挥作用。
- Vector：减少重复转换和中间落 UB，但不能破坏 MXFP8 的 scale 语义、online softmax 状态或精度路径。
- UB：状态常驻优先；bank conflict 和 padding 必须通过性能数据确认，不能凭布局猜测直接修改。
- 多核：先测各核任务数和耗时差异，再调整 tiling 分核，避免用单核微优化掩盖负载不均。

## 4. 小步优化候选

### P01 删除重复的 `sparseSeqLen` GM 标量读取

- 当前情况：`ProcessMainLoop` 已读取当前 Q block 的 `sparseLen`；`FillRunInfo` 又对同一位置执行一次 `sparseSeqLenGm.GetValue`，写入 `runInfo.actSparseLen`。
- 代码位置：`quant_block_sparse_attn_mx_kernel.h` 的 `ProcessMainLoop` 约 430 行，以及 `FillRunInfo` 约 322 行。
- 审计结果：MX 路径中 `actSparseLen` 只有写入，没有后续读取。
- 最小改法：删除第二次 GM 读取和 `MxRunInfo::actSparseLen`，直接使用外层已经取得的 `sparseLen`。
- 预期方向：减少每个 logical S2 task 的一次 Scalar GM load 和一次结构体 store。
- 风险：低。
- 验证：编译后运行 BS64/BS128、单 S2/多 S2、尾块、空 sparse 列表用例。

### P02 将 batch/head 不变量移出 S2 内循环

- 当前情况：外层已经得到 `bIdx` 和 `actualS1Size`，但 `FillRunInfo` 每个 512-token task 仍重复计算 `bIdx/n1Idx/n2Idx`，并重复读取 Q/KV actual sequence 和 `cuSeqlensQ`。
- 最小改法：以 `bn1Idx` 为粒度计算 `bIdx`、`n1Idx`、`n2Idx`、`queryTokenBase`、`actualS1Size`、`actualS2Size`，通过小型局部参数传给 `FillRunInfo`。
- 不建议：暂时不要新增另一个大型 context 结构体；优先传少量标量或使用小于 64B 的局部结构体。
- 预期方向：sparse_count 较大、同一 Q block 有多个 S2 task 时，减少除模、地址计算和 GM 标量读取。
- 风险：低到中，主要风险是 ragged batch 的 batch/head 映射传错。
- 验证：B=1/2/8 的等长和 ragged case，重点检查 GQA=5/16、actual sequence 不等长场景。

### P03 将 `GetSparseBlocksPerTask()` 提到主循环外

- 当前情况：该值仅依赖 `kvSparseBlockSize`，但在每个 Q block 内重新计算一次。
- 最小改法：在 `ProcessMainLoop` 进入 BN1/S1 循环前计算一次 `sparseBlocksPerTask`。
- 预期方向：消除重复除法、上下界判断和函数展开代码。
- 风险：低。
- 验证：BS64 应得到 8，BS128 应得到 4；两类 STC 精度和任务数量不变。

### P04 清理只写不读的 `MxRunInfo` 和 `MxConstInfo` 字段

- 当前只写不读的运行期字段：`gS1Idx`、`gIdx`、`sparseBase`、`actSparseLen`、`sparseBlockStartIdx`。
- 当前只初始化不消费的常量字段：`t1Size`、`realGSize`、`s1Size`、`s2Size`；`n1Size` 也可以用初始化局部变量替代。
- 最小改法：一次只删除一组已由全路径 `rg` 确认无读取的字段，同时删除对应赋值。
- 预期方向：减少结构体尺寸、初始化 store、栈占用和寄存器 spill 概率。
- 风险：低，但修改前应再次全局检索，防止新增分支已开始使用字段。
- 验证：全量 MX STC 编译和运行；同时记录编译器 Scalar Load/Store 或 spill 变化。

### P05 分阶段减少 `runInfo = {}` 的整结构清零

- 当前情况：`FillRunInfo` 对每个 task 执行 `runInfo = {}`。`MxRunInfo` 内含 5 组、每组最多 8 个元素的 sparse 数组，结构体明显大于 CANN Scalar 指南建议的 64B，整对象清零可能产生大量 store。
- 第一步：先完成 P04，缩小结构体。
- 第二步：列出本 task 必须重置的标量，只清零这些标量和 `[0, sparseBlockCount)` 的数组范围。
- 第三步：如 profiling 仍显示 Scalar spill，再考虑拆分结构体。
- 预期方向：减少 hot task 构造阶段的 Scalar store。
- 风险：中。漏清字段会造成跨环形槽位的脏状态，不能和其他大改同时提交。
- 验证：重点运行 BS64/BS128、负 sparse index、尾块、reverse/random、无效行、空 tensor，并连续重复执行排查偶现脏数据。

### P06 将 sparse 数组与热标量状态拆开，仅在 ScalarBound 证据充分时实施

- 当前情况：`MxRunInfo` 同时保存任务热标量和 5 组动态索引数组，并以三槽数组形式存在。CANN Scalar P1/P7 指出“大结构体 + 结构体内动态数组索引”可能破坏常量传播并引起 spill。
- 建议结构：小型 `MxTaskState` 保存 C1/V1/C2/V2 热标量，`MxSparseTileState` 单独保存 sparse block 数组。
- 实施条件：P01~P05 后仍然是 ScalarBound，且编译器/性能数据表明 `MxRunInfo` 有明显 Load/Store 或 spill。
- 预期方向：改善寄存器分配和别名分析。
- 风险：中到高，涉及 kernel、cube、vector 多处函数签名。
- 验证：作为独立提交，不与流水或搬运修改组合。

### P07 评估取消 task 内 sparse block 排序

- 当前情况：每个 task 最多对 4 个 BS128 block 或 8 个 BS64 block执行插入排序。
- 数学条件：若 K、KScale、V、VScale、mask 始终按同一 sparse 顺序拼接，则 softmax 和 `P*V` 对 KV 的一致置换不敏感，理论上不要求 token 升序。
- 最小步骤：先证明所有消费者均使用同一顺序；然后只去掉 `SortSparseBlocks`，不改其他 offset 逻辑。
- 备选：若必须排序，针对 4/8 个元素评估固定比较网络，避免动态 `while` 分支；但已排序输入上插入排序可能更便宜，必须 A/B。
- 预期方向：减少每个 task 的比较、移动和分支。
- 风险：中。mask、重复 block、负 index、reverse/random 稀疏列表必须完整回归。
- 验证：sequential/reverse/random/duplicate/negative sparse index，比较 OUT 和 LSE。

### P08 合并 K 与 KScale 的 sparse 遍历

- 当前情况：`CopyKeyToL1` 与 `CopyKeyScaleToL1` 对同一 sparse block 列表分别计算 block overlap、copyStart、copyRowNum，并分别构造 `Position/PAShape`。
- 最小改法：新增一个 `CopyKeyAndScaleToL1`，每个 block 只计算一次 overlap，再分别发出 K 和 KScale 搬运。
- 边界：只合并 Scalar 遍历，不把两个不同 GM tensor 强行合成一次 DMA。
- 预期方向：减少 BS64 最多 8 块场景下重复的 Scalar 地址计算和分支。
- 风险：中低。
- 验证：BS64/BS128、两个 256 subLoop、S2 tail、PA block table random/reverse。

### P09 缩短 `Position` 和 `PAShape` 的构造路径

- 当前情况：K、KScale、V、VScale 每个 sparse block 都从头填写 `Position` 和 `PAShape`，其中 headNum、headDim、blockSize、pageStride、rowStride 等字段在一次函数调用内不变。
- 最小改法：在 block 循环外构造不变量，循环内只更新 `s2Offset`、`copyRowNum` 和必要的目标 offset。
- 注意：如果把对象地址传给外部 API 导致 spill，可能抵消收益；应比较编译结果。
- 预期方向：减少 Scalar store 和对象构造开销。
- 风险：低到中。
- 验证：与 P08 分开做 A/B；确认每块 page/table offset 不变。

### P10 把 `RETURN_SOFTMAX_LSE` 下传为 MX 编译期模板参数

- 当前情况：`RETURN_SOFTMAX_LSE` 已进入 tiling key，但 kernel 入口使用 `(void)RETURN_SOFTMAX_LSE` 丢弃；MX vector 始终分配 `softmaxLseQueue_`，写回时再读取运行期 `isSoftmaxLseEnable`。
- 最小改法：把该布尔值传入 `QuantBlockSparseAttnMxKernel` 和 MX vector；`false` 分支用 `if constexpr` 裁掉 LSE queue 初始化、LSE 计算和 GM 写回。
- 预期方向：LSE=false 的 kernel 减少 UB 占用、运行期分支和尾部 Vector/MTE3 工作。
- 风险：中低，tiling key 已经区分 true/false，不需要新增 key 位。
- 验证：同一 case 分别运行 LSE=true/false；OUT 必须一致，false 不访问 LSE GM，true 精度不变。

### P11 复用 mask 的 `MTE2_V` event id

- 当前情况：`CopyAttenMaskToUb` 在每次确实发生 GM->UB mask 搬运时调用 `FetchEventID(MTE2_V)`。
- 最小改法：在 vector 初始化阶段分配或获取专用 event id，完成一次 set/wait 后循环复用。
- 前提：确认 event 生命周期覆盖 kernel，且同一 AIV 上不存在并发复用冲突。
- 预期方向：减少 mask 热路径中的事件查询 Scalar 开销。
- 风险：中。事件使用错误会产生偶现 NaN 或脏 mask，因此必须做重复执行稳定性测试。
- 验证：mask=3 的 sequential/reverse/random、部分 mask、无效行，连续运行至少 100 次；mask=0 不受影响。

### P12 让 V/VScale 双 buffer 真正形成 C2 的 MTE2 预取

- 当前情况：`l1VBuffers_` 已经是双 buffer，但每个 C2 K 循环仍按“取 buffer -> 搬 V/VScale -> 立即等待 -> MMAD”执行，没有明确在当前 MMAD 期间预取下一块。
- 小步方案：仅为 `kLoops==2` 写专用路径：先预取 V0；计算 V0 时向另一 buffer 发 V1/VScale1；最后计算 V1。`kLoops==1` 保持原代码。
- 预期方向：S2 有效长度大于 256 时，隐藏第二块 V/VScale 的 MTE2 延迟。
- 实施条件：流水图显示 C2 前存在明显 MTE2 wait，且 Cube 未完全覆盖该等待。
- 风险：中到高，涉及 event 顺序、tail padding、L0C 累加首末轮参数。
- 验证：S2=1~256、257~512 两组；BS64/BS128；尾块；OUT/LSE；检查无 event 死锁。

### P13 在 P12 稳定后评估 K/KScale 的 C1 预取

- 当前情况：K 使用四 buffer，但每个 subLoop 仍在 C1 前完成搬运并立即等待。
- 小步方案：只针对两个 256 subLoop 的主路径，预取 K0，计算 K0 时发 K1/KScale1，尾部保持原路径。
- 预期方向：隐藏第二个 C1 子块的 K/KScale 搬运。
- 风险：高于 P12，因为 Q 复用、C1 输出跨 AIC/AIV 同步和四 buffer 轮转同时参与。
- 验证：先做 trace，不和 P12 同时首次引入；重点检查两轮 C1 与两个 AIV 的同步。

### P14 为 KScale/VScale 小包搬运增加“可合并快路径”

- 当前情况：BS64 下单个 KScale/VScale page 约为 256B，BS128 下约为 512B，属于 CANN 搬运指南重点关注的小包；K/V data 本身约为 8KB/16KB，不是主要小包问题。
- 限制：PA block table 可把逻辑相邻块映射到不连续物理 page，不能无条件合并。
- 小步方案：先统计连续物理 page 比例。只有物理地址和 stride 满足 DMA 合并条件时，使用多 block/strided copy 快路径；否则保留当前逐 page 路径。
- 可先覆盖：`N2=1` 且物理 page 连续的简单场景，再逐步扩展。
- 预期方向：减少 scale 搬运指令数和小包启动开销。
- 风险：中到高，错误合并会直接破坏随路反量化 scale。
- 验证：sequential/reverse/random block table、N2=1/3/8、BS64/128，逐元素比较 scale 搬运结果和最终精度。

### P15 评估将 BS64/BS128 变为模板特化

- 当前情况：`kvSparseBlockSize` 运行期只支持 64/128，但 block 数、除法、group 映射和循环上限仍在运行期计算。
- 小步方案：仅新增 blocksize 模板位，分别编译 BS64/BS128；先让 `GetSparseBlocksPerTask`、`scaleBlockSize` 和循环上限常量化，不同时重写搬运逻辑。
- 预期方向：减少除法、动态循环和分支，并帮助编译器展开 4/8 block 路径。
- 代价：增加 tiling key 组合和 kernel 二进制体积。
- 风险：中到高；只有 ScalarBound 明显且两种 blocksize 都是重要场景时才值得做。
- 验证：检查新增 key 的 host/kernel 一致性、包内 JSON、两类 blocksize 的精度与编译大小。

### P16 优化 `vselrIndexes` 的 256 次 Scalar 初始化

- 当前情况：每个 AIV 初始化时通过两层循环执行 256 次 `SetValue`，生成固定 DN lane 重排表。
- 候选顺序：先确认小 case 的 Scalar 时间占比；再比较固定表搬运、Vector 生成、编译期可展开写法三种方案。
- 不建议：直接把表放 GM。这样可能只是把 Scalar store 换成一次额外 MTE2，对小 case 未必更快。
- 预期方向：降低短序列、小 sparse_count case 的启动开销。
- 风险：中低。
- 验证：表中第 `i*64+j` 个值仍为 `i+(j<<2)`；比较小 case kernel time，而不只看大 case。

### P17 根据 sparse task 数重新评估多核负载均衡

- 当前情况：MX 沿用 FP8 metadata 的 BN1/S1 分核结果，但单个 Q block 的工作量与 `sparseLen`、BS64/BS128、有效尾块数量相关。
- 第一步：记录每个 AIC 的 logical S2 task 数和总有效 token 数，确认是否存在长尾核。
- 第二步：若负载明显不均，在 host 侧用 `ceil(sparseLen / sparseBlocksPerTask)` 作为 Q block 权重重新分段，尽量不改变 kernel 的 metadata 格式。
- 预期方向：ragged batch、随机 sparse_count、一长多短场景降低多核尾部等待。
- 风险：中到高，属于 tiling 改动，不应和单核流水改动一起评估。
- 验证：B=8/16 ragged、一长多短、大 sparse_count；比较最大核时间、平均核时间和总 kernel time。

### P18 仅在 profiling 命中后尝试 UB bank padding 或 VF 算法替换

- UB bank：当前 `stage1OutQue_`、softmax 三槽状态和临时 buffer 相邻，但不能仅凭地址猜测冲突。先看 bank conflict 指标，再对单个可疑 buffer 增加 256B padding 做 A/B。
- Exp：当前 MX VF 使用 `FusedExpSub` 并带有 LN2/E8M0 语义。不要直接改成 LUT exp 或通用 Exp，除非 VectorBound 明确、存在等价官方路径且全精度回归通过。
- Cast：P 的 FP32->E4M3 和 PScale 的 float/bf16->E8M0 是 C2 输入协议的一部分，不是冗余 cast。
- 同步：AIC/AIV 的 `WaitCrossCore/SetCrossCore` 不能仅凭代码阅读删除，必须通过流水图定位重复等待。
- 风险：高，因此不进入第一轮优化。

## 5. 建议实施顺序

建议每个阶段只提交一类改动，并保留前后性能数据：

1. 建立基线：收集 kernel time、Scalar Load/Store、MTE2/Cube/Vector 利用率、等待时长、各核耗时。
2. 第一批低风险 Scalar：P01、P03、P04。每个点可单独提交，也可以在全部 A/B 均有正收益后合并。
3. 第二批循环不变量：P02、P08、P09。优先观察 BS64 和大 sparse_count。
4. 第三批编译期裁剪：P10；若 ScalarBound 仍明显，再评估 P15。
5. 第四批结构优化：P05；只有证据充分才做 P06/P07。
6. 第五批流水：先 P12，再 P13。每次只改变一条 MTE2/Cube 重叠链。
7. 第六批场景快路径：P14、P16、P17，分别面向连续 page、小 case 和多核不均衡。
8. 最后才考虑 P18。

## 6. 推荐性能与精度基线

每个点至少覆盖以下场景：

- BS128：单个 logical S2 tile 和两个以上 logical S2 tile。
- BS64：8 个 64-token block 拼成 512，以及 64-token 尾块。
- mask：`mask_mode=0/3`，含完全可见、部分 causal、无效行。
- LSE：true/false。
- sparse：sequential、reverse、random、负 index、不同 sparse_count。
- batch/head：B=1 与 ragged B>1；N2=1/3/8；GQA=5/16。
- PA：sequential/reverse/random block table，block_num 小值和大值。
- 边界：S1/S2 尾块、actual sequence 不等长、空 tensor。

性能记录建议包含：

| 指标 | 用途 |
|---|---|
| Kernel 总耗时 | 判断端到端是否真正变快 |
| Scalar Load/Store 与 spill | 验证 P01~P09、P15/P16 |
| MTE2 active/wait | 验证 P12~P14 |
| Cube active 与 Cube 前等待 | 验证 C1/C2 是否被搬运卡住 |
| Vector active 与 AIC/AIV 等待 | 验证 VF 和跨核同步 |
| 每核 task 数、有效 token、耗时 | 验证 P17 |
| OUT/LSE 精度通过率和 max error | 防止性能修改改变算法语义 |
| 重复执行稳定性 | 捕获 event、buffer 轮转导致的偶现问题 |

## 7. 可以组合的优化点

以下组合只有在各点独立验证通过后再合并：

1. **Scalar 基础组合：P01 + P03 + P04**  
   都是删除重复计算或无效状态，耦合较低，适合作为第一批。

2. **Sparse 地址计算组合：P02 + P08 + P09**  
   共同减少 task/block 级地址构造，但应先分别验证，再合并以便定位回归。

3. **LSE 裁剪组合：P10 + P04**  
   P10 编译期关闭 LSE 后，可进一步删除 false 分支不需要的运行期字段；不要提前删除 true 分支字段。

4. **C2 搬运组合：P12 + P14**  
   P12 隐藏 V/VScale 搬运，P14 减少 scale 小包。先完成 P12 的同步正确性，再添加 P14 快路径。

5. **Blocksize Scalar 组合：P03 + P08 + P15**  
   BS64/128 特化可以让 block 遍历常量化，但会增加 kernel 变体，放在通用低风险优化之后。

6. **不建议同时首次引入：P05/P06/P07、P12/P13、P15/P17**  
   这些组合一次改变的状态、同步或调度范围过大，出现性能或精度回归时难以定位。

## 8. 当前不建议直接采用的通用技巧

- 不直接删除 P/PScale cast；它们是 MXFP8 C2 输入格式的一部分。
- 不直接替换 `FusedExpSub`；LN2 和 E8M0 相关语义需要保持一致。
- 不把 PA sparse block 默认视为物理连续；scale 合并必须有可证明的快路径条件。
- 不因为已经配置双 buffer 就认定搬运已经被隐藏；必须检查指令发射顺序和流水图。
- 不直接删除跨 AIC/AIV 同步；先确认生产者、消费者和 buffer 轮转关系。
- 不为所有 UB buffer 统一加 padding；只对 profiling 命中的冲突点实验。
- 不一次性重构整个 MX kernel。优先保持 FP8 与 MX 隔离，并让每个优化点都可以独立回退。

## 9. CANN skills 参考来源

- `ops/ascendc-perf-optimize/SKILL.md`
- `ops/ascendc-perf-optimize/references/single-core-pipeline/scalar.md`
- `ops/ascendc-perf-optimize/references/single-core-pipeline/memory.md`
- `ops/ascendc-perf-optimize/references/single-core-pipeline/vec.md`
- `ops/ascendc-perf-optimize/references/single-core-pipeline/no-bound.md`
- `ops/ascendc-performance-best-practices/reference/scalar/guide.md`
- `ops/ascendc-performance-best-practices/reference/scalar/coding_principles.md`
- `ops/ascendc-performance-best-practices/reference/common/datacopy_optimization_design.md`
- `ops/ascendc-performance-best-practices/reference/common/ub_resident_design.md`
- `ops/ascendc-performance-best-practices/reference/matmul/mte2_preload_design.md`
- `ops/ascendc-performance-best-practices/reference/matmul/pingpong_design.md`
- `ops/ascendc-performance-best-practices/reference/matmul/scale_coalescing_design.md`
- `ops/ascendc-performance-best-practices/reference/reduce/templates/state_resident_design.md`
- `ops/ascendc-regbase-best-practice` 中的 pipeline、DataCopy 和 RegBase 性能实践


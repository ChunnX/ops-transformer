# MixedQuantSparseFlashMla Learning Progress

## Repository And Ownership

- 当前目录是小组实现的 `mixed_quant_sparse_flash_mla` 算子代码，主要归属文件是 `op_host/`、`op_kernel/`、`tests/pytest/`。
- `attention/common/` 是大组通用公共基础设施。当前算子可能通过条件 include 使用其中公共实现，但分析时要区分公共代码和本算子私有实现。
- 文档类输出统一放到 `/home/panjie/workspace/aichat/mixed_quant_sparse_flash_mla/`；但 `learning_progress.md` 和 `work_progress.md` 始终在当前算子目录维护。

## Tiling And Block Size

- 主计算基本块是 `S1 x S2 = 64 x 128`。
- `s1BaseSize = 64`，Cube、Vector、Kernel `constInfo` 三侧保持一致。
- `s2BaseSize = 128`，常规 S2 loop 按 128 处理，尾块使用真实剩余长度并对齐。
- `dBaseSize = 512` 只用于 Cube 侧 Q 左矩阵 L1 buffer 大小：`64 * 512 * sizeof(Q_T)`。
- `dBaseMatmulSize = 128` 用作 matmul 模板 `baseK`，控制单轮 K 方向处理粒度。

## n=64 And n=128 Strategy

- 当前 `n2Size = 1`，Q 侧 head 数主要体现为 `gSize`。
- `n=64` 时 `IS_SPLIT_G=false`，单个 AIC 覆盖完整 64 个 head；两个 AIV/subBlock 在 Vec1/Vec2 中按 M 维拆半。
- `n=128` 时 `IS_SPLIT_G=true`，相邻 AIC 按 G 轴拆成两个 64-head 分片：偶数 `aicIdx` 处理 `goIdx=0`，奇数 `aicIdx` 处理 `goIdx=64`。
- `n=128` 的 cube 配对关系是 `(AIC 2k, AIC 2k+1)`：两者处理同一个 `(B,N2,S1)` 任务，分别覆盖前后 64 个 Q head；它们共享同一组 Vec0 反量化 KV 的 GM 三缓冲 workspace，workspace 索引用 `(aicIdx >> 1)`。
- `n=128` 下不是两个 cube 合并做一个 128-head matmul，而是每个 cube 独立完成自己 64-head 分片的 BMM1、Vec1 softmax、BMM2、Vec2 写回；共享点主要是 KV 准备结果和配对 workspace。
- Vec0 稀疏 KV 处理中，`n=64` 下两个 subBlock 将 `s2RealSize=128` 分成 `[0,63]` 和 `[64,127]`；`n=128` 下两个 AIC 的四个 subBlock 近似切成四段。
- FD staging 在 `n=128` 下按逻辑 `M=128` 布局：FA 阶段前 64 head 写 `[0,63]`，后 64 head 写 `[64,127]`；FD 阶段不感知 cube 配对，只按 metadata 的 `workspaceIdx/mStartIdx/mNum` 读取逻辑 M 行。
- `n=128` FD staging 不额外增加总 workspace：每个 logical slot 的行数从 64 翻倍到 128，同时 logical slot 数按 cube pair 减半。

## MatmulK Understanding

- `MatmulK` 不是当前算子目录内的私有实现，而是来自 `attention/common/op_kernel/matmul.h` 这类大组公共代码。
- Bmm1 形态是 `Q[M,D] x K[S2,D]^T -> [M,S2]`，所以 Bmm1 的 Matmul K 轴对应 D 维。
- Bmm1 中 `param.singleK = constInfo.dSize` 表示运行时完整 K 长度，例如 512；模板 `dBaseMatmulSize=128` 表示每轮切 K 的基础块。
- 对 D=512，`MatmulK` 按 128 切成 4 轮，每轮搬 L1 到 L0A/L0B 并发一次 MMAD，后续轮累加到同一个 L0C。
- Bmm2 的 K 轴是 S2，Dv 在 Matmul N 轴上。
- Bmm1 结果是 `QK^T` score，先在 L0C，再通过 Fixpipe 到 `bmm1Buffers` UB；Vec1 对 score 做 softmax/cast 后得到 P。
- P 先落在 `stage1OutQue` UB，再写入 `l1PBuffers` L1，作为 Bmm2 的左矩阵。
- K/V 都由 Vec0 从 GM 读入量化 KV，经 UB 反量化后写入 `l1RightBuffers`；Bmm1 阶段该 L1 buffer 表示 K，Bmm2 阶段表示 V。
- `mmL0ABuffers` 在 Bmm1 中承载 Q，在 Bmm2 中承载 P；`mmL0BBuffers` 在 Bmm1 中承载 K，在 Bmm2 中承载 V。

## Current Learning Focus

- 用户当前对 Vec0 和 Vec1 阶段已有较深入理解。
- 后续学习重点放在 MM1、MM2、Vec2 阶段，以及 AIC/AIV、Cube/Vector 跨阶段同步关系。
- 回答问题时优先从数据流、buffer 角色、生产者/消费者关系和同步依赖四个角度解释，避免只停留在函数名或单段代码说明。

## MM2 And Vec2 Synchronization

- `ProcessVec2()` 入口先对 `bmm2ResBuf.WaitCrossCore()`，表示 AIV 等待 AIC 的 BMM2 结果已通过 Fixpipe 从 L0C 写入 UB。
- `runInfo.vec2MBaseSize` 来自 `runInfo.halfMRealSize`，表示当前 AIV subBlock 在 M 维上实际需要处理的行数。
- 当前代码中 `runInfo.vec2MRealSize = runInfo.vec2MBaseSize`，表示 Vec2 当前实际处理的 M 行数；它用于计算 `vec2CalcSize = vec2MRealSize * dTemplateAlign64`，并作为 `FlashUpdateNew`、`FlashUpdateLastNew`、`LastDivNew` 和最终 `DataCopyPad` 写出时的行数。
- `vec2MRealSize` 这一层命名保留了“base size”和“real size”的区分：base 表示按 subBlock 切分出来的计划块大小，real 表示本次 API/搬运真正处理的有效行数。当前 D<=128 路径未再细分，所以二者相等；若后续 Vec2 内部继续按 S1/M 尾块切分，`vec2MRealSize` 就是每个子片的真实 M 行数。
- 当 `vec2MBaseSize == 0` 时，当前 subBlock 没有有效 Vec2 行可算，但仍然必须执行 `bmm2ResBuf.SetCrossCore()` 后返回。
- 这个 `SetCrossCore()` 不是通知“Vec2 产出了有效结果”，而是归还/释放 `bmm2ResBuf` 的跨核同步状态，让 Cube 侧下一轮 `outputBuf.WaitCrossCore()` 能继续占用该 UB buffer 写 BMM2 结果。
- 如果在 `vec2MBaseSize == 0` 分支直接 `return` 而不 `SetCrossCore()`，可能导致 AIC/Cube 后续等待 `bmm2ResBuf` 可用时卡住，形成流水线同步死锁。
- Vec2 最终 `DataCopyPad(UB -> GM)` 按多行搬运理解：每个 block 是一行 Dv 输出，`blockLen = dSizeV * sizeof(OUTPUT_T)` 表示每行有效字节数，`blockCount = vec2MRealSize` 表示写出多少行。
- `srcStride = (dSizeAligned64 - dSizeV) >> 4` 表示 UB 源端相邻有效行之间的 padding gap；对 BF16/FP16 输出，`>>4` 等价于乘以 2 字节后除以 32 字节。它不是源行长度，而是搬完一行有效 Dv 后要跳过的 UB padding。
- `dstStride = attentionOutStride` 表示 GM 目的端相邻输出块之间需要跳过的字节数，用于匹配 BSND/TND 输出 layout 中当前 N/G 分片到下一条目标 Dv 向量之间的间隔。
- 当前 QSMLA 实际受限条件下 `N2=1`、`dSizeV=512` 且 `dTemplateAlign64=Align64(512)=512`，因此 `srcStride = 0`；同时 `n2G = n2Size * gSize = gSize`，所以 `attentionOutStride = (n2G - gSize) * dSizeV * sizeof(OUTPUT_T) = 0`，即 `dstStride = 0`。这两个 stride 主要是为轴占位/泛化保留的参数。

## BufferManager And Memory

- `l1BufferManager` 初始化 512 KB，实际由 `l1PBuffers`、`l1RightBuffers`、`l1QBuffers` 使用 480 KB，约剩余 32 KB。
- `l1PBuffers`: DB，两份 16 KB，共 32 KB。
- `l1RightBuffers`: 3buff，三份 128 KB，共 384 KB。
- `l1QBuffers`: single buffer，一份 64 KB。
- `l0aBufferManager` 初始化 64 KB，`mmL0ABuffers` 使用 32 KB。
- `l0bBufferManager` 初始化 64 KB，`mmL0BBuffers` 使用 64 KB。
- `l0cBufferManager` 初始化 256 KB，`mmL0CBuffers` 使用 256 KB。
- `ubBufferManager` 初始化 96 KB，由 `bmm1Buffers` 和 `bmm2Buffers` 用满。
- `bmm1Buffers` 只有 DB 两块：`mm1ResultSize=16 KB` 的 ping/pong，共 32 KB。BMM1 的 “写入 UB” 本质就是 Cube 侧 `Fixpipe(L0C -> bmm1Buffers UB)`，不能把“写入”和“Fixpipe 写出”重复算成两组 buffer，更不是四块。
- Vec1 阶段不是单一精度：`T=float`，`Q_T=bfloat16_t`。`bmm1Buffers` 中的 `QK^T` score、`commonTBuf`、`softmaxExpBuf` 按 FP32 使用；`softmaxSumBuf/softmaxMaxBuf` 也显式为 FP32；Vec1 经过 `ProcessVec1Vf<T, Q_T>` 后把 P cast 成 BF16，写入 `stage1OutQue` 和 `l1PBuffers`，供 BMM2 作为左矩阵。
- Vec1 数据路径分三段：输入 score 在 `bmm1Buffers` UB DB 中，每块 16 KB、共 32 KB；Vec1 计算结果 P 先写入 `stage1OutQue[taskIdMod2]`，这是两份 `TQue<VECOUT,1>`，每份 `33*128*sizeof(bfloat16_t)=8448 B`，不是 `BuffersPolicyDB` 但按 `taskIdMod2` 做 ping/pong；随后 P 通过 `DataCopy` 写入 `l1PBuffers` L1 DB，每块 `64*128*sizeof(bfloat16_t)=16 KB`、共 32 KB。`stage1OutQue` 和 `l1PBuffers` 不是同一块内存。
- AIV Vec 本地 UB 另由 `TPipe::InitBuffer` 申请，账面约 153.5 KB；叠加公共 BMM UB 后 AIV 侧约 249.5 KB，接近 256 KB，最终物理余量以编译资源报告为准。
- L0A manager 64 KB，`mmL0ABuffers` 使用 32 KB；L0B manager 64 KB 使用 64 KB；L0C manager 256 KB 使用 256 KB。
- 当前源码未见显式 L2 buffer 申请或 set cache 行为，`l2CacheSize_` 字段未检索到实际使用。
- `IS_SPLIT_G` 场景下 GM workspace 额外申请 `128 KB * 3 * (aicNum >> 1)`，用于 Vec0 反量化 KV 的三缓冲 GM 中转；非 split-G 场景不申请这部分。

## Empty Sequence Handling

- 当前 host check 不支持全局 shape size 为 0 的输入：`CheckSingleParaQuery()` 禁止 `q` 任意维为 0，`CheckSingleParaKey()` 禁止 `oriKv/cmpKv` 任意维为 0，`cmpSparseIndices`、block table、cuSeqLensQ 也有 shapeSize 非 0 检查。
- Kernel 侧支持的是按 batch 的 actual length 为 0：`actualS1Size == 0` 时 `s1RealSize/mRealSize` 置 0；`actualS2OriSize == 0` 时 S2 loop 置 0 并返回 no-need-calc。
- 主循环中 `s1NoNeedCalc || s2NoNeedCalc` 会直接 `continue` 跳过 BMM1/Vec0/BMM2/Vec2。
- 当某 batch `actualS2=0` 且 `actualS1>0` 时，`InitGlobalBuffer()` 的 `needInit` 检测到 `s1Size > s2Size`，`CleanOutput()` 会把整个 attentionOut 清零，避免跳过计算后输出残留。
- 当某 batch `actualS1=0` 时，对应 batch 没有有效输出行；TND 转换/Golden 中也会直接跳过该 batch 的拷贝。

## Sequence Length Terms

- `cu_seqlen_xx` 表示数据在内存中的排布长度，采用前缀和形式存储；每个值表示前面所有 batch 在 S 维累计占用的内存长度。
- `usedlen_xx` 表示每个 batch 实际参与计算的 S 长度。
- 例如 `cu_seqlen_xx = [0, 3]`、`usedlen_xx = [2]` 时，内存占用按 3 计算，实际计算长度按 2 计算。
- TND padding 指 `layout = TND` 时，Q 或 KV 的 `cu_seqlen_xx` 推导出的内存排布长度与 `usedlen_xx` 指定的实际计算长度不一致；该现象只出现在 TND layout。

## Flash Decode Metadata S2 Range

- Flash decode metadata 中 `FA_S2_START_INDEX`/`FA_S2_END_INDEX` 是 S2 block 级左闭右开区间，kernel 使用前需乘以 `s2BaseSize` 转成 token 级。
- 当 metadata `FA_S2_END_INDEX == 0` 且当前核任务结束于最后一个 batch 的最后一个基本块时，`0` 表示“不额外截断到某个 S2 block”，kernel 应使用当前任务自身的真实 `totalLen` 作为有效右边界，而不是把右边界解释为 token 0。
- 对应地，`hasNextCore` 判断也应使用转换后的 effective end；否则最后块会被误判为空范围或错误标记为仍有后继 split。
- 跨 S1 的 metadata 区间按 lexicographic 任务范围理解：首个 S1 行只应用 `S2Start`，末个 S1 行只应用 `S2End`，中间 S1 行应覆盖完整 S2。`isS2Split` 应基于当前行裁剪后的 `[rangeStart, rangeEnd)` 是否小于完整 `[0, totalLen)` 判断，不能仅根据“是否首/末任务”判断，否则中间完整 S1 行会误进 FD staging。
- FIA 使用 common `vf_flash_decode.h` 时没有在算子私有文件里声明 `using AscendC::Reg::StoreDist`，而是在包含 VF 头之前先包含 `kernel_operator.h`、`kernel_operator_list_tensor_intf.h`、`kernel_tiling/kernel_tiling.h`、matmul 相关头等 CANN 上下文；QSMLA 应对齐这种 include 顺序，不修改 common 头文件。
- FD 前用于保护 GM staging 生产/消费顺序的 `SyncAll()` 必须让 `hasLoad == 0` 的空核也参与；否则有效任务较少时空核提前 return，active 核在 FD 前全局屏障等待不到所有参与核而卡死。
- Metadata 结束坐标是 `(BN2_END, M_END, S2_END)` 组合坐标；当 S2 split 发生在同一个 BN/M 上时，`M_END` 可能为 0 但 `S2_END != 0` 仍表示当前核有有效工作。Kernel 将右开 BN2 结束转换成循环上界时，不能只检查 `M_END != 0`，还必须检查 `S2_END != 0`。
- `ComputeS1LoopInfo()` 处理最后一个 BN 的 `gs1LoopEndIdx` 时，同样必须把 `(M_END, S2_END)` 作为组合结束坐标判断；只有 `M_END == 0 && S2_END == 0` 才表示虚拟下一个 batch 的 S1 为 0 并回退到当前 batch 的真实 S1 结尾。S2 切分下可能出现 `M_END == 0 && S2_END != 0`，这里的 `M_END=0` 是当前 batch 的 M 坐标为 0，不是下一个 batch 的 0，因此不能回退为真实 S1 结尾。
- `ProcessFlashDecode()` 中 `blockSumUb` 会在 `ComputeScaleValue_VF` 后保存 scale，并被后续 `ReduceFinalRes_const_VF` 在 V pipe 读取；chunk 间复用 `blockMaxUb/blockSumUb` 前必须先等上一 chunk 最后一次 V 计算完成，否则下一 chunk 的 MTE2 读入可能抢先覆盖旧 scale，表现为偶现、点位不固定的精度问题。
- `ReduceFinalRes_const_VF()` 内部会读取 `partialOFp32` 和 `blockSumUb`，并在 `splitIdx > 0` 时读写同一块 `chunkAccumO` 做累加；`SetFlag<V_MTE2>` 前需要先 `PipeBarrier<PIPE_V>()`，让该事件真正表示本轮 V pipe 已经读完 `partialOFp32`、写完 `chunkAccumO`，否则下一 split 的 MTE2/V 可能过早复用这些 UB。
- 当前 FA 写 staging 的 `s2SplitIdxCounter` 与 workspace 常量都隐含“每个 staging 起点最多 2 个 S2 split”：`MAX_S2_SPLIT_NUM=2`，host workspace 按 2 个 split/slot 分配，FA 写端用 `firstFdDataWorkspaceIdx + s2SplitIdx` 定位 partialO/max/sum。counter 只应在当前核 S1 循环范围的最后一个有效 S1 task 确实发生 S2 split 时推进，即 `gS1Index == runParam.gs1LoopEndIdx - 1`；末个 S2 loop 才会实际写 FD staging。若 metadata 允许同一 `(BN,M/G)` 被 3 个及以上核切分，需要 host metadata/workspace 共同扩展。
- `runParam.s2SplitIdx` 会通过 `SetRunInfo()` 传给当前 S1 task 的每个 S2 loop，但实际参与 staging 地址计算只在 `isS2Split && s2LoopCount == s2LoopLimit` 的末轮：Vec1 末轮拷出 max/sum，Vec2 末轮拷出 partial attention output。因此它标识的是“当前核负责的这个 S1 task 对应的 split slot”，不是每个普通 S2 基本块都独立消费的索引。

## Skills

- 已从 `/home/panjie/.config/opencode/skills/` 安装缺失 skills 到 `/home/panjie/.codex/skills/`，安装方式为 symlink。
- 新安装的 skills 包括 `ascendc-crash-debug`、`ascendc-direct-invoke-template`、`ascendc-regbase-best-practice`、`npu-arch`、`ops-precision-standard`、`ops-profiling`、`torch-ascendc-op-extension`。
- 新 skills 需要重启 Codex 后才会出现在可用技能列表。

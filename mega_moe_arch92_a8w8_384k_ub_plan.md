# A6 A8W8 384 KiB UB 优化方案（最终整合版）

更新日期：2026-09-12。源仓：`D:\gitcode\ops-transformer_910b2`。

## 1. 范围、结论与使用方式

本文件合并原 UB 候选排查和源码复核文档，统一修正事实与公式，完整保留逐阶段占用账本。
这是供人工评估的**最终整合文档**，不是已选定、已实施或已证明提速的唯一方案；互斥候选继续保留。
本次只整理 Markdown，不修改 kernel、Host tiling、tensor 仓、同步或打点。

### 1.1 依据与边界

- 范围为当前 arch92 A8W8 MTE / 1C1V 路径，BF16 输入输出，FP8 activation/weight，DN 和 NZ 权重布局；A8W4/A4W4不在本方案实施范围。
- 初次源码复核基线为 `503814d1a`；占用账本以2026-09-10读取的 `11f0378b50d81364519b74ba9aaa5a3ec805694a` 工作目录为据。
- 2026-09-12合并时 HEAD 仍为上述版本，已核对影响本文件的入口、UB分配、Host容量与关键常量；源仓现有代码改动保持原样。后续代码变化应按函数重新定位。
- A6物理UB按用户提供的384 KiB计算，容量上限仍以平台实际可用值为准。`1 KiB=1024 B`，384 KiB为393216 B，不能把它写成384000 B。
- 这是静态分析；示例算术可以核算，硬件支持、精度覆盖和性能收益未经过本次A6编译、模拟或profiling验证。
- 本地 [tensor BlockMmad](/D:/gitcode/ops-tensor/include/blaze/gemm/block/block_mmad_qbmm_mx.h:504) 仍有MX operand固定slot0的精度fallback；Transformer还独立关闭L0C ping-pong。二者不能当作UB优化顺带解除。

### 1.2 统一结论

1. 不是所有阶段都锁死在256 KiB：Dispatch、SendMask、Unpermute已使用平台UB容量自适应。应找固定地址、生命周期浪费与协议限制，而不是统一替换一个容量常量。
2. 当前普通prefetch epilogue容量是132 KiB；非prefetch及共享GMM1是256 KiB。把prefetch恢复256行后的264 KiB是**候选预算**，不是当前占用。
3. Gate与gluRes同址，Up与后续FP8/scale/scratch跨子阶段复用；不能把这些Tensor逐个相加。
4. Reset零值从Init起就要保留；共享输入准备已经复用Quant输出槽。启动阶段重排地址必须反映这两个事实。
5. MXFP8是每32元素独立scale。当前量化Combine等待整行是实现选择，按256列tile量化在数学上可行，但短scale写和通信完成协议仍须验证。
6. 共享GMM2产生在普通MoE之后，不需要跨越前面的MoE存活；局部融合的主要约束是生产/消费分核和输出规模。
7. UB容得下不代表Cube算得了、跨核访问可行或同步正确。UB直通还会限制AIC领先量，必须按完整wave评估背压。
8. 当前Quant、Dispatch、Unpermute已有不同程度的分槽同步；邻接Set/Wait不等于零重叠。未调用helper的地址常量也不计运行期UB。

### 1.3 阅读导航

- 理解流程和GM保留原因：第2章。
- 人工逐项算容量和地址：第3章，特别是3.3至3.10。
- 选择优化方向：第4章C01..C19；它们是候选集合，不是强制实施顺序。
- 判断互斥、别名和跨阶段常驻：第5章。
- 设计少量有效验证：第6章。

## 2. 当前执行模型与数据归属

### 2.1 当前入口与任务归属

[kernel 入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44) 指定 `KERNEL_TYPE_MIX_AIC_1_1`，实际模板进入 arch92 A8W8 Wave。
当前范围是 BF16 输入输出、匹配的 FP8 activation/weight、MTE 通信，权重包括 DN 和 NZ。

[Wave 入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:358) 的顺序为：

```text
AIV: 本卡 Quant -> SendMask -> Reset -> 共享专家输入拆分
AIC/AIV: 共享专家 GMM1 + activation
跨 rank 同步
普通 MoE wave:
    Dispatch 与 GMM1/activation 交错
    当前 wave 的 GMM2/Combine
导出专家计数
共享专家 GMM2
AIV: 跨 rank 同步 -> Unpermute
```

这个顺序只描述各自执行流，不代表 AIC 和 AIV 在所有阶段边界严格同进同退。
AIV 会在消费当前 GMM1 输出前推进下一段 Dispatch，而 AIC 可以继续产生 GM 输出。
因此，把输出改成 UB 直通时，必须把 Dispatch 对同一 AIV UB 的使用也放进所有权协议。

### 2.2 数据依赖和保留必要性

| 数据 | 生产端 -> 消费端 | 当前介质 | 从源码得到的约束 |
| --- | --- | --- | --- |
| 本卡量化输入 | Quant -> 各 rank Dispatch | peer GM | 多 rank、多专家复用，不能仅放生产 AIV 的 UB |
| Dispatch FP8 与 scale | Dispatch -> 多个 GMM1 N tile | 本地 GM | 存在跨 AIC 扇出 |
| 普通 GMM1 Gate/Up | AIC -> 对应 AIV activation | 本地 GM | UB 直通有可能，但需给落后消费者保留空间 |
| 共享 GMM1 Gate/Up | AIC -> 配对 AIV activation | UB 单槽 | 现有代码路径已存在，运行覆盖情况另需确认 |
| activation FP8 与 scale | activation -> GMM2 | 本地 GM | 多核产生、多个 GMM2 AIC 读取 |
| 普通 GMM2 BF16 | AIC -> Combine | 本地 GM | 非量化按 tile 消费，量化当前按整行消费 |
| shared GMM2 BF16 | AIC -> Unpermute | 本地 GM | 在普通 MoE 之后产生，受分核和输出规模约束 |
| Dispatch prefix | AIV -> 自身后续 Dispatch/计数导出 | UB + GM 备份 | 可常驻，也可缩减成当前专家状态 |

### 2.3 容量与同步必须分开看

[Host 平台容量查询](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2045) 已读取平台 UB 容量。
Dispatch、SendMask 和 Unpermute 的 Host 自适应公式已使用这个值，不能说所有路径都被整体锁死在 256 KiB。

真正要分别检查的是固定地址、生命周期、输出协议以及是否有消费者等待。

`SetFlag/WaitFlag` 紧挨着出现，并不自动意味着所有硬件流水串行。
例如 `MTE2_V` 表达 Vector 消费前需要输入就绪，不能仅凭 C++ 调用顺序推断下一槽 MTE2 无法提前执行。
`MTE3_S`、`V_S` 等会影响 Scalar 推进的等待，和针对单一流水线的等待需要分开分析。
具体重叠程度必须通过编译后的指令或 profiler 判断。

## 3. 当前 UB 占用账本

本章公式描述当前布局；候选布局另见第 4 章。地址与容量均以单个 AIV 的 UB 为单位。

### 3.1 先区分四种“占用”

1. **有效数据量**：例如当前只处理 25 行，就只有这 25 行的有效元素。
2. **布局范围**：代码为最大 tile 或 ring 槽规定的地址区间，包含 padding；尾块通常不会移动后续 buffer 的基址。
3. **同时存活的数据**：已经产生且后续仍要读取的数据，以及尚未完成的异步搬运所引用的区域。
4. **最高地址**：所有相关布局区间末尾的最大值。不能仅凭有效数据量之和判断是否越界。

本文的 `E` 是相对 UB 基址 0 的排他末尾，区间统一写为 `[起点, 末尾)`。
判断越界使用 `E <= U`，其中 `U` 是平台传给 Host 的可用 UB 字节数；按 384 KiB 示例时 `U=393216`。
代码中的 `LocalTensor(..., address, count)` 是按地址构造视图，不是按函数自动申请、释放的内存。
`VECIN`、`VECOUT`、`VECCALC` 在这里不能被当作三块互不相干的 UB；相同物理地址的视图会互相覆盖。
`ReinterpretCast`、切片以及同址的 Te Tensor/LocalTensor 也不额外占一份空间。
MicroAPI 的 `RegTensor` 是寄存器，不按 UB Tensor 累加；L1/L0A/L0B/L0C 也不属于本账本。

记号和单位如下，除明确标注元素数外，容量均为字节：

| 记号 | 含义 |
| --- | --- |
| `A_a(x)` | 向上对齐：`ceil(x/a)*a`；必须注意对齐对象是字节还是元素 |
| `H` | tokenHiddenDim，即输入 H / GMM2 输出 N；本章讨论 BF16 输入输出、FP8 activation/weight |
| `R, E_r, C` | worldSize、本卡 MoE 专家数、参与任务分配的 AIV 核数 |
| `B, T, P` | 本卡 token 数、topK、TopkWeightsPrefetch 开关（0 或 1） |
| `g` | 每行 MX scale 有效数：`ceil(H/32)`，每个 E8M0 scale 为 1 B |
| `O` | Dispatch 输入记录大小：`A_256(H)+A_32(g)+P*A_32(4*T)` |
| `I` | Quant 的单个 BF16 输入槽：`2*A_128(H)`；这里先对齐元素数，再乘 2 |
| `Q_s, B_s` | SendMask 的 routeItemsPerBatch、bufferCount |
| `Q_d, B_d` | Dispatch 的 routeItemsPerBatch、bufferCount；不保证等于 SendMask 配置 |
| `B_u, T_u` | Unpermute 的 inputBufferCount、tokensPerBatch |
| `M_e, N_e` | 激活 epilogue 的容量 shape；当前 `N_e=256`，`M_e` 为 128 或 256 |
| `m, n` | 当前调用实际处理的行、列数，不能代替容量 shape 计算全部基址 |

`INT32_PER_256B=8`、`META_INFO_SIZE=8`，本处一条 metadata 是 `8*4=32 B`，不是 256 B。
普通 metadata 包含来源 rank/token/topk 索引；prefetch 时还存对应路由选中的一个 weight。
它与 Quant/Dispatch 记录末尾的“这个 token 全部 T 个 FP32 weight”不是同一份布局。

### 3.2 到底分几套 UB 布局

从当前 A6 A8W8 Wave 入口看，可分为 **5 类主布局**；把 GMM2 的量化/非量化分支拆开是 **6 套**。
执行阶段更多，因为若干阶段共用一套地址，普通 MoE 中又会反复切换布局。
下表 U0..U4 是 UB 布局编号，与硬件 L0/L1 存储层级无关。

| 布局 | 使用阶段 | 地址从哪里开始 | 与其他布局的关系 |
| --- | --- | --- | --- |
| U0 启动联合布局 | Quant、SendMask、Reset、共享输入准备 | 0 | 四阶段当前在一套布局中划出多个区域；共享输入准备明确复用 Quant 输出槽 |
| U1 Dispatch 布局 | 计数、前缀和、路由搬运、最后导出专家计数 | 0 | 与激活/Combine 覆盖；当前 cumsum 借 GM 备份跨越覆盖 |
| U2 激活布局 | 普通或共享 GMM1 输出消费、SwiGLU/其他激活、MX 量化 | 0 | Gate/Up 与后续激活/量化数据阶段内复用；共享与普通调用不相加 |
| U3a 非量化 Combine | 普通 GMM2 BF16 tile 搬回并发往源 rank | 数据 0，metadata 200 KiB | 与 Dispatch/激活覆盖；固定 metadata 基址产生空洞 |
| U3b 量化 Combine | 普通 GMM2 整行搬回、MX 量化、发送 | 0 | 和 U3a 是模板分支，不同时分配 |
| U4 Unpermute | 接收普通专家结果、叠加共享结果、最终累加输出 | 0 | 最后重用前面布局；内部仍有输入 ring 和独立输出槽 |

阶段执行顺序见第 2.1 节；这里仅补充影响布局选择的分支和所有权关系。

共享 GMM2 当前写 GM，不另外给它加一份 UB 输出矩阵；AIV 在 Unpermute 中等待对应共享结果 ready 后读 GM。
当前普通 MoE 使用 `USE_GM_HANDOFF=true`，AIC 的普通 GMM1/GMM2 可以领先于 AIV；GM 中的待消费结果不计入 UB。
共享 GMM1 的通用路径未传该开关，使用 UB 单槽生产消费协议。不能把普通和共享路径混为一谈。
当前入口没有调用旧 `WaveCombineScratch`/`RunWaveExpertCombineStage` 那套 helper，
因此不能把其 64 KiB 基址、1536 条 metadata、184 KiB ready 扫描区再加到当前路径中。

源码：[Wave 入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:358)、
[Wave 交错调度](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)、
[架构策略](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:19)。

### 3.3 U0：Quant / SendMask / Reset / 共享输入准备

#### 当前地址表

定义：

```text
F = resetWorkspaceConfig.flagElementCount
若 P=1，F = max(F, (E_r*maxTilesPerExpert+1)*INT_CACHELINE)
Z = 4*A_8(min(ceil(F/C), 2048))          // resetTensor 字节，最大 8192
A = A_32(4*ceil(R*E_r/C))              // sendCntAcc 字节，按最多拥有的专家数预留
J = Q_s/8 + 32                        // 一个 SendMask 槽
```

`Q_s` 按 256 个 route item 对齐，故以下基址自然满足 32 B 对齐。表中每行紧接上一行：

| Tensor / 区域 | 起点 | 大小 | 用途与关系 |
| --- | --- | --- | --- |
| `topkIdsTensor` | 0 | `4*Q_s` | SendMask 的输入专家 id；不是 top-k weight |
| `resetTensor` | `4*Q_s` | `Z` | Init 时已 Duplicate 为零，直到 Reset 发完仍须保留 |
| `mxTempTensor` | `4*Q_s+Z` | 2048 | Quant 的 maxExp/halfScale；prefetch BF16 weight 转 FP32 时也借用此区 |
| `xOutTensor0/1` | `4*Q_s+Z+2048` | `2*O` | 两个量化输出记录槽；共享准备的 copyBuffer0/1 就是这两个视图 |
| `xInTensor0/1` | `4*Q_s+Z+2048+2*O` | `2*I` | 两个 BF16 输入槽 |
| `sendMaskTensor` | `4*Q_s+Z+2048+2*O+2*I` | `B_s*J` | 每槽 mask 加 32 B 尾部，末批写入 count；不是整张全局 mask |
| `sendGatherOutTensor` | 上一行末尾 | `4*Q_s` | GatherMask 的输出区，目前真实提供给算子，不能仅凭结果未使用就删除空间 |
| `sendCntAccTensor` | 上一行末尾 | `A` | 跨 route batch 累计本核各专家的 count |

```text
E_U0 = 8*Q_s + Z + 2048 + 2*O + 2*I + B_s*(Q_s/8+32) + A
```

这里 `2*O+2*I` 是已经存在的双 buffer，不是每次循环又申请两份。
`mxTempTensor` 内 `maxExp` 从头开始，`halfScale` 从 `uint16_t` 元素偏移 `A_32(g)` 开始，
不是另加两块 2 KiB。BF16 top-k 转换先使用同一 scratch，转换完成后才由量化覆盖。
当前 H<=8192、T<=32；扩大范围时必须重新检查固定 2 KiB scratch，而不是默认永远足够。

#### 活跃范围和可以讨论的复用

| 执行阶段 | 当前需保留的主要区域 | 本阶段触及的最高地址 / 保守边界 |
| --- | --- | --- |
| Quant | `Z + 2048 + 2*O + 2*I`；reset 零值虽没参与 Quant，也不能破坏 | `4*Q_s+Z+2048+2*O+2*I` |
| SendMask | `topkIds + GatherOut + mask ring + sendCntAcc`；另保留 reset 零值 | `E_U0`；已结束的 Quant 区域仍夹在中间 |
| Reset | `resetTensor`，供一批或多批 MTE3 反复读 | `4*Q_s+Z`；返回并不代表已回收所有异步读 |
| 共享输入准备 | `xOut0/1`；若 Reset 的 MTE3 尚未排空，其零值区也仍需保留 | copy 槽布局末尾 `4*Q_s+Z+2048+2*O` |

共享准备实际只搬 `A_256(H)+A_32(g)`，不搬 weight 尾部，但槽容量仍是 `O`，不能额外算一组 shared buffer。
它从 Quant 已写好的 peer GM 记录重新读取，拆成独立 data/scale GM 张量。
“复用槽地址”不等于“沿用 Quant 最后一行的内容”：两次遍历之间已执行 SendMask/Reset，且通常有多行。

若未来重排为同一 arena，容量候选可表达为下面的保守上界，**不是现有布局已实现的节省**：

```text
QuantScratch = 2048 + 2*O + 2*I
MaskScratch  = 8*Q_s + B_s*(Q_s/8+32) + A
SharedScratch = 2*O

保留跨阶段零值：Z + max(QuantScratch, MaskScratch, SharedScratch)
把清零移到 Reset 开始，并严格排空阶段异步操作：
               max(QuantScratch, MaskScratch, Z, SharedScratch)
```

容量公式成立的前提是允许重新落地址，且不新增重叠执行。现有 `Duplicate(resetTensor,0)` 在 Init 内，
所以只把 Quant/SendMask 地址挪到零值区会导致 Reset 向 GM flag 写入非零内容。
Quant 末尾回收两个 `MTE3_MTE2` event；SendMask 末尾回收各槽 `MTE3_V` event。
若新布局随后由 Scalar/MTE2 覆盖先前区域，还要核查跨流水的完整排空，不能把一种 event 当作所有流水全屏障。
Reset 自身没有尾部 drain；当前后面的 shared prepare / Process 中 `PIPE_ALL` 保护后续布局切换。

源码：[地址构造与提前清零](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:464)、
[Quant 与 scratch 复用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_quant.h:37)、
[SendMask](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_send_mask.h:117)、
[Reset](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_workspace_reset.h:73)、
[共享输入准备](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_shared_expert_input.h:34)。

### 3.4 U1：Dispatch 和专家计数导出

定义 `C_sum=A_32(4*R*E_r)`、`E_out=A_32(4*E_r)`、`V=A_32(4*Q_d)`。

| Tensor / 区域 | 起点 | 大小 | 存活范围 |
| --- | --- | --- | --- |
| `expertTokenCntTensor` | 0 | 32 | 当前专家总数写 GM 前须保留 |
| `cumsumInfoTensor` | 32 | `C_sum` | 当前专家定位和路由搬运；历史前缀还要供下一专家和最终导出使用 |
| `sendCntTensor` | `32+C_sum` | `32*R` | 每个 rank 一个有效 int32，搬入按 32 B 槽排布，不是 `4*R` |
| `maskBatchTensor` | `32+C_sum+32*R` | `Q_d/8` | 当前 rank 的一批位图；U32 视图与 U8 视图同址 |
| `validTopkIndexTensor` | 上一行末尾 | `V` | Gather 后的有效路由索引，持续供本批 token 搬运读取 |
| `topkIndexTensor` | 上一行末尾 | `V` | Gather 输入索引，与有效索引不是同址 |
| `copyTmp` ring | 上一行末尾 | `B_d*O` | MTE2 拉取 peer 记录，MTE3 拆写 data/scale，prefetch 还读取 weight |
| `expertTokenNumsOutTensor` | 上一行末尾 | `E_out` | 最终计数导出使用；普通 Dispatch 期间仍留着该地址范围 |
| `metaInfoTensor` ring | 上一行末尾 | `32*B_d` | 与 copyTmp 同槽编号，Scalar 填 metadata，MTE3 写到 GM |

```text
E_U1 = 32 + C_sum + 32*R + Q_d/8 + 2*V + B_d*O + E_out + 32*B_d
Q_d 按 256 对齐时：2*V + Q_d/8 = 65*Q_d/8
```

copyTmp 的 data、scale、weight 切片都包含在 `O` 中，不额外申请。
一个 token 槽的完整预算是 `O+32`，但数据 ring 与 metadata ring 在物理上分开放置。
缩槽数量必须同时改两者及 event 回绕；只缩 copyTmp 会破坏槽编号对应关系。
有效索引在 token 复制过程中仍被读取，prefetch 取 weight 时也使用它，不能在 Gather 返回后马上覆盖。

#### cumsum 为什么不能直接认为已经常驻

`cumsumInfoTensor[e*R+r]` 是按专家、rank 顺序累计的前缀；`cumsumRevCntInRank` 是独立的 Scalar 累计值，
不是又一份 UB 数组。后者能够跨函数保留，并不意味着前者在激活覆盖 UB 后仍有效。
当前 cumsum 区 `[32,32+C_sum)` 落在下一次 Gate/激活区内，所以：

```text
专家 e 计数：从 GM 恢复历史前缀（e>0） -> 写入本专家 R 项 -> 回写 GM
本专家 Dispatch：读取 UB 前缀定位 -> 完成 token 搬运
随后激活 / Combine：允许覆盖上述 UB
最终 Export：再从 GM 恢复全部前缀 -> 相邻专家末项做差 -> 输出计数
```

当前支持的总 MoE 专家约束 `R*E_r<=2048` 下，整张表最多 8 KiB。
这是每 AIV 的容量；GM 备份按核隔离。多次读写流量与这 8 KiB 容量不是一回事。
最终 Export 只需要 cumsum 和 `E_out` 两块有效区域，但仍沿用原地址，
其最高地址为 `E_U1-32*B_d`，不能误写成 `C_sum+E_out`。

常驻候选必须找一片 **所有跨越阶段都不覆盖** 的区域，并从 Dispatch、SendMask/Unpermute 等实际跨越阶段的预算中扣除。
若只要求“第一次 Dispatch 到最终 Export”常驻，不必再跨越启动 Quant 或最终 Unpermute；
但必须覆盖全部普通激活和 Combine，以及同一 AIV 上的后续 Dispatch。
放在 256 KiB 以上不自动安全：Host 已可能用 384 KiB 预算扩大 Dispatch ring 或 route batch。

Dispatch 拷贝函数末尾回收每个使用槽的 `MTE3_MTE2` 和 `MTE3_S`，分别保护数据槽和 Scalar metadata 重用。
计数回写后有 `PIPE_ALL`；最终 Export 的输出也有尾部 event 回收。
常驻化应保留这些数据就绪关系，不能连同 GM cumsum 拷贝把所有等待一起删除。

源码：[Dispatch 布局](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:374)、
[token/metadata 搬运与回收](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:92)、
[前缀更新](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:496)、
[计数导出](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:555)。

### 3.5 U2：GMM1 输出、激活与 MX 量化

#### 容量 shape 和实际计算 shape 要分开

普通 GMM1 的外层调度 `GMM1_TILE_M=256`；普通 epilogue 在 `P=1` 时按 128 行子块消费，在 `P=0` 时容量为 256 行。
共享 epilogue 固定为 256 行、prefetch=false。两者使用同一片 UB 的不同时间段，不能相加为两套常驻。
BlockMmad 内部 adaptive tile 的 M（例如 32）不等于外层 UB layout 的 M，更不等于实际 m（例如 25）。
UB 表应按实际调用的 epilogue 模板容量计算，不应看到日志 `tile_m=32` 就将整个 UB 布局缩为 32 行。

令 `S=M_e*N_e`（元素数），BF16 输入下当前有效 shape 的地址如下：

| Tensor / 视图 | 起点 B | 大小 B | 与前后阶段的别名关系 |
| --- | --- | --- | --- |
| Gate `l0cOutUbFirst_` | 0 | `2*S` | 输入完成消费后，同区域容纳激活结果 |
| Up `l0cOutUbSecond_` | `2*S` | `2*S` | 激活完成后，其区域被 FP8/scale/scratch 覆盖 |
| `gluRes_` | 0 | `2*S` | 与 Gate 完全同址，不另加一个 BF16 tile |
| `quantOutput_` | `2*S` | `S` | 位于原 Up 区前半部分 |
| `quantScaleOutput_` | `3*S` | `S/32` | 紧凑 scale，仍在原 Up 区内 |
| `maxExp_` | `3*S+S/32` | `2*S/32` | 临时 uint16 指数 |
| `halfScale_` | `3*S+3*S/32` | `2*S/32` | 临时 uint16 量化乘数 |
| `weightUb_`（仅普通 P=1） | `4*S` | `32*M_e` | 逐 token metadata，内含一个被选中 weight，不是 `M_e*T` 个 weight |

```text
量化子阶段布局末尾 = 3*S + 5*S/32
Gate/Up 布局末尾 = 4*S
E_U2 = max(4*S, 3*S+5*S/32, 4*S+P*32*M_e)
     = 4*S + P*32*M_e                 // 当前 128/256 x 256 的有效配置
```

上述 Up 起点只对当前支持的 shape 成立。源码 `kUbSecondOffset` 判断 `4*S<=256*1024`，
否则竟将第二块起点置为 0；另有生产/消费代码通过“128，否则 256”选固定 Up 偏移。
因此尝试 M=160/176/384 或改变 N，不能只修改模板参数后套 `4*S` 公式就认为正确。
需要同时修正生产端、GM->UB 拷贝、epilogue 的基址、行距与容量判断。

#### 当前具体地址（KiB）

| 区域 | M_e=128, N_e=256 | M_e=256, N_e=256 |
| --- | --- | --- |
| Gate / gluRes | `[0,64)` | `[0,128)` |
| Up（激活之前） | `[64,128)` | `[128,256)` |
| FP8（激活之后） | `[64,96)` | `[128,192)` |
| scale | `[96,97)` | `[192,194)` |
| maxExp | `[97,99)` | `[194,198)` |
| halfScale | `[99,101)` | `[198,202)` |
| metadata（如果启用 P） | `[128,132)` | `[256,264)` |

当前普通 P=1 是 **132 KiB**；普通 P=0、共享 GMM1 是 **256 KiB**。
“256 行加 metadata 共 264 KiB”是扩大 prefetch epilogue 的候选值，不是当前 P=1 的实际模板容量。
尾块 m 小于容量 M_e 时，部分区域未写满，但固定起点不会自动紧缩。

#### 为什么不能把下一 tile 直接塞进 Up 区

```text
Gate+Up 到齐
  -> 激活：读 Gate/Up，结果写到 Gate 原地址
  -> 求 maxExp、scale/halfScale
  -> 读 gluRes，写 FP8；此时覆盖原 Up 区
  -> MTE3 将 FP8 与 scale 写 GM
  -> MTE3 不再读取后，才允许下一次覆盖输出区域
```

Up 的最后一次激活读完成，意味着“原 Up 值可丢弃”，不意味着这块地址空闲，因为它马上变成量化输出区。
同理，Gate 的原值已经消耗，但 gluRes 会继续被量化读取。
现有 epilogue 末尾等待 `MTE3_V`、`MTE3_S`；普通 GM handoff 分支再进入下一次 GM->UB 拷贝。
共享单槽路径还使用 `WaitForCube/NotifyCube` 和 AIC 的 `WaitForVector/NotifyVector` 防止覆盖未消费的数据。

若做两组完整流水 bank，不能只给 Gate/Up 乘 2 而让输出和 scratch 仍固定引用 bank0。
一个保守候选为 `2*(4*S+P*32*M_e)+常驻区`，每组输出/scratch 随本组 Up 地址复用。
例如 128x256、P=1、常驻区8 KiB：`2*132+8=272 KiB`。
这是可讨论的布局预算，不代表代码已有双 bank；还需要 credit/ACK、event、bank 基址及跨 Dispatch 的所有权设计。
若想让部分 scratch 跨 bank 共用，需另外证明两个 bank 不会同时读写那块 scratch。

源码：[epilogue 类型](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:144)、
[Gate/Up 基址](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h:213)、
[量化别名布局](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h:268)、
[激活与量化次序](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h:639)、
[128 行消费](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:345)。

### 3.6 U3a：非量化 GMM2 / Combine

当前普通路径 AIC 将 GMM2 写 GM，AIV 等 ready 后将一个 tile 搬回 UB，再逐行发往源 rank。
外层 `config.tileM=256`，UB 行距是 `L1_TILE_N=256`，不随尾 N 缩短。

| 区域 | 起点 B | 大小 B |
| --- | --- | --- |
| BF16 tile / `l0cOutUbGMM2` | 0 | `2*256*256=131072`（128 KiB 容量） |
| 中间空洞 | 131072 | 73728（72 KiB） |
| `metaInfoTensor` | 204800（200 KiB） | `32*m`，`m<=256` |

```text
有效元素数据量 = 2*m*n + 32*m
按容量 tile 计的区域大小之和 = 131072 + 32*m
E_U3a = max(131072, 204800+32*m)
满 256 行：区域大小之和 136 KiB，但最高地址为 208 KiB
```

因此“只用了 136 KiB”不能解释当前全部地址边界，“占到 208 KiB”也不表示中间 72 KiB 里都装了数据。
尾行少时 metadata 更小，但其起点仍是 200 KiB。
这个空洞可以列为当前 Combine 阶段内的候选用途，不能据此认为跨 GMM1 也空闲：GMM1 Gate/Up 会覆盖这里。
若扩大 GMM2 tile，应检查 `2*M*N<=204800` 才不撞现有 metadata 起点，并重新核算 metadata 大小、行距和搬运协议。
只把可用 UB 从 256 改为 384 KiB，不会自动移动 200 KiB 固定地址。

Te 的 `tensorBlockUb` 与 LocalTensor `l0cOutUbGMM2` 指向同一区域，不计两份。
metadata 在当前逐行发送期间要保留；BF16 tile 至最后一次 MTE3 读完前也必须保留。
当前函数尾的 `MTE3_MTE2` 保护下一次 GM->UB 覆盖。改成双 tile 时必须把 metadata 生命周期一起考虑。
共享 GMM2 不执行这个普通 Combine epilogue，结果写共享 GM 区，供后续 Unpermute 读取。

源码：[固定 metadata 基址](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:226)、
[GM handoff epilogue](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:697)、
[外层 tileM](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:283)。

### 3.7 U3b：量化 GMM2 / Combine

当前一次负责 `t<=256` 个 token 的 metadata，但数据部分仅有 **两个整行槽**，不是 256 行矩阵。
调度等待这些行所属 group 的全部 GMM2 N tile ready，再逐行做量化发送。
设：

```text
h32 = A_32(H)                        // BF16 元素数
s   = A_2(ceil(H/32))                // scale 元素/字节数，偶数对齐
Q_c = A_32(A_256(H)+s)               // Combine 的 FP8+scale 发送记录，无 top-k weight
L   = 2*h32 + Q_c                    // 一个槽的字节数
F_c = 4*(A_32(s)+s/2)               // floatTemp 字节数；括号内部是 float 元素数
```

`Q_c` 与前面的 `O` 有不同的布局来源：前者没有 weight，且 scale 存储采用偶数对齐后整体对齐。
当前合法 H 条件下某些取值恰好相等，也不应在代码整改时把两个概念随意合并。

| 区域 | 起点 B | 大小 B | 关系 |
| --- | --- | --- | --- |
| metadata | 0 | `32*t` | 整批保留 |
| 槽0 BF16 | `32*t` | `2*h32` | 一整行输入 |
| 槽0 FP8+scale | `32*t+2*h32` | `Q_c` | 与输入分开，不是原地压缩 |
| 槽1 BF16 | `32*t+L` | `2*h32` | 第二个整行输入 |
| 槽1 FP8+scale | `32*t+L+2*h32` | `Q_c` | 第二个发送区 |
| `floatTemp` | `32*t+2*L` | `F_c` | 两槽共用一份量化 scratch |

```text
E_U3b = 32*t + 2*(2*A_32(H)+Q_c) + 4*(A_32(s)+s/2)
```

`floatTemp` 被重新解释为 uint16 maxExp 区，halfScale 指针位于 float 元素偏移 `A_32(s)`。
这意味着 halfScale 起点按 `4*A_32(s)` 字节算，不能因为数据类型是 uint16 就擅自按 2 倍计算。
把 floatTemp 压紧是候选修改，不是描述现状时可以省掉的空间。

| H，t=256 | 单 BF16 行 | 单 Q_c | 双槽 | scratch | metadata | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 8192 B | 4224 B | 24832 B | 768 B | 8192 B | 33792 B = 33 KiB |
| 8192 | 16384 B | 8448 B | 49664 B | 1536 B | 8192 B | 59392 B = 58 KiB |

若多个 AIV 分摊同一 group，`t` 可能显著小于256，只需替换 metadata 项；两行槽容量不变。
现有 `MTE3_MTE2`、`MTE2_V`、`V_S` 和函数尾部 drain 构成数据依赖。
两个槽并不等于所有环节已经完全重叠；共用 scratch 也不等于能够并发执行两次 Vector 量化。
可评估增槽、调整等待或 tile 级发送，但每种方式都要重新证明发送完之前不会改写对应 FP8/scale。

源码：[记录大小](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:327)、
[metadata 与入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:281)、
[双行槽和 scratch](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:172)、
[scratch 内部指针](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_mxfp8_utils.h:42)。

### 3.8 U4：Unpermute 输入、反量化、权重和输出

设 `K_s` 为共享专家数，`P` 仍是 top-k prefetch 开关：

```text
D_b = A_32(2*H)                      // 单 BF16 槽字节数
D_f = A_32(4*H)                      // 单 FP32 槽字节数
W_f = A_32(4*T_u*T)                  // 一批 FP32 top-k weight
W_b = A_32(2*T_u*T) 或 0             // 输入 weight 为 BF16 时才有转换区
S_b = A_32(4*g) 或 0                 // Combine 量化时：每 scale 展开为 2 个 BF16
S_f = A_32(16*g) 或 0                // Combine 量化时：每 scale 展开为 4 个 FP32
```

| 区域 | 起点 B | 大小 B | 用途 |
| --- | --- | --- | --- |
| `dataResTensor` | 0 | `(B_u+1)*D_b` | 槽0是 BF16 最终输出；槽1..B_u是输入 ring |
| `dataResFp32Tensor` | `(B_u+1)*D_b` | `(B_u+1)*D_f` | 槽0是 FP32 累加器；槽1..B_u是输入转换/反量化结果 |
| `topKWeightsTensor` | 上一行末尾 | `W_f` | 当前 weight batch，跨其中多个 token 保留 |
| BF16 weight 转换区 | 上一行末尾 | `W_b` | 转 FP32 后原 BF16 内容无需再读，但当前地址仍预留 |
| BF16 scale 展开区 | 上一行末尾 | `S_b` | 当前反量化输入使用，所有 ring 槽共用 |
| FP32 scale 展开区 | 上一行末尾 | `S_f` | 当前反量化输入使用，所有 ring 槽共用 |

```text
E_U4 = (B_u+1)*(D_b+D_f) + W_f + W_b + S_b + S_f
```

这里的 `+1` 不是多一个完整专家输入，而是“BF16 输出槽 + FP32 累加槽”。
两个地址区域分开放置；把 BF16/FP32 之和写成 `6H` 只是算总字节，并不代表每个输入槽物理上紧挨着一对。
非量化输入经 BF16->FP32 Cast；量化输入的 `Q_c` 字节直接装进 BF16 输入槽并 reinterpret 为 FP8/scale，
没有再申请一个独立 FP8 输入 buffer。缩小 BF16 槽时还需容纳共享专家的 `2H` 字节 BF16 输入。

普通专家和共享专家使用同一个输入 ring：

```text
accumulationItemIdx = localTokenIndex*(T+K_s) + expertIndex
inputBufferIdx = accumulationItemIdx % B_u
```

共享专家的 expertIndex 位于 T 之后。不能再按“共享专家数 * 一组 ring”叠加预算。
BF16/FP32 scale 展开区也是共用 scratch，不乘 `B_u`；若想并发反量化多个输入，应重新检查这个共用条件。

**prefetch 的保留空区**：当前 Host 仍为 `W_f/W_b` 分配预算，kernel 也保留相应视图，
但 `P=1` 的消费代码不再加载或乘 top-k weight。描述当前 `E_U4` 时仍计入这些地址，
可将其列为回收候选，不能直接说当前已经省掉了 weight buffer。

**输出不是严格阻塞所有下一 token 计算**：累加下一 token 的 FP32 数据发生在等待 BF16 输出槽可写之前。
当前次序是“累加 -> 等上次输出 MTE3 结束 -> Cast 到 BF16 输出槽 -> MTE3 输出”。
所以独立 FP32 累加槽可以和上一 token 的 BF16 发送并存；贸然将两者别名，会破坏这段重叠。
输入槽由各自 `V_MTE2` event 回收，输出槽由 `MTE3_V` event 回收，二者不能因为 event 数字相同就当成同一生命周期。

例如 H=8192、B_u=6、T_u=32、T=8、BF16 weight、Combine 量化：

```text
输入 ring：6*(16+32) = 288 KiB
输出+累加槽：16+32 = 48 KiB
weight：1+0.5 = 1.5 KiB
scale scratch：1+4 = 5 KiB
合计：342.5 KiB
```

这是一个明确参数示例，不是所有用例都能分到6槽。若 H=8192、T_u=1024、T=8，
同为 BF16 weight/量化，6槽会需要 `336+48+5=389 KiB`，超过384 KiB；
该组参数最多容纳5个输入槽时为 `288+48+5=341 KiB`，再由 Host 评估是否扩大 weight batch。

源码：[地址构造](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:551)、
[Host 槽数与 batch](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:843)、
[输入和共享专家复用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h:102)、
[输出生命周期](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h:201)。

### 3.9 Host 自适应预算和 kernel 地址必须一起计算

Dispatch、SendMask、Unpermute 已使用平台 `availableUbBytes`，不是统一写死256 KiB。
当前 ring 数上下限均为2..6；达到6槽后，额外空间可能被分给更大的 batch。
因此多出128 KiB并不一定是当前空闲，也不一定带来更深 ring。

| 布局 | Host 第一轮选槽所扣除的固定/批量区域 | 每加一槽成本 | 第二轮扩大什么 |
| --- | --- | --- | --- |
| SendMask | `Z+2048+2*O+2*I+A+8*Q_s` | `Q_s/8+32` | 固定 B_s 后扩大 Q_s，每 item 成本 `8+B_s/8` B |
| Dispatch | `32+C_sum+32*R+E_out+65*Q_d/8` | `O+32` | 固定 B_d 后扩大 Q_d，每 item 成本 `65/8` B |
| Unpermute | `W_f+W_b+S_b+S_f+D_b+D_f` | `D_b+D_f` | 固定 B_u 后扩大 T_u，weight 类型决定每 token 成本 |

SendMask/Dispatch 基准 Q 为 `min(A_256(B*T),12288)`，最终 Q 可以更大，但保持256 item对齐。
SendMask 槽数还受本核拥有专家数和 mask push 次数限制；两类核可能有不同配置。
Unpermute 基准 `T_u=min(floor(8192/T),本核token数)`，ring 还受 `T_u*(T+共享专家数)` 限制。
其扩大 weight batch 时预留64 B覆盖两个独立权重区的对齐余量；这64 B是预算余量，不是一个额外 Tensor。
完整 token chunk 核、尾 chunk 核的配置可能不同，无任务核应单独考虑。

如果增加常驻区 `P_keep`，只有在重新安排所有相关地址并扣除预算后，才可以写：

```text
U_stage_available = U - A_32(P_keep)
用 U_stage_available 重新求本阶段 ring/batch
最后再次验证每个区间末尾 <= U，且没有覆盖常驻区
```

不能先按384 KiB求出 Q/ring，再把常驻区放到“最上面的8 KiB”；也不能只减少 Host 预算而让 kernel 固定地址仍与常驻区相交。
同样，若保留当前 ring 深度而改成更大 M/N，Host 的预处理预算和 kernel 的 epilogue 固定地址不是同一套公式，必须分别修改与验证。

源码：[Dispatch 自适应](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:704)、
[SendMask 自适应](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:782)、
[统一预算入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:975)、
[槽数和 batch 常量](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_tiling.h:49)。

### 3.10 同一组参数下怎样读这张账本

以下只为手工算账示例：`R=8, E_r=20, C=24, B=256, T=8, P=1`，
`Q_s=Q_d=2048, B_s=B_d=6`；Reset 按最大 `Z=8192 B` 做保守预算，实际应代入 flag 数重新计算。
这组 B*T=2048，没有扩大 route batch 的必要。`C_sum=640 B, E_out=96 B, A=32 B`。

| H | 单记录 O | 单输入 I | U0 联合布局（Z按上限） | U1 Dispatch |
| --- | --- | --- | --- | --- |
| 4096 | 4256 B | 8192 B | 53280 B = 52.03125 KiB | 43392 B = 42.375 KiB |
| 8192 | 8480 B | 16384 B | 78112 B = 76.28125 KiB | 68736 B = 67.125 KiB |

不能从这个小 B 示例推断 Dispatch 永远只用67 KiB：B*T大时，Host 会增大 Q，ring/batch可接近平台预算。
对任意实际输入，应先取对应核的 tiling 配置，再代入，而不是把本文示例硬编码成新地址。

各阶段在满足已有所有权和排空条件后复用，整段运行容量的保守检查是：

```text
E_kernel = max(E_U0, E_U1, E_U2, 选中的E_U3a或E_U3b, E_U4)
```

这不是这些阶段的和。共享 GMM1 若启用，应将其256 KiB布局也放入 max；
不同物理核、不同任务和分支的 E 不一定相同，应取需要支持的最大值。
这个 max 公式仅用于当前阶段复用模型，**不能证明未来改成并行直通后的容量安全**。
例如 AIC 正往 UB 写 GMM1、同一 AIV 同时 Dispatch 时，需要计算两者同时存活区间的并集并保证不重叠，
不能继续只取 `max(Dispatch,GMM1)`。

## 4. 合并后的优化候选

以下为去重后的候选全集，保留 C01..C16 编号，并补入预处理批量化和维护类候选。
每项仅给出预期作用与成立条件；占用算术成立不代表硬件接口、精度或性能已经验证。
完整地址和生命周期以第 3 章为准。

### C01：按生命周期划分 UB，同时保留平台容量检查

详见第 3.3、3.9 节。

可以让 Quant、SendMask、Reset 在结束后复用同一片 arena，MoE 阶段另设小型常驻区。
收益是释放串行阶段互相占用的预算，供更大 batch 或更多在途数据使用。

但目前 Reset 的零值在 `SendAndQuantBuffInit` 就已经由 `Duplicate` 写好，
`ResetSyncStatus` 直接使用它，没有重新清零。这个零值事实上跨 Quant/SendMask 存活。
若复用零值区，必须把清零移到 Reset 真正开始时，并在切换 arena 前回收上一阶段异步操作；
Reset 结束后也要等待 MTE3 不再读取该零值区。也可保留独立 Z 区，仅复用其他 scratch。
两者的容量公式见第 3.3 节，不能只改地址而继续使用被覆盖的零值。

共享输入准备已经复用 Quant 的两个输出 buffer，不是第四套完全独立的存储。
这项已有复用应保留在容量计算中。

### C02：SendMask 只计数，不物化无用的 Gather 结果

源码：[SendMask Compare/Gather](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_send_mask.h:86)。

当前 CompareScalar 生成 bitmap，GatherMask 把匹配的 expert ID 写入 `sendGatherOutTensor`；
后续只使用 `batchMatchedRouteCount`，没有消费者读取 gather 输出数组。

候选是对有效 bitmap 位做向量计数，或使用可验证的计数接口，省去 Q 个 int32 的结果区。
基准 Q=12288 时该区域为 48 KiB；自适应 Q 更大时释放更多。

当前六槽 mask 下，两个 int32 数组加 bitmap 的变量预算约为 `8.75Q` 字节。
去掉 gather 数组后约为 `4.75Q + 小型计数 scratch`。可把余量给更大的 Q，或给双份 topkIds 预取。
不能只给 GatherMask 一个小目标区而仍让它写 Q 个结果；也不能改成逐 bit scalar 循环。
具体向量计数指令是否可用仍需查当前 SDK，尾 batch 只计 validLen。

### C03：Dispatch prefix 有三种存储候选

详见第 3.4 节。

当前每个专家恢复旧 prefix 并写回更长 prefix。设 R 为 rank 数、E 为本卡专家数，
单 AIV 的 prefix 搬运合计约为 `4 * R * E^2` 字节，另有最终导出 reload。

候选 A：完整表常驻 UB 顶部，大小 `Align32(4RE)`，上限 8 KiB。
常驻数据跨越的各阶段布局都必须避让，Host 不能既在 fixedBytes 中计算一次又在尾部重复扣一次。

候选 B：仅保留当前专家的 R 个 prefix，加一个专家起始偏移。
处理当前专家只需当前各 rank 边界和上一个专家总结束位置，不需要整张历史表；
最终专家计数可从已写入的 `expertRevNumsGlobalTensor` 导出。
最大当前表约 4 KiB，且不必跨 activation 长期保留所有历史 prefix。
该候选需要调整当前按 `expertIdx * R` 索引的接口。

候选 C：保留 GM 表，但仅增量写本专家段，读必要端点或当前窗口。
可在不占常驻 UB 的情况下消除近似二次的 prefix 流量。

这三种方向有不同复杂度，都应保留，而不是默认只有“全表常驻”和“整表反复 spill”。

### C04：Dispatch 的 bitmap/索引与搬运解耦

源码：[Dispatch 扫描](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:292)、[拷贝 ring](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:92)。

当前每个 rank shard 从 bitmap 起点累计匹配序号，找到自己负责的匹配区间。
多个 shard 可能重复扫描前部 bitmap。接收侧的 gather 结果确实用于定位 token，不能照搬 C02 删除。

候选包括：缓存一批索引供多次消费、预取下一批 bitmap、构建更小粒度的匹配计数目录，
或者在同一源 rank 下批量搬入多条 token 再集中写 data/scale/metadata。
目录也可在 GM 发布而在 UB 缓存；收益取决于路由稀疏度、rank 数和额外同步成本。

当前 CopyTokensAndMetaForDispatch 已有提前一条 Fetch 的流程及 per-slot 事件，不能当作完全没有流水。
bufferCount 上限 6 是当前实现常量，扩展它还要审计 event 占用，不能直接称为硬件总上限。

### C05：GMM1 的 256 行整块与 128 行双 bank 并列

详见第 3.5 节及 [GMM2 输入 ready 目标](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:479)。

单路 `256*256*BF16` 为 128 KiB，Gate+Up 合计 256 KiB。
top-k 模式实际搬入的是每 token 一条 32 B metadata，其中含选中的权重，
不是每 token 的全部 topK 权重；256 行时增加 8 KiB，共 264 KiB。

候选 A：把 top-k epilogue 从 128 行恢复为 256 行，减少子调用和固定开销。
候选 B：保留外层 256 行，用两组 128 行 Gate/Up bank 搬运与计算重叠。
两组数据共 256 KiB，加双 metadata 8 KiB 和最大 cumsum 8 KiB，共 272 KiB。
epilogue 的 scratch 若在各自 bank 内复用，需要允许传入 bank 基址。

必须联动 GMM2 的 ready 计数：
`FinalizeProblemConfig` 的 `activationTileM` 当前 top-k 固定为 128；
`WaitForGmm2InputReady` 目标为 `ceil(waveM/activationTileM)*ceil(K2/256)`。
例如 waveM=256、K2=512，旧目标为 4；改成 256 行生产后只发布 2。
如果只改 epilogue 两处常量，GMM2 会一直等不到 4。
反向采用 128 行消费也要同步增加生产、消费两侧计数。

### C06：GMM1 UB 直通，同时评估更小单块的完整双缓冲

详见第 3.5 节；[tensor UB 输出接口](/D:/gitcode/ops-tensor/include/blaze/gemm/block/block_mmad_qbmm_mx.h:214)。

普通 GMM1 当前走 GM，完整 Gate/Up tile 的逻辑本地往返为 512 KiB。
可保留 GM 队列、改成单槽 UB、三 slab 旋转、只让一路直通的 hybrid，或采用两组完整 Gate/Up。

两组完整缓冲不一定只能取 M=128。固定 N=256 的容量如下：

| M | 一组 Gate+Up | 两组 Gate+Up | 加双 metadata 和最大 cumsum |
| --- | --- | --- | --- |
| 128 | 128 KiB | 256 KiB | 272 KiB |
| 160 | 160 KiB | 320 KiB | 338 KiB |
| 176 | 176 KiB | 352 KiB | 371 KiB |
| 192 | 192 KiB | 384 KiB | 404 KiB |
| 256 | 256 KiB | 512 KiB | 536 KiB |

此表假设量化 scratch 复用各 bank 已消费区域；不证明任何 shape 已获得硬件支持。
M=160/176 比 M=128 大，又能容纳两整组，是介于128和192之间的容量候选。
其代价是更多 tile、可能重复加载权重，以及修改 256 行 flag 粒度或引入细粒度子 tile 映射。

所有 UB 直通方案都需要生产 credit、消费完成 ACK、bank 基址和退出 drain。
普通 MoE 的 AIV 还会执行下一段 Dispatch，AIC 必须先取得 UB 所有权才能写入。
只有一个“输出 ready”通知不足以防止覆盖，必须计入消费者落后和跨阶段切换。

三 slab 候选的具体次序为：当前 Gate+Up 占 slab0/1，AIC 提前把下一 Gate 放入 slab2；
当前激活及量化发送释放区域后，再把下一 Up 放入可用 slab，AIV 随后消费旋转后的配对。
每块128 KiB，三块正好384 KiB，没有额外 metadata/cumsum 余量；量化 scratch 必须在已消费 slab 内复用。
共享路径没有 top-k 区，且发生在 Dispatch prefix 存活之前，适合独立评估，但仍不能用容量推断接口或性能已验证。
普通路径可保留 GM handoff，或只让 Gate/Up 一路直通（hybrid），在 GM 往返与 UB 余量之间取舍。

### C07：GMM1 更大的 M/N，或者只增大输出组织块

可以保留 M=320/352/384、N=384/512 等候选，同时评估 `128x512`、`192x384` 或 small-M/large-N。
这些是不同的 shape 组合，不是上述 M/N 可任意交叉。
M=384、N=256 的单组 Gate/Up 为 384 KiB，单次 FP32 L0C 为 384 KiB，没有 top-k/cumsum 余量。
单组 M=320/352 则还分别剩 64/32 KiB，但是否允许需检查 A6 L0C、L0A/B、L0Scale 和指令限制。

当前 tensor operator 只在 K 上循环，M/N 的 actualShape 直接进入 L0C view 和 MMAD，
不能假设它会自动把大 M/N 分成硬件允许的小块。
还要检查 `M*K0`、`N*K0` 的 FP8 L0 占用及 L1 数据/scale 总容量，不能只算 L0C。

另一候选：调度层或输出 bank 组织为较大块，Cube 仍按当前受支持的子块计算。
这能减少部分调度/epilogue 开销，却不扩大单条 MMAD 的 M/N。

专家实际 M 在设备 Dispatch 后才知道，当前 Host 没有读取路由内容得到直方图。
真正按实际 M 自适应应放在设备收到计数后；Host 只能根据输入 shape、属性或外部统计选择策略。
外层 tile 改变时需要 AIC/AIV 和 ready 协议一致，不能仅让 AIC 自己换调度 shape。

### C08：让 activation 与 MX quant 在寄存器内按组融合

源码：[激活与量化调用次序](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h:639)。

当前先把 BF16 activation 写 UB，再读它计算最大指数，再读它量化。
候选是在同一向量计算段中完成 activation、BF16 舍入、32 元素组 scale 和 FP8 输出，
或先采用较小的片内 scratch 分段处理。

可减少 BF16 中间结果的 UB 写和重复读，以及阶段间屏障；也可能缩短输入区域存活时间。
必须保留现有 BF16 舍入点、top-k 乘法位置、clamp 和 NaN/Inf 处理。
直接用未舍入的 FP32 activation 求 scale 会改变精度语义。
该方案是向量计算改造，不应归因于单纯增加 UB，也不能保证寄存器压力增加后仍更快。

### C09：非量化 GMM2 的 GM 双 bank 或 UB 直通双 bank

详见第 3.6 节。

当前每个 256x256 BF16 tile 先写本地 GM，再加载到 UB 发送。
UB 直通可以减少每完整 tile 256 KiB 本地逻辑往返。

两个输入 bank 共 256 KiB；若 metadata 也双缓冲 16 KiB，加 cumsum 8 KiB，共 280 KiB。
如果 metadata 只保留一份，则需要在它复用前完成相关 Scalar 消费，不能声称能任意预取。

GM 分支目前给每行分配独立结果区，AIC 可以先于 AIV 推进；改成两个 UB bank 会增加背压。
应比较 GM 双 bank 预取、UB 单槽、UB 双槽在完整 wave 中的收益。
不能因为删除了 GM 流量就认定整算子一定更快。

### C10：量化 GMM2 按 N tile 直接量化发送

源码：[MXFP8 量化](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_mxfp8_utils.h:42)、[分组归约](/D:/gitcode/ops-transformer_910b2/mc2/moe_distribute_dispatch_v2/op_kernel/quantize_functions.h:131)。

MXFP8 这里按连续 32 元素分组，不是整条 H 共用一个 scale。
N tile=256、nLoc 为 256 的倍数时，每行该 tile 恰好包含 8 个独立 scale group。

因此可以让配对 AIV 直接消费 GMM2 UB tile，在 tile 内量化，并写入远端 token 记录对应的 data 和 scale 段。
数学上不需要等其他 N tile 完成，也不一定要改变各 N tile 的 AIC 归属。

设某 token 的远端记录起点为 base，完整行宽为 H，则：

```text
data 目的地址  = base + nLoc
scale 目的地址 = base + Align256(H) + nLoc / 32
完整 N=256 tile: 每行写 256 B data + 8 B scale
```

保守示例预算：BF16 双 bank 256 KiB + FP8 输出 64 KiB + scale staging 8 KiB
+ 量化 scratch 16 KiB + 双 metadata 16 KiB + cumsum 8 KiB = 368 KiB。
这是可容纳的候选布局，具体 scratch、row pitch、补齐仍要按所选量化实现校验；
也可一次只量化若干行以缩小 scratch。

必须处理有效 scale 短写的硬件粒度，避免 padding 或相邻核写入覆盖其他 tile 的 scale；
不能把 tile 的 data+scale 当作独立整行记录连续发送。
同时保证尾 N、BF16 舍入、最终通信完成与 Unpermute 可见性。
改变后，本地“所有 N tile 完成才做量化”的等待协议可以重设计，整行数据最终仍在远端拼齐。

这项候选的主要代价是更多小粒度远端写、scale 短写限制，以及新的 AIC/AIV 背压。
可保留整行 GM 量化作为互斥候选，或由 UB 量化后先写紧凑本地 FP8 区再聚合发送。

### C11：保留整行量化时，解除确定存在的跨 token 串行

详见第 3.7 节。

这里每条 token 开头的 MTE3_MTE2 同步会等待之前的发送，再开始新读取；
量化后还有 V_S 等待。它比单纯看到一个 MTE2_V 更能说明依赖过强。

候选是两槽或三槽 per-slot 所有权、延后与计算无关的 Scalar 等待、批量 metadata，
也可用更大的 token batch 减少调用次数。
增加到三至六槽只能作为实验范围，不能认定三个流水阶段就必须分配三份完整数据。
当前 H=8192、metadata 覆盖 256 个 token 时，双槽加 metadata 和 scratch 共 58 KiB（精确公式见第 3.7 节）；容量本身不构成主要瓶颈，修改依赖关系才可能带来收益。

### C12：Unpermute 保留必要输入，减少中间 FP32 存储

详见第 3.8 节。

每个 input slot 当前预留 BF16 和 FP32 两份，H=8192 时合计 48 KiB。
候选是输入保留 BF16/FP8，转换、乘权重后直接累加到唯一 FP32 accumulator，
尽量不把每个专家的完整 FP32 中间值写回 UB。每去掉一份完整 FP32 input 区，可少用 4H 字节。
通信量化的 scale 展开也可在寄存器内广播，需核对 rounding 和寄存器压力。

额外 UB 可给输入预取、多 token accumulator、更多输出槽或权重批次，均可并列试验。
top-k 已在 activation 预乘时，Host 仍可能预算 Unpermute 的 weight 区，可评估按模式取消该预算。

当前输出 MTE3 之后，下一 token 先累加到独立 FP32 区，再等 BF16 output buffer 可复用，
因此“必须新增第二 accumulator 才能重叠上一输出与下一累加”并不成立。
输入 ring 已有 per-slot event，实际预取深度要看指令时序，不能仅从邻接的 Set/Wait 判零。

### C13：共享专家输入拆分融合到 Quant

详见第 3.3 节的 Quant 和共享输入准备源码。

Quant 输出 data/scale 尚在 UB 时，可同时写共享输入两块 GM 区，省去稍后的 GM-to-UB。
两种流程都需要 peer 记录和共享 dense data/scale 写回，因此不是把这两次必要 GM store 消掉。

需要在三路输出都完成后释放输出 buffer，并保持 scale 对齐。
普通 Quant 与准备阶段使用相同 token 分核，具备融合依据。
这是减少一次重读的候选，不依赖额外 128 KiB；也可保留准备阶段独立流水。

### C14：共享专家 GMM2 与 Unpermute 的局部融合

源码：[共享 GMM2 调用位置](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:396)、[Unpermute 共享结果读取](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h:149)。

共享 GMM2 在普通 MoE 之后执行，其结果不需要跨越前面的 MoE。
但当前共享 GMM2 按 M/N tile 分核，Unpermute 按 token 分核，不保证由同一 AIV 消费对应结果。

候选是按 token/N 子块协调共享 GMM2 和 Unpermute 的归属，先为当前子块累加 MoE，
再加共享结果并输出；小 batch 场景可以更积极地保持局部结果。
这涉及跨 rank ready、多个共享专家和累加顺序，不是简单把 sharedExpertResultPtr 改成 UB。
也可以让当前 GMM2 输出继续走 GM，只按子块提前触发 Unpermute。

### C15：元数据缓存和跨阶段调度是独立候选

GMM1 每个 N tile 都读取同一批 token metadata，Combine 也反复提取 rank/token/topk。
可以缓存当前 M group 的元数据，或仅保存选中权重（每行 4 B）与已算好的目的偏移。
是否复用取决于同一个 AIV 是否再次处理同一 M group，cache tag 必须包含 expert 和 row range。

保留 GM handoff 允许 AIC 在 AIV Dispatch 期间继续计算；UB 直通则需准备 credit。
可以按小 M、大 M、共享专家、top-k 模式选择不同 handoff，或给少量 AIV 专职通信。
后一方案需要权衡参与 Vector 计算的核数，384 KiB 本身并不能解决分核负载问题。

### C16：把 UB 优化与 Cube 内部串行区分开

当前 tensor 的 MX L0 operand 强制 slot 0，源码明确保留精度 fallback；
Transformer 另外关闭了 L0C ping-pong。这是两件不同的事。

可以在精度根因得到修复后单独评估恢复 operand 双缓冲，但不能在 UB 优化中顺便解除。
如果主要耗时在 L0 搬运/MMAD 串行，扩大 UB 可能只改善边缘阶段。
已存在的 A-full-load 等 L1 策略也可作为独立候选；切换专家、输入 A 或 wave 时必须失效缓存。
这些方向不属于 384 KiB UB 带来的直接容量收益。

### C17：预处理批量化、预取与 Reset batch

Quant 已有双输入、双输出槽；可以比较单 token look-ahead 与多 token 小批量量化。
共享准备也可保持独立阶段，重排 GM->UB 和两路 GM store，而不是必须融合到 Quant。
这两者与 C13 是可选关系：融合减少一次重读，独立阶段可能有不同的流水和代码复杂度。

Reset 当前最多用2048个 int32，即8 KiB零值 tile，反复写完整 GM flag 区域。
在 C01 的阶段复用成立后，可按可用容量扩大 batch，减少循环和搬运发起次数；
也可保持8 KiB，让位于其他常驻状态。增大 batch 不减少总清零字节数，不保证带宽受限时有收益。
Host 与 kernel 的 batch、对齐和最大 flag 数必须一起更新。

Dispatch/SendMask 可在更大 Q 与更深 ring 之间权衡；当前2..6槽是实现选择，不能当作硬件总事件上限。
Quant/Unpermute 的邻接 Set/Wait 本身不证明完全串行，验证应看发射顺序、目标流水与指令时序。

### C18：布局维护、边界检查与不可达 helper

可将固定地址、对齐和容量条件集中为 arch92 UB 布局策略，对共存区间增加编译期/Host边界核算。
这是减少后续布局修改出错的候选，不要求先搭建一套新框架才能做局部优化。

未调用的 Wave Combine helper 可删除或保留在显式功能开关后；当前它们不占运行期 UB，
不能把删除定义计为直接释放物理容量。需清理的是错误的容量假设和阅读负担。
性能型编译应统一关闭诊断，再衡量各候选，不在本次文档合并中改动现有打点。

### C19：非量化 GMM2 远端直写的接口探索

只有确认 L0C copy 能表达目标 peer 地址、逐行scatter/stride和远端可见性后，
才可探索 L0C->remote 直写，进一步减少本地GM/UB中转。
当前源代码和本次核对不足以确认该完整能力，不能列为可直接实施的局部修改。
若接口或通信所有权不满足，继续比较 C09 的 GM consumer 或配对 UB->MTE3 路径。


## 5. 取舍、共存与人工决策

### 5.1 候选之间的关系

| 目标 | 并列选择 | 核心取舍 / 依赖 |
| --- | --- | --- |
| prefetch激活 | 256行整块 / 两个128行bank / 更小块融合 | 调用次数、重叠机会与ready粒度；见C05/C08 |
| GMM1交接 | GM / UB单槽 / 完整双bank / 三slab / hybrid | GM领先量与UB背压；普通AIV还执行Dispatch，见C06 |
| Dispatch prefix | 全表常驻 / 当前专家窗口 / GM增量更新 | UB持久容量、接口改动和流量；见C03 |
| GMM2量化发送 | GM整行量化 / tile量化 / 本地FP8聚合后发送 | 小粒度远端写、scale写粒度和最终可见性；见C10/C11 |
| 额外容量分配 | 更深ring / 更大batch / metadata缓存 / 常驻状态 | 必须重新代入Host预算，不能把128 KiB重复分配；见C15/C17 |
| Unpermute | 输入look-ahead / 去FP32中间区 / 多token累加 / 更大weight batch | 数据精度、寄存器压力、输入输出生命周期；见C12 |
| 共享输入 | 融合到Quant / 保留独立准备流水 | 消除重读与改变Quant多路写出依赖；见C13/C17 |
| 更大shape | 扩大单次MMAD / 仅扩大调度与输出组织块 | L0/L1限制不同；先核查接口，见C07 |

完整384 KiB三slab与同一生存区间内的cumsum、top-k或其他常驻区冲突。
普通和共享可以用不同策略；完整双bank也可取128/160/176行，不必只在128和256之间二选一。
这些是允许保留的方案分支，不应把某个容量示例当作最终推荐配置。

### 5.2 当前数据复用关系

| 两种数据 / 阶段 | 当前关系 | 人工修改时的判定条件 |
| --- | --- | --- |
| Quant scratch 与 SendMask scratch | 顺序执行，但当前落在不同地址 | 可讨论时分复用；先处理 Reset 零值的跨阶段存活和所有异步操作 |
| reset 零值与 Quant/SendMask | 当前必须同时保留零值 | 未移动清零时机前，不可覆盖；Reset 返回后还要等 MTE3 读完 |
| Quant 输出与共享输入准备 | 已共用地址，不保证共用内容 | 保留现有双槽；若消除 GM 重读，需要改变生产消费流程 |
| Dispatch cumsum 与 Gate/Up | 同址覆盖，靠 GM 恢复 | 常驻须重定位且覆盖所有普通 wave；只删回写不成立 |
| Dispatch ring 与普通 GMM1 输出 | 当前 UB 分时，待消费 GMM1 结果在 GM | 改 UB 直通后可能同时需要空间，必须增加明确 ownership/credit |
| Gate 与 gluRes | 已原地复用 | 按当前激活读写顺序成立，不可跨 tile 提前覆盖 |
| Up 与 FP8/scale/scratch | 已跨子阶段复用 | Up 被读完后还有新的量化用途，并非可给下一 tile 的空闲区 |
| metadata 与 Gate/Up | P=1 时同时存活，地址分离 | 加 tile 或 bank 必须同时考虑 metadata，不能只算矩阵 |
| 普通和共享 GMM1 | 不同阶段复用同一组地址 | 不相加；若改变调用并行关系则重新评估 |
| 非量化 Combine 的72 KiB空洞与 cumsum | 仅在该 Combine 阶段空闲 | 无法直接作为跨激活常驻区；激活会覆盖它 |
| 量化 Combine 两槽与 floatTemp | 两槽独立，scratch共用 | 增槽不自动要求复制全部scratch，但必须确保共用时无并发冲突 |
| Unpermute 普通与共享输入 | 同一输入 ring轮转 | 不按共享专家数再倍增；共享 BF16 最大行仍限制槽大小 |
| Unpermute FP32累加与BF16输出 | 地址独立，允许跨 token重叠 | 合并会改变流水，必须等上次输出结束才能覆盖 |
| Unpermute weight与其他区域 | P=0为活跃批量数据；P=1仍占布局但未消费 | P=1可作为回收候选，需同步更改Host与kernel后续基址 |
| 任意UB区与跨AIC复用输入 | UB有本核归属，GM负责多核共享 | 不能仅凭容量足够就将多消费者GM张量全部删掉 |

“把值常驻 UB”优先应评估小型、同一 AIV 后续重复使用的数据（例如 cumsum）。
Dispatch 输入、激活量化输出等可能被多个 AIC/多个 N tile读取，
若想常驻还必须解决读者归属和访问途径，不能把“省GM往返”和“多出128 KiB”直接画等号。

### 5.3 人工修改核对清单

1. 明确改的是哪个模板：普通/共享、prefetch开关、Combine量化开关、DN/NZ，以及外层shape与BlockMmad内部shape。
2. 从实际tiling取 `Q_s/B_s/Q_d/B_d/T_u/B_u`；记录核类型和是否尾任务，不能只看一个核的配置。
3. 为每个buffer写出 `[base,base+size)`，标注元素类型、对齐和行距；计算最高地址，不只相加有效元素。
4. 为每份数据记录“第一次写、最后一次读、最后一次异步读/写结束”；对象作用域不是释放时刻。
5. 将alias明确标成同一物理区；既不要重复加容量，也不要把已经有下一用途的区当空闲。
6. 保留跨阶段状态时列出它跨越的全部阶段，并检查与固定地址、动态batch、AIC写入的碰撞。
7. 修改bank或ring时同步检查data/scale/metadata/event编号、跨核ready/ACK，不能只改数组长度。
8. 修改128/256激活粒度时同步更新GMM2输入ready目标；修改N tile时同步更新Combine/shared ready计数及layout。
9. 扩大shape还须核查L1/L0C容量、BlockMmad接口范围和输出步长；384 KiB UB预算不是Cube可计算性的证明。
10. 先做离线布局和边界核算，再选最少的运行用例；无需为容量评估新增全量scalar hash或大量set_cond。

建议人工记录格式：

| 候选 | 保留/删除的原buffer | 新base/size | 必须共存的数据 | 生存区间结束依据 | Host需改项 | kernel需改项 | 未验证条件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 对应event/ACK/阶段drain | 预算或配置 | 地址/步长/同步 | 硬件限制、尾块等 |

本次合并不自动选择或实施候选；填写上述记录后再决定需要修改的代码和验证边界。

## 6. 验证计划与当前状态

### 6.1 当前状态

| 项目 | 状态 |
| --- | --- |
| 两份方案的事实纠正、去重与候选合并 | 已完成文档整理 |
| 六套UB布局的地址/对齐/容量/生命周期账本 | 已列出，示例算术已核算 |
| C01..C19中的具体实施选择 | 待人工评估，没有因本次合并自动选定 |
| kernel/Host/tensor实现整改 | 本次未实施，现有源代码保持原样 |
| A6构建、完整精度、性能验证 | 本次未运行，不宣称候选已提速或已获接口支持 |

### 6.2 验证按候选缩小范围

模拟一次约半小时，先做静态或本地小验证，再选择真正影响结论的运行用例。

| 候选类别 | 先检查什么 | 运行时关键边界 |
| --- | --- | --- |
| arena / 常驻 / ring | 区间交集、最高地址、tail配置、padding、零值初始化、Host/kernel公式一致 | 零任务核、阶段切换、最大H/专家数/route batch、ring回绕 |
| 128/256 epilogue或更大shape | MMAD接口和L0/L1预算、Gate/Up偏移、行距、producer/consumer ready目标 | M=127/128/129/255/256/257及所选新边界，N尾块 |
| UB直通 | credit、ACK、bank所有权、MTE3完成回收、Dispatch覆盖区 | 消费者落后、专家/wave切换、bank回绕、零tile核 |
| tile MX量化 | 整行与分块data/scale离线比较、BF16舍入、scale短写和padding归属 | 尾N、相邻tile并发写、远端记录拼齐与Unpermute可见性 |
| 向量融合/Unpermute | BF16/FP32舍入点、top-k乘法位置、clamp、NaN/Inf、累加次序 | 多专家、特殊值、通信量化与共享结果叠加 |
| 共享输入融合 | peer记录与共享dense数组内容、scale步长、三路MTE3输出回收 | shared开关、prefetch开关、多token及尾核 |
| 计数替代GatherMask | 向量计数接口、validLen、尾bit及目标区真实写量 | 空mask、稀疏/稠密路由、最后一批 |

模式覆盖按修改的实际范围选择：TopkWeightsPrefetch开/关、E4M3/E5M2、Combine不量化及两种FP8通信量化、DN/NZ权重、共享专家开/关。
不要求每项小改都跑全部组合，但共享接口或同步协议的变化不能只验证一条满块路径。

### 6.3 性能口径

记录 Quant、SendMask、Dispatch、GMM1、activation、GMM2、Combine、Unpermute 的阶段cycle和整算子cycle。
同时区分逻辑GM搬运字节、缓存命中、实际HBM/远端流量以及等待时间；减掉GM字节不等于同比例提速。
两个bank的背压、更多小远端写、寄存器压力和核利用率都可能抵消局部收益。

统一关闭诊断后比较性能。不要为容量评估引入全tensor scalar hash、逐bit scalar循环或大量set_cond。
必要时只对问题边界保留少量bank/sequence/窗口记录，并删除已排除阶段的打点。

## 7. 关键源码索引

本文件中的链接指向源仓或本地tensor仓；行号以当前文档基线为辅助，应同时按函数名定位。

| 主题 | 源码入口 |
| --- | --- |
| 1C1V入口、普通wave和共享阶段 | [kernel入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)、[Wave调度](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243) |
| 架构交接策略 | [USE_GM_HANDOFF/PERSIST_DISPATCH_CUMSUM](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:19) |
| 启动、Dispatch、Unpermute UB布局 | [DispatchBuffInit](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:374)、[SendAndQuantBuffInit](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:464)、[InitTokenUnpermuteBuffers](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:551) |
| Host自适应预算 | [SetAdaptiveBufferConfigs](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:975) |
| Gate/Up、量化scratch别名 | [epilogue布局](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h:213) |
| GMM1消费与GMM2 ready | [GMM1 prefetch消费](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:345)、[GMM2输入等待](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:479) |
| MX FP8分组 | [QuantMxFp8](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_mxfp8_utils.h:42) |
| 量化/非量化Combine | [CombineTokenGroup](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:172)、[GM handoff](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:697) |
| Unpermute生产消费关系 | [ProcessTokenUnpermuteBatch](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h:201) |
| 独立的Cube精度fallback | [operand slot0](/D:/gitcode/ops-tensor/include/blaze/gemm/block/block_mmad_qbmm_mx.h:504)、[L0C ping-pong开关](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:350) |

## 8. 合并范围说明

原候选排查中的shape、handoff、三slab/hybrid、常驻、启动阶段复用、各stage流水、共享输入融合、Reset batch、metadata与远端直写探索均已归入C01..C19。
源码复核新增的向量计数、prefix窗口/增量写、160/176行双bank、激活融合、tile MX量化、Unpermute中间区消除和共享GMM2局部融合也已保留。
逐阶段UB账本原有的地址、公式、示例、别名、生命周期与人工检查项分别整合在第3、5章。

被纠正的旧算法和流程描述不再作为候选前提；不保留要求读者自行在两份版本之间判断的冲突结论。
本文件是后续UB方案评估的唯一维护入口。

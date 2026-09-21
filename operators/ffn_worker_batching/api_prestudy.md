# API 预研报告

## 范围与证据

范围为 `ffn/ffn_worker_batching/op_kernel`，重点覆盖 `HEAD~4..HEAD` 的 arch35 增量，并对照原有路径。本文是 API 约束预研，不等于硬件正确性验收。

已读取 `npu-arch` 与 `ascendc-docs-search` 技能。环境未设置 ASC_DEVKIT_DIR，本机可查 SDK 为 `/home/chenpengyi/Ascend/cann-9.2.0/x86_64-linux/asc`；该目录未发现 API Markdown 文档，因此以下具体实现证据来自此 SDK 的官方头文件，不能误称为已逐项核实在线手册。

在线检索成功定位 CANN 9.2.0-beta.2 的 [DataCopyPad UB→GM 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.md)。正文抓取工具因缺少 bs4 失败。未取得正文的 API 参数上限不在本文臆定；后续条款检查须结合具体重载、SDK 检查实现与目标设备验证。

## 芯片代际

- 分析代际：新增路径为 DAV_3510 / arch35 / Ascend950；原有非 arch35 路径须分别适用原平台约束。
- 判定依据：`op_kernel/arch35`、`ffn_worker_batching_apt.cpp` 以及 Host 的 ascend950 配置。npu-arch 明确 `__DAV_C310__` 对应 DAV_3510，不能按宏中的 310 推断芯片。
- 代际特有能力：RegBase 寄存器向量编程、FP8/MXFP8/FP4 存储与运算能力。本算子的 token 搬运不执行 FP8/FP4 量化运算，只保留已经量化的字节。
- UB 大小须由 Host 平台查询；不能把文档中的典型 248 KiB 直接当作所有配置可用容量。

## 数据搬运

### DataCopy

- 常用形式：`DataCopy(LocalTensor<T>, GlobalTensor<T>, count)`；count 是元素数，参数结构重载不能与此混淆。
- 代际差异：普通块搬运仍按 32B 单位组织；非对齐长度应选相应 DataCopyPad 重载。
- 位置：`arch35/ffn_wb_a5_context.h` 的 ScheduleContextParse 使用 uint8 缓冲，因此 SCHEDULE_CONTEXT_BYTES 同时也是元素数。
- 搬入完成后进行 GetValue 的标量读取必须建立 MTE2→S 依赖，当前调用有显式事件。

### DataCopyPad / DataCopyExtParams / DataCopyPadExtParams

- 使用形式：`DataCopyPad(LocalTensor<T>, GlobalTensor<T>, DataCopyExtParams, DataCopyPadExtParams<T>)` 与反向三参数形式。
- 实现证据：SDK `impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:55` 的 CopyGmToUbufAlignV2 与 `:101` 的 CopyUbufToGmAlignV2。
- 此重载 blockLen 是**字节**；GM 侧 stride 是**字节间隙**；Normal 模式 UB 侧 stride 是 **32B 间隙单位**，实际下一 UB 块起点按 32B 向上对齐。不能把两个方向的 stride 字段一概当字节。
- padding 左右数量按 T 元素计；例如一个 int32 flag 加 7 个 int32 padding，占完整 32B。跨 session 读取 flag 时 GM 行跨度是 M×descriptorBytes，而 UB 行跨度是 32B，两者有意不同。
- blockCount 的字段为 uint16，blockLen/stride 的字段转换为 uint32；Host 必须在转换前约束范围，字段类型最大值也不代表所有硬件重载均接受该最大值。
- DataCopyPad 支持 GM 的非对齐短行；UB 起点、分块偏移及容量仍须满足相应对齐。UB 尾部补齐不代表允许向 GM 多写：输出 blockLen 必须只含有效字节。
- 代际差异：本文确认的是 DAV_3510 Normal 模式实现；不可照搬 Compact 模式或旧代际的附加限制。
- 位置：prepare 的轮询/快照/描述符清理、gather 的 token 和 MX scale、sort 的段数据和计数、group_list 分块写出。

## 内存管理与同步

### InitBuffer / AllocTensor / FreeTensor / EnQue / DeQue

- InitBuffer 的容量为字节，TQue 分配须乘队列深度；所有 TBuf、padding、辅助计数缓存均纳入 UB 峰值预算。
- AllocTensor 返回 LocalTensor；常规流水遵循 Alloc→生产→EnQue→DeQue→消费→Free。只 Alloc/Free 而未通过 EnQue/DeQue 的自管路径，不能假定获得任意 MTE/V/S 间依赖。
- TBuf 没有队列的消费者完成保护；复用它之前必须等旧消费者完成。例如 merge 计数缓存的 S→MTE2、MTE2→S 两端依赖有明确必要性。
- 代际差异：队列实现及事件资源随代际变化，但生命周期、生产消费依赖和 UB 预算原则不变。

### SetFlag / WaitFlag / FetchEventID

- 本代码 SetWaitFlag 包装按 HardEvent 获取 event ID，并在同一类型上配对 SetFlag/WaitFlag。
- MTE2_S：DMA 写 UB 后允许标量读；S_MTE2：标量读完成后允许 DMA 覆盖；V_MTE3：向量写完后允许 DMA 搬出；MTE3_V：DMA 读完后允许向量覆盖；MTE3_MTE2：DMA 搬出读完后允许下一次搬入覆盖。
- 事件是有方向的。MTE3_MTE2 不能直接作为 MTE3_V 的替代，除非能证明另有完整的依赖传递链。
- 代际差异：DAV_3510 支持相关接口，但不应将其他代际事件实现细节推广到本平台。
- 待逐条审计：ProcessRecv 的未选中行分支在 MTE3_MTE2 后 Free，下一未选中行将 Duplicate 同一缓冲；正常行循环尾部还涉及 clrBuf 的重复向量写。需核实事件/队列是否完整保护 MTE3→V 复用，不能仅凭有一个 wait 就认定安全。
- 补充编译证据：`op_host/CMakeLists.txt:22,31` 开启 `--cce-auto-sync=on`；本机 bisheng 的帮助明确其用于 memory-based operations 自动同步。因此上述手工事件覆盖疑点不能直接认定为实际竞争，更不能据此盲目添加事件。尚未获得目标 kernel 的 .s/.ll，实际分支是否补足须核查生成代码。

### PipeBarrier

- `PipeBarrier<PIPE_V>()` 用于向量指令间顺序，不能替代 MTE2→V、V→MTE3 等跨流水依赖，更不能替代跨核屏障。
- 代际差异：屏障具体实现可变，作用域仍须按模板参数理解。

### SyncAll

- 全部参与核须以相同顺序抵达对应屏障；局部无任务核也不能跳过仍有全体核参加的阶段屏障。
- SDK `dav_3510/kernel_operator_sync_impl.h:118` 中 SyncAllImpl 对 trigger/wait pipeline 建立屏障并使用跨核同步指令。
- RunPrepare 入口屏障防止 core 0 推进 polling_index 时晚启动核仍在读取旧上下文；发布屏障保证其他核读取完整 ready 快照；prepare/sort/merge/extract 阶段屏障保护跨核 GM 生产消费。
- 代际差异：DAV_3510 的 SyncAll 支持/实现不能从旧路径软件轮询推断。还需核对本工程选择的默认 isAIVOnly 与实际 launch 一致。

### TPipe::Reset

- SDK `impl/basic_api/kernel_tpipe_impl.h:612` 等待已登记的 free-buffer 事件并重置本核资源池。它不等于跨核发布屏障。
- 不应将其描述为纯指针复位，也不能用它替代针对未登记 TBuf 消费依赖的审查。
- 代际差异：资源回收实现与平台有关；跨核可见性仍由同步协议建立。

### GetValue / SetValue / DataCacheCleanAndInvalid

- LocalTensor 的 GetValue/SetValue 是标量访问 UB；必须分别与 MTE/V 生产者或消费者建立方向正确的依赖。
- 原有非 arch35 路径包含 GlobalTensor 标量访问和 DataCacheCleanAndInvalid：单 cache line 操作只覆盖指定行，不会替整个大数组刷新。
- 新 arch35 轮询采用 GM→UB 的 MTE2 搬运，再读 UB；不经过相同的标量 GM cache 访问路径，因此不能机械套用“每次轮询必须 DCCI”结论。外部生产者发布 token/ID 后再置 flag 的可见性契约仍须成立；本核事件本身不能约束外部生产者。
- 代际差异：缓存层次与指令语义随代际变化；数据缓存清理、DMA 顺序、跨核屏障是不同机制。

## 向量与精度

### Cast

- 使用形式：`Cast(dst, src, RoundMode, count)`。代码显式指定 CAST_ROUND 或 CAST_FLOOR，不依赖默认舍入。
- int32→float 对 |x|≤2^24 的整数可精确表示，超界奇数不可恢复。排序键为小范围 expert ID，与 gather 原始扁平位置不是同一个范围。
- 当前大容量 gather 在转换之前改走整数商余数路径，避免索引丢位。float 倒数乘法即便输入整数可精确表示，商仍可能受舍入影响，原有商余数修正仍有必要。
- 代际差异：支持的 dtype/舍入组合按 DAV_3510 重载核查；这里不发生 token 的 BF16→FP8 转换。

### Add / Sub / Muls / Adds / Maxs / Mins / Duplicate

- 按 count 调用和按 mask/repeat 调用具有不同参数语义；传统 repeat 上限不能直接套在 RegBase 的 C++ 循环次数上。
- masked Duplicate 的目标必须从满足对齐的 repeat 起点定位，mask bit 与该起点的 lane 对应。新增尾部初始化以 64 个 FP32 为 repeat 退回起点是为避免 keys[count] 未对齐并保证 mask 映射。
- 代际差异：Buffer API 与 RegBase 重载分开核查，不能只按同名函数匹配。

### ReduceSum

- 使用 float 累加 ready flag；padding 必须填 0，协议中的有效 flag 只能为 0/1，才能把和解释为 ready session 数。
- 返回 UB 标量被 GetValue 读取前需要 V→S。当前调用显式存在。
- 原有 `ReduceSum(work, work, work, count)` 的输入/输出/临时区同址兼容性应继续对照该 SDK 的实际重载检查，本文未取得在线正文，不将其默认认定为安全或缺陷。
- 代际差异：临时区及重叠许可属于具体 API 变体约束。

## 代际专属 API

### GetBlockIdx / GetBlockNum / GetTaskRation

- 已直接确认 SDK `dav_3510/kernel_operator_sys_var_impl.h:65`：AIV 的 GetBlockIdxImpl 返回 `get_block_idx()*get_subblockdim()+get_subblockid()`，CPU debug 分支也进行相同展平。
- 因而再乘 ratio 再加 subBlockIdx 会重复展平，新增 GetFlatAivIdx 返回 GetBlockIdx 有直接依据。总核数必须与此扁平坐标系一致，核查 launch block 数和 GetTaskRation 的乘积。
- 代际差异：该结论特指 DAV_3510，不能因旧代际用法而否定它。

### RegTensor / MaskReg / UpdateMask / Reg::DataCopy / Compares / Select

- RegBase 在 __VEC_SCOPE__ 内处理寄存器向量；UB→寄存器加载和寄存器→UB 写回须与外部 DMA 建立依赖。
- UpdateMask 的 remain 为逐轮更新状态，尾轮只开启有效 lane；不能把之后的 count 对齐处理误认为所有 API 已初始化尾部。
- EncodeLayerExperts 先把 int32 的位模式按 uint32 比较，负数转为大无符号数，连同 sentinel 被过滤；仅合法本层 expert 才加 layer offset，避免 INT32_MAX 加偏移溢出。
- 代际差异：这是 DAV_3510 RegBase 路线，旧平台不能直接复用。

## 量化存储契约

gather 以 int8 LocalTensor 搬运是位模式复制，不表示输出数值类型为 INT8。FP16/BF16 每行 2H 字节，INT8/FP8 每行 H 字节，FP4 每行 H/2 字节（Host 必须要求 H 可打包）。MX scale 为每 32 个逻辑元素一个 E8M0 字节，即 ceil(H/32) 字节；FP4 的比例因子分组仍按逻辑元素，不能再按字节 H/2 除以 32。INT8 dynamic scale 是另一个 float32 契约，不能与 MX scale 共用元素数推导。Python/Host 输出 dtype 必须在分配时确定，kernel 按字节写出不会改变 Tensor dtype。

## 未匹配核心清单但实际使用的 API

- CompareScalar/GatherMask：生成有效位并压缩；比较长度按 repeat 对齐时输入尾部也需初始化。
- Sort/Concat/MrgSort/Extract：proposal key/index 成对组织；Extract 的 repeat 对齐会读取 pair 尾部，新增初始化有必要；各重载的输入数/尾块要求应在逐条检查中核实。
- Gather/ArithProgression：构造索引或基于索引取数；索引类型、范围与有效 lane 必须一致。
- ReinterpretCast/SetGlobalBuffer/GetPhyAddr：只改变访问视图或绑定地址，不实施数值转换，也不替调用者做容量和生命周期校验。

## 日落 API

已运行 skill 的 clause.get_sunset_api.py，成功获得 CANN 920beta2 清单。针对本次 kernel 范围，未检测到清单中的日落 API/头文件使用。该结论不覆盖范围外的 Host/opapi 链接设置。

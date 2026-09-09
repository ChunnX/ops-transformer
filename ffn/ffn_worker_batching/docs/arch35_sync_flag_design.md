# arch35 syncFlag 需求、流程与核心代码解读

## 1. 背景与设计目标

本次为 Ascend950 / DAV_3510（arch35）的 FfnWorkerBatching 增加 `sync_flag` 属性。上游是 `attention_to_ffn_v2`，接收侧通过 `schedule_context` 中的设备地址访问 token 数据和就绪描述符。

原有 `need_schedule=1` 路径只检查 `polling_index` 对应的一个 micro batch，必须等 A 个 session 的 flag 全部为 1，才开始 batching。这要求每个 session 最终都给当前 FFN 发布 flag。

上游 [AttentionToFfnUrma::SetFlagToFFN](../../../mc2/attention_to_ffn_v2/op_kernel/attention_to_ffn_urma.h) 在 `syncFlag=1` 对应的模板分支中，会跳过发送状态为 0 的 FFN。因此，没有给某个 FFN 发送 token 的 session 可以不向它置 flag。如果接收侧仍等 A 个 flag，就可能一直等不到全部数据。上游模板名 `isSync` 容易引起误解；这里依据实际分支行为及 README，采用 **0=同步、1=异步** 的定义。

目标是：没有就绪描述符时继续等待；发现就绪描述符后处理这次扫描观察到的集合，不为其它 session 等待。不增加攒批时间窗口，也不要求达到最小 token 数。

## 2. 属性契约与边界

新增可选 INT 属性 `sync_flag`，ACLNN 参数名为 `syncFlag`，合法值为 0/1，默认 0。

| need_schedule | sync_flag | 执行行为 |
| --- | --- | --- |
| 0 | 0/1 | 原 NORM batching 路径，不读 flag，不推进 polling_index |
| 1 | 0（默认） | 原同步 RECV：只等入口 polling_index 对应 micro batch 的 A 个 flag 全就绪 |
| 1 | 1 | 异步 RECV：循环扫描 micro batch，首次发现至少一个 ready session 就处理该轮观察结果 |

属性追加在 `layer_num` 后，索引为 5，原有属性索引保持不变。公共算子声明需要更新，以生成图属性和 ACLNN 接口；新增调度逻辑只在 arch35 的 tiling/kernel 中实现，旧架构 kernel/tiling 不变。自动生成的 ACLNN C 接口在 `layerNum` 后增加 `syncFlag`，调用方须使用新头文件重新编译，不保证旧二进制 ABI 兼容。

关键约束：

- 同一次异步调用只消费一个 micro batch 中选中的 session，不跨 micro batch 合批。输出容量仍为 `Y=A*BS*K`。
- `flag=1` 表示描述符已发布，不表示一定有有效 token。已发布但 expert_ids 全被 mask 的描述符仍要完成消费，允许 `actual_token_num=0`。
- 未选中 session 不参与输出，其原始 flag、layer_id、expert_ids 和 token_data 均不被本次消费改写。选中 session 的 token_data 和 layer_id 也保持不变。
- 同一个 schedule_context 的消费者调用必须串行。生产者必须先完成 token/ids 写入，再发布 flag，并遵守槽位复用协议。
- 本次不增加设备内超时、退出信号、按 layer 筛选或跨 micro batch 合批。若始终没有 ready 描述符，异步扫描会持续等待。

## 3. 术语与内存协议

### 3.1 形状及描述符寻址

| 符号/字段 | 含义 |
| --- | --- |
| A | attention session 数 |
| M | 每个 session 的 micro batch 槽数 |
| BS | micro batch size |
| K | 每个 token 的专家槽数，与上下文 selected_expert_num 对应 |
| H | 逻辑 hidden size |
| F | 一个描述符的 int32 word 数，当前为 `2+BS*K` |
| polling_index | 下一次扫描起点，保存在 schedule_context 中 |
| selectedMb | 本次扫描最终选中的 micro batch |
| flatIds | 排序前的扁平 expert_id workspace |

描述符定义取自公共 `attention_ffn_schedule.h`，逻辑布局为：

```text
FfnDataDesc(session=a, micro_batch=m)
    int32 flag                 // 0：未发布；1：已发布
    int32 layer_id             // 本算子不改写
    int32 expert_ids[BS*K]
```

token_info 按 `[A, M, F]` 排列，所以该描述符起点的 int32 偏移为：

```text
descOffset(a, m) = a*M*F + m*F
flagOffset      = descOffset
idsOffset       = descOffset + 2
```

轮询固定 m 时，每次跨步读取 A 个 flag。相邻源 flag 相隔 `M*F` 个 int32；`DataCopyExtParams.srcStride` 不包含本次已复制的一个 int32，因此设为 `(M*F-1)*sizeof(int32_t)`。

### 3.2 为什么需要独立快照

若仅将等待条件从 `readyNum>=A` 改成 `readyNum>0`，随后仍读取并清理所有 session，就会把未发布或尚在写入的数据纳入输出。若各核自行读取实时 flag，又可能因生产者在两次读取之间置位而得到不同集合，导致读入和释放范围不一致。

因此 0 号核将**停止等待的那轮 DMA 观察结果**写入私有 workspace；prepare 和 release 均使用这份结果。这里的快照是一次跨步 DMA 观察到的集合，不保证 A 个 flag 是在同一物理时刻原子读取的。未被本次观察到的到达数据留待后续调用。

### 3.3 readyWs 布局

Host 的 `wsReady` 以 int32 word 为单位，相对 userWorkspace。kernel 乘 `sizeof(int32_t)` 得到字节地址。

```text
readyWs + 0              32B 头块：低 8B 存 uint64 selectedMb
readyWs + 32             32B：低 4B 存 session 0 的 int32 flag
readyWs + 64             32B：低 4B 存 session 1 的 int32 flag
...
readyWs + (A+1)*32       段末尾
```

flag 的块内尾部由 `DataCopyPad` 补零，因此 padded 数组的求和仍等于 ready session 数。沿用 padded 布局直接发布快照，避免再做一次压缩和索引转换。头块只写入并读取 selectedMb 对应的有效 8B，其余头块字节没有业务含义。

快照载荷大小为 `(A+1)*32B`。A=1024 时为 32800B；起点对齐另可能产生最多 28B 空隙。只有 `need_schedule=1 && sync_flag=1` 分配快照载荷，NORM 和同步 RECV 不访问该载荷。workspace 内容只用于本次调用，不承担跨调用的已消费状态记录。

## 4. 完整执行流程

```mermaid
flowchart TD
    A[各核解析 schedule_context 和 tiling] --> B{need_schedule=1?}
    B -- 否 --> N[原 NORM prepare]
    B -- 是 --> C[SyncAll：所有核先读完入口上下文]
    C --> D[0号核从入口 polling_index 扫描 flag]
    D --> E{sync_flag=1?}
    E -- 否 --> F{当前 micro batch 的 A 个 flag 全 ready?}
    F -- 否 --> D
    E -- 是 --> G{本轮至少一个 flag ready?}
    G -- 否 --> H[局部扫描索引加1并对M取模]
    H --> D
    G -- 是 --> I[保存 flag 快照和 selectedMb]
    I --> J[共享 polling_index 更新为 selectedMb 的下一项]
    F -- 是 --> J
    J --> K[SyncAll：发布扫描结果]
    K --> L[异步各核读取 selectedMb；prepare 依快照选择行]
    L --> P[SyncAll：flatIds 完整可用]
    N --> P
    P --> Q[原排序、归并、提取、group_list、gather]
    Q --> R{异步 RECV?}
    R -- 否 --> T[返回]
    R -- 是 --> S[SyncAll：所有核完成 gather]
    S --> U[依同一快照清所选 expert_ids，再清 flag]
    U --> T
```

图中“从入口扫描”仅指初始设置。异步空轮沿用更新后的局部扫描索引，不重新把它置回入口值；同步空轮则一直检查入口项。

### 4.1 读取上下文与扫描准备

`FfnWorkerBatchingA5` 调用 `ScheduleContextParse<isRecv>`。每个核各自获得上下文中的 A/M/BS/K、buffer 地址和入口 polling_index，并计算：

```cpp
ctx.asyncRecv = isRecv && tilingData->syncFlag == 1;
```

之后的第一个 `SyncAll()` 保证所有核都已读取入口上下文。否则 0 号核若提前推进共享 polling_index，起步较晚的核可能读到下一项，形成跨核索引不一致。

### 4.2 等待并选中一次扫描结果

`FfnWbA5RecvWait::Process` 只有 0 号向量核执行扫描，其它核在调用方的发布屏障等待。

```text
m = 入口 polling_index
重复：
    flags = 读取各 session 的 (session, m).flag，并补零对齐
    readyNum = sum(flags)             // 协议 flag 只取 0/1
    同步且 readyNum >= A：结束等待
    异步且 readyNum > 0：保存 flags 和 m，结束等待
    异步且没有 ready：m = (m+1) % M
    同步且尚未全 ready：继续检查同一个 m
共享 polling_index = (m+1) % M
```

必须跳过空 micro batch：异步上游不保证每一项都会被置位，在一个空项上一直等，会阻塞其它已经有数据的项。扫描绕回原项后可发现迟到的数据，无需为某个 micro batch 维护跨调用的“A 个 session 已处理计数”。

### 4.3 发布快照，统一各核当前索引

异步路径先把 flags 写到 readyWs 的 A 个块，再把 selectedMb 写到头块，最后推进共享 polling_index。各次需要复用 UB 的写入均等待 MTE3 完成；调用方随后执行 `SyncAll()`。

通过发布屏障后，各核从 readyWs 头块读取 selectedMb，覆盖自己的 `ctx.curMicroBatchID`。这里区分两个索引：

| 位置 | 含义 | 谁使用 |
| --- | --- | --- |
| readyWs 头块 / 各核 ctx.curMicroBatchID | 本次消费的 micro batch | prepare 寻址、gather 数据及 micro_batch_ids 输出、release |
| schedule_context.ffn.polling_index | 下次调用的扫描起点 | 下次消费者调用 |

因此不能在 prepare 中重新从共享 polling_index 取“当前 micro batch”。它已经是下一次调用的起点。

### 4.4 依快照构造 flatIds

`FfnWbPrepareArch35::Init` 按 session 行把工作分给各核。同步/NORM 沿用 host 给出的每轮行数；异步每轮处理一个 session，因为就绪行可以不连续，这个 1 表示选择粒度，不是硬件核数或容量。

RECV 中每个 session 的 flatIds 行跨度为 `align(BS*K, 8)` 个 int32，尾部补齐到 32B。异步分两种情况：

- **选中行**：读取该 session、selectedMb 对应的 expert_ids，补齐部分填 `MASK_SENTINEL`，复制到原 session 的 flatIds 行位置。
- **未选中行**：不读取其原始 expert_ids，只在 workspace 对应行填 `MASK_SENTINEL`。release 阶段也直接跳过该行。

`MASK_SENTINEL=2147483647`，大于现有排序过滤阈值 `expertStart=1000000`，所以未选中行不会进入有效 token 输出。保留原 session 行位置意味着 gather 仍可从原下标解码 session、token 和专家槽，不需要增加压缩 session 表。

### 4.5 复用原排序及输出流程

prepare 完成后，经 `SyncAll()` 进入原有流程：段内排序 → 多轮归并 → 提取有效 expert_id 和 gather_idx → group_list → gather。

本次没有改变排序、group_list、六种 token_dtype 的搬运和 scale 布局。selectedMb 通过 ctx 传给 gather，确保输出 micro_batch_ids 和 token 源地址指向本次实际选择的项。`actual_token_num` 取过滤后的有效 token 数，输出张量仍按原协议只使用有效前缀。

### 4.6 gather 完成后释放所选槽位

异步 prepare 的第一次调用使用 `release=false`，只构造 flatIds，不清原描述符。若此时清 flag，生产者可能开始复用 token 槽，而其它核仍在读取旧 token。

`RunEmitAndGather` 返回后，异步路径执行一个 `SyncAll()`，再以 `release=true` 调用 `ProcessRecv`。这是必要的，因为按 session 切分的清理核，不一定是负责搬运该 session 输出 token 的 gather 核。

release 使用与 prepare 相同的快照，执行：

```text
未选中 session：完全跳过
选中 session：
    expert_ids 全写 MASK_SENTINEL
    等待 ids 的 MTE3 回写完成
    flag 写 0
    等待 flag 搬出完成后复用 UB
```

先清 ids，再清 flag，避免生产者看见空闲槽后写入的新 ids 被本次清理覆盖。layer_id 和 token_data 不清理。即使 `actual_token_num=0`，也必须执行释放，以消费已发布但全 mask 的描述符。

同步路径仍在原 prepare 阶段清理所有行，没有改为异步的延迟释放时序。

## 5. 核心代码导航与解读

源码链接以文件为单位，函数名作为稳定定位点，避免补充注释后行号失效。

| 文件与定位点 | 主要职责 | 阅读重点 |
| --- | --- | --- |
| [ffn_worker_batching_def.cpp](../op_host/ffn_worker_batching_def.cpp)：属性注册 | 声明 sync_flag 默认值 | 公共注册不代表旧架构启用异步调度 |
| [ffn_worker_batching_tiling_arch35.cpp](../op_host/ffn_worker_batching_tiling_arch35.cpp)：ParseOptionalAttrs | 读取属性索引 5，拒绝 -1/2 等非法值 | 未提供属性时保留 0；NORM 也校验合法域 |
| 同文件：DoOpTiling / GetTilingKey | 下发 syncFlag，选择 NORM/RECV | key 仍为 100/101，syncFlag 与 tokenDtype 不增加 key 组合 |
| 同文件：SplitPrepare / LayoutWorkspace | 预算 UB 和 readyWs | 异步读取单行快照额外用一个 UB 块；workspace 偏移以 word 计 |
| [ffn_worker_batching_arch35_tiling_def.h](../op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h) | Host/kernel 共用 ABI | syncFlag 是策略值，wsReady 是空间偏移 |
| [ffn_wb_a5_context.h](../op_kernel/arch35/ffn_wb_a5_context.h)：ScheduleContextParse | 解释上下文并派生 asyncRecv | `isRecv && syncFlag==1` 保证 NORM 不进入异步逻辑 |
| [ffn_wb_a5_prepare.h](../op_kernel/arch35/ffn_wb_a5_prepare.h)：FfnWbA5RecvWait::Process | 扫描、保存快照、推进 cursor | Cast/ReduceSum 使用 work，不覆盖待发布的 flags |
| 同文件：FfnWbPrepareArch35::IsSelected | 查询冻结的选择集合 | 读取 `(row+1)*32B` 处低 4B，不读取实时 token_info.flag |
| 同文件：ProcessRecv | 读入/释放两个阶段 | `release=false` 与 `release=true` 的职责及未选中行分支 |
| [ffn_worker_batching_apt.cpp](../op_kernel/ffn_worker_batching_apt.cpp)：RunPrepare | 组织上下文、发布屏障和 prepare | 入口屏障、发布屏障、统一读取 selectedMb |
| 同文件：FfnWorkerBatchingA5 | 组织整个流水和最终 release | gather 后跨核等待，所有读者完成后才能清 flag |

### 5.1 等待条件是入口，快照才约束本次消费集合

```cpp
const bool ready = ctx_->asyncRecv ? readyNum > 0
                                  : static_cast<uint32_t>(readyNum) >= ctx_->A;
```

这一行只负责决定何时结束等待，不能单独保证部分接收正确。它必须与保存 flags、IsSelected 过滤、未选中 flatIds 填 sentinel、延迟 release 配套，才能避免误读和误清。

### 5.2 IsSelected 的地址含义

```cpp
DataCopyPad(flag, readyGm_[(row + 1) * (ONE_BLK_SIZE / sizeof(int32_t))], cp, pad);
SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
return flag.GetValue(0) == 1;
```

`readyGm_` 的元素类型为 int32，因此块大小需要先换算成 word 数。`row+1` 跳过头块。MTE2_S 保证搬入完成后，标量分支才读取该 flag。prepare 和 release 调用同一个函数，保证它们处理同一集合。

### 5.3 ProcessRecv 的三个有效组合

| 调用组合 | 读原始 ids | 写 flatIds | 清原始 ids/flag |
| --- | --- | --- | --- |
| 同步，release=false | 全部行 | 全部行 | 全部行，维持原时序 |
| 异步，release=false | 仅选中行 | 选中行写 ids，其余写 sentinel | 不清 |
| 异步，release=true | 不读 | 不写 | 仅选中行，在全核 gather 完成后清理 |

`release=true` 由入口代码限定在异步路径调用，不是单独暴露给用户的属性。

## 6. 同步关系与正确性依据

| 同步位置 | 依赖关系 | 缺失时可能出现的问题 |
| --- | --- | --- |
| Waiter 前 SyncAll | 所有核读完入口上下文 → 0 号核推进 cursor | 不同核读取到本次/下次两个索引 |
| flag 队列 EnQue/DeQue 及 V_S | flag DMA → 向量求和 → 标量 ready 判定 | 使用未完成的 DMA 或求和结果 |
| 快照写入完成后再释放 UB | MTE3 读取 flags → flags 缓冲复用 | 快照数据被下一轮覆盖 |
| Waiter 后 SyncAll | 0 号核发布快照 → 各核读取快照 | 部分核读取未发布的 selectedMb/flags |
| prepare 后 SyncAll | 各核生成 flatIds → 排序 | 排序看到未写完的行 |
| gather 后 SyncAll | 全部核读 token 完成 → release | 生产者复用仍被读取的 token 槽 |
| release 中 ids 回写等待 | ids 失效回写完成 → flag 清零 | 新一代 ids 被旧消费的迟到清理覆盖 |

`TPipe::Reset()` 负责阶段间本核资源复用，不能替代跨核的 `SyncAll()`。等待指令用于建立具体的数据依赖，不是通过固定延时猜测其它核已经完成。

## 7. A=4、M=3 的执行示例

假设入口 polling_index=0，每行给出该 micro batch 下四个 session 的 flag：

| micro batch | flag 向量 | 扫描结果 |
| --- | --- | --- |
| 0 | [0,0,0,0] | 跳过 |
| 1 | [0,1,0,1] | 选中，停止扫描 |
| 2 | [1,0,0,0] | 本次不扫描、不消费，保留给后续调用 |

本次 `selectedMb=1`，快照选择 session 1 和 3，共享 polling_index 更新为 2。各核把 micro batch 1 中这两个 session 的 ids 搬入 flatIds，对 session 0 和 2 对应的 workspace 行填 sentinel。

若快照形成后 session 0 的 micro batch 1 才发布 flag，它不属于本次集合。完成 gather 后，本次只清 `(session=1, mb=1)` 和 `(session=3, mb=1)` 的 ids/flag；迟到的 `(0,1)` 和已就绪的 `(0,2)` 都保留。

下次调用从 micro batch 2 开始，可先处理 `(0,2)`；后续扫描绕回 micro batch 1 时再处理迟到的 `(0,1)`。这里的轮询顺序没有等待 A 个 session 到齐的限制，但并不承诺某一项的固定墙钟时间上限。

## 8. 测试设计与验收证据

### 8.1 静态 kernel / ACLNN ST

[golden.py](../tests/assets/golden.py) 单独生成 ready 矩阵。未就绪行的设备输入保留正常 expert_ids，只在 CPU golden 的副本中屏蔽这些行。如果 kernel 错误地读取全部行，会产生额外 token，无法通过比较。

新增 38 条用例由以下部分组成：六种 dtype × 六类场景（部分 ready、末尾 ready、全部 ready、ready 全 mask、M=3 跳过空项、NORM 下 syncFlag=1），再加 A=1024 的部分 ready FP16 与末尾 ready FP4 两条边界用例。kernel 和 ACLNN 各增加同样的 38 条。

静态 ST 至少预置一个 ready 描述符，不用于证明无数据时的持续等待。正常输出按独立 golden 比较有效 token、索引和 scale；时序和原始描述符副作用交给下面的真实地址测试。

### 8.2 Host UT

[Host tiling UT](../tests/ut/op_host/test_ffn_worker_batching_tiling.cpp) 检查默认值、0/1 下发、非法值拒绝、key 100/101 不变、wsReady 对齐、workspace 不重叠及容量覆盖。need_schedule=0 的合法属性也被覆盖。

### 8.3 动态等待与握手测试

[verify_sync_flag_state.py](../tests/st/arch35/verify_sync_flag_state.py) 用 aclrtMalloc 分配真实设备缓冲，把真实指针放入 schedule_context，不启用内嵌偏移测试模式。六个场景复用同一设备上下文，检查跨调用状态：

| 场景 | 验证内容 |
| --- | --- |
| empty_wait_then_one_ready | 初始无 flag；启动后延迟 200ms，确认 stream 未完成，再发布一个 flag，期望返回 |
| skip_empty_mb_and_preserve_others | 跳过空项，只消费选中描述符，其余数据保持 |
| cursor_wrap_and_partial_ready | cursor 对 M 回绕，处理部分 session |
| reuse_consumed_slot | 被消费过的槽写入新一代数据后可以再次处理 |
| ready_with_no_valid_tokens | ready 全 mask 返回 0，并完成清理 |
| sync_waits_for_all | 同步对照：已有一个 ready 仍需等待其它三个发布 |

每次调用比较输出 token、四类索引、group_list、actual_token_num，以及完整描述符表、polling_index、token_data。完整描述符比较还用于发现误清未选中行或误改 layer_id。200ms 是测试观察窗口，不是算子内部超时。

### 8.4 已完成结果与本次文档更新范围

功能实现轮已完成：Host UT **33/33**、kernel ST **106/106**、ACLNN ST **105/105**、真实地址状态测试 **6/6**，无失败或跳过。详见 [验证报告](arch35_sync_flag_validation.md) 和 [逐用例结果](arch35_sync_flag_results.csv)。

本次后续更新只补充源码注释与设计说明，不改变属性、算法、同步指令、内存布局和测试执行逻辑。通过 C/C++ 非注释 token、Python AST 对比确认这一点；未因注释更新重复执行上板回归。验证报告中的源码 SHA256 对应上一轮实际构建测试的版本，注释更新后的源码哈希自然不同。

验证边界仍为单算子功能和协议检查，未执行 attention_to_ffn_v2 多机 URMA 端到端联调，未设性能提升验收目标。

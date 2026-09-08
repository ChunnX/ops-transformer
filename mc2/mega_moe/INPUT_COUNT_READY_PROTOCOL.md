# MTE 输入 count 就绪协议

共享专家 GMM1/激活只依赖本卡输入。MTE 路径删除其前面的入口跨卡握手，保留本卡
`SyncAll<false>()` 和出口 `CrossRankSyncInWorldSize`。Dispatch 仍校验完整 count 表，
count 布局、24-bit count + 8-bit epoch 编码及 VF cumsum 不变。

## 修改前后对比

修改前，每个 AIV 完成自身量化后就生成并发送 index/count，接着清零 workspace、
整理共享输入。所有核经过本卡 `SyncAll<false>()` 后，AIV 还需要完成入口跨卡握手及
其末尾的 `SyncAll<true>()`，才能进入共享激活或 Dispatch。AIC 不执行跨卡握手，
但其共享 GMM1 与 AIV 激活存在 UB/ping-pong 交接，因此仍会受到 AIV 跨卡等待的影响。

修改后，在本核量化写回后发布本地 ready，保持 index 提前发送；只在发布 count 前
等待所有本卡量化生产者，随后照常清零和整理共享输入。本卡同步通过后直接进入共享
计算，远端依赖由原有 count epoch 校验承担。

| 项目 | 修改前 | 修改后 |
|---|---|---|
| 本卡量化依赖 | 入口本卡/跨卡同步共同收口 | 发 count 前等待所有本卡量化 ready |
| index/count 发送 | 位于 workspace 清零、共享输入整理之前 | 仍位于这两项工作之前 |
| 入口跨卡握手 | AIV 参与，末尾另有 AIV 全核同步 | 删除 |
| 共享计算入口 | 配对 AIV 可能阻塞在跨卡握手 | 本卡 `SyncAll<false>()` 后即可推进 |
| 接收 count | 逐槽验证 epoch，整表就绪后 VF cumsum | 保持原方式，epoch 同时承担输入就绪通知 |
| ready 状态 | 原逐 AIV 同步计数 | 复用同一独占 64B 槽，不新增跨卡消息 |
| 本卡同步、出口跨卡同步 | 保留 | 保留 |

预期收益来自移除入口跨卡等待、让共享计算提前推进；新增的本地轮询有成本。
尚无多卡性能数据，不将编译或 CPU 协议测试结果视为性能提升证据。

## 发送与消费顺序

```text
每个 AIV：
量化自己的 token（允许空任务）
  → MTE3_S 完成等待
  → 发布本核完整输入计数（量化 ready）
  → 生成、发送自己负责的 compact index
  → 等待所有本卡 AIV 的量化 ready
  → 收口 index MTE3 写入，发送 epoch | count
  → workspace 清零、共享输入整理
  → 本卡 SyncAll<false>
  → 共享 GMM1/激活及 MoE 阶段

接收 AIV1：
本卡 SyncAll<false>
  → 读取本轮 epoch
  → 逐槽确认 count epoch，计算 prefix
  → Dispatch 拉取源卡 token/scale
```

量化按 token 分工，而路由按专家分工。发送某个专家 count 的核必须等其他量化核，
不能只检查自身。等待放在 index 发送之后、count 发布之前，使快核仍可提前处理 index。
count 继续先于 workspace 清零发送。没有专家发送任务的 AIV 直接继续本地准备，
但必须在此前发布量化 ready；没有 token 的 AIV 也必须发布。

发送端已有 route ring 的 `MTE3_V` 完成等待，并在 count 发布前显式加入
`PipeBarrier<PIPE_MTE3>()`。协议依赖这些完成/顺序操作对 peer window 的写入语义；
CPU 模型测试不证明硬件互连的可见性，仍需设备多卡验证。

## 状态地址与同址复用

不新增、清零或通过 MTE 搬运 ready 数组，复用原入口握手的逐 AIV 同步计数槽：

```text
rankSyncBase + 48 KiB + aivIndex * 64B
```

每槽第一个 int32 保存完整轮次计数，其余字节不写。该区域与前部 rank 信号区、
60 KiB 之后的通信数据区、workspace 清零区分离。每个槽仅由所属 AIV 使用标量
`WriteGmByPassDCache` 更新；其他 AIV 使用 `ReadGmByPassDCache` 读取并退避，
不使用 AtomicAdd，也不混用 MTE 清零和标量发布。

沿用原通信窗口的初始化及调用约定：首次使用时同步计数有效初始化，所有参与 rank
按相同顺序调用，窗口布局与活动 AIV 集合保持兼容。

若上次出口计数为 `s`：

| 时刻 | 槽值 | 条件 |
|---|---|---|
| 本轮量化尚未完成 | `s` | 不满足本轮 ready |
| 本轮量化写回完成 | `s + 1` | 本核发布量化 ready |
| 发布 count 前 | 所有量化槽等于 `s + 1` | 所有源 token 可读 |
| 出口同步 | `s + 2` | 保留原出口推进 |

本地等待比较完整 32-bit 计数；仅跨卡 count 使用 `(sequence & 0x7f) | 0x80`。
计数递增通过无符号加法实现，避免有符号溢出。每轮仍推进两次，保持原 count epoch
和出口握手序列。不得在 kernel 入口把槽清零，否则会覆盖提前发布的新值。

同址复用的两个保护边界：

1. 输入 `SyncAll<false>` 在所有 count 发送者完成 ready 读取之后。任意 AIV 都不可能
   越过它推进出口计数，而另一个本卡发送者还在等输入计数。
2. 接收端读取 slot 0 的 epoch 发生在输入同步之后、MoE 阶段末尾的 AIV 同步之前，
   这期间 slot 0 不会被出口更新。当前完整 count 校验和保留的出口握手还约束了
   各 rank 跨轮次的进度；未来若拆掉整表等待或出口同步，必须重新审查窗口复用。

## 验证

从仓库根目录运行：

```bash
python3 mc2/mega_moe/tests/ut/op_kernel/arch35/run_input_sync_test.py
```

测试直接编译生产 `mega_moe_input_sync.h`，仅用 CPU mock 替代 Ascend 标量访问、
MTE3_S 完成等待和时钟。覆盖慢量化生产者、空 token/无 route 任务、零 count、
两 rank 并发、1560 轮重复调用、8-bit epoch 回绕、计数跨符号边界及 32-bit 回绕。
地址哨兵和单写者检查覆盖槽偏移、越界写及 ready 早于量化写回；AIC 构建验证不访问状态。
测试启用 UBSan，并设置进程超时捕获等待不退出。

本地验证（2026-09-08）：上述 AIV/AIC 协议测试通过。CANN 9.2 beta 编译器以
Ascend950PR_9599 为目标编译通过 A8W8 的 6 个 MTE tiling key，以及 W4 的 18 个
MTE tiling key，覆盖 combine 量化选项和 topk weights prefetch。编译覆盖本次五个
内核头文件的修改，提交检查随后仅调整格式。编译使用仅用于验证的 Hcomm 版本屏蔽 shim，避免当前 SDK
缺少上游 URMA API 导致无关编译失败；URMA 未在此次编译验证范围内。

设备验收仍需多卡、共享专家开启/关闭、A8W8/A8W4/A4W4、空专家、小 bs 空任务核、
跨 epoch 回绕的连续调用，并与基线比较精度、首个共享 GMM1/激活时间和算子总耗时。

# URMA 队列 Dump 方案对比与选型总结

> 面向场景：生产环境中发生 AIC/AIV error 时，自动定位 URMA SQE/CQE/WQEBB 值。
>
> 对比对象：
> - **方案 A**（采用）：Kernel 侧队列快照——算子 kernel 在每个通信 turn 将 WQE/CQE 原始字节写入 GM workspace，异常回调自动 dump。
> - **PR #92**（不采用）：asc-comm `feat(hcomm): add dfx channel dump helper`——异常回调仅记录状态，host 安全点调 `HcommChannelInfoDump(channel)` 启动 helper AIV kernel 读取队列现场。

---

## 一、方案 A：Kernel 侧队列快照（采用）

### 1.1 核心思路

WQE/CQE 常驻 UB（Unified Buffer），Host 侧 `aclrtMemcpy` D2H 无法读取。
不依赖事后从 host 启动 device 代码，而是让**已经在 device 上运行的算子 kernel 自己做快照**：

```
正常运行（每个通信 turn，常开，<0.5% 开销）:
  AIV block i（驱动 peer i 的 channel）:
    ① 快照计数器 (sqHead/sqTail/cqHead/cqTail) → GM 快照段
    Commit()
    ② 回读刚提交的 WQE 原始字节 (__gm__ 读 sqVa) → GM 快照段（WQE 环形历史 + turn 戳）
    Wait()
    ③ 回读最新 CQE 原始字节 (__gm__ 读 scqVa) → GM 快照段
    每 turn 合并 1 次 DataCacheCleanAndInvalid（照抄 mc2_runtime_info.h:140-146 的 FlushCache 模式）

AIC/AIV error 发生:
  kernel 死亡（快照已安全落 GM）→ ExceptionDumpParseFunc 回调
  → Mc2DumpTilingData 按段循环 dump（mc2_exception_dump.h:936-957，零改动）
  → workspace_seg2_type13_*.bin 自动产出 ← 快照就在里面
  → dump_analysis 解析 → 一眼定位时间线
```

### 1.2 为什么这个架构成立

仓库已有完整的 kernel→GM→host dump 链路，快照只需复用：

| 已有设施 | 位置 | 作用 |
|---------|------|------|
| `Mc2RuntimeInfo` | `mc2_runtime_info.h:82-137` | 算子 kernel 每个 turn 把执行进度（turn/position/phase/commit/wait）写入 GM workspace，用 `DataCacheCleanAndInvalid` 刷 cache，异常时 host D2H 读出 |
| Workspace 段自动 dump | `mc2_exception_dump.h:936-957` | `Mc2DumpTilingData` 按 `wsLayout.segments` 逐段自动 dump——新增段无需改 host 代码 |
| `MC2ExceptionDump::ExceptionDump<Policy>` | `mc2/common/op_kernel/mc2_exception_dump.h` | MegaMoe 在用的 kernel 侧结构化 dump 框架（kernel 写 GM，host 解析） |

可行性关键证据：**算子 kernel 本身就跑在 AIV 上**（AIV 直驱 URMA），hcomm 库的 kernel 侧 API 内部就是 `(__gm__ ChannelEntity*)channelHandle` 解引用——算子 kernel 具备和 PR #92 helper kernel 完全相同的队列读取能力。

另外对 error 场景有额外好处：AIV error 时该核 UB 内容可能已损坏，而快照是每次写入都 cache-flush 到 GM 的——比事后读 UB 更可靠。

### 1.3 快照数据格式

```cpp
struct QueueSnapshotSlot {           // 每个 peer 一个，~4KB
    uint32_t magic;                  // 0x5A5A5A5A 有效性校验
    uint32_t turn;                   // 当前 turn（与 RuntimeInfo.execTurn 对齐）
    uint32_t peer;                   // 本 block 驱动的 peer
    uint32_t sqHead, sqTail, cqHead, cqTail;   // 实时计数器
    uint8_t  wqeLatest[256];         // 最近提交的 WQE 原始字节
    uint32_t cqeStatus, cqeSubstatus;// 最新 CQE 解码字段
    uint8_t  cqeLatest[256];         // 最新 CQE 原始字节
    uint8_t  wqeRing[8][256];        // 最近 8 个 WQE 历史（含 turn 戳）
    uint32_t wqeRingTurn[8];
    uint32_t ringPos;
};
```

总占用：16 block 上限 × 4KB = 64KB（新 workspace segment，type=13）。

### 1.4 落盘与定位链路

```
error → 回调 → Mc2DumpTilingData 自动 dump 所有段（现有逻辑，零改动）
  → workspace_seg2_type13_*.bin   ← 快照段
  → workspace_seg1_type12_*.bin   ← RuntimeInfo（已有）
  → tools/dump_analysis 扩展解析器:
     "turn 5: WQE(opcode=0x6, rmtEid=0xdf..., sgeNum=2) 提交
      turn 5: CQE status=0 正常完成
      turn 6: WQE(opcode=0x3, ...) 提交  ← 最后快照
      turn 6: 无 CQE 记录
      RuntimeInfo: execTurn=6, commit=6, wait=5 → 卡在 Wait#6
      队列: sqTail=7, sqHead=6 → WQE#6 pending，远端未完成
      结论: peer X 的 WQE#6 (opcode 0x3, 目标地址 0x...) 未完成"
```

WQE/CQE 字段解码复用已 vendored 的 `AscendC::HcommUrmaSqeCtx` / `HcommUrmaJfcCqeCtx`（`mc2_hcomm_dfx_types.h:242-293`）。

### 1.5 性能影响

每 turn 额外 ~0.5μs，占单轮耗时（100μs~1ms）的 <0.5%，与已存在的 RuntimeInfo 打点开销同量级。

| 操作 | 数据量 | 耗时估算 |
|------|--------|---------|
| ① 计数器快照 | ~16B 读 + 16B 写 | ~50ns |
| ② WQE 回读（写 wqeLatest + ring 槽位） | ~256B 读 + ~512B 写 | ~150ns |
| ③ CQE 回读 | ~256B 读 + ~256B 写 | ~150ns |
| cache flush（合并为每 turn 1 次） | 1-2 cache line op | ~100-200ns |
| **合计** | **~1.2KB/turn** | **~0.5μs** |

| 维度 | 数值 | 对比基准 |
|------|------|---------|
| 关键路径（在 SyncAll 前，计入） | ~0.5μs/turn | 现有 RuntimeInfo 打点 ~0.5μs/turn → 总量翻倍仍 <1% |
| GM 写带宽 | ~1.2KB × 100 turn ≈ 120KB/kernel | 通信流量 ~700MB → 0.02% |
| workspace | 固定 64KB | LIB_API 段 ~16MB → 0.4% |

性能三道保险：uint64 字宽拷贝（禁止逐字节）；每 block 只快照自己驱动的 channel；每 turn 合并 1 次 flush。

### 1.6 实施步骤

| # | 步骤 | 文件 | 改动量 |
|---|------|------|--------|
| 0 | Phase 0 验证：ST 加 trap() 触发 error，确认回调触发、现有 WQE/CQE D2H 是否真的失败、workspace dump 正常 | 仅测试代码 | 0.5 天 |
| 1 | 新增 kernel 快照引擎 | 新文件 `mc2/common/op_kernel/mc2_queue_snapshot.h`（~250 行） | 1 天 |
| 2 | tiling 增加快照 segment（type=13，64KB）+ 偏移传递 | `all_gather_matmul_v3_tiling.cpp:785-791`、`mc2_tiling_struct.h` | 0.5 天 |
| 3 | kernel 侧接入 ①②③ 快照点（每个通信 block 快照自己驱动的 channel） | `all_gather_mx_matmul_urma_impl.h:279-301`、`all_gather_matmul_v3_apt.cpp:45-49` | 1 天 |
| 4 | `tools/dump_analysis` 增加快照段解析器（时间线重建） | `mc2/tools/dump_analysis/` | 1 天 |
| 5 | ST 端到端验证：trap() → 自动落盘 → 解析器输出时间线 | 测试 | 1 天 |

host 侧 dump 机制（`mc2_exception_dump.h`）**零改动**。

---

## 二、PR #92 方案概述

asc-comm PR #92（`feat(hcomm): add dfx channel dump helper`）提供了两阶段 dump 架构：

### 2.1 架构

```
launch 前: HcommChannelExceptionCallbackRegister(kernelSymbol)   ← op_api/torch_extension 层
launch kernel
aclrtSynchronizeStream 返回错误                                    ← op_api/torch_extension 层
host 安全点: HcommChannelInfoDump(channel)                        ← op_api/torch_extension 层
  → 检查 pending 异常状态（回调只做 atomic store，极轻量）
  → 有异常: DumpChannelInfo → 逐通道 D2H 读 ChannelEntity/SqContext/CqContext
  → WQE/CQE: 启动 helper AIV kernel (QueueReader) 逐字节拷贝 UB → device result buffer → D2H 回 host
  → ParseAndPrintChannelDump: 解码 + printf 到 stdout
```

### 2.2 核心组件

| 组件 | 作用 |
|------|------|
| `HcommChannelExceptionCallbackRegister(kernelSymbol)` | launch 前注册异常回调，内部调用 `aclrtGetFuncBySymbol` → `aclrtFunctionGetBinary` → `aclrtBinarySetExceptionCallback` |
| `HcommChannelInfoDump(channel)` | host 安全点调用，传入 ChannelHandle 作为 device VA |
| `QueueReader`（helper AIV kernel） | 在独立 stream 上 launch，逐字节拷贝 UB-resident 队列条目到 device result buffer，再 D2H |
| `HcommDfxChannelDumpData`（结构化 blob） | 自描述 dump 数据（magic/version），collect 写入、parse 打印 |

### 2.3 设计亮点

- 回调极轻量（仅 atomic store），D2H/launch/打印全部推迟到 host 安全点
- WQE/CQE 通过 helper kernel 可靠读取（解决 UB 不可直接 D2H 的问题）
- 窗口采样：前 4 个 slot + head/tail 附近 4 个 slot
- 正常路径零开销

---

## 三、优缺点对比

| 维度 | 方案 A（kernel 快照，采用） | PR #92（helper kernel 事后读，不采用） |
|------|---------------------------|--------------------------------------|
| **数据新鲜度** | 最后一次快照点（Commit 后） | error 时刻的实时队列状态 |
| **error CQE**（硬件在 error 瞬间写的） | 无法捕获（kernel 已死） | 可捕获（dump 发生在 error 后） |
| **WQE 历史**（回放轨迹） | ring 历史 + turn 戳，可重建时间线 | 单次快照，无时间维度 |
| **正常运行开销** | ~0.5μs/turn（<0.5%） | 零 |
| **kernel 侵入** | 需改 2 个通信 impl | 零（不改算子） |
| **触发可靠性** | 快照早已落 GM，error 后只需 D2H | 依赖 error 后还能 launch AIV kernel |
| **自动化程度** | 零集成，error → 自动落盘 | 需调用方集成（注册 + sync 后调 dump） |
| **构建复杂度** | 复用现有构建系统 | 需 Bisheng ASC 编译器独立编译 .so + dlopen |

### 各自的决定性优势

**PR #92 赢在"事后实时"**：
- error 时刻硬件写入的 error CQE（status/substatus 错误码）只有它能读到
- 正常路径零开销，不改算子 kernel

**方案 A 赢在"三个不依赖"**：
1. **不依赖 error 后设备还能跑 kernel**——PR #92 的 QueueReader 要在 error 后 launch AIV kernel，若 error 导致设备/流进入 fatal 态（生产最常见情况），helper kernel 跑不起来，dump 静默失败。方案 A 快照在 error 前已 flush 到 GM，error 后只做 D2H
2. **不依赖调用方集成**——PR #92 需 launch 前注册 + sync 失败后调 `HcommChannelInfoDump(channel)`，方案 A error → 自动落盘
3. **不依赖 UB 完好**——AIV error 时该核 UB 可能已损坏，快照走 GM 更可靠

### 故障场景矩阵

| 场景 | PR #92 | 方案 A | 说明 |
|------|--------|--------|------|
| 通信 hang（Wait 卡死，最常见） | 取决于设备能否 launch kernel | 完整定位 | hang 时队列静止，最后快照 ≈ 实时状态，两者等效 |
| AIC/AIV 硬件 error（chip fatal） | helper kernel 大概率跑不了 | 快照已在 GM | 方案 A 的决定性场景 |
| CQE 完成错误（带 error status） | 能读到 error CQE | 靠最后正常 CQE + commit/wait 差推断 | PR #92 的决定性场景 |
| 偶发/难复现问题 | 需要预先集成调用代码 | 历史 ring 提供回放轨迹 | 时间维度只有快照方案有 |

---

## 四、为什么不选择 PR #92

### 4.1 公开 API 在我们的架构里接不上（硬伤）

PR #92 的设计假设"算子作者自己 launch kernel、自己 sync、自己持有 handle"——hcomm 示例确实如此。但我们是 ACLNN 框架路径，三个前提全不成立：

| PR #92 的前提 | 我们的现状 | 调研结论 |
|--------------|-----------|---------|
| 调用方持有 `ChannelHandle` 传给 `HcommChannelInfoDump(channel)` | handle 只存在 device 内存，host 侧 `CreateContext` 后栈变量即销毁（`comm_channel_builder_manager.cpp:80`） | op_api 层拿不到 |
| launch 前用 kernel symbol 注册回调 | `aclOpExecutor` 是不透明结构体，kernel 由框架生成的 `aclnnInner*` 内部 launch | symbol 不可获取 |
| 依赖 `aclrtGetFuncBySymbol` / `aclrtFunctionGetBinary` / `aclrtBinarySetExceptionCallback` | 这三个 API 在我们的 CANN SDK 和整个仓库中完全不存在（未声明、未 stub、未 dlsym） | 无从调用 |
| 在 `aclrtSynchronizeStream` 失败后的 host 安全点调 dump | 生产代码 fire-and-forget（`ACLNN_CMD` 异步），没有任何 sync 点 | 触发点不存在 |

### 4.2 退化方案（只借 QueueReader）构建成本高

绕开公开 API、只抄 helper kernel 的 QueueReader 组件，就必须：
- 用 Bisheng ASC 编译器独立编译 `.so`（`<<<>>>` 语法要求，op_host 是普通 C++）
- 通过 dlopen 动态加载
- 额外维护一套独立构建体系

### 4.3 生产 error 场景下触发不可靠

PR #92 的 QueueReader 要在 error 发生之后 launch 一个 AIV kernel 去读队列。但生产环境最常见的恰恰是 error 导致设备/流进入 fatal 态——那时 helper kernel 根本跑不起来，dump 静默失败，什么都没留下。

方案 A 的快照在 error 之前每 turn 已 flush 到 GM，error 后只需 D2H（现有回调已验证可行），触发链路 100% 可靠。

### 4.4 异常回调机制冲突风险

我们的异常处理通过 `IMPL_OP(Op).ExceptionDumpParseFunc(wrapper)` 注册（metadef 框架层，内部对接 ACL runtime 异常回调）。PR #92 通过 `aclrtBinarySetExceptionCallback` 在 binary 层注册。两者是否可共存未经验证，存在冲突风险。

---

## 五、结论

PR #92 是"理想触发模型下的事后实时取证"，在 hcomm 示例场景（算子作者直接 launch kernel、控制 sync、持有 handle）下是正确设计。但它的触发模型在我们 ACLNN 框架架构里三个前提全不满足（无 kernel symbol、无 sync 点、无 host handle），且在 error 后设备 fatal 的生产场景下 helper kernel 不可靠。

方案 A（kernel 侧队列快照）用 <0.5% 开销 + 改 kernel 的代价，换来：
- 触发链路 100% 可靠（不依赖 error 后设备可用）
- 零集成（error → 自动落盘）
- 历史回放轨迹（ring + turn 戳）
- host 侧 dump 机制零改动

唯一能力缺口是 error 瞬间的 error CQE，靠 Phase 0 实测 post-error D2H 能否读到 CQ 补充；补不上则接受"最后 CQE + commit/wait 差推断"的精度损失。

两方案本质互补（事后实时 vs 事前留痕），但在我们的约束下，方案 A 是唯一能在生产环境全自动落地的选择。

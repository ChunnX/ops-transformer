# DispatchSendKernel 低频诊断打印设计

## 目标

在不修改一级通信状态推进逻辑的前提下，为 `DispatchSendKernel` 增加低频诊断打印，定位循环无法退出时具体卡在哪个条件。

## 打印内容

1. sender 开始处理目标 server 时打印一次：
   - `rankId_`
   - `localExpertId` 对应的 `roundTag`
   - `targetServerForSender`
   - `senderIdx`
   - 初始 `cursor`
   - 初始 `count`
   - `realComputeCoreNum`

2. 轮询达到阈值后打印状态，后续按固定周期低频打印：
   - 当前 `cursor` 和 `count`
   - `cursor > count` 是否成立
   - 第一个 `done < roundTag` 的 compute core 及其 done 值
   - 当 `0 <= cursor < count` 时，cursor 对应 slot 的 meta state
   - 当前轮询次数

3. sender 正常退出时打印一次最终状态。

## 频率控制

- 第一次诊断打印阈值：`1 << 20` 次轮询。
- 后续打印周期：每增加 `1 << 20` 次轮询打印一次。
- 正常快速完成时只产生入口和退出两条打印。

## 约束

- 不修改 `count`、`cursor`、meta state 或 done。
- 不增加 `Drain`、同步或通信调用。
- 不在每次循环中打印。
- 诊断代码仅放在 `DispatchSendKernel` 内，便于问题确认后整体删除。

## 验证

- 检查代码格式和编译语法。
- 检查 diff，确认除诊断打印外没有改变通信控制流。
- 实机运行时根据打印区分：
  - `cursor > count`：跨次状态清理异常。
  - `cursor < count && metaState == 0`：slot 已计数但未发布。
  - `cursor < count && metaState == 1`：slot 已发布但发送侧未完成状态推进。
  - `cursor == count && unfinishedCore >= 0`：compute done 条件未满足。

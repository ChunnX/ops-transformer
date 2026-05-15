# Cache Clean、Flush 与同步边界

当代码新增、删除或质疑 `DataCacheCleanAndInvalid`、flush wrapper、`Sync*`、`SetFlag/WaitFlag`、跨核可见性边界时读取本文件。

## 判断顺序

1. 明确 producer 和 consumer。
   - producer engine：AIV、AIC、FIX、MTE2、MTE3、scalar helper。
   - consumer engine：AIC load、AIV load、GM reader、外部输出、跨核 reader。

2. 明确已有顺序保证。
   - `SetFlag/WaitFlag`：显式 pipeline 顺序。
   - `SyncMte3ToMte2`：GM 写后经 MTE2 读。
   - `SyncFixToMte2`：Fixpipe 写后 MTE2 读。
   - `SyncFixToS`：Fixpipe 写后 scalar 观察。
   - 跨核 flag：不同 core/subblock 交接。

3. 对比 origin/reference 行为。
   - origin 相同 GM workspace handoff 只靠 flag/wait 且无 DCCI，则不要默认加 DCCI。
   - 新路径不同于 origin 时，先写出缺失的具体边界，再加 DCCI。

4. 每次只改一种边界。
   - 删除 DCCI 但保留 sync 是一个优化。
   - 删除 sync 是另一个优化，要重新走精度和性能 gate。

## 规则

- 不要因为“AIC 可能看不到 AIV 写入”这种模糊判断加 cache clean/flush。
- 不要只因为某个 GM write 靠近 flush 就保留它；先证明它是真消费者还是副作用。
- 不要在“删除 cache clean”的 phase 里同时删 `Sync*`。
- 小 shape PASS、长 shape FAIL 时，按真实边界失败处理，不当作噪声。
- cache 操作如果按行或按 block 执行，要计算真实 scalar 成本；一个概念上的 clean 可能是上百条 cacheline DCCI。

## 通用案例教训

- “4 个 cache clean”可能实际是 4 个 block helper，每个 helper 又按多行 cacheline 清理。
- Full input preload 配合 per-line clean 可能精度安全，但 scalar 成本抵消 MTE2 收益。
- 只移除显式 DCCI 且保留原有 sync/order 时可能正确并提速；DCCI cleanup 不证明 sync 冗余。
- DCCI 和 sync 一起删，即使短 shape PASS，也可能长 shape 失败。
